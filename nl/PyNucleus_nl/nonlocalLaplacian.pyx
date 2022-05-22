###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from libc.math cimport ceil
import numpy as np
cimport numpy as np
cimport cython

include "config.pxi"

from libc.math cimport sin, cos, M_PI as pi
from PyNucleus_base.myTypes import INDEX, REAL, ENCODE, BOOL
from PyNucleus_base import uninitialized
from PyNucleus_base.intTuple cimport intTuple
from PyNucleus_fem.mesh import mesh0d, mesh1d
from PyNucleus_fem.functions cimport function, constant
from PyNucleus_fem.DoFMaps cimport P0_DoFMap, P1_DoFMap, P2_DoFMap
from PyNucleus_fem.meshCy cimport sortEdge, encode_edge, decode_edge, encode_face
from PyNucleus_fem.femCy cimport local_matrix_t
from PyNucleus_fem.femCy import assembleMatrix, mass_1d_sym_scalar_anisotropic, mass_2d_sym_scalar_anisotropic
from PyNucleus_fem.quadrature import simplexXiaoGimbutas
from PyNucleus_base.sparsityPattern cimport sparsityPattern
from PyNucleus_base.linear_operators cimport (CSR_LinearOperator,
                                              SSS_LinearOperator,
                                              Dense_LinearOperator,
                                              Dense_SubBlock_LinearOperator,
                                              diagonalOperator,
                                              TimeStepperLinearOperator,
                                              nullOperator)
from . nonlocalLaplacianBase import MASK
from . twoPointFunctions cimport constantTwoPoint
from . fractionalOrders cimport (fractionalOrderBase,
                                 constFractionalOrder,
                                 variableFractionalOrder,
                                 variableFractionalLaplacianScaling)
from . kernels import getFractionalKernel

from . clusterMethodCy import (assembleFarFieldInteractions,
                               getDoFBoxesAndCells,
                               getFractionalOrders,
                               getFractionalOrdersDiagonal,
                               getAdmissibleClusters,
                               symmetrizeNearFieldClusters,
                               trimTree)
from . clusterMethodCy cimport (refinementType,
                                GEOMETRIC,
                                MEDIAN,
                                BARYCENTER)
import logging
from logging import INFO
import warnings
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py cimport MPI
include "panelTypes.pxi"

LOGGER = logging.getLogger(__name__)


cdef class IndexManager:
    cdef:
        DoFMap dm
        indexSet myDofs
        public INDEX_t[::1] localDoFs
        INDEX_t[::1] permutedDoFsLocal
        INDEX_t[:, ::1] idxCellFlip
        LinearOperator A
        sparsityPattern sP
        public dict cache
        intTuple hv

    def __init__(self, DoFMap dm, LinearOperator A=None, cellPairIdentifierSize=1, indexSet myDofs=None, sparsityPattern sP=None):
        cdef:
            INDEX_t[:, ::1] idxCellFlip
            INDEX_t j, offset
        self.dm = dm
        self.myDofs = myDofs
        self.localDoFs = uninitialized((2*self.dm.dofs_per_element), dtype=INDEX)
        self.permutedDoFsLocal = uninitialized((2*self.dm.dofs_per_element), dtype=INDEX)
        self.hv = intTuple.create(uninitialized(cellPairIdentifierSize, dtype=INDEX))
        self.A = A
        self.sP = sP
        if self.dm.mesh.dim == 1:
            idxCellFlip = uninitialized((2, self.dm.dofs_per_element), dtype=INDEX)
            for j in range(self.dm.dofs_per_vertex):
                idxCellFlip[0, j] = j
                idxCellFlip[0, self.dm.dofs_per_vertex+j] = self.dm.dofs_per_vertex+j

                idxCellFlip[1, j] = self.dm.dofs_per_vertex+j
                idxCellFlip[1, self.dm.dofs_per_vertex+j] = j
            offset = 2*self.dm.dofs_per_vertex
            for j in range(self.dm.dofs_per_cell):
                idxCellFlip[0, offset+j] = offset+j
                idxCellFlip[1, offset+self.dm.dofs_per_cell-1-j] = offset+j

        elif self.dm.mesh.dim == 2:
            idxCellFlip = uninitialized((3, self.dm.dofs_per_element), dtype=INDEX)
            for j in range(self.dm.dofs_per_vertex):
                idxCellFlip[0, j] = j
                idxCellFlip[0, self.dm.dofs_per_vertex+j] = self.dm.dofs_per_vertex+j
                idxCellFlip[0, 2*self.dm.dofs_per_vertex+j] = 2*self.dm.dofs_per_vertex+j

                idxCellFlip[1, j] = self.dm.dofs_per_vertex+j
                idxCellFlip[1, self.dm.dofs_per_vertex+j] = 2*self.dm.dofs_per_vertex+j
                idxCellFlip[1, 2*self.dm.dofs_per_vertex+j] = j

                idxCellFlip[2, j] = 2*self.dm.dofs_per_vertex+j
                idxCellFlip[2, self.dm.dofs_per_vertex+j] = j
                idxCellFlip[2, 2*self.dm.dofs_per_vertex+j] = self.dm.dofs_per_vertex+j
        else:
            raise NotImplementedError()
        self.idxCellFlip = idxCellFlip
        self.cache = {}

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void getDoFsElem(self, INDEX_t cellNo):
        cdef:
            INDEX_t p, dof
        for p in range(self.dm.dofs_per_element):
            self.localDoFs[p] = self.dm.cell2dof(cellNo, p)
        if self.myDofs is not None:
            for p in range(self.dm.dofs_per_element):
                dof = self.localDoFs[p]
                if not self.myDofs.inSet(dof):
                    self.localDoFs[p] = -1

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline BOOL_t getDoFsElemElem(self, INDEX_t cellNo1, INDEX_t cellNo2):
        cdef:
            INDEX_t p, dof
            BOOL_t canSkip = True
        for p in range(self.dm.dofs_per_element):
            dof = self.dm.cell2dof(cellNo1, p)
            self.localDoFs[p] = dof
            canSkip = canSkip and dof < 0
        for p in range(self.dm.dofs_per_element):
            dof = self.dm.cell2dof(cellNo2, p)
            self.localDoFs[self.dm.dofs_per_element+p] = dof
            canSkip = canSkip and dof < 0
        return canSkip

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void addToMatrixElemSym(self, const REAL_t[::1] contrib, REAL_t fac):
        cdef:
            INDEX_t k, p, q, I, J
        k = 0
        for p in range(self.dm.dofs_per_element):
            I = self.localDoFs[p]
            if I >= 0:
                self.A.addToEntry(I, I, fac*contrib[k])
                k += 1
                for q in range(p+1, self.dm.dofs_per_element):
                    J = self.localDoFs[q]
                    if J >= 0:
                        self.A.addToEntry(I, J, fac*contrib[k])
                        self.A.addToEntry(J, I, fac*contrib[k])
                    k += 1
            else:
                k += self.dm.dofs_per_element-p

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void addToSparsityElemElemSym(self):
        # Add symmetric 'contrib' to elements i and j in symmetric fashion
        cdef:
            INDEX_t k, p, q, I, J
        k = 0
        for p in range(2*self.dm.dofs_per_element):
            I = self.localDoFs[p]
            if I >= 0:
                self.sP.add(I, I)
                k += 1
                for q in range(p+1, 2*self.dm.dofs_per_element):
                    J = self.localDoFs[q]
                    if J >= 0:
                        self.sP.add(I, J)
                        self.sP.add(J, I)
                    k += 1
            else:
                k += 2*self.dm.dofs_per_element-p

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void addToMatrixElemElemSym(self, const REAL_t[::1] contrib, REAL_t fac):
        # Add symmetric 'contrib' to elements i and j in symmetric fashion
        cdef:
            INDEX_t k, p, q, I, J
        k = 0
        for p in range(2*self.dm.dofs_per_element):
            I = self.localDoFs[p]
            if I >= 0:
                self.A.addToEntry(I, I, fac*contrib[k])
                k += 1
                for q in range(p+1, 2*self.dm.dofs_per_element):
                    J = self.localDoFs[q]
                    if J >= 0:
                        self.A.addToEntry(I, J, fac*contrib[k])
                        self.A.addToEntry(J, I, fac*contrib[k])
                    k += 1
            else:
                k += 2*self.dm.dofs_per_element-p

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void addToSparsityElemElem(self):
        # Add general 'contrib' to elements i and j
        cdef:
            INDEX_t k, p, q, I, J
        k = 0
        for p in range(2*self.dm.dofs_per_element):
            I = self.localDoFs[p]
            if I >= 0:
                for q in range(2*self.dm.dofs_per_element):
                    J = self.localDoFs[q]
                    if J >= 0:
                        self.sP.add(I, J)
                    k += 1
            else:
                k += 2*self.dm.dofs_per_element

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void addToMatrixElemElem(self, const REAL_t[::1] contrib, REAL_t fac):
        # Add general 'contrib' to elements i and j
        cdef:
            INDEX_t k, p, q, I, J
        k = 0
        for p in range(2*self.dm.dofs_per_element):
            I = self.localDoFs[p]
            if I >= 0:
                for q in range(2*self.dm.dofs_per_element):
                    J = self.localDoFs[q]
                    if J >= 0:
                        self.A.addToEntry(I, J, fac*contrib[k])
                    k += 1
            else:
                k += 2*self.dm.dofs_per_element

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef tupleDictMASK buildMasksForClusters(self, list clusterList, bint symmetricCells, INDEX_t *startCluster):
        cdef:
            nearFieldClusterPair cluster = clusterList[0]
            MASK_t cellMask1, cellMask2
            indexSet cellsUnion = cluster.cellsUnion
            indexSetIterator it = cellsUnion.getIter(), it2 = cellsUnion.getIter()
            indexSet clusterDofs1, clusterDofs2
            INDEX_t cellNo1 = -1, cellNo2 = -1
            INDEX_t[::1] cellPair = uninitialized((2), dtype=INDEX)
            INDEX_t[::1] cellPair2 = uninitialized((2), dtype=INDEX)
            tupleDictMASK masks = tupleDictMASK(self.dm.mesh.num_cells, deleteHits=False, logicalAndHits=True, length_inc=20)
            INDEX_t p, I
            dict cellMasks1, cellMasks2
            MASK_t mask, mask1, mask2, cellMask11, cellMask12, cellMask21, cellMask22, k
            INDEX_t dofs_per_element = self.dm.dofs_per_element

        cellMask1, cellMask2 = 0, 0
        for cluster in clusterList[startCluster[0]:]:
            startCluster[0] += 1
            cellsUnion = cluster.cellsUnion
            cellMasks1 = {}
            cellMasks2 = {}
            clusterDofs1 = cluster.n1.get_dofs()
            clusterDofs2 = cluster.n2.get_dofs()

            it.setIndexSet(cellsUnion)

            while it.step():
                cellNo1 = it.i
                mask1 = 0
                mask2 = 0
                k = 1
                for p in range(dofs_per_element):
                    I = self.dm.cell2dof(cellNo1, p)
                    if I >= 0:
                        if clusterDofs1.inSet(I):
                            mask1 |= k
                        if clusterDofs2.inSet(I):
                            mask2 |= k
                    k = k << 1
                cellMasks1[cellNo1] = mask1
                cellMasks2[cellNo1] = mask2

            if not symmetricCells:
                # TODO: Think some more about this branch, maybe this can be improved.
                it.reset()
                it2.setIndexSet(cellsUnion)
                while it.step():
                    cellNo1 = it.i
                    cellPair[0] = cellNo1
                    cellMask11 = cellMasks1[cellNo1]
                    cellMask12 = cellMasks2[cellNo1]
                    it2.reset()
                    while it2.step():
                        cellNo2 = it2.i
                        if ((cellNo1 > cellNo2) and symmetricCells):
                            continue
                        cellMask21 = cellMasks1[cellNo2]
                        cellMask22 = cellMasks2[cellNo2]
                        cellMask1 = cellMask11 | (cellMask21 << dofs_per_element)
                        cellMask2 = cellMask12 | (cellMask22 << dofs_per_element)
                        if (cellMask1 == 0) or (cellMask2 == 0):
                            continue
                        cellPair[1] = cellNo2
                        mask = self.getElemElemSymMask(cellMask1, cellMask2)
                        # does a logical "and" if there already is an entry
                        masks.enterValue(cellPair, mask)
            else:
                it.setIndexSet(cluster.n1.cells)
                it2.setIndexSet(cluster.n2.cells)
                while it.step():
                    cellNo1 = it.i
                    cellPair[0] = cellNo1
                    cellPair2[1] = cellNo1
                    cellMask11 = cellMasks1[cellNo1]
                    cellMask12 = cellMasks2[cellNo1]
                    it2.reset()
                    while it2.step():
                        cellNo2 = it2.i
                        cellMask21 = cellMasks1[cellNo2]
                        cellMask22 = cellMasks2[cellNo2]
                        if cellNo1 > cellNo2:
                            cellMask1 = cellMask21 | (cellMask11 << dofs_per_element)
                            cellMask2 = cellMask22 | (cellMask12 << dofs_per_element)
                            if (cellMask1 == 0) or (cellMask2 == 0):
                                continue
                            cellPair2[0] = cellNo2
                            mask = self.getElemElemSymMask(cellMask1, cellMask2)
                            # does a logical "and" if there already is an entry
                            masks.enterValue(cellPair2, mask)
                        else:
                            cellMask1 = cellMask11 | (cellMask21 << dofs_per_element)
                            cellMask2 = cellMask12 | (cellMask22 << dofs_per_element)
                            if (cellMask1 == 0) or (cellMask2 == 0):
                                continue
                            cellPair[1] = cellNo2
                            mask = self.getElemElemSymMask(cellMask1, cellMask2)
                            # does a logical "and" if there already is an entry
                            masks.enterValue(cellPair, mask)

            if masks.nnz > 10000000:
                break

        return masks

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline MASK_t getElemSymEntryMask(self, INDEX_t cellNo1, INDEX_t I, INDEX_t J):
        # Add symmetric 'contrib' to elements i and j in symmetric fashion
        cdef:
            INDEX_t p, q, K, L
            MASK_t k = 1
            MASK_t mask = 0
        for p in range(self.dm.dofs_per_element):
            K = self.dm.cell2dof(cellNo1, p)
            for q in range(p, self.dm.dofs_per_element):
                L = self.dm.cell2dof(cellNo1, q)
                if (I == K and J == L) or (J == K and I == L):
                    mask |= k
                k = k << 1
        return mask

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline MASK_t getElemElemSymMask(self, MASK_t mask_dofs1, MASK_t mask_dofs2):
        # Add symmetric 'contrib' to elements i and j in symmetric fashion
        cdef:
            INDEX_t p, q
            MASK_t k = 1
            MASK_t mask = 0
        for p in range(2*self.dm.dofs_per_element):
            if (mask_dofs1 & (1 << p)):
                for q in range(p, 2*self.dm.dofs_per_element):
                    if (mask_dofs2 & (1 << q)):
                        mask |= k
                    k = k << 1
            else:
                k = k << (2*self.dm.dofs_per_element-p)
        return mask

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline MASK_t getElemElemSymEntryMask(self, INDEX_t cellNo1, INDEX_t cellNo2, INDEX_t I, INDEX_t J):
        # Add symmetric 'contrib' to elements i and j in symmetric fashion
        cdef:
            INDEX_t p, q, K, L
            MASK_t k = 1
            MASK_t mask = 0
        for p in range(2*self.dm.dofs_per_element):
            if p < self.dm.dofs_per_element:
                K = self.dm.cell2dof(cellNo1, p)
            else:
                K = self.dm.cell2dof(cellNo2, p-self.dm.dofs_per_element)

            for q in range(p, 2*self.dm.dofs_per_element):
                if q < self.dm.dofs_per_element:
                    L = self.dm.cell2dof(cellNo1, q)
                else:
                    L = self.dm.cell2dof(cellNo2, q-self.dm.dofs_per_element)
                if (I == K and J == L) or (J == K and I == L):
                    mask |= k
                k = k << 1
        return mask

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void addToMatrixElemElemSymMasked(self, const REAL_t[::1] contrib, REAL_t fac, MASK_t mask):
        # Add symmetric 'contrib' to elements i and j in symmetric fashion
        cdef:
            INDEX_t k, p, q, I, J
        k = 0
        for p in range(2*self.dm.dofs_per_element):
            I = self.localDoFs[p]
            if mask & (1 << k):
                self.A.addToEntry(I, I, fac*contrib[k])
            k += 1
            for q in range(p+1, 2*self.dm.dofs_per_element):
                if mask & (1 << k):
                    J = self.localDoFs[q]
                    self.A.addToEntry(I, J, fac*contrib[k])
                    self.A.addToEntry(J, I, fac*contrib[k])
                k += 1

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void addToMatrixElemElemMasked(self, const REAL_t[::1] contrib, REAL_t fac, MASK_t mask):
        # Add symmetric 'contrib' to elements i and j in symmetric fashion
        cdef:
            INDEX_t k, p, q, I, J
        k = 0
        for p in range(2*self.dm.dofs_per_element):
            I = self.localDoFs[p]
            for q in range(2*self.dm.dofs_per_element):
                J = self.localDoFs[q]
                if mask & (1 << k):
                    self.A.addToEntry(I, J, fac*contrib[k])
                k += 1

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void addToCache(self, REAL_t[::1] contrib, INDEX_t[::1] ID, INDEX_t perm, BOOL_t inv=False):
        cdef:
            intTuple hv = intTuple.create(ID)
        contribNew = uninitialized((contrib.shape[0]), dtype=REAL)
        self.permute(contrib, contribNew, perm, inv)
        self.cache[hv] = contribNew

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void permute(self, REAL_t[::1] contrib, REAL_t[::1] contribNew, INDEX_t perm, BOOL_t inv=False):
        cdef:
            INDEX_t K, p, q
            INDEX_t k, i, j
            INDEX_t dofs_per_element = self.dm.dofs_per_element
            INDEX_t dofs_per_element2 = 2*dofs_per_element
            BOOL_t perm0 = perm & 1
            INDEX_t perm1 = (perm >> 1) & 3
            INDEX_t perm2 = (perm >> 3) & 3
            INDEX_t[::1] permutedDoFsLocal = self.permutedDoFsLocal
        if inv and self.dm.dim == 2:
            if perm1 == 1:
                perm1 = 2
            elif perm1 == 2:
                perm1 = 1

            if perm2 == 1:
                perm2 = 2
            elif perm2 == 2:
                perm2 = 1
            if perm0:
                perm1, perm2 = perm2, perm1

        for p in range(dofs_per_element2):
            if perm0:
                i = p+dofs_per_element
                if i >= dofs_per_element2:
                    i -= dofs_per_element2
            else:
                i = p
            if (i < dofs_per_element):
                i = self.idxCellFlip[perm1, i]
            else:
                i = dofs_per_element + self.idxCellFlip[perm2, i-dofs_per_element]
            permutedDoFsLocal[p] = i

        K = 0
        for p in range(dofs_per_element2):
            i = permutedDoFsLocal[p]

            k = 2*dofs_per_element*i-(i*(i+1) >> 1) + i
            contribNew[K] = contrib[k]
            K += 1

            for q in range(p+1, dofs_per_element2):
                j = permutedDoFsLocal[q]

                if i > j:
                    k = dofs_per_element2*j-(j*(j+1) >> 1) + i
                else:
                    k = dofs_per_element2*i-(i*(i+1) >> 1) + j
                contribNew[K] = contrib[k]
                K += 1

    
    def __repr__(self):
        s = ''
        s += 'Cache size: {}'.format(len(self.cache))
        return s


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline MASK_t getElemSymMask(DoFMap DoFMap, INDEX_t cellNo1, INDEX_t I, INDEX_t J):
    # Add symmetric 'contrib' to elements i and j in symmetric fashion
    cdef:
        INDEX_t p, q, K, L
        MASK_t k = 1
        MASK_t mask = 0
    for p in range(DoFMap.dofs_per_element):
        K = DoFMap.cell2dof(cellNo1, p)
        for q in range(p, DoFMap.dofs_per_element):
            L = DoFMap.cell2dof(cellNo1, q)
            if (I == K and J == L) or (J == K and I == L):
                mask |= k
            k = k << 1
    return mask


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline MASK_t getElemElemSymMask(DoFMap DoFMap, INDEX_t cellNo1, INDEX_t cellNo2, INDEX_t I, INDEX_t J):
    # Add symmetric 'contrib' to elements i and j in symmetric fashion
    cdef:
        INDEX_t p, q, K, L
        MASK_t k = 1
        MASK_t mask = 0
    for p in range(2*DoFMap.dofs_per_element):
        if p < DoFMap.dofs_per_element:
            K = DoFMap.cell2dof(cellNo1, p)
        else:
            K = DoFMap.cell2dof(cellNo2, p-DoFMap.dofs_per_element)

        for q in range(p, 2*DoFMap.dofs_per_element):
            if q < DoFMap.dofs_per_element:
                L = DoFMap.cell2dof(cellNo1, q)
            else:
                L = DoFMap.cell2dof(cellNo2, q-DoFMap.dofs_per_element)
            if (I == K and J == L) or (J == K and I == L):
                mask |= k
            k = k << 1
    return mask


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline REAL_t extractElemSymMasked(DoFMap DoFMap, const REAL_t[::1] contrib, REAL_t fac, MASK_t mask):
    # Add symmetric 'contrib' to elements i and j in symmetric fashion
    cdef:
        INDEX_t k, p, q
        REAL_t s = 0.
    k = 0
    for p in range(DoFMap.dofs_per_element):
        for q in range(p, DoFMap.dofs_per_element):
            if mask & (1 << k):
                s += fac*contrib[k]
            k += 1
    return s


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline REAL_t extractElemElemSymMasked(DoFMap DoFMap, const REAL_t[::1] contrib, REAL_t fac, MASK_t mask):
    # Add symmetric 'contrib' to elements i and j in symmetric fashion
    cdef:
        INDEX_t k, p, q
        REAL_t s = 0.
    k = 0
    for p in range(2*DoFMap.dofs_per_element):
        for q in range(p, 2*DoFMap.dofs_per_element):
            if mask & (1 << k):
                s += fac*contrib[k]
            k += 1
    return s


cdef class nonlocalBuilder:
    def __init__(self,
                 meshBase mesh,
                 DoFMap dm,
                 Kernel kernel,
                 dict params={},
                 bint zeroExterior=True,
                 MPI.Comm comm=None,
                 FakePLogger PLogger=None,
                 DoFMap dm2=None,
                 **kwargs):
        if 'boundary' in kwargs:
            warnings.warn('"boundary" parameter deprecated', DeprecationWarning)
            zeroExterior = kwargs['boundary']

        self.dm = dm
        self.mesh = self.dm.mesh
        assert self.dm.mesh == mesh
        if dm2 is not None:
            self.dm2 = dm2
            assert type(self.dm) == type(self.dm2)
            assert self.dm.mesh == self.dm2.mesh
        self.kernel = kernel
        if self.kernel.finiteHorizon:
            self.zeroExterior = False
        else:
            self.zeroExterior = zeroExterior
        self.comm = comm
        self.params = params

        assert isinstance(self.kernel.horizon, constant)
        assert kernel.dim == mesh.dim
        assert kernel.dim == dm.mesh.dim

        # volume integral
        self.local_matrix = self.getLocalMatrix(params)

        if self.local_matrix.symmetricLocalMatrix:
            self.contrib = uninitialized(((2*self.dm.dofs_per_element)*(2*self.dm.dofs_per_element+1)//2), dtype=REAL)
        else:
            self.contrib = uninitialized(((2*self.dm.dofs_per_element)**2), dtype=REAL)

        self.local_matrix.setMesh1(self.dm.mesh)
        if self.dm2 is None:
            self.local_matrix.setMesh2(self.dm.mesh)
        else:
            self.local_matrix.setMesh2(self.dm2.mesh)

        LOGGER.debug(self.local_matrix)

        # surface integrals
        self.local_matrix_zeroExterior = self.getLocalMatrixBoundaryZeroExterior(params, infHorizon=True)
        self.local_matrix_surface = self.getLocalMatrixBoundaryZeroExterior(params, infHorizon=False)

        if self.local_matrix_zeroExterior is not None:
            self.local_matrix_zeroExterior.setMesh1(self.dm.mesh)
            self.local_matrix_surface.setMesh1(self.dm.mesh)
            if self.local_matrix_zeroExterior.symmetricLocalMatrix:
                self.contribZeroExterior = uninitialized((self.dm.dofs_per_element*(self.dm.dofs_per_element+1)//2), dtype=REAL)
            else:
                self.contribZeroExterior = uninitialized(((self.dm.dofs_per_element)**2), dtype=REAL)
            LOGGER.debug(self.local_matrix_zeroExterior)
            LOGGER.debug(self.local_matrix_surface)
        else:
            self.contribZeroExterior = uninitialized((0), dtype=REAL)

        if PLogger is not None:
            self.PLogger = PLogger
        else:
            self.PLogger = FakePLogger()

    @property
    def d2c(self):
        if self._d2c is None:
            self._d2c = self.dm.getPatchLookup()
        return self._d2c

    cdef inline double_local_matrix_t getLocalMatrix(self, dict params):
        cdef:
            bint symmetric, genKernel, forceNonSym
            fractionalOrderBase s
        target_order = params.get('target_order', None)
        genKernel = params.get('genKernel', False)
        forceNonSym = params.get('forceNonSym', False)
        autoQuad = params.get('automaticQuadrature', False)
        symmetric = not forceNonSym and self.kernel.symmetric
        if genKernel:
             LOGGER.warning('General kernel not implemented for boundary term')
        elif isinstance(self.kernel, FractionalKernel):
            s = self.kernel.s
            assert ((s.min < 1.) and (s.max < 1.)) or ((s.min > 1.) and (s.max > 1.)), "smin={}, smax={} not supported".format(s.min, s.max)

            if isinstance(self.dm, P0_DoFMap):
                if self.mesh.dim == 1:
                    if s.min > 0. and s.max < 0.5:
                        local_matrix = fractionalLaplacian1D_P0(self.kernel,
                                                                mesh=self.mesh,
                                                                DoFMap=self.dm,
                                                                target_order=target_order)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif isinstance(self.dm, P1_DoFMap):
                if self.mesh.dim == 1:
                    if s.min > 0. and s.max < 1.:
                        if symmetric:
                            if autoQuad:
                                 raise NotImplementedError()
                            else:
                                local_matrix = fractionalLaplacian1D_P1(self.kernel,
                                                                        mesh=self.mesh,
                                                                        DoFMap=self.dm,
                                                                        target_order=target_order)
                        else:
                             raise NotImplementedError()
                    
                    else:
                        raise NotImplementedError(self.kernel)
                elif self.mesh.dim == 2:
                    if s.min > 0. and s.max < 1.:
                        if symmetric:
                            if autoQuad:
                                 raise NotImplementedError()
                            else:
                                local_matrix = fractionalLaplacian2D_P1(self.kernel,
                                                                        mesh=self.mesh,
                                                                        DoFMap=self.dm,
                                                                        target_order=target_order)
                        else:
                            raise NotImplementedError(self.kernel)
                    
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            
            else:
                raise NotImplementedError()
        else:
            if self.mesh.dim == 1:
                local_matrix = integrable1D(self.kernel,
                                            mesh=self.mesh,
                                            DoFMap=self.dm,
                                            target_order=target_order)
            elif self.mesh.dim == 2:
                local_matrix = integrable2D(self.kernel,
                                            mesh=self.mesh,
                                            DoFMap=self.dm,
                                            target_order=target_order)
            else:
                raise NotImplementedError()
        return local_matrix

    cdef inline double_local_matrix_t getLocalMatrixBoundaryZeroExterior(self, dict params, BOOL_t infHorizon):
        cdef:
            bint genKernel
            fractionalOrderBase s
        target_order = params.get('target_order', None)
        genKernel = params.get('genKernel', False)
        if isinstance(self.kernel, FractionalKernel):
            s = self.kernel.s
            assert ((s.min < 1.) and (s.max < 1.)) or ((s.min > 1.) and (s.max > 1.))
            assert isinstance(self.kernel.horizon, constant)
            if infHorizon:
                kernelInfHorizon = self.kernel.getModifiedKernel(horizon=constant(np.inf))
            else:
                kernelInfHorizon = self.kernel
            if genKernel:
                LOGGER.warning('General kernel not implemented for boundary term')
            if isinstance(self.dm, P0_DoFMap):
                if self.mesh.dim == 1:
                    if s.min > 0. and s.max < 0.5:
                        local_matrix = fractionalLaplacian1D_P0_boundary(kernelInfHorizon,
                                                                         mesh=self.mesh,
                                                                         DoFMap=self.dm,
                                                                         target_order=target_order)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif isinstance(self.dm, P1_DoFMap):
                if self.mesh.dim == 1:
                    if s.min > 0. and s.max < 1.:
                        local_matrix = fractionalLaplacian1D_P1_boundary(kernelInfHorizon,
                                                                         mesh=self.mesh,
                                                                         DoFMap=self.dm,
                                                                         target_order=target_order)
                    
                    else:
                        raise NotImplementedError()
                elif self.mesh.dim == 2:
                    if s.min > 0. and s.max < 1.:
                        local_matrix = fractionalLaplacian2D_P1_boundary(kernelInfHorizon,
                                                                         mesh=self.mesh,
                                                                         DoFMap=self.dm,
                                                                         target_order=target_order)
                    
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            
            else:
                raise NotImplementedError()
        else:
            local_matrix = None
        return local_matrix

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getDense(self, BOOL_t trySparsification=False):
        cdef:
            INDEX_t cellNo1, cellNo2
            LinearOperator A = None
            REAL_t[::1] contrib = self.contrib, contribZeroExterior = self.contribZeroExterior
            INDEX_t start, end
            meshBase surface
            IndexManager iM
            INDEX_t i, j, explicitZerosRow
            np.int64_t explicitZeros
            REAL_t[:, ::1] data
            REAL_t sparsificationThreshold = 0.8
            BOOL_t symmetricLocalMatrix = self.local_matrix.symmetricLocalMatrix
            BOOL_t symmetricCells = self.local_matrix.symmetricCells

        if self.comm:
            start = <INDEX_t>np.ceil(self.mesh.num_cells*self.comm.rank/self.comm.size)
            end = <INDEX_t>np.ceil(self.mesh.num_cells*(self.comm.rank+1)/self.comm.size)
        else:
            start = 0
            end = self.mesh.num_cells

        if (trySparsification
            and (self.comm is None or self.comm.size == 1)
            and not self.zeroExterior
            and self.dm2 is None
            and self.kernel.finiteHorizon
            and (self.mesh.volume*(1.-sparsificationThreshold) > self.kernel.getHorizonValue()**self.mesh.dim)):

            with self.PLogger.Timer('build sparsity pattern'):

                sP = sparsityPattern(self.dm.num_dofs)
                iM = IndexManager(self.dm, None, sP=sP)

                for cellNo1 in range(start, end):
                    self.local_matrix.setCell1(cellNo1)
                    for cellNo2 in range(cellNo1, self.mesh.num_cells):
                        self.local_matrix.setCell2(cellNo2)
                        if iM.getDoFsElemElem(cellNo1, cellNo2):
                            continue
                        panel = self.local_matrix.getPanelType()
                        if cellNo1 == cellNo2:
                            if panel != IGNORED:
                                if self.local_matrix.symmetricLocalMatrix:
                                    iM.addToSparsityElemElemSym()
                                else:
                                    iM.addToSparsityElemElem()
                        else:
                            if self.local_matrix.symmetricCells:
                                if panel != IGNORED:
                                    if self.local_matrix.symmetricLocalMatrix:
                                        iM.addToSparsityElemElemSym()
                                    else:
                                        iM.addToSparsityElemElem()
                            else:
                                if panel != IGNORED:
                                    if self.local_matrix.symmetricLocalMatrix:
                                        iM.addToSparsityElemElemSym()
                                    else:
                                        iM.addToSparsityElemElem()
                                self.local_matrix.swapCells()
                                panel = self.local_matrix.getPanelType()
                                if panel != IGNORED:
                                    if iM.getDoFsElemElem(cellNo2, cellNo1):
                                        continue
                                    if self.local_matrix.symmetricLocalMatrix:
                                        iM.addToSparsityElemElemSym()
                                    else:
                                        iM.addToSparsityElemElem()
                                self.local_matrix.swapCells()
                indptr, indices = sP.freeze()
                useSymmetricMatrix = self.local_matrix.symmetricLocalMatrix and self.local_matrix.symmetricCells
                if useSymmetricMatrix:
                    A = SSS_LinearOperator(indices, indptr,
                                           np.zeros((indices.shape[0]), dtype=REAL),
                                           np.zeros((self.dm.num_dofs), dtype=REAL))
                    ratio = ((A.nnz+A.num_rows)/REAL(A.num_rows))/REAL(A.num_columns)
                else:
                    A = CSR_LinearOperator(indices, indptr,
                                           np.zeros((indices.shape[0]), dtype=REAL))
                    ratio = (A.nnz/REAL(A.num_rows))/REAL(A.num_columns)
                LOGGER.warning('Assembling into sparse{} matrix, since {}% of entries are zero.'.format(', symmetric' if useSymmetricMatrix else '',
                                                                                                     100.*(1.-ratio)))
                trySparsification = False
        else:
            if self.dm2 is None:
                A = Dense_LinearOperator(np.zeros((self.dm.num_dofs, self.dm.num_dofs), dtype=REAL))
            else:
                A = Dense_LinearOperator(np.zeros((self.dm.num_dofs, self.dm2.num_dofs), dtype=REAL))

        if self.dm2 is None:
            iM = IndexManager(self.dm, A)
        else:
            LOGGER.warning('Efficiency of assembly with 2 DoFMaps is bad.')
            dmCombined = self.dm.combine(self.dm2)
            B = SubMatrixAssemblyOperator(A,
                                          np.arange(self.dm.num_dofs, dtype=INDEX),
                                          np.arange(self.dm.num_dofs, self.dm.num_dofs+self.dm2.num_dofs, dtype=INDEX))
            iM = IndexManager(dmCombined, B)

        # Omega x Omega
        with self.PLogger.Timer('interior'):
            for cellNo1 in range(start, end):
                self.local_matrix.setCell1(cellNo1)
                for cellNo2 in range(cellNo1, self.mesh.num_cells):
                    self.local_matrix.setCell2(cellNo2)
                    if iM.getDoFsElemElem(cellNo1, cellNo2):
                        continue
                    panel = self.local_matrix.getPanelType()
                    if cellNo1 == cellNo2:
                        if panel != IGNORED:
                            self.local_matrix.eval(contrib, panel)
                            if symmetricLocalMatrix:
                                iM.addToMatrixElemElemSym(contrib, 1.)
                            else:
                                iM.addToMatrixElemElem(contrib, 1.)
                    else:
                        if symmetricCells:
                            if panel != IGNORED:
                                self.local_matrix.eval(contrib, panel)
                                # If the kernel is symmetric, the contributions from (cellNo1, cellNo2) and (cellNo2, cellNo1)
                                # are the same. We multiply by 2 to account for the contribution from cells (cellNo2, cellNo1).
                                if symmetricLocalMatrix:
                                    iM.addToMatrixElemElemSym(contrib, 2.)
                                else:
                                    iM.addToMatrixElemElem(contrib, 2.)
                        else:
                            if panel != IGNORED:
                                self.local_matrix.eval(contrib, panel)
                                if symmetricLocalMatrix:
                                    iM.addToMatrixElemElemSym(contrib, 1.)
                                else:
                                    iM.addToMatrixElemElem(contrib, 1.)
                            self.local_matrix.swapCells()
                            panel = self.local_matrix.getPanelType()
                            if panel != IGNORED:
                                if iM.getDoFsElemElem(cellNo2, cellNo1):
                                    continue
                                self.local_matrix.eval(contrib, panel)
                                if symmetricLocalMatrix:
                                    iM.addToMatrixElemElemSym(contrib, 1.)
                                else:
                                    iM.addToMatrixElemElem(contrib, 1.)
                            self.local_matrix.swapCells()

        # Omega x Omega^C
        if self.zeroExterior:
            with self.PLogger.Timer('zeroExterior'):
                surface = self.mesh.get_surface_mesh()

                self.local_matrix_zeroExterior.setMesh2(surface)

                for cellNo1 in range(start, end):
                    iM.getDoFsElem(cellNo1)
                    self.local_matrix_zeroExterior.setCell1(cellNo1)
                    for cellNo2 in range(surface.num_cells):
                        self.local_matrix_zeroExterior.setCell2(cellNo2)
                        panel = self.local_matrix_zeroExterior.getPanelType()
                        self.local_matrix_zeroExterior.eval(contribZeroExterior, panel)
                        # if local_matrix_zeroExterior.symmetricLocalMatrix:
                        iM.addToMatrixElemSym(contribZeroExterior, 1.)
                        # else:
                        #     raise NotImplementedError()
        if self.comm:
            self.comm.Allreduce(MPI.IN_PLACE, A.data)
        if trySparsification:
            explicitZeros = 0
            data = A.data
            nr = A.num_rows
            for i in range(A.num_rows):
                explicitZerosRow = 0
                for j in range(A.num_columns):
                    if data[i, j] == 0.:
                        explicitZerosRow += 1
                explicitZeros += explicitZerosRow
                if not (explicitZerosRow > sparsificationThreshold*A.num_columns):
                    nr = i+1
                    break
            ratio = (explicitZeros/REAL(nr))/REAL(A.num_columns)
            if ratio > sparsificationThreshold:
                LOGGER.warning('Converting dense to sparse matrix, since {}% of entries are zero.'.format(100.*ratio))
                return CSR_LinearOperator.from_dense(A)
            else:
                LOGGER.warning('Not converting dense to sparse matrix, since only {}% of entries are zero.'.format(100.*ratio))
        return A

    
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef REAL_t getEntryCluster(self, INDEX_t I, INDEX_t J):
        cdef:
            tree_node n1, n2, n3
            list clusters = []
            nearFieldClusterPair c1, c2, c3
            arrayIndexSet aI1, aI2, aI3
            REAL_t[:, :, ::1] fake_boxes = uninitialized((0, 0, 0), dtype=REAL)
            INDEX_t[::1] I_view = np.array([I], dtype=INDEX)
            INDEX_t[::1] J_view = np.array([J], dtype=INDEX)
            arrayIndexSetIterator it = arrayIndexSetIterator()
            list d2c = self.d2c
            LinearOperator A
            REAL_t[:, ::1] mat = np.zeros((1, 1), dtype=REAL)
        if I == J:
            aI3 = arrayIndexSet(I_view)
            n3 = tree_node(None, aI3, fake_boxes)

            cells = set()
            it.setIndexSet(aI3)
            while it.step():
                cells |= d2c[it.i]
            n3._cells = arrayIndexSet()
            n3._cells.fromSet(cells)

            c3 = nearFieldClusterPair(n3, n3)
            clusters.append(c3)
        else:
            aI1 = arrayIndexSet(I_view)
            aI2 = arrayIndexSet(J_view)
            n1 = tree_node(None, aI1, fake_boxes)
            n2 = tree_node(None, aI2, fake_boxes)

            cells = set()
            it.setIndexSet(aI1)
            while it.step():
                cells |= d2c[it.i]
            n1._cells = arrayIndexSet()
            n1._cells.fromSet(cells)

            cells = set()
            it.setIndexSet(aI2)
            while it.step():
                cells |= d2c[it.i]
            n2._cells = arrayIndexSet()
            n2._cells.fromSet(cells)

            c1 = nearFieldClusterPair(n1, n2)
            c2 = nearFieldClusterPair(n2, n1)
            clusters.append(c1)
            clusters.append(c2)
        A = Dense_SubBlock_LinearOperator(I_view,
                                          J_view,
                                          self.dm.num_dofs,
                                          self.dm.num_dofs,
                                          mat)
        self.assembleClusters(clusters, Anear=A)
        return mat[0, 0]

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef REAL_t getEntry(self, INDEX_t I, INDEX_t J):
        cdef:
            INDEX_t cellNo1, cellNo2
            INDEX_t[:,::1] surface_cells
            MASK_t mask
            indexSet cellsUnion = arrayIndexSet()
            indexSet cellsInter = arrayIndexSet()
            indexSetIterator it1 = arrayIndexSetIterator()
            indexSetIterator it2 = arrayIndexSetIterator()
            dm = self.dm
            REAL_t entry = 0.
        cellsUnion.fromSet(self.d2c[I] | self.d2c[J])
        cellsInter.fromSet(self.d2c[I] & self.d2c[J])

        assert isinstance(self.kernel.horizon, constant) and self.kernel.horizon.value == np.inf

        # (supp phi_I \cup supp phi_J)^2
        it1.setIndexSet(cellsUnion)
        it2.setIndexSet(cellsUnion)
        while it1.step():
            cellNo1 = it1.i
            self.local_matrix.setCell1(cellNo1)
            it2.reset()
            while it2.step():
                cellNo2 = it2.i
                if cellNo2 < cellNo1:
                    continue
                mask = getElemElemSymMask(dm, cellNo1, cellNo2, I, J)
                if mask == 0:
                    continue
                self.local_matrix.setCell2(cellNo2)
                panel = self.local_matrix.getPanelType()
                if cellNo1 == cellNo2:
                    self.local_matrix.eval(self.contrib, panel, mask)
                    if self.local_matrix.symmetricLocalMatrix:
                        entry += extractElemElemSymMasked(dm, self.contrib, 1., mask)
                    else:
                        raise NotImplementedError()
                else:
                    if self.local_matrix.symmetricCells:
                        if panel != IGNORED:
                            self.local_matrix.eval(self.contrib, panel, mask)
                            # multiply by 2 to account for the contribution from cells (cellNo2, cellNo1)
                            if self.local_matrix.symmetricLocalMatrix:
                                entry += extractElemElemSymMasked(dm, self.contrib, 2., mask)
                            else:
                                raise NotImplementedError()
                    else:
                        if panel != IGNORED:
                            self.local_matrix.eval(self.contrib, panel, mask)
                            # multiply by 2 to account for the contribution from cells (cellNo2, cellNo1)
                            if self.local_matrix.symmetricLocalMatrix:
                                entry += extractElemElemSymMasked(dm, self.contrib, 1., mask)
                            else:
                                raise NotImplementedError()
                        self.local_matrix.swapCells()
                        mask = getElemElemSymMask(dm, cellNo2, cellNo1, I, J)
                        panel = self.local_matrix.getPanelType()
                        if panel != IGNORED:
                            self.local_matrix.eval(self.contrib, panel, mask)
                            if self.local_matrix.symmetricLocalMatrix:
                                entry += extractElemElemSymMasked(dm, self.contrib, 1., mask)
                            else:
                                raise NotImplementedError()
        # (supp phi_I \cup supp phi_J) x (supp phi_I \cup supp phi_J)^C
        if not self.kernel.variable:
            if self.zeroExterior:
                # zeroExterior of (supp phi_I \cup supp phi_J)
                if self.mesh.dim == 1:
                    surface_cells = boundaryVertices(self.mesh.cells, cellsUnion)
                elif self.mesh.dim == 2:
                    surface_cells = boundaryEdges(self.mesh.cells, cellsUnion)
                else:
                    raise NotImplementedError()

                self.local_matrix_zeroExterior.setVerticesCells2(self.mesh.vertices, surface_cells)

                it1.setIndexSet(cellsInter)
                while it1.step():
                    cellNo1 = it1.i
                    self.local_matrix_zeroExterior.setCell1(cellNo1)
                    mask = getElemSymMask(dm, cellNo1, I, J)
                    for cellNo2 in range(surface_cells.shape[0]):
                        self.local_matrix_zeroExterior.setCell2(cellNo2)
                        panel = self.local_matrix_zeroExterior.getPanelType()
                        self.local_matrix_zeroExterior.eval(self.contribZeroExterior, panel)
                        entry += extractElemSymMasked(dm, self.contribZeroExterior, 1., mask)
        else:
            # (supp phi_I \cup supp phi_J) x (Omega \ (supp phi_I \cup supp phi_J))
            # TODO: This can be done using surface integrals instead
            it1.setIndexSet(cellsUnion)
            while it1.step():
                cellNo1 = it1.i
                self.local_matrix.setCell1(cellNo1)

                for cellNo2 in set(range(self.mesh.num_cells))-cellsUnion.toSet():
                    self.local_matrix.setCell2(cellNo2)
                    mask = getElemElemSymMask(dm, cellNo1, cellNo2, I, J)
                    panel = self.local_matrix.getPanelType()
                    if panel != IGNORED:
                        if self.local_matrix.symmetricLocalMatrix:
                            # multiply by 2 to account for the 2 symmetric contributions
                            self.local_matrix.eval(self.contrib, panel)
                            entry += extractElemElemSymMasked(dm, self.contrib, 1., mask)
                        else:
                            raise NotImplementedError()

            if self.zeroExterior:
                # (supp phi_I \cup supp phi_J) x Omega^C
                surface = self.mesh.get_surface_mesh()
                self.local_matrix_zeroExterior.setMesh2(surface)

                it1.setIndexSet(cellsInter)
                while it1.step():
                    cellNo1 = it1.i
                    self.local_matrix_zeroExterior.setCell1(cellNo1)
                    mask = getElemSymMask(dm, cellNo1, I, J)
                    for cellNo2 in range(surface.num_cells):
                        self.local_matrix_zeroExterior.setCell2(cellNo2)
                        panel = self.local_matrix_zeroExterior.getPanelType()
                        self.local_matrix_zeroExterior.eval(self.contribZeroExterior, panel)
                        entry += extractElemSymMasked(dm, self.contribZeroExterior, 1., mask)
        return entry

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef LinearOperator assembleClusters(self, list Pnear, bint forceUnsymmetric=False, LinearOperator Anear=None, dict jumps={}, indexSet myDofs=None, str prefix=''):
        cdef:
            INDEX_t cellNo1, cellNo2, cellNo3
            REAL_t fac
            REAL_t[::1] contrib = self.contrib, contribZeroExterior = self.contribZeroExterior
            meshBase surface
            INDEX_t[:, ::1] cells = self.mesh.cells, surface_cells, fake_cells
            indexSet cellsInter
            indexSet clusterDofs1, clusterDofs2
            FilteredAssemblyOperator Anear_filtered = None
            INDEX_t[::1] cellPair = uninitialized((2), dtype=INDEX)
            nearFieldClusterPair cluster
            panelType panel
            tupleDictMASK masks = None
            ENCODE_t hv, hv2
            MASK_t mask = 0
            # INDEX_t vertex1, vertex2
            bint useSymmetricMatrix
            bint useSymmetricCells
            INDEX_t vertexNo, i
            INDEX_t[::1] edge = uninitialized((2), dtype=INDEX)
            REAL_t evalShift = 1e-9
            local_matrix_t mass
            indexSetIterator it = arrayIndexSetIterator()
            INDEX_t startCluster

        if Anear is None:
            useSymmetricMatrix = self.local_matrix.symmetricLocalMatrix and self.local_matrix.symmetricCells and not forceUnsymmetric
            with self.PLogger.Timer(prefix+'build near field sparsity pattern'):
                Anear = getSparseNearField(self.dm, Pnear, symmetric=useSymmetricMatrix)
            LOGGER.info('Anear: {}'.format(Anear))

        Anear_filtered = FilteredAssemblyOperator(Anear)

        useSymmetricCells = self.local_matrix.symmetricCells

        iM = IndexManager(self.dm, Anear)

        use_masks = self.params.get('use_masks', True)

        with self.PLogger.Timer(prefix+'interior'):
            # This corresponds to
            #  C(d,s) \int_D \int_D (u(x)-u(y)) (v(x)-v(y)) /|x-y|^{d+2s}
            # where
            #  D = (supp u) \cup (supp v).,
            # We only update unknows that are in the cluster pair.

            if not use_masks:
                # This loop does the correct thing, but we are wasting a lot of
                # element x element evaluations.
                for cluster in Pnear:
                    cellsUnion = cluster.cellsUnion

                    clusterDofs1 = cluster.n1.get_dofs()
                    clusterDofs2 = cluster.n2.get_dofs()
                    Anear_filtered.setFilter(clusterDofs1, clusterDofs2)
                    iM = IndexManager(self.dm, Anear_filtered)

                    for cellNo1 in cellsUnion:
                        self.local_matrix.setCell1(cellNo1)
                        for cellNo2 in cellsUnion:
                            self.local_matrix.setCell2(cellNo2)
                            panel = self.local_matrix.getPanelType()
                            if panel != IGNORED:
                                if useSymmetricCells and (cellNo1 != cellNo2):
                                    fac = 2.
                                else:
                                    fac = 1.
                                if iM.getDoFsElemElem(cellNo1, cellNo2):
                                    continue
                                self.local_matrix.eval(contrib, panel)
                                if useSymmetricCells:
                                    iM.addToMatrixElemElemSym(contrib, fac)
                                else:
                                    raise NotImplementedError()
            else:
                # Pre-record all element x element contributions.
                # This way, we only assembly over each element x element pair once.
                # We load balance the cells and only get the list for the local rank.
                assert contrib.shape[0] <= 64, "Mask type is not large enough for {} entries".format(contrib.shape[0])
                startCluster = 0
                while startCluster < len(Pnear):
                    with self.PLogger.Timer(prefix+'interior - build masks'):
                        masks = iM.buildMasksForClusters(Pnear, self.local_matrix.symmetricCells, &startCluster)

                    if (masks.getSizeInBytes() >> 20) > 20:
                        LOGGER.info('element x element pairs {}, {} MB'.format(masks.nnz, masks.getSizeInBytes() >> 20))
                    # Compute all element x element contributions
                    with self.PLogger.Timer(prefix+'interior - compute'):
                        masks.startIter()
                        while masks.next(cellPair, &mask):
                            # decode_edge(hv, cellPair)
                            cellNo1 = cellPair[0]
                            cellNo2 = cellPair[1]
                            self.local_matrix.setCell1(cellNo1)
                            self.local_matrix.setCell2(cellNo2)
                            panel = self.local_matrix.getPanelType()
                            if panel != IGNORED:
                                if useSymmetricCells and (cellNo1 != cellNo2):
                                    fac = 2.
                                else:
                                    fac = 1.
                                if iM.getDoFsElemElem(cellNo1, cellNo2):
                                    continue
                                self.local_matrix.eval(contrib, panel, mask)
                                if useSymmetricCells:
                                    iM.addToMatrixElemElemSymMasked(contrib, fac, mask)
                                else:
                                    raise NotImplementedError()
                        masks = None

        if not self.kernel.variable:
            if not self.kernel.complement:
                with self.PLogger.Timer(prefix+'cluster zeroExterior'):
                    # This corresponds to
                    #  C(d,s)/(2s) \int_D u(x) v(x) \int_E n.(x-y)/|x-y|^{d+2s}
                    # where
                    #  D = (supp u) \cap (supp v) \subset E,
                    #  E = \partial((supp u) \cap (supp v)).
                    # We only update unknows that are in the cluster pair.

                    iM = IndexManager(self.dm, Anear_filtered)

                    for cluster in Pnear:

                        cellsInter = cluster.cellsInter
                        if len(cellsInter) == 0:
                            continue

                        clusterDofs1 = cluster.n1.get_dofs()
                        clusterDofs2 = cluster.n2.get_dofs()

                        # surface of the union of clusters n1 and n2
                        if self.mesh.dim == 1:
                            surface_cells = boundaryVertices(cells, cluster.cellsUnion)
                        elif self.mesh.dim == 2:
                            surface_cells = boundaryEdges(cells, cluster.cellsUnion)
                        else:
                            raise NotImplementedError()

                        Anear_filtered.setFilter(clusterDofs1, clusterDofs2)

                        self.local_matrix_zeroExterior.setVerticesCells2(self.mesh.vertices, surface_cells)

                        it.setIndexSet(cellsInter)
                        while it.step():
                            cellNo1 = it.i
                            self.local_matrix_zeroExterior.setCell1(cellNo1)
                            iM.getDoFsElem(cellNo1)
                            for cellNo2 in range(surface_cells.shape[0]):
                                self.local_matrix_zeroExterior.setCell2(cellNo2)
                                panel = self.local_matrix_zeroExterior.getPanelType()
                                self.local_matrix_zeroExterior.eval(contribZeroExterior, panel)
                                if self.local_matrix_zeroExterior.symmetricLocalMatrix:
                                    iM.addToMatrixElemSym(contribZeroExterior, 1.)
                                else:
                                    raise NotImplementedError()
                if not self.zeroExterior and not self.kernel.finiteHorizon:
                    with self.PLogger.Timer(prefix+'zeroExterior'):
                        # Subtract the zeroExterior contribution for Omega x Omega^C that was added in the previous loop.
                        # This is for the regional fractional Laplacian.
                        surface = self.mesh.get_surface_mesh()
                        iM = IndexManager(self.dm, Anear, myDofs=myDofs)

                        self.local_matrix_zeroExterior.setMesh2(surface)

                        for cellNo1 in range(self.mesh.num_cells):
                            self.local_matrix_zeroExterior.setCell1(cellNo1)
                            iM.getDoFsElem(cellNo1)
                            for cellNo2 in range(surface.num_cells):
                                self.local_matrix_zeroExterior.setCell2(cellNo2)
                                panel = self.local_matrix_zeroExterior.getPanelType()
                                self.local_matrix_zeroExterior.eval(contribZeroExterior, panel)
                                if self.local_matrix_zeroExterior.symmetricLocalMatrix:
                                    iM.addToMatrixElemSym(contribZeroExterior, -1.)
                                else:
                                    raise NotImplementedError()
                elif not self.zeroExterior and self.kernel.finiteHorizon:
                    with self.PLogger.Timer(prefix+'zeroExterior'):
                        # Subtract the zeroExterior contribution for Omega x Omega^C that was added in the previous loop.
                        # This is for the regional fractional Laplacian.

                        if self.mesh.dim == 1:
                            vol = 2
                        elif self.mesh.dim == 2:
                            vol = 2*np.pi * self.kernel.horizonValue
                        else:
                            raise NotImplementedError()
                        coeff = constant(-vol*self.kernel.scalingValue*pow(self.kernel.horizonValue, 1-self.mesh.dim-2*self.kernel.sValue)/self.kernel.sValue)
                        qr = simplexXiaoGimbutas(2, self.mesh.dim)
                        if self.mesh.dim == 1:
                            mass = mass_1d_sym_scalar_anisotropic(coeff, self.dm, qr)
                        elif self.mesh.dim == 2:
                            mass = mass_2d_sym_scalar_anisotropic(coeff, self.dm, qr)
                        else:
                            raise NotImplementedError()

                        if myDofs is not None:
                            Anear_filtered2 = LeftFilteredAssemblyOperator(Anear)
                            Anear_filtered2.setFilter(myDofs)
                            assembleMatrix(self.mesh, self.dm, mass, A=Anear_filtered2)
                        else:
                            assembleMatrix(self.mesh, self.dm, mass, A=Anear)

            elif self.zeroExterior and not self.kernel.complement:
                with self.PLogger.Timer(prefix+'zeroExterior'):
                    # Add the zeroExterior contribution for Omega x Omega^C.
                    surface = self.mesh.get_surface_mesh()
                    iM = IndexManager(self.dm, Anear, myDofs=myDofs)
                    self.local_matrix_zeroExterior.setMesh2(surface)

                    for cellNo1 in range(self.mesh.num_cells):
                        self.local_matrix_zeroExterior.setCell1(cellNo1)
                        iM.getDoFsElem(cellNo1)
                        for cellNo2 in range(surface.num_cells):
                            self.local_matrix_zeroExterior.setCell2(cellNo2)
                            panel = self.local_matrix_zeroExterior.getPanelType()
                            self.local_matrix_zeroExterior.eval(contribZeroExterior, panel)
                            iM.addToMatrixElemSym(contribZeroExterior, 1.)

        else:
            # This corresponds to
            #  \int_D \int_E u(x) v(x) C(d, s) / |x-y|^{d+2s}
            # where
            #  D = (supp u) \cap (supp v) \subset E,
            #  E = Omega \ ((supp u) \cup (supp v)).
            # We only update unknows that are in the cluster pair.
            with self.PLogger.Timer(prefix+'cluster exterior'):
                iM = IndexManager(self.dm, Anear_filtered)

                fake_cells = uninitialized((1, self.mesh.dim), dtype=INDEX)
                for cluster in Pnear:

                    cellsInter = cluster.cellsInter
                    if len(cellsInter) == 0:
                        continue

                    clusterDofs1 = cluster.n1.get_dofs()
                    clusterDofs2 = cluster.n2.get_dofs()

                    Anear_filtered.setFilter(clusterDofs1, clusterDofs2)

                    if not self.kernel.complement:

                        # surface of the union of clusters n1 and n2
                        if self.mesh.dim == 1:
                            surface_cells = boundaryVertices(cells, cluster.cellsUnion)
                        elif self.mesh.dim == 2:
                            surface_cells = boundaryEdges(cells, cluster.cellsUnion)
                        else:
                            raise NotImplementedError()
                        self.local_matrix_surface.setVerticesCells2(self.mesh.vertices, surface_cells)

                        it.setIndexSet(cellsInter)
                        while it.step():
                            cellNo1 = it.i
                            self.local_matrix_surface.setCell1(cellNo1)
                            iM.getDoFsElem(cellNo1)
                            for cellNo2 in range(surface_cells.shape[0]):
                                self.local_matrix_surface.setCell2(cellNo2)
                                if self.mesh.dim == 1:
                                    if self.local_matrix_surface.center1[0] < self.local_matrix_surface.center2[0]:
                                        self.local_matrix_surface.center2[0] += evalShift
                                    else:
                                        self.local_matrix_surface.center2[0] -= evalShift
                                elif self.mesh.dim == 2:
                                    self.local_matrix_surface.center2[0] += evalShift*(self.local_matrix_surface.simplex2[1, 1]-self.local_matrix_surface.simplex2[0, 1])
                                    self.local_matrix_surface.center2[1] -= evalShift*(self.local_matrix_surface.simplex2[1, 0]-self.local_matrix_surface.simplex2[0, 0])
                                panel = self.local_matrix_surface.getPanelType()
                                if panel != IGNORED:
                                    self.local_matrix_surface.eval(contribZeroExterior, panel)
                                    if self.local_matrix_surface.symmetricLocalMatrix:
                                        iM.addToMatrixElemSym(contribZeroExterior, 1.)
                                    else:
                                        raise NotImplementedError()

                    for hv in jumps:
                        decode_edge(hv, cellPair)
                        if not (cluster.cellsUnion.inSet(cellPair[0]) or
                                cluster.cellsUnion.inSet(cellPair[1])):
                            if self.mesh.dim == 1:
                                fake_cells[0, 0] = jumps[hv]
                            else:
                                hv2 = jumps[hv]
                                decode_edge(hv2, edge)
                                for vertexNo in range(self.mesh.dim):
                                    fake_cells[0, vertexNo] = edge[vertexNo]
                            self.local_matrix_surface.setVerticesCells2(self.mesh.vertices, fake_cells)
                            self.local_matrix_surface.setCell2(0)
                            if self.mesh.dim == 1:
                                self.local_matrix_surface.center2[0] += evalShift
                            elif self.mesh.dim == 2:
                                self.local_matrix_surface.center2[0] += evalShift*(self.local_matrix_surface.simplex2[1, 1]-self.local_matrix_surface.simplex2[0, 1])
                                self.local_matrix_surface.center2[1] += evalShift*(self.local_matrix_surface.simplex2[0, 0]-self.local_matrix_surface.simplex2[1, 0])

                            it.setIndexSet(cellsInter)
                            while it.step():
                                cellNo3 = it.i
                                self.local_matrix_surface.setCell1(cellNo3)
                                panel = self.local_matrix_surface.getPanelType()
                                if panel != IGNORED:
                                    if self.mesh.dim == 1:
                                        if self.local_matrix_surface.center1[0] < self.local_matrix_surface.center2[0]:
                                            fac = 1.
                                        else:
                                            fac = -1.
                                    else:
                                        fac = 1.
                                    self.local_matrix_surface.eval(contribZeroExterior, panel)
                                    iM.getDoFsElem(cellNo3)
                                    if self.local_matrix_surface.symmetricLocalMatrix:
                                        iM.addToMatrixElemSym(contribZeroExterior, fac)
                                    else:
                                        raise NotImplementedError()

                            if self.mesh.dim == 1:
                                self.local_matrix_surface.center2[0] -= 2.*evalShift
                            elif self.mesh.dim == 2:
                                self.local_matrix_surface.center2[0] -= 2.*evalShift*(self.local_matrix_surface.simplex2[1, 1]-self.local_matrix_surface.simplex2[0, 1])
                                self.local_matrix_surface.center2[1] -= 2.*evalShift*(self.local_matrix_surface.simplex2[0, 0]-self.local_matrix_surface.simplex2[1, 0])

                            it.reset()
                            while it.step():
                                cellNo3 = it.i
                                self.local_matrix_surface.setCell1(cellNo3)
                                panel = self.local_matrix_surface.getPanelType()
                                if panel != IGNORED:
                                    if self.mesh.dim == 1:
                                        if self.local_matrix_surface.center1[0] < self.local_matrix_surface.center2[0]:
                                            fac = -1.
                                        else:
                                            fac = 1.
                                    else:
                                        fac = -1.
                                    self.local_matrix_surface.eval(contribZeroExterior, panel)
                                    iM.getDoFsElem(cellNo3)
                                    if self.local_matrix_surface.symmetricLocalMatrix:
                                        iM.addToMatrixElemSym(contribZeroExterior, fac)
                                    else:
                                        raise NotImplementedError()
                if not self.zeroExterior and not self.kernel.finiteHorizon:
                    with self.PLogger.Timer(prefix+'zeroExterior'):
                        # Subtract the zeroExterior contribution for Omega x Omega^C that was added in the previous loop.
                        # This is for the regional fractional Laplacian.
                        surface = self.mesh.get_surface_mesh()
                        iM = IndexManager(self.dm, Anear, myDofs=myDofs)

                        self.local_matrix_zeroExterior.setMesh2(surface)

                        for cellNo1 in range(self.mesh.num_cells):
                            self.local_matrix_zeroExterior.setCell1(cellNo1)
                            iM.getDoFsElem(cellNo1)
                            for cellNo2 in range(surface.num_cells):
                                self.local_matrix_zeroExterior.setCell2(cellNo2)
                                if self.mesh.dim == 1:
                                    if self.local_matrix_zeroExterior.center1[0] < self.local_matrix_zeroExterior.center2[0]:
                                        self.local_matrix_zeroExterior.center2[0] += evalShift
                                    else:
                                        self.local_matrix_zeroExterior.center2[0] -= evalShift
                                elif self.mesh.dim == 2:
                                    self.local_matrix_zeroExterior.center2[0] += evalShift*(self.local_matrix_zeroExterior.simplex2[1, 1]-self.local_matrix_zeroExterior.simplex2[0, 1])
                                    self.local_matrix_zeroExterior.center2[1] -= evalShift*(self.local_matrix_zeroExterior.simplex2[1, 0]-self.local_matrix_zeroExterior.simplex2[0, 0])
                                panel = self.local_matrix_zeroExterior.getPanelType()
                                self.local_matrix_zeroExterior.eval(contribZeroExterior, panel)
                                if self.local_matrix_zeroExterior.symmetricLocalMatrix:
                                    iM.addToMatrixElemSym(contribZeroExterior, -1.)
                                else:
                                    raise NotImplementedError()
                elif not self.zeroExterior and self.kernel.finiteHorizon:
                    with self.PLogger.Timer(prefix+'zeroExterior'):
                        # Subtract the contribution for Omega x (\partial B_\delta(x))
                        assert isinstance(self.kernel.horizon, constant)
                        self.local_matrix_zeroExterior.center2 = uninitialized((self.mesh.dim), dtype=REAL)
                        coeff = horizonSurfaceIntegral(self.local_matrix_zeroExterior, self.kernel.horizon.value)
                        qr = simplexXiaoGimbutas(2, self.mesh.dim)
                        if self.mesh.dim == 1:
                            mass = mass_1d_sym_scalar_anisotropic(coeff, self.dm, qr)
                        elif self.mesh.dim == 2:
                            mass = mass_2d_sym_scalar_anisotropic(coeff, self.dm, qr)
                        else:
                            raise NotImplementedError()
                        assembleMatrix(self.mesh, self.dm, mass, A=Anear)

        return Anear

    def reduceNearOp(self, LinearOperator Anear, indexSet myDofs):
        cdef:
            INDEX_t k = -1, kk, jj
            INDEX_t[::1] A_indptr = Anear.indptr, A_indices = Anear.indices
            REAL_t[::1] A_data = Anear.data, A_diagonal = None
            INDEX_t[::1] indptr, indices
            REAL_t[::1] data, diagonal = None
            LinearOperator Aother
            INDEX_t I, nnz
            indexSetIterator it = myDofs.getIter()
        counts = np.zeros((self.comm.size), dtype=INDEX)
        self.comm.Gather(np.array([Anear.nnz], dtype=INDEX), counts)
        if self.comm.rank == 0:
            LOGGER.info('Near field entries per rank: {} ({}) / {} / {} ({}) imbalance: {}'.format(counts.min(), counts.argmin(), counts.mean(), counts.max(), counts.argmax(), counts.max()/counts.min()))
        # drop entries that are not in rows of myRoot.dofs
        Anear = self.dropOffRank(Anear, myDofs)

        A_indptr = Anear.indptr

        # sum distribute matrices by stacking rows
        indptr = np.zeros((self.dm.num_dofs+1), dtype=INDEX)
        for k in range(self.dm.num_dofs):
            indptr[k+1] = A_indptr[k+1]-A_indptr[k]
        if self.comm.rank == 0:
            self.comm.Reduce(MPI.IN_PLACE, indptr, root=0)
        else:
            self.comm.Reduce(indptr, indptr, root=0)

        if self.comm.rank == 0:
            for k in range(self.dm.num_dofs):
                indptr[k+1] += indptr[k]
            nnz = indptr[self.dm.num_dofs]

            indices = uninitialized((nnz), dtype=INDEX)
            data = uninitialized((nnz), dtype=REAL)
            if isinstance(Anear, SSS_LinearOperator):
                diagonal = np.zeros((self.dm.num_dofs), dtype=REAL)

            for p in range(self.comm.size):
                if p == 0:
                    Aother = Anear
                else:
                    Aother = self.comm.recv(source=p)

                A_indptr = Aother.indptr
                A_indices = Aother.indices
                A_data = Aother.data

                for I in range(self.dm.num_dofs):
                    kk = indptr[I]
                    for jj in range(A_indptr[I], A_indptr[I+1]):
                        indices[kk] = A_indices[jj]
                        data[kk] = A_data[jj]
                        kk += 1

                if isinstance(Aother, SSS_LinearOperator):
                    A_diagonal = Aother.diagonal
                    for I in range(self.dm.num_dofs):
                        diagonal[I] += A_diagonal[I]

            if isinstance(Anear, SSS_LinearOperator):
                Anear = SSS_LinearOperator(indices, indptr, data, diagonal)
            else:
                Anear = CSR_LinearOperator(indices, indptr, data)
        else:
            self.comm.send(Anear, dest=0)
        self.comm.Barrier()

        if self.comm.rank != 0:
            Anear = None
        else:
            LOGGER.info('Anear reduced: {}'.format(Anear))
        # Anear = self.comm.bcast(Anear, root=0)
        return Anear

    def dropOffRank(self, LinearOperator Anear, indexSet myDofs):
        cdef:
            INDEX_t k = -1, kk, jj
            INDEX_t[::1] A_indptr = Anear.indptr, A_indices = Anear.indices
            REAL_t[::1] A_data = Anear.data, A_diagonal = None
            INDEX_t[::1] indptr, indices
            REAL_t[::1] data, diagonal = None
            indexSetIterator it = myDofs.getIter()
        # drop entries that are not in rows of myRoot.dofs
        indptr = np.zeros((self.dm.num_dofs+1), dtype=INDEX)
        while it.step():
            k = it.i
            indptr[k+1] = A_indptr[k+1]-A_indptr[k]
        for k in range(self.dm.num_dofs):
            indptr[k+1] += indptr[k]
        indices = uninitialized((indptr[self.dm.num_dofs]), dtype=INDEX)
        data = uninitialized((indptr[self.dm.num_dofs]), dtype=REAL)
        it.reset()
        while it.step():
            k = it.i
            kk = indptr[k]
            for jj in range(A_indptr[k], A_indptr[k+1]):
                indices[kk] = A_indices[jj]
                data[kk] = A_data[jj]
                kk += 1
        if isinstance(Anear, SSS_LinearOperator):
            A_diagonal = Anear.diagonal
            diagonal = np.zeros((self.dm.num_dofs), dtype=REAL)
            it.reset()
            while it.step():
                k = it.i
                diagonal[k] = A_diagonal[k]
            Anear = SSS_LinearOperator(indices, indptr, data, diagonal)
        else:
            Anear = CSR_LinearOperator(indices, indptr, data)
        return Anear

    def getDiagonal(self):
        cdef:
            diagonalOperator D
            INDEX_t I
            INDEX_t start, end
        D = diagonalOperator(np.zeros((self.dm.num_dofs), dtype=REAL))
        if self.comm:
            start = <INDEX_t>np.ceil(self.dm.num_dofs*self.comm.rank/self.comm.size)
            end = <INDEX_t>np.ceil(self.dm.num_dofs*(self.comm.rank+1)/self.comm.size)
        else:
            start = 0
            end = self.dm.num_dofs
        if self.kernel.variable:
            for I in range(start, end):
                D.setEntry(I, I, self.getEntryCluster(I, I))
        else:
            for I in range(start, end):
                D.setEntry(I, I, self.getEntry(I, I))
        if self.comm:
            self.comm.Allreduce(MPI.IN_PLACE, D.data)
        return D

    def getDiagonalCluster(self):
        cdef:
            diagonalOperator D
            tree_node n
            nearFieldClusterPair c
            INDEX_t I
            list clusters = []
            REAL_t[:, :, ::1] fake_boxes = uninitialized((0, 0, 0), dtype=REAL)
            list d2c = self.d2c
        D = diagonalOperator(np.zeros((self.dm.num_dofs), dtype=REAL))
        for I in range(self.dm.num_dofs):
            n = tree_node(None, set([I]), fake_boxes)
            n._cells = d2c[I]
            c = nearFieldClusterPair(n, n)
            clusters.append(c)
        D = self.assembleClusters(clusters, Anear=D)
        if self.comm:
            self.comm.Allreduce(MPI.IN_PLACE, D.data)
        return D

    def getKernelBlocksAndJumps(self):
        cdef:
            meshBase mesh = self.mesh
            DoFMap DoFMap = self.dm
            fractionalOrderBase s = self.kernel.s
            REAL_t[::1] orders = None
            REAL_t[::1] dofOrders
            REAL_t cellOrder
            dict blocks
            INDEX_t[::1] cellPair = uninitialized((2), dtype=INDEX)
            INDEX_t[::1] edge = uninitialized((2), dtype=INDEX)
            INDEX_t cellNo, dofNo, dof, cellNo1, cellNo2, vertexNo1, vertexNo2, vertex1, vertex2, i
            ENCODE_t hv
        orders = getFractionalOrdersDiagonal(s, mesh)
        dofOrders = -np.inf*np.ones((DoFMap.num_dofs), dtype=REAL)
        for cellNo in range(mesh.num_cells):
            cellOrder = orders[cellNo]
            for dofNo in range(DoFMap.dofs_per_element):
                dof = DoFMap.cell2dof(cellNo, dofNo)
                if dof >= 0:
                    if dofOrders[dof] == -np.inf:
                        dofOrders[dof] = cellOrder
                    elif dofOrders[dof] < np.inf:
                        if dofOrders[dof] != cellOrder:
                            dofOrders[dof] = np.inf
        blocks = {}
        for dof in range(DoFMap.num_dofs):
            try:
                blocks[dofOrders[dof]].add(dof)
            except KeyError:
                blocks[dofOrders[dof]] = set([dof])
        LOGGER.debug('Block sizes: '+str({key: len(blocks[key]) for key in blocks}))

        jumps = {}
        cellConnectivity = mesh.getCellConnectivity(mesh.dim)
        for cellNo1 in range(mesh.num_cells):
            for cellNo2 in cellConnectivity[cellNo1]:
                if orders[cellNo1] != orders[cellNo2]:
                    sortEdge(cellNo1, cellNo2, cellPair)
                    hv = encode_edge(cellPair)
                    if mesh.dim == 1:
                        for vertexNo1 in range(mesh.dim+1):
                            vertex1 = mesh.cells[cellNo1, vertexNo1]
                            for vertexNo2 in range(mesh.dim+1):
                                vertex2 = mesh.cells[cellNo2, vertexNo2]
                                if vertex1 == vertex2:
                                    jumps[hv] = vertex1
                                    break
                    else:
                        i = 0
                        for vertexNo1 in range(mesh.dim+1):
                            vertex1 = mesh.cells[cellNo1, vertexNo1]
                            for vertexNo2 in range(mesh.dim+1):
                                vertex2 = mesh.cells[cellNo2, vertexNo2]
                                if vertex1 == vertex2:
                                    edge[i] = vertex1
                                    i += 1
                                    break
                        hv2 = encode_edge(edge)
                        jumps[hv] = hv2
        return blocks, jumps

    def getTree(self, BOOL_t doDistributedAssembly, INDEX_t maxLevels, INDEX_t minClusterSize, INDEX_t minMixedClusterSize, REAL_t eta, INDEX_t minFarFieldBlockSize):
        cdef:
            REAL_t[:, :, ::1] boxes = None
            list cells = []
            REAL_t[:, ::1] coords = None
            INDEX_t i, j, dof, num_cluster_dofs
            REAL_t[:, ::1] box
            dict blocks = {}, jumps = {}
            indexSet dofs, clusterDofs, subDofs, blockDofs, myDofs = None
            indexSetIterator it
            REAL_t key
            INDEX_t[::1] part = None
            set myCells
            tree_node root, myRoot, n
            refinementType refType
            BOOL_t splitEveryDim

        rT = self.params.get('refinementType', 'MEDIAN')
        refType = {'geometric': GEOMETRIC,
                   'GEOMETRIC': GEOMETRIC,
                   'median': MEDIAN,
                   'MEDIAN': MEDIAN,
                   'barycenter': BARYCENTER,
                   'BARYCENTER': BARYCENTER}[rT]
        splitEveryDim = self.params.get('splitEveryDim', False)

        with self.PLogger.Timer('prepare tree'):
            boxes, cells = getDoFBoxesAndCells(self.dm.mesh, self.dm, self.comm)
            coords = self.dm.getDoFCoordinates()

            dofs = arrayIndexSet(np.arange(self.dm.num_dofs, dtype=INDEX), sorted=True)
            root = tree_node(None, dofs, boxes)

            if doDistributedAssembly:
                from PyNucleus_fem.meshPartitioning import regularDofPartitioner, metisDofPartitioner, PartitionerException

                partitioner = self.params.get('partitioner', 'regular')

                if partitioner == 'regular':
                    rVP = regularDofPartitioner(dm=self.dm)
                    try:
                        part, _ = rVP.partitionDofs(self.comm.size, irregular=True)
                    except PartitionerException:
                        doDistributedAssembly = False
                        LOGGER.warning('Falling back to serial assembly')
                    del rVP
                elif partitioner == 'metis':
                    dP = metisDofPartitioner(dm=self.dm, matrixPower=self.params.get('metis_matrixPower', 1))
                    try:
                        part, _ = dP.partitionDofs(self.comm.size, ufactor=self.params.get('metis_ufactor', 30))
                    except PartitionerException:
                        doDistributedAssembly = False
                        LOGGER.warning('Falling back to serial assembly')
                    del dP
                elif partitioner == 'metis-weighted':

                    self.params['partitioner'] = 'metis'
                    (root, myRoot, _, tempMyDofs, _,
                     doDistributedAssemblyTemp, _) = self.getTree(doDistributedAssembly, maxLevels, minClusterSize, minMixedClusterSize, eta, minFarFieldBlockSize)
                    # get the admissible cluster pairs
                    Pnear, _ = self.getAdmissibleClusters(root, myRoot, doDistributedAssemblyTemp, eta, minFarFieldBlockSize)
                    tempAnear = getSparseNearField(self.dm, Pnear, False)
                    tempAnear = self.reduceNearOp(tempAnear, tempMyDofs)

                    if self.comm.rank == 0:
                        weights = np.array(tempAnear.indptr)
                        weights = weights[1:]-weights[:-1]
                    else:
                        weights = None
                    self.comm.bcast(weights)

                    dP = metisDofPartitioner(dm=self.dm, matrixPower=self.params.get('metis_matrixPower', 1))
                    try:
                        part, _ = dP.partitionDofs(self.comm.size, ufactor=self.params.get('metis_ufactor', 30), dofWeights=weights)
                    except PartitionerException:
                        doDistributedAssembly = False
                        LOGGER.warning('Falling back to serial assembly')
                    del dP
                    self.params['partitioner'] = 'metis-weighted'
                else:
                    raise NotImplementedError(partitioner)
            if doDistributedAssembly:
                num_dofs = 0
                for p in range(self.comm.size):
                    subDofs = arrayIndexSet(np.where(np.array(part, copy=False) == p)[0].astype(INDEX), sorted=True)
                    num_dofs += subDofs.getNumEntries()
                    root.children.append(tree_node(root, subDofs, boxes, canBeAssembled=not self.kernel.variable))
                assert self.dm.num_dofs == num_dofs
                root._dofs = None

                myRoot = root.children[self.comm.rank]
                myDofs = myRoot.get_dofs()
            else:
                myRoot = root

            if self.kernel.variable:
                blocks, jumps = self.getKernelBlocksAndJumps()
                for n in root.leaves():
                    clusterDofs = n.get_dofs()
                    num_cluster_dofs = clusterDofs.getNumEntries()
                    num_dofs = 0
                    for key in sorted(blocks):
                        blockDofs = arrayIndexSet()
                        blockDofs.fromSet(blocks[key])
                        subDofs = blockDofs.inter(clusterDofs)
                        if subDofs.getNumEntries() > 0:
                            num_dofs += subDofs.getNumEntries()
                            n.children.append(tree_node(n, subDofs, boxes, mixed_node=key == np.inf))
                    assert num_dofs == num_cluster_dofs, (num_dofs, num_cluster_dofs)
                    n._dofs = None
                LOGGER.info('Jumps: {}, Block sizes: {}, Leaf nodes: {}'.format(len(jumps), str({key: len(blocks[key]) for key in blocks}), len(list(root.leaves()))))

            for n in root.leaves():
                # if not n.mixed_node:
                n.refine(boxes, coords, maxLevels=maxLevels, minSize=minClusterSize, minMixedSize=minMixedClusterSize, refType=refType, splitEveryDim=splitEveryDim)

            # recursively set tree node ids
            root.set_id()

            # enter cells in leaf nodes
            it = arrayIndexSetIterator()
            for n in root.leaves():
                myCells = set()
                it.setIndexSet(n.dofs)
                while it.step():
                    dof = it.i
                    myCells |= cells[dof]
                n._cells = arrayIndexSet()
                n._cells.fromSet(myCells)
            del cells

            # update boxes (if we stopped at maxLevels, before each DoF has
            # only it's support as box)
            for n in root.leaves():
                box = n.box
                it.setIndexSet(n.dofs)
                while it.step():
                    dof = it.i
                    for i in range(self.dm.mesh.dim):
                        for j in range(2):
                            boxes[dof, i, j] = box[i, j]

            if maxLevels <= 0:
                maxLevels = root.numLevels+maxLevels

            return root, myRoot, boxes, myDofs, jumps, doDistributedAssembly, maxLevels

    def getAdmissibleClusters(self,
                              tree_node root, tree_node myRoot,
                              BOOL_t doDistributedAssembly,
                              REAL_t eta,
                              INDEX_t minFarFieldBlockSize,
                              BOOL_t assembleOnRoot=True):
        cdef:
            dict Pfar = {}
            list Pnear = []
            INDEX_t lvl, id1, id2
            nearFieldClusterPair cPnear
            farFieldClusterPair cP
            tree_node n1, n2
            dict added
            INDEX_t N
        with self.PLogger.Timer('admissible clusters'):
            if doDistributedAssembly:
                for n in root.children:
                    getAdmissibleClusters(self.local_matrix.kernel, myRoot, n,
                                          eta=eta, farFieldInteractionSize=minFarFieldBlockSize,
                                          Pfar=Pfar, Pnear=Pnear)
                counts = np.zeros((self.comm.size), dtype=INDEX)
                symmetrizeNearFieldClusters(Pnear)
                self.comm.Gather(np.array([len(Pnear)], dtype=INDEX), counts)
                LOGGER.info('Near field cluster pairs per rank: {} ({}) / {} / {} ({})'.format(counts.min(), counts.argmin(), counts.mean(), counts.max(), counts.argmax()))

                nnz = 0.
                for cPnear in Pnear:
                    n1 = cPnear.n1
                    n2 = cPnear.n2
                    nnz += n1.get_num_dofs()*n2.get_num_dofs()
                self.comm.Gather(np.array([nnz], dtype=INDEX), counts)
                LOGGER.info('Near field entries per rank: {} ({}) / {} / {} ({})'.format(counts.min(), counts.argmin(), counts.mean(), counts.max(), counts.argmax()))

                if assembleOnRoot:
                    # collect far field on rank 0
                    farField = []
                    for lvl in Pfar:
                        for cP in Pfar[lvl]:
                            # "lvl+1", since the ranks are children of the global root
                            farField.append((lvl+1, cP.n1.id, cP.n2.id))
                    farField = np.array(farField, dtype=INDEX)
                    self.comm.Gather(np.array([farField.shape[0]], dtype=INDEX), counts)
                    if self.comm.rank == 0:
                        LOGGER.info('Far field cluster pairs per rank: {} ({}) / {} / {} ({})'.format(counts.min(), counts.argmin(), counts.mean(), counts.max(), counts.argmax()))
                        N = 0
                        for rank in range(self.comm.size):
                            N += counts[rank]
                        farFieldCollected = uninitialized((N, 3), dtype=INDEX)
                        counts *= 3
                    else:
                        farFieldCollected = None
                    self.comm.Gatherv(farField, [farFieldCollected, (counts, None)], root=0)
                    del farField

                    if self.comm.rank == 0:
                        Pfar = {}
                        added = {}
                        for k in range(farFieldCollected.shape[0]):
                            lvl, id1, id2 = farFieldCollected[k, :]
                            cP = farFieldClusterPair(root.get_node(id1),
                                                     root.get_node(id2))
                            try:
                                if (id1, id2) not in added[lvl]:
                                    Pfar[lvl].append(cP)
                                    added[lvl].add((id1, id2))
                            except KeyError:
                                Pfar[lvl] = [cP]
                                added[lvl] = set([(id1, id2)])
                        del farFieldCollected
                    else:
                        Pfar = {}
            else:
                getAdmissibleClusters(self.local_matrix.kernel, root, root,
                                      eta=eta, farFieldInteractionSize=minFarFieldBlockSize,
                                      Pfar=Pfar, Pnear=Pnear)
            if self.params.get('trim', True):
                trimTree(root, Pnear, Pfar, self.comm)
        return Pnear, Pfar

    def getH2(self, BOOL_t returnNearField=False, returnTree=False):
        cdef:
            meshBase mesh = self.mesh
            DoFMap DoFMap = self.dm
            tree_node root, myRoot
            fractionalOrderBase s = self.kernel.s
            REAL_t[:, :, ::1] boxes
            dict Pfar
            list Pnear
            LinearOperator h2 = None, Anear = None
            dict jumps
            indexSet myDofs
            INDEX_t interpolation_order, maxLevels, minClusterSize, minMixedClusterSize, minFarFieldBlockSize
            BOOL_t forceUnsymmetric, doDistributedAssembly = False, assembleOnRoot = True
        assert isinstance(self.kernel, FractionalKernel), 'H2 is only implemented for fractional kernels'

        target_order = self.local_matrix.target_order
        eta = self.params.get('eta', 3.)

        iO = self.params.get('interpolation_order', None)
        if iO is None:
            loggamma = abs(np.log(0.25))
            interpolation_order = max(np.ceil((2*target_order+max(mesh.dim+2*s.max, 2))*abs(np.log(mesh.hmin/mesh.diam))/loggamma/3.), 2)
        else:
            interpolation_order = iO
        mL = self.params.get('maxLevels', None)
        if mL is None:
            # maxLevels = max(int(np.around(np.log2(DoFMap.num_dofs)/mesh.dim-np.log2(interpolation_order))), 0)
            maxLevels = 200
        else:
            maxLevels = mL
        mCS = self.params.get('minClusterSize', None)
        if mCS is None:
            minClusterSize = interpolation_order**mesh.dim//2
        else:
            minClusterSize = mCS
        minMixedClusterSize = minClusterSize
        if self.kernel.finiteHorizon:
            minMixedClusterSize = max(min(self.kernel.horizon.value//(2*mesh.h)-1, minClusterSize), 1)
        mFFBS = self.params.get('minFarFieldBlockSize', None)
        if mFFBS is None:
            # For this value, size(kernelInterpolant) == size(dense block)
            # If we choose a smaller value for minFarFieldBlockSize, then we use more memory,
            # but we might save time, since the assembly of a far field block is cheaper than a near field block.
            minFarFieldBlockSize = interpolation_order**(2*mesh.dim)
        else:
            minFarFieldBlockSize = mFFBS
        forceUnsymmetric = self.params.get('forceUnsymmetric', False)
        doDistributedAssembly = self.comm is not None and self.comm.size > 1 and DoFMap.num_dofs > self.comm.size
        assembleOnRoot = self.params.get('assembleOnRoot', True)
        LOGGER.info('interpolation_order: {}, maxLevels: {}, minClusterSize: {}, minMixedClusterSize: {}, minFarFieldBlockSize: {}, distributedAssembly: {}, eta: {}'.format(interpolation_order, maxLevels, minClusterSize, minMixedClusterSize, minFarFieldBlockSize, doDistributedAssembly, eta))

        # construct the cluster tree
        (root, myRoot, boxes, myDofs, jumps,
         doDistributedAssembly, maxLevels) = self.getTree(doDistributedAssembly, maxLevels, minClusterSize, minMixedClusterSize, eta, minFarFieldBlockSize)
        # get the admissible cluster pairs
        Pnear, Pfar = self.getAdmissibleClusters(root, myRoot, doDistributedAssembly, eta, minFarFieldBlockSize, assembleOnRoot=assembleOnRoot)
        lenPfar = len(Pfar)
        if doDistributedAssembly:
            lenPfar = self.comm.bcast(lenPfar)

        # tempAnear = getSparseNearField(self.dm, Pnear, False)
        # tempAnear = self.reduceNearOp(tempAnear, myDofs)

        # if self.comm.rank == 0:
        #     weights = np.array(tempAnear.indptr)
        #     weights = weights[1:]-weights[:-1]
        # else:
        #     weights = None
        # self.comm.bcast(weights)

        # if self.comm.rank == 0:
        #     import matplotlib.pyplot as plt
        #     w = self.dm.zeros()
        #     w2 = np.array(weights).astype(REAL)
        #     w.assign(w2)
        #     w.plot(flat=True)
        #     print('w2.max', w2.argmax(), w2.max())
        #     plt.show()

        if lenPfar > 0:
            # get near field matrix
            with self.PLogger.Timer('near field'):
                Anear = self.assembleClusters(Pnear, jumps=jumps, myDofs=myDofs, forceUnsymmetric=forceUnsymmetric)
            if doDistributedAssembly:
                if assembleOnRoot:
                    with self.PLogger.Timer('reduceNearOp'):
                        Anear = self.reduceNearOp(Anear, myDofs)
                else:
                    Anear = self.dropOffRank(Anear, myDofs)

            with self.PLogger.Timer('leaf values'):
                # get leave values
                if s.max < 1.:
                    root.enterLeafValues(mesh, DoFMap, interpolation_order, boxes, self.comm, assembleOnRoot=assembleOnRoot)
                elif s.min > 1:
                    root.enterLeafValuesGrad(mesh, DoFMap, interpolation_order, boxes, self.comm)
                else:
                    raise NotImplementedError()

            if self.comm is None or (assembleOnRoot and self.comm.rank == 0) or (not assembleOnRoot):
                with self.PLogger.Timer('far field'):
                    # get kernel interpolations
                    assembleFarFieldInteractions(self.local_matrix.kernel, Pfar, interpolation_order, DoFMap)

                with self.PLogger.Timer('transfer matrices'):
                    # get transfer matrices
                    root.prepareTransferOperators(interpolation_order)

                if self.comm is None or (assembleOnRoot and self.comm.rank == 0):
                    h2 = H2Matrix(root, Pfar, Anear)
                else:
                    local_h2 = H2Matrix(root, Pfar, Anear)
                    h2 = DistributedH2Matrix(local_h2, self.comm)
            else:
                h2 = nullOperator(self.dm.num_dofs, self.dm.num_dofs)

        elif len(Pnear) == 0:
            h2 = nullOperator(self.dm.num_dofs, self.dm.num_dofs)
        else:
            LOGGER.info('Near field pairs: {}, far field pairs: {}, tree nodes: {}'.format(len(Pnear), lenPfar, root.nodes))
            with self.PLogger.Timer('dense operator'):
                h2 = self.getDense()
        LOGGER.info('{}'.format(h2))
        if returnNearField:
            if returnTree:
                return h2, Pnear, root
            else:
                return h2, Pnear
        else:
            if returnTree:
                return h2, root
            else:
                return h2

    def getH2FiniteHorizon(self, LinearOperator Ainf=None):
        A = horizonCorrected(self.mesh, self.dm, self.kernel, self.comm, Ainf, logging=isinstance(self.PLogger, (PLogger, LoggingPLogger)))
        return A


cdef class horizonSurfaceIntegral(function):
    cdef:
        nonlocalLaplacian local_matrix
        REAL_t horizon
        REAL_t[:, ::1] quadNodes
        REAL_t[::1] quadWeights
        REAL_t inc

    def __init__(self, nonlocalLaplacian local_matrix, REAL_t horizon):
        cdef:
            INDEX_t k, numQuadNodes
        self.local_matrix = local_matrix
        self.horizon = horizon
        if self.local_matrix.dim == 1:
            self.quadNodes = uninitialized((2, 1), dtype=REAL)
            self.quadWeights = uninitialized((2), dtype=REAL)
            self.quadNodes[0, 0] = self.horizon
            self.quadNodes[1, 0] = -self.horizon
            self.quadWeights[0] = 1.
            self.quadWeights[1] = 1.
        elif self.local_matrix.dim == 2:
            numQuadNodes = 10
            self.quadNodes = uninitialized((numQuadNodes, 2), dtype=REAL)
            self.quadWeights = uninitialized((numQuadNodes), dtype=REAL)
            inc = 2*pi/numQuadNodes
            for k in range(numQuadNodes):
                self.quadNodes[k, 0] = self.horizon*cos(inc*k)
                self.quadNodes[k, 1] = self.horizon*sin(inc*k)
                self.quadWeights[k] = inc*self.horizon
        else:
            raise NotImplementedError()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t fac = 0.
            INDEX_t k, j
            REAL_t s
            INDEX_t dim = self.local_matrix.dim
        for j in range(self.local_matrix.dim):
            self.local_matrix.center1[j] = x[j]

        for k in range(self.quadNodes.shape[0]):
            for j in range(self.local_matrix.dim):
                self.local_matrix.center2[j] = x[j]+self.quadNodes[k, j]
            self.local_matrix.kernel.evalParams(self.local_matrix.center1,
                                                self.local_matrix.center2)
            s = self.local_matrix.kernel.sValue
            fac -= self.local_matrix.kernel.scalingValue*pow(self.horizon, 1-dim-2*s)/s * self.quadWeights[k]
        return fac



cdef class horizonCorrected(TimeStepperLinearOperator):
    cdef:
        meshBase mesh
        DoFMap dm
        MPI.Comm comm
        public LinearOperator Ainf
        public LinearOperator mass
        public Kernel kernel
        BOOL_t logging
        BOOL_t initialized

    def __init__(self, meshBase mesh, DoFMap dm, FractionalKernel kernel, MPI.Comm comm=None, LinearOperator Ainf=None, BOOL_t logging=False):
        self.mesh = mesh
        self.dm = dm
        self.kernel = kernel
        self.comm = comm
        self.logging = logging
        assert isinstance(kernel.horizon, constant)

        if Ainf is None:
            scaling = constantTwoPoint(0.5)
            infiniteKernel = kernel.getModifiedKernel(horizon=constant(np.inf), scaling=scaling)
            infBuilder = nonlocalBuilder(self.mesh, self.dm, infiniteKernel, zeroExterior=True, comm=self.comm, logging=self.logging)
            self.Ainf = infBuilder.getH2()
        else:
            self.Ainf = Ainf
        self.mass = self.dm.assembleMass(sss_format=True)
        TimeStepperLinearOperator.__init__(self, self.Ainf, self.mass, 1.0)
        self.initialized = False

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def setKernel(self, Kernel kernel):
        cdef:
            REAL_t s, horizon, C, vol
            dict jumps
            ENCODE_t hv, hv2
            INDEX_t[::1] cellPair, edge
            REAL_t evalShift, fac
            INDEX_t vertexNo, cellNo3
            panelType panel
            INDEX_t[:, ::1] fake_cells

        assert isinstance(kernel.horizon, constant)
        assert not isinstance(kernel.scaling, variableFractionalLaplacianScaling)
        horizon = kernel.horizon.value
        assert horizon < np.inf
        C = kernel.scaling.value

        if isinstance(kernel.s, constFractionalOrder):
            assert kernel.s.value == self.kernel.s.value

            if (self.initialized and
                (self.kernel.s.value == kernel.s.value) and
                (self.kernel.horizonValue == horizon) and
                (self.kernel.scaling.value == C)):
                return

        self.kernel = kernel

        complementKernel = kernel.getComplementKernel()
        builder = nonlocalBuilder(self.mesh, self.dm, complementKernel, zeroExterior=True, comm=self.comm, logging=self.logging)
        correction = builder.getH2()

        self.S = self.Ainf
        self.facS = 2.*C

        if isinstance(self.kernel.s, constFractionalOrder):
            if self.mesh.dim == 1:
                vol = 2
            elif self.mesh.dim == 2:
                vol = 2*np.pi * horizon
            else:
                raise NotImplementedError()
            s = self.kernel.sValue
            self.M = -correction + (-vol*C*pow(horizon, 1-self.mesh.dim-2*s)/s) * self.mass
        else:
            self.mass.setZero()

            builder.local_matrix_zeroExterior.center2 = uninitialized((self.mesh.dim), dtype=REAL)
            coeff = horizonSurfaceIntegral(builder.local_matrix_zeroExterior, horizon)
            qr = simplexXiaoGimbutas(2, self.mesh.dim)
            if self.mesh.dim == 1:
                if isinstance(self.dm, P1_DoFMap):
                    mass = mass_1d_sym_scalar_anisotropic(coeff, self.dm, qr)
                else:
                    raise NotImplementedError()
            elif self.mesh.dim == 2:
                if isinstance(self.dm, P1_DoFMap):
                    mass = mass_2d_sym_scalar_anisotropic(coeff, self.dm, qr)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

            assembleMatrix(self.mesh, self.dm, mass, A=self.mass)

            _, jumps = builder.getKernelBlocksAndJumps()
            iM = IndexManager(self.dm, self.mass)
            fake_cells = uninitialized((1, self.mesh.dim), dtype=INDEX)
            cellPair = uninitialized((2), dtype=INDEX)
            edge = uninitialized((2), dtype=INDEX)
            evalShift = 1e-9
            contribZeroExterior = uninitialized((self.dm.dofs_per_element*(self.dm.dofs_per_element+1)//2), dtype=REAL)
            builder.local_matrix_surface.setMesh1(self.mesh)
            for hv in jumps:
                decode_edge(hv, cellPair)

                if self.mesh.dim == 1:
                    fake_cells[0, 0] = jumps[hv]
                else:
                    hv2 = jumps[hv]
                    decode_edge(hv2, edge)
                    for vertexNo in range(self.mesh.dim):
                        fake_cells[0, vertexNo] = edge[vertexNo]
                builder.local_matrix_surface.setVerticesCells2(self.mesh.vertices, fake_cells)
                builder.local_matrix_surface.setCell2(0)
                if self.mesh.dim == 1:
                    builder.local_matrix_surface.center2[0] += evalShift
                elif self.mesh.dim == 2:
                    builder.local_matrix_surface.center2[0] += evalShift*(builder.local_matrix_surface.simplex2[1, 1]-builder.local_matrix_surface.simplex2[0, 1])
                    builder.local_matrix_surface.center2[1] += evalShift*(builder.local_matrix_surface.simplex2[0, 0]-builder.local_matrix_surface.simplex2[1, 0])

                for cellNo3 in range(self.mesh.num_cells):
                    builder.local_matrix_surface.setCell1(cellNo3)
                    panel = builder.local_matrix_surface.getPanelType()
                    if panel != IGNORED:
                        if self.mesh.dim == 1:
                            if builder.local_matrix_surface.center1[0] < builder.local_matrix_surface.center2[0]:
                                fac = 1.
                            else:
                                fac = -1.
                        else:
                            fac = 1.
                        builder.local_matrix_surface.eval(contribZeroExterior, panel)
                        iM.getDoFsElem(cellNo3)
                        if builder.local_matrix_surface.symmetricLocalMatrix:
                            iM.addToMatrixElemSym(contribZeroExterior, -2*C*fac)
                        else:
                            raise NotImplementedError()

                if self.mesh.dim == 1:
                    builder.local_matrix_surface.center2[0] -= 2.*evalShift
                elif self.mesh.dim == 2:
                    builder.local_matrix_surface.center2[0] -= 2.*evalShift*(builder.local_matrix_surface.simplex2[1, 1]-builder.local_matrix_surface.simplex2[0, 1])
                    builder.local_matrix_surface.center2[1] -= 2.*evalShift*(builder.local_matrix_surface.simplex2[0, 0]-builder.local_matrix_surface.simplex2[1, 0])

                for cellNo3 in range(self.mesh.num_cells):
                    builder.local_matrix_surface.setCell1(cellNo3)
                    panel = builder.local_matrix_surface.getPanelType()
                    if panel != IGNORED:
                        if self.mesh.dim == 1:
                            if builder.local_matrix_surface.center1[0] < builder.local_matrix_surface.center2[0]:
                                fac = -1.
                            else:
                                fac = 1.
                        else:
                            fac = -1.
                        builder.local_matrix_surface.eval(contribZeroExterior, panel)
                        iM.getDoFsElem(cellNo3)
                        if builder.local_matrix_surface.symmetricLocalMatrix:
                            iM.addToMatrixElemSym(contribZeroExterior, -2*C*fac)
                        else:
                            raise NotImplementedError()

            self.M = -correction + self.mass
        self.facM = 1.
        self.initialized = True

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        assert self.initialized
        return TimeStepperLinearOperator.matvec(self, x, y)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec_no_overwrite(self,
                                     REAL_t[::1] x,
                                     REAL_t[::1] y) except -1:
        assert self.initialized
        return TimeStepperLinearOperator.matvec_no_overwrite(self, x, y)


def assembleFractionalLaplacian(meshBase mesh,
                                DoFMap DoFMap,
                                fractionalOrderBase s,
                                function horizon=constant(np.inf),
                                target_order=None,
                                bint zeroExterior=True,
                                bint genKernel=False,
                                MPI.Comm comm=None,
                                bint forceNonSym=False,
                                **kwargs):
    warnings.warn('"assembleFractionalLaplacian" deprecated, use "assembleNonlocalOperator"', DeprecationWarning)
    params = {'target_order': target_order,
              'genKernel': genKernel,
              'forceNonSym': forceNonSym}
    return assembleNonlocalOperator(mesh, DoFMap, s, horizon, params, zeroExterior, comm, **kwargs)


def assembleNonlocalOperator(meshBase mesh,
                             DoFMap DoFMap,
                             fractionalOrderBase s,
                             function horizon=constant(np.inf),
                             dict params={},
                             bint zeroExterior=True,
                             MPI.Comm comm=None,
                             **kwargs):
    kernel = getFractionalKernel(mesh.dim, s, horizon)
    builder = nonlocalBuilder(mesh, DoFMap, kernel, params, zeroExterior, comm, **kwargs)
    return builder.getDense()



def assembleFractionalLaplacianDiagonal(meshBase mesh,
                                        DoFMap DoFMap,
                                        fractionalOrderBase s,
                                        function horizon=constant(np.inf),
                                        dict params={},
                                        bint zeroExterior=True,
                                        comm=None,
                                        **kwargs):
    kernel = getFractionalKernel(mesh.dim, s, horizon)
    builder = nonlocalBuilder(mesh, DoFMap, kernel, params, zeroExterior, comm, **kwargs)
    return builder.getDiagonal()


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef LinearOperator getSparseNearField(DoFMap DoFMap, list Pnear, bint symmetric=False):
    cdef:
        sparsityPattern sP
        INDEX_t I = -1, J = -1
        nearFieldClusterPair clusterPair
        indexSet dofs1, dofs2
        indexSetIterator it1 = arrayIndexSetIterator(), it2 = arrayIndexSetIterator()
    sP = sparsityPattern(DoFMap.num_dofs)
    if symmetric:
        for clusterPair in Pnear:
            dofs1 = clusterPair.n1.get_dofs()
            dofs2 = clusterPair.n2.get_dofs()
            it1.setIndexSet(dofs1)
            it2.setIndexSet(dofs2)
            while it1.step():
                I = it1.i
                it2.reset()
                while it2.step():
                    J = it2.i
                    if I > J:
                        sP.add(I, J)
    else:
        for clusterPair in Pnear:
            dofs1 = clusterPair.n1.get_dofs()
            dofs2 = clusterPair.n2.get_dofs()
            it1.setIndexSet(dofs1)
            it2.setIndexSet(dofs2)
            while it1.step():
                I = it1.i
                it2.reset()
                while it2.step():
                    J = it2.i
                    sP.add(I, J)
    indptr, indices = sP.freeze()
    data = np.zeros((indices.shape[0]), dtype=REAL)
    if symmetric:
        diagonal = np.zeros((DoFMap.num_dofs), dtype=REAL)
        A = SSS_LinearOperator(indices, indptr, data, diagonal)
    else:
        A = CSR_LinearOperator(indices, indptr, data)
    return A


cdef class nearFieldClusterPair:
    def __init__(self, tree_node n1, tree_node n2):
        cdef:
            indexSet cells1, cells2
        self.n1 = n1
        self.n2 = n2
        cells1 = self.n1.get_cells()
        cells2 = self.n2.get_cells()
        self.cellsUnion = cells1.union(cells2)
        self.cellsInter = cells1.inter(cells2)
        assert len(cells1)+len(cells2) == len(self.cellsUnion)+len(self.cellsInter), (cells1.toSet(),
                                                                                      cells2.toSet(),
                                                                                      self.cellsInter.toSet(),
                                                                                      self.cellsUnion.toSet())

    def plot(self, color='red'):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        dim = self.n1.box.shape[0]
        if dim == 1:
            box1 = self.n1.box
            box2 = self.n2.box
            plt.gca().add_patch(patches.Rectangle((box1[0, 0], box2[0, 0]), box1[0, 1]-box1[0, 0], box2[0, 1]-box2[0, 0], fill=True, alpha=0.5, facecolor=color))
        else:
            for dof1 in self.n1.dofs:
                for dof2 in self.n2.dofs:
                    plt.gca().add_patch(patches.Rectangle((dof1-0.5, dof2-0.5), 1., 1., fill=True, alpha=0.5, facecolor=color))

    def HDF5write(self, node):
        node.attrs['n1'] = self.n1.id
        node.attrs['n2'] = self.n2.id

    @staticmethod
    def HDF5read(node, nodes):
        cP = nearFieldClusterPair(nodes[int(node.attrs['n1'])],
                                  nodes[int(node.attrs['n2'])])
        return cP

    def __repr__(self):
        return 'nearFieldClusterPair<{}, {}>'.format(self.n1, self.n2)


cdef inline int balanceCluster(INDEX_t[::1] csums, INDEX_t cost):
    cdef:
        INDEX_t k, i
        INDEX_t csize = csums.shape[0]
    # k = argmin(csums)
    k = 0
    for i in range(1, csize):
        if csums[i] < csums[k]:
            k = i
    # add cost estimate
    csums[k] += cost
    # prevent overflow
    if csums[k] > 1e7:
        j = csums[k]
        for i in range(csize):
            csums[i] -= j
    return k


cdef class SubMatrixAssemblyOperator(LinearOperator):
    cdef:
        LinearOperator A
        dict lookupI, lookupJ

    def __init__(self, LinearOperator A, INDEX_t[::1] I, INDEX_t[::1] J):
        LinearOperator.__init__(self,
                                A.num_rows,
                                A.num_columns)
        self.A = A
        self.lookupI = {}
        self.lookupJ = {}
        for i in range(I.shape[0]):
            self.lookupI[I[i]] = i
        for i in range(J.shape[0]):
            self.lookupJ[J[i]] = i

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void addToEntry(self, INDEX_t I, INDEX_t J, REAL_t val):
        cdef:
            INDEX_t i, j
        i = self.lookupI.get(I, -1)
        j = self.lookupJ.get(J, -1)
        if i >= 0 and j >= 0:
            self.A.addToEntry(i, j, val)


cdef class FilteredAssemblyOperator(LinearOperator):
    cdef:
        LinearOperator A
        indexSet dofs1, dofs2

    def __init__(self, LinearOperator A):
        self.A = A

    cdef void setFilter(self, indexSet dofs1, indexSet dofs2):
        self.dofs1 = dofs1
        self.dofs2 = dofs2

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void addToEntry(self, INDEX_t I, INDEX_t J, REAL_t val):
        if self.dofs1.inSet(I) and self.dofs2.inSet(J):
            self.A.addToEntry(I, J, val)


cdef class LeftFilteredAssemblyOperator(LinearOperator):
    cdef:
        LinearOperator A
        indexSet dofs1

    def __init__(self, LinearOperator A):
        self.A = A

    cdef void setFilter(self, indexSet dofs1):
        self.dofs1 = dofs1

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline void addToEntry(self, INDEX_t I, INDEX_t J, REAL_t val):
        if self.dofs1.inSet(I):
            self.A.addToEntry(I, J, val)


def assembleNearField(list Pnear,
                      meshBase mesh,
                      DoFMap DoFMap,
                      fractionalOrderBase s,
                      function horizon=constant(np.inf),
                      dict params={},
                      bint zeroExterior=True,
                      comm=None,
                      **kwargs):
    kernel = getFractionalKernel(mesh.dim, s, horizon)
    builder = nonlocalBuilder(mesh, DoFMap, kernel, params, zeroExterior, comm, logging=True, **kwargs)
    A = builder.assembleClusters(Pnear)
    return A


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef INDEX_t[:, ::1] boundaryVertices(INDEX_t[:, ::1] cells, indexSet cellIds):
    cdef:
        INDEX_t c0, c1, i, k
        np.ndarray[INDEX_t, ndim=2] bvertices_mem
        INDEX_t[:, ::1] bvertices_mv
        set bvertices = set()
        indexSetIterator it = cellIds.getIter()

    while it.step():
        i = it.i
        c0, c1 = cells[i, 0], cells[i, 1]
        try:
            bvertices.remove(c0)
        except KeyError:
            bvertices.add(c0)
        try:
            bvertices.remove(c1)
        except KeyError:
            bvertices.add(c1)
    bvertices_mem = uninitialized((len(bvertices), 1), dtype=INDEX)
    bvertices_mv = bvertices_mem
    i = 0
    for k in bvertices:
        bvertices_mv[i, 0] = k
        i += 1
    return bvertices_mem


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef INDEX_t[:, ::1] boundaryEdges(INDEX_t[:, ::1] cells, indexSet cellIds):
    cdef:
        INDEX_t c0, c1, c2, i, k
        ENCODE_t hv
        INDEX_t[:, ::1] temp = uninitialized((3, 2), dtype=INDEX)
        INDEX_t[::1] e0 = temp[0, :]
        INDEX_t[::1] e1 = temp[1, :]
        INDEX_t[::1] e2 = temp[2, :]
        np.ndarray[INDEX_t, ndim=2] bedges_mem
        INDEX_t[:, ::1] bedges_mv
        dict bedges = dict()
        bint orientation
        indexSetIterator it = cellIds.getIter()

    while it.step():
        i = it.i
        c0, c1, c2 = cells[i, 0], cells[i, 1], cells[i, 2]
        sortEdge(c0, c1, e0)
        sortEdge(c1, c2, e1)
        sortEdge(c2, c0, e2)
        for k in range(3):
            hv = encode_edge(temp[k, :])
            try:
                del bedges[hv]
            except KeyError:
                bedges[hv] = (cells[i, k] == temp[k, 0])
    bedges_mem = uninitialized((len(bedges), 2), dtype=INDEX)
    bedges_mv = bedges_mem

    i = 0
    for hv in bedges:
        orientation = bedges[hv]
        decode_edge(hv, e0)
        if orientation:
            bedges_mv[i, 0], bedges_mv[i, 1] = e0[0], e0[1]
        else:
            bedges_mv[i, 0], bedges_mv[i, 1] = e0[1], e0[0]
        i += 1
    return bedges_mem
