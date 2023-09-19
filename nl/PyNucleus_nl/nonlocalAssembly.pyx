###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from libc.math cimport ceil
import numpy as np
cimport numpy as np

include "config.pxi"

from libc.math cimport sin, cos, M_PI as pi
from libcpp.map cimport map
from cpython.long cimport PyLong_FromSsize_t
from PyNucleus_base.myTypes import INDEX, REAL, COMPLEX, ENCODE, BOOL
from PyNucleus_base import uninitialized
from PyNucleus_base.intTuple cimport intTuple
from PyNucleus_base.ip_norm cimport (ip_distributed_nonoverlapping,
                                     norm_distributed_nonoverlapping)
from PyNucleus_fem.mesh import mesh0d, mesh1d
from PyNucleus_fem.functions cimport function, constant
from PyNucleus_fem.DoFMaps cimport P0_DoFMap, P1_DoFMap, P2_DoFMap, P3_DoFMap, Product_DoFMap
from PyNucleus_fem.meshCy cimport sortEdge, encode_edge, decode_edge, encode_face
from PyNucleus_fem.femCy cimport local_matrix_t
from PyNucleus_fem.femCy import assembleMatrix, mass_1d_sym_scalar_anisotropic, mass_2d_sym_scalar_anisotropic
from PyNucleus_fem.quadrature import simplexXiaoGimbutas
from PyNucleus_base.sparsityPattern cimport sparsityPattern
from PyNucleus_base.linear_operators cimport (CSR_LinearOperator,
                                              SSS_LinearOperator,
                                              Dense_LinearOperator,
                                              VectorLinearOperator,
                                              ComplexVectorLinearOperator,
                                              Dense_VectorLinearOperator,
                                              Dense_SubBlock_LinearOperator,
                                              diagonalOperator,
                                              TimeStepperLinearOperator,
                                              nullOperator,
                                              sparseGraph,
                                              ComplexCSR_LinearOperator,
                                              ComplexSSS_LinearOperator,
                                              ComplexDense_LinearOperator,
                                              ComplexdiagonalOperator)
from PyNucleus_fem.splitting import dofmapSplitter
from PyNucleus_fem import dofmapFactory
from . twoPointFunctions cimport constantTwoPoint
from . fractionalOrders cimport (fractionalOrderBase,
                                 constFractionalOrder,
                                 piecewiseConstantFractionalOrder,
                                 variableFractionalOrder,
                                 singleVariableUnsymmetricFractionalOrder)
from . kernelNormalization cimport variableFractionalLaplacianScaling
from . kernels import getFractionalKernel
from . clusterMethodCy import (assembleFarFieldInteractions,
                               getDoFBoxesAndCells,
                               getFractionalOrders,
                               getAdmissibleClusters,
                               getCoveringClusters,
                               symmetrizeNearFieldClusters,
                               trimTree)
from . clusterMethodCy cimport (refinementType,
                                refinementParams,
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

cdef REAL_t INTERFACE_DOF = np.inf

LOGGER = logging.getLogger(__name__)


include "nonlocalAssembly_REAL.pxi"
include "nonlocalAssembly_COMPLEX.pxi"


# These functions are used by getEntry

cdef inline MASK_t getElemSymMask(DoFMap DoFMap, INDEX_t cellNo1, INDEX_t I, INDEX_t J):
    # Add symmetric 'contrib' to elements i and j in symmetric fashion
    cdef:
        INDEX_t p, q, K, L
        MASK_t k = <MASK_t> 1
        MASK_t mask
    mask.reset()
    for p in range(DoFMap.dofs_per_element):
        K = DoFMap.cell2dof(cellNo1, p)
        for q in range(p, DoFMap.dofs_per_element):
            L = DoFMap.cell2dof(cellNo1, q)
            if (I == K and J == L) or (J == K and I == L):
                mask |= k
            k = k << 1
    return mask


cdef inline MASK_t getElemElemSymMask(DoFMap DoFMap, INDEX_t cellNo1, INDEX_t cellNo2, INDEX_t I, INDEX_t J):
    # Add symmetric 'contrib' to elements i and j in symmetric fashion
    cdef:
        INDEX_t p, q, K, L
        MASK_t k = <MASK_t> 1
        MASK_t mask
    mask.reset()
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


cdef class horizonSurfaceIntegral(function):
    # x -> \int_{B_2(x, horizon)} kernel(x,y) dy
    cdef:
        Kernel kernel
        REAL_t horizon
        REAL_t[:, ::1] quadNodes
        REAL_t[::1] quadWeights, y

    def __init__(self, Kernel kernel, REAL_t horizon):
        cdef:
            INDEX_t k, numQuadNodes
            REAL_t inc
        self.kernel = kernel
        self.horizon = horizon
        if self.kernel.dim == 1:
            self.quadNodes = uninitialized((2, 1), dtype=REAL)
            self.quadWeights = uninitialized((2), dtype=REAL)
            self.quadNodes[0, 0] = self.horizon
            self.quadNodes[1, 0] = -self.horizon
            self.quadWeights[0] = 1.
            self.quadWeights[1] = 1.
        elif self.kernel.dim == 2:
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
        self.y = uninitialized((self.kernel.dim), dtype=REAL)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t fac = 0.
            INDEX_t k, j
            REAL_t val
            INDEX_t dim = self.kernel.dim
        for k in range(self.quadNodes.shape[0]):
            for j in range(dim):
                self.y[j] = x[j]+self.quadNodes[k, j]
            self.kernel.evalParams(x, self.y)
            val = self.kernel.eval(x, self.y)
            # val = self.kernel.scalingValue*pow(self.horizon, 1-dim-2*s)/s
            fac -= val * self.quadWeights[k]
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
            coeff = horizonSurfaceIntegral(builder.local_matrix_zeroExterior.kernel, horizon)
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

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        assert self.initialized
        return TimeStepperLinearOperator.matvec(self, x, y)

    cdef INDEX_t matvec_no_overwrite(self,
                                     REAL_t[::1] x,
                                     REAL_t[::1] y) except -1:
        assert self.initialized
        return TimeStepperLinearOperator.matvec_no_overwrite(self, x, y)


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


cdef LinearOperator getSparseNearField(DoFMap DoFMap, list Pnear, bint symmetric=False, tree_node myRoot=None):
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
    elif myRoot is not None:
        for clusterPair in Pnear:
            if clusterPair.n1.getParent(1).id != myRoot.id:
                continue
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
        self.n1 = n1
        self.n2 = n2

    cdef void set_cells(self):
        cdef:
            indexSet cells1, cells2
        cells1 = self.n1.get_cells()
        cells2 = self.n2.get_cells()
        self.cellsUnion = cells1.union(cells2)
        self.cellsInter = cells1.inter(cells2)
        assert len(cells1)+len(cells2) == len(self.cellsUnion)+len(self.cellsInter), (cells1.toSet(),
                                                                                      cells2.toSet(),
                                                                                      self.cellsInter.toSet(),
                                                                                      self.cellsUnion.toSet())

    def set_cells_py(self):
        self.set_cells()

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
