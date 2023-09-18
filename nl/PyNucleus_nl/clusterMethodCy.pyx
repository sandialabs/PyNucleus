###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes import INDEX, REAL
from PyNucleus_base import uninitialized
from libc.math cimport sqrt, sin, cos, atan2, M_PI as pi
from itertools import product
from scipy.special import gamma
import numpy as np
from PyNucleus_base.linear_operators cimport (LinearOperator,
                                              CSR_LinearOperator,
                                              sparseGraph,
                                              Multiply_Linear_Operator,
                                              IJOperator,
                                              sparseGraph)
from PyNucleus_base.blas cimport gemv, gemvT, mydot, matmat, norm, assign
from PyNucleus_base.io cimport DistributedMap, Import
from PyNucleus_base.sparsityPattern cimport sparsityPattern
from PyNucleus_fem.splitting import dofmapSplitter
from . nonlocalOperator cimport variableFractionalOrder
from . nonlocalAssembly cimport nearFieldClusterPair
from . kernelsCy cimport Kernel
from PyNucleus_fem.DoFMaps cimport DoFMap, P0_DoFMap, P1_DoFMap, P2_DoFMap, P3_DoFMap
from PyNucleus_fem.meshCy cimport meshBase
from PyNucleus_fem.functions cimport constant
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI

COMPRESSION = 'gzip'


def getRefinementParams(meshBase mesh, Kernel kernel, dict params={}):
    cdef:
        REAL_t singularity = kernel.max_singularity
        refinementParams refParams

    target_order = params.get('target_order', 2.)
    refParams.eta = params.get('eta', 3.)

    iO = params.get('interpolation_order', None)
    if iO is None:
        loggamma = abs(np.log(0.25))
        refParams.interpolation_order = max(np.ceil((2*target_order+max(-singularity, 2))*abs(np.log(mesh.hmin/mesh.diam))/loggamma/3.), 2)
    else:
        refParams.interpolation_order = iO
    mL = params.get('maxLevels', None)
    if mL is None:
        # maxLevels = max(int(np.around(np.log2(DoFMap.num_dofs)/mesh.dim-np.log2(refParams.interpolation_order))), 0)
        refParams.maxLevels = 200
    else:
        refParams.maxLevels = mL
    refParams.maxLevelsMixed = refParams.maxLevels
    mCS = params.get('minClusterSize', None)
    if mCS is None:
        refParams.minSize = refParams.interpolation_order**mesh.dim//2
    else:
        refParams.minSize = mCS
    if kernel.finiteHorizon:
        refParams.minMixedSize = max(min(kernel.horizon.value//(2*mesh.h)-1, refParams.minSize), 1)
    else:
        refParams.minMixedSize = refParams.minSize
    mFFBS = params.get('minFarFieldBlockSize', None)
    if mFFBS is None:
        # For this value, size(kernelInterpolant) == size(dense block)
        # If we choose a smaller value for minFarFieldBlockSize, then we use more memory,
        # but we might save time, since the assembly of a far field block is cheaper than a near field block.
        refParams.farFieldInteractionSize = refParams.interpolation_order**(2*mesh.dim)
    else:
        refParams.farFieldInteractionSize = mFFBS

    rT = params.get('refinementType', 'MEDIAN')
    refParams.refType = {'geometric': GEOMETRIC,
                         'GEOMETRIC': GEOMETRIC,
                         'median': MEDIAN,
                         'MEDIAN': MEDIAN,
                         'barycenter': BARYCENTER,
                         'BARYCENTER': BARYCENTER}[rT]

    refParams.splitEveryDim = params.get('splitEveryDim', False)

    refParams.attemptRefinement = True

    return refParams


cdef inline void merge_boxes(REAL_t[:, ::1] box1,
                             REAL_t[:, ::1] box2,
                             REAL_t[:, ::1] new_box):
    cdef INDEX_t i
    for i in range(box1.shape[0]):
        new_box[i, 0] = min(box1[i, 0], box2[i, 0])
        new_box[i, 1] = max(box1[i, 1], box2[i, 1])


cdef inline void merge_boxes2(REAL_t[:, ::1] box1,
                              REAL_t[:, :, ::1] box2,
                              INDEX_t dof,
                              REAL_t[:, ::1] new_box):
    cdef INDEX_t i
    for i in range(box1.shape[0]):
        new_box[i, 0] = min(box1[i, 0], box2[dof, i, 0])
        new_box[i, 1] = max(box1[i, 1], box2[dof, i, 1])


cdef inline bint inBox(const REAL_t[:, ::1] box,
                       const REAL_t[::1] vector):
    cdef:
        bint t = True
        INDEX_t i
    for i in range(box.shape[0]):
        t = t and (box[i, 0] <= vector[i]) and (vector[i] < box[i, 1])
    return t


cdef inline REAL_t minDist2FromBox(const REAL_t[:, ::1] box,
                                     const REAL_t[::1] vector):
    cdef:
        INDEX_t i
        REAL_t d2min = 0.
    for i in range(box.shape[0]):
        if vector[i] <= box[i, 0]:
            d2min += (vector[i]-box[i, 0])**2
        elif vector[i] >= box[i, 1]:
            d2min += (vector[i]-box[i, 1])**2
    return d2min


cdef inline tuple distsFromBox(const REAL_t[:, ::1] box,
                               const REAL_t[::1] vector):
    cdef:
        INDEX_t i
        REAL_t d2min = 0., d2max = 0.
    for i in range(box.shape[0]):
        if vector[i] <= box[i, 0]:
            d2min += (vector[i]-box[i, 0])**2
            d2max += (vector[i]-box[i, 1])**2
        elif vector[i] >= box[i, 1]:
            d2min += (vector[i]-box[i, 1])**2
            d2max += (vector[i]-box[i, 0])**2
        else:
            d2max += max((vector[i]-box[i, 0])**2, (vector[i]-box[i, 1])**2)
    return sqrt(d2min), sqrt(d2max)


cdef inline bint distFromSimplex(const REAL_t[:, ::1] simplex,
                                 const REAL_t[::1] vector,
                                 const REAL_t radius):
    cdef:
        INDEX_t i, j
        REAL_t t, p, q
        REAL_t w_mem[2]
        REAL_t z_mem[2]
        REAL_t[::1] w = w_mem
        REAL_t[::1] z = z_mem
    for i in range(3):
        for j in range(2):
            w[j] = simplex[(i+1) % 3, j] - simplex[i, j]
            z[j] = simplex[i, j]-vector[j]
        t = 1./mydot(w, w)
        p = 2*mydot(w, z)*t
        q = (mydot(z, z)-radius**2)*t
        q = 0.25*p**2-q
        if q > 0:
            q = sqrt(q)
            t = -0.5*p+q
            if (0 <= t) and (t <= 1):
                return True
            t = -0.5*p-q
            if (0 <= t) and (t <= 1):
                return True
    return False


cdef REAL_t distBoxes(REAL_t[:, ::1] box1, REAL_t[:, ::1] box2):
    cdef:
        REAL_t dist = 0., a2, b1
        INDEX_t i
    for i in range(box1.shape[0]):
        if box1[i, 0] > box2[i, 0]:
            b1 = box2[i, 1]
            a2 = box1[i, 0]
        else:
            b1 = box1[i, 1]
            a2 = box2[i, 0]
        dist += max(a2-b1, 0)**2
    return sqrt(dist)


cdef REAL_t maxDistBoxes(REAL_t[:, ::1] box1, REAL_t[:, ::1] box2):
    cdef:
        REAL_t dist = 0., a2, b1
        INDEX_t i
    for i in range(box1.shape[0]):
        if box1[i, 0] > box2[i, 0]:
            b1 = box2[i, 0]
            a2 = box1[i, 1]
        else:
            b1 = box1[i, 0]
            a2 = box2[i, 1]
        dist += max(a2-b1, 0)**2
    return sqrt(dist)

cdef REAL_t diamBox(REAL_t[:, ::1] box):
    cdef:
        REAL_t d = 0.
        INDEX_t i
    for i in range(box.shape[0]):
        d += (box[i, 1]-box[i, 0])**2
    return sqrt(d)


cdef class tree_node:
    def __init__(self, tree_node parent, indexSet dofs, REAL_t[:, :, ::1] boxes, bint mixed_node=False, bint canBeAssembled=True):
        cdef:
            INDEX_t dof = -1
            indexSetIterator it
        self.parent = parent
        self.dim = boxes.shape[1]
        self.children = []
        self._num_dofs = -1
        self._dofs = dofs
        self.mixed_node = mixed_node
        self.canBeAssembled = canBeAssembled
        self._irregularLevelsOffset = 0
        if parent is None:
            self.levelNo = 0
            self.id = 0
            self.distFromRoot = 0
        else:
            self.levelNo = self.parent.levelNo+1
            self.distFromRoot = self.parent.distFromRoot+1
        if self.dim > 0:
            self.box = uninitialized((self.dim, 2), dtype=REAL)
            self.box[:, 0] = np.inf
            self.box[:, 1] = -np.inf
            it = self.get_dofs().getIter()
            while it.step():
                dof = it.i
                merge_boxes2(self.box, boxes, dof, self.box)

    def __eq__(self, tree_node other):
        for j in range(self.dim):
            if ((abs(self.box[j, 0]-other.box[j, 0]) > 1e-10) or
                (abs(self.box[j, 1]-other.box[j, 1]) > 1e-10)):
                return False
        return True

    cdef indexSet get_dofs(self):
        cdef:
            indexSet dofs
            tree_node c
        if self.get_is_leaf():
            return self._dofs
        else:
            dofs = arrayIndexSet()
            for c in self.children:
                dofs = dofs.union(c.get_dofs())
            return dofs

    cdef indexSet get_local_dofs(self):
        cdef:
            indexSet dofs
            tree_node c
        if self.get_is_leaf():
            return self._local_dofs
        else:
            print('ouch!')
            dofs = arrayIndexSet()
            for c in self.children:
                dofs = dofs.union(c.get_local_dofs())
            return dofs

    @property
    def dofs(self):
        return self.get_dofs()

    @property
    def local_dofs(self):
        return self.get_local_dofs()

    cdef INDEX_t get_num_dofs(self):
        if self._num_dofs < 0:
            self._num_dofs = self.get_dofs().getNumEntries()
        return self._num_dofs

    @property
    def num_dofs(self):
        if self._num_dofs < 0:
            self._num_dofs = self.get_dofs().getNumEntries()
        return self._num_dofs

    cdef indexSet get_cells(self):
        cdef:
            indexSet s
            tree_node c
        if self.get_is_leaf():
            return self._cells
        else:
            s = arrayIndexSet()
            for c in self.children:
                s = s.union(c.get_cells())
            return s

    @property
    def cells(self):
        return self.get_cells()

    def get_nodes(self):
        if self.get_is_leaf():
            return 1
        else:
            return 1+sum([c.nodes for c in self.children])

    nodes = property(fget=get_nodes)

    def get_irregularLevelsOffset(self):
        if self._irregularLevelsOffset < 0:
            if self.parent is not None:
                self._irregularLevelsOffset = self.parent.irregularLevelsOffset
            elif self.parent is None:
                raise Exception()
        return self._irregularLevelsOffset

    def set_irregularLevelsOffset(self, INDEX_t irregularLevelsOffset):
        self._irregularLevelsOffset = irregularLevelsOffset

    irregularLevelsOffset = property(fget=get_irregularLevelsOffset,
                                     fset=set_irregularLevelsOffset)

    cdef get_num_root_children(self):
        if self.parent is None:
            return self.get_num_children(self.irregularLevelsOffset)
        else:
            return self.parent.get_num_root_children()

    cdef get_num_children(self, INDEX_t levelOffset=1):
        cdef:
            tree_node n
            INDEX_t nc = 0
        if levelOffset > 1:
            for n in self.children:
                nc += n.get_num_children(levelOffset-1)
            return nc
        else:
            return len(self.children)

    def refine(self,
               REAL_t[:, :, ::1] boxes,
               REAL_t[:, ::1] coords,
               refinementParams refParams,
               BOOL_t recursive=True):
        cdef:
            refinementType refType = refParams.refType
            indexSet dofs = self.get_dofs()
            INDEX_t num_initial_dofs = dofs.getNumEntries(), dim, i = -1, j, num_dofs, k
            REAL_t[:, ::1] subbox
            indexSet s
            indexSetIterator it = dofs.getIter()
            set sPre, sPre0, sPre1
            list preSets
            INDEX_t nD = 0
            REAL_t[:, ::1] center
            REAL_t[::1] splitPoint
            REAL_t[::1] coords0, coords1
            list blob
            INDEX_t splitDimension
            REAL_t m0 = 0., m1 = 0., median = 0., maxBoxSize
            list children = []
            INDEX_t maxNumChildren, lvlNo, numRootChildren, lvlID
        if not self.mixed_node:
            if (self.levelNo+1 >= refParams.maxLevels) or (num_initial_dofs <= refParams.minSize):
                return
        else:
            if (self.levelNo+1 >= refParams.maxLevelsMixed) or (num_initial_dofs <= refParams.minMixedSize):
                return
        dim = self.box.shape[0]

        if dim == 1:
            maxNumChildren = 2
            lvlNo = self.levelNo
            numRootChildren = self.get_num_root_children()
            if refType == GEOMETRIC:
                # divide box into equal sized subboxes
                m0 = 0.5*(self.box[0, 0] + self.box[0, 1])
            elif refType == BARYCENTER:
                # divide box at bary-center of DoF coords
                m0 = 0.
                it.reset()
                while it.step():
                    i = it.i
                    m0 += coords[i, 0]
                m0 /= num_initial_dofs
            elif refType == MEDIAN:
                coords0 = np.zeros((num_initial_dofs), dtype=REAL)
                it.reset()
                k = 0
                while it.step():
                    i = it.i
                    coords0[k] = coords[i, 0]
                    k += 1
                m0 = np.median(coords0)

            subbox = uninitialized((dim, 2), dtype=REAL)
            subbox[0, 0] = self.box[0, 0]
            subbox[0, 1] = m0
            sPre0 = set()
            sPre1 = set()
            it.reset()
            while it.step():
                i = it.i
                if inBox(subbox, coords[i, :]):
                    sPre0.add(i)
                else:
                    sPre1.add(i)

            s = arrayIndexSet()
            s.fromSet(sPre0)
            num_dofs = s.getNumEntries()
            if num_dofs >= refParams.minSize and num_dofs < num_initial_dofs:
                nD += num_dofs
                children.append(tree_node(self, s, boxes, mixed_node=self.mixed_node))
                if lvlNo > 0:
                    lvlID = self.id-(numRootChildren*(maxNumChildren**(lvlNo-1)-1)//(maxNumChildren-1)+1)
                    children[0].id = numRootChildren*(maxNumChildren**lvlNo-1)//(maxNumChildren-1) + 1 + maxNumChildren*lvlID + 0
                else:
                    children[0].id = 1
            else:
                return

            s = arrayIndexSet()
            s.fromSet(sPre1)
            num_dofs = s.getNumEntries()
            if num_dofs >= refParams.minSize and num_dofs < num_initial_dofs:
                nD += num_dofs
                children.append(tree_node(self, s, boxes, mixed_node=self.mixed_node))
                if lvlNo > 0:
                    lvlID = self.id-(numRootChildren*(maxNumChildren**(lvlNo-1)-1)//(maxNumChildren-1)+1)
                    children[1].id = numRootChildren*(maxNumChildren**lvlNo-1)//(maxNumChildren-1) + 1 + maxNumChildren*lvlID + 1
                else:
                    children[1].id = 2
            else:
                return
        elif dim == 2 and (not self.mixed_node and refParams.splitEveryDim):
            maxNumChildren = 2**dim
            lvlNo = self.levelNo
            numRootChildren = self.get_num_root_children()
            if refType != MEDIAN:
                if refType == GEOMETRIC:
                    # divide box into equal sized subboxes
                    splitPoint = uninitialized((dim), dtype=REAL)
                    for i in range(dim):
                        splitPoint[i] = 0.5*(self.box[i, 0] + self.box[i, 1])
                elif refType == BARYCENTER:
                    # divide box at bary-center of DoF coords
                    splitPoint = np.zeros((dim), dtype=REAL)
                    it.reset()
                    while it.step():
                        i = it.i
                        for j in range(dim):
                            splitPoint[j] += coords[i, j]
                    for i in range(dim):
                        splitPoint[i] /= num_initial_dofs
                elif refType == MEDIAN:
                    # divide box at median of DoF coords
                    center = np.zeros((num_initial_dofs, dim), dtype=REAL)
                    it.reset()
                    k = 0
                    while it.step():
                        i = it.i
                        for j in range(dim):
                            center[k, j] = coords[i, j]
                        k += 1
                    splitPoint = np.median(center, axis=0)
                    del center

                preSets = [set() for k in range(2**dim)]
                it.reset()
                while it.step():
                    k = 0
                    i = it.i
                    for j in range(dim):
                        if coords[i, j] > splitPoint[j]:
                            k += 1 << j
                    preSets[k].add(i)
                for k in range(2**dim):
                    s = arrayIndexSet()
                    s.fromSet(preSets[k])
                    num_dofs = s.getNumEntries()
                    if num_dofs >= refParams.minSize and num_dofs < num_initial_dofs:
                        nD += num_dofs
                        children.append(tree_node(self, s, boxes, mixed_node=self.mixed_node))
                        if lvlNo > 0:
                            lvlID = self.id-(numRootChildren*(maxNumChildren**(lvlNo-1)-1)//(maxNumChildren-1)+1)
                            children[k].id = numRootChildren*(maxNumChildren**lvlNo-1)//(maxNumChildren-1) + 1 + maxNumChildren*lvlID + k
                        else:
                            children[k].id = k+1
                    elif num_dofs == 0:
                        pass
                    else:
                        return
            else:
                # refType == MEDIAN
                # divide box at median of DoF coords
                coords0 = np.zeros((num_initial_dofs), dtype=REAL)
                it.reset()
                k = 0
                while it.step():
                    i = it.i
                    coords0[k] = coords[i, 0]
                    k += 1
                m0 = np.median(coords0)
                for p in range(2):
                    blob = []
                    it.reset()
                    if p == 0:
                        while it.step():
                            i = it.i
                            if coords[i, 0] < m0:
                                blob.append(i)
                    else:
                        while it.step():
                            i = it.i
                            if coords[i, 0] >= m0:
                                blob.append(i)
                    coords1 = np.zeros((len(blob)), dtype=REAL)
                    k = 0
                    for i in blob:
                        coords1[k] = coords[i, 1]
                        k += 1
                    m1 = np.median(coords1)
                    sPre0 = set()
                    sPre1 = set()
                    for i in blob:
                        if coords[i, 1] < m1:
                            sPre0.add(i)
                        else:
                            sPre1.add(i)

                    s = arrayIndexSet()
                    s.fromSet(sPre0)
                    num_dofs = s.getNumEntries()
                    if num_dofs >= refParams.minSize and num_dofs < num_initial_dofs:
                        nD += num_dofs
                        children.append(tree_node(self, s, boxes, mixed_node=self.mixed_node))
                        if lvlNo > 0:
                            lvlID = self.id-(numRootChildren*(maxNumChildren**(lvlNo-1)-1)//(maxNumChildren-1)+1)
                            children[p*dim+0].id = numRootChildren*(maxNumChildren**lvlNo-1)//(maxNumChildren-1) + 1 + maxNumChildren*lvlID + p*dim+0
                        else:
                            children[p*dim+0].id = p*dim+1
                    elif num_dofs == 0:
                        pass
                    else:
                        return
                    s = arrayIndexSet()
                    s.fromSet(sPre1)
                    num_dofs = s.getNumEntries()
                    if num_dofs >= refParams.minSize and num_dofs < num_initial_dofs:
                        nD += num_dofs
                        children.append(tree_node(self, s, boxes, mixed_node=self.mixed_node))
                        if lvlNo > 0:
                            lvlID = self.id-(numRootChildren*(maxNumChildren**(lvlNo-1)-1)//(maxNumChildren-1)+1)
                            children[p*dim+1].id = numRootChildren*(maxNumChildren**lvlNo-1)//(maxNumChildren-1) + 1 + maxNumChildren*lvlID + p*dim+1
                        else:
                            children[p*dim+1].id = p*dim+2
                    else:
                        return
        else:
            # dim == 2, mixed_node or not splitEveryDim
            maxNumChildren = 2
            lvlNo = self.levelNo
            numRootChildren = self.get_num_root_children()

            splitDimension = 0
            maxBoxSize = self.box[0, 1]-self.box[0, 0]
            for i in range(1, dim):
                if self.box[i, 1]-self.box[i, 0] > maxBoxSize:
                    splitDimension = i
                    maxBoxSize = self.box[i, 1]-self.box[i, 0]
            if refType == MEDIAN:
                coords0 = np.zeros((num_initial_dofs), dtype=REAL)
                it.reset()
                k = 0
                while it.step():
                    i = it.i
                    coords0[k] = coords[i, splitDimension]
                    k += 1
                median = np.median(coords0)
                del coords0
            # split along larger box dimension
            subbox = uninitialized((dim, 2), dtype=REAL)

            for i in range(dim):
                subbox[i, 0] = self.box[i, 0]-1e-12
                subbox[i, 1] = self.box[i, 1]+1e-12
            if refType == GEOMETRIC:
                subbox[splitDimension, 1] = (self.box[splitDimension, 0] + self.box[splitDimension, 1])*0.5
            elif refType == MEDIAN:
                subbox[splitDimension, 1] = median
            sPre0 = set()
            sPre1 = set()
            it.reset()
            while it.step():
                i = it.i
                if inBox(subbox, coords[i, :]):
                    sPre0.add(i)
                else:
                    sPre1.add(i)

            s = arrayIndexSet()
            s.fromSet(sPre0)
            num_dofs = s.getNumEntries()
            if num_dofs >= refParams.minSize and num_dofs < num_initial_dofs:
                nD += num_dofs
                children.append(tree_node(self, s, boxes, mixed_node=self.mixed_node))
                if lvlNo > 0:
                    lvlID = self.id-(numRootChildren*(maxNumChildren**(lvlNo-1)-1)//(maxNumChildren-1)+1)
                    children[0].id = numRootChildren*(maxNumChildren**lvlNo-1)//(maxNumChildren-1) + 1 + maxNumChildren*lvlID + 0
                else:
                    children[0].id = 1
            else:
                return

            s = arrayIndexSet()
            s.fromSet(sPre1)
            num_dofs = s.getNumEntries()
            if num_dofs >= refParams.minSize and num_dofs < num_initial_dofs:
                nD += num_dofs
                children.append(tree_node(self, s, boxes, mixed_node=self.mixed_node))
                if lvlNo > 0:
                    lvlID = self.id-(numRootChildren*(maxNumChildren**(lvlNo-1)-1)//(maxNumChildren-1)+1)
                    children[1].id = numRootChildren*(maxNumChildren**lvlNo-1)//(maxNumChildren-1) + 1 + maxNumChildren*lvlID + 1
                else:
                    children[1].id = 2
            else:
                return
        if len(children) <= 1:
            return
        self.children = children

        assert nD == 0 or nD == num_initial_dofs, (nD, num_initial_dofs, np.array(self.box), [np.array(c.box) for c in self.children])
        if nD == num_initial_dofs:
            self._dofs = None
        else:
            assert self.get_is_leaf()

        if recursive:
            for k in range(len(self.children)):
                self.children[k].refine(boxes, coords, refParams, recursive)

    def idsAreUnique(self, bitArray used=None):
        cdef:
            tree_node c
            BOOL_t unique = True
        if used is None:
            used = bitArray(maxElement=self.get_max_id())
        for c in self.children:
            unique &= c.idsAreUnique(used)
        if not used.inSet(self.id):
            used.set(self.id)
        else:
            print('ID {} not unique!'.format(self.id))
            unique = False
        return unique

    cdef BOOL_t get_is_leaf(self):
        return len(self.children) == 0

    def get_is_leaf_py(self):
        return len(self.children) == 0

    isLeaf = property(fget=get_is_leaf_py)

    def leaves(self):
        cdef:
            tree_node i, j
        if self.get_is_leaf():
            yield self
        else:
            for i in self.children:
                for j in i.leaves():
                    yield j

    def get_tree_nodes(self):
        cdef:
            tree_node i, j
        yield self
        for i in self.children:
            for j in i.get_tree_nodes():
                yield j

    def get_tree_nodes_up_to_level(self, INDEX_t level):
        cdef:
            tree_node i, j
        yield self
        if level > 0:
            for i in self.children:
                for j in i.get_tree_nodes_up_to_level(level-1):
                    yield j

    cdef INDEX_t _getLevels(self):
        cdef:
            tree_node c
            INDEX_t l
        if self.get_is_leaf():
            return 1
        else:
            l = 0
            for c in self.children:
                l = max(l, c._getLevels())
            return 1+l

    def _getLevels_py(self):
        return self._getLevels()

    numLevels = property(fget=_getLevels_py)

    cdef INDEX_t _getParentLevels(self):
        if self.parent is None:
            return 0
        else:
            return self.parent._getParentLevels() + 1

    def getParent(self, INDEX_t parentLevel=0):
        if parentLevel >= 0:
            parentLevel = parentLevel-self._getParentLevels()-1
            assert parentLevel < 0
        if parentLevel == -1:
            return self
        else:
            return self.parent.getParent(parentLevel + 1)

    def plot(self, level=0, plotType='box', DoFMap dm=None, BOOL_t recurse=False, BOOL_t printClusterIds=False, BOOL_t printNumDoFs=False, transferOperatorColor='purple', coefficientsColor='red', horizontal=True, levelSkip=1, skip_angle=0.):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        cdef:
            indexSet dofs
            indexSetIterator it
            INDEX_t dof, k, j
            REAL_t[:, ::1] points
            INDEX_t[::1] idx
            REAL_t[::1] x, y
            REAL_t[:, ::1] dofCoords

        if plotType in ('treeDoFCoords', 'treeDoF'):
            if plotType == 'treeDoFCoords':
                assert dm is not None
                dofCoords = dm.getDoFCoordinates()
            elif plotType == 'treeDoF':
                assert self.dim == 1
            if self.parent is not None:
                dofs = self.get_dofs()
                center = np.zeros((self.dim), dtype=REAL)
                it = dofs.getIter()
                k = 0
                while it.step():
                    dof = it.i
                    if plotType == 'treeDoFCoords':
                        for j in range(self.dim):
                            center[j] += dofCoords[dof, j]
                    else:
                        center[0] += dof
                    k += 1
                for j in range(self.dim):
                    center[j] /= len(dofs)

                dofs = self.parent.get_dofs()
                pcenter = np.zeros((self.dim), dtype=REAL)
                it = dofs.getIter()
                k = 0
                while it.step():
                    dof = it.i
                    if plotType == 'treeDoFCoords':
                        for j in range(self.dim):
                            pcenter[j] += dofCoords[dof, j]
                    else:
                        pcenter[0] += dof
                    k += 1
                for j in range(self.dim):
                    pcenter[j] /= len(dofs)

                if self.dim == 1:
                    if horizontal:
                        plt.plot([pcenter[0], center[0]], [level+levelSkip, level], c='k')
                        level = level-levelSkip
                    else:
                        plt.plot([level-levelSkip, level], [pcenter[0], center[0]], c='k')
                        level = level+levelSkip
                else:
                    offset = np.array(dofCoords).max(axis=0)-np.array(dofCoords).min(axis=0)
                    offset[0] = np.cos(skip_angle)*offset[0]
                    offset[1] = -np.sin(skip_angle)*offset[1]
                    mesh = dm.mesh.copy()
                    for k in range(mesh.num_vertices):
                        for j in range(self.dim):
                            mesh.vertices[k, j] += offset[j]*level
                    mesh.plot(vertices=False)
                    plt.gca().add_patch(patches.Rectangle((self.box[0, 0]+offset[0]*level, self.box[1, 0]+offset[1]*level),
                                                          self.box[0, 1]-self.box[0, 0],
                                                          self.box[1, 1]-self.box[1, 0],
                                                          fill=False,
                                                          color='red' if self.mixed_node else 'blue'))
                    plt.plot([pcenter[0]+offset[0]*(level-levelSkip), center[0]+offset[0]*level],
                             [pcenter[1]+offset[1]*(level-levelSkip), center[1]+offset[1]*level], c='k')
                    level = level+levelSkip
            else:
                offset = np.array(dofCoords).max(axis=0)-np.array(dofCoords).min(axis=0)
                offset[0] = np.cos(skip_angle)*offset[0]
                offset[1] = -np.sin(skip_angle)*offset[1]
                level = level-levelSkip
                mesh = dm.mesh.copy()
                for k in range(mesh.num_vertices):
                    for j in range(self.dim):
                        mesh.vertices[k, j] += offset[j]*level
                mesh.plot(vertices=False)
                plt.gca().add_patch(patches.Rectangle((self.box[0, 0]+offset[0]*level, self.box[1, 0]+offset[1]*level),
                                                          self.box[0, 1]-self.box[0, 0],
                                                          self.box[1, 1]-self.box[1, 0],
                                                          fill=False,
                                                          color='red' if self.mixed_node else 'blue'))

                level = level+levelSkip

            if recurse:
                for c in self.children:
                    c.plot(level=level, plotType=plotType, dm=dm, recurse=recurse,
                           printClusterIds=printClusterIds, printNumDoFs=printNumDoFs, horizontal=horizontal,
                           levelSkip=levelSkip,
                           skip_angle=skip_angle)

        elif plotType == 'DoFCoords':
            # plot the convex hull of the dofs of each tree node
            assert dm is not None
            dofCoords = dm.getDoFCoordinates()
            from scipy.spatial import ConvexHull
            if self.dim == 1:
                dofs = self.get_dofs()
                points = uninitialized((len(dofs), self.dim), dtype=REAL)
                it = dofs.getIter()
                k = 0
                while it.step():
                    dof = it.i
                    for j in range(self.dim):
                        points[k, j] = dofCoords[dof, j]
                    k += 1
                y = level*np.ones((len(dofs)), dtype=REAL)
                plt.plot(np.array(points)[:, 0], np.array(y), color='red' if self.mixed_node else 'blue')
                if printClusterIds and printNumDoFs:
                    label = 'id={},nd={}'.format(self.id, self.num_dofs)
                elif printClusterIds:
                    label = '{}'.format(self.id)
                elif printNumDoFs:
                    label = '{}'.format(self.num_dofs)
                if printClusterIds or printNumDoFs:
                    myCenter = np.mean(self.box, axis=1)
                    plt.text(myCenter[0], level, s=label,
                             horizontalalignment='center',
                             verticalalignment='center')
                if recurse:
                    for c in self.children:
                        c.plot(level=level+1, plotType=plotType, dofCoords=dofCoords, recurse=recurse,
                               printClusterIds=printClusterIds, printNumDoFs=printNumDoFs)
            elif self.dim == 2:
                dofs = self.get_dofs()
                points = uninitialized((len(dofs), self.dim), dtype=REAL)
                it = dofs.getIter()
                k = 0
                while it.step():
                    dof = it.i
                    for j in range(self.dim):
                        points[k, j] = dofCoords[dof, j]
                    k += 1
                if len(dofs) > 2:
                    hull = ConvexHull(points, qhull_options='Qt QJ')
                    idx = hull.vertices
                else:
                    idx = np.arange(len(dofs), dtype=INDEX)
                x = uninitialized((idx.shape[0]+1), dtype=REAL)
                y = uninitialized((idx.shape[0]+1), dtype=REAL)
                for k in range(idx.shape[0]):
                    x[k] = points[idx[k], 0]
                    y[k] = points[idx[k], 1]
                x[idx.shape[0]] = points[idx[0], 0]
                y[idx.shape[0]] = points[idx[0], 1]
                # plt.plot(x, y, color='red' if self.mixed_node else 'blue')

                import matplotlib.patches as patches
                if recurse and self.parent is None:
                    numPlots = self.numLevels
                    numRows = int(np.sqrt(numPlots))
                    numCols = numPlots//numRows
                    numRows = numPlots//numCols
                    if numRows*numCols<numPlots:
                        numRows += 1
                    plt.subplots(numRows, numCols)
                    for k in range(self.numLevels):
                        plt.gcf().axes[k].set_xlim([self.box[0, 0], self.box[0, 1]])
                        plt.gcf().axes[k].set_ylim([self.box[1, 0], self.box[1, 1]])
                        plt.gcf().axes[k].set_aspect('equal')

                if recurse:
                    ax = plt.gcf().axes[level]
                else:
                    ax = plt.gca()

                ax.plot(np.array(x), np.array(y), color='red' if self.mixed_node else 'blue')
                if printClusterIds and printNumDoFs:
                    label = 'id={},nd={}'.format(self.id, self.num_dofs)
                elif printClusterIds:
                    label = '{}'.format(self.id)
                elif printNumDoFs:
                    label = '{}'.format(self.num_dofs)
                if printClusterIds or printNumDoFs:
                    myCenter = np.mean(self.box, axis=1)
                    ax.text(myCenter[0], myCenter[1], s=label,
                            horizontalalignment='center',
                            verticalalignment='center')

                if recurse and not self.get_is_leaf():
                    for c in self.children:
                        c.plot(level+1, plotType, dm, recurse, printClusterIds, printNumDoFs)
            else:
                raise NotImplementedError()
        elif plotType == 'DoF':
            if self.dim == 1:
                dofs = self.get_dofs()

                spacing = 0.2

                try:
                    self.parent.transferOperator
                    if horizontal:
                        # box1 = [min(dofs)+spacing, min(dofs)+self.transferOperator.shape[0]-spacing]
                        box1 = [0.5*(min(dofs)+max(dofs))-0.5*self.transferOperator.shape[0]+spacing, 0.5*(min(dofs)+max(dofs))+0.5*self.transferOperator.shape[0]-spacing]
                        box2 = [level+spacing, level+self.transferOperator.shape[0]-spacing]
                    else:
                        box1 = [level+spacing, level+self.transferOperator.shape[0]-spacing]
                        # box2 = [max(dofs)-self.transferOperator.shape[0]+spacing, max(dofs)-spacing]
                        box2 = [0.5*(min(dofs)+max(dofs))-0.5*self.transferOperator.shape[0]+spacing, 0.5*(min(dofs)+max(dofs))+0.5*self.transferOperator.shape[0]-spacing]
                    plt.gca().add_patch(patches.Rectangle((box1[0], box2[0]), box1[1]-box1[0], box2[1]-box2[0], fill=True, facecolor=transferOperatorColor))

                    if printClusterIds and printNumDoFs:
                        label = 'id={},nd={}'.format(self.id, self.num_dofs)
                    elif printClusterIds:
                        label = '{}'.format(self.id)
                    elif printNumDoFs:
                        label = '{}'.format(self.num_dofs)
                    if printClusterIds or printNumDoFs:
                        myCenter = np.mean(self.box, axis=1)
                        plt.text(myCenter[0], level, s=label,
                                 horizontalalignment='center',
                                 verticalalignment='center')
                    if horizontal:
                        level += self.transferOperator.shape[0]
                    else:
                        level -= self.transferOperator.shape[0]
                except:
                    pass
                try:
                    self.value
                    if horizontal:
                        box1 = [min(dofs)+spacing, max(dofs)-spacing]
                        box2 = [level+spacing, level+self.transferOperator.shape[0]-spacing]
                    else:
                        box1 = [level+spacing, level+self.transferOperator.shape[0]-spacing]
                        box2 = [min(dofs)+spacing, max(dofs)-spacing]
                    plt.gca().add_patch(patches.Rectangle((box1[0], box2[0]), box1[1]-box1[0], box2[1]-box2[0], fill=True, facecolor=coefficientsColor))

                    if horizontal:
                        level += self.transferOperator.shape[0]
                    else:
                        level -= self.transferOperator.shape[0]
                except:
                    pass
                returnLevel = level
                if recurse:
                    for c in self.children:
                        returnLevel = c.plot(level=level, plotType=plotType, dm=dm, recurse=recurse,
                                             printClusterIds=printClusterIds, printNumDoFs=printNumDoFs, horizontal=horizontal, transferOperatorColor=transferOperatorColor, coefficientsColor=coefficientsColor)
                return returnLevel

        elif plotType == 'box':
            # plot the box for each tree node
            import matplotlib.patches as patches
            if self.dim == 2:
                if recurse and self.parent is None:
                    numPlots = self.numLevels
                    numRows = int(np.sqrt(numPlots))
                    numCols = numPlots//numRows
                    numRows = numPlots//numCols
                    if numRows*numCols<numPlots:
                        numRows += 1
                    plt.subplots(numRows, numCols)
                    for k in range(self.numLevels):
                        plt.gcf().axes[k].set_xlim([self.box[0, 0], self.box[0, 1]])
                        plt.gcf().axes[k].set_ylim([self.box[1, 0], self.box[1, 1]])
                        plt.gcf().axes[k].set_aspect('equal')
                    for k in range(self.numLevels, len(plt.gcf().axes)):
                        plt.gcf().axes[k].set_axis_off()
                if recurse:
                    ax = plt.gcf().axes[level]
                else:
                    ax = plt.gca()

                ax.add_patch(patches.Rectangle((self.box[0, 0], self.box[1, 0]),
                                               self.box[0, 1]-self.box[0, 0],
                                               self.box[1, 1]-self.box[1, 0],
                                               fill=False,
                                               color='red' if self.mixed_node else 'blue'))
                if recurse and not self.get_is_leaf():
                    myCenter = np.mean(self.box, axis=1)
                    for c in self.children:
                        # cCenter = np.mean(c.box, axis=1)
                        # plt.arrow(myCenter[0], myCenter[1], cCenter[0]-myCenter[0], cCenter[1]-myCenter[1])
                        # plt.text(cCenter[0], cCenter[1], s=str(level+1))
                        c.plot(level=level+1, plotType='box', recurse=recurse)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(plotType)

    cdef void prepareTransferOperators(self, INDEX_t m, transferMatrixBuilder tMB=None):
        cdef:
            tree_node c
            INDEX_t interactionSize = m**self.dim
        if tMB is None:
            tMB = transferMatrixBuilder(m, self.dim)
        if not self.get_is_leaf():
            for c in self.children:
                c.prepareTransferOperators(m, tMB)
        if self.parent is not None:
            self.transferOperator = uninitialized((interactionSize, interactionSize),
                                                  dtype=REAL)
            tMB.build(self.parent.box, self.box,
                      self.transferOperator)
        self.coefficientsUp = uninitialized((interactionSize), dtype=REAL)
        self.coefficientsDown = uninitialized((interactionSize), dtype=REAL)

    def prepareTransferOperators_py(self, INDEX_t m):
        self.prepareTransferOperators(m)

    cdef void upwardPass(self, REAL_t[::1] x, INDEX_t componentNo=0, BOOL_t skip_leaves=False, BOOL_t local=False):
        cdef:
            INDEX_t i, dof = -1, k = 0
            tree_node c
            indexSetIterator it
            REAL_t temp_x
        if self.get_is_leaf():
            if skip_leaves:
                # for i in range(self.coefficientsUp.shape[0]):
                #     self.coefficientsUp[i] = x[self.coefficientsUpOffset+i]
                pass
            else:
                self.coefficientsUp[:] = 0.0
                if not local:
                    it = self.get_dofs().getIter()
                    while it.step():
                        dof = it.i
                        temp_x = x[dof]
                        for i in range(self.coefficientsUp.shape[0]):
                            self.coefficientsUp[i] += temp_x*self.value[componentNo, k, i]
                        k += 1
                else:
                    it = self.get_local_dofs().getIter()
                    while it.step():
                        dof = it.i
                        temp_x = x[dof]
                        for i in range(self.coefficientsUp.shape[0]):
                            self.coefficientsUp[i] += temp_x*self.value[componentNo, k, i]
                        k += 1
        else:
            self.coefficientsUp[:] = 0.0
            for c in self.children:
                c.upwardPass(x, componentNo, skip_leaves, local)
                gemv(c.transferOperator, c.coefficientsUp, self.coefficientsUp, 1.)

    def upwardPass_py(self, REAL_t[::1] x, INDEX_t componentNo=0, BOOL_t skip_leaves=False):
        self.upwardPass(x, componentNo, skip_leaves)

    cdef void resetCoefficientsDown(self):
        cdef:
            tree_node c
        self.coefficientsDown[:] = 0.0
        if not self.get_is_leaf():
            for c in self.children:
                c.resetCoefficientsDown()

    def resetCoefficientsDown_py(self):
        self.resetCoefficientsDown()

    cdef void resetCoefficientsUp(self):
        cdef:
            tree_node c
        self.coefficientsUp[:] = 0.0
        if not self.get_is_leaf():
            for c in self.children:
                c.resetCoefficientsUp()

    def resetCoefficientsUp_py(self):
        self.resetCoefficientsUp()

    cdef void downwardPass(self, REAL_t[::1] y, INDEX_t componentNo=0, BOOL_t local=False):
        cdef:
            INDEX_t i, dof = -1, k = 0
            REAL_t val
            tree_node c
            indexSetIterator it
        if self.get_is_leaf():
            if not local:
                it = self.get_dofs().getIter()
                while it.step():
                    dof = it.i
                    val = 0.0
                    for i in range(self.coefficientsDown.shape[0]):
                        val += self.value[componentNo, k, i]*self.coefficientsDown[i]
                    y[dof] += val
                    k += 1
            else:
                it = self.get_local_dofs().getIter()
                while it.step():
                    dof = it.i
                    val = 0.0
                    for i in range(self.coefficientsDown.shape[0]):
                        val += self.value[componentNo, k, i]*self.coefficientsDown[i]
                    y[dof] += val
                    k += 1
        else:
            for c in self.children:
                gemvT(c.transferOperator, self.coefficientsDown, c.coefficientsDown, 1.)
                c.downwardPass(y, componentNo, local)

    def downwardPass_py(self, REAL_t[::1] y, INDEX_t componentNo=0):
        self.downwardPass(y, componentNo)

    def enterLeafValues(self,
                        meshBase mesh,
                        DoFMap DoFMap,
                        INDEX_t order,
                        REAL_t[:, :, ::1] boxes,
                        comm=None,
                        BOOL_t assembleOnRoot=True,
                        BOOL_t local=False):
        cdef:
            INDEX_t i, k, I, l, j, p, dim, manifold_dim, dof = -1, r, start, end
            REAL_t[:, ::1] coeff, simplex, local_vals, PHI, xi, x
            REAL_t[::1] eta, fvals
            REAL_t vol, beta, omega
            tree_node n
            simplexQuadratureRule qr
            indexSetIterator it = arrayIndexSetIterator()
            productIterator pit = productIterator(order, mesh.dim)
            transferMatrixBuilder tMB
            REAL_t[:, ::1] transferOperator
        dim = mesh.dim
        manifold_dim = mesh.manifold_dim
        # Sauter Schwab p. 428
        if isinstance(DoFMap, P0_DoFMap):
            quadOrder = order+1
        elif isinstance(DoFMap, P1_DoFMap):
            quadOrder = order+2
        elif isinstance(DoFMap, P2_DoFMap):
            quadOrder = order+3
        elif isinstance(DoFMap, P3_DoFMap):
            quadOrder = order+4
        else:
            raise NotImplementedError()
        qr = simplexXiaoGimbutas(quadOrder, dim, manifold_dim)

        # get values of basis function in quadrature nodes
        PHI = uninitialized((DoFMap.dofs_per_element, qr.num_nodes), dtype=REAL)
        for i in range(DoFMap.dofs_per_element):
            for j in range(qr.num_nodes):
                PHI[i, j] = DoFMap.localShapeFunctions[i](qr.nodes[:, j])

        coeff = np.zeros((DoFMap.num_dofs, order**dim), dtype=REAL)
        simplex = uninitialized((manifold_dim+1, dim), dtype=REAL)
        local_vals = uninitialized((DoFMap.dofs_per_element, order**dim), dtype=REAL)

        eta = np.cos((2.0*np.arange(order, 0, -1, dtype=REAL)-1.0) / (2.0*order) * np.pi)
        xi = uninitialized((order, dim), dtype=REAL)
        x = uninitialized((qr.num_nodes, dim), dtype=REAL)
        fvals = uninitialized((qr.num_nodes), dtype=REAL)

        if comm:
            start = <INDEX_t>np.ceil(mesh.num_cells*comm.rank/comm.size)
            end = <INDEX_t>np.ceil(mesh.num_cells*(comm.rank+1)/comm.size)
        else:
            start = 0
            end = mesh.num_cells

        # loop over elements
        for i in range(start, end):
            mesh.getSimplex(i, simplex)
            vol = qr.getSimplexVolume(simplex)
            # get quadrature nodes
            qr.nodesInGlobalCoords(simplex, x)

            # loop over element dofs
            for k in range(DoFMap.dofs_per_element):
                I = DoFMap.cell2dof(i, k)
                if I >= 0:
                    # get Chebyshev nodes of box associated with DoF
                    for j in range(order):
                        for l in range(dim):
                            xi[j, l] = (boxes[I, l, 1]-boxes[I, l, 0])*0.5 * (eta[j]+1.0) + boxes[I, l, 0]
                    # loop over interpolating polynomial basis
                    r = 0
                    pit.reset()
                    while pit.step():
                        # evaluation of the pit.idx-Chebyshev polynomial
                        # at the quadrature nodes are saved in fvals
                        fvals[:] = 1.0
                        for q in range(dim):
                            l = pit.idx[q]
                            beta = 1.0
                            for j in range(order):
                                if j != l:
                                    beta *= xi[l, q]-xi[j, q]

                            # loop over quadrature nodes
                            for j in range(qr.num_nodes):
                                # evaluate l-th polynomial at j-th quadrature node
                                if abs(x[j, q]-xi[l, q]) > 1e-9:
                                    omega = 1.0
                                    for p in range(order):
                                        if p != l:
                                            omega *= x[j, q]-xi[p, q]
                                    fvals[j] *= omega/beta
                        # integrate chebyshev polynomial * local basis function over element
                        local_vals[k, r] = 0.0
                        for j in range(qr.num_nodes):
                            local_vals[k, r] += vol*fvals[j]*PHI[k, j]*qr.weights[j]
                        r += 1

            # enter data into vector coeff
            for k in range(DoFMap.dofs_per_element):
                I = DoFMap.cell2dof(i, k)
                if I >= 0:
                    for l in range(order**dim):
                        coeff[I, l] += local_vals[k, l]
        if comm is not None and comm.size > 1:
            if assembleOnRoot:
                if comm.rank == 0:
                    comm.Reduce(MPI.IN_PLACE, coeff, root=0)
                else:
                    comm.Reduce(coeff, coeff, root=0)
            else:
                comm.Allreduce(MPI.IN_PLACE, coeff)
        if comm is None or (assembleOnRoot and comm.rank == 0) or (not assembleOnRoot):
            tMB = transferMatrixBuilder(order, dim)
            transferOperator = uninitialized((order**dim, order**dim), dtype=REAL)
            # distribute entries of coeff to tree leaves
            for n in self.leaves():
                n.value = np.zeros((1, len(n.dofs), order**dim), dtype=REAL)
                if not local:
                    it.setIndexSet(n.dofs)
                else:
                    it.setIndexSet(n.local_dofs)
                k = 0
                while it.step():
                    dof = it.i

                    tMB.build(n.box, boxes[dof, :, :], transferOperator)
                    for i in range(order**dim):
                        for j in range(order**dim):
                            n.value[0, k, i] += transferOperator[i, j]*coeff[dof, j]
                    k += 1

    def enterLeafValuesGrad(self,
                            meshBase mesh,
                            DoFMap DoFMap,
                            INDEX_t order,
                            REAL_t[:, :, ::1] boxes,
                            comm=None):
        cdef:
            INDEX_t i, k, I, l, j, p, dim, dof, r, start, end
            REAL_t[:, ::1] simplex, local_vals, PHI, xi, x, box, gradients
            REAL_t[:, :, ::1] coeff
            REAL_t[::1] eta, fvals
            REAL_t vol, beta, omega
            tree_node n
            simplexQuadratureRule qr
            INDEX_t[:, ::1] cells = mesh.cells
            REAL_t[:, ::1] vertices = mesh.vertices
        dim = mesh.dim
        # Sauter Schwab p. 428
        if isinstance(DoFMap, P1_DoFMap):
            quadOrder = order+1
        else:
            raise NotImplementedError()
        qr = simplexXiaoGimbutas(quadOrder, dim)

        coeff = np.zeros((DoFMap.num_dofs, dim, order**dim), dtype=REAL)
        simplex = uninitialized((dim+1, dim), dtype=REAL)
        local_vals = uninitialized((DoFMap.dofs_per_element, order**dim), dtype=REAL)
        gradients = uninitialized((DoFMap.dofs_per_element, dim), dtype=REAL)

        eta = np.cos((2.0*np.arange(order, 0, -1, dtype=REAL)-1.0) / (2.0*order) * np.pi)
        xi = uninitialized((order, dim), dtype=REAL)
        x = uninitialized((qr.num_nodes, dim), dtype=REAL)
        fvals = uninitialized((qr.num_nodes), dtype=REAL)

        if comm:
            start = <INDEX_t>np.ceil(mesh.num_cells*comm.rank/comm.size)
            end = <INDEX_t>np.ceil(mesh.num_cells*(comm.rank+1)/comm.size)
        else:
            start = 0
            end = mesh.num_cells

        # loop over elements
        for i in range(start, end):
            mesh.getSimplex(i, simplex)
            vol = qr.getSimplexVolume(simplex)
            # get quadrature nodes
            qr.nodesInGlobalCoords(simplex, x)

            # loop over element dofs
            for k in range(DoFMap.dofs_per_element):
                I = DoFMap.cell2dof(i, k)
                if I >= 0:
                    # get box for dof
                    # TODO: avoid slicing
                    box = boxes[I, :, :]
                    # get Chebyshev nodes of box
                    for j in range(order):
                        for l in range(dim):
                            xi[j, l] = (box[l, 1]-box[l, 0])*0.5 * (eta[j]+1.0) + box[l, 0]
                    # loop over interpolating polynomial basis
                    r = 0
                    for idx in product(*([range(order)]*dim)):
                        # evaluation of the idx-Chebyshev polynomial
                        # at the quadrature nodes are saved in fvals
                        fvals[:] = 1.0
                        for q in range(dim):
                            l = idx[q]
                            beta = 1.0
                            for j in range(order):
                                if j != l:
                                    beta *= xi[l, q]-xi[j, q]

                            # loop over quadrature nodes
                            for j in range(qr.num_nodes):
                                # evaluate l-th polynomial at j-th quadrature node
                                if abs(x[j, q]-xi[l, q]) > 1e-9:
                                    omega = 1.0
                                    for p in range(order):
                                        if p != l:
                                            omega *= x[j, q]-xi[p, q]
                                    fvals[j] *= omega/beta
                        # integrate chebyshev polynomial * local basis function over element
                        #TODO: deal with multiple components, get gradient
                        local_vals[k, r] = 0.0
                        for j in range(qr.num_nodes):
                            local_vals[k, r] += vol*fvals[j]*qr.weights[j]
                        r += 1
            # get gradients
            if dim == 1:
                det = simplex[1, 0]-simplex[0, 0]
                gradients[0, 0] = 1./det
                gradients[1, 0] = -1./det
            elif dim == 2:
                det = (simplex[1, 1]-simplex[2, 1])*(simplex[0, 0]-simplex[2, 0])+(simplex[2, 0]-simplex[1, 0])*(simplex[0, 1]-simplex[2, 1])
                gradients[0, 0] = (simplex[1, 1]-simplex[2, 1])/det
                gradients[0, 1] = (simplex[2, 0]-simplex[1, 0])/det
                gradients[1, 0] = (simplex[2, 1]-simplex[0, 1])/det
                gradients[1, 1] = (simplex[0, 0]-simplex[2, 0])/det
                gradients[2, 0] = -gradients[0, 0]-gradients[1, 0]
                gradients[2, 1] = -gradients[0, 1]-gradients[1, 1]

            # enter data into vector coeff
            for k in range(DoFMap.dofs_per_element):
                I = DoFMap.cell2dof(i, k)
                if I >= 0:
                    for j in range(dim):
                        for l in range(order**dim):
                            coeff[I, j, l] += local_vals[k, l]*gradients[k, j]
        if comm:
            comm.Allreduce(MPI.IN_PLACE, coeff)
        # distribute entries of coeff to tree leaves
        for n in self.leaves():
            n.value = uninitialized((dim, len(n.dofs), order**dim), dtype=REAL)
            for k, dof in enumerate(sorted(n.dofs)):
                for j in range(dim):
                    for i in range(order**dim):
                        n.value[j, k, i] = coeff[dof, j, i]

    def set_id(self, INDEX_t maxID=0, INDEX_t distFromRoot=0):
        self.id = maxID
        self.distFromRoot = distFromRoot
        maxID += 1
        for c in self.children:
            maxID = c.set_id(maxID, distFromRoot+1)
        return maxID

    def get_max_id(self):
        cdef:
            INDEX_t id = self.id
            tree_node c
        for c in self.children:
            id = max(id, c.get_max_id())
        return id

    cdef BOOL_t trim(self, bitArray keep):
        cdef:
            tree_node c
            BOOL_t delNode, c_delNode
            list newChildren = []
            BOOL_t delAllChildren = True
        delNode = not keep.inSet(self.id)
        for c in self.children:
            c_delNode = c.trim(keep)
            if not c_delNode:
                delNode = False
                newChildren.append(c)
            delAllChildren &= c_delNode
        if not self.get_is_leaf() and len(newChildren) == 0:
            # self._cells = self.get_cells()
            self._dofs = self.get_dofs()
            self.children = []
            assert self.get_is_leaf()
            assert self.num_dofs == self.get_dofs().getNumEntries()
        elif delAllChildren:
            # None of the children are used, but some of their children are.
            # Remove the children, adopt their children.
            newChildren = []
            for c in self.children:
                for c2 in c.children:
                    newChildren.append(c2)
                    c2.parent = self
            self.children = newChildren
            assert self.num_dofs == self.get_dofs().getNumEntries()
        return delNode

    def HDF5write(self, node):
        myNode = node.create_group(str(self.id))
        if self.parent:
            myNode.attrs['parent'] = self.parent.id
        else:
            myNode.attrs['parent'] = -1
            node.attrs['numNodes'] = self.nodes
        myNode.create_dataset('children',
                              data=[c.id for c in self.children],
                              compression=COMPRESSION)
        for c in self.children:
            c.HDF5write(node)
        myNode.attrs['dim'] = self.dim
        myNode.create_dataset('_dofs',
                              data=list(self.dofs.toSet()),
                              compression=COMPRESSION)
        if self.get_is_leaf():
            myNode.create_dataset('_cells',
                                  data=list(self._cells),
                                  compression=COMPRESSION)

        try:
            myNode.create_dataset('transferOperator',
                                  data=np.array(self.transferOperator, copy=False),
                                  compression=COMPRESSION)
            node.attrs['M'] = self.transferOperator.shape[0]
        except:
            pass
        try:
            myNode.create_dataset('value',
                                  data=np.array(self.value, copy=False),
                                  compression=COMPRESSION)
        except:
            pass
        myNode.create_dataset('box', data=np.array(self.box, copy=False),
                              compression=COMPRESSION)

    @staticmethod
    def HDF5read(node):
        nodes = []
        boxes = uninitialized((0, 0, 0), dtype=REAL)
        try:
            M = node.attrs['M']
        except:
            M = 0
        for _ in range(node.attrs['numNodes']):
            n = tree_node(None, set(), boxes)
            nodes.append(n)
        for id in node:
            n = nodes[int(id)]
            myNode = node[id]
            n.dim = myNode.attrs['dim']
            dofs = arrayIndexSet()
            dofs.fromSet(set(myNode['_dofs']))
            n.dofs = dofs
            try:
                n._cells = set(myNode['_cells'])
                n.box = np.array(myNode['box'], dtype=REAL)
            except:
                pass
            try:
                n.transferOperator = np.array(myNode['transferOperator'],
                                              dtype=REAL)
            except:
                pass
            try:
                n.value = np.array(myNode['value'], dtype=REAL)
            except:
                pass
            n.coefficientsUp = uninitialized((M), dtype=REAL)
            n.coefficientsDown = uninitialized((M), dtype=REAL)
            n.id = int(id)
            nodes.append(n)
        for id in node:
            myNode = node[id]
            if myNode.attrs['parent'] >= 0:
                nodes[int(id)].parent = nodes[myNode.attrs['parent']]
            else:
                root = nodes[int(id)]
            for c in list(myNode['children']):
                nodes[int(id)].children.append(nodes[c])
        return root, nodes

    def HDF5writeNew(self, node):
        cdef:
            INDEX_t c = -1
            tree_node n
            indexSetIterator it = arrayIndexSetIterator()
            INDEX_t dim = self.box.shape[0], i, j
        numNodes = self.nodes
        indptrChildren = uninitialized((numNodes+1), dtype=INDEX)
        boxes = uninitialized((numNodes, dim, 2), dtype=REAL)
        for n in self.get_tree_nodes():
            indptrChildren[n.id+1] = len(n.children)
            for i in range(dim):
                for j in range(2):
                    boxes[n.id, i, j] = n.box[i, j]
        indptrChildren[0] = 0
        for i in range(1, numNodes+1):
            indptrChildren[i] += indptrChildren[i-1]
        nnz = indptrChildren[numNodes]
        indicesChildren = uninitialized((nnz), dtype=INDEX)
        for n in self.get_tree_nodes():
            k = indptrChildren[n.id]
            for cl in n.children:
                indicesChildren[k] = cl.id
                k += 1
        children = sparseGraph(indicesChildren, indptrChildren, numNodes, numNodes)
        node.create_group('children')
        children.HDF5write(node['children'])
        node.create_dataset('boxes', data=boxes, compression=COMPRESSION)
        del children

        M = self.children[0].transferOperator.shape[0]
        transferOperators = uninitialized((numNodes, M, M), dtype=REAL)
        for n in self.get_tree_nodes():
            try:
                if n.transferOperator.shape[0] > 0:
                    transferOperators[n.id, :, :] = n.transferOperator
                else:
                    transferOperators[n.id, :, :] = 0.0
            except:
                pass
        node.create_dataset('transferOperators', data=transferOperators,
                            compression=COMPRESSION)

        indptrDofs = uninitialized((numNodes+1), dtype=INDEX)
        for n in self.get_tree_nodes():
            indptrDofs[n.id+1] = len(n.dofs)
        indptrDofs[0] = 0
        for i in range(1, numNodes+1):
            indptrDofs[i] += indptrDofs[i-1]
        nnz = indptrDofs[numNodes]
        indicesDofs = uninitialized((nnz), dtype=INDEX)
        maxDof = -1
        for n in self.get_tree_nodes():
            k = indptrDofs[n.id]
            it.setIndexSet(n.dofs)
            while it.step():
                c = it.i
                indicesDofs[k] = c
                maxDof = max(maxDof, c)
                k += 1
        dofs = sparseGraph(indicesDofs, indptrDofs, numNodes, maxDof+1)
        node.create_group('dofs')
        dofs.HDF5write(node['dofs'])
        del dofs

        indptrCells = uninitialized((numNodes+1), dtype=INDEX)
        for n in self.get_tree_nodes():
            if n.get_is_leaf():
                indptrCells[n.id+1] = len(n.cells)
            else:
                indptrCells[n.id+1] = 0
        indptrCells[0] = 0
        for i in range(1, numNodes+1):
            indptrCells[i] += indptrCells[i-1]
        nnz = indptrCells[numNodes]
        indicesCells = uninitialized((nnz), dtype=INDEX)
        maxCell = -1
        for n in self.get_tree_nodes():
            if n.get_is_leaf():
                k = indptrCells[n.id]
                for c in n.cells:
                    indicesCells[k] = c
                    maxCell = max(maxCell, c)
                    k += 1
        cells = sparseGraph(indicesCells, indptrCells, numNodes, maxCell+1)
        node.create_group('cells')
        cells.HDF5write(node['cells'])
        del cells

        noCoefficients = next(self.leaves()).value.shape[0]
        values = uninitialized((maxDof+1, noCoefficients, M), dtype=REAL)
        mapping = {}
        k = 0
        for n in self.leaves():
            mapping[n.id] = k, k+n.value.shape[1]
            k += n.value.shape[1]
            values[mapping[n.id][0]:mapping[n.id][1], :, :] = np.swapaxes(n.value, 0, 1)
        node.create_group('mapping')
        keys = uninitialized((len(mapping)), dtype=INDEX)
        vals = uninitialized((len(mapping), 2), dtype=INDEX)
        k = 0
        for i in mapping:
            keys[k] = i
            vals[k][0] = mapping[i][0]
            vals[k][1] = mapping[i][1]
            k += 1
        node['mapping'].create_dataset('keys', data=keys, compression=COMPRESSION)
        node['mapping'].create_dataset('vals', data=vals, compression=COMPRESSION)
        node.create_dataset('values', data=values,
                            compression=COMPRESSION)

        node.attrs['dim'] = self.dim
        node.attrs['M'] = M

    @staticmethod
    def HDF5readNew(node):
        cdef:
            list nodes
            LinearOperator children
            REAL_t[:, :, ::1] boxes
            INDEX_t M
            tree_node n
            INDEX_t k
            dict mapping
            INDEX_t[::1] keys
            INDEX_t[:, ::1] vals
            indexSet cluster_dofs
            LinearOperator dofs
        children = LinearOperator.HDF5read(node['children'])
        nodes = [0]*children.shape[0]
        M = node.attrs['M']

        transferOperators = np.array(node['transferOperators'], dtype=REAL)
        boxes = np.array(node['boxes'], dtype=REAL)
        tree = readNode(nodes, 0, None, boxes, children, M, transferOperators)
        dofs = LinearOperator.HDF5read(node['dofs'])
        cells = LinearOperator.HDF5read(node['cells'])
        keys = np.array(node['mapping']['keys'], dtype=INDEX)
        vals = np.array(node['mapping']['vals'], dtype=INDEX)
        mapping = {}
        for k in range(keys.shape[0]):
            mapping[keys[k]] = k
        values = np.array(node['values'], dtype=REAL)
        for n in tree.leaves():
            n._dofs = arrayIndexSet(dofs.indices[dofs.indptr[n.id]:dofs.indptr[n.id+1]], sorted=True)
            n._cells = arrayIndexSet(cells.indices[cells.indptr[n.id]:cells.indptr[n.id+1]], sorted=True)
            n.value = np.ascontiguousarray(np.swapaxes(np.array(values[vals[mapping[n.id], 0]:vals[mapping[n.id], 1], :, :], dtype=REAL), 0, 1))
        # setDoFsFromChildren(tree)
        return tree, nodes

    cpdef INDEX_t findCell(self, meshBase mesh, REAL_t[::1] vertex, REAL_t[:, ::1] simplex, REAL_t[::1] bary):
        cdef:
            tree_node c
            INDEX_t cellNo = -1
        if minDist2FromBox(self.box, vertex) > 0.:
            return -1
        if self.get_is_leaf():
            for cellNo in self.cells:
                if mesh.vertexInCell(vertex, cellNo, simplex, bary):
                    return cellNo
                return -1
        else:
            for c in self.children:
                cellNo = c.findCell(mesh, vertex, simplex, bary)
                if cellNo >= 0:
                    break
            return cellNo

    cpdef set findCells(self, meshBase mesh, REAL_t[::1] vertex, REAL_t r, REAL_t[:, ::1] simplex):
        cdef:
            set cells = set()
            REAL_t h = mesh.h
            REAL_t rmin = r-h
            REAL_t rmax = r+h
            REAL_t dmin, dmax
            tree_node c
            INDEX_t cellNo
        dmin, dmax = distsFromBox(self.box, vertex)
        if (dmax <= rmin) or (dmin >= rmax):
            return cells
        if self.get_is_leaf():
            for cellNo in self._cells:
                mesh.getSimplex(cellNo, simplex)
                if distFromSimplex(simplex, vertex, r):
                    cells.add(cellNo)
        else:
            for c in self.children:
                cells |= c.findCells(mesh, vertex, r, simplex)
        return cells

    def __repr__(self):
        m = '['
        for i in range(self.box.shape[0]):
            if i == 0:
                m += '['
            else:
                m += ', ['
            for j in range(self.box.shape[1]):
                if j > 0:
                    m += ', '
                m += str(self.box[i, j])
            m += ']'
        m += ']'
        return 'node({}, nd={})'.format(m, self.num_dofs)

    cdef void upwardPassMatrix(self, dict coefficientsUp):
        cdef:
            INDEX_t k, i, m, j
            INDEX_t[::1] dof, dofs
            REAL_t[:, ::1] transfer, transfers
            tree_node c
        if self.id in coefficientsUp:
            return
        elif self.get_is_leaf():
            coefficientsUp[self.id] = (self.value[0, :, :], np.array(self.dofs.toArray()))
        else:
            transfers = np.zeros((self.num_dofs, self.coefficientsUp.shape[0]), dtype=REAL)
            dofs = uninitialized((self.num_dofs), dtype=INDEX)
            k = 0
            for c in self.children:
                if c.id not in coefficientsUp:
                    c.upwardPassMatrix(coefficientsUp)
                transfer, dof = coefficientsUp[c.id]
                for i in range(dof.shape[0]):
                    dofs[k] = dof[i]
                    for m in range(self.coefficientsUp.shape[0]):
                        for j in range(c.transferOperator.shape[1]):
                            transfers[k, m] += transfer[i, j]*c.transferOperator[m, j]
                    k += 1
            coefficientsUp[self.id] = (transfers, dofs)

    def findNodeForDoF(self, INDEX_t dof):
        cdef:
            tree_node c, c2
        if self.get_is_leaf():
            if self._dofs.inSet(dof):
                return self
            else:
                return None
        else:
            for c in self.children:
                c2 = c.findNodeForDoF(dof)
                if c2 is not None:
                    return c2

    def constructBasisMatrix(self, LinearOperator B, REAL_t[::1] coefficientsUp, INDEX_t offset=0):
        cdef:
            INDEX_t i, dof = -1, k = 0
            tree_node c
            indexSetIterator it
        if self.get_is_leaf():
            self.coefficientsUpOffset = offset
            self.coefficientsUp = coefficientsUp[offset:offset+self.coefficientsUp.shape[0]]
            it = self.get_dofs().getIter()
            while it.step():
                dof = it.i
                for i in range(self.coefficientsUp.shape[0]):
                    B.setEntry(offset+i, dof, self.value[0, k, i])
                k += 1
            offset += self.coefficientsUp.shape[0]
        else:
            for c in self.children:
                offset = c.constructBasisMatrix(B, coefficientsUp, offset)
        return offset

    def partition(self, DoFMap dm, MPI.Comm comm, REAL_t[:, :, ::1] boxes, BOOL_t canBeAssembled, BOOL_t mixed_node, dict params={}):
        from PyNucleus_fem.meshPartitioning import regularDofPartitioner, metisDofPartitioner

        cdef:
            indexSet subDofs
            INDEX_t num_dofs
            INDEX_t[::1] part = None

        assert self.parent is None
        assert self.id == 0

        if 'user-provided partition' in params:
            part = params['user-provided partition']
            assert part.shape[0] == dm.num_dofs, "User provided partitioning does not match number of degrees of freedom: {} != {}".format(part.shape[0], dm.num_dofs)
            assert np.min(part) == 0, "User provided partitioning leaves ranks empty."
            assert np.max(part) == comm.size-1, "User provided partitioning leaves ranks empty."
            assert np.unique(part).shape[0] == comm.size, "User provided partitioning leaves ranks empty."
        else:
            partitioner = params.get('partitioner', 'regular')
            if partitioner == 'regular':
                rVP = regularDofPartitioner(dm=dm)
                part, _ = rVP.partitionDofs(comm.size, irregular=True)
                del rVP
            elif partitioner == 'metis':
                dP = metisDofPartitioner(dm=dm, matrixPower=params.get('metis_matrixPower', 1))
                part, _ = dP.partitionDofs(comm.size, ufactor=params.get('metis_ufactor', 30))
                del dP
            else:
                raise NotImplementedError(partitioner)

        num_dofs = 0
        for p in range(comm.size):
            subDofs = arrayIndexSet(np.where(np.array(part, copy=False) == p)[0].astype(INDEX), sorted=True)
            num_dofs += subDofs.getNumEntries()
            self.children.append(tree_node(self, subDofs, boxes, canBeAssembled=canBeAssembled, mixed_node=mixed_node))
            self.children[p].id = p+1
        assert dm.num_dofs == num_dofs
        self._dofs = None

cdef tree_node readNode(list nodes, INDEX_t myId, parent, REAL_t[:, :, ::1] boxes, LinearOperator children, INDEX_t M, REAL_t[:, :, ::1] transferOperators):
    cdef:
        indexSet bA = arrayIndexSet()
        tree_node n = tree_node(parent, bA, boxes)
        INDEX_t i, j
    n.id = myId
    nodes[myId] = n
    n.transferOperator = uninitialized((transferOperators.shape[1],
                                        transferOperators.shape[2]),
                                       dtype=REAL)
    n.box = uninitialized((boxes.shape[1],
                           boxes.shape[2]),
                          dtype=REAL)
    for i in range(transferOperators.shape[1]):
        for j in range(transferOperators.shape[2]):
            n.transferOperator[i, j] = transferOperators[myId, i, j]
    for i in range(boxes.shape[1]):
        for j in range(boxes.shape[2]):
            n.box[i, j] = boxes[myId, i, j]
    n.coefficientsUp = uninitialized((M), dtype=REAL)
    n.coefficientsDown = uninitialized((M), dtype=REAL)
    for i in range(children.indptr[myId], children.indptr[myId+1]):
        n.children.append(readNode(nodes, children.indices[i], n, boxes, children, M, transferOperators))
    return n


cdef indexSet setDoFsFromChildren(tree_node n):
    if n.isLeaf:
        return n.dofs
    else:
        dofs = arrayIndexSet()
        for c in n.children:
            dofs.union(setDoFsFromChildren(c))
        n.dofs = dofs
        return dofs


cdef class transferMatrixBuilder:
    def __init__(self, INDEX_t m, INDEX_t dim):
        self.m = m
        self.dim = dim
        self.omega = uninitialized((m, dim), dtype=REAL)
        self.beta = uninitialized((m, dim), dtype=REAL)
        self.xiC = uninitialized((m, dim), dtype=REAL)
        self.xiP = uninitialized((m, dim), dtype=REAL)
        self.eta = np.cos((2.0*np.arange(m, 0, -1)-1.0) / (2.0*m) * np.pi)
        self.pit = productIterator(m, dim)
        self.pit2 = productIterator(m, dim)

    cdef void build(self,
                    REAL_t[:, ::1] boxP,
                    REAL_t[:, ::1] boxC,
                    REAL_t[:, ::1] T):
        cdef:
            INDEX_t dim = self.dim, i, j, l, k, I, J
            REAL_t[:, ::1] omega = self.omega
            REAL_t[:, ::1] beta = self.beta
            REAL_t[:, ::1] xiC = self.xiC
            REAL_t[:, ::1] xiP = self.xiP
            REAL_t[::1] eta = self.eta
            INDEX_t m = self.m

        for i in range(m):
            for j in range(dim):
                xiC[i, j] = (boxC[j, 1]-boxC[j, 0])*0.5*(eta[i]+1.0)+boxC[j, 0]
                xiP[i, j] = (boxP[j, 1]-boxP[j, 0])*0.5*(eta[i]+1.0)+boxP[j, 0]
        for j in range(m):
            for l in range(dim):
                omega[j, l] = xiC[j, l]-xiP[0, l]
                for k in range(1, m):
                    omega[j, l] *= xiC[j, l]-xiP[k, l]
                beta[j, l] = 1.0
                for k in range(m):
                    if k != j:
                        beta[j, l] *= xiP[j, l]-xiP[k, l]
        T[:, :] = 1.0
        I = 0
        self.pit.reset()
        while self.pit.step():
            J = 0
            self.pit2.reset()
            while self.pit2.step():
                for k in range(dim):
                    i = self.pit.idx[k]
                    j = self.pit2.idx[k]
                    if abs(xiP[i, k]-xiC[j, k]) > 1e-8:
                        T[I, J] *= omega[j, k]/(xiC[j, k]-xiP[i, k])/beta[i, k]
                J += 1
            I += 1


cdef class farFieldClusterPair:
    def __init__(self, tree_node n1, tree_node n2):
        self.n1 = n1
        self.n2 = n2

    def plot(self, color='blue'):
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

    cpdef void apply(farFieldClusterPair self, REAL_t[::1] x, REAL_t[::1] y):
        gemv(self.kernelInterpolant, x, y, 1.)

    def __repr__(self):
        return 'farFieldClusterPair<{}, {}>'.format(self.n1, self.n2)


cdef class productIterator:
    def __init__(self, INDEX_t m, INDEX_t dim):
        self.m = m
        self.dim = dim
        self.idx = np.zeros((dim), dtype=INDEX)

    cdef void reset(self):
        cdef:
            INDEX_t i
        for i in range(self.dim-1):
            self.idx[i] = 0
        self.idx[self.dim-1] = -1

    cdef BOOL_t step(self):
        cdef:
            INDEX_t i
        i = self.dim-1
        self.idx[i] += 1
        while self.idx[i] == self.m:
            self.idx[i] = 0
            if i>0:
                i -= 1
                self.idx[i] += 1
            else:
                return False
        return True


def assembleFarFieldInteractions(Kernel kernel, dict Pfar, INDEX_t m, DoFMap dm, BOOL_t bemMode=False):
    cdef:
        INDEX_t lvl
        REAL_t[:, ::1] box1, box2, x, y
        INDEX_t k, i, j, p
        REAL_t[::1] eta
        REAL_t eta_p
        farFieldClusterPair cP
        INDEX_t dim = dm.mesh.dim
        REAL_t[:, ::1] dofCoords = None
        INDEX_t dof1, dof2
        indexSetIterator it = arrayIndexSetIterator()
        INDEX_t kiSize = m**dim
        productIterator pit = productIterator(m, dim)
        BOOL_t kernel_variable = kernel.variable

    eta = np.cos((2.0*np.arange(m, 0, -1)-1.0) / (2.0*m) * np.pi)

    x = uninitialized((kiSize, dim))
    y = uninitialized((kiSize, dim))

    for lvl in Pfar:
        for cP in Pfar[lvl]:
            box1 = cP.n1.box
            box2 = cP.n2.box
            k = 0
            pit.reset()
            while pit.step():
                for j in range(dim):
                    p = pit.idx[j]
                    eta_p = eta[p]+1.0
                    x[k, j] = (box1[j, 1]-box1[j, 0])*0.5 * eta_p + box1[j, 0]
                    y[k, j] = (box2[j, 1]-box2[j, 0])*0.5 * eta_p + box2[j, 0]
                k += 1
            cP.kernelInterpolant = uninitialized((kiSize, kiSize), dtype=REAL)
            if not bemMode:
                for i in range(kiSize):
                    for j in range(kiSize):
                        if kernel_variable:
                            kernel.evalParamsPtr(dim, &x[i, 0], &y[j, 0])
                        cP.kernelInterpolant[i, j] = -2.0*kernel.evalPtr(dim, &x[i, 0], &y[j, 0])
            else:
                for i in range(kiSize):
                    for j in range(kiSize):
                        if kernel_variable:
                            kernel.evalParamsPtr(dim, &x[i, 0], &y[j, 0])
                        cP.kernelInterpolant[i, j] = kernel.evalPtr(dim, &x[i, 0], &y[j, 0])


cdef class H2Matrix(LinearOperator):
    def __init__(self,
                 tree_node tree,
                 dict Pfar,
                 LinearOperator Anear,
                 FakePLogger PLogger=None):
        self.tree = tree
        self.Pfar = Pfar
        self.Anear = Anear
        LinearOperator.__init__(self, Anear.shape[0], Anear.shape[1])
        if PLogger is not None:
            self.PLogger = PLogger
        else:
            self.PLogger = FakePLogger()
        self.skip_leaves_upward = False

    def constructBasis(self):
        num_leaves = len(list(self.tree.leaves()))
        md = self.tree.children[0].transferOperator.shape[0]
        B = IJOperator(num_leaves*md, self.shape[1])
        self.leafCoefficientsUp = uninitialized((num_leaves*md), dtype=REAL)
        self.tree.constructBasisMatrix(B, self.leafCoefficientsUp)
        self.basis = B.to_csr_linear_operator()
        self.skip_leaves_upward = True

    def isSparse(self):
        return False

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        cdef:
            INDEX_t level, componentNo
            tree_node n1, n2
            farFieldClusterPair clusterPair
        if self.Anear.nnz > 0:
            with self.PLogger.Timer("h2 matvec near"):
                self.Anear.matvec(x, y)
        else:
            y[:] = 0.
        if len(self.Pfar) > 0:
            if self.skip_leaves_upward:
                self.basis.matvec(x, self.leafCoefficientsUp)
            for componentNo in range(next(self.tree.leaves()).value.shape[0]):
                with self.PLogger.Timer("h2 upwardPass"):
                    self.tree.upwardPass(x, componentNo, skip_leaves=self.skip_leaves_upward)
                    self.tree.resetCoefficientsDown()
                with self.PLogger.Timer("h2 far field"):
                    for level in self.Pfar:
                        for clusterPair in self.Pfar[level]:
                            n1, n2 = clusterPair.n1, clusterPair.n2
                            clusterPair.apply(n2.coefficientsUp, n1.coefficientsDown)
                with self.PLogger.Timer("h2 downwardPass"):
                    self.tree.downwardPass(y, componentNo)
        return 0

    cdef INDEX_t matvec_no_overwrite(self,
                                     REAL_t[::1] x,
                                     REAL_t[::1] y) except -1:
        cdef:
            INDEX_t level, componentNo
            tree_node n1, n2
            farFieldClusterPair clusterPair
        self.Anear.matvec_no_overwrite(x, y)
        if len(self.Pfar) > 0:
            if self.skip_leaves_upward:
                self.basis.matvec(x, self.leafCoefficientsUp)
            for componentNo in range(next(self.tree.leaves()).value.shape[0]):
                self.tree.upwardPass(x, componentNo, skip_leaves=self.skip_leaves_upward)
                self.tree.resetCoefficientsDown()
                for level in self.Pfar:
                    for clusterPair in self.Pfar[level]:
                        n1, n2 = clusterPair.n1, clusterPair.n2
                        clusterPair.apply(n2.coefficientsUp, n1.coefficientsDown)
                self.tree.downwardPass(y, componentNo)
        return 0

    cdef INDEX_t matvec_submat(self,
                               REAL_t[::1] x,
                               REAL_t[::1] y,
                               list right_list,
                               tree_node left) except -1:
        cdef:
            INDEX_t level, componentNo
            tree_node n1, n2, right
            farFieldClusterPair clusterPair
        if self.Anear.nnz > 0:
            with self.PLogger.Timer("h2 matvec near"):
                self.Anear.matvec(x, y)
        else:
            y[:] = 0.
        if len(self.Pfar) > 0:
            if self.skip_leaves_upward:
                self.basis.matvec(x, self.leafCoefficientsUp)
            for componentNo in range(next(self.tree.leaves()).value.shape[0]):
                with self.PLogger.Timer("h2 upwardPass"):
                    for right in right_list:
                        right.upwardPass(x, componentNo, skip_leaves=self.skip_leaves_upward)
                    left.resetCoefficientsDown()
                with self.PLogger.Timer("h2 far field"):
                    for level in self.Pfar:
                        for clusterPair in self.Pfar[level]:
                            n1, n2 = clusterPair.n1, clusterPair.n2
                            clusterPair.apply(n2.coefficientsUp, n1.coefficientsDown)
                with self.PLogger.Timer("h2 downwardPass"):
                    left.downwardPass(y, componentNo)
        return 0


    property diagonal:
        def __get__(self):
            return self.Anear.diagonal

    property tree_size:
        def __get__(self):
            cdef:
                INDEX_t transfers_size = 0
                INDEX_t leaf_size = 0
                tree_node n
            # transfer operators
            for n in self.tree.get_tree_nodes():
                try:
                    transfers_size += n.transferOperator.shape[0]*n.transferOperator.shape[1]
                except AttributeError:
                    pass
            # leaf values
            for n in self.tree.leaves():
                try:
                    leaf_size += n.value.shape[0]*n.value.shape[1]*n.value.shape[2]
                except AttributeError:
                    pass
            return transfers_size + leaf_size

    property num_far_field_cluster_pairs:
        def __get__(self):
            clusters = 0
            for lvl in self.Pfar:
                clusters += len(self.Pfar[lvl])
            return clusters

    property cluster_size:
        def __get__(self):
            cdef:
                INDEX_t lvl
                farFieldClusterPair cP
                kernelInterpolantSize = 0
            for lvl in self.Pfar:
                for cP in self.Pfar[lvl]:
                    kernelInterpolantSize += cP.kernelInterpolant.shape[0]*cP.kernelInterpolant.shape[1]
            return kernelInterpolantSize

    property nearField_size:
        def __get__(self):
            if isinstance(self.Anear, Dense_LinearOperator):
                return self.Anear.num_rows*self.Anear.num_columns
            elif isinstance(self.Anear, Multiply_Linear_Operator):
                return self.Anear.A.nnz
            else:
                return self.Anear.nnz

    def __repr__(self):
        return '<%dx%d %s %f fill from near field, %f fill from tree, %f fill from clusters, %d tree nodes, %d far-field cluster pairs>' % (self.num_rows,
                                                                                                                                            self.num_columns,
                                                                                                                                            self.__class__.__name__,
                                                                                                                                            self.nearField_size/self.num_rows/self.num_columns,
                                                                                                                                            self.tree_size/self.num_rows/self.num_columns,
                                                                                                                                            self.cluster_size/self.num_rows/self.num_columns,
                                                                                                                                            self.tree.nodes,
                                                                                                                                            self.num_far_field_cluster_pairs)

    def getMemorySize(self):
        return self.Anear.getMemorySize() + self.cluster_size*sizeof(REAL_t) + self.tree_size*sizeof(REAL_t)

    def HDF5write(self, node, version=2, Pnear=None):
        cdef:
            INDEX_t K, S, j, lvl, d1, d2
            farFieldClusterPair clusterPair
            REAL_t[::1] kernelInterpolants
            INDEX_t[:, ::1] nodeIds
            REAL_t[:, ::1] sVals
        node.attrs['type'] = 'h2'

        node.create_group('Anear')
        self.Anear.HDF5write(node['Anear'])

        node.create_group('tree')
        if version == 2:
            self.tree.HDF5writeNew(node['tree'])
        elif version == 1:
            self.tree.HDF5write(node['tree'])
        else:
            raise NotImplementedError()
        node.attrs['version'] = version

        K = 0
        j = 0
        for lvl in self.Pfar:
            for clusterPair in self.Pfar[lvl]:
                K += clusterPair.kernelInterpolant.shape[0]*clusterPair.kernelInterpolant.shape[1]
                j += 1
        kernelInterpolants = uninitialized((K), dtype=REAL)
        nodeIds = uninitialized((j, 5), dtype=INDEX)
        K = 0
        j = 0
        for lvl in self.Pfar:
            for clusterPair in self.Pfar[lvl]:
                S = clusterPair.kernelInterpolant.shape[0]*clusterPair.kernelInterpolant.shape[1]
                for d1 in range(clusterPair.kernelInterpolant.shape[0]):
                    for d2 in range(clusterPair.kernelInterpolant.shape[1]):
                        kernelInterpolants[K] = clusterPair.kernelInterpolant[d1, d2]
                        K += 1
                nodeIds[j, 0] = clusterPair.n1.id
                nodeIds[j, 1] = clusterPair.n2.id
                nodeIds[j, 2] = clusterPair.kernelInterpolant.shape[0]
                nodeIds[j, 3] = clusterPair.kernelInterpolant.shape[1]
                nodeIds[j, 4] = lvl
                j += 1
        g = node.create_group('Pfar')
        g.create_dataset('kernelInterpolants', data=kernelInterpolants, compression=COMPRESSION)
        g.create_dataset('nodeIds', data=nodeIds, compression=COMPRESSION)

        if Pnear is not None:
            node2 = node.create_group('Pnear')
            k = 0
            for clusterPairNear in Pnear:
                node2.create_group(str(k))
                clusterPairNear.HDF5write(node2[str(k)])
                k += 1

    @staticmethod
    def HDF5read(node, returnPnear=False):
        cdef:
            dict Pfar
            INDEX_t lvl, K, j, d1, d2
            farFieldClusterPair cP
            INDEX_t[:, ::1] nodeIds
            REAL_t[:, ::1] sVals

        Anear = LinearOperator.HDF5read(node['Anear'])

        try:
            version = node.attrs['version']
        except:
            version = 1
        if version == 2:
            tree, nodes = tree_node.HDF5readNew(node['tree'])
        else:
            tree, nodes = tree_node.HDF5read(node['tree'])

        Pfar = {}
        nodeIds = np.array(node['Pfar']['nodeIds'], dtype=INDEX)
        kernelInterpolants = np.array(node['Pfar']['kernelInterpolants'], dtype=REAL)
        K = 0
        for j in range(nodeIds.shape[0]):
            lvl = nodeIds[j, 4]
            if lvl not in Pfar:
                Pfar[lvl] = []
            cP = farFieldClusterPair(nodes[nodeIds[j, 0]],
                                     nodes[nodeIds[j, 1]])
            d1 = nodeIds[j, 2]
            d2 = nodeIds[j, 3]
            cP.kernelInterpolant = uninitialized((d1, d2), dtype=REAL)
            for d1 in range(cP.kernelInterpolant.shape[0]):
                for d2 in range(cP.kernelInterpolant.shape[1]):
                    cP.kernelInterpolant[d1, d2] = kernelInterpolants[K]
                    K += 1
            Pfar[lvl].append(cP)

        if returnPnear:
            Pnear = []
            for k in node['Pnear']:
                Pnear.append(nearFieldClusterPair.HDF5read(node['Pnear'][k], nodes))
            return H2Matrix(tree, Pfar, Anear), Pnear
        else:
            return H2Matrix(tree, Pfar, Anear)

    def toarray(self):
        cdef:
            INDEX_t minLvl, maxLvl, lvl, i, j, I, J, id
            farFieldClusterPair cP
            tree_node n1, n2
            dict lvlNodes, delNodes
            dict coefficientsUp
            REAL_t[:, ::1] dense
            REAL_t[:, ::1] tr1, tr2, d
            INDEX_t[::1] dofs1, dofs2
        dense = self.Anear.toarray()
        if len(self.Pfar) > 0:
            minLvl = min(self.Pfar)
            maxLvl = max(self.Pfar)
        else:
            minLvl = 100
            maxLvl = 0
        lvlNodes = {lvl : set() for lvl in range(minLvl, maxLvl+1)}
        delNodes = {lvl : set() for lvl in range(minLvl, maxLvl+1)}
        for lvl in self.Pfar:
            for cP in self.Pfar[lvl]:
                lvlNodes[lvl].add(cP.n1.id)
                lvlNodes[lvl].add(cP.n2.id)
        for lvl in range(minLvl+1, maxLvl+1):
            lvlNodes[lvl] |= lvlNodes[lvl-1]
        lvlNodes[maxLvl] = set(list(range(self.tree.get_max_id()+1)))
        for lvl in range(minLvl, maxLvl):
            delNodes[lvl] = lvlNodes[lvl+1]-lvlNodes[lvl]
        del lvlNodes

        coefficientsUp = {}
        for lvl in reversed(sorted(self.Pfar.keys())):
            for cP in self.Pfar[lvl]:
                n1 = cP.n1
                n2 = cP.n2
                n1.upwardPassMatrix(coefficientsUp)
                n2.upwardPassMatrix(coefficientsUp)
                tr1, dofs1 = coefficientsUp[n1.id]
                tr2, dofs2 = coefficientsUp[n2.id]
                d = np.dot(tr1, np.dot(cP.kernelInterpolant, tr2.T))
                for i in range(dofs1.shape[0]):
                    I = dofs1[i]
                    for j in range(dofs2.shape[0]):
                        J = dofs2[j]
                        dense[I, J] = d[i, j]

            for id in delNodes[lvl]:
                if id in coefficientsUp:
                    del coefficientsUp[id]
        del coefficientsUp
        return np.array(dense, copy=False)

    def plot(self, Pnear=[], fill='box', nearFieldColor='red', farFieldColor='blue', kernelApproximationColor='yellow', shiftCoefficientColor='red', rankColor=None):
        import matplotlib.pyplot as plt
        if self.tree.dim == 1:
            if fill == 'box':
                for c in Pnear:
                    c.plot()
                for lvl in self.Pfar:
                    for c in self.Pfar[lvl]:
                        c.plot()
                plt.xlim([self.tree.box[0, 0], self.tree.box[0, 1]])
                plt.ylim([self.tree.box[0, 0], self.tree.box[0, 1]])
            elif fill == 'dof':
                import matplotlib.patches as patches
                nd = self.shape[0]
                spacing = 0.2
                for c in Pnear:
                    box1 = [min(c.n1.dofs)+spacing, max(c.n1.dofs)-spacing]
                    box2 = [nd-max(c.n2.dofs)+spacing, nd-min(c.n2.dofs)-spacing]
                    plt.gca().add_patch(patches.Rectangle((box1[0], box2[0]), box1[1]-box1[0], box2[1]-box2[0], fill=True, facecolor=nearFieldColor))
                for lvl in self.Pfar:
                    for c in self.Pfar[lvl]:
                        if farFieldColor is not None:
                            box1 = [min(c.n1.dofs)+spacing, max(c.n1.dofs)-spacing]
                            box2 = [nd-max(c.n2.dofs)+spacing, nd-min(c.n2.dofs)-spacing]
                            plt.gca().add_patch(patches.Rectangle((box1[0], box2[0]), box1[1]-box1[0], box2[1]-box2[0], fill=True, facecolor=farFieldColor))

                        k = c.kernelInterpolant.shape[0]

                        if shiftCoefficientColor is not None:
                            box1 = [min(c.n1.dofs), min(c.n1.dofs)+k-1]
                            box2 = [nd-min(c.n2.dofs), nd-max(c.n2.dofs)]
                            plt.gca().add_patch(patches.Rectangle((box1[0], box2[0]), box1[1]-box1[0], box2[1]-box2[0], fill=True, facecolor=shiftCoefficientColor))

                            box1 = [min(c.n1.dofs), max(c.n1.dofs)]
                            box2 = [nd-min(c.n2.dofs), nd-min(c.n2.dofs)-k+1]
                            plt.gca().add_patch(patches.Rectangle((box1[0], box2[0]), box1[1]-box1[0], box2[1]-box2[0], fill=True, facecolor=shiftCoefficientColor))

                        if kernelApproximationColor is not None:
                            box1 = [min(c.n1.dofs), min(c.n1.dofs)+k-1]
                            box2 = [nd-min(c.n2.dofs), nd-min(c.n2.dofs)-k+1]
                            plt.gca().add_patch(patches.Rectangle((box1[0], box2[0]), box1[1]-box1[0], box2[1]-box2[0], fill=True, facecolor=kernelApproximationColor))

                        if rankColor is not None:
                            plt.text(0.5*(min(c.n1.dofs)+max(c.n1.dofs)), nd-0.5*(min(c.n2.dofs)+max(c.n2.dofs)), str(k),
                                     color=rankColor,
                                     horizontalalignment='center',
                                     verticalalignment='center')

                plt.xlim([0, nd])
                plt.ylim([0, nd])
                plt.axis('equal')
            else:
                 raise NotImplementedError(fill)
        elif self.tree.dim == 2:
            Z = np.zeros((self.num_rows, self.num_columns), dtype=INDEX)
            for c in Pnear:
                for dof1 in c.n1.dofs:
                    for dof2 in c.n2.dofs:
                        Z[dof1, dof2] = 2
            for lvl in self.Pfar:
                for c in self.Pfar[lvl]:
                    for dof1 in c.n1.dofs:
                        for dof2 in c.n2.dofs:
                            Z[dof1, dof2] = 1
            plt.pcolormesh(Z)


cdef class DistributedH2Matrix_globalData(LinearOperator):
    """
    Distributed H2 matrix, operating on global vectors
    """
    def __init__(self, LinearOperator localMat, comm):
        self.localMat = localMat
        self.comm = comm
        super(DistributedH2Matrix_globalData, self).__init__(localMat.num_rows, localMat.num_columns)

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        self.comm.Bcast(x, root=0)
        self.localMat(x, y)
        self.comm.Allreduce(MPI.IN_PLACE, y)
        return 0

    def __repr__(self):
        return '<Rank %d/%d, %s>' % (self.comm.rank, self.comm.size, self.localMat)

    property diagonal:
        def __get__(self):
            d = self.localMat.Anear.diagonal
            self.comm.Allreduce(MPI.IN_PLACE, d)
            return d

    def getMemorySize(self):
        return self.localMat.getMemorySize()


cdef class DistributedLinearOperator(LinearOperator):
    """
    Distributed linear operator, operating on local vectors
    """
    def __init__(self, CSR_LinearOperator localMat, tree_node tree, list Pnear, MPI.Comm comm, DoFMap dm, DoFMap local_dm, CSR_LinearOperator lclR, CSR_LinearOperator lclP):
        cdef:
            tree_node n
        super(DistributedLinearOperator, self).__init__(local_dm.num_dofs, local_dm.num_dofs)
        self.localMat = localMat
        self.tree = tree
        self.Pnear = Pnear
        self.comm = comm
        self.dm = dm

        self.node_lookup = {}
        for n in self.tree.get_tree_nodes():
            self.node_lookup[n.id] = n

        self.lclRoot = self.tree.children[self.comm.rank]
        self.lcl_node_lookup = {}
        for n in self.lclRoot.get_tree_nodes():
            self.lcl_node_lookup[n.id] = n

        self.lcl_dm = local_dm
        self.lclR = lclR
        self.lclP = lclP

        self.setupNear()

    def __repr__(self):
        return '<Rank %d/%d, %s, %d local size>' % (self.comm.rank, self.comm.size, self.localMat, self.lcl_dm.num_dofs)

    def getMemorySize(self):
        return self.localMat.getMemorySize()

    cdef void setupNear(self):
        cdef:
            nearFieldClusterPair cP
            list remoteReceives_list
            INDEX_t commSize = self.comm.size, remoteRank
            INDEX_t local_dof, global_dof, k
            dict global_to_local
            INDEX_t[::1] indptr, indices
            REAL_t[::1] data
            INDEX_t[::1] new_indptr, new_indices
            REAL_t[::1] new_data
            INDEX_t jj, jjj, J
        remoteReceives_list = [set() for p in range(commSize)]
        for cP in self.Pnear:
            if cP.n1.id in self.lcl_node_lookup and cP.n2.id not in self.lcl_node_lookup:
                remoteRank = cP.n2.getParent(1).id-1
                remoteReceives_list[remoteRank] |= cP.n2.dofs.toSet()
            elif cP.n1.id not in self.lcl_node_lookup and cP.n2.id in self.lcl_node_lookup:
                remoteRank = cP.n1.getParent(1).id-1
                remoteReceives_list[remoteRank] |= cP.n1.dofs.toSet()

        counterReceives = np.array([len(remoteReceives_list[p]) for p in range(commSize)], dtype=INDEX)
        remoteReceives = np.concatenate([np.sort(list(remoteReceives_list[p])) for p in range(commSize)]).astype(INDEX)
        counterSends = np.zeros((commSize), dtype=INDEX)
        self.comm.Alltoall(counterReceives, counterSends)
        remoteSends = np.zeros(counterSends.sum(), dtype=INDEX)

        offsetsReceives = np.concatenate(([0], np.cumsum(counterReceives)[:commSize-1])).astype(INDEX)
        offsetsSends = np.concatenate(([0], np.cumsum(counterSends)[:commSize-1])).astype(INDEX)
        self.comm.Alltoallv([remoteReceives, (counterReceives, offsetsReceives)],
                            [remoteSends, (counterSends, offsetsSends)])

        self.near_offsetReceives = offsetsReceives
        self.near_offsetSends = offsetsSends
        self.near_remoteReceives = remoteReceives
        self.near_remoteSends = remoteSends
        self.near_counterReceives = counterReceives
        self.near_counterSends = counterSends
        self.near_dataReceives = uninitialized((np.sum(counterReceives)), dtype=REAL)
        self.near_dataSends = uninitialized((np.sum(counterSends)), dtype=REAL)

        global_to_local = {}
        for local_dof in range(self.lcl_dm.num_dofs):
            global_dof = self.lclR.indices[local_dof]
            global_to_local[global_dof] = local_dof
        for k in range(self.near_remoteSends.shape[0]):
            global_dof = self.near_remoteSends[k]
            local_dof = global_to_local[global_dof]
            self.near_remoteSends[k] = local_dof

        self.rowIdx = self.lclR.indices[:self.lcl_dm.num_dofs]
        self.colIdx = np.concatenate((self.rowIdx,
                                      self.near_remoteReceives))
        self.near_remoteReceives = np.arange(self.lcl_dm.num_dofs,
                                             self.lcl_dm.num_dofs+self.near_remoteReceives.shape[0],
                                             dtype=INDEX)

        for local_dof in range(self.lcl_dm.num_dofs,
                               self.lcl_dm.num_dofs+self.near_remoteReceives.shape[0]):
            global_dof = self.colIdx[local_dof]
            global_to_local[global_dof] = local_dof

        indptr = self.localMat.indptr
        indices = self.localMat.indices
        data = self.localMat.data
        new_indptr = uninitialized((self.rowIdx.shape[0]+1), dtype=INDEX)
        new_indices = uninitialized((self.localMat.nnz), dtype=INDEX)
        new_data = uninitialized((self.localMat.nnz), dtype=REAL)
        new_indptr[0] = 0
        for local_dof in range(self.rowIdx.shape[0]):
            global_dof = self.rowIdx[local_dof]
            new_indptr[local_dof+1] = new_indptr[local_dof] + indptr[global_dof+1]-indptr[global_dof]
            jjj = new_indptr[local_dof]
            for jj in range(indptr[global_dof], indptr[global_dof+1]):
                J = indices[jj]
                new_indices[jjj] = global_to_local[J]
                new_data[jjj] = data[jj]
                jjj += 1
        self.localMat.indptr = new_indptr
        self.localMat.indices = new_indices
        self.localMat.data = new_data
        self.localMat.num_rows = self.rowIdx.shape[0]
        self.localMat.num_columns = self.colIdx.shape[0]

    cdef void communicateNear(self, REAL_t[::1] src, REAL_t[::1] target):
        cdef:
            INDEX_t i
        # pack
        for i in range(self.near_remoteSends.shape[0]):
            self.near_dataSends[i] = src[self.near_remoteSends[i]]
        # comm
        self.comm.Alltoallv([self.near_dataSends, (self.near_counterSends, self.near_offsetSends)],
                            [self.near_dataReceives, (self.near_counterReceives, self.near_offsetReceives)])
        # unpack
        for i in range(self.near_remoteReceives.shape[0]):
            target[self.near_remoteReceives[i]] = self.near_dataReceives[i]

    cdef INDEX_t matvec(self, REAL_t[::1] x, REAL_t[::1] y) except -1:
        cdef:
            INDEX_t level
            tree_node n1, n2
            farFieldClusterPair clusterPair
            CSR_LinearOperator localMat = self.localMat
        # near field
        xTemp = uninitialized((self.localMat.shape[1]), dtype=REAL)
        xTemp[:self.localMat.shape[0]] = x
        self.communicateNear(x, xTemp)
        localMat(xTemp, y)

    cpdef tuple convert(self):
        cdef:
            dict gid_to_lid_rowmap
            INDEX_t[:, ::1] lcl_A_rowmap_a
            INDEX_t k, dof, p
            DistributedMap distRowMap
            INDEX_t[:, ::1] lcl_Anear_colmap_a
            DistributedMap distColMap
            dict clusterID_to_lid
            INDEX_t lid, clusterID, cluster_gid, cluster_lid
            tree_node n
            INDEX_t[::1] lcl_clusterGIDs
            INDEX_t[:, ::1] lcl_clusterMap, lcl_ghosted_clusterMap
            DistributedMap distClusterMap
            INDEX_t lcl_coeffmap_size, lcl_ghosted_coeffmap_size
            INDEX_t[:, ::1] lcl_coeffmap, lcl_ghosted_coeffmap
            INDEX_t[::1] lcl_blocksizes
            dict gid_to_clusterID
            dict gid_cluster_to_lid_coeff
            INDEX_t[::1] lid_cluster_to_gid_coeff, ghosted_lid_cluster_to_gid_coeff
            INDEX_t offset, gid_coeff, lid_coeff, lid_dof, dofNo
            IJOperator lcl_basisMatrix, lcl_transfer, lcl_transferBlockGraph, lcl_kernelApproximation, lcl_kernelBlockGraph
            INDEX_t blockSize
            INDEX_t i, k1, k2
            INDEX_t lid_coeff1, lid_coeff2
            INDEX_t cluster_gid1, cluster_gid2
            INDEX_t lid_cluster1, lid_cluster2
            farFieldClusterPair cP
            MPI.Comm comm = self.comm
            INDEX_t commSize = comm.size
            INDEX_t commRank = comm.rank
            INDEX_t minDistFromRoot
        # rowmap
        gid_to_lid_rowmap = {}
        lcl_A_rowmap_a = np.zeros((self.lclRoot.num_dofs, 2), dtype=INDEX)
        lcl_A_rowmap_a[:, 0] = self.rowIdx
        lcl_A_rowmap_a[:, 1] = commRank
        for dof in range(self.lclRoot.num_dofs):
            gid_to_lid_rowmap[self.rowIdx[dof]] = dof

        distRowMap = DistributedMap(comm, lcl_A_rowmap_a)

        # Anear colmap
        lcl_Anear_colmap_a = np.zeros((self.lclRoot.num_dofs+self.near_remoteReceives.shape[0], 2), dtype=INDEX)
        lcl_Anear_colmap_a[:, 0] = self.colIdx
        for k in range(self.lclRoot.num_dofs):
            lcl_Anear_colmap_a[k, 1] = commRank
        for p in range(commSize):
            for k in range(self.near_offsetReceives[p],
                           self.near_offsetReceives[p]+self.near_counterReceives[p]):
                lcl_Anear_colmap_a[self.lclRoot.num_dofs+k, 1] = p

        distColMap = DistributedMap(comm, lcl_Anear_colmap_a)

        # Anear
        lclAnear = self.localMat

        return (lclAnear,
                distRowMap,
                distColMap)

    property diagonal:
        def __get__(self):
            return self.localMat.Anear.diagonal


cdef class DistributedH2Matrix_localData(LinearOperator):
    """
    Distributed H2 matrix, operating on local vectors
    """
    def __init__(self, H2Matrix localMat, list Pnear, MPI.Comm comm, DoFMap dm, DoFMap local_dm, CSR_LinearOperator lclR, CSR_LinearOperator lclP):
        cdef:
            tree_node n
        super(DistributedH2Matrix_localData, self).__init__(local_dm.num_dofs, local_dm.num_dofs)
        assert isinstance(localMat.Anear, CSR_LinearOperator)
        self.localMat = localMat
        self.Pnear = Pnear
        self.comm = comm
        self.dm = dm

        self.node_lookup = {}
        for n in self.localMat.tree.get_tree_nodes():
            self.node_lookup[n.id] = n

        self.lclRoot = self.localMat.tree.children[self.comm.rank]
        self.lcl_node_lookup = {}
        for n in self.lclRoot.get_tree_nodes():
            self.lcl_node_lookup[n.id] = n

        self.lcl_dm = local_dm
        self.lclR = lclR
        self.lclP = lclP

        self.setupFar()
        self.setupNear()

    def __repr__(self):
        return '<Rank %d/%d, %s, %d local size>' % (self.comm.rank, self.comm.size, self.localMat, self.lcl_dm.num_dofs)

    cdef void setupNear(self):
        cdef:
            nearFieldClusterPair cP
            list remoteReceives_list
            INDEX_t commSize = self.comm.size, remoteRank
            INDEX_t local_dof, global_dof, k
            dict global_to_local
            INDEX_t[::1] indptr, indices
            REAL_t[::1] data
            INDEX_t[::1] new_indptr, new_indices
            REAL_t[::1] new_data
            INDEX_t jj, jjj, J
        remoteReceives_list = [set() for p in range(commSize)]
        for cP in self.Pnear:
            if cP.n1.id in self.lcl_node_lookup and cP.n2.id not in self.lcl_node_lookup:
                remoteRank = cP.n2.getParent(1).id-1
                remoteReceives_list[remoteRank] |= cP.n2.dofs.toSet()
            elif cP.n1.id not in self.lcl_node_lookup and cP.n2.id in self.lcl_node_lookup:
                remoteRank = cP.n1.getParent(1).id-1
                remoteReceives_list[remoteRank] |= cP.n1.dofs.toSet()

        counterReceives = np.array([len(remoteReceives_list[p]) for p in range(commSize)], dtype=INDEX)
        remoteReceives = np.concatenate([np.sort(list(remoteReceives_list[p])) for p in range(commSize)]).astype(INDEX)
        counterSends = np.zeros((commSize), dtype=INDEX)
        self.comm.Alltoall(counterReceives, counterSends)
        remoteSends = np.zeros(counterSends.sum(), dtype=INDEX)

        offsetsReceives = np.concatenate(([0], np.cumsum(counterReceives)[:commSize-1])).astype(INDEX)
        offsetsSends = np.concatenate(([0], np.cumsum(counterSends)[:commSize-1])).astype(INDEX)
        self.comm.Alltoallv([remoteReceives, (counterReceives, offsetsReceives)],
                            [remoteSends, (counterSends, offsetsSends)])

        self.near_offsetReceives = offsetsReceives
        self.near_offsetSends = offsetsSends
        self.near_remoteReceives = remoteReceives
        self.near_remoteSends = remoteSends
        self.near_counterReceives = counterReceives
        self.near_counterSends = counterSends
        self.near_dataReceives = uninitialized((np.sum(counterReceives)), dtype=REAL)
        self.near_dataSends = uninitialized((np.sum(counterSends)), dtype=REAL)

        global_to_local = {}
        for local_dof in range(self.lcl_dm.num_dofs):
            global_dof = self.lclR.indices[local_dof]
            global_to_local[global_dof] = local_dof
        for k in range(self.near_remoteSends.shape[0]):
            global_dof = self.near_remoteSends[k]
            local_dof = global_to_local[global_dof]
            self.near_remoteSends[k] = local_dof

        self.rowIdx = self.lclR.indices[:self.lcl_dm.num_dofs]
        self.colIdx = np.concatenate((self.rowIdx,
                                      self.near_remoteReceives))
        self.near_remoteReceives = np.arange(self.lcl_dm.num_dofs,
                                             self.lcl_dm.num_dofs+self.near_remoteReceives.shape[0],
                                             dtype=INDEX)

        for local_dof in range(self.lcl_dm.num_dofs,
                               self.lcl_dm.num_dofs+self.near_remoteReceives.shape[0]):
            global_dof = self.colIdx[local_dof]
            global_to_local[global_dof] = local_dof

        indptr = self.localMat.Anear.indptr
        indices = self.localMat.Anear.indices
        data = self.localMat.Anear.data
        new_indptr = uninitialized((self.rowIdx.shape[0]+1), dtype=INDEX)
        new_indices = uninitialized((self.localMat.Anear.nnz), dtype=INDEX)
        new_data = uninitialized((self.localMat.Anear.nnz), dtype=REAL)
        new_indptr[0] = 0
        for local_dof in range(self.rowIdx.shape[0]):
            global_dof = self.rowIdx[local_dof]
            new_indptr[local_dof+1] = new_indptr[local_dof] + indptr[global_dof+1]-indptr[global_dof]
            jjj = new_indptr[local_dof]
            for jj in range(indptr[global_dof], indptr[global_dof+1]):
                J = indices[jj]
                new_indices[jjj] = global_to_local[J]
                new_data[jjj] = data[jj]
                jjj += 1
        self.localMat.Anear.indptr = new_indptr
        self.localMat.Anear.indices = new_indices
        self.localMat.Anear.data = new_data
        self.localMat.Anear.num_rows = self.rowIdx.shape[0]
        self.localMat.Anear.num_columns = self.colIdx.shape[0]

    cdef void communicateNear(self, REAL_t[::1] src, REAL_t[::1] target):
        cdef:
            INDEX_t i
        # pack
        for i in range(self.near_remoteSends.shape[0]):
            self.near_dataSends[i] = src[self.near_remoteSends[i]]
        # comm
        self.comm.Alltoallv([self.near_dataSends, (self.near_counterSends, self.near_offsetSends)],
                            [self.near_dataReceives, (self.near_counterReceives, self.near_offsetReceives)])
        # unpack
        for i in range(self.near_remoteReceives.shape[0]):
            target[self.near_remoteReceives[i]] = self.near_dataReceives[i]

    cdef void setupFar(self):
        cdef:
            INDEX_t lvl, remoteRank, k, off1, off2, p, id, j
            INDEX_t commSize = self.comm.size
            farFieldClusterPair cP
            tree_node n
            list remoteReceives_list
            INDEX_t[::1] sublist
            INDEX_t[::1] counterReceives, remoteReceives, offsetsReceives
            INDEX_t[::1] counterSends, remoteSends, offsetsSends
            INDEX_t[::1] dataCounterReceives, dataOffsetReceives
            INDEX_t[::1] dataCounterSends, dataOffsetSends
        remoteReceives_list = [set() for p in range(commSize)]
        for lvl in self.localMat.Pfar:
            for cP in self.localMat.Pfar[lvl]:
                n = cP.n2.getParent(1)
                if self.lclRoot != n:
                    remoteRank = n.id-1
                    remoteReceives_list[remoteRank].add(cP.n2.id)
        counterReceives = uninitialized((commSize), dtype=INDEX)
        k = 0
        for p in range(commSize):
            counterReceives[p] = len(remoteReceives_list[p])
            k += counterReceives[p]
        remoteReceives = uninitialized((k), dtype=INDEX)
        k = 0
        for p in range(commSize):
            sublist = np.sort(list(remoteReceives_list[p])).astype(INDEX)
            for j in range(counterReceives[p]):
                remoteReceives[k+j] = sublist[j]
            k += counterReceives[p]
        assert np.min(remoteReceives) > 0, remoteReceives

        counterSends = np.zeros((commSize), dtype=INDEX)
        self.comm.Alltoall(counterReceives, counterSends)
        remoteSends = np.zeros(np.sum(counterSends), dtype=INDEX)

        offsetsReceives = np.concatenate(([0], np.cumsum(counterReceives)[:commSize-1])).astype(INDEX)
        offsetsSends = np.concatenate(([0], np.cumsum(counterSends)[:commSize-1])).astype(INDEX)
        self.comm.Alltoallv([remoteReceives, (counterReceives, offsetsReceives)],
                            [remoteSends, (counterSends, offsetsSends)])
        assert np.min(remoteSends) > 0, remoteSends

        dataCounterReceives = np.zeros((commSize), dtype=INDEX)
        k = 0
        for p in range(commSize):
            off1 = offsetsReceives[p]
            if p == commSize-1:
                off2 = remoteReceives.shape[0]
            else:
                off2 = offsetsReceives[p+1]
            for j in range(off1, off2):
                id = remoteReceives[j]
                n = self.node_lookup[id]
                dataCounterReceives[k] += n.coefficientsUp.shape[0]
            k += 1

        dataOffsetReceives = np.zeros((commSize), dtype=INDEX)
        k = 1
        for p in range(commSize-1):
            off1 = offsetsReceives[p]
            off2 = offsetsReceives[p+1]
            dataOffsetReceives[k] = dataOffsetReceives[k-1]
            for j in range(off1, off2):
                id = remoteReceives[j]
                n = self.node_lookup[id]
                dataOffsetReceives[k] += n.coefficientsUp.shape[0]
            k += 1

        dataCounterSends = np.zeros((commSize), dtype=INDEX)
        k = 0
        for p in range(commSize):
            off1 = offsetsSends[p]
            if p == commSize-1:
                off2 = remoteSends.shape[0]
            else:
                off2 = offsetsSends[p+1]
            for j in range(off1, off2):
                id = remoteSends[j]
                n = self.node_lookup[id]
                dataCounterSends[k] += n.coefficientsUp.shape[0]
            k += 1

        dataOffsetSends = np.zeros((commSize), dtype=INDEX)
        k = 1
        for p in range(commSize-1):
            off1 = offsetsSends[p]
            off2 = offsetsSends[p+1]
            dataOffsetSends[k] = dataOffsetSends[k-1]
            for j in range(off1, off2):
                id = remoteSends[j]
                n = self.node_lookup[id]
                dataOffsetSends[k] += n.coefficientsUp.shape[0]
            k += 1

        self.far_offsetsReceives = offsetsReceives
        self.far_offsetsSends = offsetsSends
        self.far_counterReceives = counterReceives
        self.far_counterSends = counterSends
        self.far_remoteReceives = remoteReceives
        self.far_remoteSends = remoteSends
        self.far_dataCounterReceives = dataCounterReceives
        self.far_dataCounterSends = dataCounterSends
        self.far_dataOffsetReceives = dataOffsetReceives
        self.far_dataOffsetSends = dataOffsetSends
        self.far_dataReceives = np.zeros((np.sum(dataCounterReceives)), dtype=REAL)
        self.far_dataSends = np.zeros((np.sum(dataCounterSends)), dtype=REAL)

    cdef void communicateFar(self):
        cdef:
            INDEX_t commSize = self.comm.size
            INDEX_t p, off1, off2, k, lclSize, j, i, id
            tree_node n
        # pack
        for p in range(commSize):
            off1 = self.far_offsetsSends[p]
            if p == commSize-1:
                off2 = self.far_remoteSends.shape[0]
            else:
                off2 = self.far_offsetsSends[p+1]
            k = self.far_dataOffsetSends[p]
            for i in range(off1, off2):
                id = self.far_remoteSends[i]
                n = self.lcl_node_lookup[id]
                lclSize = n.coefficientsUp.shape[0]
                for j in range(lclSize):
                    self.far_dataSends[k+j] = n.coefficientsUp[j]
                k += lclSize
        # comm
        self.comm.Alltoallv([self.far_dataSends, (self.far_dataCounterSends, self.far_dataOffsetSends)],
                            [self.far_dataReceives, (self.far_dataCounterReceives, self.far_dataOffsetReceives)])
        # unpack
        for p in range(commSize):
            off1 = self.far_offsetsReceives[p]
            if p == commSize-1:
                off2 = self.far_remoteReceives.shape[0]
            else:
                off2 = self.far_offsetsReceives[p+1]
            k = self.far_dataOffsetReceives[p]
            for i in range(off1, off2):
                id = self.far_remoteReceives[i]
                n = self.node_lookup[id]
                lclSize = n.coefficientsUp.shape[0]
                for j in range(lclSize):
                    n.coefficientsUp[j] = self.far_dataReceives[k+j]
                k += lclSize

    cdef INDEX_t matvec(self, REAL_t[::1] x, REAL_t[::1] y) except -1:
        cdef:
            INDEX_t level
            tree_node n1, n2
            farFieldClusterPair clusterPair
            H2Matrix localMat = self.localMat
        # near field
        xTemp = uninitialized((self.localMat.Anear.shape[1]), dtype=REAL)
        xTemp[:self.localMat.Anear.shape[0]] = x
        self.communicateNear(x, xTemp)
        localMat.Anear(xTemp, y)
        # far field
        self.lclRoot.upwardPass(x, 0, skip_leaves=False, local=True)
        self.communicateFar()
        self.lclRoot.resetCoefficientsDown()
        for level in localMat.Pfar:
            for clusterPair in localMat.Pfar[level]:
                n1, n2 = clusterPair.n1, clusterPair.n2
                clusterPair.apply(n2.coefficientsUp, n1.coefficientsDown)
        self.lclRoot.downwardPass(y, 0, local=True)

    cpdef tuple convert(self):
        cdef:
            dict gid_to_lid_rowmap
            INDEX_t[:, ::1] lcl_A_rowmap_a
            INDEX_t k, dof, p
            DistributedMap distRowMap
            INDEX_t[:, ::1] lcl_Anear_colmap_a
            DistributedMap distColMap
            dict clusterID_to_lid
            INDEX_t lid, clusterID, cluster_gid, cluster_lid
            tree_node n
            INDEX_t[::1] lcl_clusterGIDs
            INDEX_t[:, ::1] lcl_clusterMap, lcl_ghosted_clusterMap
            DistributedMap distClusterMap
            INDEX_t lcl_coeffmap_size, lcl_ghosted_coeffmap_size
            INDEX_t[:, ::1] lcl_coeffmap, lcl_ghosted_coeffmap
            INDEX_t[::1] lcl_blocksizes
            dict gid_to_clusterID
            dict gid_cluster_to_lid_coeff
            INDEX_t[::1] lid_cluster_to_gid_coeff, ghosted_lid_cluster_to_gid_coeff
            INDEX_t offset, gid_coeff, lid_coeff, lid_dof, dofNo
            IJOperator lcl_basisMatrix, lcl_transfer, lcl_transferBlockGraph, lcl_kernelApproximation, lcl_kernelBlockGraph
            INDEX_t blockSize
            INDEX_t i, k1, k2
            INDEX_t lid_coeff1, lid_coeff2
            INDEX_t cluster_gid1, cluster_gid2
            INDEX_t lid_cluster1, lid_cluster2
            farFieldClusterPair cP
            MPI.Comm comm = self.comm
            INDEX_t commSize = comm.size
            INDEX_t commRank = comm.rank
            INDEX_t minDistFromRoot
        # rowmap
        gid_to_lid_rowmap = {}
        lcl_A_rowmap_a = np.zeros((self.lclRoot.num_dofs, 2), dtype=INDEX)
        lcl_A_rowmap_a[:, 0] = self.rowIdx
        lcl_A_rowmap_a[:, 1] = commRank
        for dof in range(self.lclRoot.num_dofs):
            gid_to_lid_rowmap[self.rowIdx[dof]] = dof

        distRowMap = DistributedMap(comm, lcl_A_rowmap_a)

        # Anear colmap
        lcl_Anear_colmap_a = np.zeros((self.lclRoot.num_dofs+self.near_remoteReceives.shape[0], 2), dtype=INDEX)
        lcl_Anear_colmap_a[:, 0] = self.colIdx
        for k in range(self.lclRoot.num_dofs):
            lcl_Anear_colmap_a[k, 1] = commRank
        for p in range(commSize):
            for k in range(self.near_offsetReceives[p],
                           self.near_offsetReceives[p]+self.near_counterReceives[p]):
                lcl_Anear_colmap_a[self.lclRoot.num_dofs+k, 1] = p

        distColMap = DistributedMap(comm, lcl_Anear_colmap_a)

        # Anear
        lclAnear = self.localMat.Anear

        minDistFromRoot = comm.allreduce(min(self.localMat.Pfar.keys()), op=MPI.MIN)

        # clustermap
        clusterID_to_lid = {}
        lid = 0
        for n in self.lclRoot.get_tree_nodes():
            clusterID = n.id
            if clusterID not in clusterID_to_lid:
                clusterID_to_lid[clusterID] = lid
                lid += 1

        sizes = comm.gather(len(clusterID_to_lid))
        if commRank == 0:
            sizes = np.concatenate(([0], np.cumsum(sizes[:commSize-1])))
        offset = comm.scatter(sizes)
        clusterID_to_gid = {clusterID: offset+clusterID_to_lid[clusterID] for clusterID in clusterID_to_lid}
        lcl_clusterGIDs = np.arange(offset, offset+len(clusterID_to_lid), dtype=INDEX)
        lcl_clusterMap = np.zeros((lcl_clusterGIDs.shape[0], 2), dtype=INDEX)
        lcl_clusterMap[:, 0] = lcl_clusterGIDs
        lcl_clusterMap[:, 1] = commRank
        distClusterMap = DistributedMap(comm, lcl_clusterMap)

        # ghosted clustermap
        sendsClusterIDs = np.zeros((np.sum(self.far_counterSends)), dtype=INDEX)
        comm.Alltoallv([self.far_remoteReceives, (self.far_counterReceives, self.far_offsetsReceives)],
                       [sendsClusterIDs, (self.far_counterSends, self.far_offsetsSends)])
        receivesClusterGIDs = np.zeros_like(self.far_remoteReceives)
        sendsClusterGIDs = np.zeros_like(sendsClusterIDs)
        for k, clusterID in enumerate(sendsClusterIDs):
            sendsClusterGIDs[k] = clusterID_to_gid[clusterID]
        comm.Alltoallv([sendsClusterGIDs, (self.far_counterSends, self.far_offsetsSends)],
                       [receivesClusterGIDs, (self.far_counterReceives, self.far_offsetsReceives)])

        lcl_ghosted_clusterMap = np.zeros((lcl_clusterMap.shape[0]+receivesClusterGIDs.shape[0], 2), dtype=INDEX)
        lcl_ghosted_clusterMap[:lcl_clusterMap.shape[0], :] = lcl_clusterMap

        for k in range(self.far_remoteReceives.shape[0]):
            clusterID = self.far_remoteReceives[k]
            gid = receivesClusterGIDs[k]
            clusterID_to_gid[clusterID] = gid
            lcl_ghosted_clusterMap[lcl_clusterMap.shape[0]+k, 0] = gid
        for rank in range(commSize):
            lcl_ghosted_clusterMap[lcl_clusterMap.shape[0]+self.far_offsetsReceives[rank]:lcl_clusterMap.shape[0]+self.far_offsetsReceives[rank]++self.far_counterReceives[rank], 1] = rank

        distGhostedClusterMap = DistributedMap(comm, lcl_ghosted_clusterMap)
        gid_to_clusterID = {clusterID_to_gid[clusterID]: clusterID for clusterID in clusterID_to_gid}

        # coeffMap, blocksizes
        lcl_coeffmap_size = 0
        for cluster_lid in range(distClusterMap.getLocalNumElements()):
            cluster_gid = distClusterMap.getGlobalElement(cluster_lid)
            n = self.lcl_node_lookup[gid_to_clusterID[cluster_gid]]
            lcl_coeffmap_size += n.transferOperator.shape[0]
        sizes = comm.gather(lcl_coeffmap_size)
        if commRank == 0:
            sizes = np.concatenate(([0], np.cumsum(sizes[:commSize-1])))
        offset = comm.scatter(sizes)

        lcl_coeffmap = np.zeros((lcl_coeffmap_size, 2), dtype=INDEX)
        lcl_blocksizes = np.zeros((distClusterMap.getLocalNumElements()), dtype=INDEX)
        gid_cluster_to_lid_coeff = {}
        lid_cluster_to_gid_coeff = np.zeros((distClusterMap.getLocalNumElements()), dtype=INDEX)
        lid_coeff = 0
        for lid_cluster in range(distClusterMap.getLocalNumElements()):
            cluster_gid = distClusterMap.getGlobalElement(lid_cluster)
            n = self.lcl_node_lookup[gid_to_clusterID[cluster_gid]]
            blockSize = n.transferOperator.shape[0]
            for i in range(lid_coeff, lid_coeff+blockSize):
                lcl_coeffmap[i, 0] = offset+i
            lid_cluster_to_gid_coeff[lid_cluster] = offset+lid_coeff
            lcl_blocksizes[lid_cluster] = blockSize
            gid_cluster_to_lid_coeff[cluster_gid] = lid_coeff
            lid_coeff += blockSize
        lcl_coeffmap[:, 1] = commRank

        distCoeffMap = DistributedMap(comm, lcl_coeffmap)

        # ghosted coeffMap
        ghosted_lid_cluster_to_gid_coeff = Import(distGhostedClusterMap, distClusterMap)(lid_cluster_to_gid_coeff)

        lcl_ghosted_coeffmap_size = 0
        for cluster_lid in range(distGhostedClusterMap.getLocalNumElements()):
            cluster_gid = distGhostedClusterMap.getGlobalElement(cluster_lid)
            n = self.node_lookup[gid_to_clusterID[cluster_gid]]
            lcl_ghosted_coeffmap_size += n.transferOperator.shape[0]

        lcl_ghosted_coeffmap = np.zeros((lcl_ghosted_coeffmap_size, 2), dtype=INDEX)
        lid_coeff = 0
        for lid_cluster in range(distGhostedClusterMap.getLocalNumElements()):
            cluster_gid = distGhostedClusterMap.getGlobalElement(lid_cluster)
            n = self.node_lookup[gid_to_clusterID[cluster_gid]]
            blockSize = n.transferOperator.shape[0]
            gid_coeff = ghosted_lid_cluster_to_gid_coeff[lid_cluster]
            for i in range(blockSize):
                lcl_ghosted_coeffmap[lid_coeff+i, 0] = gid_coeff+i
            lid_coeff += blockSize

        distGhostedCoeffMap = DistributedMap(comm, lcl_ghosted_coeffmap)

        gid_lid_clusterMap = {}
        for lid in range(lcl_clusterMap.shape[0]):
            gid_lid_clusterMap[lcl_clusterMap[lid, 0]] = lid
        gid_lid_ghosted_clusterMap = {}
        for lid in range(lcl_ghosted_clusterMap.shape[0]):
            gid_lid_ghosted_clusterMap[lcl_ghosted_clusterMap[lid, 0]] = lid

        lcl_basisMatrix = IJOperator(distRowMap.getLocalNumElements(),
                                     distCoeffMap.getLocalNumElements())
        for n in self.lclRoot.get_tree_nodes():

            if n.isLeaf:
                for dofNo, dof in enumerate(n.dofs):
                    lid_dof = gid_to_lid_rowmap[dof]
                    gid_cluster = clusterID_to_gid[n.id]
                    lid_coeff = gid_cluster_to_lid_coeff[gid_cluster]
                    for coeffNo in range(n.value.shape[2]):
                        lcl_basisMatrix.setEntry(lid_dof,
                                                 lid_coeff,
                                                 n.value[0, dofNo, coeffNo])
                        lid_coeff += 1
        lcl_basisMatrix_csr = lcl_basisMatrix.to_csr_linear_operator()

        lcl_transfers = {}
        lcl_transfersBlockGraph = {}
        for n in self.lclRoot.get_tree_nodes():
            try:
                n.transferOperator
                dist = n.distFromRoot
                if dist-2 < minDistFromRoot:
                    continue
                if dist not in lcl_transfersBlockGraph:
                    lcl_transfers[dist] = IJOperator(distCoeffMap.getLocalNumElements(),
                                                     distCoeffMap.getLocalNumElements())
                    lcl_transfersBlockGraph[dist] = IJOperator(distClusterMap.getLocalNumElements(),
                                                               distClusterMap.getLocalNumElements())
                lcl_transfer = lcl_transfers[dist]
                cluster_gid1 = clusterID_to_gid[n.parent.id]
                cluster_gid2 = clusterID_to_gid[n.id]
                lid_coeff1 = gid_cluster_to_lid_coeff[cluster_gid1]
                for k1 in range(n.transferOperator.shape[0]):
                    lid_coeff2 = gid_cluster_to_lid_coeff[cluster_gid2]
                    for k2 in range(n.transferOperator.shape[1]):
                        lcl_transfer.setEntry(lid_coeff1, lid_coeff2, n.transferOperator[k1, k2])
                        lid_coeff2 += 1
                    lid_coeff1 += 1
                lcl_transferBlockGraph = lcl_transfersBlockGraph[dist]
                lcl_transferBlockGraph.setEntry(gid_lid_clusterMap[cluster_gid1],
                                                gid_lid_clusterMap[cluster_gid2],
                                                1.)
            except (AttributeError, KeyError):
                pass
        for dist in lcl_transfersBlockGraph:
            lcl_transfers[dist] = lcl_transfers[dist].to_csr_linear_operator()
            lcl_transfersBlockGraph[dist] = lcl_transfersBlockGraph[dist].to_csr_linear_operator()

        lcl_kernelBlockGraph = IJOperator(distClusterMap.getLocalNumElements(),
                                          distGhostedClusterMap.getLocalNumElements())
        lcl_kernelApproximation = IJOperator(distCoeffMap.getLocalNumElements(),
                                             distGhostedCoeffMap.getLocalNumElements())
        for lvl in self.localMat.Pfar:
            for cP in self.localMat.Pfar[lvl]:
                gid_cluster1 = clusterID_to_gid[cP.n1.id]
                gid_cluster2 = clusterID_to_gid[cP.n2.id]
                lid_cluster1 = gid_lid_clusterMap[gid_cluster1]
                lid_cluster2 = gid_lid_ghosted_clusterMap[gid_cluster2]
                lcl_kernelBlockGraph.setEntry(lid_cluster1,
                                              lid_cluster2,
                                              1.)
                lid_coeff1 = gid_cluster_to_lid_coeff[gid_cluster1]
                for k1 in range(cP.kernelInterpolant.shape[0]):
                    lid_coeff2 = distGhostedCoeffMap.getLocalElement(ghosted_lid_cluster_to_gid_coeff[lid_cluster2])
                    for k2 in range(cP.kernelInterpolant.shape[1]):
                        lcl_kernelApproximation.setEntry(lid_coeff1, lid_coeff2, cP.kernelInterpolant[k1, k2])
                        lid_coeff2 += 1
                    lid_coeff1 += 1
        lcl_kernelApproximation_csr = lcl_kernelApproximation.to_csr_linear_operator()
        lcl_kernelBlockGraph_csr = lcl_kernelBlockGraph.to_csr_linear_operator()

        return (lclAnear,
                distRowMap,
                distColMap,
                distCoeffMap, distGhostedCoeffMap,
                distClusterMap, distGhostedClusterMap,
                lcl_blocksizes,
                lcl_kernelApproximation_csr, lcl_kernelBlockGraph_csr,
                lcl_transfers, lcl_transfersBlockGraph,
                lcl_basisMatrix_csr)

    property diagonal:
        def __get__(self):
            return self.localMat.Anear.diagonal



def getDoFBoxesAndCells(meshBase mesh, DoFMap DoFMap, comm=None):
    cdef:
        INDEX_t i, j, I, k, start, end, dim = mesh.dim, manifold_dim = mesh.manifold_dim
        REAL_t[:, :, ::1] boxes = uninitialized((DoFMap.num_dofs, dim, 2), dtype=REAL)
        REAL_t[:, ::1] boxes2
        REAL_t[:, ::1] simplex = uninitialized((manifold_dim+1, dim), dtype=REAL)
        REAL_t[::1] m = uninitialized((dim), dtype=REAL), M = uninitialized((dim), dtype=REAL)
        sparsityPattern cells_pre = sparsityPattern(DoFMap.num_dofs)

    boxes[:, :, 0] = np.inf
    boxes[:, :, 1] = -np.inf

    if comm:
        start = <INDEX_t>np.ceil(mesh.num_cells*comm.rank/comm.size)
        end = <INDEX_t>np.ceil(mesh.num_cells*(comm.rank+1)/comm.size)
    else:
        start = 0
        end = mesh.num_cells
    for i in range(start, end):
        mesh.getSimplex(i, simplex)

        for k in range(dim):
            m[k] = simplex[0, k]
            M[k] = simplex[0, k]
        for j in range(manifold_dim):
            for k in range(dim):
                m[k] = min(m[k], simplex[j+1, k])
                M[k] = max(M[k], simplex[j+1, k])
        for j in range(DoFMap.dofs_per_element):
            I = DoFMap.cell2dof(i, j)
            if I >= 0:
                for k in range(dim):
                    boxes[I, k, 0] = min(boxes[I, k, 0], m[k])
                    boxes[I, k, 1] = max(boxes[I, k, 1], M[k])
    if comm:
        boxes2 = uninitialized((DoFMap.num_dofs, dim), dtype=REAL)
        for i in range(DoFMap.num_dofs):
            for j in range(dim):
                boxes2[i, j] = boxes[i, j, 0]
        comm.Allreduce(MPI.IN_PLACE, boxes2, op=MPI.MIN)
        for i in range(DoFMap.num_dofs):
            for j in range(dim):
                boxes[i, j, 0] = boxes2[i, j]
                boxes2[i, j] = boxes[i, j, 1]
        comm.Allreduce(MPI.IN_PLACE, boxes2, op=MPI.MAX)
        for i in range(DoFMap.num_dofs):
            for j in range(dim):
                boxes[i, j, 1] = boxes2[i, j]
    for i in range(mesh.num_cells):
        for j in range(DoFMap.dofs_per_element):
            I = DoFMap.cell2dof(i, j)
            if I >= 0:
                cells_pre.add(I, i)
    indptr, indices = cells_pre.freeze()
    cells = sparseGraph(indices, indptr, DoFMap.num_dofs, mesh.num_cells)
    return np.array(boxes, copy=False), cells


def getFractionalOrders(variableFractionalOrder s, meshBase mesh):
    cdef:
        REAL_t[:, ::1] centers
        INDEX_t numCells = mesh.num_cells
        INDEX_t cellNo1, cellNo2
        REAL_t[:, ::1] orders = uninitialized((numCells, numCells), dtype=REAL)

    centers = mesh.getCellCenters()

    if s.symmetric:
        for cellNo1 in range(numCells):
            for cellNo2 in range(cellNo1, numCells):
                orders[cellNo1, cellNo2] = s.eval(centers[cellNo1, :],
                                                  centers[cellNo2, :])
                orders[cellNo2, cellNo1] = orders[cellNo1, cellNo2]
    else:
        for cellNo1 in range(numCells):
            for cellNo2 in range(numCells):
                orders[cellNo1, cellNo2] = s.eval(centers[cellNo1, :],
                                                  centers[cellNo2, :])
    return orders



cpdef BOOL_t getAdmissibleClusters(Kernel kernel,
                                   tree_node n1,
                                   tree_node n2,
                                   refinementParams refParams,
                                   dict Pfar=None,
                                   list Pnear=None,
                                   INDEX_t level=0,
                                   REAL_t[:, :, ::1] boxes1=None,
                                   REAL_t[:, ::1] coords1=None,
                                   REAL_t[:, :, ::1] boxes2=None,
                                   REAL_t[:, ::1] coords2=None):
    cdef:
        tree_node t1, t2
        bint seemsAdmissible
        REAL_t dist, diam1, diam2, maxDist
        function horizon
        farFieldClusterPair cp
        BOOL_t addedFarFieldClusters = False
        INDEX_t lenNearField
        REAL_t[:, ::1] boxUnion
        REAL_t diamUnion = 0.
        REAL_t horizonValue
    dist = distBoxes(n1.box, n2.box)
    diam1 = diamBox(n1.box)
    diam2 = diamBox(n2.box)

    seemsAdmissible = refParams.eta*dist >= max(diam1, diam2) and not n1.mixed_node and not n2.mixed_node and (refParams.farFieldInteractionSize <= n1.get_num_dofs()*n2.get_num_dofs()) and n1.canBeAssembled and n2.canBeAssembled

    horizon = kernel.horizon
    assert isinstance(horizon, constant)
    horizonValue = horizon.value

    if kernel.finiteHorizon:
        maxDist = maxDistBoxes(n1.box, n2.box)
        if not kernel.complement:
            if  dist > horizonValue:
                # return True, since we don't want fully ignored cluster pairs to be merged into near field ones.
                return True
        else:
            if maxDist <= horizonValue:
                # same
                return True
        if dist <= horizonValue and horizonValue <= maxDist:
            seemsAdmissible = False
        boxUnion = uninitialized((n1.box.shape[0], 2), dtype=REAL)
        merge_boxes(n1.box, n2.box, boxUnion)
        diamUnion = diamBox(boxUnion)
    lenNearField = len(Pnear)
    if seemsAdmissible:
        cp = farFieldClusterPair(n1, n2)
        try:
            Pfar[level].append(cp)
        except KeyError:
            Pfar[level] = [cp]
        return True
    else:
        if refParams.attemptRefinement:
            if n1.get_is_leaf():
                n1.refine(boxes1, coords1, refParams, False)
            if n2.get_is_leaf():
                n2.refine(boxes2, coords2, refParams, False)
        if (n1.get_is_leaf() and n2.get_is_leaf()) or (level == refParams.maxLevels):
            Pnear.append(nearFieldClusterPair(n1, n2))
            # if kernel.finiteHorizon and len(Pnear[len(Pnear)-1].cellsInter) > 0:
            #     if diamUnion > kernel.horizon.value:
            #         print("Near field cluster pairs need to fit within horizon.\nBox1 {}\nBox2 {}\n{}, {} -> {}".format(np.array(n1.box),
            #                                                                                                             np.array(n2.box),
            #                                                                                                             diam1,
            #                                                                                                             diam2,
            #                                                                                                             diamUnion))
            return False
        elif (refParams.farFieldInteractionSize > (<np.int64_t>n1.get_num_dofs())*(<np.int64_t>n2.get_num_dofs())) and (diamUnion < horizonValue):
            Pnear.append(nearFieldClusterPair(n1, n2))
            return False
        elif n1.get_is_leaf():
            for t2 in n2.children:
                addedFarFieldClusters |= getAdmissibleClusters(kernel, n1, t2, refParams,
                                                               Pfar, Pnear,
                                                               level+1,
                                                               boxes1, coords1,
                                                               boxes2, coords2)
        elif n2.get_is_leaf():
            for t1 in n1.children:
                addedFarFieldClusters |= getAdmissibleClusters(kernel, t1, n2, refParams,
                                                               Pfar, Pnear,
                                                               level+1,
                                                               boxes1, coords1,
                                                               boxes2, coords2)
        else:
            for t1 in n1.children:
                for t2 in n2.children:
                    addedFarFieldClusters |= getAdmissibleClusters(kernel, t1, t2, refParams,
                                                                   Pfar, Pnear,
                                                                   level+1,
                                                                   boxes1, coords1,
                                                                   boxes2, coords2)
    if not addedFarFieldClusters:
        if diamUnion < horizonValue:
            del Pnear[lenNearField:]
            Pnear.append(nearFieldClusterPair(n1, n2))
    return addedFarFieldClusters


cpdef BOOL_t getCoveringClusters(Kernel kernel,
                                 tree_node n1,
                                 tree_node n2,
                                 refinementParams refParams,
                                 list Pnear=None,
                                 INDEX_t level=0,
                                 REAL_t[:, :, ::1] boxes1=None,
                                 REAL_t[:, ::1] coords1=None,
                                 REAL_t[:, :, ::1] boxes2=None,
                                 REAL_t[:, ::1] coords2=None):
    cdef:
        tree_node t1, t2
        REAL_t dist, maxDist
        function horizon
        BOOL_t addedFarFieldClusters = False
        REAL_t horizonValue
    dist = distBoxes(n1.box, n2.box)
    maxDist = maxDistBoxes(n1.box, n2.box)

    horizon = kernel.horizon
    assert isinstance(horizon, constant)
    horizonValue = horizon.value

    if (dist > horizonValue) or (n1.get_num_dofs() == 0) or (n2.get_num_dofs() == 0):
        return True
    elif maxDist <= horizonValue:
        Pnear.append(nearFieldClusterPair(n1, n2))
        return True
    else:
        if refParams.attemptRefinement:
            if n1.get_is_leaf():
                n1.refine(boxes1, coords1, refParams, False)
            if n2.get_is_leaf():
                n2.refine(boxes2, coords2, refParams, False)
        if (n1.get_is_leaf() and n2.get_is_leaf()) or (level == refParams.maxLevels):
            Pnear.append(nearFieldClusterPair(n1, n2))
            return True
        elif n1.get_is_leaf():
            for t2 in n2.children:
                addedFarFieldClusters |= getCoveringClusters(kernel, n1, t2, refParams,
                                                             Pnear,
                                                             level+1,
                                                             boxes1, coords1,
                                                             boxes2, coords2)
        elif n2.get_is_leaf():
            for t1 in n1.children:
                addedFarFieldClusters |= getCoveringClusters(kernel, t1, n2, refParams,
                                                             Pnear,
                                                             level+1,
                                                             boxes1, coords1,
                                                             boxes2, coords2)
        else:
            for t1 in n1.children:
                for t2 in n2.children:
                    addedFarFieldClusters |= getCoveringClusters(kernel, t1, t2, refParams,
                                                                 Pnear,
                                                                 level+1,
                                                                 boxes1, coords1,
                                                                 boxes2, coords2)

    # if not addedFarFieldClusters:
    #     if diamUnion < horizonValue:
    #         del Pnear[lenNearField:]
    #         Pnear.append(nearFieldClusterPair(n1, n2))
    # return addedFarFieldClusters
    return False


def symmetrizeNearFieldClusters(list Pnear):
    cdef:
        set clusters = set()
        nearFieldClusterPair cpNear
        INDEX_t id1, id2
        dict lookup = {}
    for cpNear in Pnear:
        id1 = cpNear.n1.id
        id2 = cpNear.n2.id
        clusters.add((id1, id2))
        lookup[id1] = cpNear.n1
        lookup[id2] = cpNear.n2
    while len(clusters) > 0:
        id1, id2 = clusters.pop()
        if id1 != id2:
            if (id2, id1) not in clusters:
                Pnear.append(nearFieldClusterPair(lookup[id2], lookup[id1]))
            else:
                clusters.remove((id2, id1))


def trimTree(tree_node tree, list Pnear, dict Pfar, comm, keep=[]):
    cdef:
        nearFieldClusterPair cpNear
        farFieldClusterPair cpFar
        tree_node n
        bitArray used = bitArray(maxElement=tree.get_max_id())
    for n in keep:
        used.set(n.id)
    for cpNear in Pnear:
        used.set(cpNear.n1.id)
        used.set(cpNear.n2.id)
    for lvl in Pfar:
        for cpFar in Pfar[lvl]:
            used.set(cpFar.n1.id)
            used.set(cpFar.n2.id)
    # keep the nodes associated with MPI ranks
    if (comm is not None) and len(tree.children) == comm.size:
        for n in tree.children:
            used.set(n.id)
    for n in tree.get_tree_nodes_up_to_level(tree.irregularLevelsOffset):
        used.set(n.id)
    # print('before', used.getNumEntries(), tree.get_max_id(), tree.nodes)
    tree.trim(used)
    # tree.set_id()
    # used.empty()
    # for cpNear in Pnear:
    #     used.set(cpNear.n1.id)
    #     used.set(cpNear.n2.id)
    # for lvl in Pfar:
    #     for cpFar in Pfar[lvl]:
    #         used.set(cpFar.n1.id)
    #         used.set(cpFar.n2.id)
    # print('after', used.getNumEntries(), tree.get_max_id(), tree.nodes)


cdef void insertion_sort(REAL_t[::1] a, INDEX_t start, INDEX_t end):
    cdef:
        INDEX_t i, j
        REAL_t v
    for i in range(start, end):
        v = a[i]
        j = i-1
        while j >= start:
            if a[j] <= v:
                break
            a[j+1] = a[j]
            j -= 1
        a[j+1] = v


cpdef INDEX_t[:, ::1] checkNearFarFields(DoFMap dm, list Pnear, dict Pfar):
    cdef:
        INDEX_t[:, ::1] S = np.zeros((dm.num_dofs, dm.num_dofs), dtype=INDEX)
        nearFieldClusterPair cNear
        farFieldClusterPair cFar
        INDEX_t i, j

    for cNear in Pnear:
        for i in cNear.n1.dofs:
            for j in cNear.n2.dofs:
                S[i, j] = 1
    for lvl in Pfar:
        for cFar in Pfar[lvl]:
            for i in cFar.n1.dofs:
                for j in cFar.n2.dofs:
                    S[i, j] = 2
    return S


cdef class exactSphericalIntegral2D(function):
    cdef:
        REAL_t[::1] u
        P1_DoFMap dm
        REAL_t radius
        REAL_t[:, ::1] simplex
        REAL_t[::1] u_local, w, z, bary
        public tree_node root
        INDEX_t numThetas
        REAL_t[::1] thetas

    def __init__(self, REAL_t[::1] u, P1_DoFMap dm, REAL_t radius):
        cdef:
            meshBase mesh = dm.mesh
            REAL_t[:, :, ::1] boxes = None
            list cells = []
            REAL_t[:, ::1] centers = None
            INDEX_t i, j, maxLevels, dof, I
        self.u = u
        self.dm = dm
        assert self.u.shape[0] == self.dm.num_dofs
        assert mesh.dim == 2
        self.radius = radius
        self.simplex = uninitialized((3, 2))
        self.u_local = uninitialized(3)
        boxes, cells = getDoFBoxesAndCells(mesh, dm, None)
        centers = uninitialized((dm.num_dofs, mesh.dim), dtype=REAL)
        for i in range(dm.num_dofs):
            for j in range(mesh.dim):
                centers[i, j] = 0.5*(boxes[i, j, 0]+boxes[i, j, 1])
        root = tree_node(None, set(np.arange(dm.num_dofs)), boxes)
        maxLevels = int(np.floor(0.5*np.log2(mesh.num_vertices)))
        root.refine(boxes, centers, maxLevels=maxLevels)
        # root.set_id()
        # enter cells in leaf nodes
        for n in root.leaves():
            n._cells = set()
            for dof in n.dofs:
                n._cells |= cells[dof]
        # update boxes (if we stopped at maxLevels, before each DoF has
        # only it's support as box)
        for n in root.leaves():
            for I in n.dofs:
                box = n.box
                for i in range(mesh.dim):
                    for j in range(2):
                        boxes[I, i, j] = box[i, j]
        self.root = root
        self.w = uninitialized(2)
        self.z = uninitialized(2)
        self.bary = uninitialized(3)
        self.thetas = uninitialized(6)

    cdef void findThetas(self, INDEX_t cellNo, REAL_t[::1] x):
        cdef:
            INDEX_t i
            REAL_t p, q, t, theta
        self.numThetas = 0
        for i in range(3):
            for j in range(2):
                self.w[j] = self.simplex[(i+1) % 3, j] - self.simplex[i, j]
                self.z[j] = self.simplex[i, j]-x[j]
            t = 1./mydot(self.w, self.w)
            p = 2*mydot(self.w, self.z)*t
            q = (mydot(self.z, self.z)-self.radius**2)*t
            q = 0.25*p**2-q
            if q > 0:
                q = sqrt(q)
                t = -0.5*p+q
                if (0 <= t) and (t <= 1):
                    theta = atan2(t*self.w[1]+self.z[1], t*self.w[0]+self.z[0])
                    if theta < 0:
                        theta += 2*pi
                    self.thetas[self.numThetas] = theta
                    self.numThetas += 1
                t = -0.5*p-q
                if (0 <= t) and (t <= 1):
                    theta = atan2(t*self.w[1]+self.z[1], t*self.w[0]+self.z[0])
                    if theta < 0:
                        theta += 2*pi
                    self.thetas[self.numThetas] = theta
                    self.numThetas += 1
        insertion_sort(self.thetas, 0, self.numThetas)

        theta = 0.5*(self.thetas[0]+self.thetas[1])
        self.w[0] = x[0]+self.radius*cos(theta)
        self.w[1] = x[1]+self.radius*sin(theta)
        if not self.dm.mesh.vertexInCell(self.w, cellNo, self.simplex, self.bary):
            theta = self.thetas[0]
            for i in range(self.numThetas-1):
                self.thetas[i] = self.thetas[i+1]
            self.thetas[self.numThetas-1] = theta+2*pi
        if self.numThetas == 1:
            self.numThetas = 0

    cdef REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t I = 0.
            meshBase mesh = self.dm.mesh
            P1_DoFMap dm = self.dm
            INDEX_t i, j, cellNo
            REAL_t vol, ax, ay, b, theta0, theta1
        for cellNo in self.root.findCells(mesh, x, self.radius, self.simplex):
            mesh.getSimplex(cellNo, self.simplex)

            self.findThetas(cellNo, x)
            if self.numThetas == 0:
                continue
            assert self.numThetas % 2 == 0, (np.array(self.thetas), self.numThetas)

            for i in range(dm.dofs_per_element):
                dof = dm.cell2dof(cellNo, i)
                if dof >= 0:
                    self.u_local[i] = self.u[dof]
                else:
                    self.u_local[i] = 0
            vol = ((self.simplex[0, 0]-self.simplex[1, 0])*(self.simplex[2, 1]-self.simplex[1, 1]) -
                   (self.simplex[0, 1]-self.simplex[1, 1])*(self.simplex[2, 0]-self.simplex[1, 0]))
            ax = 0
            ay = 0
            b = 0
            for i in range(dm.dofs_per_element):
                ax += self.u_local[i]*(self.simplex[(i+2) % 3, 1]-self.simplex[(i+1) % 3, 1])
                ay -= self.u_local[i]*(self.simplex[(i+2) % 3, 0]-self.simplex[(i+1) % 3, 0])
                b -= self.u_local[i]*(self.simplex[(i+1) % 3, 0]*(self.simplex[(i+2) % 3, 1]-self.simplex[(i+1) % 3, 1]) -
                                      self.simplex[(i+1) % 3, 1]*(self.simplex[(i+2) % 3, 0]-self.simplex[(i+1) % 3, 0]))
            ax /= vol
            ay /= vol
            b /= vol

            j = 0
            while j < self.numThetas:
                theta0, theta1 = self.thetas[j], self.thetas[j+1]
                j += 2
                if theta1-theta0 > theta0 + 2*pi-theta1:
                    theta0 += 2*pi
                    theta0, theta1 = theta1, theta0
                # print(theta0, theta1, j, cellNo)
                assert theta0 <= theta1, (theta0, theta1)

                I += self.radius**2 * (ax*(sin(theta1)-sin(theta0)) - ay*(cos(theta1)-cos(theta0))) + (b*self.radius + self.radius*ax*x[0] + self.radius*ay*x[1])*(theta1-theta0)
        return I
