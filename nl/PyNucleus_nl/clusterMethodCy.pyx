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
cimport cython
from PyNucleus_base.linear_operators cimport (LinearOperator,
                                               sparseGraph,
                                               Multiply_Linear_Operator)
from PyNucleus_base.blas cimport gemv, gemvT, mydot, matmat, norm, assign
from . nonlocalLaplacianBase cimport variableFractionalOrder
from . nonlocalLaplacian cimport nearFieldClusterPair
from PyNucleus_fem.DoFMaps cimport DoFMap, P1_DoFMap, P2_DoFMap
from PyNucleus_fem.meshCy cimport meshBase
from PyNucleus_fem.functions cimport constant
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI

COMPRESSION = 'gzip'


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void merge_boxes(REAL_t[:, ::1] box1,
                             REAL_t[:, ::1] box2,
                             REAL_t[:, ::1] new_box):
    cdef INDEX_t i
    for i in range(box1.shape[0]):
        new_box[i, 0] = min(box1[i, 0], box2[i, 0])
        new_box[i, 1] = max(box1[i, 1], box2[i, 1])


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void merge_boxes2(REAL_t[:, ::1] box1,
                              REAL_t[:, :, ::1] box2,
                              INDEX_t dof,
                              REAL_t[:, ::1] new_box):
    cdef INDEX_t i
    for i in range(box1.shape[0]):
        new_box[i, 0] = min(box1[i, 0], box2[dof, i, 0])
        new_box[i, 1] = max(box1[i, 1], box2[dof, i, 1])


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint inBox(const REAL_t[:, ::1] box,
                       const REAL_t[::1] vector):
    cdef:
        bint t = True
        INDEX_t i
    for i in range(box.shape[0]):
        t = t and (box[i, 0] <= vector[i]) and (vector[i] < box[i, 1])
    return t


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
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


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
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


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
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


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
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

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
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
        if self.dim > 0:
            self.box = uninitialized((self.dim, 2), dtype=REAL)
            self.box[:, 0] = np.inf
            self.box[:, 1] = -np.inf
            it = self.get_dofs().getIter()
            while it.step():
                dof = it.i
                merge_boxes2(self.box, boxes, dof, self.box)

    cdef indexSet get_dofs(self):
        cdef:
            indexSet dofs
            tree_node c
        if self.isLeaf:
            return self._dofs
        else:
            dofs = arrayIndexSet()
            for c in self.children:
                dofs = dofs.union(c.get_dofs())
            return dofs
        # return self.dofs

    @property
    def dofs(self):
        return self.get_dofs()

    @property
    def num_dofs(self):
        if self._num_dofs < 0:
            self._num_dofs = self.get_dofs().getNumEntries()
        return self._num_dofs

    cdef indexSet get_cells(self):
        cdef:
            indexSet s
            tree_node c
        if self.isLeaf:
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
        if self.isLeaf:
            return 1
        else:
            return 1+sum([c.nodes for c in self.children])

    nodes = property(fget=get_nodes)

    def refine(self,
               REAL_t[:, :, ::1] boxes,
               REAL_t[:, ::1] centers,
               INDEX_t maxLevels=200,
               INDEX_t maxLevelsMixed=200,
               INDEX_t level=0,
               INDEX_t minSize=1,
               INDEX_t minMixedSize=1):
        cdef:
            indexSet dofs = self.get_dofs()
            INDEX_t num_initial_dofs = dofs.getNumEntries(), dim, i = -1, j, num_dofs
            REAL_t[:, ::1] subbox
            indexSet s
            indexSetIterator it = dofs.getIter()
            set sPre
            INDEX_t nD = 0
        if not self.mixed_node:
            if (level >= maxLevels) or (num_initial_dofs <= minSize):
                return
        else:
            if (level >= maxLevelsMixed) or (num_initial_dofs <= minMixedSize):
                return
        dim = self.box.shape[0]
        if (not self.mixed_node) or dim == 1:
            for idx in product(*([[0, 1]]*dim)):
                subbox = uninitialized((dim, 2), dtype=REAL)
                for i, j in enumerate(idx):
                    subbox[i, 0] = self.box[i, 0] + j*(self.box[i, 1]-self.box[i, 0])/2
                    subbox[i, 1] = self.box[i, 0] + (j+1)*(self.box[i, 1]-self.box[i, 0])/2
                sPre = set()
                it.reset()
                while it.step():
                    i = it.i
                    if inBox(subbox, centers[i, :]):
                        sPre.add(i)
                s = arrayIndexSet()
                s.fromSet(sPre)
                num_dofs = s.getNumEntries()
                if num_dofs > 0 and num_dofs < num_initial_dofs:
                    nD += num_dofs
                    self.children.append(tree_node(self, s, boxes, mixed_node=self.mixed_node))
                    self.children[-1].refine(boxes, centers, maxLevels, maxLevelsMixed, level+1, minSize, minMixedSize)
        else:
            # split along larger box dimension
            for j in range(2):
                subbox = uninitialized((dim, 2), dtype=REAL)
                if self.box[0, 1]-self.box[0, 0] > self.box[1, 1]-self.box[1, 0]:
                    subbox[0, 0] = self.box[0, 0] + j*(self.box[0, 1]-self.box[0, 0])/2
                    subbox[0, 1] = self.box[0, 0] + (j+1)*(self.box[0, 1]-self.box[0, 0])/2
                    subbox[1, 0] = self.box[1, 0]
                    subbox[1, 1] = self.box[1, 1]
                else:
                    subbox[0, 0] = self.box[0, 0]
                    subbox[0, 1] = self.box[0, 1]
                    subbox[1, 0] = self.box[1, 0] + j*(self.box[1, 1]-self.box[1, 0])/2
                    subbox[1, 1] = self.box[1, 0] + (j+1)*(self.box[1, 1]-self.box[1, 0])/2
                sPre = set()
                it.reset()
                while it.step():
                    i = it.i
                    if inBox(subbox, centers[i, :]):
                        sPre.add(i)
                s = arrayIndexSet()
                s.fromSet(sPre)
                num_dofs = s.getNumEntries()
                if num_dofs > 0 and num_dofs < num_initial_dofs:
                    nD += num_dofs
                    self.children.append(tree_node(self, s, boxes, mixed_node=self.mixed_node))
                    self.children[-1].refine(boxes, centers, maxLevels, maxLevelsMixed, level+1, minSize, minMixedSize)

        assert nD == 0 or nD == num_initial_dofs
        if nD == num_initial_dofs:
            self._dofs = None
        else:
            assert self.isLeaf

    def get_is_leaf(self):
        return len(self.children) == 0

    isLeaf = property(fget=get_is_leaf)

    def leaves(self):
        cdef:
            tree_node i, j
        if self.isLeaf:
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

    def _getLevels(self):
        if self.isLeaf:
            return 1
        else:
            return 1+max([c._getLevels() for c in self.children])

    numLevels = property(fget=_getLevels)

    def plot(self, level=0, plotDoFs=False, REAL_t[:, ::1] dofCoords=None):
        import matplotlib.pyplot as plt

        cdef:
            indexSet dofs
            indexSetIterator it
            INDEX_t dof, k, j
            REAL_t[:, ::1] points
            INDEX_t[::1] idx
            REAL_t[::1] x, y

        if plotDoFs:
            from scipy.spatial import ConvexHull
            if self.dim == 2:
                assert dofCoords is not None
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
                plt.plot(x, y, color='red' if self.mixed_node else 'blue')
            else:
                raise NotImplementedError()
        else:
            import matplotlib.patches as patches
            if self.dim == 2:
                plt.gca().add_patch(patches.Rectangle((self.box[0, 0], self.box[1, 0]),
                                                      self.box[0, 1]-self.box[0, 0],
                                                      self.box[1, 1]-self.box[1, 0],
                                                      fill=False,
                                                      color='red' if self.mixed_node else 'blue'))
                if not self.isLeaf:
                    myCenter = np.mean(self.box, axis=1)
                    for c in self.children:
                        cCenter = np.mean(c.box, axis=1)
                        plt.arrow(myCenter[0], myCenter[1], cCenter[0]-myCenter[0], cCenter[1]-myCenter[1])
                        plt.text(cCenter[0], cCenter[1], s=str(level+1))
                        c.plot(level+1)
            else:
                raise NotImplementedError()

    def prepareTransferOperators(self, INDEX_t m):
        cdef:
            tree_node c
        if not self.isLeaf:
            for c in self.children:
                c.prepareTransferOperators(m)
        if self.parent is not None:
            self.transferOperator = uninitialized((m**self.dim, m**self.dim),
                                                  dtype=REAL)
            transferMatrix(self.parent.box, self.box, m,
                           self.transferOperator)
        self.coefficientsUp = uninitialized((m**self.dim), dtype=REAL)
        self.coefficientsDown = uninitialized((m**self.dim), dtype=REAL)

    def upwardPass(self, REAL_t[::1] x, INDEX_t componentNo=0):
        cdef:
            INDEX_t i, dof = -1, k = 0
            tree_node c
            indexSetIterator it
        if self.isLeaf:
            self.coefficientsUp[:] = 0.0
            it = self.get_dofs().getIter()
            while it.step():
                dof = it.i
                for i in range(self.coefficientsUp.shape[0]):
                    self.coefficientsUp[i] += x[dof]*self.value[componentNo, k, i]
                k += 1
        else:
            self.coefficientsUp[:] = 0.0
            for c in self.children:
                c.upwardPass(x, componentNo)
                gemv(c.transferOperator, c.coefficientsUp, self.coefficientsUp, 1.)

    def resetCoefficientsDown(self):
        cdef:
            tree_node c
        self.coefficientsDown[:] = 0.0
        if not self.isLeaf:
            for c in self.children:
                c.resetCoefficientsDown()

    def downwardPass(self, REAL_t[::1] y, INDEX_t componentNo=0):
        cdef:
            INDEX_t i, dof = -1, k = 0
            REAL_t val
            tree_node c
            indexSetIterator it
        if self.isLeaf:
            it = self.get_dofs().getIter()
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
                c.downwardPass(y, componentNo)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def enterLeaveValues(self,
                         meshBase mesh,
                         DoFMap DoFMap,
                         INDEX_t order,
                         REAL_t[:, :, ::1] boxes,
                         comm=None):
        cdef:
            INDEX_t i, k, I, l, j, p, dim, dof = -1, r, start, end
            REAL_t[:, ::1] coeff, simplex, local_vals, PHI, xi, x, box
            REAL_t[::1] eta, fvals
            REAL_t vol, beta, omega
            tree_node n
            simplexQuadratureRule qr
            indexSetIterator it = arrayIndexSetIterator()
        dim = mesh.dim
        # Sauter Schwab p. 428
        if isinstance(DoFMap, P1_DoFMap):
            quadOrder = order+2
        elif isinstance(DoFMap, P2_DoFMap):
            quadOrder = order+3
        else:
            raise NotImplementedError()
        qr = simplexXiaoGimbutas(quadOrder, dim)

        # get values of basis function in quadrature nodes
        PHI = uninitialized((DoFMap.dofs_per_element, qr.num_nodes), dtype=REAL)
        for i in range(DoFMap.dofs_per_element):
            for j in range(qr.num_nodes):
                PHI[i, j] = DoFMap.localShapeFunctions[i](qr.nodes[:, j])

        coeff = np.zeros((DoFMap.num_dofs, order**dim), dtype=REAL)
        simplex = uninitialized((dim+1, dim), dtype=REAL)
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
                    # get box for dof
                    # TODO: avoid slicing
                    box = boxes[I, :, :]
                    # get Chebyshev nodes of box
                    for j in range(order):
                        for l in range(dim):
                            xi[j, l] = (box[l, 1]-box[l, 0])*0.5 * (eta[j]+1.0) + box[l, 0]
                    # loop over interpolating ploynomial basis
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
        if comm and comm.size > 1:
            if comm.rank == 0:
                comm.Reduce(MPI.IN_PLACE, coeff, root=0)
            else:
                comm.Reduce(coeff, coeff, root=0)
        if comm is None or comm.rank == 0:
            # distribute entries of coeff to tree leaves
            for n in self.leaves():
                n.value = uninitialized((1, len(n.dofs), order**dim), dtype=REAL)
                it.setIndexSet(n.dofs)
                k = 0
                while it.step():
                    dof = it.i
                    for i in range(order**dim):
                        n.value[0, k, i] = coeff[dof, i]
                    k += 1

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def enterLeaveValuesGrad(self,
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
                    # loop over interpolating ploynomial basis
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

    cdef tree_node get_node(self, INDEX_t id):
        cdef:
            tree_node c
            INDEX_t k
        if self.id == id:
            return self
        else:
            for k in range(len(self.children)-1):
                if self.children[k].id <= id < self.children[k+1].id:
                    c = self.children[k]
                    return c.get_node(id)
            if self.children[len(self.children)-1].id <= id:
                c = self.children[len(self.children)-1]
                return c.get_node(id)

    cdef BOOL_t trim(self, bitArray keep):
        cdef:
            tree_node c
            BOOL_t delNode, c_delNode
            list newChildren = []
        delNode = not keep.inSet(self.id)
        for c in self.children:
            c_delNode = c.trim(keep)
            if not c_delNode:
                delNode = False
                newChildren.append(c)
        if not self.isLeaf and len(newChildren) == 0:
            self._cells = arrayIndexSet()
            for c in self.children:
                self._cells = self._cells.union(c._cells)
            self.children = []
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
        if self.isLeaf:
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
                transferOperators[n.id, :, :] = n.transferOperator
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
            if n.isLeaf:
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
            if n.isLeaf:
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef INDEX_t findCell(self, meshBase mesh, REAL_t[::1] vertex, REAL_t[:, ::1] simplex, REAL_t[::1] bary):
        cdef:
            tree_node c
            INDEX_t cellNo = -1
        if minDist2FromBox(self.box, vertex) > 0.:
            return -1
        if self.isLeaf:
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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
        if self.isLeaf:
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
        return 'node({})'.format(m)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void upwardPassMatrix(self, dict coefficientsUp):
        cdef:
            INDEX_t k, i, m, j
            INDEX_t[::1] dof, dofs
            REAL_t[:, ::1] transfer, transfers
            tree_node c
        if self.id in coefficientsUp:
            return
        elif self.isLeaf:
            coefficientsUp[self.id] = (self.value[0, :, :], np.array(self.dofs.toArray()))
        else:
            transfers = np.zeros((self.num_dofs, self.coefficientsUp.shape[0]), dtype=REAL)
            dofs = np.empty((self.num_dofs), dtype=INDEX)
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


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
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


# FIX: move it into tree_node and don't reallocate memory
cdef inline void transferMatrix(REAL_t[:, ::1] boxP,
                                REAL_t[:, ::1] boxC,
                                INDEX_t m,
                                REAL_t[:, ::1] T):
    cdef:
        INDEX_t dim, i, j, l, k, I, J
        REAL_t[:, ::1] omega, beta, xiC, xiP
    dim = boxP.shape[0]
    omega = uninitialized((m, dim), dtype=REAL)
    beta = uninitialized((m, dim), dtype=REAL)
    xiC = uninitialized((m, dim), dtype=REAL)
    xiP = uninitialized((m, dim), dtype=REAL)

    eta = np.cos((2.0*np.arange(m, 0, -1)-1.0) / (2.0*m) * np.pi)
    for i in range(m):
        for j in range(dim):
            xiC[i, j] = (boxC[j, 1]-boxC[j, 0])/2.*(eta[i]+1.0)+boxC[j, 0]
            xiP[i, j] = (boxP[j, 1]-boxP[j, 0])/2.*(eta[i]+1.0)+boxP[j, 0]
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
    for idxP in product(*([range(m)]*dim)):
        J = 0
        for idxC in product(*([range(m)]*dim)):
            for k in range(dim):
                i = idxP[k]
                j = idxC[k]
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void apply(farFieldClusterPair self, REAL_t[::1] x, REAL_t[::1] y):
        gemv(self.kernelInterpolant, x, y, 1.)

    def __repr__(self):
        return 'farFieldClusterPair<{}, {}>'.format(self.n1, self.n2)


cdef class productIterator:
    def __init__(self, INDEX_t m, INDEX_t dim):
        self.m = m
        self.dim = dim
        self.idx = np.zeros((dim), dtype=INDEX)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void reset(self):
        cdef:
            INDEX_t i
        for i in range(self.dim-1):
            self.idx[i] = 0
        self.idx[self.dim-1] = -1

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
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


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def assembleFarFieldInteractions(FractionalKernel kernel, dict Pfar, INDEX_t m, DoFMap dm):
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

    if kernel.variable:
        dofCoords = dm.getDoFCoordinates()
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
            if kernel_variable:
                it.setIndexSet(cP.n1.dofs)
                it.step()
                dof1 = it.i

                it.setIndexSet(cP.n2.dofs)
                it.step()
                dof2 = it.i

                kernel.evalParamsPtr(dim, &dofCoords[dof1, 0], &dofCoords[dof2, 0])
            for i in range(kiSize):
                for j in range(kiSize):
                    cP.kernelInterpolant[i, j] = -kernel.evalPtr(dim, &x[i, 0], &y[j, 0])
                    cP.kernelInterpolant[i, j] += -kernel.evalPtr(dim, &y[j, 0], &x[i, 0])


cdef class H2Matrix(LinearOperator):
    def __init__(self,
                 tree_node tree,
                 dict Pfar,
                 LinearOperator Anear):
        self.tree = tree
        self.Pfar = Pfar
        self.Anear = Anear
        LinearOperator.__init__(self, Anear.shape[0], Anear.shape[1])

    def isSparse(self):
        return False

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        cdef:
            INDEX_t level, componentNo
            tree_node n1, n2
            farFieldClusterPair clusterPair
        self.Anear.matvec(x, y)
        if len(self.Pfar) > 0:
            for componentNo in range(next(self.tree.leaves()).value.shape[0]):
                self.tree.upwardPass(x, componentNo)
                self.tree.resetCoefficientsDown()
                for level in self.Pfar:
                    for clusterPair in self.Pfar[level]:
                        n1, n2 = clusterPair.n1, clusterPair.n2
                        clusterPair.apply(n2.coefficientsUp, n1.coefficientsDown)
                self.tree.downwardPass(y, componentNo)
        return 0

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec_no_overwrite(self,
                                     REAL_t[::1] x,
                                     REAL_t[::1] y) except -1:
        cdef:
            INDEX_t level, componentNo
            tree_node n1, n2
            farFieldClusterPair clusterPair
        self.Anear.matvec_no_overwrite(x, y)
        if len(self.Pfar) > 0:
            for componentNo in range(next(self.tree.leaves()).value.shape[0]):
                self.tree.upwardPass(x, componentNo)
                self.tree.resetCoefficientsDown()
                for level in self.Pfar:
                    for clusterPair in self.Pfar[level]:
                        n1, n2 = clusterPair.n1, clusterPair.n2
                        clusterPair.apply(n2.coefficientsUp, n1.coefficientsDown)
                self.tree.downwardPass(y, componentNo)
        return 0

    property diagonal:
        def __get__(self):
            return self.Anear.diagonal

    property tree_size:
        def __get__(self):
            try:
                md = self.tree.children[0].transferOperator.shape[0]
            except (IndexError, AttributeError):
                # No children, ie only the root node
                # or no far field clusters
                md = 0
            nodes = self.tree.nodes
            dofs = self.shape[0]
            # size of transferMatrix * number of nodes in tree + number of dofs * leaf values
            return md**2*nodes + dofs*md

    property num_far_field_clusters:
        def __get__(self):
            clusters = 0
            for lvl in self.Pfar:
                clusters += len(self.Pfar[lvl])
            return clusters

    property cluster_size:
        def __get__(self):
            try:
                md = self.tree.children[0].transferOperator.shape[0]
            except (IndexError, AttributeError):
                # No children, ie only the root node
                # or no far field clusters
                md = 0
            # number far field cluster pairs * size of kernel interpolant matrices
            return md**2*self.num_far_field_clusters

    property nearField_size:
        def __get__(self):
            if isinstance(self.Anear, Dense_LinearOperator):
                return self.Anear.num_rows*self.Anear.num_columns
            elif isinstance(self.Anear, Multiply_Linear_Operator):
                return self.Anear.A.nnz
            else:
                return self.Anear.nnz

    def __repr__(self):
        return '<%dx%d %s %f fill from near field, %f fill from tree, %f fill from clusters, %d far-field clusters>' % (self.num_rows,
                                                                                                                        self.num_columns,
                                                                                                                        self.__class__.__name__,
                                                                                                                        self.nearField_size/self.num_rows/self.num_columns,
                                                                                                                        self.tree_size/self.num_rows/self.num_columns,
                                                                                                                        self.cluster_size/self.num_rows/self.num_columns,
                                                                                                                        self.num_far_field_clusters)

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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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
        minLvl = min(self.Pfar)
        maxLvl = max(self.Pfar)
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

    def plot(self, Pnear=[], fill='box', nearFieldColor='red', farFieldColor='blue', kernelApproximationColor='yellow', shiftCoefficientColor='red', printRank=False):
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
                for c in Pnear:
                    box1 = [min(c.n1.dofs), max(c.n1.dofs)]
                    box2 = [nd-max(c.n2.dofs), nd-min(c.n2.dofs)]
                    plt.gca().add_patch(patches.Rectangle((box1[0], box2[0]), box1[1]-box1[0], box2[1]-box2[0], fill=True, facecolor=nearFieldColor))
                for lvl in self.Pfar:
                    for c in self.Pfar[lvl]:
                        box1 = [min(c.n1.dofs), max(c.n1.dofs)]
                        box2 = [nd-max(c.n2.dofs), nd-min(c.n2.dofs)]

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

                        if printRank:
                            plt.text(0.5*(min(c.n1.dofs)+max(c.n1.dofs)), nd-0.5*(min(c.n2.dofs)+max(c.n2.dofs)), str(k),
                                     horizontalalignment='center',
                                     verticalalignment='center')

                plt.xlim([0, nd])
                plt.ylim([0, nd])
                plt.axis('equal')
            else:
                 raise NotImplementedError(fill)
        elif self.tree.dim == 2:
            Z = np.zeros((self.num_rows, self.num_columns), dtype=INDEX)
            for lvl in self.Pfar:
                for c in self.Pfar[lvl]:
                    for dof1 in c.n1.dofs:
                        for dof2 in c.n2.dofs:
                            Z[dof1, dof2] = 1
            plt.pcolormesh(Z)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def getDoFBoxesAndCells(meshBase mesh, DoFMap DoFMap, comm=None):
    cdef:
        INDEX_t i, j, I, k, start, end, dim = mesh.dim
        REAL_t[:, :, ::1] boxes = uninitialized((DoFMap.num_dofs, dim, 2), dtype=REAL)
        REAL_t[:, ::1] boxes2
        REAL_t[:, ::1] simplex = uninitialized((dim+1, dim), dtype=REAL)
        REAL_t[::1] m = uninitialized((dim), dtype=REAL), M = uninitialized((dim), dtype=REAL)
        list cells

    boxes[:, :, 0] = np.inf
    boxes[:, :, 1] = -np.inf

    cells = [set() for i in range(DoFMap.num_dofs)]

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
        for j in range(dim):
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
                cells[I].add(i)
    return np.array(boxes, copy=False), cells


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
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


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def getFractionalOrdersDiagonal(variableFractionalOrder s, meshBase mesh):
    cdef:
        REAL_t[:, ::1] centers
        INDEX_t numCells = mesh.num_cells
        INDEX_t cellNo1
        REAL_t[::1] orders = uninitialized((numCells), dtype=REAL)

    centers = mesh.getCellCenters()

    for cellNo1 in range(numCells):
        orders[cellNo1] = s.eval(centers[cellNo1, :],
                                 centers[cellNo1, :])
    return orders


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef BOOL_t getAdmissibleClusters(FractionalKernel kernel, tree_node n1, tree_node n2, INDEX_t farFieldInteractionSize, REAL_t eta=1., dict Pfar=None, list Pnear=None, INDEX_t level=0, INDEX_t maxLevels=200):
    cdef:
        tree_node t1, t2
        bint seemsAdmissible
        REAL_t dist, diam1, diam2, maxDist
        function horizon
        farFieldClusterPair cp
        BOOL_t addedFarFieldClusters = False
        INDEX_t lenNearField
        REAL_t[:, ::1] boxUnion = np.empty((n1.box.shape[0], 2))
        REAL_t diamUnion = 0.
    dist = distBoxes(n1.box, n2.box)
    diam1 = diamBox(n1.box)
    diam2 = diamBox(n2.box)

    seemsAdmissible = eta*dist >= max(diam1, diam2) and not n1.mixed_node and not n2.mixed_node and (farFieldInteractionSize <= n1.num_dofs*n2.num_dofs) and n1.canBeAssembled and n2.canBeAssembled

    if kernel.finiteHorizon:
        horizon = kernel.horizon
        assert isinstance(horizon, constant)
        maxDist = maxDistBoxes(n1.box, n2.box)
        if not kernel.complement:
            if  dist > horizon.value:
                # return True, since we don't want fully ignored cluster pairs to be merged into near field ones.
                return True
        else:
            if maxDist <= horizon.value:
                # same
                return True
        if dist <= horizon.value and horizon.value <= maxDist:
            seemsAdmissible = False
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
    elif (n1.isLeaf and n2.isLeaf) or (level == maxLevels):
        Pnear.append(nearFieldClusterPair(n1, n2))
        if kernel.finiteHorizon and len(Pnear[len(Pnear)-1].cellsInter) > 0:
            if diamUnion > kernel.horizon.value:
                print("Near field cluster pairs need to fit within horizon.\nBox1 {}\nBox2 {}\n{}, {} -> {}".format(np.array(n1.box),
                                                                                                                    np.array(n2.box),
                                                                                                                    diam1,
                                                                                                                    diam2,
                                                                                                                    diamUnion))

    elif (farFieldInteractionSize >= n1.num_dofs*n2.num_dofs) and (diamUnion < kernel.horizon.value):
        Pnear.append(nearFieldClusterPair(n1, n2))
        return False
    elif n1.isLeaf:
        for t2 in n2.children:
            addedFarFieldClusters |= getAdmissibleClusters(kernel, n1, t2, farFieldInteractionSize, eta,
                                                           Pfar, Pnear,
                                                           level+1, maxLevels)
    elif n2.isLeaf:
        for t1 in n1.children:
            addedFarFieldClusters |= getAdmissibleClusters(kernel, t1, n2, farFieldInteractionSize, eta,
                                                           Pfar, Pnear,
                                                           level+1, maxLevels)
    else:
        for t1 in n1.children:
            for t2 in n2.children:
                addedFarFieldClusters |= getAdmissibleClusters(kernel, t1, t2, farFieldInteractionSize, eta,
                                                               Pfar, Pnear,
                                                               level+1, maxLevels)
    if not addedFarFieldClusters:
        if diamUnion < kernel.horizon.value:
            del Pnear[lenNearField:]
            Pnear.append(nearFieldClusterPair(n1, n2))
    return addedFarFieldClusters


def symmetrizeNearFieldClusters(list Pnear):
    cdef:
        set clusters = set()
        nearFieldClusterPair cpNear
        farFieldClusterPair cpFar
        INDEX_t id1, id2
        dict lookup = {}
    for cpNear in Pnear:
        clusters.add((cpNear.n1.id, cpNear.n2.id))
        lookup[cpNear.n1.id] = cpNear.n1
        lookup[cpNear.n2.id] = cpNear.n2
    while len(clusters) > 0:
        id1, id2 = clusters.pop()
        if id1 != id2:
            if (id2, id1) not in clusters:
                Pnear.append(nearFieldClusterPair(lookup[id2], lookup[id1]))
            else:
                clusters.remove((id2, id1))


def trimTree(tree_node tree, list Pnear, dict Pfar):
    cdef:
        nearFieldClusterPair cpNear
        farFieldClusterPair cpFar
        bitArray used = bitArray(maxElement=tree.get_max_id())
    for cpNear in Pnear:
        used.set(cpNear.n1.id)
        used.set(cpNear.n2.id)
    for lvl in Pfar:
        for cpFar in Pfar[lvl]:
            used.set(cpFar.n1.id)
            used.set(cpFar.n2.id)
    print(used.getNumEntries(), tree.get_max_id(), tree.nodes)
    tree.trim(used)
    tree.set_id()
    print(used.getNumEntries(), tree.get_max_id(), tree.nodes)
    used.empty()
    for cpNear in Pnear:
        used.set(cpNear.n1.id)
        used.set(cpNear.n2.id)
    for lvl in Pfar:
        for cpFar in Pfar[lvl]:
            used.set(cpFar.n1.id)
            used.set(cpFar.n2.id)
    print(used.getNumEntries(), tree.get_max_id(), tree.nodes)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
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
        root.set_id()
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
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
