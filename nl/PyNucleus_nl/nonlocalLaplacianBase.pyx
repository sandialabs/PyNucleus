###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport (sin, cos, sinh, cosh, tanh, sqrt, atan2,
                        log, ceil,
                        fabs as abs, M_PI as pi, pow,
                        tgamma as gamma)
from PyNucleus_base.myTypes import INDEX, REAL, ENCODE, BOOL
from PyNucleus_base import uninitialized
from libc.stdlib cimport malloc

# With 64 bits, we can handle at most 5 DoFs per element.
MASK = np.uint64
ALL = MASK(-1)
include "panelTypes.pxi"

cdef INDEX_t MAX_INT = np.iinfo(INDEX).max
cdef REAL_t inf = np.inf

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void getSimplexAndCenter(const INDEX_t[:, ::1] cells,
                                     const REAL_t[:, ::1] vertices,
                                     const INDEX_t cellIdx,
                                     REAL_t[:, ::1] simplex,
                                     REAL_t[::1] center):
    cdef:
        INDEX_t dim = vertices.shape[1]
        INDEX_t manifold_dim = cells.shape[1]-1
        INDEX_t m, k, l
        REAL_t v, fac = 1./(manifold_dim+1)
    center[:] = 0.
    for m in range(manifold_dim+1):
        k = cells[cellIdx, m]
        for l in range(dim):
            v = vertices[k, l]
            simplex[m, l] = v
            center[l] += v
    for l in range(dim):
        center[l] *= fac


cdef class double_local_matrix_t:
    def __init__(self, INDEX_t dim, INDEX_t manifold_dim1, INDEX_t manifold_dim2):
        self.distantQuadRules = {}
        self.dim = dim
        self.symmetricLocalMatrix = True
        self.symmetricCells = True
        self.cellNo1 = -1
        self.cellNo2 = -1
        self.vol1 = np.nan
        self.vol2 = np.nan

        if dim == 1:
            self.volume1 = volume1Dsimplex
        elif dim == 2:
            self.volume1 = volume2Dsimplex
        else:
            raise NotImplementedError()

        if dim == 1 and manifold_dim2 == 1:
            self.volume2 = volume1Dsimplex
        elif dim == 1 and manifold_dim2 == 0:
            self.volume2 = volume0Dsimplex
        elif dim == 2 and manifold_dim2 == 2:
            self.volume2 = volume2Dsimplex
        elif dim == 2 and manifold_dim2 == 1:
            self.volume2 = volume1Din2Dsimplex
        else:
            raise NotImplementedError()

        if self.dim == 1:
            self.IDENTICAL = COMMON_EDGE
        elif self.dim == 2:
            self.IDENTICAL = COMMON_FACE
        else:
            raise NotImplementedError()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void precomputeSimplices(self):
        # mesh1 and mesh 2 will be the same
        cdef:
            INDEX_t cellNo1
            INDEX_t m, k, l
            REAL_t fac = 1./self.cells1.shape[1]
        self.precomputedSimplices = uninitialized((self.cells1.shape[0], self.cells1.shape[1], self.dim), dtype=REAL)
        self.precomputedCenters = np.zeros((self.cells1.shape[0], self.dim), dtype=REAL)
        for cellNo1 in range(self.cells1.shape[0]):
            for m in range(self.cells1.shape[1]):
                k = self.cells1[cellNo1, m]
                for l in range(self.vertices1.shape[1]):
                    self.precomputedSimplices[cellNo1, m, l] = self.vertices1[k, l]
                    self.precomputedCenters[cellNo1, l] += self.vertices1[k, l]
            for l in range(self.vertices1.shape[1]):
                self.precomputedCenters[cellNo1, l] *= fac

    cdef INDEX_t getCellPairIdentifierSize(self):
        return -1

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void computeCellPairIdentifierBase(self, INDEX_t[::1] ID, INDEX_t *perm):
        raise NotImplementedError()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void computeCellPairIdentifier(self, INDEX_t[::1] ID, INDEX_t *perm):
        self.computeCellPairIdentifierBase(ID, perm)

    def computeCellPairIdentifier_py(self):
        cdef:
            INDEX_t perm = 0
        ID = uninitialized((self.getCellPairIdentifierSize()), dtype=INDEX)
        self.computeCellPairIdentifier(ID, &perm)
        return ID, perm

    cdef void setMesh1(self, meshBase mesh1):
        self.setVerticesCells1(mesh1.vertices, mesh1.cells)
        self.precomputedVolumes = mesh1.volVector
        self.precomputedH = mesh1.hVector
        h1 = 2.*mesh1.h
        d = 2.*mesh1.diam
        self.h1MaxInv = 1./h1
        self.dMaxInv = 1./d

    cdef void setVerticesCells1(self, REAL_t[:, ::1] vertices1, INDEX_t[:, ::1] cells1):
        self.vertices1 = vertices1
        self.cells1 = cells1
        self.simplex1 = uninitialized((self.cells1.shape[1], self.dim), dtype=REAL)
        self.center1 = uninitialized((self.dim), dtype=REAL)
        self.cellNo1 = -1
        self.cellNo2 = -1
        if self.symmetricCells:
            # mesh1 and mesh 2 will be the same
            self.precomputeSimplices()

    cdef void setMesh2(self, meshBase mesh2):
        self.setVerticesCells2(mesh2.vertices, mesh2.cells)
        if mesh2.manifold_dim > 0:
            h2 = 2.*mesh2.h
            self.h2MaxInv = 1./h2
        else:
            self.h2MaxInv = 1.

    cdef void setVerticesCells2(self, REAL_t[:, ::1] vertices2, INDEX_t[:, ::1] cells2):
        self.vertices2 = vertices2
        self.cells2 = cells2
        self.simplex2 = uninitialized((self.cells2.shape[1], self.dim), dtype=REAL)
        self.center2 = uninitialized((self.dim), dtype=REAL)
        self.cellNo1 = -1
        self.cellNo2 = -1

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void setCell1(self, INDEX_t cellNo1):
        if self.cellNo1 == cellNo1:
            return
        self.cellNo1 = cellNo1
        if not self.symmetricCells:
            getSimplexAndCenter(self.cells1, self.vertices1, self.cellNo1, self.simplex1, self.center1)
            self.vol1 = self.volume1(self.simplex1)
        else:
            self.simplex1 = self.precomputedSimplices[cellNo1, :, :]
            self.center1 = self.precomputedCenters[cellNo1, :]
            self.vol1 = self.precomputedVolumes[cellNo1]

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void setCell2(self, INDEX_t cellNo2):
        if self.cellNo2 == cellNo2:
            return
        self.cellNo2 = cellNo2
        if not self.symmetricCells:
            getSimplexAndCenter(self.cells2, self.vertices2, self.cellNo2, self.simplex2, self.center2)
            self.vol2 = self.volume2(self.simplex2)
        else:
            self.simplex2 = self.precomputedSimplices[cellNo2, :, :]
            self.center2 = self.precomputedCenters[cellNo2, :]
            self.vol2 = self.precomputedVolumes[cellNo2]

    def setMesh1_py(self, meshBase mesh1):
        self.setMesh1(mesh1)

    def setMesh2_py(self, meshBase mesh2):
        self.setMesh2(mesh2)

    def setCell1_py(self, INDEX_t cellNo1):
        self.setCell1(cellNo1)

    def setCell2_py(self, INDEX_t cellNo2):
        self.setCell2(cellNo2)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void swapCells(self):
        self.cellNo1, self.cellNo2 = self.cellNo2, self.cellNo1
        self.simplex1, self.simplex2 = self.simplex2, self.simplex1
        self.center1, self.center2 = self.center2, self.center1

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void setSimplex1(self, REAL_t[:, ::1] simplex1):
        self.simplex1 = simplex1
        self.getSimplexCenter(self.simplex1, self.center1)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void setSimplex2(self, REAL_t[:, ::1] simplex2):
        self.simplex2 = simplex2
        self.getSimplexCenter(self.simplex2, self.center2)

    def __call__(self,
                 REAL_t[::1] contrib,
                 panelType panel):
        return self.eval(contrib, panel)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        raise NotImplementedError()

    def eval_py(self,
                REAL_t[::1] contrib,
                panel):
        self.eval(contrib, panel)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        raise NotImplementedError()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t get_h_simplex(self, const REAL_t[:, ::1] simplex):
        raise NotImplementedError()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t get_h_surface_simplex(self, const REAL_t[:, ::1] simplex):
        raise NotImplementedError()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void getSimplexCenter(self,
                               const REAL_t[:, ::1] simplex,
                               REAL_t[::1] center):
        cdef:
            INDEX_t i, j
            REAL_t fac
        center[:] = 0.
        for i in range(simplex.shape[0]):
            for j in range(simplex.shape[1]):
                center[j] += simplex[i, j]
        fac = 1./simplex.shape[0]
        for j in range(simplex.shape[1]):
            center[j] *= fac

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef panelType getProtoPanelType(self):
        # Given two cells, determines their relationship:
        # - COMMON_FACE
        # - COMMON_EDGE
        # - COMMON_VERTEX
        # - DISTANT
        cdef:
            INDEX_t k, i, j
            panelType panel = 0
        if self.symmetricCells:
            if self.cellNo1 > self.cellNo2:
                return IGNORED
        if (self.cells1.shape[1] == self.cells2.shape[1]) and (self.cellNo1 == self.cellNo2):
            return self.IDENTICAL
        for k in range(self.cells2.shape[1]):
            i = self.cells2[self.cellNo2, k]
            for j in range(self.cells1.shape[1]):
                if i == self.cells1[self.cellNo1, j]:
                    panel -= 1
                    break
        return panel

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void computeCenterDistance(self):
        cdef:
            INDEX_t j
            REAL_t d2 = 0.
        for j in range(self.dim):
            d2 += (self.center1[j]-self.center2[j])**2
        self.dcenter2 = d2

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void computeExtremeDistances(self):
        cdef:
            INDEX_t i, k, j
            INDEX_t noSimplex1 = self.simplex1.shape[0]
            INDEX_t noSimplex2 = self.simplex2.shape[0]
            REAL_t d2
            REAL_t dmin2 = inf
            REAL_t dmax2 = 0.
        for i in range(noSimplex1):
            for k in range(noSimplex2):
                d2 = 0.
                for j in range(self.dim):
                    d2 += (self.simplex1[i, j] - self.simplex2[k, j])**2
                dmin2 = min(dmin2, d2)
                dmax2 = max(dmax2, d2)
        self.dmin2 = dmin2
        self.dmax2 = dmax2

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef panelType getPanelType(self):
        raise NotImplementedError()

    cdef void addQuadRule(self, panelType panel):
        raise NotImplementedError()

    def addQuadRule_py(self, panelType panel):
        self.addQuadRule(panel)

    def __repr__(self):
        return '{}\n'.format(self.__class__.__name__)


cdef class specialQuadRule:
    def __init__(self,
                 quadratureRule qr,
                 REAL_t[:, ::1] PSI=None,
                 REAL_t[:, :, ::1] PSI3=None,
                 REAL_t[:, ::1] PHI=None,
                 REAL_t[:, :, ::1] PHI3=None):
        self.qr = qr
        if PSI is not None:
            self.PSI = PSI
        if PSI3 is not None:
            self.PSI3 = PSI3
        if PHI is not None:
            self.PHI = PHI
        if PHI3 is not None:
            self.PHI3 = PHI3


cdef class nonlocalLaplacian(double_local_matrix_t):
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh, DoFMap dm,
                 num_dofs=None, INDEX_t manifold_dim2=-1):
        cdef:
            shapeFunction sf
            INDEX_t i
        if manifold_dim2 < 0:
            manifold_dim2 = mesh.manifold_dim
        double_local_matrix_t.__init__(self, mesh.dim, mesh.manifold_dim, manifold_dim2)
        if num_dofs is None:
            self.num_dofs = dm.num_dofs
        else:
            self.num_dofs = num_dofs
        self.hmin = mesh.hmin
        self.H0 = mesh.diam/sqrt(8)
        self.DoFMap = dm
        self.localShapeFunctions = malloc(self.DoFMap.dofs_per_element*sizeof(void*))
        for i in range(self.DoFMap.dofs_per_element):
            sf = dm.localShapeFunctions[i]
            (<void**>(self.localShapeFunctions+i*sizeof(void*)))[0] = <void*>sf
        self.specialQuadRules = {}
        self.distantQuadRulesPtr = <void**>malloc(100*sizeof(void*))
        for i in range(100):
            self.distantQuadRulesPtr[i] = NULL

        self.kernel = kernel

        if self.kernel.variable:
            self.symmetricCells = self.kernel.symmetric
            self.symmetricLocalMatrix = self.kernel.symmetric
        else:
            self.symmetricCells = True
            self.symmetricLocalMatrix = True

        if self.kernel.variableHorizon:
            self.symmetricCells = False

    cdef void getNearQuadRule(self, panelType panel):
        raise NotImplementedError()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void computeCellPairIdentifier(self, INDEX_t[::1] ID, INDEX_t *perm):
        assert not self.kernel.variable
        if self.kernel.finiteHorizon:
            self.computeExtremeDistances()
            if self.dmax2 <= self.kernel.getHorizonValue2():
                # entirely within horizon
                self.computeCellPairIdentifierBase(ID, perm)
            elif self.dmin2 >= self.kernel.getHorizonValue2():
                # entirely outside of horizon
                ID[0] = IGNORED
            else:
                # on horizon
                ID[0] = ON_HORIZON
        else:
            self.computeCellPairIdentifierBase(ID, perm)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef panelType getPanelType(self):
        # Given two cells, determines their relationship:
        # - COMMON_FACE
        # - COMMON_EDGE
        # - COMMON_VERTEX
        # - DISTANT
        # - IGNORED
        cdef:
            panelType panel
            REAL_t d, h1, h2
        panel = self.getProtoPanelType()

        if panel == IGNORED:
            return IGNORED

        if self.kernel.variable:
            self.kernel.evalParams(self.center1, self.center2)

        if panel == DISTANT:
            if self.kernel.interaction.getRelativePosition(self.simplex1, self.simplex2) == REMOTE:
                return IGNORED

            self.computeCenterDistance()
            d = sqrt(self.dcenter2)

            if self.symmetricCells:
                h1 = self.precomputedH[self.cellNo1]
            else:
                h1 = self.get_h_simplex(self.simplex1)
            if self.cells1.shape[1] == self.cells2.shape[1]:
                if self.symmetricCells:
                    h2 = self.precomputedH[self.cellNo2]
                else:
                    h2 = self.get_h_simplex(self.simplex2)
            else:
                h2 = self.get_h_surface_simplex(self.simplex2)
            panel = self.getQuadOrder(h1, h2, d)
        elif self.kernel.variable:
            self.getNearQuadRule(panel)
        return panel

    def __repr__(self):
        return (super(nonlocalLaplacian, self).__repr__() +
                'kernel:                        {}\n'.format(self.kernel))

    cdef inline shapeFunction getLocalShapeFunction(self, INDEX_t local_dof):
        return (<shapeFunction>((<void**>(self.localShapeFunctions+local_dof*sizeof(void*)))[0]))


cdef class nonlocalLaplacian1D(nonlocalLaplacian):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap DoFMap, num_dofs=None, manifold_dim2=-1, **kwargs):
        super(nonlocalLaplacian1D, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2, **kwargs)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t get_h_simplex(self, const REAL_t[:, ::1] simplex):
        cdef:
            REAL_t h2
        h2 = abs(simplex[1, 0]-simplex[0, 0])
        return h2

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t get_h_surface_simplex(self, const REAL_t[:, ::1] simplex):
        return 1.

    cdef INDEX_t getCellPairIdentifierSize(self):
        return 3

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void computeCellPairIdentifierBase(self, INDEX_t[::1] ID, INDEX_t * perm):
        # use h1, h2 and midpoint distance as identifier
        cdef:
            REAL_t h1, h2, d

        h1 = self.simplex1[1, 0]-self.simplex1[0, 0]
        h2 = self.simplex2[1, 0]-self.simplex2[0, 0]
        d = self.center2[0]-self.center1[0]

        perm[0] = (d < 0)
        perm[0] += (h1 < 0) << 1
        perm[0] += (h2 < 0) << 3

        h1 = abs(h1)
        h2 = abs(h2)
        d = abs(d)

        ID[0] = <INDEX_t>(MAX_INT*d*self.dMaxInv)
        ID[1] = <INDEX_t>(MAX_INT*h1*self.h1MaxInv)
        ID[2] = <INDEX_t>(MAX_INT*h2*self.h2MaxInv)


cdef class nonlocalLaplacian2D(nonlocalLaplacian):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap DoFMap, num_dofs=None, manifold_dim2=-1, **kwargs):
        super(nonlocalLaplacian2D, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2, **kwargs)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t get_h_simplex(self, const REAL_t[:, ::1] simplex):
        cdef:
            INDEX_t i, j
            REAL_t hmax = 0., h2
        for i in range(2):
            for j in range(i+1, 3):
                h2 = (simplex[j, 0]-simplex[i, 0])*(simplex[j, 0]-simplex[i, 0]) + (simplex[j, 1]-simplex[i, 1])*(simplex[j, 1]-simplex[i, 1])
                hmax = max(hmax, h2)
        return sqrt(hmax)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t get_h_surface_simplex(self, const REAL_t[:, ::1] simplex):
        cdef:
            INDEX_t k
            REAL_t h2
        h2 = 0.
        for k in range(2):
            h2 += (simplex[1, k]-simplex[0, k])**2
        return sqrt(h2)

    cdef INDEX_t getCellPairIdentifierSize(self):
        return 9

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void computeCellPairIdentifierBase(self, INDEX_t[::1] ID, INDEX_t * perm):
        cdef:
            REAL_t d, d1, d2
            REAL_t v00, v01, v10, v11, v20, v21
            REAL_t c00, c01, c10, c11, c20, c21
            INDEX_t rot
            BOOL_t permCells

        d1 = self.center2[0]-self.center1[0]
        d2 = self.center2[1]-self.center1[1]

        d = sqrt(d1*d1 + d2*d2)
        ID[0] = <INDEX_t>(MAX_INT*d*self.dMaxInv)

        if d < 1e-9:
            d1 = 1.
            d2 = 0.
            permCells = False
            perm[0] = False
        else:
            permCells = (d1 < 0)
            perm[0] = permCells
            d = 1.0/d
            d1 *= d
            d2 *= d
            if permCells:
                d1 = -d1
                d2 = -d2

        # 1st simplex

        if not permCells:
            v00 = self.simplex1[0, 0]-self.center1[0]
            v01 = self.simplex1[0, 1]-self.center1[1]
            v10 = self.simplex1[1, 0]-self.center1[0]
            v11 = self.simplex1[1, 1]-self.center1[1]
            v20 = self.simplex1[2, 0]-self.center1[0]
            v21 = self.simplex1[2, 1]-self.center1[1]
        else:
            v00 = self.simplex2[0, 0]-self.center2[0]
            v01 = self.simplex2[0, 1]-self.center2[1]
            v10 = self.simplex2[1, 0]-self.center2[0]
            v11 = self.simplex2[1, 1]-self.center2[1]
            v20 = self.simplex2[2, 0]-self.center2[0]
            v21 = self.simplex2[2, 1]-self.center2[1]

        c00 = v00*d1 + v01*d2
        c10 = v10*d1 + v11*d2
        c20 = v20*d1 + v21*d2

        d1, d2 = -d2, d1

        c01 = v00*d1 + v01*d2
        c11 = v10*d1 + v11*d2
        c21 = v20*d1 + v21*d2

        d1, d2 = d2, -d1

        if c00 != c10:
            if c00 > c10:
                if c00 != c20:
                    if c00 > c20:
                        rot = 0
                    else:
                        rot = 2
                elif c01 > c21:
                    rot = 0
                else:
                    rot = 2
            elif c10 != c20:
                if c10 > c20:
                    rot = 1
                else:
                    rot = 2
            elif c11 > c21:
                rot = 1
            else:
                rot = 2
        elif c01 > c11:
            if c00 > c20:
                rot = 0
            else:
                rot = 2
        elif c10 > c20:
            rot = 1
        else:
            rot = 2

        if not permCells:
            perm[0] += (rot << 1)
        else:
            perm[0] += (rot << 3)

        if rot == 0:
            pass
        elif rot == 1:
            c00, c10 = c10, c20
            c01, c11 = c11, c21
        else:
            c00, c10 = c20, c00
            c01, c11 = c21, c01

        ID[1] = <INDEX_t>(MAX_INT*c00*self.h1MaxInv)
        ID[2] = <INDEX_t>(MAX_INT*c10*self.h1MaxInv)

        ID[3] = <INDEX_t>(MAX_INT*c01*self.h1MaxInv)
        ID[4] = <INDEX_t>(MAX_INT*c11*self.h1MaxInv)

        # 2nd simplex

        if not permCells:
            v00 = self.simplex2[0, 0]-self.center2[0]
            v01 = self.simplex2[0, 1]-self.center2[1]
            v10 = self.simplex2[1, 0]-self.center2[0]
            v11 = self.simplex2[1, 1]-self.center2[1]
            v20 = self.simplex2[2, 0]-self.center2[0]
            v21 = self.simplex2[2, 1]-self.center2[1]
        else:
            v00 = self.simplex1[0, 0]-self.center1[0]
            v01 = self.simplex1[0, 1]-self.center1[1]
            v10 = self.simplex1[1, 0]-self.center1[0]
            v11 = self.simplex1[1, 1]-self.center1[1]
            v20 = self.simplex1[2, 0]-self.center1[0]
            v21 = self.simplex1[2, 1]-self.center1[1]


        c00 = v00*d1 + v01*d2
        c10 = v10*d1 + v11*d2
        c20 = v20*d1 + v21*d2

        d1, d2 = -d2, d1

        c01 = v00*d1 + v01*d2
        c11 = v10*d1 + v11*d2
        c21 = v20*d1 + v21*d2

        if c00 != c10:
            if c00 > c10:
                if c00 != c20:
                    if c00 > c20:
                        rot = 0
                    else:
                        rot = 2
                elif c01 > c21:
                    rot = 0
                else:
                    rot = 2
            elif c10 != c20:
                if c10 > c20:
                    rot = 1
                else:
                    rot = 2
            elif c11 > c21:
                rot = 1
            else:
                rot = 2
        elif c01 > c11:
            if c00 > c20:
                rot = 0
            else:
                rot = 2
        elif c10 > c20:
            rot = 1
        else:
            rot = 2

        if not permCells:
            perm[0] += (rot << 3)
        else:
            perm[0] += (rot << 1)

        if rot == 0:
            pass
        elif rot == 1:
            c00, c10 = c10, c20
            c01, c11 = c11, c21
        else:
            c00, c10 = c20, c00
            c01, c11 = c21, c01

        ID[5] = <INDEX_t>(MAX_INT*c00*self.h2MaxInv)
        ID[6] = <INDEX_t>(MAX_INT*c10*self.h2MaxInv)

        ID[7] = <INDEX_t>(MAX_INT*c01*self.h2MaxInv)
        ID[8] = <INDEX_t>(MAX_INT*c11*self.h2MaxInv)
