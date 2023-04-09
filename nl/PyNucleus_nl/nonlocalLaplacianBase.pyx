###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
cimport numpy as np
from libc.math cimport (sin, cos, sinh, cosh, tanh, sqrt, atan2,
                        log, ceil,
                        fabs as abs, M_PI as pi, pow,
                        tgamma as gamma)
from PyNucleus_base.myTypes import INDEX, REAL, ENCODE, BOOL
from PyNucleus_base import uninitialized
from libc.stdlib cimport malloc, free
from . interactionDomains cimport CUT

# With 64 bits, we can handle at most 5 DoFs per element.
MASK = np.uint64
ALL = MASK(np.iinfo(MASK).max)
include "panelTypes.pxi"

cdef INDEX_t MAX_INT = np.iinfo(INDEX).max
cdef REAL_t inf = np.inf


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

        self.center1 = uninitialized((self.dim), dtype=REAL)
        self.center2 = uninitialized((self.dim), dtype=REAL)

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

    cdef void computeCellPairIdentifierBase(self, INDEX_t[::1] ID, INDEX_t *perm):
        raise NotImplementedError()

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
        self.cellNo1 = -1
        self.cellNo2 = -1

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

    cdef void swapCells(self):
        self.cellNo1, self.cellNo2 = self.cellNo2, self.cellNo1
        self.simplex1, self.simplex2 = self.simplex2, self.simplex1
        self.center1, self.center2 = self.center2, self.center1

    cdef void setSimplex1(self, REAL_t[:, ::1] simplex1):
        self.simplex1 = simplex1
        self.getSimplexCenter(self.simplex1, self.center1)
        self.vol1 = self.volume1(self.simplex1)

    cdef void setSimplex2(self, REAL_t[:, ::1] simplex2):
        self.simplex2 = simplex2
        self.getSimplexCenter(self.simplex2, self.center2)
        self.vol2 = self.volume2(self.simplex2)

    def __call__(self,
                 REAL_t[::1] contrib,
                 panelType panel):
        return self.eval(contrib, panel)

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        raise NotImplementedError()

    def eval_py(self,
                REAL_t[::1] contrib,
                panel):
        self.eval(contrib, panel)

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        raise NotImplementedError()

    cdef REAL_t get_h_simplex(self, const REAL_t[:, ::1] simplex):
        raise NotImplementedError()

    cdef REAL_t get_h_surface_simplex(self, const REAL_t[:, ::1] simplex):
        raise NotImplementedError()

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

    cdef void computeCenterDistance(self):
        cdef:
            INDEX_t j
            REAL_t d2 = 0.
        for j in range(self.dim):
            d2 += (self.center1[j]-self.center2[j])**2
        self.dcenter2 = d2

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


cdef panelType MAX_PANEL = 120


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
        self.distantQuadRulesPtr = <void**>malloc(MAX_PANEL*sizeof(void*))
        for i in range(MAX_PANEL):
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

    def __del__(self):
        free(self.localShapeFunctions)
        free(self.distantQuadRulesPtr)

    cdef void getNearQuadRule(self, panelType panel):
        raise NotImplementedError()

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
            REAL_t alpha
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
        if panel < 0:
            alpha = self.kernel.getSingularityValue()
            if alpha == 0.:
                self.kernel.interaction.getRelativePosition(self.simplex1, self.simplex2)
        return panel

    def __repr__(self):
        return (super(nonlocalLaplacian, self).__repr__() +
                'kernel:                        {}\n'.format(self.kernel))

    cdef inline shapeFunction getLocalShapeFunction(self, INDEX_t local_dof):
        return (<shapeFunction>((<void**>(self.localShapeFunctions+local_dof*sizeof(void*)))[0]))

    cdef void addQuadRule(self, panelType panel):
        cdef:
            simplexQuadratureRule qr0, qr1
            doubleSimplexQuadratureRule qr2
            specialQuadRule sQR
            REAL_t[:, ::1] PSI
            INDEX_t I, k, i, j
            INDEX_t numQuadNodes0, numQuadNodes1, dofs_per_element
            shapeFunction sf
        qr0 = simplexXiaoGimbutas(panel, self.dim)
        qr1 = qr0
        qr2 = doubleSimplexQuadratureRule(qr0, qr1)
        numQuadNodes0 = qr0.num_nodes
        numQuadNodes1 = qr1.num_nodes
        dofs_per_element = self.DoFMap.dofs_per_element
        PSI = uninitialized((2*dofs_per_element,
                             qr2.num_nodes), dtype=REAL)
        # phi_i(x) - phi_i(y) = phi_i(x)
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    PSI[I, k] = sf(qr0.nodes[:, i])
                    k += 1
        # phi_i(x) - phi_i(y) = -phi_i(y)
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    PSI[I+dofs_per_element, k] = -sf(qr1.nodes[:, j])
                    k += 1
        sQR = specialQuadRule(qr2, PSI)
        self.distantQuadRules[panel] = sQR
        self.distantQuadRulesPtr[panel] = <void*>(self.distantQuadRules[panel])

        if numQuadNodes0 > self.x.shape[0]:
            self.x = uninitialized((numQuadNodes0, self.dim), dtype=REAL)
        if numQuadNodes1 > self.y.shape[0]:
            self.y = uninitialized((numQuadNodes1, self.dim), dtype=REAL)
        if numQuadNodes0*numQuadNodes1 > self.temp.shape[0]:
            self.temp = uninitialized((numQuadNodes0*numQuadNodes1), dtype=REAL)

    cdef void addQuadRule_nonSym(self, panelType panel):
        cdef:
            simplexQuadratureRule qr0, qr1
            doubleSimplexQuadratureRule qr2
            specialQuadRule sQR
            REAL_t[:, ::1] PSI
            REAL_t[:, :, ::1] PHI
            INDEX_t I, k, i, j
            INDEX_t numQuadNodes0, numQuadNodes1, dofs_per_element
            shapeFunction sf
        qr0 = simplexXiaoGimbutas(panel, self.dim)
        qr1 = qr0
        qr2 = doubleSimplexQuadratureRule(qr0, qr1)
        numQuadNodes0 = qr0.num_nodes
        numQuadNodes1 = qr1.num_nodes
        dofs_per_element = self.DoFMap.dofs_per_element
        PSI = uninitialized((2*dofs_per_element,
                             qr2.num_nodes), dtype=REAL)
        PHI = uninitialized((2,
                             2*dofs_per_element,
                             qr2.num_nodes), dtype=REAL)
        # phi_i(x) - phi_i(y) = phi_i(x)
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    PSI[I, k] = sf(qr0.nodes[:, i])
                    PHI[0, I, k] = sf(qr0.nodes[:, i])
                    PHI[1, I, k] = 0.
                    k += 1
        # phi_i(x) - phi_i(y) = -phi_i(y)
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    PSI[I+dofs_per_element, k] = -sf(qr1.nodes[:, j])
                    PHI[0, I+dofs_per_element, k] = 0.
                    PHI[1, I+dofs_per_element, k] = sf(qr1.nodes[:, j])
                    k += 1
        sQR = specialQuadRule(qr2, PSI, PHI3=PHI)
        self.distantQuadRules[panel] = sQR
        self.distantQuadRulesPtr[panel] = <void*>(self.distantQuadRules[panel])

        if numQuadNodes0 > self.x.shape[0]:
            self.x = uninitialized((numQuadNodes0, self.dim), dtype=REAL)
        if numQuadNodes1 > self.y.shape[0]:
            self.y = uninitialized((numQuadNodes1, self.dim), dtype=REAL)
        if numQuadNodes0*numQuadNodes1 > self.temp.shape[0]:
            self.temp = uninitialized((numQuadNodes0*numQuadNodes1), dtype=REAL)
            self.temp2 = uninitialized((numQuadNodes0*numQuadNodes1), dtype=REAL)

    cdef void getNonSingularNearQuadRule(self, panelType panel):
        cdef:
            simplexQuadratureRule qr0, qr1
            doubleSimplexQuadratureRule qr2
            specialQuadRule sQR
            REAL_t[:, ::1] PSI
            INDEX_t I, k, i, j
            INDEX_t numQuadNodes0, numQuadNodes1, dofs_per_element
            shapeFunction sf
        try:
            sQR = <specialQuadRule>(self.distantQuadRules[MAX_PANEL+panel])
        except KeyError:
            quadOrder = <panelType>max(ceil(self.target_order), 2)
            qr0 = simplexXiaoGimbutas(quadOrder, self.dim)
            qr1 = qr0
            qr2 = doubleSimplexQuadratureRule(qr0, qr1)
            numQuadNodes0 = qr0.num_nodes
            numQuadNodes1 = qr1.num_nodes
            dofs_per_element = self.DoFMap.dofs_per_element
            PSI = uninitialized((2*dofs_per_element,
                                 qr2.num_nodes),
                                dtype=REAL)

            # phi_i(x) - phi_i(y) = phi_i(x)
            for I in range(dofs_per_element):
                sf = self.getLocalShapeFunction(I)
                k = 0
                for i in range(numQuadNodes0):
                    for j in range(numQuadNodes1):
                        PSI[I, k] = sf(qr0.nodes[:, i])
                        k += 1
            # phi_i(x) - phi_i(y) = -phi_i(y)
            for I in range(dofs_per_element):
                sf = self.getLocalShapeFunction(I)
                k = 0
                for i in range(numQuadNodes0):
                    for j in range(numQuadNodes1):
                        PSI[I+dofs_per_element, k] = -sf(qr1.nodes[:, j])
                        k += 1
            sQR = specialQuadRule(qr2, PSI)
            self.distantQuadRules[MAX_PANEL+panel] = sQR
            self.distantQuadRulesPtr[MAX_PANEL+panel] = <void*>(self.distantQuadRules[MAX_PANEL+panel])
            if numQuadNodes0 > self.x.shape[0]:
                self.x = uninitialized((numQuadNodes0, self.dim), dtype=REAL)
            if numQuadNodes1 > self.y.shape[0]:
                self.y = uninitialized((numQuadNodes1, self.dim), dtype=REAL)
            if qr2.num_nodes > self.temp.shape[0]:
                self.temp = uninitialized((qr2.num_nodes), dtype=REAL)

    cdef void eval_distant(self,
                           REAL_t[::1] contrib,
                           panelType panel,
                           MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J
            REAL_t vol, val, vol1 = self.vol1, vol2 = self.vol2
            doubleSimplexQuadratureRule qr2
            REAL_t[:, ::1] PSI
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            INDEX_t dim = simplex1.shape[1]
            REAL_t c1, c2, PSI_I, PSI_J
            transformQuadratureRule qr0trans, qr1trans
            INDEX_t dofs_per_element, numQuadNodes0, numQuadNodes1
            REAL_t a_b1[3]
            REAL_t a_A1[3][3]
            REAL_t a_A2[3][3]
            REAL_t[::1] b1
            REAL_t[:, ::1] A1, A2
            BOOL_t cutElements = False

        if self.kernel.finiteHorizon:
            # check if the horizon might cut the elements
            if self.kernel.interaction.relPos == CUT:
                cutElements = True
            if self.kernel.complement:
                cutElements = False
                # TODO: cutElements should be set to True, but
                #       need to figure out the element
                #       transformation.

        contrib[:] = 0.

        if not cutElements:
            vol = vol1*vol2
            if panel < 0:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[MAX_PANEL+panel])
            else:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[panel])
            qr2 = <doubleSimplexQuadratureRule>(sQR.qr)
            PSI = sQR.PSI
            qr2.rule1.nodesInGlobalCoords(simplex1, self.x)
            qr2.rule2.nodesInGlobalCoords(simplex2, self.y)

            k = 0
            for i in range(qr2.rule1.num_nodes):
                for j in range(qr2.rule2.num_nodes):
                    self.temp[k] = qr2.weights[k]*self.kernel.evalPtr(dim,
                                                                      &self.x[i, 0],
                                                                      &self.y[j, 0])
                    k += 1

            k = 0
            for I in range(2*self.DoFMap.dofs_per_element):
                for J in range(I, 2*self.DoFMap.dofs_per_element):
                    if mask & (1 << k):
                        val = 0.
                        for i in range(qr2.num_nodes):
                            val += self.temp[i] * PSI[I, i] * PSI[J, i]
                        contrib[k] = val*vol
                    k += 1
        else:
            if panel < 0:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[MAX_PANEL+panel])
            else:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[panel])
            qr2 = <doubleSimplexQuadratureRule>(sQR.qr)
            if sQR.qrTransformed0 is not None:
                qr0trans = sQR.qrTransformed0
            else:
                qr0 = qr2.rule1
                qr0trans = transformQuadratureRule(qr0)
                sQR.qrTransformed0 = qr0trans
            if sQR.qrTransformed1 is not None:
                qr1trans = sQR.qrTransformed1
            else:
                qr1 = qr2.rule2
                qr1trans = transformQuadratureRule(qr1)
                sQR.qrTransformed1 = qr1trans

            numQuadNodes0 = qr0trans.num_nodes
            numQuadNodes1 = qr1trans.num_nodes

            vol = vol1*vol2
            dofs_per_element = self.DoFMap.dofs_per_element

            A1 = a_A1
            b1 = a_b1
            A2 = a_A2

            self.kernel.interaction.startLoopSubSimplices_Simplex(simplex1, simplex2)
            while self.kernel.interaction.nextSubSimplex_Simplex(A1, b1, &c1):
                qr0trans.setAffineBaryTransform(A1, b1)
                qr0trans.nodesInGlobalCoords(simplex1, self.x)
                for i in range(qr0trans.num_nodes):
                    self.kernel.interaction.startLoopSubSimplices_Node(self.x[i, :], simplex2)
                    while self.kernel.interaction.nextSubSimplex_Node(A2, &c2):
                        qr1trans.setLinearBaryTransform(A2)
                        qr1trans.nodesInGlobalCoords(simplex2, self.y)
                        for j in range(qr1trans.num_nodes):
                            val = qr0trans.weights[i]*qr1trans.weights[j]*self.kernel.evalPtr(dim, &self.x[i, 0], &self.y[j, 0])
                            val *= c1 * c2 * vol
                            k = 0
                            for I in range(2*dofs_per_element):
                                if I < dofs_per_element:
                                    PSI_I = self.getLocalShapeFunction(I).evalStrided(&qr0trans.nodes[0, i], numQuadNodes0)
                                else:
                                    PSI_I = -self.getLocalShapeFunction(I-dofs_per_element).evalStrided(&qr1trans.nodes[0, j], numQuadNodes1)
                                for J in range(I, 2*dofs_per_element):
                                    if mask & (1 << k):
                                        if J < dofs_per_element:
                                            PSI_J = self.getLocalShapeFunction(J).evalStrided(&qr0trans.nodes[0, i], numQuadNodes0)
                                        else:
                                            PSI_J = -self.getLocalShapeFunction(J-dofs_per_element).evalStrided(&qr1trans.nodes[0, j], numQuadNodes1)
                                        contrib[k] += val * PSI_I*PSI_J
                                    k += 1

    cdef void eval_distant_nonsym(self,
                                  REAL_t[::1] contrib,
                                  panelType panel,
                                  MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J
            REAL_t vol, val, vol1 = self.vol1, vol2 = self.vol2
            doubleSimplexQuadratureRule qr2
            REAL_t[:, ::1] PSI
            REAL_t[:, :, ::1] PHI
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            INDEX_t dim = simplex1.shape[1]
            BOOL_t cutElements = False
            REAL_t w

        if self.kernel.finiteHorizon:
            # check if the horizon might cut the elements
            if self.kernel.interaction.relPos == CUT:
                cutElements = True
            if self.kernel.complement:
                cutElements = False
                # TODO: cutElements should be set to True, but
                #       need to figure out the element
                #       transformation.

        contrib[:] = 0.

        if not cutElements:
            vol = vol1*vol2
            if panel < 0:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[MAX_PANEL+panel])
            else:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[panel])
            qr2 = <doubleSimplexQuadratureRule>(sQR.qr)
            PSI = sQR.PSI
            PHI = sQR.PHI3
            qr2.rule1.nodesInGlobalCoords(simplex1, self.x)
            qr2.rule2.nodesInGlobalCoords(simplex2, self.y)

            k = 0
            for i in range(qr2.rule1.num_nodes):
                for j in range(qr2.rule2.num_nodes):
                    w = qr2.weights[k]
                    self.temp[k] = w * self.kernel.evalPtr(dim,
                                                           &self.x[i, 0],
                                                           &self.y[j, 0])
                    self.temp2[k] = w * self.kernel.evalPtr(dim,
                                                            &self.y[j, 0],
                                                            &self.x[i, 0])
                    k += 1

            k = 0
            for I in range(2*self.DoFMap.dofs_per_element):
                for J in range(2*self.DoFMap.dofs_per_element):
                    if mask & (1 << k):
                        val = 0.
                        for i in range(qr2.num_nodes):
                            val += (self.temp[i] * PHI[0, I, i] - self.temp2[i] * PHI[1, I, i]) * PSI[J, i]
                        contrib[k] = val*vol
                    k += 1
        else:
            raise NotImplementedError()


cdef class nonlocalLaplacian1D(nonlocalLaplacian):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap DoFMap, num_dofs=None, manifold_dim2=-1, **kwargs):
        super(nonlocalLaplacian1D, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2, **kwargs)

    cdef REAL_t get_h_simplex(self, const REAL_t[:, ::1] simplex):
        cdef:
            REAL_t h2
        h2 = abs(simplex[1, 0]-simplex[0, 0])
        return h2

    cdef REAL_t get_h_surface_simplex(self, const REAL_t[:, ::1] simplex):
        return 1.

    cdef INDEX_t getCellPairIdentifierSize(self):
        return 3

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

    cdef REAL_t get_h_simplex(self, const REAL_t[:, ::1] simplex):
        cdef:
            INDEX_t i, j
            REAL_t hmax = 0., h2
        for i in range(2):
            for j in range(i+1, 3):
                h2 = (simplex[j, 0]-simplex[i, 0])*(simplex[j, 0]-simplex[i, 0]) + (simplex[j, 1]-simplex[i, 1])*(simplex[j, 1]-simplex[i, 1])
                hmax = max(hmax, h2)
        return sqrt(hmax)

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
