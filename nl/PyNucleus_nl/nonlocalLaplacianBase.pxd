###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cimport numpy as np
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, ENCODE_t, BOOL_t
from PyNucleus_fem.quadrature cimport (simplexQuadratureRule, quadratureRule,
                                       doubleSimplexQuadratureRule, GaussJacobi,
                                       simplexDuffyTransformation, simplexXiaoGimbutas,
                                       transformQuadratureRule)
from PyNucleus_fem.DoFMaps cimport DoFMap, shapeFunction
from PyNucleus_fem.meshCy cimport meshBase
from PyNucleus_fem.functions cimport function, constant
from PyNucleus_fem.femCy cimport volume_t
from PyNucleus_fem.meshCy cimport (volume0Dsimplex,
                                   volume1Dsimplex,
                                   volume2Dsimplex,
                                   volume1Din2Dsimplex,
                                   volume3Dsimplex,
                                   volume2Din3Dsimplex)
from . twoPointFunctions cimport (twoPointFunction,
                                  constantTwoPoint)
from . interactionDomains cimport REMOTE
from . fractionalOrders cimport (fractionalOrderBase,
                                 constFractionalOrder,
                                 variableFractionalOrder)
from . kernelsCy cimport (Kernel,
                          FractionalKernel)
from . clusterMethodCy cimport tree_node
ctypedef INDEX_t panelType

from . bitset cimport MASK_t


cdef class PermutationIndexer:
    cdef:
        INDEX_t N
        INDEX_t[::1] onesCountLookup, factorials, lehmer

    cdef INDEX_t rank(self, INDEX_t[::1] perm)


cdef class double_local_matrix_t:
    cdef:
        REAL_t[:, ::1] vertices1, vertices2
        INDEX_t[:, ::1] cells1, cells2
        public dict distantQuadRules
        public DoFMap DoFMap
        INDEX_t dim
        public bint symmetricLocalMatrix
        public bint symmetricCells
        public INDEX_t cellNo1, cellNo2
        REAL_t[:, :, ::1] precomputedSimplices
        REAL_t[:, ::1] precomputedCenters
        REAL_t[::1] precomputedVolumes
        REAL_t[::1] precomputedH
        REAL_t[:, ::1] simplex1, simplex2
        REAL_t[::1] center1, center2
        volume_t volume1, volume2
        public REAL_t vol1, vol2
        panelType IDENTICAL
        REAL_t dmin2, dmax2, dcenter2
        REAL_t h1MaxInv, h2MaxInv, dMaxInv
        PermutationIndexer pI_volume, pI_surface
        public INDEX_t[::1] perm1, perm2, perm
        public INDEX_t[:, ::1] precomputedVolumeSimplexPermutations
        public INDEX_t[:, ::1] precomputedSurfaceSimplexPermutations
        public INDEX_t[:, ::1] precomputedDoFPermutations
    cdef void precomputePermutations(self)
    cdef void precomputeSimplices(self)
    cdef INDEX_t getCellPairIdentifierSize(self)
    cdef void computeCellPairIdentifierBase(self, INDEX_t[::1] ID, INDEX_t *perm)
    cdef void computeCellPairIdentifier(self, INDEX_t[::1] ID, INDEX_t *perm)
    cdef void setMesh1(self, meshBase mesh1)
    cdef void setMesh2(self, meshBase mesh2)
    cdef void setVerticesCells1(self, REAL_t[:, ::1] vertices1, INDEX_t[:, ::1] cells1)
    cdef void setVerticesCells2(self, REAL_t[:, ::1] vertices2, INDEX_t[:, ::1] cells2)
    cdef void setCell1(self, INDEX_t cellNo1)
    cdef void setCell2(self, INDEX_t cellNo2)
    cdef void setSimplex1(self, REAL_t[:, ::1] simplex1)
    cdef void setSimplex2(self, REAL_t[:, ::1] simplex2)
    cdef void swapCells(self)
    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=*)
    cdef void evalVector(self,
                         REAL_t[:, ::1] contrib,
                         panelType panel,
                         MASK_t mask=*)
    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d)
    cdef panelType getProtoPanelType(self)
    cdef void computeCenterDistance(self)
    cdef void computeExtremeDistances(self)
    cpdef panelType getPanelType(self)
    cdef void addQuadRule(self, panelType panel)
    cdef REAL_t get_h_simplex(self, const REAL_t[:, ::1] simplex)
    cdef REAL_t get_h_surface_simplex(self, const REAL_t[:, ::1] simplex)
    cdef void getSimplexCenter(self,
                               const REAL_t[:, ::1] simplex,
                               REAL_t[::1] center)


cdef class nonlocalLaplacian(double_local_matrix_t):
    cdef:
        public REAL_t H0, hmin, num_dofs
        void* localShapeFunctions
        dict specialQuadRules
        public Kernel kernel
        REAL_t[:, ::1] x, y
        void** distantQuadRulesPtr
        REAL_t[::1] temp, temp2
        public REAL_t[::1] n, w
    cdef void getNearQuadRule(self, panelType panel)
    cdef inline shapeFunction getLocalShapeFunction(self, INDEX_t local_dof)
    cdef void addQuadRule(self, panelType panel)
    cdef void addQuadRule_nonSym(self, panelType panel)
    cdef void addQuadRule_boundary(self, panelType panel)
    cdef void getNonSingularNearQuadRule(self, panelType panel)
    cdef void eval_distant(self, REAL_t[::1] contrib, panelType panel, MASK_t mask=*)
    cdef void eval_distant_nonsym(self, REAL_t[::1] contrib, panelType panel, MASK_t mask=*)
    cdef void eval_distant_boundary(self, REAL_t[::1] contrib, panelType panel, MASK_t mask=*)


cdef class specialQuadRule:
    cdef:
        public quadratureRule qr
        public REAL_t[:, ::1] PSI
        public REAL_t[:, :, ::1] PSI3
        public REAL_t[:, ::1] PHI
        public REAL_t[:, :, ::1] PHI3
        public transformQuadratureRule qrTransformed0
        public transformQuadratureRule qrTransformed1


cdef class nonlocalLaplacian1D(nonlocalLaplacian):
    cdef:
        public REAL_t target_order, quad_order_diagonal
        dict distantPSI
        INDEX_t[::1] idx


cdef class nonlocalLaplacian2D(nonlocalLaplacian):
    cdef:
        public REAL_t target_order, quad_order_diagonal, quad_order_diagonalV
        INDEX_t[::1] idx1, idx2, idx3, idx4


