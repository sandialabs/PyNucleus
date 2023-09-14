###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cimport numpy as np
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, ENCODE_t, BOOL_t
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
                          ComplexKernel,
                          FractionalKernel)
from . clusterMethodCy cimport tree_node
ctypedef INDEX_t panelType

from . bitset cimport MASK_t


cdef class PermutationIndexer:
    cdef:
        INDEX_t N
        INDEX_t[::1] onesCountLookup, factorials, lehmer

    cdef INDEX_t rank(self, INDEX_t[::1] perm)


include "nonlocalOperator_decl_REAL.pxi"
include "nonlocalOperator_decl_COMPLEX.pxi"


cdef class specialQuadRule:
    cdef:
        public quadratureRule qr
        public REAL_t[:, ::1] PSI
        public REAL_t[:, :, ::1] PSI3
        public REAL_t[:, ::1] PHI
        public REAL_t[:, :, ::1] PHI3
        public transformQuadratureRule qrTransformed0
        public transformQuadratureRule qrTransformed1


cdef class nonlocalLaplacian1D(nonlocalOperator):
    cdef:
        public REAL_t target_order, quad_order_diagonal
        dict distantPSI
        INDEX_t[::1] idx


cdef class nonlocalLaplacian2D(nonlocalOperator):
    cdef:
        public REAL_t target_order, quad_order_diagonal, quad_order_diagonalV
        INDEX_t[::1] idx1, idx2, idx3, idx4


