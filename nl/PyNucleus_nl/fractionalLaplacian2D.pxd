###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, ENCODE_t, BOOL_t
from PyNucleus_fem.quadrature cimport (quadratureRule,
                                       simplexQuadratureRule,
                                       quadQuadratureRule)
from PyNucleus_fem.DoFMaps cimport DoFMap
from PyNucleus_fem.meshCy cimport meshBase
from . nonlocalOperator cimport (double_local_matrix_t,
                                 nonlocalLaplacian2D,
                                 specialQuadRule,
                                 singularityCancelationQuadRule,
                                 panelType,
                                 MASK_t)
from . fractionalOrders cimport fractionalOrderBase
from . kernels cimport (Kernel,
                        FractionalKernel)


cdef class fractionalLaplacian2DZeroExterior(nonlocalLaplacian2D):
    pass

cdef class singularityCancelationQuadRule2D(singularityCancelationQuadRule):
    pass


cdef class fractionalLaplacian2D(nonlocalLaplacian2D):
    cdef:
        REAL_t singularityCancelationIntegrandWithinElement
        REAL_t singularityCancelationIntegrandAcrossElements


cdef class fractionalLaplacian2D_nonsym(fractionalLaplacian2D):
    pass

cdef class fractionalLaplacian2D_nonsym2(fractionalLaplacian2D_nonsym):
    pass


cdef class fractionalLaplacian2D_boundary(fractionalLaplacian2DZeroExterior):
    pass
