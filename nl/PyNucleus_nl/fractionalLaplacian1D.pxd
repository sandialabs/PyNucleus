###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, ENCODE_t, BOOL_t
from PyNucleus_fem.quadrature cimport (quadratureRule,
                                       simplexQuadratureRule,
                                       quadQuadratureRule,
                                       doubleSimplexQuadratureRule)
from PyNucleus_fem.DoFMaps cimport DoFMap, P0_DoFMap, P1_DoFMap, P2_DoFMap
from PyNucleus_fem.functions cimport function
from . nonlocalLaplacianBase cimport (double_local_matrix_t,
                                      nonlocalLaplacian1D,
                                      panelType,
                                      MASK_t,
                                      specialQuadRule)
from . fractionalOrders cimport fractionalOrderBase
from . kernelsCy cimport (Kernel,
                          FractionalKernel)


cdef class fractionalLaplacian1DZeroExterior(nonlocalLaplacian1D):
    cdef:
        public quadratureRule qrVertex
        public REAL_t[:, ::1] PHI_dist, PHI_sep, PHI_vertex
        dict distantPHI

cdef class singularityCancelationQuadRule1D(quadratureRule):
    pass


cdef class fractionalLaplacian1D(nonlocalLaplacian1D):
    cdef:
        public quadratureRule qrId, qrVertex
        REAL_t[:, ::1] PSI_id, PSI_vertex
        REAL_t singularityCancelationIntegrandWithinElement
        REAL_t singularityCancelationIntegrandAcrossElements


cdef class fractionalLaplacian1D_nonsym(fractionalLaplacian1D):
    cdef:
        REAL_t[:, :, ::1] PHI_id, PHI_vertex


cdef class fractionalLaplacian1D_boundary(fractionalLaplacian1DZeroExterior):
    pass


