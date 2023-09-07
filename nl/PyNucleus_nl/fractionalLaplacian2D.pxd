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
from . nonlocalLaplacianBase cimport (double_local_matrix_t,
                                      nonlocalLaplacian2D,
                                      specialQuadRule,
                                      panelType,
                                      MASK_t)
from . fractionalOrders cimport fractionalOrderBase
from . kernelsCy cimport (Kernel,
                          FractionalKernel)


cdef class fractionalLaplacian2DZeroExterior(nonlocalLaplacian2D):
    cdef:
        public REAL_t[:, :, ::1] PHI_edge, PSI_edge, PHI_vertex, PSI_vertex
        public REAL_t[:, ::1] PHI_edge2, PHI_vertex2


cdef class singularityCancelationQuadRule2D(quadratureRule):
    pass


cdef class fractionalLaplacian2D(nonlocalLaplacian2D):
    cdef:
        public quadratureRule qrEdge, qrVertex, qrId
        public REAL_t[:, ::1] PSI_edge, PSI_id, PSI_vertex
        REAL_t singularityCancelationIntegrandWithinElement
        REAL_t singularityCancelationIntegrandAcrossElements


cdef class fractionalLaplacian2D_nonsym(fractionalLaplacian2D):
    cdef:
        public REAL_t[:, :, ::1] PHI_edge, PHI_id, PHI_vertex


cdef class fractionalLaplacian2D_boundary(fractionalLaplacian2DZeroExterior):
    cdef:
        public quadQuadratureRule qrVertex0, qrVertex1
        public quadratureRule qrEdge, qrVertex


