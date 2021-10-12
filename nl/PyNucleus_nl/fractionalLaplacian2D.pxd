###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, ENCODE_t, BOOL_t
from PyNucleus_fem.quadrature cimport (simplexQuadratureRule, quadQuadratureRule,
                             doubleSimplexQuadratureRule, GaussJacobi,
                             simplexDuffyTransformation, simplexXiaoGimbutas)
from PyNucleus_fem.DoFMaps cimport DoFMap
from PyNucleus_fem.meshCy cimport meshBase
from PyNucleus_fem.functions cimport function
from . nonlocalLaplacianBase cimport (double_local_matrix_t,
                                        nonlocalLaplacian2D,
                                        specialQuadRule,
                                        panelType, MASK_t)
from . interactionDomains cimport CUT
from . fractionalOrders cimport fractionalOrderBase
from . kernels2 cimport (Kernel,
                         FractionalKernel)


cdef class fractionalLaplacian2DZeroExterior(nonlocalLaplacian2D):
    cdef:
        public REAL_t[:, :, ::1] PHI_edge, PSI_edge, PHI_vertex, PSI_vertex
        dict distantPHI
        public REAL_t[::1] n, w


cdef class fractionalLaplacian2D_P1(nonlocalLaplacian2D):
    cdef:
        public quadQuadratureRule qrEdge0, qrEdge1, qrVertex, qrId
        REAL_t[:, :, ::1] PSI_edge, PSI_id, PSI_vertex


cdef class fractionalLaplacian2D_P1_boundary(fractionalLaplacian2DZeroExterior):
    cdef:
        public quadQuadratureRule qrEdge, qrVertex0, qrVertex1


