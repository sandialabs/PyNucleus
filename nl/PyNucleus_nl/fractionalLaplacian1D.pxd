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
from PyNucleus_fem.DoFMaps cimport DoFMap, P0_DoFMap, P1_DoFMap, P2_DoFMap
from PyNucleus_fem.functions cimport function
from . nonlocalLaplacianBase cimport (double_local_matrix_t,
                                        nonlocalLaplacian1D,
                                        panelType,
                                        MASK_t,
                                        specialQuadRule)
from . interactionDomains cimport CUT
from . fractionalOrders cimport fractionalOrderBase
from . kernelsCy cimport (Kernel,
                          FractionalKernel)


cdef class fractionalLaplacian1DZeroExterior(nonlocalLaplacian1D):
    cdef:
        public quadQuadratureRule qrVertex
        public REAL_t[:, ::1] PHI_dist, PHI_sep, PHI_vertex
        dict distantPHI


cdef class fractionalLaplacian1D_P1(nonlocalLaplacian1D):
    cdef:
        public quadQuadratureRule qrId, qrVertex
        REAL_t[:, ::1] PSI_id, PSI_vertex


cdef class fractionalLaplacian1D_P1_boundary(fractionalLaplacian1DZeroExterior):
    pass



cdef class fractionalLaplacian1D_P0(nonlocalLaplacian1D):
    cdef:
        public quadQuadratureRule qrId, qrVertex0, qrVertex1
        REAL_t[:, ::1] PSI_id, PSI_vertex0, PSI_vertex1


cdef class fractionalLaplacian1D_P0_boundary(fractionalLaplacian1DZeroExterior):
    pass
