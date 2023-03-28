###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, ENCODE_t, BOOL_t
from PyNucleus_fem.quadrature cimport (simplexQuadratureRule,
                             transformQuadratureRule,
                             quadratureRule,
                             quadQuadratureRule,
                             doubleSimplexQuadratureRule, GaussJacobi,
                             simplexDuffyTransformation, simplexXiaoGimbutas)

from PyNucleus_fem.DoFMaps cimport DoFMap
from PyNucleus_fem.femCy cimport volume_t
from PyNucleus_base.ip_norm cimport mydot
from PyNucleus_fem.meshCy cimport (vectorProduct,
                                   volume0D,
                                   volume1D, volume1Dnew,
                                   volume1D_in_2D,
                                   volume2Dnew,
                                   volume3D, volume3Dnew)
from . nonlocalLaplacianBase cimport (double_local_matrix_t,
                                      nonlocalLaplacian1D,
                                      nonlocalLaplacian2D,
                                      panelType, MASK_t,
                                      specialQuadRule)
from . interactionDomains cimport CUT
from . fractionalOrders cimport fractionalOrderBase
from . kernelsCy cimport Kernel

include "config.pxi"



cdef class integrable1D(nonlocalLaplacian1D):
    cdef:
        public quadQuadratureRule qrId, qrVertex0, qrVertex1
        REAL_t[:, ::1] PSI_id, PSI_vertex0, PSI_vertex1

    cdef void getNearQuadRule(self, panelType panel)


cdef class integrable2D(nonlocalLaplacian2D):
    cdef:
        INDEX_t[::1] idx
        public quadQuadratureRule qrEdge0, qrEdge1, qrVertex, qrId
        REAL_t[:, :, ::1] PSI_edge, PSI_id, PSI_vertex

    cdef void getNearQuadRule(self, panelType panel)
