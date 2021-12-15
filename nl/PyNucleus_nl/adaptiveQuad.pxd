###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, ENCODE_t, BOOL_t
from PyNucleus_fem.DoFMaps cimport DoFMap, P1_DoFMap
from PyNucleus_fem.quadrature cimport simplexQuadratureRule, simplexXiaoGimbutas
from . nonlocalLaplacianBase cimport (double_local_matrix_t,
                                        nonlocalLaplacian1D,
                                        nonlocalLaplacian2D,
                                        panelType,
                                        MASK_t,
                                        specialQuadRule)
from . fractionalOrders cimport (fractionalOrderBase,
                                 constFractionalOrder,
                                 variableFractionalOrder)
from . kernelsCy cimport (Kernel,
                          FractionalKernel)
from . nonlocalLaplacianBase cimport nonlocalLaplacian1D
from . nonlocalLaplacianBase cimport nonlocalLaplacian2D


cdef class fractionalLaplacian1D_P1_automaticQuadrature(nonlocalLaplacian1D):
    cdef:
        REAL_t abstol, reltol
        void *user_ptr
        object integrandId
        object integrandVertex
        object integrandDistant


cdef class fractionalLaplacian1D_P1_nonsymAutomaticQuadrature(nonlocalLaplacian1D):
    cdef:
        REAL_t abstol, reltol
        void *user_ptr
        object integrandId
        object integrandVertex1
        object integrandVertex2
        object integrandDistant
        REAL_t[::1] temp3
        dict distantPHIx, distantPHIy


cdef class fractionalLaplacian2D_P1_automaticQuadrature(nonlocalLaplacian2D):
    cdef:
        REAL_t abstol, reltol
        void *user_ptr
        object integrandId
        object integrandEdge
        object integrandVertex
        object integrandDistant
