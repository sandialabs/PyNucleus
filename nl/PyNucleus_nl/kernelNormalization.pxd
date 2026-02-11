###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cimport numpy as np
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t
from PyNucleus_fem.functions cimport function
from . twoPointFunctions cimport (twoPointFunction,
                                  constantTwoPoint,
                                  parametrizedTwoPointFunction)
from . fractionalOrders cimport fractionalOrderBase
from . interactionDomains cimport interactionDomain

include "kernel_params_decl.pxi"


cdef class memoizedFun:
    cdef:
        dict memory
        int hit, miss
    cdef REAL_t eval(self, REAL_t x)


cdef class constantFractionalLaplacianScaling(constantTwoPoint):
    cdef:
        public INDEX_t dim
        public REAL_t s
        public REAL_t horizon
        public BOOL_t normalized
        public BOOL_t boundary
        public INDEX_t derivative
        public REAL_t tempered
        public INDEX_t termNo
        public REAL_t[::1] values


cdef class variableFractionalLaplacianScaling(parametrizedTwoPointFunction):
    cdef:
        public INDEX_t dim
        public fractionalOrderBase sFun
        public function horizonFun
        public BOOL_t normalized
        public BOOL_t boundary
        public INDEX_t derivative
        public INDEX_t termNo
        public REAL_t[::1] values


cdef class variableFractionalLaplacianScalingWithDifferentHorizon(variableFractionalLaplacianScaling):
    pass


cdef class constantIntegrableScaling(constantTwoPoint):
    cdef:
        kernelType kType
        INDEX_t dim
        REAL_t horizon
        interactionDomain interaction
        REAL_t gaussian_variance
        REAL_t exponentialRate


cdef class variableIntegrableScaling(parametrizedTwoPointFunction):
    cdef:
        kernelType kType
        interactionDomain interaction
        INDEX_t dim
        function horizonFun
        twoPointFunction phi


cdef class variableIntegrableScalingWithDifferentHorizon(variableIntegrableScaling):
    pass
