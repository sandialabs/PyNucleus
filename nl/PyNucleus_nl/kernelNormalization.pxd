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
        INDEX_t dim
        REAL_t s, horizon
        REAL_t tempered


cdef class constantFractionalLaplacianScalingBoundary(constantTwoPoint):
    cdef:
        INDEX_t dim
        REAL_t s, horizon, tempered


cdef class constantFractionalLaplacianScalingDerivative(twoPointFunction):
    cdef:
        INDEX_t dim
        REAL_t s
        REAL_t horizon
        REAL_t horizon2
        BOOL_t normalized
        BOOL_t boundary
        INDEX_t derivative
        REAL_t tempered
        REAL_t C
        REAL_t fac
        REAL_t fac2


cdef class variableFractionalLaplacianScaling(parametrizedTwoPointFunction):
    cdef:
        INDEX_t dim
        fractionalOrderBase sFun
        function horizonFun
        REAL_t facInfinite, facFinite
        twoPointFunction phi
        BOOL_t normalized
        BOOL_t boundary
        INDEX_t derivative
        memoizedFun digamma


cdef class variableIntegrableScaling(parametrizedTwoPointFunction):
    cdef:
        kernelType kType
        interactionDomain interaction
        INDEX_t dim
        function horizonFun
        twoPointFunction phi


cdef class variableFractionalLaplacianScalingBoundary(parametrizedTwoPointFunction):
    cdef:
        INDEX_t dim
        fractionalOrderBase sFun
        function horizonFun
        REAL_t facInfinite, facFinite
        twoPointFunction phi
        BOOL_t normalized


cdef class variableFractionalLaplacianScalingWithDifferentHorizon(variableFractionalLaplacianScaling):
    pass


cdef class variableIntegrableScalingWithDifferentHorizon(variableIntegrableScaling):
    pass


cdef class constantIntegrableScaling(constantTwoPoint):
    cdef:
        kernelType kType
        INDEX_t dim
        REAL_t horizon
        interactionDomain interaction
        REAL_t gaussian_variance
        REAL_t exponentialRate


