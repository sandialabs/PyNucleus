###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cimport numpy as np
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, BOOL_t
from PyNucleus_fem.functions cimport function
from . twoPointFunctions cimport (twoPointFunction,
                                  constantTwoPoint,
                                  parametrizedTwoPointFunction)
from . interactionDomains cimport interactionDomain

include "kernel_params_decl.pxi"


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
        INDEX_t derivative
        REAL_t tempered
        REAL_t C
        REAL_t fac


cdef class variableFractionalLaplacianScaling(parametrizedTwoPointFunction):
    cdef:
        INDEX_t dim
        fractionalOrderBase sFun
        function horizonFun
        REAL_t facInfinite, facFinite
        twoPointFunction phi
        BOOL_t normalized
        INDEX_t derivative


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


cdef class fractionalOrderBase(twoPointFunction):
    cdef:
        public REAL_t min, max


cdef class constFractionalOrder(fractionalOrderBase):
    cdef:
        public REAL_t value


cdef class variableFractionalOrder(fractionalOrderBase):
    cdef:
        void *c_params
    cdef void setFractionalOrderFun(self, void* params)


cdef class extendedFunction(function):
    cdef REAL_t eval(self, REAL_t[::1])
    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x)


cdef class singleVariableUnsymmetricFractionalOrder(variableFractionalOrder):
    cdef:
        public extendedFunction sFun

cdef class piecewiseConstantFractionalOrder(variableFractionalOrder):
    cdef:
        public function blockIndicator
        REAL_t[:, ::1] sVals


cdef class constantIntegrableScaling(constantTwoPoint):
    cdef:
        kernelType kType
        INDEX_t dim
        REAL_t horizon
        interactionDomain interaction
