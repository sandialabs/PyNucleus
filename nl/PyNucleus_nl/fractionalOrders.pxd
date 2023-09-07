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

include "kernel_params_decl.pxi"


cdef class fractionalOrderBase(twoPointFunction):
    cdef:
        public REAL_t min, max
        public INDEX_t numParameters
    cdef void evalGrad(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] grad)
    cdef REAL_t evalGradPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, INDEX_t vectorSize, REAL_t* grad)


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
    cdef void evalGrad(self, REAL_t[::1] x, REAL_t[::1] grad)
    cdef void evalGradPtr(self, INDEX_t dim, REAL_t* x, INDEX_t vectorSize, REAL_t* grad)


cdef class singleVariableUnsymmetricFractionalOrder(variableFractionalOrder):
    cdef:
        public extendedFunction sFun
    cdef void evalGrad(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] grad)
    cdef REAL_t evalGradPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, INDEX_t vectorSize, REAL_t* grad)



cdef class piecewiseConstantFractionalOrder(variableFractionalOrder):
    cdef:
        public function blockIndicator
        REAL_t[:, ::1] sVals
