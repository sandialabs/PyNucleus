###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


cimport numpy as np
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, BOOL_t
from PyNucleus_fem.functions cimport function
from . kernelsCy cimport kernelCy


cdef class twoPointFunction:
    cdef:
        public BOOL_t symmetric
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y)
    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y)


cdef class productTwoPoint(twoPointFunction):
    cdef:
        twoPointFunction f1, f2


cdef class constantTwoPoint(twoPointFunction):
    cdef:
        public REAL_t value


cdef class leftRightTwoPoint(twoPointFunction):
    cdef:
        public REAL_t ll, lr, rl, rr, interface


cdef class matrixTwoPoint(twoPointFunction):
    cdef:
        public REAL_t[:, ::1] mat
        REAL_t[::1] n


cdef class temperedTwoPoint(twoPointFunction):
    cdef:
        public REAL_t lambdaCoeff
        public INDEX_t dim


cdef class smoothedLeftRightTwoPoint(twoPointFunction):
    cdef:
        public REAL_t vl, vr, r, slope, fac


cdef class parametrizedTwoPointFunction(twoPointFunction):
    cdef:
        void *params
    cdef void setParams(self, void *params)
    cdef void* getParams(self)


cdef class productParametrizedTwoPoint(parametrizedTwoPointFunction):
    cdef:
        twoPointFunction f1, f2
