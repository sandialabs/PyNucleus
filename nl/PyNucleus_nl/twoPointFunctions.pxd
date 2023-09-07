###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cimport numpy as np
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t
from PyNucleus_fem.functions cimport function, complexFunction


include "twoPointFunctions_decl_REAL.pxi"
include "twoPointFunctions_decl_COMPLEX.pxi"


cdef class leftRightTwoPoint(twoPointFunction):
    cdef:
        public REAL_t ll, lr, rl, rr, interface


cdef class interfaceTwoPoint(twoPointFunction):
    cdef:
        public REAL_t horizon1, horizon2, interface
        public BOOL_t left


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


cdef class unsymTwoPoint(twoPointFunction):
    cdef:
        public REAL_t l, r


cdef class inverseTwoPoint(twoPointFunction):
    cdef:
        twoPointFunction f
