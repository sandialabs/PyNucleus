###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t
from libc.math cimport sqrt
from cython.view cimport array as carray

cpdef carray uninitializedINDEX(tuple shape)
cpdef carray uninitializedREAL(tuple shape)

ctypedef fused SCALAR_t:
    REAL_t
    COMPLEX_t

cdef void assign(SCALAR_t[::1] y, SCALAR_t[::1] x)
cdef void assignScaled(SCALAR_t[::1] y, SCALAR_t[::1] x, SCALAR_t alpha)
cdef void assign3(SCALAR_t[::1] z, SCALAR_t[::1] x, SCALAR_t alpha, SCALAR_t[::1] y, SCALAR_t beta)
cdef void update(SCALAR_t[::1] x, SCALAR_t[::1] y)
cdef void updateScaled(SCALAR_t[::1] x, SCALAR_t[::1] y, SCALAR_t alpha)
cdef void scaleScalar(SCALAR_t[::1] x, SCALAR_t alpha)
cdef void updateScaledVector(REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] alpha)
cdef SCALAR_t mydot(SCALAR_t[::1] v0, SCALAR_t[::1] v1)
cdef REAL_t norm(SCALAR_t[::1] x)
cdef void gemv(SCALAR_t[:, ::1] A,
               SCALAR_t[::1] x,
               SCALAR_t[::1] y,
               SCALAR_t beta=*)
cdef void gemvF(SCALAR_t[::1, :] A,
                SCALAR_t[::1] x,
                SCALAR_t[::1] y,
                SCALAR_t beta=*)
cdef void gemvT(SCALAR_t[:, ::1] A,
                SCALAR_t[::1] x,
                SCALAR_t[::1] y,
                SCALAR_t beta=*)
cdef void matmat(SCALAR_t[:, ::1] A,
                 SCALAR_t[:, ::1] B,
                 SCALAR_t[:, ::1] C)
cdef void spmv(INDEX_t[::1] indptr, INDEX_t[::1] indices, SCALAR_t[::1] data, SCALAR_t[::1] x, SCALAR_t[::1] y, BOOL_t overwrite=*, BOOL_t trans=*)
cdef void spres(INDEX_t[::1] indptr, INDEX_t[::1] indices, SCALAR_t[::1] data, SCALAR_t[::1] x, SCALAR_t[::1] rhs, SCALAR_t[::1] result)
