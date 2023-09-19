###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from scipy.linalg.cython_blas cimport dcopy, dscal, daxpy, ddot, dnrm2
from scipy.linalg.cython_blas cimport zcopy, zscal, zaxpy, zdotc
from scipy.linalg.cython_blas cimport dgemv, zgemv
from scipy.linalg.cython_blas cimport dgemm, zgemm


cdef void assign(SCALAR_t[::1] y, SCALAR_t[::1] x):
    cdef:
        double* x_ptr
        double* y_ptr
        double complex* x_ptr_c
        double complex* y_ptr_c
        int n = x.shape[0]
        int inc = 1
    if SCALAR_t is COMPLEX_t:
        x_ptr_c = &x[0]
        y_ptr_c = &y[0]
        zcopy(&n, x_ptr_c, &inc, y_ptr_c, &inc)
    else:
        x_ptr = &x[0]
        y_ptr = &y[0]
        dcopy(&n, x_ptr, &inc, y_ptr, &inc)

cdef void assignScaled(SCALAR_t[::1] y, SCALAR_t[::1] x, SCALAR_t alpha):
    cdef:
        double* x_ptr
        double* y_ptr
        double complex* x_ptr_c
        double complex* y_ptr_c
        int n = x.shape[0]
        int inc = 1
    if SCALAR_t is COMPLEX_t:
        x_ptr_c = &x[0]
        y_ptr_c = &y[0]
        zcopy(&n, x_ptr_c, &inc, y_ptr_c, &inc)
        zscal(&n, &alpha, y_ptr_c, &inc)
    else:
        x_ptr = &x[0]
        y_ptr = &y[0]
        dcopy(&n, x_ptr, &inc, y_ptr, &inc)
        dscal(&n, &alpha, y_ptr, &inc)

cdef void assign3(SCALAR_t[::1] z, SCALAR_t[::1] x, SCALAR_t alpha, SCALAR_t[::1] y, SCALAR_t beta):
    cdef:
        double* x_ptr
        double* y_ptr
        double* z_ptr
        double complex* x_ptr_c
        double complex* y_ptr_c
        double complex* z_ptr_c
        int n = x.shape[0]
        int inc = 1
    if SCALAR_t is COMPLEX_t:
        x_ptr_c = &x[0]
        y_ptr_c = &y[0]
        z_ptr_c = &z[0]
        zcopy(&n, x_ptr_c, &inc, z_ptr_c, &inc)
        if alpha != 1.0:
            zscal(&n, &alpha, z_ptr_c, &inc)
        zaxpy(&n, &beta, y_ptr_c, &inc, z_ptr_c, &inc)
    else:
        x_ptr = &x[0]
        y_ptr = &y[0]
        z_ptr = &z[0]
        dcopy(&n, x_ptr, &inc, z_ptr, &inc)
        if alpha != 1.0:
            dscal(&n, &alpha, z_ptr, &inc)
        daxpy(&n, &beta, y_ptr, &inc, z_ptr, &inc)

cdef void update(SCALAR_t[::1] x, SCALAR_t[::1] y):
    cdef:
        double* x_ptr
        double* y_ptr
        double complex* x_ptr_c
        double complex* y_ptr_c
        SCALAR_t alpha = 1.0
        int n = x.shape[0]
        int inc = 1
    if SCALAR_t is COMPLEX_t:
        x_ptr_c = &x[0]
        y_ptr_c = &y[0]
        zaxpy(&n, &alpha, y_ptr_c, &inc, x_ptr_c, &inc)
    else:
        x_ptr = &x[0]
        y_ptr = &y[0]
        daxpy(&n, &alpha, y_ptr, &inc, x_ptr, &inc)

cdef void updateScaled(SCALAR_t[::1] x, SCALAR_t[::1] y, SCALAR_t alpha):
    cdef:
        double* x_ptr
        double* y_ptr
        double complex* x_ptr_c
        double complex* y_ptr_c
        int n = x.shape[0]
        int inc = 1
    if SCALAR_t is COMPLEX_t:
        x_ptr_c = &x[0]
        y_ptr_c = &y[0]
        zaxpy(&n, &alpha, y_ptr_c, &inc, x_ptr_c, &inc)
    else:
        x_ptr = &x[0]
        y_ptr = &y[0]
        daxpy(&n, &alpha, y_ptr, &inc, x_ptr, &inc)

cdef void scaleScalar(SCALAR_t[::1] x, SCALAR_t alpha):
    cdef:
        double* x_ptr
        double complex* x_ptr_c
        int n = x.shape[0]
        int inc = 1
    if SCALAR_t is COMPLEX_t:
        x_ptr_c = &x[0]
        zscal(&n, &alpha, x_ptr_c, &inc)
    else:
        x_ptr = &x[0]
        dscal(&n, &alpha, x_ptr, &inc)

cdef SCALAR_t mydot(SCALAR_t[::1] v0, SCALAR_t[::1] v1):
    cdef:
        SCALAR_t s = 0.0
        double* v0_ptr
        double* v1_ptr
        double complex* v0_ptr_c
        double complex* v1_ptr_c
        int n = v0.shape[0]
        int inc = 1
    if SCALAR_t is COMPLEX_t:
        v0_ptr_c = &v0[0]
        v1_ptr_c = &v1[0]
        s = zdotc(&n, v0_ptr_c, &inc, v1_ptr_c, &inc)
    else:
        v0_ptr = &v0[0]
        v1_ptr = &v1[0]
        s = ddot(&n, v0_ptr, &inc, v1_ptr, &inc)
    return s

cdef REAL_t norm(SCALAR_t[::1] x):
    cdef:
        REAL_t s = 0.0
        double* x_ptr
        double complex* x_ptr_c
        int n = x.shape[0]
        int inc = 1
    if SCALAR_t is COMPLEX_t:
        x_ptr_c = &x[0]
        s = zdotc(&n, x_ptr_c, &inc, x_ptr_c, &inc).real
        return sqrt(s)
    else:
        x_ptr = &x[0]
        return dnrm2(&n, x_ptr, &inc)

cdef void gemv(SCALAR_t[:, ::1] A, SCALAR_t[::1] x, SCALAR_t[::1] y, SCALAR_t beta=0.):
    cdef:
        double* A_ptr
        double* x_ptr
        double* y_ptr
        double complex* A_ptr_c
        double complex* x_ptr_c
        double complex* y_ptr_c
        int m = A.shape[1]
        int n = A.shape[0]
        SCALAR_t alpha = 1.
        int lda = A.shape[1]
        int incx = 1
        int incy = 1
    if SCALAR_t is COMPLEX_t:
        A_ptr_c = &A[0, 0]
        x_ptr_c = &x[0]
        y_ptr_c = &y[0]
        zgemv('t', &m, &n, &alpha, A_ptr_c, &lda, x_ptr_c, &incx, &beta, y_ptr_c, &incy)
    else:
        A_ptr = &A[0, 0]
        x_ptr = &x[0]
        y_ptr = &y[0]
        dgemv('t', &m, &n, &alpha, A_ptr, &lda, x_ptr, &incx, &beta, y_ptr, &incy)

cdef void gemvF(SCALAR_t[::1, :] A, SCALAR_t[::1] x, SCALAR_t[::1] y, SCALAR_t beta=0.):
    cdef:
        double* A_ptr
        double* x_ptr
        double* y_ptr
        double complex* A_ptr_c
        double complex* x_ptr_c
        double complex* y_ptr_c
        int m = A.shape[0]
        int n = A.shape[1]
        SCALAR_t alpha = 1.
        int lda = A.shape[0]
        int incx = 1
        int incy = 1
    if SCALAR_t is COMPLEX_t:
        A_ptr_c = &A[0, 0]
        x_ptr_c = &x[0]
        y_ptr_c = &y[0]
        zgemv('n', &m, &n, &alpha, A_ptr_c, &lda, x_ptr_c, &incx, &beta, y_ptr_c, &incy)
    else:
        A_ptr = &A[0, 0]
        x_ptr = &x[0]
        y_ptr = &y[0]
        dgemv('n', &m, &n, &alpha, A_ptr, &lda, x_ptr, &incx, &beta, y_ptr, &incy)


cdef void gemvT(SCALAR_t[:, ::1] A, SCALAR_t[::1] x, SCALAR_t[::1] y, SCALAR_t beta=0.):
    cdef:
        double* A_ptr
        double* x_ptr
        double* y_ptr
        double complex* A_ptr_c
        double complex* x_ptr_c
        double complex* y_ptr_c
        int m = A.shape[0]
        int n = A.shape[1]
        SCALAR_t alpha = 1.
        int lda = A.shape[0]
        int incx = 1
        int incy = 1
    if SCALAR_t is COMPLEX_t:
        A_ptr_c = &A[0, 0]
        x_ptr_c = &x[0]
        y_ptr_c = &y[0]
        zgemv('n', &m, &n, &alpha, A_ptr_c, &lda, x_ptr_c, &incx, &beta, y_ptr_c, &incy)
    else:
        A_ptr = &A[0, 0]
        x_ptr = &x[0]
        y_ptr = &y[0]
        dgemv('n', &m, &n, &alpha, A_ptr, &lda, x_ptr, &incx, &beta, y_ptr, &incy)


cdef void matmat(SCALAR_t[:, ::1] A, SCALAR_t[:, ::1] B, SCALAR_t[:, ::1] C):
    cdef:
        double* A_ptr
        double* B_ptr
        double* C_ptr
        double complex* A_ptr_c
        double complex* B_ptr_c
        double complex* C_ptr_c
        int m = B.shape[1]
        int k = B.shape[0]
        int n = A.shape[0]
        SCALAR_t alpha = 1.
        SCALAR_t beta = 0.
        int lda = A.shape[1]
        int ldb = B.shape[1]
        int ldc = C.shape[1]
    if SCALAR_t is COMPLEX_t:
        A_ptr_c = &A[0, 0]
        B_ptr_c = &B[0, 0]
        C_ptr_c = &C[0, 0]
        zgemm('n', 'n', &m, &n, &k, &alpha, B_ptr_c, &ldb, A_ptr_c, &lda, &beta, C_ptr_c, &ldc)
    else:
        A_ptr = &A[0, 0]
        B_ptr = &B[0, 0]
        C_ptr = &C[0, 0]
        dgemm('n', 'n', &m, &n, &k, &alpha, B_ptr, &ldb, A_ptr, &lda, &beta, C_ptr, &ldc)
