###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
cimport numpy as np
cimport cython
from . myTypes import INDEX

include "config.pxi"

cdef:
    INDEX_t MAX_INT = np.iinfo(INDEX).max
    REAL_t NAN = np.nan


def uninitialized(*args, **kwargs):
    IF FILL_UNINITIALIZED:
        if 'dtype' in kwargs and np.issubdtype(kwargs['dtype'], np.integer):
            kwargs['fill_value'] = np.iinfo(kwargs['dtype']).min
        else:
            kwargs['fill_value'] = NAN
        return np.full(*args, **kwargs)
    ELSE:
        return np.empty(*args, **kwargs)


def uninitialized_like(*args, **kwargs):
    IF FILL_UNINITIALIZED:
        if 'dtype' in kwargs and np.issubdtype(kwargs['dtype'], np.integer):
            kwargs['fill_value'] = np.iinfo(kwargs['dtype']).min
        else:
            kwargs['fill_value'] = NAN
        return np.full_like(*args, **kwargs)
    ELSE:
        return np.empty_like(*args, **kwargs)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef carray uninitializedINDEX(tuple shape):
    cdef:
        carray a = carray(shape, 4, 'i')
        size_t s, i
    IF FILL_UNINITIALIZED:
        s = 1
        for i in range(len(shape)):
            s *= shape[i]
        for i in range(s):
            (<INDEX_t*>a.data)[i] = MAX_INT
    return a


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef carray uninitializedREAL(tuple shape):
    cdef:
        carray a = carray(shape, 8, 'd')
        size_t s, i
    IF FILL_UNINITIALIZED:
        s = 1
        for i in range(len(shape)):
            s *= shape[i]
        for i in range(s):
            (<REAL_t*>a.data)[i] = NAN
    return a


IF USE_BLAS:

    from scipy.linalg.cython_blas cimport dcopy, dscal, daxpy, ddot, dnrm2
    from scipy.linalg.cython_blas cimport zcopy, zscal, zaxpy, zdotc
    from scipy.linalg.cython_blas cimport dgemv, zgemv
    from scipy.linalg.cython_blas cimport dgemm, zgemm


    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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


    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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


    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
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

ELSE:
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void assign(SCALAR_t[::1] y, const SCALAR_t[::1] x):
        cdef:
            INDEX_t i
        for i in range(x.shape[0]):
            y[i] = x[i]

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void assignScaled(SCALAR_t[::1] y, const SCALAR_t[::1] x, SCALAR_t alpha):
        cdef:
            INDEX_t i
        for i in range(x.shape[0]):
            y[i] = alpha*x[i]


    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void assign3(SCALAR_t[::1] z, const SCALAR_t[::1] x, SCALAR_t alpha, const SCALAR_t[::1] y, SCALAR_t beta):
        cdef:
            INDEX_t i
        for i in range(x.shape[0]):
            z[i] = alpha*x[i] + beta*y[i]

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update(SCALAR_t[::1] x, SCALAR_t[::1] y):
        cdef:
            INDEX_t i
        if SCALAR_t is COMPLEX_t:
            for i in range(x.shape[0]):
                x[i] = x[i] + y[i]
        else:
            for i in range(x.shape[0]):
                x[i] += y[i]

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void updateScaled(SCALAR_t[::1] x, SCALAR_t[::1] y, SCALAR_t alpha):
        cdef:
            INDEX_t i
        if SCALAR_t is COMPLEX_t:
            for i in range(x.shape[0]):
                x[i] = x[i]+alpha*y[i]
        else:
            for i in range(x.shape[0]):
                x[i] += alpha*y[i]

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void scaleScalar(SCALAR_t[::1] x, SCALAR_t alpha):
        cdef:
            INDEX_t i
        if SCALAR_t is COMPLEX_t:
            for i in range(x.shape[0]):
                x[i] *= alpha
        else:
            for i in range(x.shape[0]):
                x[i] = x[i]*alpha

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef SCALAR_t mydot(SCALAR_t[::1] v0, SCALAR_t[::1] v1) nogil:
        cdef:
            int i
            SCALAR_t s = 0.0
        if SCALAR_t is COMPLEX_t:
            for i in range(v0.shape[0]):
                s += v0[i].conjugate()*v1[i]
        else:
            for i in range(v0.shape[0]):
                s += v0[i]*v1[i]
        return s

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t norm(SCALAR_t[::1] x):
        cdef:
            int i
            REAL_t s = 0.0
        if SCALAR_t is COMPLEX_t:
            for i in range(x.shape[0]):
                s += (x[i].conjugate()*x[i]).real
        else:
            for i in range(x.shape[0]):
                s += x[i]*x[i]
        return sqrt(s)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void gemv(SCALAR_t[:, ::1] A, SCALAR_t[::1] x, SCALAR_t[::1] y, SCALAR_t beta=0.):
        cdef:
            INDEX_t i, j
            SCALAR_t s
        if SCALAR_t is COMPLEX_t:
            if beta != 0.:
                for i in range(A.shape[0]):
                    s = 0.
                    for j in range(A.shape[1]):
                        s = s + A[i, j]*x[j]
                    y[i] = beta*y[i]+s
            else:
                for i in range(A.shape[0]):
                    s = 0.
                    for j in range(A.shape[1]):
                        s = s + A[i, j]*x[j]
                    y[i] = s
        else:
            if beta != 0.:
                for i in range(A.shape[0]):
                    s = 0.
                    for j in range(A.shape[1]):
                        s += A[i, j]*x[j]
                    y[i] = beta*y[i]+s
            else:
                for i in range(A.shape[0]):
                    s = 0.
                    for j in range(A.shape[1]):
                        s += A[i, j]*x[j]
                    y[i] = s


    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void gemvT(SCALAR_t[:, ::1] A, SCALAR_t[::1] x, SCALAR_t[::1] y, SCALAR_t beta=0.):
        cdef:
            INDEX_t i, j
        if SCALAR_t is COMPLEX_t:
            if beta != 0.:
                for i in range(A.shape[0]):
                    y[i] = y[i]*beta
            else:
                y[:] = 0.
            for j in range(A.shape[1]):
                for i in range(A.shape[0]):
                    y[i] = y[i]+A[j, i]*x[j]
        else:
            if beta != 0.:
                for i in range(A.shape[0]):
                    y[i] *= beta
            else:
                y[:] = 0.
            for j in range(A.shape[1]):
                for i in range(A.shape[0]):
                    y[i] += A[j, i]*x[j]


    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void matmat(SCALAR_t[:, ::1] A, SCALAR_t[:, ::1] B, SCALAR_t[:, ::1] C):
        cdef:
            INDEX_t i, j, k
        C[:, :] = 0.
        if SCALAR_t is COMPLEX_t:
            for i in range(A.shape[0]):
                for j in range(B.shape[0]):
                    for k in range(B.shape[1]):
                        C[i, k] = C[i, k] + A[i, j]*B[j, k]
        else:
            for i in range(A.shape[0]):
                for j in range(B.shape[0]):
                    for k in range(B.shape[1]):
                        C[i, k] += A[i, j]*B[j, k]



@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void updateScaledVector(REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] alpha):
    cdef:
        INDEX_t i
    for i in range(x.shape[0]):
        x[i] += alpha[i]*y[i]


IF USE_MKL:
    ctypedef INDEX_t MKL_INT

    cdef extern from "mkl/mkl_spblas.h":
        void mkl_cspblas_dcsrgemv (const char *transa , const MKL_INT *m , const REAL_t *a , const MKL_INT *ia , const MKL_INT *ja , const REAL_t *x , REAL_t *y );
        void mkl_dcsrmv (const char *transa , const MKL_INT *m , const MKL_INT *k , const REAL_t *alpha , const char *matdescra ,
                         const REAL_t *val , const MKL_INT *indx , const MKL_INT *pntrb , const MKL_INT *pntre ,
                         const REAL_t *x , const REAL_t *beta , REAL_t *y );
        # void mkl_zcsrmv (const char *transa , const MKL_INT *m , const MKL_INT *k , const COMPLEX_t *alpha , const char *matdescra ,
        #                  const COMPLEX_t *val , const MKL_INT *indx , const MKL_INT *pntrb , const MKL_INT *pntre ,
        #                  const COMPLEX_t *x , const COMPLEX_t *beta , COMPLEX_t *y );

    cdef void spmv(INDEX_t[::1] indptr, INDEX_t[::1] indices, SCALAR_t[::1] data, SCALAR_t[::1] x, SCALAR_t[::1] y, BOOL_t overwrite=True):
        cdef:
            char transA = 78
            INDEX_t num_rows = indptr.shape[0]-1

        assert overwrite

        if SCALAR_t is COMPLEX_t:
            # mkl_cspblas_zcsrgemv(&transA, &num_rows, &data[0], &indptr[0], &indices[0], &x[0], &y[0])
            pass
        else:
            mkl_cspblas_dcsrgemv(&transA, &num_rows, &data[0], &indptr[0], &indices[0], &x[0], &y[0])


    cdef void spres(INDEX_t[::1] indptr, INDEX_t[::1] indices, SCALAR_t[::1] data, SCALAR_t[::1] x, SCALAR_t[::1] rhs, SCALAR_t[::1] result):
        cdef:
            char transA = 78
            SCALAR_t alpha = -1.
            SCALAR_t beta = 1.
            char matdscr[6]
            INDEX_t inc = 1
            INDEX_t num_rows = indptr.shape[0]-1
            INDEX_t num_columns = x.shape[0]

        matdscr[0] = 71
        matdscr[2] = 78
        matdscr[3] = 67

        assign(result, rhs)
        if SCALAR_t is COMPLEX_t:
            pass
            # mkl_dcsrmv(&transA, &num_rows, &num_columns, &alpha, &matdscr[0],
            #            &data[0], &indices[0], &indptr[0], &indptr[1],
            #            &x[0], &beta, &result[0])
        else:
            mkl_dcsrmv(&transA, &num_rows, &num_columns, &alpha, &matdscr[0],
                       &data[0], &indices[0], &indptr[0], &indptr[1],
                       &x[0], &beta, &result[0])

ELSE:

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cdef void spmv(INDEX_t[::1] indptr, INDEX_t[::1] indices, SCALAR_t[::1] data, SCALAR_t[::1] x, SCALAR_t[::1] y, BOOL_t overwrite=True):
        cdef:
            INDEX_t i, jj, j
            SCALAR_t temp
        if SCALAR_t is COMPLEX_t:
            for i in range(indptr.shape[0]-1):
                temp = 0.
                for jj in range(indptr[i], indptr[i+1]):
                    j = indices[jj]
                    temp = temp + data[jj]*x[j]
                if overwrite:
                    y[i] = temp
                else:
                    y[i] = y[i]+temp
        else:
            for i in range(indptr.shape[0]-1):
                temp = 0.
                for jj in range(indptr[i], indptr[i+1]):
                    j = indices[jj]
                    temp += data[jj]*x[j]
                if overwrite:
                    y[i] = temp
                else:
                    y[i] += temp

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cdef void spres(INDEX_t[::1] indptr, INDEX_t[::1] indices, SCALAR_t[::1] data, SCALAR_t[::1] x, SCALAR_t[::1] rhs, SCALAR_t[::1] result):
        cdef:
            INDEX_t i, jj, j
            SCALAR_t temp
            INDEX_t num_rows = indptr.shape[0]-1
        if SCALAR_t is COMPLEX_t:
            for i in range(num_rows):
                temp = rhs[i]
                for jj in range(indptr[i], indptr[i+1]):
                    j = indices[jj]
                    temp = temp-data[jj]*x[j]
                result[i] = temp
        else:
            for i in range(num_rows):
                temp = rhs[i]
                for jj in range(indptr[i], indptr[i+1]):
                    j = indices[jj]
                    temp -= data[jj]*x[j]
                result[i] = temp
