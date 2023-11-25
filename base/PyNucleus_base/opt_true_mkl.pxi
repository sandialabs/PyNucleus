###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

ctypedef INDEX_t MKL_INT

cdef extern from "mkl/mkl_spblas.h":
    void mkl_cspblas_dcsrgemv (const char *transa , const MKL_INT *m , const REAL_t *a , const MKL_INT *ia , const MKL_INT *ja , const REAL_t *x , REAL_t *y)
    void mkl_dcsrmv (const char *transa , const MKL_INT *m , const MKL_INT *k , const REAL_t *alpha , const char *matdescra ,
                     const REAL_t *val , const MKL_INT *indx , const MKL_INT *pntrb , const MKL_INT *pntre ,
                     const REAL_t *x , const REAL_t *beta , REAL_t *y)
    # void mkl_zcsrmv (const char *transa , const MKL_INT *m , const MKL_INT *k , const COMPLEX_t *alpha , const char *matdescra ,
    #                  const COMPLEX_t *val , const MKL_INT *indx , const MKL_INT *pntrb , const MKL_INT *pntre ,
    #                  const COMPLEX_t *x , const COMPLEX_t *beta , COMPLEX_t *y )

cdef void spmv(INDEX_t[::1] indptr, INDEX_t[::1] indices, SCALAR_t[::1] data, SCALAR_t[::1] x, SCALAR_t[::1] y, BOOL_t overwrite=True, BOOL_t trans=False):
    cdef:
        char transA
        INDEX_t num_rows = indptr.shape[0]-1

    assert overwrite

    if not transA:
        transA = 78
    else:
        if SCALAR_t is COMPLEX_t:
            transA = 67
        else:
            transA = 84

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
