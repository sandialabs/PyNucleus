###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

ctypedef INDEX_t MKL_INT

cdef extern from "mkl/mkl_spblas.h":
    void mkl_dcsrsm (const char *transa , const MKL_INT *m , const MKL_INT *n , const REAL_t *alpha , const char *matdescra ,
                     const REAL_t *val , const MKL_INT *indx , const MKL_INT *pntrb , const MKL_INT *pntre ,
                     const REAL_t *b , const MKL_INT *ldb , REAL_t *c , const MKL_INT *ldc );

cdef inline void trisolve_mkl(INDEX_t[::1] indptr,
                              INDEX_t[::1] indices,
                              REAL_t[::1] data,
                              REAL_t[::1] b,
                              REAL_t[::1] y,
                              BOOL_t forward=True,
                              BOOL_t unitDiagonal=False):
    cdef:
        char transA
        REAL_t alpha = 1.
        char matdscr[6]
        INDEX_t inc = 1
        INDEX_t n = indptr.shape[0]-1
        INDEX_t one = 1
    matdscr[0] = 84
    if forward:
        transA = 84
    else:
        transA = 78
    matdscr[1] = 85
    if unitDiagonal:
        matdscr[2] = 85
    else:
        matdscr[2] = 78
    matdscr[3] = 67
    mkl_dcsrsm(&transA, &n, &one, &alpha, &matdscr[0], &data[0], &indices[0], &indptr[0], &indptr[1], &b[0], &one, &y[0], &one)


cdef class ichol_solver(solver):
    def __init__(self, LinearOperator A=None, INDEX_t num_rows=-1):
        solver.__init__(self, A, num_rows)
        self.temp = uninitialized((self.num_rows), dtype=REAL)

    cpdef void setup(self, LinearOperator A=None):
        cdef:
            INDEX_t i
        if A is not None:
            self.A = A
        if isinstance(self.A, CSR_LinearOperator):
            self.indices, self.indptr, self.data, self.diagonal = ichol_csr(self.A)
        elif isinstance(self.A, SSS_LinearOperator):
            # self.indices, self.indptr, self.data, self.diagonal = ichol_sss(A)
            self.indices, self.indptr, self.data, self.diagonal = ichol_csr(self.A.to_csr_linear_operator())
        else:
            try:
                B = self.A.to_csr_linear_operator()
                self.indices, self.indptr, self.data, self.diagonal = ichol_csr(B)
            except:
                raise NotImplementedError()

        from . linear_operators import diagonalOperator
        T = CSR_LinearOperator(self.indices, self.indptr, self.data).to_csr()+diagonalOperator(self.diagonal).to_csr()
        self.L = CSR_LinearOperator.from_csr(T)
        self.initialized = True

    cdef int solve(self, vector_t b, vector_t x) except -1:
        solver.solve(self, b, x)
        self.temp[:] = 0.0
        trisolve_mkl(self.L.indptr, self.L.indices, self.L.data, b, self.temp, forward=True, unitDiagonal=False)
        trisolve_mkl(self.L.indptr, self.L.indices, self.L.data, self.temp, x, forward=False, unitDiagonal=False)
        return 1

    def __str__(self):
        return 'Incomplete Cholesky'
