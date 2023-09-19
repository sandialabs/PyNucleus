###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class chol_solver(solver):
    def __init__(self, LinearOperator A, INDEX_t num_rows=-1):
        solver.__init__(self, A, num_rows)

    cpdef void setup(self, LinearOperator A=None):
        cdef:
            LinearOperator B = None
            INDEX_t i, j, k
        if A is not None:
            self.A = A

        if not isinstance(self.A, (SSS_LinearOperator,
                                   CSR_LinearOperator,
                                   Dense_LinearOperator)):
            if self.A.isSparse():
                B = self.A.to_csr_linear_operator()
            else:
                B = Dense_LinearOperator(np.ascontiguousarray(self.A.toarray()))
        else:
            B = self.A

        self.denseFactor = False
        if isinstance(B, Dense_LinearOperator):
            from scipy.linalg import cholesky
            L = cholesky(B.toarray(), lower=True)
            self.Lflat = uninitialized((L.shape[0]*(L.shape[1]+1)//2), dtype=REAL)
            k = 0
            for i in range(L.shape[0]):
                for j in range(i+1):
                    self.Lflat[k] = L[i, j]
                    k += 1
            self.denseFactor = True
            self.temp = uninitialized((L.shape[0]), dtype=REAL)
        else:
            raise NotImplementedError("Cholmod not available, install \"scikit-sparse\".")
        self.initialized = True

    cdef int solve(self, vector_t b, vector_t x) except -1:
        cdef:
            INDEX_t i, j, k
            REAL_t val
            REAL_t[::1] temp = self.temp, Lflat = self.Lflat
        solver.solve(self, b, x)
        if not self.denseFactor:
            np.array(x, copy=False, dtype=REAL)[:] = self.Ainv(np.array(b, copy=False, dtype=REAL))
        else:
            k = 0
            for i in range(x.shape[0]):
                val = b[i]
                for j in range(i):
                    val -= Lflat[k]*temp[j]
                    k += 1
                temp[i] = val/Lflat[k]
                k += 1
            for i in range(x.shape[0]-1, -1, -1):
                val = temp[i]
                for j in range(i+1, x.shape[0]):
                    k = ((j*(j+1)) >> 1) + i
                    val -= Lflat[k]*x[j]
                k = ((i*(i+1)) >> 1) + i
                x[i] = val/Lflat[k]
        return 1

    def __str__(self):
        return 'Cholesky'
