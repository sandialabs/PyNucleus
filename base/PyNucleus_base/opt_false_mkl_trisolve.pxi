###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

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

        for i in range(self.diagonal.shape[0]):
            self.diagonal[i] = 1./self.diagonal[i]
        self.initialized = True

    cdef int solve(self, vector_t b, vector_t x) except -1:
        solver.solve(self, b, x)
        self.temp[:] = 0.0
        forward_solve_sss_noInverse(self.indptr, self.indices,
                                    self.data, self.diagonal,
                                    b, self.temp, unitDiagonal=False)
        backward_solve_sss_noInverse(self.indptr, self.indices,
                                     self.data, self.diagonal,
                                     self.temp, x)
        return 1

    def __str__(self):
        return 'Incomplete Cholesky'
