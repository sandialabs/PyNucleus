###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


cdef class pardiso_lu_solver(solver):
    def __init__(self, LinearOperator A, INDEX_t num_rows=-1):
        solver.__init__(self, A, num_rows)

    cpdef void setup(self, LinearOperator A=None):
        cdef:
            INDEX_t i, j, explicitZeros, explicitZerosRow
            REAL_t[:, ::1] data

        if A is not None:
            self.A = A

        if not isinstance(self.A, (SSS_LinearOperator,
                                   CSR_LinearOperator,
                                   Dense_LinearOperator)):
            if self.A.isSparse():
                self.A = self.A.to_csr_linear_operator()
            else:
                self.A = Dense_LinearOperator(np.ascontiguousarray(self.A.toarray()))
        try_sparsification = False
        sparsificationThreshold = 0.9
        if isinstance(self.A, Dense_LinearOperator) and try_sparsification:
            explicitZeros = 0
            data = self.A.data
            for i in range(self.A.num_rows):
                explicitZerosRow = 0
                for j in range(self.A.num_columns):
                    if data[i, j] == 0.:
                        explicitZerosRow += 1
                explicitZeros += explicitZerosRow
                if not (explicitZerosRow > sparsificationThreshold*self.A.num_columns):
                    break
            if explicitZeros > sparsificationThreshold*self.A.num_rows*self.A.num_columns:
                print('Converting dense to sparse matrix, since {}% of entries are zero.'.format(100.*explicitZeros/REAL(self.A.num_rows*self.A.num_columns)))
                self.A = CSR_LinearOperator.from_dense(self.A)
        if isinstance(self.A, (SSS_LinearOperator,
                               CSR_LinearOperator)):
            from pypardiso import PyPardisoSolver
            try:
                self.Ainv = PyPardisoSolver()
                self.Asp = self.A.to_csr()
                self.Ainv.factorize(self.Asp)
            except RuntimeError:
                print(self.A, np.array(self.A.data))
                raise
        elif isinstance(self.A, Dense_LinearOperator):
            from scipy.linalg import lu_factor
            self.lu, self.perm = lu_factor(self.A.data)
        else:
            raise NotImplementedError('Cannot use operator of type "{}"'.format(type(self.A)))
        self.initialized = True

    cdef int solve(self, vector_t b, vector_t x) except -1:
        cdef:
            REAL_t[::1] temp
        solver.solve(self, b, x)
        if isinstance(self.A, (SSS_LinearOperator, CSR_LinearOperator)):
            temp = self.Ainv.solve(self.Asp,
                                   np.array(b, copy=False, dtype=REAL))
            assign(x, temp)
        else:
            from scipy.linalg import lu_solve
            assign(x, b)
            lu_solve((self.lu, self.perm),
                     np.array(x, copy=False, dtype=REAL),
                     overwrite_b=True)
        return 1

    def __str__(self):
        return 'LU (MKL Pardiso)'
