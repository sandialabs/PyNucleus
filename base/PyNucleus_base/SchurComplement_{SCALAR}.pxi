###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}SchurComplement({SCALAR_label}LinearOperator):
    def __init__(self,
                 {SCALAR_label}LinearOperator A,
                 INDEX_t[::1] indices,
                 *args, **kwargs):
        self.A = A
        self.R1 = {SCALAR_label}CSR_LinearOperator(indices, np.arange((indices.shape[0]+1), dtype=INDEX), np.ones((indices.shape[0]), dtype={SCALAR}))
        self.R1.num_columns = self.A.shape[0]
        self.P1 = self.R1.transpose()

        temp = np.ones((self.A.shape[0]), dtype=bool)
        for dof in indices:
            temp[dof] = False
        indices2 = np.where(temp)[0].astype(INDEX)
        del temp
        self.R2 = {SCALAR_label}CSR_LinearOperator(indices2, np.arange((indices2.shape[0]+1), dtype=INDEX), np.ones((indices2.shape[0]), dtype={SCALAR}))
        self.R2.num_columns = self.A.shape[0]
        self.P2 = self.R2.transpose()

        self.A11 = self.R1 * self.A * self.P1
        self.A12 = self.R1 * self.A * self.P2
        self.A21 = self.R2 * self.A * self.P1
        self.A22 = self.R2 * self.A * self.P2
        super({SCALAR_label}SchurComplement, self).__init__(self.R1.num_rows, self.R1.num_rows)
        kwargs['A'] = self.A22
        # TODO: construct complex solver
        self.invA22 = solverFactory(*args, **kwargs)
        self.invA22.setup()

        self.temporaryMemory = uninitialized((self.A22.shape[0]), dtype={SCALAR})
        self.temporaryMemory2 = uninitialized((self.A22.shape[0]), dtype={SCALAR})
        self.temporaryMemory3 = uninitialized((self.A11.shape[0]), dtype={SCALAR})

    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        self.A21.matvec(x, self.temporaryMemory)
        self.invA22(self.temporaryMemory, self.temporaryMemory2)
        self.A12.matvec(self.temporaryMemory2, self.temporaryMemory3)
        self.A11.matvec(x, y)
        updateScaled(y, self.temporaryMemory3, -1.)
        return 0

    cdef INDEX_t matvec_no_overwrite(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] y) except -1:
        self.A21.matvec(x, self.temporaryMemory)
        self.invA22(self.temporaryMemory, self.temporaryMemory2)
        self.A12.matvec(self.temporaryMemory2, self.temporaryMemory3)
        self.A11.matvec_no_overwrite(x, y)
        updateScaled(y, self.temporaryMemory3, -1.)
        return 0

    def isSparse(self):
        return False

    def to_csr(self):
        raise NotImplementedError()

    def toarray(self):
        from scipy.linalg import inv
        invA22 = inv(self.A22.toarray())
        return self.A11.toarray() - self.A12.toarray().dot(invA22.dot(self.A21.toarray()))

    def to_csr_linear_operator(self):
        raise NotImplementedError()

    def __repr__(self):
        return 'SchurComplement({}, {}x{})'.format(self.A, self.num_rows, self.num_columns)
