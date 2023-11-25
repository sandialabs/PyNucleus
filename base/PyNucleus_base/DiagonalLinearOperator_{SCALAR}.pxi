###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}diagonalOperator({SCALAR_label}LinearOperator):
    def __init__(self, {SCALAR}_t[::1] diagonal):
        {SCALAR_label}LinearOperator.__init__(self, diagonal.shape[0], diagonal.shape[0])
        self.data = diagonal

    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        cdef:
            INDEX_t i
        y[:] = 0.
        for i in range(self.num_rows):
            y[i] = self.data[i]*x[i]
        return 0

    cdef INDEX_t matvec_no_overwrite(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] y) except -1:
        cdef:
            INDEX_t i
        for i in range(self.num_rows):
            y[i] += self.data[i]*x[i]
        return 0

    cdef {SCALAR}_t getEntry(self, INDEX_t I, INDEX_t J):
        if I == J:
            return self.data[I]
        else:
            return 0.

    cdef void setEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        if I == J:
            self.data[I] = val

    cdef void addToEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        if I == J:
            self.data[I] += val
        else:
            raise NotImplementedError()

    def get_diagonal(self):
        return np.array(self.data, copy=False)

    diagonal = property(fget=get_diagonal)

    def isSparse(self):
        return True

    def to_csr(self):
        from scipy.sparse import csr_matrix
        return csr_matrix((self.data,
                           np.arange(self.num_columns, dtype=INDEX),
                           np.arange(self.num_rows+1, dtype=INDEX)),
                          shape=(self.num_rows, self.num_columns))

    def to_csr_linear_operator(self):
        B = self.to_csr()
        return CSR_LinearOperator(B.indices, B.indptr, B.data)

    def __getstate__(self):
        return (np.array(self.data, dtype={SCALAR}),)

    def __setstate__(self, state):
        self.data = state[0]
        self.num_rows = self.data.shape[0]
        self.num_columns = self.data.shape[0]

    def HDF5write(self, node):
        node.create_dataset('data', data=np.array(self.data,
                                                  copy=False),
                            compression=COMPRESSION)
        node.attrs['type'] = 'diagonal'

    @staticmethod
    def HDF5read(node):
        return diagonalOperator(np.array(node['data'], dtype={SCALAR}))


cdef class {SCALAR_label}invDiagonal({SCALAR_label}diagonalOperator):
    def __init__(self, {SCALAR_label}LinearOperator A):
        {SCALAR_label}diagonalOperator.__init__(self, 1./np.array(A.diagonal))
