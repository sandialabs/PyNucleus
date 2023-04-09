###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}IJOperator({SCALAR_label}LinearOperator):
    def __init__(self, INDEX_t numRows, INDEX_t numCols):
        super({SCALAR_label}IJOperator, self).__init__(numRows, numCols)
        self.entries = {}

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void setEntry({SCALAR_label}IJOperator self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        cdef:
            ENCODE_t hv = MAX_VAL*<ENCODE_t>I+<ENCODE_t>J
        self.entries[hv] = val

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void addToEntry({SCALAR_label}IJOperator self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        cdef:
            ENCODE_t hv = MAX_VAL*<ENCODE_t>I+<ENCODE_t>J
            REAL_t oldVal
        oldVal = self.entries.pop(hv, 0.)
        self.entries[hv] = oldVal+val

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef {SCALAR}_t getEntry({SCALAR_label}IJOperator self, INDEX_t I, INDEX_t J):
        cdef:
            ENCODE_t hv = MAX_VAL*<ENCODE_t>I+<ENCODE_t>J
        return self.entries.get(hv, 0.)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getData(self):
        cdef:
            INDEX_t[::1] I, J
            {SCALAR}_t[::1] data
            INDEX_t numEntries = len(self.entries)
            ENCODE_t hv
            INDEX_t k = 0
        I = np.empty((numEntries), dtype=INDEX)
        J = np.empty((numEntries), dtype=INDEX)
        data = np.empty((numEntries), dtype={SCALAR})
        for hv in self.entries:
            I[k] = hv // MAX_VAL
            J[k] = hv % MAX_VAL
            data[k] = self.entries[hv]
            k += 1
        return (np.array(I, copy=False),
                np.array(J, copy=False),
                np.array(data, copy=False))

    def to_csr_linear_operator(self):
        cdef:
            INDEX_t[::1] indptr = np.zeros((self.num_rows+1), dtype=INDEX)
            INDEX_t[::1] indices = uninitialized((len(self.entries)), dtype=INDEX)
            REAL_t[::1] data = uninitialized((len(self.entries)), dtype=REAL)
            ENCODE_t hv
            INDEX_t I
        for hv in self.entries:
            I = hv // MAX_VAL
            indptr[I+1] += 1
        for I in range(self.num_rows):
            indptr[I+1] += indptr[I]
        for hv in self.entries:
            I = hv // MAX_VAL
            indices[indptr[I]] = hv % MAX_VAL
            data[indptr[I]] = self.entries[hv]
            indptr[I] += 1
        for I in range(self.num_rows-1, 0, -1):
            indptr[I] = indptr[I-1]
        indptr[0] = 0
        A = {SCALAR_label}CSR_LinearOperator(indices, indptr, data)
        A.num_columns = self.num_columns
        return A
