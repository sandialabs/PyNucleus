###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


cdef class {SCALAR_label}SSS_LinearOperator({SCALAR_label}LinearOperator):
    """
    Sparse symmetric matrix that saves the lower triangular part.
    """
    def __init__(self,
                 INDEX_t[::1] indices,
                 INDEX_t[::1] indptr,
                 {SCALAR}_t[::1] data,
                 {SCALAR}_t[::1] diagonal,
                 int NoThreads=1):
        {SCALAR_label}LinearOperator.__init__(self,
                                  indptr.shape[0]-1,
                                  indptr.shape[0]-1)
        self.indices = indices
        self.indptr = indptr
        self.data = data
        self.diagonal = diagonal
        self.indices_sorted = False
        self.NoThreads = NoThreads

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec({SCALAR_label}SSS_LinearOperator self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        cdef:
            INDEX_t i, j, k
            {SCALAR}_t temp
        y[:] = 0.
        if self.NoThreads > 1:
            with nogil, parallel(num_threads=self.NoThreads):
                for i in prange(self.num_rows, schedule='static'):
                    temp = self.diagonal[i]*x[i]
                    for j in range(self.indptr[i], self.indptr[i+1]):
                        temp = temp + self.data[j]*x[self.indices[j]]
                    y[i] = temp
                for i in prange(self.num_rows, schedule='static'):
                    for j in range(self.indptr[i], self.indptr[i+1]):
                        y[self.indices[j]] += self.data[j]*x[i]
        else:
            for i in range(self.num_rows):
                temp = self.diagonal[i]*x[i]
                for j in range(self.indptr[i], self.indptr[i+1]):
                    k = self.indices[j]
                    temp += self.data[j]*x[k]
                    y[k] += self.data[j]*x[i]
                y[i] += temp
        return 0

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec_no_overwrite({SCALAR_label}SSS_LinearOperator self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] y) except -1:
        cdef:
            INDEX_t i, j, k
            {SCALAR}_t temp
        for i in range(self.num_rows):
            temp = self.diagonal[i]*x[i]
            for j in range(self.indptr[i], self.indptr[i+1]):
                k = self.indices[j]
                temp += self.data[j]*x[k]
                y[k] += self.data[j]*x[i]
            y[i] += temp
        return 0

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void setEntry({SCALAR_label}SSS_LinearOperator self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        cdef:
            INDEX_t i, low, mid, high
        if I == J:
            self.diagonal[I] = val
        elif I > J:
            low = self.indptr[I]
            high = self.indptr[I+1]
            if high-low < 20:
                for i in range(low, high):
                    if self.indices[i] == J:
                        self.data[i] = val
                        break
            else:
                # This should scale better than previous implementation,
                # if we have a high number of non-zeros per row.
                while self.indices[low] != J:
                    if high-low <= 1:
                        raise IndexError()
                    mid = (low+high) >> 1
                    if self.indices[mid] <= J:
                        low = mid
                    else:
                        high = mid
                self.data[low] = val

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void addToEntry({SCALAR_label}SSS_LinearOperator self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        cdef:
            INDEX_t i, low, mid, high
        if I == J:
            self.diagonal[I] += val
        elif I > J:
            low = self.indptr[I]
            high = self.indptr[I+1]
            if high-low < 20:
                for i in range(low, high):
                    if self.indices[i] == J:
                        self.data[i] += val
                        break
            else:
                # This should scale better than previous implementation,
                # if we have a high number of non-zeros per row.
                while self.indices[low] != J:
                    if high-low <= 1:
                        # raise IndexError()
                        return
                    mid = (low+high) >> 1
                    if self.indices[mid] <= J:
                        low = mid
                    else:
                        high = mid
                self.data[low] += val

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef {SCALAR}_t getEntry({SCALAR_label}SSS_LinearOperator self, INDEX_t I, INDEX_t J):
        cdef:
            INDEX_t low, high, i
        if I == J:
            return self.diagonal[I]
        if I < J:
            I, J = J, I
        low = self.indptr[I]
        high = self.indptr[I+1]
        if high-low < 20:
            for i in range(low, high):
                if self.indices[i] == J:
                    return self.data[i]
        else:
            # This should scale better than previous implementation,
            # if we have a high number of non-zeros per row.
            while self.indices[low] != J:
                if high-low <= 1:
                    return 0.
                mid = (low+high) >> 1
                if self.indices[mid] <= J:
                    low = mid
                else:
                    high = mid
            return self.data[low]

    def isSparse(self):
        return True

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def to_csr(self):
        cdef:
            np.ndarray[INDEX_t, ndim=1] indptr_mem = np.zeros((self.num_rows+1),
                                                              dtype=INDEX)
            INDEX_t[::1] indptr = indptr_mem
            np.ndarray[INDEX_t, ndim=1] indices_mem
            INDEX_t[::1] indices
            np.ndarray[{SCALAR}_t, ndim=1] data_mem
            {SCALAR}_t[::1] data
            INDEX_t i, jj, j, nnz
        for i in range(self.num_rows):
            indptr[i+1] += 1
            for jj in range(self.indptr[i], self.indptr[i+1]):
                j = self.indices[jj]
                indptr[i+1] += 1
                indptr[j+1] += 1
        for i in range(self.num_rows):
            indptr[i+1] += indptr[i]
        nnz = indptr[indptr.shape[0]-1]
        indices_mem = uninitialized((nnz), dtype=INDEX)
        indices = indices_mem
        data_mem = uninitialized((nnz), dtype={SCALAR})
        data = data_mem
        for i in range(self.num_rows):
            indices[indptr[i]] = i
            data[indptr[i]] = self.diagonal[i]
            indptr[i] += 1
            for jj in range(self.indptr[i], self.indptr[i+1]):
                j = self.indices[jj]
                indices[indptr[i]] = j
                data[indptr[i]] = self.data[jj]
                indptr[i] += 1
                indices[indptr[j]] = i
                data[indptr[j]] = self.data[jj]
                indptr[j] += 1
        for i in range(self.num_rows, 0, -1):
            indptr[i] = indptr[i-1]
        indptr[0] = 0
        from scipy.sparse import csr_matrix
        return csr_matrix((data_mem, indices_mem, indptr_mem),
                          shape=self.shape)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def to_lower_csc(self):
        cdef:
            np.ndarray[INDEX_t, ndim=1] indptr_mem = np.zeros((self.num_rows+1),
                                                              dtype=INDEX)
            INDEX_t[::1] indptr = indptr_mem
            np.ndarray[INDEX_t, ndim=1] indices_mem
            INDEX_t[::1] indices
            np.ndarray[{SCALAR}_t, ndim=1] data_mem
            {SCALAR}_t[::1] data
            INDEX_t i, jj, j, nnz
        for i in range(self.num_rows):
            indptr[i+1] += 1
            for jj in range(self.indptr[i], self.indptr[i+1]):
                j = self.indices[jj]
                indptr[j+1] += 1
        for i in range(self.num_rows):
            indptr[i+1] += indptr[i]
        nnz = indptr[indptr.shape[0]-1]
        indices_mem = uninitialized((nnz), dtype=INDEX)
        indices = indices_mem
        data_mem = uninitialized((nnz), dtype={SCALAR})
        data = data_mem
        for i in range(self.num_rows):
            indices[indptr[i]] = i
            data[indptr[i]] = self.diagonal[i]
            indptr[i] += 1
            for jj in range(self.indptr[i], self.indptr[i+1]):
                j = self.indices[jj]
                indices[indptr[j]] = i
                data[indptr[j]] = self.data[jj]
                indptr[j] += 1
        for i in range(self.num_rows, 0, -1):
            indptr[i] = indptr[i-1]
        indptr[0] = 0
        from scipy.sparse import csc_matrix
        return csc_matrix((data_mem, indices_mem, indptr_mem),
                          shape=self.shape)

    def to_csr_linear_operator(self):
        B = self.to_csr()
        return {SCALAR_label}CSR_LinearOperator(B.indices, B.indptr, B.data)

    def to_csc(self):
        A = self.to_csr()
        from scipy.sparse import csc_matrix
        return csc_matrix((A.data, A.indices, A.indptr),
                          shape=self.shape)

    def getnnz(self):
        return self.indptr[-1]+self.num_rows

    nnz = property(fget=getnnz)

    def getMemorySize(self):
        return ((self.indptr.shape[0]+self.indices.shape[0])*sizeof(INDEX_t) +
                (self.data.shape[0]+self.diagonal.shape[0])*sizeof({SCALAR}_t))

    def __repr__(self):
        sizeInMB = self.getMemorySize() >> 20
        if sizeInMB > 100:
            return '<%dx%d %s with %d stored elements, %d MB>' % (self.num_rows,
                                                                  self.num_columns,
                                                                  self.__class__.__name__,
                                                                  self.nnz,
                                                                  sizeInMB)
        else:
            return '<%dx%d %s with %d stored elements>' % (self.num_rows,
                                                           self.num_columns,
                                                           self.__class__.__name__,
                                                           self.nnz)

    def HDF5write(self, node):
        node.create_dataset('indices', data=np.array(self.indices,
                                                     copy=False),
                            compression=COMPRESSION)
        node.create_dataset('indptr', data=np.array(self.indptr,
                                                    copy=False),
                            compression=COMPRESSION)
        node.create_dataset('data', data=np.array(self.data,
                                                  copy=False),
                            compression=COMPRESSION)
        node.create_dataset('diagonal', data=np.array(self.diagonal,
                                                      copy=False),
                            compression=COMPRESSION)
        node.attrs['type'] = 'sss'

    @staticmethod
    def HDF5read(node):
        return {SCALAR_label}SSS_LinearOperator(np.array(node['indices'], dtype=INDEX),
                                  np.array(node['indptr'], dtype=INDEX),
                                  np.array(node['data'], dtype={SCALAR}),
                                  np.array(node['diagonal'], dtype={SCALAR}))

    def __getstate__(self):
        return (np.array(self.indices, dtype=INDEX),
                np.array(self.indptr, dtype=INDEX),
                np.array(self.data, dtype={SCALAR}),
                np.array(self.diagonal, dtype={SCALAR}),
                self.num_rows,
                self.num_columns)

    def __setstate__(self, state):
        self.indices = state[0]
        self.indptr = state[1]
        self.data = state[2]
        self.diagonal = state[3]
        self.num_rows = state[4]
        self.num_columns = state[5]

    def sort_indices(self):
        sort_indices{SCALAR_label}(self.indptr, self.indices, self.data)
        self.indices_sorted = True

    def setZero(self):
        cdef:
            INDEX_t i

        for i in range(self.data.shape[0]):
            self.data[i] = 0.
        for i in range(self.diagonal.shape[0]):
            self.diagonal[i] = 0.

    def copy(self):
        data = np.array(self.data, copy=True)
        diagonal = np.array(self.diagonal, copy=True)
        other = {SCALAR_label}SSS_LinearOperator(self.indices, self.indptr, data, diagonal)
        return other

    def scale(self, {SCALAR}_t scaling):
        cdef:
            INDEX_t i
        for i in range(self.num_rows):
            self.diagonal[i] *= scaling
        for i in range(self.data.shape[0]):
            self.data[i] *= scaling
