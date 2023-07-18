###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}CSR_LinearOperator({SCALAR_label}LinearOperator):
    def __init__(self,
                 INDEX_t[::1] indices,
                 INDEX_t[::1] indptr,
                 {SCALAR}_t[::1] data,
                 int NoThreads=1):
        {SCALAR_label}LinearOperator.__init__(self,
                                  indptr.shape[0]-1,
                                  indptr.shape[0]-1)
        self.indices = indices
        self.indptr = indptr
        self.data = data
        self.NoThreads = NoThreads
        self.indices_sorted = False

    cdef INDEX_t matvec({SCALAR_label}CSR_LinearOperator self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        spmv(self.indptr, self.indices, self.data, x, y)
        return 0

    cdef void _residual({SCALAR_label}CSR_LinearOperator self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] rhs,
                        {SCALAR}_t[::1] result,
                        BOOL_t simpleResidual=False):
        if simpleResidual:
            assign(result, rhs)
        else:
            spres(self.indptr, self.indices, self.data, x, rhs, result)

    cdef INDEX_t matvec_no_overwrite({SCALAR_label}CSR_LinearOperator self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] y) except -1:
        spmv(self.indptr, self.indices, self.data, x, y, overwrite=False)
        return 0

    cdef INDEX_t matvec_multi({SCALAR_label}CSR_LinearOperator self,
                              {SCALAR}_t[:, ::1] x,
                              {SCALAR}_t[:, ::1] y) except -1:
        cdef:
            INDEX_t i, jj, j, k, numVecs = x.shape[1]
            {SCALAR}_t[::1] temp = uninitialized((numVecs), dtype={SCALAR})
        for i in range(self.num_rows):
            for k in range(numVecs):
                temp[k] = 0.0
            for jj in range(self.indptr[i], self.indptr[i+1]):
                j = self.indices[jj]
                for k in range(numVecs):
                    temp[k] += self.data[jj]*x[j, k]
            for k in range(numVecs):
                y[i, k] = temp[k]
        return 0

    def isSparse(self):
        return True

    def to_csr(self):
        from scipy.sparse import csr_matrix
        return csr_matrix((np.array(self.data, copy=False),
                           np.array(self.indices, copy=False),
                           np.array(self.indptr, copy=False)),
                          shape=self.shape)

    @staticmethod
    def from_csr(matrix):
        A = {SCALAR_label}CSR_LinearOperator(matrix.indices, matrix.indptr, matrix.data)
        A.num_rows = matrix.shape[0]
        A.num_columns = matrix.shape[1]
        return A

    @staticmethod
    def from_dense(matrix, REAL_t tolerance=0.):
        cdef:
            INDEX_t i, j, nnz, k, jj
            {SCALAR}_t[:, ::1] data
            INDEX_t[::1] indptr, indices
            {SCALAR}_t[::1] values
        if isinstance(matrix, {SCALAR_label}Dense_LinearOperator):
            data = matrix.data
        else:
            data = matrix
        indptr = np.zeros((data.shape[0]+1), dtype=INDEX)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if abs(data[i, j]) > tolerance:
                    indptr[i+1] += 1
        for i in range(data.shape[0]):
            indptr[i+1] += indptr[i]
        nnz = indptr[data.shape[0]]
        indices = np.empty((nnz), dtype=INDEX)
        values = np.empty((nnz), dtype={SCALAR})
        for i in range(data.shape[0]):
            k = 0
            for j in range(data.shape[1]):
                if abs(data[i, j]) > tolerance:
                    jj = indptr[i]+k
                    indices[jj] = j
                    values[jj] = data[i, j]
                    k += 1
        A = {SCALAR_label}CSR_LinearOperator(indices, indptr, values)
        A.num_rows = data.shape[0]
        A.num_columns = data.shape[1]
        return A

    def to_csr_linear_operator(self):
        return self

    def toarray(self):
        return self.to_csr().toarray()

    cdef void setEntry({SCALAR_label}CSR_LinearOperator self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        cdef:
            INDEX_t i, low, mid, high
        low = self.indptr[I]
        high = self.indptr[I+1]
        if high-low < 20:
            for i in range(low, high):
                if self.indices[i] == J:
                    self.data[i] = val
                    break
        else:
            while self.indices[low] != J:
                if high-low <= 1:
                    raise IndexError()
                mid = (low+high) >> 1
                if self.indices[mid] <= J:
                    low = mid
                else:
                    high = mid
            self.data[low] = val

    cdef void addToEntry({SCALAR_label}CSR_LinearOperator self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        cdef:
            INDEX_t i, low, mid, high
        low = self.indptr[I]
        high = self.indptr[I+1]
        if high-low < 20:
            for i in range(low, high):
                if self.indices[i] == J:
                    self.data[i] += val
                    break
        else:
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

    cdef {SCALAR}_t getEntry({SCALAR_label}CSR_LinearOperator self, INDEX_t I, INDEX_t J):
        cdef:
            INDEX_t i, low, mid, high
        low = self.indptr[I]
        high = self.indptr[I+1]
        if high-low < 20:
            for i in range(low, high):
                if self.indices[i] == J:
                    return self.data[i]
        else:
            while self.indices[low] != J:
                if high-low <= 1:
                    return 0.
                mid = (low+high) >> 1
                if self.indices[mid] <= J:
                    low = mid
                else:
                    high = mid
            return self.data[low]

    def to_sss(self):
        cdef:
            np.ndarray[INDEX_t, ndim=1] indptr_mem = np.zeros((self.num_rows+1),
                                                              dtype=INDEX)
            INDEX_t[::1] indptr = indptr_mem
            np.ndarray[INDEX_t, ndim=1] indices_mem
            INDEX_t[::1] indices
            np.ndarray[{SCALAR}_t, ndim=1] data_mem, diagonal_mem
            {SCALAR}_t[::1] data, diagonal
            INDEX_t i, jj, j, nnz
        for i in range(self.num_rows):
            for jj in range(self.indptr[i], self.indptr[i+1]):
                j = self.indices[jj]
                if j < i:
                    indptr[i+1] += 1
        for i in range(self.num_rows):
            indptr[i+1] += indptr[i]
        nnz = indptr[indptr.shape[0]-1]
        indices_mem = uninitialized((nnz), dtype=INDEX)
        indices = indices_mem
        data_mem = uninitialized((nnz), dtype={SCALAR})
        data = data_mem
        diagonal_mem = uninitialized((self.num_rows), dtype={SCALAR})
        diagonal = diagonal_mem
        for i in range(self.num_rows):
            for jj in range(self.indptr[i], self.indptr[i+1]):
                j = self.indices[jj]
                if j < i:
                    indices[indptr[i]] = j
                    data[indptr[i]] = self.data[jj]
                    indptr[i] += 1
                elif j == i:
                    diagonal[i] = self.data[jj]
        for i in range(self.num_rows, 0, -1):
            indptr[i] = indptr[i-1]
        indptr[0] = 0
        return SSS_LinearOperator(indices, indptr, data, diagonal)

    def get_diagonal(self):
        cdef:
            INDEX_t i, jj
            np.ndarray[{SCALAR}_t, ndim=1] diag_mem = np.zeros((self.num_rows),
                                                             dtype={SCALAR})
            {SCALAR}_t[::1] d = diag_mem

        for i in range(self.num_rows):
            for jj in range(self.indptr[i], self.indptr[i+1]):
                if self.indices[jj] == i:
                    d[i] = self.data[jj]
                    break
        return diag_mem

    diagonal = property(fget=get_diagonal)

    def getnnz(self):
        return self.indptr[self.indptr.shape[0]-1]

    nnz = property(fget=getnnz)

    def getMemorySize(self):
        return ((self.indptr.shape[0]+self.indices.shape[0])*sizeof(INDEX_t) +
                self.data.shape[0]*sizeof({SCALAR}_t))

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
        node.attrs['type'] = 'csr'
        node.attrs['num_rows'] = self.num_rows
        node.attrs['num_columns'] = self.num_columns

    @staticmethod
    def HDF5read(node):
        B = {SCALAR_label}CSR_LinearOperator(np.array(node['indices'], dtype=INDEX),
                                  np.array(node['indptr'], dtype=INDEX),
                                  np.array(node['data'], dtype={SCALAR}))
        B.num_rows = node.attrs['num_rows']
        B.num_columns = node.attrs['num_columns']
        assert B.indptr.shape[0]-1 == B.num_rows
        return B

    def __getstate__(self):
        return (np.array(self.indices, dtype=INDEX),
                np.array(self.indptr, dtype=INDEX),
                np.array(self.data, dtype={SCALAR}),
                self.num_rows,
                self.num_columns)

    def __setstate__(self, state):
        self.indices = state[0]
        self.indptr = state[1]
        self.data = state[2]
        self.num_rows = state[3]
        self.num_columns = state[4]

    def copy(self):
        data = np.array(self.data, copy=True)
        other = {SCALAR_label}CSR_LinearOperator(self.indices, self.indptr, data)
        return other

    def sort_indices(self):
        sort_indices{SCALAR_label}(self.indptr, self.indices, self.data)
        self.indices_sorted = True

    def isSorted(self):
        """
        Check if column indices are sorted.
        """
        cdef:
            INDEX_t i, nnz, s, p, q
        nnz = self.indptr[self.indptr.shape[0]-1]
        for i in range(self.indptr.shape[0]-1):
            s = self.indptr[i]
            if s ==  nnz:
                continue
            p = self.indices[s]
            for q in self.indices[self.indptr[i]+1:self.indptr[i+1]]:
                if q <= p:
                    return False
                else:
                    p = q
        return True

    def restrictMatrix(self, {SCALAR_label}LinearOperator A, {SCALAR_label}LinearOperator Ac):
        if self.num_rows == Ac.num_rows:
            multiply(self, A, Ac)
        if self.num_columns == Ac.num_rows:
            multiply2(self, A, Ac)

    def scale(self, {SCALAR}_t scaling):
        scaleScalar(self.data, scaling)

    def scaleLeft(self, {SCALAR}_t[::1] scaling):
        cdef:
            INDEX_t i, jj
            {SCALAR}_t d
        assert self.num_rows == scaling.shape[0]
        for i in range(self.num_rows):
            d = scaling[i]
            for jj in range(self.indptr[i], self.indptr[i+1]):
                self.data[jj] *= d

    def scaleRight(self, {SCALAR}_t[::1] scaling):
        cdef:
            INDEX_t i, jj, j
        assert self.num_columns == scaling.shape[0]
        for i in range(self.num_rows):
            for jj in range(self.indptr[i], self.indptr[i+1]):
                j = self.indices[jj]
                self.data[jj] *= scaling[j]

    def setZero(self):
        cdef:
            INDEX_t i
        for i in range(self.data.shape[0]):
            self.data[i] = 0.

    def eliminate_zeros(self):
        cdef:
            INDEX_t[::1] indptrNew = np.zeros((self.num_rows+1), dtype=INDEX)
            INDEX_t[::1] indicesNew
            {SCALAR}_t[::1] dataNew
            INDEX_t i, jj, j, k
            {SCALAR}_t v

        for i in range(self.num_rows):
            indptrNew[i+1] = indptrNew[i]
            for jj in range(self.indptr[i], self.indptr[i+1]):
                v = self.data[jj]
                if v != 0.:
                    indptrNew[i+1] += 1
        indicesNew = uninitialized((indptrNew[self.num_rows]), dtype=INDEX)
        dataNew = uninitialized((indptrNew[self.num_rows]), dtype={SCALAR})
        k = 0
        for i in range(self.num_rows):
            for jj in range(self.indptr[i], self.indptr[i+1]):
                v = self.data[jj]
                if v != 0.:
                    j = self.indices[jj]
                    indicesNew[k] = j
                    dataNew[k] = v
                    k += 1
        self.indptr = indptrNew
        self.indices = indicesNew
        self.data = dataNew

    def sliceRows(self, INDEX_t[::1] rowIndices):
        temp = self.to_csr()
        temp = temp[rowIndices, :]
        return {SCALAR_label}CSR_LinearOperator(temp.indices, temp.indptr, temp.data)

    def sliceColumns(self, INDEX_t[::1] columnIndices):
        temp = self.to_csr()
        temp = temp[:, columnIndices]
        return {SCALAR_label}CSR_LinearOperator(temp.indices, temp.indptr, temp.data)

    def transpose(self):
        return transpose(self, inplace=False)

    cpdef {SCALAR_label}CSR_LinearOperator getBlockDiagonal(self, sparseGraph blocks):
        cdef:
            INDEX_t[::1] indptr = np.zeros((self.num_rows+1), dtype=INDEX)
            INDEX_t[::1] indices
            {SCALAR}_t[::1] data
            INDEX_t blkIdx, ii, i, jj, j, kk, nnz, temp
        for blkIdx in range(blocks.num_rows):
            for ii in range(blocks.indptr[blkIdx], blocks.indptr[blkIdx+1]):
                i = blocks.indices[ii]
                for jj in range(self.indptr[i], self.indptr[i+1]):
                    j = self.indices[jj]
                    for kk in range(blocks.indptr[blkIdx], blocks.indptr[blkIdx+1]):
                        if blocks.indices[kk] == j:
                            indptr[i] += 1
                            break
        nnz = 0
        for i in range(self.num_rows):
            temp = indptr[i]
            indptr[i] = nnz
            nnz += temp
        indptr[self.num_rows] = nnz
        nnz = indptr[self.num_rows]
        indices = uninitialized((nnz), dtype=INDEX)
        data = uninitialized((nnz), dtype={SCALAR})
        for blkIdx in range(blocks.num_rows):
            for ii in range(blocks.indptr[blkIdx], blocks.indptr[blkIdx+1]):
                i = blocks.indices[ii]
                for jj in range(self.indptr[i], self.indptr[i+1]):
                    j = self.indices[jj]
                    for kk in range(blocks.indptr[blkIdx], blocks.indptr[blkIdx+1]):
                        if blocks.indices[kk] == j:
                            indices[indptr[i]] = j
                            data[indptr[i]] = self.data[jj]
                            indptr[i] += 1
                            break
        for i in range(self.num_rows, 0, -1):
            indptr[i] = indptr[i-1]
        indptr[0] = 0
        blockA = {SCALAR_label}CSR_LinearOperator(indices, indptr, data)
        return blockA


cdef BOOL_t sort_indices{SCALAR_label}(INDEX_t[::1] indptr,
                       INDEX_t[::1] indices,
                       {SCALAR}_t[::1] data):
    cdef:
        INDEX_t n, i, jj, j, kk
        {SCALAR}_t d
        BOOL_t wasSorted = True
    n = indptr.shape[0]-1
    if indices.shape[0] == data.shape[0]:
        for i in range(n):
            for jj in range(indptr[i], indptr[i+1]):
                j = indices[jj]
                d = data[jj]
                kk = jj
                while indptr[i] < kk and j < indices[kk-1]:
                    wasSorted = False
                    indices[kk] = indices[kk-1]
                    data[kk] = data[kk-1]
                    kk -= 1
                indices[kk] = j
                data[kk] = d
    else:
        for i in range(n):
            for jj in range(indptr[i], indptr[i+1]):
                j = indices[jj]
                kk = jj
                while indptr[i] < kk and j < indices[kk-1]:
                    wasSorted = False
                    indices[kk] = indices[kk-1]
                    kk -= 1
                indices[kk] = j
    return wasSorted


