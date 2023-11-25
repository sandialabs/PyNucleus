###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class tupleDict{VALUE}:
    def __init__(self,
                 INDEX_t num_dofs,
                 {LENGTH_t} initial_length=0,
                 {LENGTH_t} length_inc=3,
                 BOOL_t deleteHits=True,
                 BOOL_t logicalAndHits=False):
        cdef:
            INDEX_t i
        self.num_dofs = num_dofs
        self.initial_length = initial_length
        self.length_inc = length_inc
        self.nnz = 0
        self.counts = np.zeros((num_dofs), dtype={LENGTH_dtype})
        self.lengths = initial_length*np.ones((num_dofs), dtype={LENGTH_dtype})
        self.indexL = <INDEX_t **>malloc(num_dofs*sizeof(INDEX_t *))
        self.vals = <{VALUE_t} **>malloc(num_dofs*sizeof({VALUE_t} *))
        # reserve initial memory for array of variable column size
        for i in range(num_dofs):
            self.indexL[i] = <INDEX_t *>malloc(self.initial_length * sizeof(INDEX_t))
            self.vals[i] = <{VALUE_t} *>malloc(self.initial_length * sizeof({VALUE_t}))
        self.deleteHits = deleteHits
        self.logicalAndHits = logicalAndHits
        # self.invalid = np.iinfo({VALUE_dtype}).max
        self.invalid = {INVALID}
        self.invalidIndex = np.iinfo({LENGTH_dtype}).max

    cdef INDEX_t getSizeInBytes(self):
        cdef:
            INDEX_t s, i, l
        s = self.num_dofs * (2*sizeof({LENGTH_t}) + sizeof(INDEX_t*) + sizeof({VALUE_t}*))
        l = 0
        for i in range(self.num_dofs):
            l += self.lengths[i]
        s += l*(sizeof(INDEX_t)+sizeof({VALUE_t}))
        return s

    cdef inline BOOL_t findIndex(self, INDEX_t I, INDEX_t J):
        cdef:
            {LENGTH_t} m, low, high, mid
            INDEX_t K

        if self.counts[I] < 20:
            for m in range(self.counts[I]):
                K = self.indexL[I][m]
                if K == J:
                    self.index = m
                    return True
                elif K > J:
                    self.index = m
                    return False
            else:
                self.index = self.counts[I]
                return False
        else:
            low = 0
            high = self.counts[I]
            while self.indexL[I][low] != J:
                if high-low <= 1:
                    if self.indexL[I][low] > J:
                        self.index = low
                    else:
                        self.index = low+1
                    return False
                mid = (low+high) >> 1
                if self.indexL[I][mid] <= J:
                    low = mid
                else:
                    high = mid
            self.index = low
            return True

    cdef inline void increaseSize(self, INDEX_t I, {LENGTH_t} increment):
        self.lengths[I] += increment
        self.indexL[I] = <INDEX_t *>realloc(self.indexL[I], (self.lengths[I]) * sizeof(INDEX_t))
        self.vals[I] = <{VALUE_t} *>realloc(self.vals[I], (self.lengths[I]) * sizeof({VALUE_t}))

    cdef {VALUE_t} enterValue(self, const INDEX_t[::1] e, {VALUE_t} val):
        cdef:
            INDEX_t m, n, I = e[0], J = e[1]

        if self.findIndex(I, J):  # J is already present
            m = self.index
            if self.deleteHits:
                val = self.vals[I][m]
                for n in range(m+1, self.counts[I]):
                    self.indexL[I][n-1] = self.indexL[I][n]
                    self.vals[I][n-1] = self.vals[I][n]
                self.counts[I] -= 1
                self.nnz -= 1
            elif self.logicalAndHits:
                self.vals[I][m] |= val
                val = self.vals[I][m]
            else:
                val = self.vals[I][m]
            return val
        else:
            # J was not present
            m = self.index
            # Do we need more space?
            if self.counts[I] == self.lengths[I]:
                self.increaseSize(I, self.length_inc)
            # move previous indices out of the way
            for n in range(self.counts[I], m, -1):
                self.indexL[I][n] = self.indexL[I][n-1]
                self.vals[I][n] = self.vals[I][n-1]
            # insert in empty spot
            self.indexL[I][m] = J
            self.vals[I][m] = val
            self.counts[I] += 1
            self.nnz += 1
            return val

    cdef {VALUE_t} removeValue(self, const INDEX_t[::1] e):
        cdef:
            INDEX_t m, n, I = e[0], J = e[1]
            {VALUE_t} val

        if self.findIndex(I, J):  # J is already present
            m = self.index
            val = self.vals[I][m]
            for n in range(m+1, self.counts[I]):
                self.indexL[I][n-1] = self.indexL[I][n]
                self.vals[I][n-1] = self.vals[I][n]
            self.counts[I] -= 1
            self.nnz -= 1
            return val
        return self.invalid

    cpdef {VALUE_t} enterValue_py(self, const INDEX_t[::1] e, {VALUE_t} val):
        self.enterValue(e, val)

    cpdef {VALUE_t} removeValue_py(self, const INDEX_t[::1] e):
        self.removeValue(e)

    cdef {VALUE_t} getValue(self, const INDEX_t[::1] e):
        cdef:
            INDEX_t m
        if self.findIndex(e[0], e[1]):  # J is already present
            return self.vals[e[0]][self.index]
        else:
            return self.invalid

    def __getitem__(self, INDEX_t[::1] edge):
        return self.getValue(edge)

    def __dealloc__(self):
        cdef:
            INDEX_t i
        for i in range(self.num_dofs):
            free(self.indexL[i])
            free(self.vals[i])
        free(self.indexL)
        free(self.vals)
        malloc_trim(0)

    cdef void startIter(self):
        self.i = 0
        while self.i < self.num_dofs and self.counts[self.i] == 0:
            self.i += 1
        self.jj = 0

    cdef BOOL_t next(self, INDEX_t[::1] e, {VALUE_t} * val):
        cdef:
            INDEX_t i = self.i, jj = self.jj, j
        if i < self.num_dofs:
            j = self.indexL[i][jj]
            val[0] = self.vals[i][jj]
        else:
            return False
        e[0] = i
        e[1] = j
        if jj < self.counts[i]-1:
            self.jj += 1
        else:
            self.jj = 0
            i += 1
            while i < self.num_dofs and self.counts[i] == 0:
                i += 1
            self.i = i
        return True

    cdef tuple getData(self):
        cdef:
            INDEX_t[::1] indexL
            {VALUE_t}[::1] vals
            INDEX_t i, j, k
        indexL = np.empty((self.nnz), dtype=INDEX)
        vals = np.empty((self.nnz), dtype={VALUE_dtype})
        k = 0
        for i in range(self.num_dofs):
            for j in range(self.counts[i]):
                indexL[k] = self.indexL[i][j]
                vals[k] = self.vals[i][j]
                k += 1
        return indexL, vals

    def __getstate__(self):
        indexL, vals = self.getData()
        return (self.num_dofs, self.length_inc, self.deleteHits, self.logicalAndHits, np.array(self.counts), np.array(indexL), np.array(vals))

    def __setstate__(self, state):
        cdef:
            INDEX_t[::1] indexL = state[5]
            {VALUE_t}[::1] vals = state[6]
            INDEX_t i, j, k
        self.__init__(state[0], 0, state[1], state[2], state[3])
        self.lengths = state[4]
        k = 0
        for i in range(self.num_dofs):
            self.counts[i] = self.lengths[i]
            self.indexL[i] = <INDEX_t *>malloc(self.lengths[i] * sizeof(INDEX_t))
            self.vals[i] = <{VALUE_t} *>malloc(self.lengths[i] * sizeof({VALUE_t}))
            for j in range(self.counts[i]):
                self.indexL[i][j] = indexL[k]
                self.vals[i][j] = vals[k]
                k += 1
        self.nnz = k

    cpdef void merge(self, tupleDict{VALUE} other):
        cdef:
            INDEX_t[::1] e = np.empty((2), dtype=INDEX)
            {VALUE_t} val
            INDEX_t i
        assert self.num_dofs == other.num_dofs
        other.startIter()
        while other.next(e, &val):
            self.enterValue(e, val)

    cpdef void mergeData(self, {LENGTH_t}[::1] counts, INDEX_t[::1] indexL, {VALUE_t}[::1] vals):
        cdef:
            INDEX_t[::1] e = np.empty((2), dtype=INDEX)
            INDEX_t i, k
        assert self.num_dofs == counts.shape[0]
        k = 0
        for i in range(self.num_dofs):
            e[0] = i
            for j in range(counts[i]):
                e[1] = indexL[k]
                self.enterValue(e, vals[k])
                k += 1

    def isCorrect(self):
        cdef:
            INDEX_t i, j
        for i in range(self.num_dofs):
            for j in range(self.counts[i]-1):
                if self.indexL[i][j] > self.indexL[i][j+1]:
                    print(i, j, self.indexL[i][j], self.indexL[i][j+1])
                    return False
        return True

    def toDict(self):
        cdef:
            INDEX_t e[2]
            {VALUE_t} val
            dict d = {}
        self.startIter()
        while self.next(e, &val):
            d[(e[0], e[1])] = val
        return d
