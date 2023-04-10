###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}Dense_LinearOperator({SCALAR_label}LinearOperator):
    def __init__(self,
                 {SCALAR}_t[:, ::1] data):
        {SCALAR_label}LinearOperator.__init__(self,
                                  data.shape[0],
                                  data.shape[1])
        self.data = data

    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        gemv(self.data, x, y)
        return 0

    cdef INDEX_t matvec_no_overwrite(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] y) except -1:
        gemv(self.data, x, y, 1.)
        return 0

    cdef INDEX_t matvec_multi(self,
                              {SCALAR}_t[:, ::1] x,
                              {SCALAR}_t[:, ::1] y) except -1:
        cdef:
            INDEX_t i, j, k
            INDEX_t numVecs = y.shape[1]
            {SCALAR}_t[::1] temp = uninitialized((numVecs), dtype={SCALAR})
        y[:, :] = 0.
        for i in range(self.num_rows):
            temp[:] = 0.
            for j in range(self.num_columns):
                for k in range(numVecs):
                    temp[k] += self.data[i, j]*x[j, k]
            for k in range(numVecs):
                y[i, k] = temp[k]
        return 0

    property diagonal:
        def __get__(self):
            cdef INDEX_t i
            diag = uninitialized((min(self.num_rows, self.num_columns)),
                            dtype={SCALAR})
            for i in range(min(self.num_rows, self.num_columns)):
                diag[i] = self.data[i, i]
            return diag

    def scale(self, {SCALAR}_t scaling):
        cdef:
            INDEX_t i, j
        for i in range(self.num_rows):
            for j in range(self.num_columns):
                self.data[i, j] *= scaling

    def isSparse(self):
        return False

    def toarray(self):
        return np.array(self.data, copy=False, dtype={SCALAR})

    cdef {SCALAR}_t getEntry(self, INDEX_t I, INDEX_t J):
        return self.data[I, J]

    cdef void setEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        self.data[I, J] = val

    cdef void addToEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        self.data[I, J] += val

    def HDF5write(self, node):
        node.create_dataset('data', data=np.array(self.data,
                                                  copy=False),
                            compression=COMPRESSION)
        node.attrs['type'] = 'dense{SCALAR_label}'

    @staticmethod
    def HDF5read(node):
        return Dense_LinearOperator(np.array(node['data'], dtype={SCALAR}))

    @staticmethod
    def zeros(INDEX_t num_rows, INDEX_t num_columns):
        return Dense_LinearOperator(np.zeros((num_rows, num_columns), dtype={SCALAR}))

    @staticmethod
    def ones(INDEX_t num_rows, INDEX_t num_columns):
        return Dense_LinearOperator(np.ones((num_rows, num_columns), dtype={SCALAR}))

    @staticmethod
    def empty(INDEX_t num_rows, INDEX_t num_columns):
        return Dense_LinearOperator(uninitialized((num_rows, num_columns), dtype={SCALAR}))

    def transpose(self):
        return {SCALAR_label}Dense_LinearOperator(np.ascontiguousarray(self.toarray().T))

    def getMemorySize(self):
        return self.data.shape[0]*self.data.shape[1]*sizeof({SCALAR}_t)

    def __repr__(self):
        sizeInMB = self.getMemorySize() >> 20
        if sizeInMB > 100:
            return '<%dx%d %s, %d MB>' % (self.num_rows,
                               self.num_columns,
                               self.__class__.__name__,
                               sizeInMB)
        else:
            return '<%dx%d %s>' % (self.num_rows,
                               self.num_columns,
                               self.__class__.__name__)


cdef class {SCALAR_label}Dense_SubBlock_LinearOperator({SCALAR_label}LinearOperator):
    def __init__(self, INDEX_t[::1] I, INDEX_t[::1] J, INDEX_t num_rows, INDEX_t num_columns, {SCALAR}_t[:, :] mem=None):
        cdef:
            INDEX_t i
        if mem is None:
            mem = np.zeros((I.shape[0], J.shape[0]), dtype={SCALAR})
        self.data = mem
        {SCALAR_label}LinearOperator.__init__(self,
                                              num_rows,
                                              num_columns)
        self.lookupI = {}
        self.lookupJ = {}
        for i in range(I.shape[0]):
            self.lookupI[I[i]] = i
        for i in range(J.shape[0]):
            self.lookupJ[J[i]] = i

    cdef {SCALAR}_t getEntry(self, INDEX_t I, INDEX_t J):
        cdef:
            INDEX_t i, j
        i = self.lookupI.get(I, -1)
        j = self.lookupJ.get(J, -1)
        if i >= 0 and j >= 0:
            return self.data[i, j]
        else:
            return 0.

    cdef void setEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        cdef:
            INDEX_t i, j
        i = self.lookupI.get(I, -1)
        j = self.lookupJ.get(J, -1)
        if i >= 0 and j >= 0:
            self.data[i, j] = val

    cdef void addToEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        cdef:
            INDEX_t i, j
        i = self.lookupI.get(I, -1)
        j = self.lookupJ.get(J, -1)
        if i >= 0 and j >= 0:
            self.data[i, j] += val
