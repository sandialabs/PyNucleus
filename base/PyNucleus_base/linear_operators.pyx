###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
cimport numpy as np
from . myTypes import INDEX, REAL, COMPLEX
from . blas cimport gemv
from . blas import uninitialized
from cython.parallel cimport prange, parallel

COMPRESSION = 'gzip'

include "LinearOperator_REAL.pxi"
include "LinearOperator_COMPLEX.pxi"

include "LinearOperatorWrapper_REAL.pxi"
include "LinearOperatorWrapper_COMPLEX.pxi"

include "DenseLinearOperator_REAL.pxi"
include "DenseLinearOperator_COMPLEX.pxi"

include "CSR_LinearOperator_REAL.pxi"
include "CSR_LinearOperator_COMPLEX.pxi"

include "SSS_LinearOperator_REAL.pxi"
include "SSS_LinearOperator_COMPLEX.pxi"

include "DiagonalLinearOperator_REAL.pxi"
include "DiagonalLinearOperator_COMPLEX.pxi"

cdef ENCODE_t MAX_VAL = pow(<ENCODE_t>2, <ENCODE_t>31)

include "IJOperator_REAL.pxi"
include "IJOperator_COMPLEX.pxi"


def transpose(S, inplace=True):
    cdef:
        INDEX_t i, j, c, temp
        INDEX_t nrow = S.shape[0], ncol = S.shape[1]
        INDEX_t[::1] indices_mv = S.indices
        INDEX_t[::1] indptr_mv = S.indptr
        np.ndarray[INDEX_t, ndim=1] newindices = uninitialized((S.nnz), dtype=INDEX)
        np.ndarray[INDEX_t, ndim=1] newindptr = np.zeros((ncol+1), dtype=INDEX)
        INDEX_t[::1] newindices_mv = newindices
        INDEX_t[::1] newindptr_mv = newindptr
        REAL_t[::1] data_mv
        np.ndarray[REAL_t, ndim=1] newdata
        REAL_t[::1] newdata_mv

    if hasattr(S, 'data'):
        data_mv = S.data
        newdata = uninitialized((S.nnz), dtype=REAL)
        newdata_mv = newdata

    # count up occurrences of columns
    for i in range(nrow):
        for j in range(indptr_mv[i], indptr_mv[i+1]):
            c = indices_mv[j]
            newindptr_mv[c+1] += 1
    # make it into indptr array by cumsum
    for j in range(1, ncol+1):
        newindptr_mv[j] += newindptr_mv[j-1]
    # fill new indices and data, use new indptr to index position
    if hasattr(S, 'data'):
        for i in range(nrow):
            for j in range(indptr_mv[i], indptr_mv[i+1]):
                c = indices_mv[j]
                newindices_mv[newindptr_mv[c]] = i
                newdata_mv[newindptr_mv[c]] = data_mv[j]
                newindptr_mv[c] += 1
    else:
        for i in range(nrow):
            for j in range(indptr_mv[i], indptr_mv[i+1]):
                c = indices_mv[j]
                newindices_mv[newindptr_mv[c]] = i
                newindptr_mv[c] += 1
    # set new indptr back by one position
    temp = 0
    for i in range(ncol+1):
        newindptr_mv[i], temp = temp, newindptr_mv[i]
    if inplace:
        S.indices = newindices
        S.indptr = newindptr
        if hasattr(S, 'data'):
            S.data = newdata
        S._shape = (ncol, nrow)
    else:
        # if hasattr(S, 'data'):
        #     return csr_matrix((newdata, newindices, newindptr))
        if isinstance(S, restrictionOp):
            return prolongationOp(newindices, newindptr, ncol, nrow)
        elif isinstance(S, prolongationOp):
            return restrictionOp(newindices, newindptr, ncol, nrow)
        elif isinstance(S, CSR_LinearOperator):
            A = CSR_LinearOperator(newindices, newindptr, newdata)
            A.num_rows = ncol
            A.num_columns = nrow
            return A
        else:
            raise NotImplementedError()


# We are using this for cascade calculation of residuals.
cdef class Triple_Product_Linear_Operator(LinearOperator):
    def __init__(self,
                 LinearOperator A,
                 LinearOperator B,
                 LinearOperator C,
                 REAL_t[::1] temporaryMemory=None,
                 REAL_t[::1] temporaryMemory2=None):
        assert A.num_columns == B.num_rows
        assert B.num_columns == C.num_rows
        super(Triple_Product_Linear_Operator, self).__init__(A.num_rows,
                                                             C.num_columns)
        self.A = A
        self.B = B
        self.C = C
        if temporaryMemory is not None:
            assert temporaryMemory.shape[0] == self.B.num_columns
            self.temporaryMemory = temporaryMemory
        else:
            self.temporaryMemory = uninitialized((self.B.num_columns), dtype=REAL)
        if temporaryMemory2 is not None:
            assert temporaryMemory2.shape[0] == self.A.num_columns
            self.temporaryMemory2 = temporaryMemory2
        else:
            self.temporaryMemory2 = uninitialized((self.A.num_columns), dtype=REAL)

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        self.C(x, self.temporaryMemory)
        self.B(self.temporaryMemory, self.temporaryMemory2)
        self.A(self.temporaryMemory2, y)
        return 0


cdef class split_CSR_LinearOperator(LinearOperator):
    def __init__(self,
                 CSR_LinearOperator A1,
                 CSR_LinearOperator A2):
        LinearOperator.__init__(self,
                                  A1.indptr.shape[0]-1,
                                  A1.indptr.shape[0]-1)
        self.A1 = A1
        self.A2 = A2

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        self.A1.matvec(x, y)
        self.A2.matvec_no_overwrite(x, y)
        return 0

    def to_csr(self):
        return self.A1.to_csr() + self.A2.to_csr()

    def getnnz(self):
        return self.A1.nnz+self.A2.nnz

    nnz = property(fget=getnnz)

    def __repr__(self):
        return '<%dx%d %s with %d stored elements>' % (self.num_rows,
                                                       self.num_columns,
                                                       self.__class__.__name__,
                                                       self.nnz)

    def HDF5write(self, node):
        grpA1 = node.create_group('A1')
        self.A1.HDF5write(grpA1)
        grpA2 = node.create_group('A2')
        self.A2.HDF5write(grpA2)
        node.attrs['type'] = 'split_csr'

    @staticmethod
    def HDF5read(node):
        A1 = CSR_LinearOperator.HDF5read(node['A1'])
        A2 = CSR_LinearOperator.HDF5read(node['A2'])
        return split_CSR_LinearOperator(A1, A2)

    def sort_indices(self):
        self.A1.sort_indices()
        self.A2.sort_indices()

    def get_indices_sorted(self):
        return self.A1.indices_sorted and self.A2.indices_sorted

    indices_sorted = property(fget=get_indices_sorted)


cdef class sparseGraph(LinearOperator):
    def __init__(self, INDEX_t[::1] indices, INDEX_t[::1] indptr,
                 INDEX_t num_rows, INDEX_t num_columns):
        self.indices = indices
        self.indptr = indptr
        super(sparseGraph, self).__init__(num_rows, num_columns)
        self.indices_sorted = False

    def copy(self):
        return sparseGraph(self.indices.copy(),
                              self.indptr.copy(),
                              self.num_rows,
                              self.num_columns)

    def transpose(self):
        newindices = uninitialized(self.nnz, INDEX)
        newindptr = np.zeros(self.num_columns+1, INDEX)
        cdef:
            INDEX_t i, j, c, temp
            INDEX_t[::1] indices_mv = self.indices
            INDEX_t[::1] indptr_mv = self.indptr
            INDEX_t[::1] newindices_mv = newindices
            INDEX_t[::1] newindptr_mv = newindptr
        for i in range(self.num_rows):
            for j in range(indptr_mv[i], indptr_mv[i+1]):
                c = indices_mv[j]
                newindptr_mv[c+1] += 1
        for j in range(1, self.num_columns+1):
            newindptr_mv[j] += newindptr_mv[j-1]
        for i in range(self.num_rows):
            for j in range(indptr_mv[i], indptr_mv[i+1]):
                c = indices_mv[j]
                newindices_mv[newindptr_mv[c]] = i
                newindptr_mv[c] += 1
        temp = 0
        for i in range(self.num_columns+1):
            newindptr_mv[i], temp = temp, newindptr_mv[i]
        self.indices = newindices
        self.indptr = newindptr
        self.num_columns, self.num_rows = self.num_rows, self.num_columns
        self.shape = (self.num_rows, self.num_columns)

    def getnnz(self):
        return self.indptr[self.indptr.shape[0]-1]

    nnz = property(fget=getnnz)

    def getshape(self):
        return (self.num_rows, self.num_columns)

    def setshape(self, val):
        self.num_rows, self.num_columns = val

    shape = property(fget=getshape, fset=setshape)

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

    def sliceRow(self, slice):
        cdef:
            INDEX_t last = 0
            INDEX_t length
            INDEX_t newRowIdx = 0
            INDEX_t i, j, k
            INDEX_t[::1] indices_mv = self.indices
            INDEX_t[::1] indptr_mv = self.indptr
            INDEX_t[::1] ind
        for i in slice:
            length = indptr_mv[i+1]-indptr_mv[i]
            ind = indices_mv[indptr_mv[i]:indptr_mv[i+1]]
            indptr_mv[newRowIdx+1] = last+length
            for k, j in enumerate(range(indptr_mv[newRowIdx], indptr_mv[newRowIdx+1])):
                indices_mv[j] = ind[k]
            newRowIdx += 1
            last += length
        self.indices = self.indices[:last]
        self.indptr = self.indptr[:newRowIdx+1]
        self.num_rows = newRowIdx
        self.shape = (self.num_rows, self.num_columns)

    def sliceColumn(self, slice):
        self.transpose()
        self.sliceRow(slice)
        self.transpose()

    def to_csr(self):
        from scipy.sparse import csr_matrix
        return csr_matrix((np.ones(len(self.indices)), self.indices,
                           self.indptr),
                          shape=(self.num_rows, self.num_columns))

    def todense(self):
        return self.to_csr().todense()

    def __getstate__(self):
        return {'indices': self.indices,
                'indptr': self.indptr,
                'num_rows': self.num_rows,
                'num_columns': self.num_columns}

    def __setstate__(self, value):
        self.__init__(value['indices'], value['indptr'], value['num_rows'], value['num_columns'])

    def __repr__(self):
        return '<%dx%d %s with %d stored elements>' % (self.num_rows, self.num_columns, self.__class__.__name__, self.nnz)

    cdef INDEX_t matvec(self, REAL_t[::1] x, REAL_t[::1] y) except -1:
        cdef:
            INDEX_t[::1] indices_mv = self.indices
            INDEX_t[::1] indptr_mv = self.indptr
            INDEX_t i, j
            REAL_t sum
        for i in range(self.num_rows):
            sum = 0.0
            for j in range(indptr_mv[i], indptr_mv[i+1]):
                sum += x[indices_mv[j]]
            y[i] = sum
        return 0

    cdef public INDEX_t matvec_no_overwrite(self, REAL_t[::1] x, REAL_t[::1] y) except -1:
        cdef:
            INDEX_t[::1] indices_mv = self.indices
            INDEX_t[::1] indptr_mv = self.indptr
            INDEX_t i, j
            REAL_t sum
        for i in range(self.num_rows):
            sum = 0.0
            for j in range(indptr_mv[i], indptr_mv[i+1]):
                sum += x[indices_mv[j]]
            y[i] += sum
        return 0

    def sort_indices(self):
        cdef REAL_t[::1] temp = uninitialized((0), dtype=REAL)
        sort_indices(self.indptr, self.indices, temp)
        self.indices_sorted = True

    def HDF5write(self, node):
        node.create_dataset('indices', data=np.array(self.indices,
                                                     copy=False),
                            compression=COMPRESSION)
        node.create_dataset('indptr', data=np.array(self.indptr,
                                                    copy=False),
                            compression=COMPRESSION)
        node.attrs['num_rows'] = self.num_rows
        node.attrs['num_columns'] = self.num_columns
        node.attrs['type'] = 'sparseGraph'

    @staticmethod
    def HDF5read(node):
        return restrictionOp(np.array(node['indices'], dtype=INDEX),
                             np.array(node['indptr'], dtype=INDEX),
                             node.attrs['num_rows'], node.attrs['num_columns'])


cdef class restrictionOp(sparseGraph):
    def __init__(self, INDEX_t[::1] indices, INDEX_t[::1] indptr,
                 INDEX_t num_rows, INDEX_t num_columns, int NoThreads=1):
        super(restrictionOp, self).__init__(indices, indptr,
                                            num_rows, num_columns)
        self.NoThreads = NoThreads

    cdef INDEX_t matvec(self, REAL_t[::1] x, REAL_t[::1] y) except -1:
        cdef:
            INDEX_t i, j
            REAL_t sum
        if self.NoThreads > 1:
            with nogil, parallel(num_threads=self.NoThreads):
                for i in prange(self.num_rows, schedule='static'):
                    sum = 0.0
                    for j in range(self.indptr[i], self.indptr[i+1]):
                        sum = sum + x[self.indices[j]]
                    y[i] = x[i] + 0.5*sum
        else:
            for i in range(self.num_rows):
                sum = 0.0
                for j in range(self.indptr[i], self.indptr[i+1]):
                    sum += x[self.indices[j]]
                y[i] = x[i] + 0.5*sum
        return 0

    def to_csr(self):
        from scipy.sparse import eye
        return (eye(self.num_rows, self.num_columns, dtype=REAL, format='csr') +
                0.5*super(restrictionOp, self).to_csr())

    def to_csr_linear_operator(self):
        B = self.to_csr()
        C = CSR_LinearOperator(B.indices, B.indptr, B.data)
        C.num_rows = B.shape[0]
        C.num_columns = B.shape[1]
        return C

    def HDF5write(self, node):
        node.create_dataset('indices', data=np.array(self.indices,
                                                     copy=False),
                            compression=COMPRESSION)
        node.create_dataset('indptr', data=np.array(self.indptr,
                                                    copy=False),
                            compression=COMPRESSION)
        node.attrs['num_rows'] = self.num_rows
        node.attrs['num_columns'] = self.num_columns
        node.attrs['type'] = 'restriction'

    @staticmethod
    def HDF5read(node):
        return restrictionOp(np.array(node['indices'], dtype=INDEX),
                             np.array(node['indptr'], dtype=INDEX),
                             node.attrs['num_rows'], node.attrs['num_columns'])

    def restrictMatrix(self, LinearOperator A, LinearOperator Ac):
        multiply_restr(self, A, Ac)


cdef class prolongationOp(sparseGraph):
    def __init__(self, INDEX_t[::1] indices, INDEX_t[::1] indptr,
                 INDEX_t num_rows, INDEX_t num_columns, int NoThreads=1):
        super(prolongationOp, self).__init__(indices, indptr, num_rows, num_columns)
        self.NoThreads = NoThreads

    cdef INDEX_t matvec(self, REAL_t[::1] x, REAL_t[::1] y) except -1:
        cdef:
            INDEX_t i, j
            REAL_t sum
        if self.NoThreads > 1:
            with nogil, parallel(num_threads=self.NoThreads):
                for i in prange(self.num_rows, schedule='static'):
                    sum = 0.0
                    for j in range(self.indptr[i], self.indptr[i+1]):
                        sum = sum + x[self.indices[j]]
                    y[i] = 0.5*sum
                for i in prange(self.num_columns, schedule='static'):
                    y[i] = y[i] + x[i]
        else:
            for i in range(self.num_rows):
                sum = 0.0
                for j in range(self.indptr[i], self.indptr[i+1]):
                    sum += x[self.indices[j]]
                y[i] = 0.5*sum
            for i in range(self.num_columns):
                y[i] += x[i]
        return 0

    def to_csr(self):
        from scipy.sparse import eye
        return (eye(self.num_rows, self.num_columns, dtype=REAL, format='csr') +
                0.5*super(prolongationOp, self).to_csr())

    def to_csr_linear_operator(self):
        B = self.to_csr()
        C = CSR_LinearOperator(B.indices, B.indptr, B.data)
        C.num_rows = B.shape[0]
        C.num_columns = B.shape[1]
        return C

    def HDF5write(self, node):
        node.create_dataset('indices', data=np.array(self.indices,
                                                     copy=False),
                            compression=COMPRESSION)
        node.create_dataset('indptr', data=np.array(self.indptr,
                                                    copy=False),
                            compression=COMPRESSION)
        node.attrs['num_rows'] = self.num_rows
        node.attrs['num_columns'] = self.num_columns
        node.attrs['type'] = 'prolongation'

    @staticmethod
    def HDF5read(node):
        return prolongationOp(np.array(node['indices'], dtype=INDEX),
                              np.array(node['indptr'], dtype=INDEX),
                              node.attrs['num_rows'], node.attrs['num_columns'])


######################################################################
# Matrix restriction R*A*R.T for restrictionOp R

cdef inline REAL_t getEntry_restr(INDEX_t i,
                                    INDEX_t j,
                                    INDEX_t[::1] R_indptr,
                                    INDEX_t[::1] R_indices,
                                    INDEX_t[::1] A_indptr,
                                    INDEX_t[::1] A_indices,
                                    REAL_t[::1] A_data):
    cdef:
        INDEX_t kk, k, mm1, m1, mm2, m2
        REAL_t sum = 0., sumI
    # calculate entry at (i, j) as combination (i, k), (k, m), (m, j)
    # but the last one is transposed, so we have (i, k), (k, m), (j, m)
    # J*A*R^t
    for kk in range(R_indptr[i], R_indptr[i+1]):
        k = R_indices[kk]
        sumI = 0.
        mm1 = A_indptr[k]
        mm2 = R_indptr[j]
        # Find matches between (k, m1) and (j, m2)
        # A*J^t
        while (mm1 < A_indptr[k+1]) and (mm2 < R_indptr[j+1]):
            m1 = A_indices[mm1]
            m2 = R_indices[mm2]
            if m1 < m2:
                mm1 += 1
            elif m1 > m2:
                mm2 += 1
            else:
                sumI += A_data[mm1]*0.5
                mm1 += 1
                mm2 += 1
        # A*I^t
        mm1 = A_indptr[k]
        while (mm1 < A_indptr[k+1]) and (A_indices[mm1] < j):
            mm1 += 1
        if (mm1 < A_indptr[k+1]) and (A_indices[mm1] == j):
            sumI += A_data[mm1]
        sum += 0.5*sumI
    # I*A*R^t
    mm1 = A_indptr[i]
    mm2 = R_indptr[j]
    # Find matches between (i, m1) and (j, m2)
    while (mm1 < A_indptr[i+1]) and (mm2 < R_indptr[j+1]):
        m1 = A_indices[mm1]
        m2 = R_indices[mm2]
        if m1 < m2:
            mm1 += 1
        elif m1 > m2:
            mm2 += 1
        else:
            sum += A_data[mm1]*0.5
            mm1 += 1
            mm2 += 1
    mm1 = A_indptr[i]
    while (mm1 < A_indptr[i+1]) and (A_indices[mm1] < j):
        mm1 += 1
    if (mm1 < A_indptr[i+1]) and (A_indices[mm1] == j):
        sum += A_data[mm1]
    return sum


cdef inline REAL_t getEntryFromD_restr(INDEX_t i,
                                         INDEX_t j,
                                         INDEX_t[::1] R_indptr,
                                         INDEX_t[::1] R_indices,
                                         INDEX_t[::1] A_indptr,
                                         INDEX_t[::1] A_indices,
                                         REAL_t[::1] A_diagonal):
    cdef:
        INDEX_t kk, k, mm1, m1, mm2, m2
        REAL_t sum
    # J*D*J.t
    # Find matches between (k, m1) and (j, m2)
    mm1 = R_indptr[i]
    mm2 = R_indptr[j]
    sum = 0.
    while (mm1 < R_indptr[i+1]) and (mm2 < R_indptr[j+1]):
        m1 = R_indices[mm1]
        m2 = R_indices[mm2]
        if m1 < m2:
            mm1 += 1
        elif m1 > m2:
            mm2 += 1
        else:
            sum += 0.25*A_diagonal[m1]
            mm1 += 1
            mm2 += 1
    # J*D*I.t
    mm1 = R_indptr[i]
    while (mm1 < R_indptr[i+1]) and (R_indices[mm1] < j):
        mm1 += 1
    if (mm1 < R_indptr[i+1]) and (R_indices[mm1] == j):
        sum += 0.5*A_diagonal[j]
    # I*D*J.t
    mm2 = R_indptr[j]
    while (mm2 < R_indptr[j+1]) and (R_indices[mm2] < i):
        mm2 += 1
    if (mm2 < R_indptr[j+1]) and (R_indices[mm2] == i):
        sum += 0.5*A_diagonal[i]
    return sum


def multiply_restr(restrictionOp R, LinearOperator A, LinearOperator Ac):
    cdef:
        INDEX_t i, jj, j, kk, k, mm1, m1, mm2, m2
        REAL_t sum
        INDEX_t[::1] indices = Ac.indices, indptr = Ac.indptr
        REAL_t[::1] data = Ac.data, diagonal
        INDEX_t[::1] R_indptr = R.indptr, R_indices = R.indices
        INDEX_t[::1] A_indptr = A.indptr, A_indices = A.indices
        REAL_t[::1] A_data = A.data, A_diagonal
    # R can be written as I + J, entries of I are 1.0, entries of J are 0.5.
    # If A is in SSS format, it can be written as D + L + L.t,
    # the L part is handled as the A of a CSR matrix.
    for i in range(indptr.shape[0]-1):
        for jj in range(indptr[i], indptr[i+1]):
            j = indices[jj]
            data[jj] += getEntry_restr(i, j,
                                       R_indptr,
                                       R_indices,
                                       A_indptr,
                                       A_indices,
                                       A_data)
    if isinstance(A, SSS_LinearOperator):
        A_diagonal = A.diagonal
        diagonal = Ac.diagonal
        # R*L.t*R.t
        for i in range(indptr.shape[0]-1):
            for jj in range(indptr[i], indptr[i+1]):
                j = indices[jj]
                data[jj] += getEntry_restr(j, i,
                                           R_indptr,
                                           R_indices,
                                           A_indptr,
                                           A_indices,
                                           A_data)
        for i in range(indptr.shape[0]-1):
            # I*D*I.t
            diagonal[i] += A_diagonal[i]
            # R*L*R.t + R*L.t*R.t
            diagonal[i] += 2*getEntry_restr(i, i,
                                            R_indptr,
                                            R_indices,
                                            A_indptr,
                                            A_indices,
                                            A_data)
            for jj in range(indptr[i], indptr[i+1]):
                j = indices[jj]
                data[jj] += getEntryFromD_restr(i, j,
                                                R_indptr,
                                                R_indices,
                                                A_indptr,
                                                A_indices,
                                                A_diagonal)
            diagonal[i] += getEntryFromD_restr(i, i,
                                               R_indptr,
                                               R_indices,
                                               A_indptr,
                                               A_indices,
                                               A_diagonal)


######################################################################
# Matrix restriction R*A*R.T for CSR matrix R

cdef inline REAL_t getEntry(const INDEX_t i,
                              const INDEX_t j,
                              const INDEX_t[::1] R_indptr,
                              const INDEX_t[::1] R_indices,
                              const REAL_t[::1] R_data,
                              const INDEX_t[::1] A_indptr,
                              const INDEX_t[::1] A_indices,
                              const REAL_t[::1] A_data):
    cdef:
        INDEX_t k, kk, mm1, m1, mm2, m2
        REAL_t sum = 0., sumI
    # calculate entry at (i, j) as combination (i, k), (k, m), (m, j)
    # but the last one is transposed, so we have (i, k), (k, m), (j, m)
    # R*A*R^t
    for kk in range(R_indptr[i], R_indptr[i+1]):
        k = R_indices[kk]
        sumI = 0.
        mm1 = A_indptr[k]
        mm2 = R_indptr[j]
        # Find matches between (k, m1) and (j, m2)
        # A*R^t
        while (mm1 < A_indptr[k+1]) and (mm2 < R_indptr[j+1]):
            m1 = A_indices[mm1]
            m2 = R_indices[mm2]
            if m1 < m2:
                mm1 += 1
            elif m1 > m2:
                mm2 += 1
            else:
                sumI += A_data[mm1]*R_data[mm2]
                mm1 += 1
                mm2 += 1
        sum += R_data[kk]*sumI
    return sum


cdef inline REAL_t getEntryFromD(const INDEX_t i,
                                   const INDEX_t j,
                                   const INDEX_t[::1] R_indptr,
                                   const INDEX_t[::1] R_indices,
                                   const REAL_t[::1] R_data,
                                   const INDEX_t[::1] A_indptr,
                                   const INDEX_t[::1] A_indices,
                                   const REAL_t[::1] A_diagonal):
    cdef:
        INDEX_t mm1, m1, mm2, m2
        REAL_t sum
    # R*D*R.t
    # Find matches between (k, m1) and (j, m2)
    mm1 = R_indptr[i]
    mm2 = R_indptr[j]
    sum = 0.
    while (mm1 < R_indptr[i+1]) and (mm2 < R_indptr[j+1]):
        m1 = R_indices[mm1]
        m2 = R_indices[mm2]
        if m1 < m2:
            mm1 += 1
        elif m1 > m2:
            mm2 += 1
        else:
            sum += R_data[mm1]*A_diagonal[m1]*R_data[mm2]
            mm1 += 1
            mm2 += 1
    return sum


def multiply2(CSR_LinearOperator P, LinearOperator A, LinearOperator Ac):
    cdef:
        INDEX_t k, ll, l, ii, i, jj, j
        REAL_t akl, pki, plj
        INDEX_t[::1] P_indptr = P.indptr, P_indices = P.indices
        INDEX_t[::1] A_indptr = A.indptr, A_indices = A.indices
        REAL_t[::1] P_data = P.data, A_data = A.data
        REAL_t[::1] A_diagonal

    for k in range(A_indptr.shape[0]-1):
        for ll in range(A_indptr[k], A_indptr[k+1]):
            l = A_indices[ll]
            akl = A_data[ll]
            for ii in range(P_indptr[k], P_indptr[k+1]):
                i = P_indices[ii]
                pki = P_data[ii]
                for jj in range(P_indptr[l], P_indptr[l+1]):
                    j = P_indices[jj]
                    plj = P_data[jj]
                    Ac.addToEntry(i, j, pki*akl*plj)
    if isinstance(A, SSS_LinearOperator):
        for k in range(A_indptr.shape[0]-1):
            for ll in range(A_indptr[k], A_indptr[k+1]):
                l = A_indices[ll]
                akl = A_data[ll]
                for ii in range(P_indptr[k], P_indptr[k+1]):
                    i = P_indices[ii]
                    pki = P_data[ii]
                    for jj in range(P_indptr[l], P_indptr[l+1]):
                        j = P_indices[jj]
                        plj = P_data[jj]
                        Ac.addToEntry(j, i, pki*akl*plj)

        A_diagonal = A.diagonal
        for k in range(A_indptr.shape[0]-1):
            akl = A_diagonal[k]
            for ii in range(P_indptr[k], P_indptr[k+1]):
                i = P_indices[ii]
                pki = P_data[ii]
                for jj in range(P_indptr[k], P_indptr[k+1]):
                    j = P_indices[jj]
                    plj = P_data[jj]
                    Ac.addToEntry(i, j, pki*akl*plj)


def multiply(CSR_LinearOperator R, LinearOperator A, LinearOperator Ac):
    cdef:
        INDEX_t i, jj, j
        INDEX_t[::1] indices = Ac.indices, indptr = Ac.indptr
        REAL_t[::1] data = Ac.data, diagonal
        INDEX_t[::1] R_indptr = R.indptr, R_indices = R.indices
        INDEX_t[::1] A_indptr = A.indptr, A_indices = A.indices
        REAL_t[::1] R_data = R.data, A_data = A.data, A_diagonal
    # If A is in SSS format, it can be written as D + L + L.t,
    # the L part is handled as the A of a CSR matrix.
    for i in range(indptr.shape[0]-1):
        for jj in range(indptr[i], indptr[i+1]):
            j = indices[jj]
            data[jj] += getEntry(i, j,
                                 R_indptr,
                                 R_indices,
                                 R_data,
                                 A_indptr,
                                 A_indices,
                                 A_data)
    if isinstance(A, SSS_LinearOperator):
        A_diagonal = A.diagonal
        diagonal = Ac.diagonal
        # R*L.t*R.t
        for i in range(indptr.shape[0]-1):
            for jj in range(indptr[i], indptr[i+1]):
                j = indices[jj]
                data[jj] += getEntry(j, i,
                                     R_indptr,
                                     R_indices,
                                     R_data,
                                     A_indptr,
                                     A_indices,
                                     A_data)
        for i in range(indptr.shape[0]-1):
            # R*L*R.t + R*L.t*R.t
            diagonal[i] += 2*getEntry(i, i,
                                      R_indptr,
                                      R_indices,
                                      R_data,
                                      A_indptr,
                                      A_indices,
                                      A_data)
            for jj in range(indptr[i], indptr[i+1]):
                j = indices[jj]
                data[jj] += getEntryFromD(i, j,
                                          R_indptr,
                                          R_indices,
                                          R_data,
                                          A_indptr,
                                          A_indices,
                                          A_diagonal)
            diagonal[i] += getEntryFromD(i, i,
                                         R_indptr,
                                         R_indices,
                                         R_data,
                                         A_indptr,
                                         A_indices,
                                         A_diagonal)


cdef class blockOperator(LinearOperator):
    def __init__(self, list subblocks):
        cdef:
            INDEX_t n, m, i, j, M
        self.blockShape = (len(subblocks), len(subblocks[0]))
        self.blockInptrLeft = np.zeros((self.blockShape[0]+1), dtype=INDEX)
        self.blockInptrRight = np.zeros((self.blockShape[1]+1), dtype=INDEX)
        n = 0
        m = 0
        M = -1
        for i in range(self.blockShape[0]):
            assert len(subblocks[i]) == self.blockShape[1]
            n += subblocks[i][0].shape[0]
            self.blockInptrLeft[i+1] = self.blockInptrLeft[i]+subblocks[i][0].shape[0]
            self.blockInptrRight[0] = 0
            m = 0
            for j in range(self.blockShape[1]):
                m += subblocks[i][j].shape[1]
                self.blockInptrRight[j+1] = self.blockInptrRight[j]+subblocks[i][j].shape[1]
            if M >= 0:
                assert m == M
            else:
                M = m
        super(blockOperator, self).__init__(n, m)
        self.subblocks = subblocks
        self.temp = uninitialized((self.num_rows), dtype=REAL)

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        cdef:
            INDEX_t i, j, k
            LinearOperator lo
        y[:] = 0.
        for i in range(self.blockShape[0]):
            lo = self.subblocks[i][0]
            lo.matvec(x[self.blockInptrRight[0]:self.blockInptrRight[1]],
                      y[self.blockInptrLeft[i]:self.blockInptrLeft[i+1]])
            for j in range(1, self.blockShape[1]):
                lo = self.subblocks[i][j]
                lo.matvec(x[self.blockInptrRight[j]:self.blockInptrRight[j+1]],
                          self.temp[self.blockInptrLeft[i]:self.blockInptrLeft[i+1]])
                for k in range(self.blockInptrLeft[i], self.blockInptrLeft[i+1]):
                    y[k] += self.temp[k]
                # lo.matvec_no_overwrite(x[self.blockInptrRight[j]:self.blockInptrRight[j+1]],
                #                        y[self.blockInptrLeft[i]:self.blockInptrLeft[i+1]])
        return 0

    def toarray(self):
        cdef:
            INDEX_t i, j
        B = uninitialized((self.num_rows, self.num_columns), dtype=REAL)
        for i in range(self.blockShape[0]):
            for j in range(self.blockShape[1]):
                lo = self.subblocks[i][j]
                B[self.blockInptrLeft[i]:self.blockInptrLeft[i+1],
                  self.blockInptrRight[j]:self.blockInptrRight[j+1]] = lo.toarray()
        return B

    def isSparse(self):
        cdef:
            BOOL_t sparse = True
            INDEX_t i, j
        for i in range(self.blockShape[0]):
            for j in range(self.blockShape[1]):
                lo = self.subblocks[i][j]
                sparse &= lo.isSparse()
        return sparse


cdef class blockDiagonalOperator(blockOperator):
    def __init__(self, list diagonalBlocks):
        subblocks = []
        numBlocks = len(diagonalBlocks)
        for i in range(numBlocks):
            d = diagonalBlocks[i]
            row = []
            for j in range(i):
                row.append(nullOperator(d.shape[0], diagonalBlocks[j].shape[1]))
            row.append(d)
            for j in range(i+1, numBlocks):
                row.append(nullOperator(d.shape[0], diagonalBlocks[j].shape[1]))
            subblocks.append(row)
        super(blockDiagonalOperator, self).__init__(subblocks)


cdef class nullOperator(LinearOperator):
    def __init__(self, INDEX_t num_rows, INDEX_t num_columns):
        super(nullOperator, self).__init__(num_rows, num_columns)

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        cdef:
            INDEX_t i
        for i in range(self.num_rows):
            y[i] = 0.
        return 0

    cdef INDEX_t matvec_no_overwrite(self,
                                     REAL_t[::1] x,
                                     REAL_t[::1] y) except -1:
        return 0

    def toarray(self):
        return np.zeros((self.num_rows, self.num_columns), dtype=REAL)

    def get_diagonal(self):
        return np.zeros((min(self.num_rows, self.num_columns)), dtype=REAL)

    diagonal = property(fget=get_diagonal)


cdef class identityOperator(LinearOperator):
    def __init__(self, INDEX_t num_rows, REAL_t alpha=1.0):
        super(identityOperator, self).__init__(num_rows, num_rows)
        self.alpha = alpha

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        cdef:
            INDEX_t i
        for i in range(self.num_rows):
            y[i] = self.alpha*x[i]
        return 0

    cdef INDEX_t matvec_no_overwrite(self,
                                     REAL_t[::1] x,
                                     REAL_t[::1] y) except -1:
        cdef:
            INDEX_t i
        for i in range(self.num_rows):
            y[i] += self.alpha*x[i]
        return 0

    def isSparse(self):
        return True

    def to_csr(self):
        from scipy.sparse import csr_matrix
        indptr = np.arange((self.num_rows+1), dtype=INDEX)
        indices = np.arange((self.num_rows), dtype=INDEX)
        data = self.alpha*np.ones((self.num_rows), dtype=REAL)
        return csr_matrix((data,
                           indices,
                           indptr),
                          shape=self.shape)

    def toarray(self):
        return self.alpha*np.eye(self.num_rows, dtype=REAL)

    def get_diagonal(self):
        return np.ones((self.num_rows), dtype=REAL)

    diagonal = property(fget=get_diagonal)


cdef class blockLowerInverse(blockOperator):
    def __init__(self, subblocks, diagonalInverses):
        if isinstance(subblocks, blockOperator):
            super(blockLowerInverse, self).__init__(subblocks.subblocks)
        else:
            super(blockLowerInverse, self).__init__(subblocks)
        for i in range(self.blockShape[0]):
            for j in range(i+1, self.blockShape[1]):
                assert isinstance(self.subblocks[i][j], nullOperator)
        self.diagonalInverses = diagonalInverses

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        cdef:
            INDEX_t i, j, k
            LinearOperator lo
        y[:] = 0.
        for i in range(self.blockShape[0]):
            for j in range(i):
                lo = self.subblocks[i][j]
                lo.matvec(y[self.blockInptrRight[j]:self.blockInptrRight[j+1]],
                          self.temp[self.blockInptrLeft[i]:self.blockInptrLeft[i+1]])
                for k in range(self.blockInptrLeft[i], self.blockInptrLeft[i+1]):
                    y[k] += self.temp[k]
            for k in range(self.blockInptrLeft[i], self.blockInptrLeft[i+1]):
                self.temp[k] = x[k] - y[k]
            lo = self.diagonalInverses[i]
            lo.matvec(self.temp[self.blockInptrRight[i]:self.blockInptrRight[i+1]],
                      y[self.blockInptrLeft[i]:self.blockInptrLeft[i+1]])
        return 0


cdef class blockUpperInverse(blockOperator):
    def __init__(self, subblocks, diagonalInverses):
        if isinstance(subblocks, blockOperator):
            super(blockUpperInverse, self).__init__(subblocks.subblocks)
        else:
            super(blockUpperInverse, self).__init__(subblocks)
        for i in range(self.blockShape[0]):
            for j in range(i):
                assert isinstance(self.subblocks[i][j], nullOperator)
        self.diagonalInverses = diagonalInverses

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        cdef:
            INDEX_t i, j, k
            LinearOperator lo
        y[:] = 0.
        for i in range(self.blockShape[0]-1, -1, -1):
            for j in range(i+1, self.blockShape[0]):
                lo = self.subblocks[i][j]
                lo.matvec(y[self.blockInptrRight[j]:self.blockInptrRight[j+1]],
                          self.temp[self.blockInptrLeft[i]:self.blockInptrLeft[i+1]])
                for k in range(self.blockInptrLeft[i], self.blockInptrLeft[i+1]):
                    y[k] += self.temp[k]
            for k in range(self.blockInptrLeft[i], self.blockInptrLeft[i+1]):
                self.temp[k] = x[k] - y[k]
            lo = self.diagonalInverses[i]
            lo.matvec(self.temp[self.blockInptrRight[i]:self.blockInptrRight[i+1]],
                      y[self.blockInptrLeft[i]:self.blockInptrLeft[i+1]])
        return 0


cdef class wrapRealToComplex(ComplexLinearOperator):
    def __init__(self, LinearOperator A):
        super(wrapRealToComplex, self).__init__(A.num_rows, A.num_columns)
        self.realA = A
        self.temporaryMemory = uninitialized((A.num_columns), dtype=REAL)
        self.temporaryMemory2 = uninitialized((A.num_rows), dtype=REAL)

    cdef INDEX_t matvec(self,
                        COMPLEX_t[::1] x,
                        COMPLEX_t[::1] y) except -1:
        cdef:
            INDEX_t i
            COMPLEX_t I = 1j
        for i in range(self.num_columns):
            self.temporaryMemory[i] = x[i].real
        self.realA(self.temporaryMemory, self.temporaryMemory2)
        for i in range(self.num_rows):
            y[i] = self.temporaryMemory2[i]
        for i in range(self.num_columns):
            self.temporaryMemory[i] = x[i].imag
        self.realA(self.temporaryMemory, self.temporaryMemory2)
        for i in range(self.num_rows):
            y[i] = y[i]+I*self.temporaryMemory2[i]

        return 0

    def to_csr_linear_operator(self):
        B = self.realA.to_csr()
        return ComplexCSR_LinearOperator(B.indices, B.indptr, np.array(B.data).astype(COMPLEX))

    def to_csr(self):
        return self.to_csr_linear_operator().to_csr()


cdef class wrapRealToComplexCSR(ComplexLinearOperator):
    def __init__(self, CSR_LinearOperator A):
        super(wrapRealToComplexCSR, self).__init__(A.num_rows, A.num_columns)
        self.realA = A

    cdef INDEX_t matvec(self,
                        COMPLEX_t[::1] x,
                        COMPLEX_t[::1] y) except -1:
        cdef:
            INDEX_t i, j
            COMPLEX_t temp
            INDEX_t[::1] indptr = self.realA.indptr, indices = self.realA.indices
            REAL_t[::1] data = self.realA.data
        for i in range(self.num_rows):
            temp = 0.0
            for j in range(indptr[i], indptr[i+1]):
                temp += data[j]*x[indices[j]]
            y[i] = temp
        return 0

    def to_csr_linear_operator(self):
        return ComplexCSR_LinearOperator(self.realA.indices, self.realA.indptr, np.array(self.realA.data).astype(COMPLEX))

    def to_csr(self):
        return self.to_csr_linear_operator().to_csr()


cdef class HelmholtzShiftOperator(ComplexLinearOperator):
    def __init__(self, CSR_LinearOperator S, CSR_LinearOperator M, COMPLEX_t shift):
        super(HelmholtzShiftOperator, self).__init__(S.num_rows, S.num_columns)
        self.M = M
        self.S = S
        self._diagonal = uninitialized((self.num_rows), dtype=COMPLEX)
        self.setShift(shift)

    def setShift(self, COMPLEX_t shift):
        cdef:
            REAL_t[::1] d1, d2
            INDEX_t i
        self.shift = shift
        d1 = self.S.diagonal
        d2 = self.M.diagonal
        for i in range(self.num_rows):
            self._diagonal[i] = d1[i] + shift*d2[i]

    cdef INDEX_t matvec(self,
                        COMPLEX_t[::1] x,
                        COMPLEX_t[::1] y) except -1:
        cdef:
            INDEX_t i, j
            COMPLEX_t temp
            INDEX_t[::1] Sindptr = self.S.indptr, Sindices = self.S.indices
            REAL_t[::1] Sdata = self.S.data
            INDEX_t[::1] Mindptr = self.M.indptr, Mindices = self.M.indices
            REAL_t[::1] Mdata = self.M.data
        for i in range(self.num_rows):
            temp = 0.0
            for j in range(Sindptr[i], Sindptr[i+1]):
                temp += Sdata[j]*x[Sindices[j]]
            y[i] = temp
        for i in range(self.num_rows):
            temp = 0.0
            for j in range(Mindptr[i], Mindptr[i+1]):
                temp += Mdata[j]*x[Mindices[j]]
            y[i] = y[i]+self.shift*temp
        return 0

    def get_diagonal(self):
        return self._diagonal

    def set_diagonal(self, COMPLEX_t[::1] diagonal):
        self._diagonal = diagonal

    diagonal = property(fget=get_diagonal, fset=set_diagonal)

    def to_csr(self):
        return self.S.to_csr()+self.shift*self.M.to_csr()

    def to_csr_linear_operator(self):
        A = self.to_csr()
        return ComplexCSR_LinearOperator(A.indices, A.indptr, A.data)

    def getnnz(self):
        return self.M.nnz+self.S.nnz

    nnz = property(fget=getnnz)


cdef class debugOperator(LinearOperator):
    cdef:
        LinearOperator A
        str name

    def __init__(self, LinearOperator A, str name=""):
        super(debugOperator, self).__init__(A.num_rows, A.num_columns)
        self.A = A
        self.name = name

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        if self.name != "":
            print(self.name)
        print(self.shape, y.shape[0], x.shape[0])
        print('x', np.linalg.norm(x), np.array(x))
        self.A(x, y)
        print('y', np.linalg.norm(y), np.array(y))


cdef class sumMultiplyOperator(LinearOperator):
    cdef:
        public REAL_t[::1] coeffs
        REAL_t[::1] z
        public list ops

    def __init__(self, list ops, REAL_t[::1] coeffs):
        assert len(ops) > 0
        assert len(ops) == coeffs.shape[0]
        shape = ops[0].shape
        super(sumMultiplyOperator, self).__init__(shape[0], shape[1])
        for i in range(1, len(ops)):
            shape2 = ops[i].shape
            assert (shape[0] == shape2[0]) and (shape[1] == shape2[1])
        self.ops = ops
        self.coeffs = coeffs
        self.z = uninitialized((self.ops[0].shape[0]), dtype=REAL)

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        cdef:
            INDEX_t i
            LinearOperator op
        op = self.ops[0]
        op.matvec(x, y)
        scaleScalar(y, self.coeffs[0])
        for i in range(1, self.coeffs.shape[0]):
            op = self.ops[i]
            op.matvec(x, self.z)
            assign3(y, y, 1.0, self.z, self.coeffs[i])
        return 0

    cdef INDEX_t matvec_no_overwrite(self,
                                     REAL_t[::1] x,
                                     REAL_t[::1] y) except -1:
        cdef:
            INDEX_t i
            LinearOperator op
        for i in range(self.coeffs.shape[0]):
            op = self.ops[i]
            op.matvec(x, self.z)
            assign3(y, y, 1.0, self.z, self.coeffs[i])
        return 0

    def toarray(self):
        return sum([c*op.toarray() for c, op in zip(self.coeffs, self.ops)])

    def to_csr(self):
        return sum([c*op.csr() for c, op in zip(self.coeffs, self.ops)])

    def get_diagonal(self):
        return sum([c*np.array(op.diagonal, copy=False) for c, op in zip(self.coeffs, self.ops)])

    diagonal = property(fget=get_diagonal)

    def isSparse(self):
        cdef:
            BOOL_t sparse = True
            INDEX_t i
        for i in range(len(self.ops)):
            sparse &= self.ops[i].isSparse()
        return sparse


cdef class interpolationOperator(sumMultiplyOperator):
    cdef:
        public REAL_t[::1] nodes
        public REAL_t[:, ::1] W, W_prime, W_2prime
        public REAL_t left, right, val
        public INDEX_t derivative

    def __init__(self, list ops, REAL_t[::1] nodes, REAL_t left, REAL_t right):
        cdef:
            INDEX_t i
            INDEX_t numNodes = nodes.shape[0]
        coeffs = np.nan*np.ones((numNodes), dtype=REAL)
        super(interpolationOperator, self).__init__(ops, coeffs)
        self.nodes = nodes
        self.left = left
        self.right = right
        self.val = np.nan
        self.derivative = -1

        for i in range(self.nodes.shape[0]-1):
            assert self.nodes[i] < self.nodes[i+1]

        self.W = np.zeros((numNodes, numNodes), dtype=REAL)
        self.W_prime = np.zeros((numNodes, numNodes), dtype=REAL)
        self.W_2prime = np.zeros((numNodes, numNodes), dtype=REAL)

    def set(self, REAL_t val, int derivative=0):
        cdef:
            INDEX_t k, j, i
            REAL_t[::1] nodes = self.nodes
            INDEX_t numNodes = nodes.shape[0]
            REAL_t[:, ::1] W = self.W
            REAL_t[:, ::1] W_prime, W_2prime
        assert self.left <= val
        assert val <= self.right
        self.val = val
        self.derivative = derivative

        if derivative == 0:
            for i in range(numNodes):
                for j in range(numNodes):
                    if i == j:
                        W[i, j] = 1.
                    else:
                        W[i, j] = 0.

            for k in range(1, numNodes):
                for j in range(numNodes-k):
                    for i in range(numNodes):
                        W[j, i] = (W[j, i]*(val-nodes[k+j]) - W[1+j, i]*(val-nodes[j])) / (nodes[j] - nodes[k+j])
            for i in range(numNodes):
                self.coeffs[i] = W[0, i]
        elif derivative == 1:
            W_prime = self.W_prime
            for i in range(numNodes):
                for j in range(numNodes):
                    if i == j:
                        W[i, j] = 1.
                    else:
                        W[i, j] = 0.
                    W_prime[i, j] = 0.

            for k in range(1, numNodes):
                for j in range(numNodes-k):
                    for i in range(numNodes):
                        W_prime[j, i] = (W_prime[j, i]*(val-nodes[k+j]) + W[j, i] - W_prime[1+j, i]*(val-nodes[j]) - W[1+j, i]) / (nodes[j] - nodes[k+j])
                for j in range(numNodes-k):
                    for i in range(numNodes):
                        W[j, i] = (W[j, i]*(val-nodes[k+j]) - W[1+j, i]*(val-nodes[j])) / (nodes[j] - nodes[k+j])
            for i in range(numNodes):
                self.coeffs[i] = W_prime[0, i]
        elif derivative == 2:
            W_prime = self.W_prime
            W_2prime = self.W_2prime
            for i in range(numNodes):
                for j in range(numNodes):
                    if i == j:
                        W[i, j] = 1.
                    else:
                        W[i, j] = 0.
                    W_prime[i, j] = 0.
                    W_2prime[i, j] = 0.

            for k in range(1, numNodes):
                for j in range(numNodes-k):
                    for i in range(numNodes):
                        W_2prime[j, i] = (W_2prime[j, i]*(val-nodes[k+j]) + 2*W_prime[j, i] - W_2prime[1+j, i]*(val-nodes[j]) - 2*W_prime[1+j, i]) / (nodes[j] - nodes[k+j])
                for j in range(numNodes-k):
                    for i in range(numNodes):
                        W_prime[j, i] = (W_prime[j, i]*(val-nodes[k+j]) + W[j, i] - W_prime[1+j, i]*(val-nodes[j]) - W[1+j, i]) / (nodes[j] - nodes[k+j])
                for j in range(numNodes-k):
                    for i in range(numNodes):
                        W[j, i] = (W[j, i]*(val-nodes[k+j]) - W[1+j, i]*(val-nodes[j])) / (nodes[j] - nodes[k+j])
            for i in range(numNodes):
                self.coeffs[i] = W_2prime[0, i]
        else:
            raise NotImplementedError('derivative {} not implemented'.format(derivative))

    def getNumInterpolationNodes(self):
        return self.nodes.shape[0]

    numInterpolationNodes = property(fget=getNumInterpolationNodes)

    def __repr__(self):
        return '<%dx%d %s with %d interpolation nodes>' % (self.num_rows, self.num_columns, self.__class__.__name__, self.numInterpolationNodes)

    def HDF5write(self, node):
        node.attrs['type'] = 'interpolationOperator'
        node.attrs['left'] = self.left
        node.attrs['right'] = self.right
        node.create_dataset('nodes', data=np.array(self.nodes, copy=False))
        for i in range(len(self.ops)):
            grp = node.create_group(str(i))
            self.ops[i].HDF5write(grp)

    @staticmethod
    def HDF5read(node):
        left = node.attrs['left']
        right = node.attrs['right']
        nodes = np.array(node['nodes'], dtype=REAL)
        ops = []
        for i in range(nodes.shape[0]):
            ops.append(LinearOperator.HDF5read(node[str(i)]))
        return interpolationOperator(ops, nodes, left, right)

    cpdef void assure_constructed(self):
        for i in range(len(self.ops)):
            if isinstance(self.ops[i], delayedConstructionOperator):
                self.ops[i].assure_constructed()


cdef class multiIntervalInterpolationOperator(LinearOperator):
    cdef:
        public list ops
        INDEX_t selected
        readonly REAL_t left
        readonly REAL_t right

    def __init__(self, list intervals, list nodes, list ops):
        shape = ops[0][0].shape
        super(multiIntervalInterpolationOperator, self).__init__(shape[0], shape[1])
        self.ops = []
        self.left = np.inf
        self.right = -np.inf
        for k in range(len(intervals)):
            left, right = intervals[k]
            self.left = min(self.left, left)
            self.right = max(self.right, right)
            self.ops.append(interpolationOperator(ops[k], nodes[k], left, right))
        self.selected = -1

    def get(self):
        if self.selected != -1:
            return self.ops[self.selected].val
        else:
            return np.nan

    def set(self, REAL_t val, BOOL_t derivative=False):
        cdef:
            interpolationOperator op
            INDEX_t k
            REAL_t left, right
        assert self.left <= val, (val, self.left)
        assert val <= self.right, (val, self.right)
        for k in range(len(self.ops)):
            op = self.ops[k]
            left, right = op.left, op.right
            if (left <= val) and (val <= right):
                op.set(val, derivative)
                self.selected = k
                break
        else:
            assert False

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        cdef:
            interpolationOperator op
        assert self.selected != -1
        op = self.ops[self.selected]
        op.matvec(x, y)
        return 0

    def toarray(self):
        assert self.selected != -1
        return self.ops[self.selected].toarray()

    def to_csr(self):
        assert self.selected != -1
        return self.ops[self.selected].to_csr()

    def get_diagonal(self):
        assert self.selected != -1
        return self.ops[self.selected].diagonal

    diagonal = property(fget=get_diagonal)

    def getNumInterpolationNodes(self):
        cdef:
            INDEX_t numInterpolationNodes = 0
            INDEX_t k
            interpolationOperator op
        for k in range(len(self.ops)):
            op = self.ops[k]
            numInterpolationNodes += op.numInterpolationNodes
        return numInterpolationNodes

    numInterpolationNodes = property(fget=getNumInterpolationNodes)

    def getSelectedOp(self):
        assert self.selected != -1
        return self.ops[self.selected]

    def __repr__(self):
        return '<%dx%d %s with %d intervals and %d interpolation nodes>' % (self.num_rows, self.num_columns, self.__class__.__name__, len(self.ops), self.numInterpolationNodes)

    def isSparse(self):
        return self.getSelectedOp().isSparse()

    def HDF5write(self, node):
        node.attrs['type'] = 'multiIntervalInterpolationOperator'
        for i in range(len(self.ops)):
            grp = node.create_group(str(i))
            self.ops[i].HDF5write(grp)

    @staticmethod
    def HDF5read(node):
        numOps = len(node)
        ops = []
        nodes = []
        intervals = []
        for i in range(numOps):
            op = LinearOperator.HDF5read(node[str(i)])
            ops.append(op.ops)
            nodes.append(op.nodes)
            intervals.append((op.left, op.right))
        return multiIntervalInterpolationOperator(intervals, nodes, ops)

    cpdef void assure_constructed(self):
        for i in range(len(self.ops)):
            self.ops[i].assure_constructed()


cdef class delayedConstructionOperator(LinearOperator):
    def __init__(self, INDEX_t numRows, INDEX_t numCols):
        super(delayedConstructionOperator, self).__init__(numRows, numCols)
        self.isConstructed = False
        self.params = {}

    def construct(self):
        raise NotImplementedError()

    cpdef int assure_constructed(self) except -1:
        if not self.isConstructed:
            self.A = self.construct()
            assert self.A.num_rows == self.num_rows, "A.num_rows = {} != self.num_rows = {}".format(self.A.num_rows, self.num_rows)
            assert self.A.num_columns == self.num_columns, "A.num_columns = {} != self.num_columns = {}".format(self.A.num_columns, self.num_columns)
            self.isConstructed = True
            return 0

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        self.assure_constructed()
        self.A.matvec(x, y)
        return 0

    def toarray(self):
        self.assure_constructed()
        return self.A.toarray()

    def to_csr(self):
        self.assure_constructed()
        return self.A.to_csr()

    def get_diagonal(self):
        self.assure_constructed()
        return self.A.diagonal

    diagonal = property(fget=get_diagonal)

    def getnnz(self):
        if hasattr(self.A, 'nnz'):
            return self.A.nnz
        else:
            return

    nnz = property(fget=getnnz)

    def setParams(self, **kwargs):
        for key in kwargs:
            if key not in self.params or self.params[key] != kwargs[key]:
                self.isConstructed = False
                self.params[key] = kwargs[key]

    def isSparse(self):
        self.assure_constructed()
        return self.A.isSparse()

    def HDF5write(self, node):
        self.assure_constructed()
        self.A.HDF5write(node)
