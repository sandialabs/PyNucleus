###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . myTypes import INDEX, REAL
import numpy as np
from . myTypes cimport INDEX_t, REAL_t
from . blas import uninitialized
cimport numpy as np
cimport cython
from . linear_operators cimport (LinearOperator,
                                 CSR_LinearOperator,
                                 sparseGraph,
                                 restrictionOp,
                                 prolongationOp)


class combinedOperator(LinearOperator):
    def __init__(self, operators, factors=None):
        if factors is None:
            factors = [1.0]*len(operators)
        for i in range(len(operators)):
            if isinstance(operators[i], tuple):
                operators[i] = sparseGraph(*operators[i])
        self.operators = operators
        self.factors = factors
        self.ndim = 2
        super(combinedOperator, self).__init__(operators[0].shape[0], operators[0].shape[1])

    def sliceRow(self, slice):
        for i in range(len(self.operators)):
            self.operators[i].sliceRow(slice)
        self.shape = self.operators[0].shape

    def sliceColumn(self, slice):
        for i in range(len(self.operators)):
            self.operators[i].sliceColumn(slice)
        self.shape = self.operators[0].shape

    def matvec(self, x):
        y = self.factors[0]*(self.operators[0]*x)
        for op, fac in zip(self.operators[1:], self.factors[1:]):
            y += fac*(op*x)
        return y

    def toCSR(self):
        C = self.factors[0]*self.operators[0].toCSR()
        for op, fac in zip(self.operators[1:], self.factors[1:]):
            C = C + fac*op.toCSR()
        return C

    def __add__(self, other):
        return combinedOperator(self.operators + other.operators,
                                self.factors + other.factors)

    def __rmul__(self, other):
        factors = self.factors[:]
        for i in range(len(factors)):
            factors[i] *= other
        return combinedOperator(self.operators[:], factors)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dropRowsInPlace(S, INDEX_t[::1] rowIndices):
    cdef:
        INDEX_t i, j = 0, k, ri, m = 0, l = 0
        INDEX_t nrows = S.shape[0]
        INDEX_t[::1] indptr = S.indptr
        INDEX_t[::1] indices = S.indices
        REAL_t[::1] data

    if rowIndices.shape[0] == 0:
        return

    if hasattr(S, 'data'):
        data = S.data

    ri = rowIndices[m]  # First row to be dropped
    if hasattr(S, 'data'):
        for i in range(nrows):
            if (i == ri):
                # don't do anything, just select next row that needs to be dropped
                m += 1
                if m < rowIndices.shape[0]:
                    ri = rowIndices[m]
            else:
                for k in range(indptr[i], indptr[i+1]):
                    indices[j] = indices[k]
                    data[j] = data[k]
                    j += 1
                l += 1
                indptr[l] = j
    else:
        for i in range(nrows):
            if (i == ri):
                # don't do anything, just select next row that needs to be dropped
                m += 1
                if m < rowIndices.shape[0]:
                    ri = rowIndices[m]
            else:
                for k in range(indptr[i], indptr[i+1]):
                    indices[j] = indices[k]
                    j += 1
                l += 1
                indptr[l] = j
    S.indices = S.indices[:j]
    S.indptr = S.indptr[:l+1]
    if hasattr(S, 'data'):
        S.data = S.data[:j]
    # if isinstance(S, csr_matrix):
    #     S._shape = (nrows-rowIndices.shape[0], S.shape[1])
    # else:
    S.shape = (nrows-rowIndices.shape[0], S.shape[1])


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dropColsInPlace(S, INDEX_t[::1] col_idx):
    cdef:
        INDEX_t[::1] indptr = S.indptr
        INDEX_t[::1] indices = S.indices
        INDEX_t nrows = S.shape[0]
        INDEX_t ncols = S.shape[1]
        REAL_t[::1] data
        INDEX_t p, i, k, jj, j, z = len(col_idx)-1

    if hasattr(S, 'data'):
        data = S.data
    p = 0
    for i in range(len(indptr)-1):
        k = 0
        for jj in range(indptr[i], indptr[i+1]):
            j = indices[jj]
            while j > col_idx[k] and k < z:
                k += 1
            if j > col_idx[k]:
                indices[p] = j-k-1
                data[p] = data[jj]
                p += 1
            elif j < col_idx[k]:
                indices[p] = j-k
                data[p] = data[jj]
                p += 1
        indptr[i] = p

    for i in range(len(indptr)-1, 0, -1):
        indptr[i] = indptr[i-1]
    indptr[0] = 0
    S.indices = S.indices[:p]
    if hasattr(S, 'data'):
        S.data = S.data[:p]

    # if isinstance(S, csr_matrix):
    #     S._shape = (nrows, ncols-len(col_idx))
    # else:
    S.shape = (nrows, ncols-len(col_idx))


# stolen from scipy
cdef _node_degrees(INDEX_t[::1] ind,
                   INDEX_t[::1] ptr,
                   INDEX_t num_rows):
    """
    Find the degree of each node (matrix row) in a graph represented
    by a sparse CSR or CSC matrix.
    """
    cdef INDEX_t ii, jj
    cdef INDEX_t[::1] degree = np.zeros(num_rows, dtype=INDEX)

    for ii in range(num_rows):
        degree[ii] = ptr[ii + 1] - ptr[ii]
        for jj in range(ptr[ii], ptr[ii + 1]):
            if ind[jj] == ii:
                # add one if the diagonal is in row ii
                degree[ii] += 1
                break
    return degree


# stolen from scipy
cpdef void cuthill_mckee(sparseGraph graph,
                         INDEX_t[::1] order,
                         BOOL_t reverse=False):
    """
    Cuthill-McKee ordering of a sparse symmetric CSR or CSC matrix.
    We follow the original Cuthill-McKee paper and always start the routine
    at a node of lowest degree for each connected component.
    """
    cdef:
        INDEX_t[::1] ind = graph.indices
        INDEX_t[::1] ptr = graph.indptr
        INDEX_t num_rows = graph.num_rows
    cdef INDEX_t N = 0, N_old, level_start, level_end, temp
    cdef INDEX_t zz, ii, jj, kk, ll, level_len
    cdef INDEX_t[::1] reverse_order
    cdef INDEX_t[::1] degree = _node_degrees(ind, ptr, num_rows)
    cdef INDEX_t[::1] inds = np.argsort(degree).astype(INDEX)
    cdef INDEX_t[::1] rev_inds = np.argsort(inds).astype(INDEX)
    cdef INDEX_t[::1] temp_degrees = np.zeros(np.max(degree), dtype=INDEX)
    cdef INDEX_t i, j, seed, temp2

    # loop over zz takes into account possible disconnected graph.
    for zz in range(num_rows):
        if inds[zz] != -1:   # Do BFS with seed=inds[zz]
            seed = inds[zz]
            order[N] = seed
            N += 1
            inds[rev_inds[seed]] = -1
            level_start = N - 1
            level_end = N

            while level_start < level_end:
                for ii in range(level_start, level_end):
                    i = order[ii]
                    N_old = N

                    # add unvisited neighbors
                    for jj in range(ptr[i], ptr[i + 1]):
                        # j is node number connected to i
                        j = ind[jj]
                        if inds[rev_inds[j]] != -1:
                            inds[rev_inds[j]] = -1
                            order[N] = j
                            N += 1

                    # Add values to temp_degrees array for insertion sort
                    level_len = 0
                    for kk in range(N_old, N):
                        temp_degrees[level_len] = degree[order[kk]]
                        level_len += 1

                    # Do insertion sort for nodes from lowest to highest degree
                    for kk in range(1,level_len):
                        temp = temp_degrees[kk]
                        temp2 = order[N_old+kk]
                        ll = kk
                        while (ll > 0) and (temp < temp_degrees[ll-1]):
                            temp_degrees[ll] = temp_degrees[ll-1]
                            order[N_old+ll] = order[N_old+ll-1]
                            ll -= 1
                        temp_degrees[ll] = temp
                        order[N_old+ll] = temp2

                # set next level start and end ranges
                level_start = level_end
                level_end = N

        if N == num_rows:
            break

    if reverse:
        reverse_order = uninitialized((num_rows), dtype=INDEX)
        for i in range(num_rows):
            reverse_order[num_rows-1-i] = order[i]
        for i in range(num_rows):
            order[i] = reverse_order[i]
