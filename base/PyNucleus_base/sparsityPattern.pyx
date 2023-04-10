###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from libc.stdlib cimport malloc, realloc, free
import numpy as np
cimport numpy as np
from . myTypes import INDEX, REAL
from . blas import uninitialized


cdef class sparsityPattern:
    def __init__(self, INDEX_t num_dofs, np.uint16_t initial_length=9):
        cdef:
            INDEX_t i
        self.num_dofs = num_dofs
        self.initial_length = initial_length
        self.nnz = 0
        self.counts = np.zeros((num_dofs), dtype=INDEX)
        self.lengths = initial_length*np.ones((num_dofs), dtype=np.uint16)
        self.indexL = <INDEX_t **>malloc(num_dofs*sizeof(INDEX_t *))
        # reserve initial memory for array of variable column size
        for i in range(num_dofs):
            self.indexL[i] = <INDEX_t *>malloc(self.initial_length*sizeof(INDEX_t))

    cdef inline BOOL_t findIndex(self, INDEX_t I, INDEX_t J):
        cdef:
            uint16_t m, low, high, mid
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

    cdef inline void add(self, INDEX_t I, INDEX_t J):
        cdef:
            INDEX_t m, n
        if not self.findIndex(I, J):
            # J was not present
            # Do we need more space?
            if self.counts[I] == self.lengths[I]:
                self.indexL[I] = <INDEX_t *>realloc(self.indexL[I], (self.lengths[I]+self.initial_length)*sizeof(INDEX_t))
                self.lengths[I] += self.initial_length
            # where should we insert?
            m = self.index
            # move previous indices out of the way
            for n in range(self.counts[I], m, -1):
                self.indexL[I][n] = self.indexL[I][n-1]
            # insert in empty spot
            self.indexL[I][m] = J
            self.counts[I] += 1
            self.nnz += 1

    cdef freeze(self):
        cdef:
            INDEX_t i, j, k, nnz
            INDEX_t[::1] indptr, indices
            np.ndarray[INDEX_t, ndim=1] indptr_mem, indices_mem
        # del self.lengths

        # write indices list of lists into array
        indices_mem = uninitialized((self.nnz), dtype=INDEX)
        indices = indices_mem
        k = 0
        for i in range(self.num_dofs):
            for j in range(self.counts[i]):
                indices[k] = self.indexL[i][j]
                k += 1
            free(self.indexL[i])
        free(self.indexL)

        # fill indptr array
        indptr_mem = uninitialized((self.num_dofs+1), dtype=INDEX)
        indptr = indptr_mem
        nnz = 0
        for i in range(self.num_dofs):
            indptr[i] = nnz
            nnz += self.counts[i]
        indptr[self.num_dofs] = nnz
        return indptr_mem, indices_mem

    def add_python(self, INDEX_t I, INDEX_t J):
        self.add(I, J)

    def freeze_python(self):
        return self.freeze()
