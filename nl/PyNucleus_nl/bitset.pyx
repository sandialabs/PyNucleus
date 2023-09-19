###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from libc.stdlib cimport malloc, realloc, free

include "malloc.pxi"


cdef str MASK2Str(MASK_t a):
    s = ''
    for i in range(a.size()):
        s+= str(int(a[i]))
    return s


cdef class tupleDictMASK:
    def __init__(self,
                 INDEX_t num_dofs,
                 np.uint16_t initial_length=0,
                 np.uint16_t length_inc=3,
                 BOOL_t deleteHits=True,
                 BOOL_t logicalAndHits=False):
        cdef:
            INDEX_t i
            MASK_t MASK_INVALID
        self.num_dofs = num_dofs
        self.initial_length = initial_length
        self.length_inc = length_inc
        self.nnz = 0
        self.counts = np.zeros((num_dofs), dtype=np.uint16)
        self.lengths = initial_length*np.ones((num_dofs), dtype=np.uint16)
        self.indexL = <INDEX_t **>malloc(num_dofs*sizeof(INDEX_t *))
        self.vals = <MASK_t **>malloc(num_dofs*sizeof(MASK_t *))
        # reserve initial memory for array of variable column size
        for i in range(num_dofs):
            self.indexL[i] = <INDEX_t *>malloc(self.initial_length *
                                               sizeof(INDEX_t))
            self.vals[i] = <MASK_t *>malloc(self.initial_length *
                                               sizeof(MASK_t))
        self.deleteHits = deleteHits
        self.logicalAndHits = logicalAndHits
        MASK_INVALID.set()
        self.invalid = MASK_INVALID
        self.invalidIndex = np.iinfo(np.uint16).max

    cdef INDEX_t getSizeInBytes(self):
        cdef:
            INDEX_t s, i, l
        s = self.num_dofs * (2*sizeof(np.uint16_t) + sizeof(INDEX_t*) + sizeof(MASK_t*))
        l = 0
        for i in range(self.num_dofs):
            l += self.lengths[i]
        s += l*(sizeof(INDEX_t)+sizeof(MASK_t))
        return s

    cdef inline BOOL_t findIndex(self, INDEX_t I, INDEX_t J):
        cdef:
            np.uint16_t m, low, high, mid
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

    cdef inline void increaseSize(self, INDEX_t I, np.uint16_t increment):
        self.lengths[I] += increment
        self.indexL[I] = <INDEX_t *>realloc(self.indexL[I],
                                            (self.lengths[I]) *
                                            sizeof(INDEX_t))
        self.vals[I] = <MASK_t *>realloc(self.vals[I],
                                            (self.lengths[I]) *
                                            sizeof(MASK_t))

    cdef MASK_t enterValue(self, const INDEX_t[::1] e, MASK_t val):
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

    cdef MASK_t removeValue(self, const INDEX_t[::1] e):
        cdef:
            INDEX_t m, n, I = e[0], J = e[1]
            MASK_t val

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

    cdef MASK_t getValue(self, const INDEX_t[::1] e):
        cdef:
            INDEX_t m
        if self.findIndex(e[0], e[1]):  # J is already present
            return self.vals[e[0]][self.index]
        else:
            return self.invalid

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

    cdef BOOL_t next(self, INDEX_t[::1] e, MASK_t * val):
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
