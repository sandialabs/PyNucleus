###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from libc.stdlib cimport malloc, realloc, free
from libc.stdlib cimport qsort
from . myTypes import INDEX
cimport cython

include "config.pxi"

IF HAVE_MALLOC_H:
    cdef extern from "malloc.h" nogil:
        int malloc_trim(size_t pad)
ELSE:
    cdef int malloc_trim(size_t pad):
        pass

# def return_memory_to_OS():
#     malloc_trim(0)

include "tupleDict_INDEX.pxi"
include "tupleDict_MASK.pxi"


cdef class indexSet:
    cdef BOOL_t inSet(self, INDEX_t i):
        raise NotImplementedError()

    def inSet_py(self, INDEX_t i):
        return self.inSet(i)

    cpdef void fromSet(self, set s):
        raise NotImplementedError()

    cpdef set toSet(self):
        raise NotImplementedError()

    cpdef INDEX_t[::1] toArray(self):
        raise NotImplementedError()

    cdef indexSetIterator getIter(self):
        raise NotImplementedError()

    def __iter__(self):
        return self.getIter()

    def getIter_py(self):
        return self.getIter()

    cdef INDEX_t getNumEntries(self):
        raise NotImplementedError()

    def __len__(self):
        return self.getNumEntries()

    cpdef void empty(self):
        raise NotImplementedError()

    cpdef indexSet union(self, indexSet other):
        raise NotImplementedError()

    cpdef indexSet inter(self, indexSet other):
        raise NotImplementedError()

    cpdef indexSet setminus(self, indexSet other):
        raise NotImplementedError()

    cpdef BOOL_t isSorted(self):
        cdef:
            indexSetIterator it = self.getIter()
            INDEX_t i, j
            BOOL_t sorted = True

        if it.step():
            i = it.i
            while it.step():
                j = it.i
                sorted = sorted & (i < j)
                i = j
        return sorted



cdef class indexSetIterator:
    def __init__(self):
        pass

    cdef void setIndexSet(self, indexSet iS):
        self.iS = iS
        self.reset()

    cdef void reset(self):
        raise NotImplementedError()

    cdef BOOL_t step(self):
        raise NotImplementedError()

    def __next__(self):
        if self.step():
            return self.i
        else:
            raise StopIteration


cdef inline int compareIndices(const void *pa, const void *pb) noexcept nogil:
    cdef:
        INDEX_t a = (<INDEX_t *> pa)[0]
        INDEX_t b = (<INDEX_t *> pb)[0]
    return a > b


cdef class arrayIndexSet(indexSet):
    def __init__(self, INDEX_t[::1] I = None, BOOL_t sorted=False):
        if I is not None:
            if not sorted:
                qsort(&I[0], I.shape[0], sizeof(INDEX_t), compareIndices)
            self.I = I
        else:
            self.I = np.empty((0), dtype=INDEX)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef BOOL_t inSet(self, INDEX_t i):
        cdef:
            INDEX_t low = 0
            INDEX_t high = self.I.shape[0]
            INDEX_t mid
        if high-low < 20:
            for mid in range(low, high):
                if self.I[mid] == i:
                    return True
            return False
        else:
            while self.I[low] != i:
                if high-low <= 1:
                    return False
                mid = (low+high) >> 1
                if self.I[mid] <= i:
                    low = mid
                else:
                    high = mid
            return True

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void fromSet(self, set s):
        cdef:
            INDEX_t i, k
        self.I = np.empty((len(s)), dtype=INDEX)
        k = 0
        for i in s:
            self.I[k] = i
            k += 1
        qsort(&self.I[0], self.I.shape[0], sizeof(INDEX_t), compareIndices)

    cpdef set toSet(self):
        cdef:
            INDEX_t k
            set s = set()
        for k in range(self.I.shape[0]):
            s.add(self.I[k])
        return s

    cpdef INDEX_t[::1] toArray(self):
        return self.I

    cdef indexSetIterator getIter(self):
        return arrayIndexSetIterator(self)

    cdef INDEX_t getNumEntries(self):
        return self.I.shape[0]

    cpdef void empty(self):
        self.I = np.empty((0), dtype=INDEX)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef indexSet union(self, indexSet other_):
        cdef:
            arrayIndexSet other = other_
            INDEX_t l1, l2
            INDEX_t k1, k2, k, i1, i2
            arrayIndexSet newIS

        l1 = self.getNumEntries()
        l2 = other.getNumEntries()

        k1 = 0
        k2 = 0
        k = 0
        while (k1 < l1) and (k2 < l2):
            i1 = self.I[k1]
            i2 = other.I[k2]
            if i1 == i2:
                k += 1
                k1 += 1
                k2 += 1
            elif i1 < i2:
                k += 1
                k1 += 1
            else:
                k += 1
                k2 += 1
        if k1 == l1:
            k += l2-k2
        else:
            k += l1-k1

        newIS = arrayIndexSet(np.empty((k), dtype=INDEX), True)

        k1 = 0
        k2 = 0
        k = 0
        while (k1 < l1) and (k2 < l2):
            i1 = self.I[k1]
            i2 = other.I[k2]
            if i1 == i2:
                newIS.I[k] = i1
                k += 1
                k1 += 1
                k2 += 1
            elif i1 < i2:
                newIS.I[k] = i1
                k += 1
                k1 += 1
            else:
                newIS.I[k] = i2
                k += 1
                k2 += 1
        if k1 == l1:
            for k1 in range(k2, l2):
                newIS.I[k] = other.I[k1]
                k += 1
        else:
            for k2 in range(k1, l1):
                newIS.I[k] = self.I[k2]
                k += 1

        return newIS

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef indexSet inter(self, indexSet other_):
        cdef:
            arrayIndexSet other = other_
            INDEX_t l1, l2
            INDEX_t k1, k2, k, i1 = -1, i2 = -1
            arrayIndexSet newIS

        l1 = self.getNumEntries()
        l2 = other.getNumEntries()

        k1 = 0
        k2 = 0
        k = 0
        while (k1 < l1) and (k2 < l2):
            i1 = self.I[k1]
            i2 = other.I[k2]
            if i1 == i2:
                k += 1
                k1 += 1
                k2 += 1
            elif i1 < i2:
                k1 += 1
            else:
                k2 += 1

        newIS = arrayIndexSet(np.empty((k), dtype=INDEX), True)

        k1 = 0
        k2 = 0
        k = 0
        while (k1 < l1) and (k2 < l2):
            i1 = self.I[k1]
            i2 = other.I[k2]
            if i1 == i2:
                newIS.I[k] = i1
                k += 1
                k1 += 1
                k2 += 1
            elif i1 < i2:
                k1 += 1
            else:
                k2 += 1

        return newIS

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef indexSet setminus(self, indexSet other_):
        cdef:
            arrayIndexSet other = other_
            INDEX_t l1, l2
            INDEX_t k1, k2, k, i1 = -1, i2 = -1
            arrayIndexSet newIS

        l1 = self.getNumEntries()
        l2 = other.getNumEntries()

        k1 = 0
        k2 = 0
        k = 0
        while (k1 < l1) and (k2 < l2):
            i1 = self.I[k1]
            i2 = other.I[k2]
            if i1 == i2:
                k1 += 1
                k2 += 1
            elif i1 < i2:
                k += 1
                k1 += 1
            else:
                k2 += 1

        while (k1 < l1):
            k += 1
            k1 += 1

        newIS = arrayIndexSet(np.empty((k), dtype=INDEX), True)

        k1 = 0
        k2 = 0
        k = 0
        while (k1 < l1) and (k2 < l2):
            i1 = self.I[k1]
            i2 = other.I[k2]
            if i1 == i2:
                k1 += 1
                k2 += 1
            elif i1 < i2:
                newIS.I[k] = i1
                k += 1
                k1 += 1
            else:
                k2 += 1

        while (k1 < l1):
            i1 = self.I[k1]
            newIS.I[k] = i1
            k += 1
            k1 += 1

        return newIS


cdef class unsortedArrayIndexSet(arrayIndexSet):
    def __init__(self, INDEX_t[::1] I = None):
        if I is not None:
            self.I = I
        else:
            self.I = np.empty((0), dtype=INDEX)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef BOOL_t inSet(self, INDEX_t i):
        cdef:
            INDEX_t j
        for j in range(self.I.shape[0]):
            if self.I[j] == i:
                return True
        return False

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void fromSet(self, set s):
        cdef:
            INDEX_t i, k
        self.I = np.empty((len(s)), dtype=INDEX)
        k = 0
        for i in s:
            self.I[k] = i
            k += 1

    cpdef set toSet(self):
        cdef:
            INDEX_t k
            set s = set()
        for k in range(self.I.shape[0]):
            s.add(self.I[k])
        return s

    cpdef INDEX_t[::1] toArray(self):
        return self.I

    cdef indexSetIterator getIter(self):
        return arrayIndexSetIterator(self)

    cdef INDEX_t getNumEntries(self):
        return self.I.shape[0]

    cpdef void empty(self):
        self.I = np.empty((0), dtype=INDEX)

    cpdef indexSet union(self, indexSet other):
        raise NotImplementedError()

    cpdef indexSet inter(self, indexSet other):
        raise NotImplementedError()

    cpdef indexSet setminus(self, indexSet other):
        raise NotImplementedError()


cdef class arrayIndexSetIterator(indexSetIterator):
    def __init__(self, arrayIndexSet aIS=None):
        if aIS is not None:
            self.setIndexSet(aIS)

    cdef void reset(self):
        self.k = -1

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef BOOL_t step(self):
        cdef:
            arrayIndexSet aIS = self.iS
        self.k += 1
        if self.k < aIS.I.shape[0]:
            self.i = aIS.I[self.k]
            return True
        else:
            return False


cdef class bitArray(indexSet):
    def __init__(self, INDEX_t hintMaxLength=1, INDEX_t maxElement=0):
        self.length = max(hintMaxLength, maxElement/(sizeof(MEM_t)*8)+1)
        self.a = <MEM_t *>malloc(self.length*sizeof(MEM_t))
        for j in range(self.length):
            self.a[j] = 0

    @cython.cdivision(True)
    cdef void set(self, INDEX_t i):
        cdef:
            INDEX_t k = i/(sizeof(MEM_t)*8)
            INDEX_t n = i-k*sizeof(MEM_t)*8
            INDEX_t j, l
            MEM_t one = 1
        if k >= self.length:
            l = self.length
            self.length = k+1
            self.a = <MEM_t *>realloc(self.a, self.length * sizeof(MEM_t))
            for j in range(l, self.length):
                self.a[j] = 0
        self.a[k] |= one << n

    def set_py(self, INDEX_t i):
        self.set(i)

    @cython.cdivision(True)
    cdef BOOL_t inSet(self, INDEX_t i):
        cdef:
            INDEX_t k = i/(sizeof(MEM_t)*8)
            INDEX_t n = i-k*sizeof(MEM_t)*8
        if 0 <= k < self.length:
            return (self.a[k] >> n) & 1
        else:
            return False

    cpdef set toSet(self):
        cdef:
            set s = set()
            indexSetIterator it = self.getIter()
        while it.step():
            s.add(it.i)
        return s

    cpdef void fromSet(self, set s):
        cdef:
            INDEX_t i
        self.empty()
        for i in s:
            self.set(i)

    cdef INDEX_t getNumEntries(self):
        cdef:
            INDEX_t k, c = 0
            MEM_t v
        for k in range(self.length):
            v = self.a[k]
            for _ in range(sizeof(MEM_t)*8):
                if v & 1:
                    c += 1
                v = v >> 1
        return c

    cpdef void empty(self):
        cdef:
            INDEX_t j
        for j in range(self.length):
            self.a[j] = 0

    def __dealloc__(self):
        free(self.a)

    cdef indexSetIterator getIter(self):
        return bitArrayIterator(self)

    cpdef indexSet union(self, indexSet other_):
        cdef:
            bitArray other = other_
            bitArray bA = bitArray(max(self.length, other.length))
            INDEX_t k

        for k in range(min(self.length, other.length)):
            bA.a[k] = self.a[k] | other.a[k]
        if self.length > other.length:
            for k in range(other.length, self.length):
                bA.a[k] = self.a[k]
        else:
            for k in range(self.length, other.length):
                bA.a[k] = other.a[k]
        return bA

    cpdef indexSet inter(self, indexSet other_):
        cdef:
            bitArray other = other_
            bitArray bA = bitArray(min(self.length, other.length))
            INDEX_t k

        for k in range(min(self.length, other.length)):
            bA.a[k] = self.a[k] & other.a[k]
        return bA


cdef class bitArrayIterator(indexSetIterator):
    def __init__(self, bitArray bA=None):
        if bA is not None:
            self.setIndexSet(bA)

    cdef void reset(self):
        self.k = -1
        self.n = sizeof(MEM_t)*8-1
        self.i = -1

    cdef BOOL_t step(self):
        cdef:
            bitArray bA = self.iS
            INDEX_t k0, n0, k, n
            MEM_t v

        if self.n == sizeof(MEM_t)*8-1:
            k0 = self.k+1
            n0 = 0
        else:
            k0 = self.k
            n0 = self.n+1
        v = bA.a[k0]
        v = v >> n0
        for n in range(n0, sizeof(MEM_t)*8):
            if v & 1:
                self.k = k0
                self.n = n
                self.i = self.k*sizeof(MEM_t)*8+self.n
                return True
            v = v >> 1

        for k in range(k0+1, bA.length):
            v = bA.a[k]
            for n in range(sizeof(MEM_t)*8):
                if v & 1:
                    self.k = k
                    self.n = n
                    self.i = self.k*sizeof(MEM_t)*8+self.n
                    return True
                v = v >> 1
        return False
