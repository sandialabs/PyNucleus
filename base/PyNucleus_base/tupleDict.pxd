###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cimport numpy as np
from . myTypes cimport INDEX_t, BOOL_t

include "tupleDict_decl_INDEX.pxi"

ctypedef np.uint64_t MEM_t


cdef class indexSet:
    cdef BOOL_t inSet(self, INDEX_t i)
    cdef INDEX_t position(self, INDEX_t i)
    cpdef void fromSet(self, set s)
    cpdef set toSet(self)
    cpdef INDEX_t[::1] toArray(self)
    cdef indexSetIterator getIter(self)
    cdef INDEX_t getNumEntries(self)
    cpdef void empty(self)
    cpdef indexSet union(self, indexSet other)
    cpdef indexSet inter(self, indexSet other)
    cpdef indexSet setminus(self, indexSet other)
    cpdef BOOL_t isSorted(self)


cdef class indexSetIterator:
    cdef:
        indexSet iS
        readonly INDEX_t i
    cdef void setIndexSet(self, indexSet iS)
    cdef void reset(self)
    cdef BOOL_t step(self)


cdef class rangeIndexSet(indexSet):
    cdef:
        INDEX_t start, end, increment


cdef class rangeIndexSetIterator(indexSetIterator):
    cdef:
        INDEX_t k


cdef class arrayIndexSet(indexSet):
    cdef:
        INDEX_t[::1] indexArray


cdef class unsortedArrayIndexSet(arrayIndexSet):
    pass


cdef class arrayIndexSetIterator(indexSetIterator):
    cdef:
        INDEX_t k


cdef class bitArray(indexSet):
    cdef:
        readonly INDEX_t length
        MEM_t* a
        bitArrayIterator it

    cdef void set(self, INDEX_t i)


cdef class bitArrayIterator(indexSetIterator):
    cdef:
        bitArray bA
        INDEX_t k
        size_t n
