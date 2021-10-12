###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy


cdef enum:
    INDEX_SIZE = sizeof(INDEX_t)


cdef class intTuple:
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set(self, INDEX_t * t, int size):
        self.size = size
        self.entries = <INDEX_t *>malloc(size*INDEX_SIZE)
        memcpy(&self.entries[0], &t[0], size*INDEX_SIZE)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void assign(self, INDEX_t * t):
        memcpy(&self.entries[0], &t[0], self.size*INDEX_SIZE)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void assignNonOwning(self, INDEX_t * t):
        self.entries = t

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void get(self, INDEX_t * t):
        memcpy(&t[0], &self.entries[0], self.size*INDEX_SIZE)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    cdef intTuple create(INDEX_t[::1] t):
        cdef:
            intTuple tt = intTuple()
        tt.set(&t[0], t.shape[0])
        return tt

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    cdef intTuple createNonOwning(INDEX_t[::1] t):
        cdef:
            intTuple tt = intTuple()
        tt.size = t.shape[0]
        tt.entries = &t[0]
        return tt

    @staticmethod
    def createPy(INDEX_t[::1] t):
        return intTuple.create(t)

    @staticmethod
    cdef intTuple create2(INDEX_t a, INDEX_t b):
        cdef:
            intTuple t = intTuple()
        t.size = 2
        t.entries = <INDEX_t *>malloc(2*INDEX_SIZE)
        t.entries[0] = a
        t.entries[1] = b
        return t

    @staticmethod
    def create2Py(INDEX_t a, INDEX_t b):
        return intTuple.create2(a, b)

    @staticmethod
    cdef intTuple create3(INDEX_t a, INDEX_t b, INDEX_t c):
        cdef:
            intTuple t = intTuple()
        t.size = 3
        t.entries = <INDEX_t *>malloc(3*INDEX_SIZE)
        t.entries[0] = a
        t.entries[1] = b
        t.entries[2] = c
        return t

    @staticmethod
    def create3Py(INDEX_t a, INDEX_t b, INDEX_t c):
        return intTuple.create3(a, b, c)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __hash__(self):
        cdef:
            INDEX_t hash_val = 2166136261
            INDEX_t i
            char * entries = <char*>self.entries
        for i in range(self.size*INDEX_SIZE):
            hash_val = hash_val ^ entries[i]
            hash_val = hash_val * 16777619
        return hash_val

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __eq__(self, intTuple other):
        cdef:
            INDEX_t i
        for i in range(self.size):
            if self.entries[i] != other.entries[i]:
                return False
        return True

    def __repr__(self):
        s = '<'
        s += ','.join([str(self.entries[i]) for i in range(self.size)])
        s += '>'
        return s
