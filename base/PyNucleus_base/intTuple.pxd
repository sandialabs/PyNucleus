###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from . myTypes cimport INDEX_t, BOOL_t


cdef class intTuple:
    cdef:
        INDEX_t * entries
        int size

    cdef void set(self, INDEX_t * t, int size)
    cdef void assign(self, INDEX_t * t)
    cdef void assignNonOwning(self, INDEX_t * t)
    cdef void get(self, INDEX_t * t)
    @staticmethod
    cdef intTuple create(INDEX_t[::1] t)
    @staticmethod
    cdef intTuple createNonOwning(INDEX_t[::1] t)
    @staticmethod
    @staticmethod
    cdef intTuple create2(INDEX_t a, INDEX_t b)
    @staticmethod
    cdef intTuple create3(INDEX_t a, INDEX_t b, INDEX_t c)


cdef class productIterator:
    cdef:
        INDEX_t m
        INDEX_t dim
        INDEX_t[::1] idx
    cdef void reset(self)
    cdef BOOL_t step(self)
