###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . myTypes cimport INDEX_t, REAL_t, BOOL_t
from numpy cimport uint16_t


cdef class sparsityPattern:
    cdef:
        INDEX_t ** indexL
        INDEX_t[::1] counts
        uint16_t initial_length
        uint16_t[::1] lengths
        INDEX_t num_dofs, nnz
        uint16_t index

    cdef inline BOOL_t findIndex(self, INDEX_t I, INDEX_t J)
    cdef inline void add(self, INDEX_t I, INDEX_t J)
    cdef freeze(self)
