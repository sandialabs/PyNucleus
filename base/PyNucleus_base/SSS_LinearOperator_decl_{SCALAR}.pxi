###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . blas import uninitialized


cdef class {SCALAR_label}SSS_LinearOperator({SCALAR_label}LinearOperator):
    cdef:
        public INDEX_t[::1] indptr, indices
        public {SCALAR}_t[::1] data, diagonal
        public BOOL_t indices_sorted
        public int NoThreads
    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1
    cdef INDEX_t matvec_no_overwrite(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] y) except -1
    cdef void setEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val)
    cdef {SCALAR}_t getEntry({SCALAR_label}SSS_LinearOperator self, INDEX_t I, INDEX_t J)


