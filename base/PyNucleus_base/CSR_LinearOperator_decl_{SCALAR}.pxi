###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from . blas cimport spmv, spres
from . blas import uninitialized

cdef class {SCALAR_label}CSR_LinearOperator({SCALAR_label}LinearOperator):
    cdef:
        public INDEX_t[::1] indptr, indices
        public {SCALAR}_t[::1] data
        public int NoThreads
        public BOOL_t indices_sorted
    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1
    cdef INDEX_t matvec_no_overwrite(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] y) except -1
    cdef INDEX_t matvec_multi(self,
                              {SCALAR}_t[:, ::1] x,
                              {SCALAR}_t[:, ::1] y) except -1
    cdef {SCALAR}_t getEntry(self, INDEX_t I, INDEX_t J)
    cdef void setEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val)
    cpdef {SCALAR_label}CSR_LinearOperator getBlockDiagonal(self, sparseGraph blocks)
