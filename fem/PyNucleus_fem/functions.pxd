###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes cimport REAL_t, INDEX_t, COMPLEX_t, BOOL_t
from numpy cimport uint8_t



cdef class function:
    cdef REAL_t eval(self, REAL_t[::1] x)


cdef class constant(function):
    cdef:
        public REAL_t value


ctypedef REAL_t(*volume_t)(REAL_t[:, ::1])


cdef class complexFunction:
    cdef COMPLEX_t eval(self, REAL_t[::1] x)


cdef class vectorFunction:
    cdef:
        public INDEX_t rows
    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] vals)


cdef class componentVectorFunction(vectorFunction):
    cdef:
        public list components


cdef class matrixFunction:
    cdef:
        list components
        INDEX_t rows
        INDEX_t columns
        public BOOL_t symmetric
    cdef void eval(self, REAL_t[::1] x, REAL_t[:, ::1] vals)
