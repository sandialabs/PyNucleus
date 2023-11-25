###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t
from libc.math cimport sqrt
from . blas cimport mydot, norm
cimport numpy as np
from mpi4py cimport MPI

ctypedef fused SCALAR_t:
    REAL_t
    COMPLEX_t

ctypedef REAL_t[::1] vector_t
ctypedef COMPLEX_t[::1] complex_vector_t


######################################################################
# INNER products and norms

cdef class ipBase:
    cdef REAL_t eval(self,
                     vector_t v1, vector_t v2,
                     BOOL_t acc1=*, BOOL_t acc2=*,
                     BOOL_t asynchronous=*)


cdef class normBase:
    cdef REAL_t eval(self,
                     vector_t v,
                     BOOL_t acc=*,
                     BOOL_t asynchronous=*)


cdef class norm_serial(normBase):
    pass


cdef class ip_serial(ipBase):
    pass


cdef class ip_distributed_nonoverlapping(ipBase):
    cdef:
        MPI.Comm comm
        ip_serial localIP


cdef class norm_distributed_nonoverlapping(normBase):
    cdef:
        MPI.Comm comm
        ip_serial localIP


cdef class ip_distributed(ipBase):
    cdef:
        object overlap
        MPI.Comm comm
        INDEX_t level
        public vector_t temporaryMemory
        ip_serial localIP


cdef class norm_distributed(normBase):
    cdef:
        object overlap
        MPI.Comm comm
        INDEX_t level
        public vector_t temporaryMemory
        ip_serial localIP


cdef class complexipBase:
    cdef COMPLEX_t eval(self,
                        complex_vector_t v1, complex_vector_t v2,
                        BOOL_t acc1=*, BOOL_t acc2=*,
                        BOOL_t asynchronous=*)


cdef class complexNormBase:
    cdef REAL_t eval(self,
                     complex_vector_t v,
                     BOOL_t acc=*,
                     BOOL_t asynchronous=*)


cdef class wrapRealNormToComplex(complexNormBase):
    cdef:
        normBase norm
        vector_t temporaryMemory


cdef class wrapRealInnerToComplex(complexipBase):
    cdef:
        ipBase inner
        vector_t temporaryMemory, temporaryMemory2
