###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from . myTypes import INDEX, REAL, COMPLEX
from . blas import uninitialized
cimport cython

include "config.pxi"

import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI


######################################################################
# Inner products and norms

cdef class ipBase:
    def __init__(self):
        pass

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self,
                 vector_t v1, vector_t v2,
                 BOOL_t acc1=False, BOOL_t acc2=False,
                 BOOL_t asynchronous=False):
        return self.eval(v1, v2, acc1, acc2, asynchronous)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self,
                       vector_t v1, vector_t v2,
                       BOOL_t acc1=False, BOOL_t acc2=False,
                       BOOL_t asynchronous=False):
        raise NotImplementedError()


cdef class normBase:
    def __init__(self):
        pass

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self,
                 vector_t v,
                 BOOL_t acc=False,
                 BOOL_t asynchronous=False):
        return self.eval(v, acc, asynchronous)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self,
                       vector_t v,
                       BOOL_t acc=False,
                       BOOL_t asynchronous=False):
        raise NotImplementedError()


cdef class ip_noop(ipBase):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self,
                       vector_t v1, vector_t v2,
                       BOOL_t acc1=False, BOOL_t acc2=False,
                       BOOL_t asynchronous=False):
        return 10.


cdef class norm_noop(normBase):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self,
                       vector_t v,
                       BOOL_t acc=False,
                       BOOL_t asynchronous=False):
        return 10.


cdef class ip_serial(ipBase):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self,
                       vector_t v1, vector_t v2,
                       BOOL_t acc1=False, BOOL_t acc2=False,
                       BOOL_t asynchronous=False):
        return mydot(v1, v2)


cdef class norm_serial(normBase):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self,
                       vector_t v,
                       BOOL_t acc=False,
                       BOOL_t asynchronous=False):
        return norm(v)


cdef class ip_distributed_nonoverlapping(ipBase):
    def __init__(self, comm):
        self.comm = comm
        self.localIP = ip_serial()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self,
                       vector_t v1, vector_t v2,
                       BOOL_t acc1=False, BOOL_t acc2=False,
                       BOOL_t asynchronous=False):
        cdef:
            REAL_t temp_mem[1]
            REAL_t[::1] temp = temp_mem
        temp[0] = self.localIP.eval(v1, v2)
        self.comm.Allreduce(MPI.IN_PLACE, temp)
        return temp[0]


cdef class norm_distributed_nonoverlapping(normBase):
    def __init__(self, comm):
        self.comm = comm
        self.localIP = ip_serial()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self,
                       vector_t v,
                       BOOL_t acc=False,
                       BOOL_t asynchronous=False):
        cdef:
            REAL_t temp_mem[1]
            REAL_t[::1] temp = temp_mem
        temp[0] = self.localIP.eval(v, v)
        self.comm.Allreduce(MPI.IN_PLACE, temp)
        return sqrt(temp[0])



cdef class ip_distributed(ipBase):
    def __init__(self, overlap, INDEX_t level=-1):
        self.overlap = overlap
        self.comm = overlap.comm
        self.level = level
        self.temporaryMemory = uninitialized((0), dtype=REAL)
        self.localIP = ip_serial()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self,
                       vector_t v1, vector_t v2,
                       BOOL_t acc1=False, BOOL_t acc2=False,
                       BOOL_t asynchronous=False):
        cdef:
            REAL_t n
            vector_t u = self.temporaryMemory
        assert v1.shape[0] == v2.shape[0]
        if v1.shape[0] > u.shape[0]:
            self.temporaryMemory = uninitialized((v1.shape[0]), dtype=REAL)
            u = self.temporaryMemory
        if acc1 == acc2:
            if acc1:
                self.overlap.distribute(v1, u)
                n = self.overlap.reduce(self.localIP(v2, u), asynchronous)
            else:
                self.overlap.accumulate(v1, u, level=self.level, asynchronous=asynchronous)
                n = self.overlap.reduce(self.localIP(v2, u), asynchronous)
        else:
            if not acc1:
                # self.overlap.accumulate(v1, u, level=self.level, asynchronous=asynchronous)
                # self.overlap.distribute(u)
                n = self.localIP.eval(v2, v1)
                n = self.overlap.reduce(n, asynchronous)
            else:
                # self.overlap.accumulate(v2, u, level=self.level, asynchronous=asynchronous)
                # self.overlap.distribute(u)
                n = self.localIP.eval(v1, v2)
                n = self.overlap.reduce(n, asynchronous)
        return n


cdef class norm_distributed(normBase):
    def __init__(self, overlap, INDEX_t level=-1):
        self.overlap = overlap
        self.comm = overlap.comm
        self.level = level
        self.temporaryMemory = uninitialized((0), dtype=REAL)
        self.localIP = ip_serial()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self,
                       vector_t v,
                       BOOL_t acc=False,
                       BOOL_t asynchronous=False):
        cdef:
            vector_t u = self.temporaryMemory
            REAL_t n, nb
        if v.shape[0] > u.shape[0]:
            self.temporaryMemory = uninitialized((v.shape[0]), dtype=REAL)
            u = self.temporaryMemory
        if acc:
            self.overlap.distribute(v, u, level=self.level)
        else:
            self.overlap.accumulate(v, u, level=self.level, asynchronous=asynchronous)
        nb = self.localIP.eval(v, u)
        n = self.overlap.reduce(nb, asynchronous)
        n = sqrt(n)
        return n


cdef class complexipBase:
    def __init__(self):
        pass

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self,
                 complex_vector_t v1, complex_vector_t v2,
                 BOOL_t acc1=False, BOOL_t acc2=False,
                 BOOL_t asynchronous=False):
        return self.eval(v1, v2, acc1, acc2, asynchronous)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef COMPLEX_t eval(self,
                        complex_vector_t v1,
                        complex_vector_t v2,
                        BOOL_t acc1=False,
                        BOOL_t acc2=False,
                        BOOL_t asynchronous=False):
        return 10.


cdef class complexNormBase:
    def __init__(self):
        pass

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self,
                 complex_vector_t v,
                 BOOL_t acc=False,
                 BOOL_t asynchronous=False):
        return self.eval(v, acc, asynchronous)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self,
                       complex_vector_t v,
                       BOOL_t acc=False,
                       BOOL_t asynchronous=False):
        return 10.


cdef class wrapRealNormToComplex(complexNormBase):
    def __init__(self, normBase norm):
        self.norm = norm
        self.temporaryMemory = uninitialized((0), dtype=REAL)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self,
                       complex_vector_t x,
                       BOOL_t acc=False,
                       BOOL_t asynchronous=False):
        cdef:
            INDEX_t i
            REAL_t s = 0.0
        if x.shape[0] != self.temporaryMemory.shape[0]:
            self.temporaryMemory = uninitialized((x.shape[0]), dtype=REAL)
        for i in range(x.shape[0]):
            self.temporaryMemory[i] = x[i].real
        s += self.norm.eval(self.temporaryMemory, acc)**2
        for i in range(x.shape[0]):
            self.temporaryMemory[i] = x[i].imag
        s += self.norm.eval(self.temporaryMemory, acc)**2
        return sqrt(s)


cdef class wrapRealInnerToComplex(complexipBase):
    def __init__(self, ipBase inner):
        self.inner = inner
        self.temporaryMemory = uninitialized((0), dtype=REAL)
        self.temporaryMemory2 = uninitialized((0), dtype=REAL)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef COMPLEX_t eval(self,
                        complex_vector_t x,
                        complex_vector_t y,
                        BOOL_t acc1=False,
                        BOOL_t acc2=False,
                        BOOL_t asynchronous=False):
        cdef:
            INDEX_t i
            COMPLEX_t s = 0.0
            COMPLEX_t I = 1j
        if x.shape[0] != self.temporaryMemory.shape[0]:
            self.temporaryMemory = uninitialized((x.shape[0]), dtype=REAL)
            self.temporaryMemory2 = uninitialized((x.shape[0]), dtype=REAL)
        for i in range(x.shape[0]):
            self.temporaryMemory[i] = x[i].real
        for i in range(y.shape[0]):
            self.temporaryMemory2[i] = y[i].real
        # Re * Re
        s = self.inner.eval(self.temporaryMemory, self.temporaryMemory2, acc1, acc2)
        for i in range(y.shape[0]):
            self.temporaryMemory2[i] = y[i].imag
        # Re * Im
        s = s + I * self.inner.eval(self.temporaryMemory, self.temporaryMemory2, acc1, acc2)
        for i in range(x.shape[0]):
            self.temporaryMemory[i] = x[i].imag
        # Im * Im
        s = s + self.inner.eval(self.temporaryMemory, self.temporaryMemory2, acc1, acc2)
        for i in range(y.shape[0]):
            self.temporaryMemory2[i] = y[i].real
        # Im * Re
        s = s - I * self.inner.eval(self.temporaryMemory, self.temporaryMemory2, acc1, acc2)
        return s
