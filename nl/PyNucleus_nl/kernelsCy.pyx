###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


cimport cython

cdef extern from "kernels.hpp":
    cdef cppclass kernel_t:
        kernel_t()
        REAL_t eval(REAL_t *x, REAL_t *y) nogil

    cdef cppclass fractional_kernel_t(kernel_t):
        fractional_kernel_t(REAL_t s_, REAL_t C_)

    # ctypedef REAL_t (*kernel_callback_t)(REAL_t *x, REAL_t *y, void* user_data)

    cdef cppclass callback_kernel_t(kernel_t):
        callback_kernel_t(kernel_callback_t kernel_callback_, void* user_data)


cdef class kernelCy:
    def __init__(self):
        pass

    cdef void setCallback(self, kernel_callback_t callback):
        self.callback = callback

    cdef void setParams(self, void* params):
        self.params = params

    cdef void setKernel(self, void *user_data, size_t pos):
        (<kernel_t**>(user_data+pos))[0] = new callback_kernel_t(self.callback[0], self.params)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.callback(&x[0], &y[0], self.params)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t evalPtr(self, REAL_t* x, REAL_t* y):
        return self.callback(x, y, self.params)
