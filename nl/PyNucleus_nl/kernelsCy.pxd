###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes cimport REAL_t

ctypedef REAL_t (*kernel_callback_t)(REAL_t *x, REAL_t *y, void* user_data)


cdef class kernelCy:
    cdef:
        kernel_callback_t callback
        void *params
    cdef void setCallback(self, kernel_callback_t callback)
    cdef void setParams(self, void* params)
    cdef void setKernel(self, void *user_data, size_t pos)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y)
    cdef REAL_t evalPtr(self, REAL_t* x, REAL_t* y)
