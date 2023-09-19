###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

def uninitialized(*args, **kwargs):
    return np.empty(*args, **kwargs)


def uninitialized_like(like, **kwargs):
    return np.empty_like(like, **kwargs)


cpdef carray uninitializedINDEX(tuple shape):
    cdef:
        carray a = carray(shape, 4, 'i')
        Py_ssize_t s, i
    return a


cpdef carray uninitializedREAL(tuple shape):
    cdef:
        carray a = carray(shape, 8, 'd')
        Py_ssize_t s, i
    return a
