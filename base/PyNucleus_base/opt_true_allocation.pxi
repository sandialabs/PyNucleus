###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

def uninitialized(*args, **kwargs):
    if 'dtype' in kwargs and np.issubdtype(kwargs['dtype'], np.integer):
        kwargs['fill_value'] = np.iinfo(kwargs['dtype']).min
    else:
        kwargs['fill_value'] = NAN
    return np.full(*args, **kwargs)


def uninitialized_like(like, **kwargs):
    if np.issubdtype(np.array(like, copy=False).dtype, np.integer):
        kwargs['fill_value'] = np.iinfo(like.dtype).min
    else:
        kwargs['fill_value'] = NAN
    return np.full_like(like, **kwargs)


cpdef carray uninitializedINDEX(tuple shape):
    cdef:
        carray a = carray(shape, 4, 'i')
        Py_ssize_t s, i
    s = 1
    for i in range(len(shape)):
        s *= shape[i]
    for i in range(s):
        (<INDEX_t*>a.data)[i] = MAX_INT
    return a


cpdef carray uninitializedREAL(tuple shape):
    cdef:
        carray a = carray(shape, 8, 'd')
        Py_ssize_t s, i
    s = 1
    for i in range(len(shape)):
        s *= shape[i]
    for i in range(s):
        (<REAL_t*>a.data)[i] = NAN
    return a
