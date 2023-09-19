###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef void spmv(INDEX_t[::1] indptr, INDEX_t[::1] indices, SCALAR_t[::1] data, SCALAR_t[::1] x, SCALAR_t[::1] y, BOOL_t overwrite=True):
    cdef:
        INDEX_t i, jj, j
        SCALAR_t temp
    if SCALAR_t is COMPLEX_t:
        for i in range(indptr.shape[0]-1):
            temp = 0.
            for jj in range(indptr[i], indptr[i+1]):
                j = indices[jj]
                temp = temp + data[jj]*x[j]
            if overwrite:
                y[i] = temp
            else:
                y[i] = y[i]+temp
    else:
        for i in range(indptr.shape[0]-1):
            temp = 0.
            for jj in range(indptr[i], indptr[i+1]):
                j = indices[jj]
                temp += data[jj]*x[j]
            if overwrite:
                y[i] = temp
            else:
                y[i] += temp

cdef void spres(INDEX_t[::1] indptr, INDEX_t[::1] indices, SCALAR_t[::1] data, SCALAR_t[::1] x, SCALAR_t[::1] rhs, SCALAR_t[::1] result):
    cdef:
        INDEX_t i, jj, j
        SCALAR_t temp
        INDEX_t num_rows = indptr.shape[0]-1
    if SCALAR_t is COMPLEX_t:
        for i in range(num_rows):
            temp = rhs[i]
            for jj in range(indptr[i], indptr[i+1]):
                j = indices[jj]
                temp = temp-data[jj]*x[j]
            result[i] = temp
    else:
        for i in range(num_rows):
            temp = rhs[i]
            for jj in range(indptr[i], indptr[i+1]):
                j = indices[jj]
                temp -= data[jj]*x[j]
            result[i] = temp
