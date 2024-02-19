###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef void assign(SCALAR_t[::1] y, const SCALAR_t[::1] x):
    cdef:
        INDEX_t i
    for i in range(x.shape[0]):
        y[i] = x[i]

cdef void assignScaled(SCALAR_t[::1] y, const SCALAR_t[::1] x, SCALAR_t alpha):
    cdef:
        INDEX_t i
    for i in range(x.shape[0]):
        y[i] = alpha*x[i]


cdef void assign3(SCALAR_t[::1] z, const SCALAR_t[::1] x, SCALAR_t alpha, const SCALAR_t[::1] y, SCALAR_t beta):
    cdef:
        INDEX_t i
    for i in range(x.shape[0]):
        z[i] = alpha*x[i] + beta*y[i]

cdef void update(SCALAR_t[::1] x, SCALAR_t[::1] y):
    cdef:
        INDEX_t i
    if SCALAR_t is COMPLEX_t:
        for i in range(x.shape[0]):
            x[i] = x[i] + y[i]
    else:
        for i in range(x.shape[0]):
            x[i] += y[i]

cdef void updateScaled(SCALAR_t[::1] x, SCALAR_t[::1] y, SCALAR_t alpha):
    cdef:
        INDEX_t i
    if SCALAR_t is COMPLEX_t:
        for i in range(x.shape[0]):
            x[i] = x[i]+alpha*y[i]
    else:
        for i in range(x.shape[0]):
            x[i] += alpha*y[i]

cdef void scaleScalar(SCALAR_t[::1] x, SCALAR_t alpha):
    cdef:
        INDEX_t i
    if SCALAR_t is COMPLEX_t:
        for i in range(x.shape[0]):
            x[i] *= alpha
    else:
        for i in range(x.shape[0]):
            x[i] = x[i]*alpha

cdef SCALAR_t mydot(SCALAR_t[::1] v0, SCALAR_t[::1] v1):
    cdef:
        int i
        SCALAR_t s = 0.0
    if SCALAR_t is COMPLEX_t:
        for i in range(v0.shape[0]):
            s += v0[i].conjugate()*v1[i]
    else:
        for i in range(v0.shape[0]):
            s += v0[i]*v1[i]
    return s

cdef REAL_t norm(SCALAR_t[::1] x):
    cdef:
        int i
        REAL_t s = 0.0
    if SCALAR_t is COMPLEX_t:
        for i in range(x.shape[0]):
            s += (x[i].conjugate()*x[i]).real
    else:
        for i in range(x.shape[0]):
            s += x[i]*x[i]
    return sqrt(s)

cdef void gemv(SCALAR_t[:, ::1] A, SCALAR_t[::1] x, SCALAR_t[::1] y, SCALAR_t beta=0.):
    cdef:
        INDEX_t i, j
        SCALAR_t s
    if SCALAR_t is COMPLEX_t:
        if beta != 0.:
            for i in range(A.shape[0]):
                s = 0.
                for j in range(A.shape[1]):
                    s = s + A[i, j]*x[j]
                y[i] = beta*y[i]+s
        else:
            for i in range(A.shape[0]):
                s = 0.
                for j in range(A.shape[1]):
                    s = s + A[i, j]*x[j]
                y[i] = s
    else:
        if beta != 0.:
            for i in range(A.shape[0]):
                s = 0.
                for j in range(A.shape[1]):
                    s += A[i, j]*x[j]
                y[i] = beta*y[i]+s
        else:
            for i in range(A.shape[0]):
                s = 0.
                for j in range(A.shape[1]):
                    s += A[i, j]*x[j]
                y[i] = s


cdef void gemvF(SCALAR_t[::1, :] A, SCALAR_t[::1] x, SCALAR_t[::1] y, SCALAR_t beta=0.):
    cdef:
        INDEX_t i, j
    if SCALAR_t is COMPLEX_t:
        if beta != 0.:
            for i in range(A.shape[0]):
                y[i] = beta*y[i]
        for j in range(A.shape[1]):
            for i in range(A.shape[0]):
                y[i] = y[i] + A[i, j]*x[j]
    else:
        if beta != 0.:
            for i in range(A.shape[0]):
                y[i] *= beta
        for j in range(A.shape[1]):
            for i in range(A.shape[0]):
                y[i] += A[i, j]*x[j]


cdef void gemvT(SCALAR_t[:, ::1] A, SCALAR_t[::1] x, SCALAR_t[::1] y, SCALAR_t beta=0.):
    cdef:
        INDEX_t i, j
    if SCALAR_t is COMPLEX_t:
        if beta != 0.:
            for i in range(A.shape[0]):
                y[i] = y[i]*beta
        else:
            y[:] = 0.
        for i in range(A.shape[1]):
            for j in range(A.shape[0]):
                y[i] = y[i]+A[j, i]*x[j]
    else:
        if beta != 0.:
            for i in range(A.shape[0]):
                y[i] *= beta
        else:
            y[:] = 0.
        for i in range(A.shape[1]):
            for j in range(A.shape[0]):
                y[i] += A[j, i]*x[j]


cdef void matmat(SCALAR_t[:, ::1] A, SCALAR_t[:, ::1] B, SCALAR_t[:, ::1] C):
    cdef:
        INDEX_t i, j, k
    C[:, :] = 0.
    if SCALAR_t is COMPLEX_t:
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                for k in range(B.shape[1]):
                    C[i, k] = C[i, k] + A[i, j]*B[j, k]
    else:
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                for k in range(B.shape[1]):
                    C[i, k] += A[i, j]*B[j, k]
