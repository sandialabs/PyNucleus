###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from . myTypes import INDEX, REAL, COMPLEX
from . myTypes cimport INDEX_t, REAL_t, COMPLEX_t
from . linear_operators cimport (sort_indices,
                                 LinearOperator,
                                 CSR_LinearOperator,
                                 SSS_LinearOperator,
                                 LinearOperator_wrapper,
                                 TimeStepperLinearOperator,
                                 ComplexLinearOperator,
                                 wrapRealToComplex)
from . solvers cimport cg_solver, gmres_solver, complex_gmres_solver
from . ip_norm import wrapRealInnerToComplex, wrapRealNormToComplex, ip_serial, norm_serial
from . ip_norm cimport ipBase, normBase, complexipBase, complexNormBase
from . utilsCy import UniformOnUnitSphere
from . blas cimport assign, assignScaled, assign3, update, updateScaled, mydot, gemvF
from . blas import uninitialized
from . convergence cimport (convergenceMaster, noOpConvergenceMaster,
                            convergenceClient, noOpConvergenceClient)

include "config.pxi"


cpdef REAL_t accumulate_serial(REAL_t[::1] x):
    pass


def ichol_csr(A):
    cdef:
        np.ndarray[INDEX_t, ndim=1] indptr_mem = np.zeros_like(A.indptr,
                                                               dtype=INDEX)
        np.ndarray[INDEX_t, ndim=1] indices_mem
        np.ndarray[REAL_t, ndim=1] data_mem, diagonal_mem = uninitialized((A.num_rows), dtype=REAL)
        INDEX_t[::1] Aindptr = A.indptr, Aindices = A.indices
        REAL_t[::1] Adata = A.data
        INDEX_t[::1] indptr = indptr_mem, indices
        REAL_t[::1] data, diagonal = diagonal_mem
        INDEX_t i, ii, jj, j, nnz, kk, k, hh
        INDEX_t num_rows = A.num_rows
    # step 1: build indptr
    for i in range(num_rows):
        for jj in range(Aindptr[i], Aindptr[i+1]):
            j = Aindices[jj]
            if j < i:
                indptr[j+1] += 1
    for i in range(num_rows):
        indptr[i+1] += indptr[i]
    # step 2: build indices and initial data
    nnz = indptr[num_rows]
    indices_mem = uninitialized((nnz), dtype=INDEX)
    indices = indices_mem
    data_mem = np.zeros((nnz), dtype=REAL)
    data = data_mem
    for i in range(num_rows):
        for ii in range(Aindptr[i], Aindptr[i+1]):
            if Aindices[ii] == i:
                diagonal[i] = Adata[ii]
                break
        for jj in range(Aindptr[i], Aindptr[i+1]):
            j = Aindices[jj]
            if j < i:
                indices[indptr[j]] = i
                data[indptr[j]] = Adata[jj]
                indptr[j] += 1
    for i in range(num_rows, 0, -1):
        indptr[i] = indptr[i-1]
    indptr[0] = 0
    sort_indices(indptr, indices, data)
    # step 3: perform Cholesky
    for i in range(num_rows):
        diagonal[i] = sqrt(diagonal[i])
        for jj in range(indptr[i], indptr[i+1]):
            data[jj] /= diagonal[i]
            j = indices[jj]
            diagonal[j] -= data[jj]*data[jj]
            for kk in range(indptr[j], indptr[j+1]):
                k = indices[kk]
                for hh in range(jj, indptr[i+1]):
                    if indices[hh] == k:
                        data[kk] -= data[hh]*data[jj]
                        break
    return indices_mem, indptr_mem, data_mem, diagonal_mem


def ichol_sss(SSS_LinearOperator A):
    cdef:
        np.ndarray[INDEX_t, ndim=1] indptr_mem = np.zeros_like(A.indptr,
                                                               dtype=INDEX)
        np.ndarray[INDEX_t, ndim=1] indices_mem
        np.ndarray[REAL_t, ndim=1] data_mem, diagonal_mem = uninitialized((A.num_rows), dtype=REAL)
        INDEX_t[::1] Aindptr = A.indptr, Aindices = A.indices
        REAL_t[::1] Adata = A.data, Adiagonal = A.diagonal
        INDEX_t[::1] indptr = indptr_mem, indices
        REAL_t[::1] data, diagonal = diagonal_mem
        INDEX_t i, ii, jj, j, nnz, kk, k, hh
        INDEX_t num_rows = A.num_rows
    # step 1: build indptr
    for i in range(num_rows):
        for jj in range(Aindptr[i], Aindptr[i+1]):
            j = Aindices[jj]
            if j < i:
                indptr[j+1] += 1
    for i in range(num_rows):
        indptr[i+1] += indptr[i]
    # step 2: build indices and initial data
    nnz = indptr[num_rows]
    indices_mem = uninitialized((nnz), dtype=INDEX)
    indices = indices_mem
    data_mem = np.zeros((nnz), dtype=REAL)
    data = data_mem
    for i in range(num_rows):
        diagonal[i] = Adiagonal[i]
        for jj in range(Aindptr[i], Aindptr[i+1]):
            j = Aindices[jj]
            if j < i:
                indices[indptr[j]] = i
                data[indptr[j]] = Adata[jj]
                indptr[j] += 1
    for i in range(num_rows, 0, -1):
        indptr[i] = indptr[i-1]
    indptr[0] = 0
    sort_indices(indptr, indices, data)
    # step 3: perform Cholesky
    for i in range(num_rows):
        diagonal[i] = sqrt(diagonal[i])
        for jj in range(indptr[i], indptr[i+1]):
            data[jj] /= diagonal[i]
            j = indices[jj]
            diagonal[j] -= data[jj]*data[jj]
            for kk in range(indptr[j], indptr[j+1]):
                k = indices[kk]
                for hh in range(jj, indptr[i+1]):
                    if indices[hh] == k:
                        data[kk] -= data[hh]*data[jj]
                        break
    return indices_mem, indptr_mem, data_mem, diagonal_mem


# Assumes that indices are ordered
cdef void forward_solve_csc(INDEX_t[::1] indptr,
                            INDEX_t[::1] indices,
                            SCALAR_t[::1] data,
                            SCALAR_t[::1] b,
                            SCALAR_t[::1] y,
                            BOOL_t unitDiagonal):
    cdef:
        INDEX_t n = b.shape[0], i, j, i1, i2
    if SCALAR_t is REAL_t:
        if unitDiagonal:
            for j in range(n):
                i1 = indptr[j]
                i2 = indptr[j+1]
                y[j] = b[j]-y[j]
                # FIX: Should I start with i1 here, not i1+1?
                # Maybe SuperLU saves the ones on the diagonal?
                for i in range(i1+1, i2):
                    y[indices[i]] += data[i]*y[j]
        else:
            for j in range(n):
                i1 = indptr[j]
                i2 = indptr[j+1]
                y[j] = (b[j]-y[j])/data[i1]
                for i in range(i1+1, i2):
                    y[indices[i]] += data[i]*y[j]
    else:
        if unitDiagonal:
            for j in range(n):
                i1 = indptr[j]
                i2 = indptr[j+1]
                y[j] = b[j]-y[j]
                # FIX: Should I start with i1 here, not i1+1?
                # Maybe SuperLU saves the ones on the diagonal?
                for i in range(i1+1, i2):
                    y[indices[i]] = y[indices[i]] + data[i]*y[j]
        else:
            for j in range(n):
                i1 = indptr[j]
                i2 = indptr[j+1]
                y[j] = (b[j]-y[j])/data[i1]
                for i in range(i1+1, i2):
                    y[indices[i]] = y[indices[i]] + data[i]*y[j]


IF USE_MKL_TRISOLVE:

    ctypedef INDEX_t MKL_INT

    cdef extern from "mkl/mkl_spblas.h":
        void mkl_dcsrsm (const char *transa , const MKL_INT *m , const MKL_INT *n , const REAL_t *alpha , const char *matdescra ,
                         const REAL_t *val , const MKL_INT *indx , const MKL_INT *pntrb , const MKL_INT *pntre ,
                         const REAL_t *b , const MKL_INT *ldb , REAL_t *c , const MKL_INT *ldc );

    cdef inline void trisolve_mkl(INDEX_t[::1] indptr,
                                  INDEX_t[::1] indices,
                                  REAL_t[::1] data,
                                  REAL_t[::1] b,
                                  REAL_t[::1] y,
                                  BOOL_t forward=True,
                                  BOOL_t unitDiagonal=False):
        cdef:
            char transA
            REAL_t alpha = 1.
            char matdscr[6]
            INDEX_t inc = 1
            INDEX_t n = indptr.shape[0]-1
            INDEX_t one = 1
        matdscr[0] = 84
        if forward:
            transA = 84
        else:
            transA = 78
        matdscr[1] = 85
        if unitDiagonal:
            matdscr[2] = 85
        else:
            matdscr[2] = 78
        matdscr[3] = 67
        mkl_dcsrsm(&transA, &n, &one, &alpha, &matdscr[0], &data[0], &indices[0], &indptr[0], &indptr[1], &b[0], &one, &y[0], &one)


# Assumes that indices are ordered
cdef void forward_solve_sss(const INDEX_t[::1] indptr,
                            const INDEX_t[::1] indices,
                            const REAL_t[::1] data,
                            const REAL_t[::1] diagonal,
                            const REAL_t[::1] b,
                            REAL_t[::1] y,
                            BOOL_t unitDiagonal=False):
    cdef:
        INDEX_t n = b.shape[0], i, j
    if unitDiagonal:
        for j in range(n):
            y[j] = b[j]-y[j]
            for i in range(indptr[j], indptr[j+1]):
                y[indices[i]] += data[i]*y[j]
    else:
        for j in range(n):
            y[j] = (b[j]-y[j])/diagonal[j]
            for i in range(indptr[j], indptr[j+1]):
                y[indices[i]] += data[i]*y[j]


# Assumes that indices are ordered
cdef void forward_solve_sss_noInverse(const INDEX_t[::1] indptr,
                                      const INDEX_t[::1] indices,
                                      const REAL_t[::1] data,
                                      const REAL_t[::1] invDiagonal,
                                      const REAL_t[::1] b,
                                      REAL_t[::1] y,
                                      BOOL_t unitDiagonal=False):
    cdef:
        INDEX_t n = b.shape[0], i, j
    if unitDiagonal:
        for j in range(n):
            y[j] = b[j]-y[j]
            for i in range(indptr[j], indptr[j+1]):
                y[indices[i]] += data[i]*y[j]
    else:
        for j in range(n):
            y[j] = (b[j]-y[j])*invDiagonal[j]
            for i in range(indptr[j], indptr[j+1]):
                y[indices[i]] += data[i]*y[j]


# Assumes that indices are ordered
cdef inline void backward_solve_csc(INDEX_t[::1] indptr,
                                    INDEX_t[::1] indices,
                                    SCALAR_t[::1] data,
                                    SCALAR_t[::1] b,
                                    SCALAR_t[::1] y):
    cdef:
        INDEX_t n = b.shape[0], i, j, i1, i2
    if SCALAR_t is REAL_t:
        for j in range(n-1, -1, -1):
            i1 = indptr[j]
            i2 = indptr[j+1]
            y[j] = (b[j]-y[j])/data[i2-1]
            for i in range(i1, i2-1):
                y[indices[i]] += data[i]*y[j]
    else:
        for j in range(n-1, -1, -1):
            i1 = indptr[j]
            i2 = indptr[j+1]
            y[j] = (b[j]-y[j])/data[i2-1]
            for i in range(i1, i2-1):
                y[indices[i]] = y[indices[i]]+data[i]*y[j]


# Assumes that indices are ordered
cdef inline void backward_solve_csr(INDEX_t[::1] indptr,
                                    INDEX_t[::1] indices,
                                    REAL_t[::1] data,
                                    REAL_t[::1] b,
                                    REAL_t[::1] y):
    cdef:
        INDEX_t n = b.shape[0], i, j, jj
        REAL_t temp
    for i in range(n-1, -1, -1):
        temp = b[i]
        jj = indptr[i+1]-1
        j = indices[jj]
        while j > i:
            temp -= data[jj]*y[j]
            jj -= 1
            j = indices[jj]
        y[i] = temp/data[jj]


# Assumes that indices are ordered
cdef inline void backward_solve_sss(const INDEX_t[::1] indptr,
                                    const INDEX_t[::1] indices,
                                    const REAL_t[::1] data,
                                    const REAL_t[::1] diagonal,
                                    const REAL_t[::1] b,
                                    REAL_t[::1] y):
    cdef:
        INDEX_t n = b.shape[0], i, jj
        REAL_t temp
    for i in range(n-1, -1, -1):
        temp = b[i]
        for jj in range(indptr[i], indptr[i+1]):
            temp -= data[jj]*y[indices[jj]]
        y[i] = temp/diagonal[i]


# Assumes that indices are ordered
cdef void backward_solve_sss_noInverse(const INDEX_t[::1] indptr,
                                       const INDEX_t[::1] indices,
                                       const REAL_t[::1] data,
                                       const REAL_t[::1] invDiagonal,
                                       const REAL_t[::1] b,
                                       REAL_t[::1] y):
    cdef:
        INDEX_t n = b.shape[0], i, jj
        REAL_t temp
    for i in range(n-1, -1, -1):
        temp = b[i]
        for jj in range(indptr[i], indptr[i+1]):
            temp -= data[jj]*y[indices[jj]]
        y[i] = temp*invDiagonal[i]


# Assumes that indices are ordered
cpdef solve_LU(INDEX_t[::1] Lindptr, INDEX_t[::1] Lindices, REAL_t[::1] Ldata,
               INDEX_t[::1] Uindptr, INDEX_t[::1] Uindices, REAL_t[::1] Udata,
               INDEX_t[::1] perm_r, INDEX_t[::1] perm_c,
               REAL_t[::1] b):
    cdef:
        INDEX_t n = b.shape[0], i, j
        np.ndarray[REAL_t, ndim=1] temp1_mem = np.zeros((n), dtype=REAL)
        np.ndarray[REAL_t, ndim=1] temp2_mem = uninitialized((n), dtype=REAL)
        REAL_t[::1] temp1 = temp1_mem
        REAL_t[::1] temp2 = temp2_mem
    for i in range(n):
        temp2[perm_r[i]] = b[i]
    forward_solve_csc(Lindptr, Lindices, Ldata, temp2, temp1,
                      unitDiagonal=True)
    temp2[:] = 0.
    backward_solve_csc(Uindptr, Uindices, Udata, temp1, temp2)
    for i in range(n):
        temp1[i] = temp2[perm_c[i]]
    return temp1_mem



cdef class ILU_solver:
    cdef:
        public INDEX_t[::1] Lindptr, Lindices
        public REAL_t[::1] Ldata
        public INDEX_t[::1] Uindptr, Uindices
        public REAL_t[::1] Udata
        public INDEX_t[::1] perm_c, perm_r
        REAL_t[::1] temp1
        REAL_t[::1] temp2

    def __init__(self, num_rows):
        self.temp1 = uninitialized((num_rows), dtype=REAL)
        self.temp2 = uninitialized((num_rows), dtype=REAL)

    def setup(self, A, fill_factor=1.):
        from scipy.sparse.linalg import spilu
        Clu = spilu(A.to_csr().tocsc(), fill_factor=fill_factor)
        self.Lindices = Clu.L.indices
        self.Lindptr = Clu.L.indptr
        self.Ldata = Clu.L.data
        self.Uindices = Clu.U.indices
        self.Uindptr = Clu.U.indptr
        self.Udata = Clu.U.data
        self.perm_r = Clu.perm_r
        self.perm_c = Clu.perm_c

    cpdef solve(self, REAL_t[::1] b, REAL_t[::1] x):
        cdef INDEX_t i
        self.temp1[:] = 0.
        for i in range(x.shape[0]):
            self.temp2[self.perm_r[i]] = b[i]
        forward_solve_csc(self.Lindptr, self.Lindices, self.Ldata,
                          self.temp2, self.temp1,
                          unitDiagonal=True)
        self.temp2[:] = 0.
        backward_solve_csc(self.Uindptr, self.Uindices, self.Udata,
                           self.temp1, self.temp2)
        for i in range(x.shape[0]):
            x[i] = self.temp2[self.perm_c[i]]

    def asPreconditioner(self):
        return LinearOperator_wrapper(self.temp1.shape[0],
                                        self.temp1.shape[0],
                                        self.solve)


# Assumes that indices are ordered
cpdef solve_cholesky(INDEX_t[::1] Lindptr,
                     INDEX_t[::1] Lindices,
                     REAL_t[::1] Ldata,
                     REAL_t[::1] b):
    cdef:
        INDEX_t n = b.shape[0], i, j
        np.ndarray[REAL_t, ndim=1] temp_mem = np.zeros((n), dtype=REAL)
        REAL_t[::1] temp = temp_mem
    forward_solve_csc(Lindptr, Lindices, Ldata, b, temp,
                      unitDiagonal=False)
    backward_solve_csr(Lindptr, Lindices, Ldata, temp, temp)
    return temp_mem


cdef class cholesky_solver:
    cdef:
        public INDEX_t[::1] indptr, indices
        public REAL_t[::1] data, diagonal, temp
        CSR_LinearOperator L

    def __init__(self, num_rows):
        self.temp = uninitialized((num_rows), dtype=REAL)

    def setup(self, A):
        cdef:
            INDEX_t i
        if isinstance(A, CSR_LinearOperator):
            self.indices, self.indptr, self.data, self.diagonal = ichol_csr(A)
        elif isinstance(A, SSS_LinearOperator):
            # self.indices, self.indptr, self.data, self.diagonal = ichol_sss(A)
            self.indices, self.indptr, self.data, self.diagonal = ichol_csr(A.to_csr_linear_operator())
        elif isinstance(A, TimeStepperLinearOperator):
            B = A.to_csr_linear_operator()
            self.indices, self.indptr, self.data, self.diagonal = ichol_csr(B)
        else:
            raise NotImplementedError()

        IF USE_MKL_TRISOLVE:
            from . linear_operators import diagonalOperator
            T = CSR_LinearOperator(self.indices, self.indptr, self.data).to_csr()+diagonalOperator(self.diagonal).to_csr()
            self.L = CSR_LinearOperator.from_csr(T)
        ELSE:
            for i in range(self.diagonal.shape[0]):
                self.diagonal[i] = 1./self.diagonal[i]

    cpdef solve(self, REAL_t[::1] b, REAL_t[::1] x):
        self.temp[:] = 0.0
        IF USE_MKL_TRISOLVE:
            trisolve_mkl(self.L.indptr, self.L.indices, self.L.data, b, self.temp, forward=True, unitDiagonal=False)
            trisolve_mkl(self.L.indptr, self.L.indices, self.L.data, self.temp, x, forward=False, unitDiagonal=False)
        ELSE:
            forward_solve_sss_noInverse(self.indptr, self.indices,
                                    self.data, self.diagonal,
                                    b, self.temp, unitDiagonal=False)
            backward_solve_sss_noInverse(self.indptr, self.indices,
                                         self.data, self.diagonal,
                                         self.temp, x)

    def asPreconditioner(self):
        return LinearOperator_wrapper(self.diagonal.shape[0],
                                        self.diagonal.shape[0],
                                        self.solve)


cpdef void bicgstab(LinearOperator A,
                    const REAL_t[::1] b,
                    REAL_t[::1] x,
                    REAL_t[::1] x0=None,
                    REAL_t tol=1e-10,
                    int maxiter=20,
                    list residuals=None,
                    LinearOperator precond=None,
                    ipBase inner=ip_serial(),
                    normBase norm=norm_serial(),
                    accumulate=accumulate_serial,
                    BOOL_t use2norm=True,
                    REAL_t[::1] temporaryMemory=None):
    """
    Stabilized Biconjugate Gradient iteration.

    In a distributed solve, we want:
    A:        accumulated to distributed
    precond:  distributed to accumulated
    b:        distributed
    x:        accumulated
    x0:       accumulated

    In the unpreconditioned distributed case, set precond to accumulate.

    If use2norm is False, use Preconditioner norm of residual as
    stopping criterion, otherwise use 2-norm of residual.

    Memory requirement:
    8*dim with preconditioner,
    6*dim without.
    """

    cdef:
        INDEX_t i, k, dim = A.shape[0]
        REAL_t[::1] r0, r, p, p2, s, s2, temp, temp2
        REAL_t kapppa, kappaNew, alpha, omega, beta

    if temporaryMemory is None:
        temporaryMemory = uninitialized((8*dim), dtype=REAL)

    else:
        if precond is not None:
            assert temporaryMemory.shape[0] >= 8*dim
        else:
            assert temporaryMemory.shape[0] >= 6*dim

    r0 = temporaryMemory[:dim]
    r = temporaryMemory[dim:2*dim]
    p = temporaryMemory[2*dim:3*dim]
    s = temporaryMemory[3*dim:4*dim]
    temp = temporaryMemory[4*dim:5*dim]
    temp2 = temporaryMemory[5*dim:6*dim]
    if precond is not None:
        p2 = temporaryMemory[6*dim:7*dim]
        s2 = temporaryMemory[7*dim:]
    else:
        p2 = p
        s2 = s


    if residuals is None:
        residuals = []
    else:
        assert len(residuals) == 0

    if x0 is None:
        for i in range(dim):
            x[i] = 0.
            p[i] = r[i] = r0[i] = b[i]
    else:
        for i in range(dim):
            x[i] = x0[i]
        A(x, temp)
        for i in range(dim):
            p[i] = r0[i] = r[i] = b[i] - temp[i]
    accumulate(r0)

    kappa = inner(r, r0, False, True)
    residuals.append(sqrt(kappa))
    for k in range(maxiter):
        if precond is not None:
            precond(p, p2)
        A(p2, temp)
        alpha = kappa / inner(temp, r0, False, True)
        for i in range(dim):
            s[i] = r[i]-alpha*temp[i]
        if precond is not None:
            precond(s, s2)
        A(s2, temp2)
        omega = inner(temp2, s, False, True) / norm(temp2, False)**2
        for i in range(dim):
            x[i] += alpha*p2[i] + omega*s2[i]
        for i in range(dim):
            r[i] = s[i] - omega*temp2[i]
        if use2norm:
            residuals.append(norm(r, False))
        else:
            raise NotImplementedError()
        if residuals[k+1] < tol:
            return
        kappaNew = inner(r, r0, False, True)
        beta = kappaNew/kappa * alpha/omega
        kappa = kappaNew
        for i in range(dim):
            p[i] = r[i] + beta*(p[i] - omega*temp[i])


cpdef int cg(LinearOperator A,
             REAL_t[::1] b,
             REAL_t[::1] x,
             REAL_t[::1] x0=None,
             REAL_t tol=1e-10,
             int maxiter=20,
             list residuals=None,
             LinearOperator precond=None,
             ipBase inner=ip_serial(),
             normBase norm=norm_serial(),
             BOOL_t use2norm=False,
             BOOL_t relativeTolerance=False):
    cdef:
        cg_solver solver = cg_solver(A)
        int numIter
    if precond is not None:
        solver.setPreconditioner(precond)
    solver.tolerance = tol
    solver.maxIter = maxiter
    solver.use2norm = use2norm
    solver.relativeTolerance = relativeTolerance
    if x0 is not None:
        solver.setInitialGuess(x0)
    solver.inner = inner
    solver.norm = norm
    solver.setup()
    numIter = solver.solve(b, x)
    if residuals is not None:
        residuals += solver.residuals
    return numIter


cpdef flexible_cg(A,
                  REAL_t[::1] b,
                  x0=None,
                  REAL_t tol=1e-10,
                  int maxiter=20,
                  residuals=None,
                  precond=None,
                  inner=ip_serial):
    cdef:
        np.ndarray[REAL_t, ndim=1] rold_mem = uninitialized(b.shape[0],
                                                         dtype=REAL)
        REAL_t beta, beta2, alpha
        REAL_t[::1] r, rold = rold_mem, p, Ap, Br, x
        int dim = b.shape[0]

        int i, j
    if x0 is None:
        x = b.copy()
    else:
        x = x0

    if residuals is None:
        residuals = []
    else:
        assert len(residuals) == 0

    # Krylov space spans whole solution space after dim-1 iterations
    maxiter = min(maxiter, dim)

    if precond is None:
        r = b - A.dot(x)
        p = r.copy()
        beta = sqrt(inner(r, r))
        residuals.append(beta)
        if not beta <= tol:
            beta2 = beta**2
            for i in range(maxiter):
                Ap = A.dot(np.array(p, copy=False, dtype=REAL))
                alpha = beta2/inner(p, Ap)
                for j in range(dim):
                    x[j] += alpha*p[j]
                    r[j] -= alpha*Ap[j]
                for j in range(dim):
                    rold[j] = r[j] - rold[j]
                beta = sqrt(inner(r, rold))
                residuals.append(beta)
                if beta <= tol:
                    break
                beta = beta**2
                for j in range(dim):
                    p[j] = r[j] + beta/beta2*p[j]
                beta2 = inner(r, r)
                for j in range(dim):
                    rold[j] = r[j]
        return np.array(x, copy=False, dtype=REAL)
    else:
        r = b - A*x
        p = precond*r
        beta2 = inner(r, p, False, True)
        residuals.append(sqrt(inner(r, r, False, False)))
        if not residuals[0] <= tol:
            for i in range(maxiter):
                Ap = A*np.array(p, copy=False, dtype=REAL)
                alpha = beta2/inner(p, Ap, True, False)
                for j in range(dim):
                    x[j] += alpha*p[j]
                    r[j] -= alpha*Ap[j]
                for j in range(dim):
                    rold[j] = r[j] - rold[j]
                Br = precond*np.array(r, copy=False, dtype=REAL)
                beta = inner(rold, Br, False, True)
                residuals.append(sqrt(inner(r, r, False, False)))
                if residuals[i+1] <= tol:
                    break
                for j in range(dim):
                    p[j] = Br[j] + beta/beta2*p[j]
                beta2 = inner(r, Br, False, True)
                for j in range(dim):
                    rold[j] = r[j]
        return np.array(x, copy=False, dtype=REAL)


cpdef int gmres(LinearOperator A,
                REAL_t[::1] b,
                REAL_t[::1] x,
                REAL_t[::1] x0=None,
                int maxiter=20,
                int restarts=1,
                REAL_t tol=1e-5,
                list residuals=None,
                LinearOperator Lprecond=None,
                LinearOperator Rprecond=None,
                ipBase inner=ip_serial(),
                normBase norm=norm_serial(),
                convergenceMaster convMaster=None,
                convergenceClient convClient=None,
                BOOL_t flexible=False,
                BOOL_t relativeTolerance=False):
    cdef:
        gmres_solver solver = gmres_solver(A)
        int numIter
    if Rprecond is not None:
        solver.setPreconditioner(Rprecond, False)
    elif Lprecond is not None:
        solver.setPreconditioner(Lprecond, True)
    solver.tolerance = tol
    solver.maxIter = maxiter
    solver.restarts = restarts
    solver.relativeTolerance = relativeTolerance
    solver.flexible = flexible
    if x0 is not None:
        solver.setInitialGuess(x0)
    solver.inner = inner
    solver.norm = norm
    solver.convMaster = convMaster
    solver.convClient = convClient
    solver.setup()
    numIter = solver.solve(b, x)
    if residuals is not None:
        residuals += solver.residuals
    return numIter


cpdef int gmresComplex(ComplexLinearOperator A,
                       COMPLEX_t[::1] b,
                       COMPLEX_t[::1] x,
                       COMPLEX_t[::1] x0=None,
                       int maxiter=20,
                       int restarts=1,
                       REAL_t tol=1e-5,
                       list residuals=None,
                       ComplexLinearOperator Lprecond=None,
                       ComplexLinearOperator Rprecond=None,
                       complexipBase inner=wrapRealInnerToComplex(ip_serial()),
                       complexNormBase norm=wrapRealNormToComplex(norm_serial()),
                       convergenceMaster convMaster=None,
                       convergenceClient convClient=None,
                       BOOL_t flexible=False,
                       BOOL_t relativeTolerance=False):
    cdef:
        complex_gmres_solver solver = complex_gmres_solver(A)
        int numIter
    if Rprecond is not None:
        solver.setPreconditioner(Rprecond, False)
    elif Lprecond is not None:
        solver.setPreconditioner(Lprecond, True)
    solver.tolerance = tol
    solver.maxIter = maxiter
    solver.restarts = restarts
    solver.relativeTolerance = relativeTolerance
    solver.flexible = flexible
    if x0 is not None:
        solver.setInitialGuess(x0)
    solver.setNormInner(norm, inner)
    solver.convMaster = convMaster
    solver.convClient = convClient
    solver.setup()
    numIter = solver.solve(b, x)
    if residuals is not None:
        residuals += solver.residuals
    return numIter


cpdef void bicgstabComplex(ComplexLinearOperator A,
                           const COMPLEX_t[::1] b,
                           COMPLEX_t[::1] x,
                           COMPLEX_t[::1] x0=None,
                           REAL_t tol=1e-10,
                           int maxiter=20,
                           list residuals=None,
                           ComplexLinearOperator precond=None,
                           complexipBase inner=wrapRealInnerToComplex(ip_serial()),
                           complexNormBase norm=wrapRealNormToComplex(norm_serial()),
                           BOOL_t use2norm=True,
                           COMPLEX_t[::1] temporaryMemory=None):
    """
    Stabilized Biconjugate Gradient iteration.

    In a distributed solve, we want:
    A:        accumulated to distributed
    precond:  distributed to accumulated
    b:        distributed
    x:        accumulated
    x0:       accumulated

    In the unpreconditioned distributed case, set precond to accumulate.

    If use2norm is False, use Preconditioner norm of residual as
    stopping criterion, otherwise use 2-norm of residual.

    Memory requirement:
    8*dim with preconditioner,
    6*dim without.
    """

    cdef:
        INDEX_t i, k, dim = A.shape[0]
        COMPLEX_t[::1] r0, r, p, p2, s, s2, temp, temp2
        COMPLEX_t kapppa, kappaNew, alpha, omega, beta

    if temporaryMemory is None:
        temporaryMemory = uninitialized((8*dim), dtype=COMPLEX)

    else:
        if precond is not None:
            assert temporaryMemory.shape[0] >= 8*dim
        else:
            assert temporaryMemory.shape[0] >= 6*dim

    r0 = temporaryMemory[:dim]
    r = temporaryMemory[dim:2*dim]
    p = temporaryMemory[2*dim:3*dim]
    s = temporaryMemory[3*dim:4*dim]
    temp = temporaryMemory[4*dim:5*dim]
    temp2 = temporaryMemory[5*dim:6*dim]
    if precond is not None:
        p2 = temporaryMemory[6*dim:7*dim]
        s2 = temporaryMemory[7*dim:]
    else:
        p2 = p
        s2 = s


    if residuals is None:
        residuals = []

    if x0 is None:
        for i in range(dim):
            x[i] = 0.
            p[i] = r[i] = r0[i] = b[i]
    else:
        for i in range(dim):
            x[i] = x0[i]
        A(x, temp)
        for i in range(dim):
            p[i] = r0[i] = r[i] = b[i] - temp[i]
    # accumulate(r0)

    kappa = inner(r0, r, False, False)
    residuals.append(sqrt(abs(kappa)))
    for k in range(maxiter):
        if precond is not None:
            precond(p, p2)
        A(p2, temp)
        alpha = kappa / inner(r0, temp, False, False)
        for i in range(dim):
            s[i] = r[i]-alpha*temp[i]
        if precond is not None:
            precond(s, s2)
        A(s2, temp2)
        omega = inner(temp2, s, False, True) / norm(temp2, False)**2
        for i in range(dim):
            x[i] = x[i] + alpha*p2[i] + omega*s2[i]
        for i in range(dim):
            r[i] = s[i] - omega*temp2[i]
        if use2norm:
            residuals.append(norm(r, False))
        else:
            raise NotImplementedError()
        if residuals[k+1] < tol:
            return
        kappaNew = inner(r0, r, False, False)
        beta = kappaNew/kappa * alpha/omega
        kappa = kappaNew
        for i in range(dim):
            p[i] = r[i] + beta*(p[i] - omega*temp[i])


def estimateSpectralRadius(LinearOperator A,
                           normBase norm=norm_serial(),
                           REAL_t eps=1e-3,
                           INDEX_t kMax=100):
    """
    Estimate the absolute value of the largest eigenvalue
    using the power method.
    """
    x = UniformOnUnitSphere(A.shape[0])
    lold = 0
    l = 1
    k = 0
    while np.absolute(l-lold) > eps and k <= kMax:
        x = A.dot(x)
        lold = l
        l = norm(x, False)
        x /= l
        k += 1
    return l


cpdef arnoldi(LinearOperator A,
              REAL_t[::1] x0=None,
              int maxiter=20,
              REAL_t tol=1e-10,
              LinearOperator Lprecond=None,
              LinearOperator Rprecond=None,
              ipBase inner=ip_serial(),
              normBase norm=norm_serial(),
              REAL_t[::1] temporaryMemory=None,
              REAL_t[::1, :] temporaryMemoryQ=None,
              REAL_t[::1, :] temporaryMemoryH=None):
    """
    GMRES iteration.

    In a distributed solve, we want:
    A:        accumulated to distributed
    Lprecond:  distributed to accumulated
    b:        distributed
    x0:       accumulated
    x:        accumulated

    In the unpreconditioned distributed case, set Lprecond to accumulate.

    Memory requirement:
    dim * (maxiter+1) for Q
    (maxiter+1) * maxiter for H
    (4*maxiter + 2) + 2*dim for c,s,gamma,y and r,Ar
    """
    cdef:
        int i = -1, j, dim = A.shape[0], l
        REAL_t[::1, :] Q, H
        REAL_t[::1] r, Ar

    if temporaryMemory.shape[0] >= 2*dim:
        r = temporaryMemory[:dim]
        Ar = temporaryMemory[dim:2*dim]
    else:
        r = uninitialized((dim), dtype=REAL)
        Ar = uninitialized((dim), dtype=REAL)
    if ((temporaryMemoryQ.shape[0] == dim) and
        (temporaryMemoryQ.shape[1] >= maxiter+1)):
        Q = temporaryMemoryQ
    else:
        Q = uninitialized((dim, maxiter+1), dtype=REAL, order='F')
    if ((temporaryMemoryH.shape[0] >= maxiter+1) and
        (temporaryMemoryH.shape[1] >= maxiter)):
        H = temporaryMemoryH
    else:
        H = np.zeros((maxiter+1, maxiter), dtype=REAL, order='F')

    if x0 is None:
        x0 = np.random.rand(dim)
    for j in range(dim):
        r[j] = x0[j]

    for l in range(dim):
        Q[l, 0] = r[l]/norm(r, True)                                      # acc
    for i in range(maxiter):
        ##############################
        # Arnoldi iteration
        for l in range(dim):
            r[l] = Q[l, i]                                           # acc
        if Rprecond:
            # FIX: Use inplace multiplication
            r = Rprecond*np.array(r, copy=False, dtype=REAL)
        A(r, Ar)                                                     # dist
        if Lprecond:
            Lprecond(Ar, r)                                          # acc
        else:
            for j in range(dim):
                r[j] = Ar[j]
        for j in range(i+1):
            H[j, i] = inner(Q[:, j], r, True, True)
            for l in range(dim):
                r[l] -= H[j, i]*Q[l, j]                              # acc
        H[i+1, i] = norm(r, True)
        if abs(H[i+1, i]) > tol:
            for l in range(dim):
                Q[l, i+1] = r[l]/H[i+1, i]                           # acc
        else:
            return np.array(H, dtype=REAL)[:i+1, :i]
    return np.array(H, dtype=REAL)


def lanczos(A, x=None, numIter=5):
    norm = np.linalg.norm
    inner = np.vdot
    if x is None:
        x = np.ones((A.shape[0]))/np.sqrt(A.shape[0])
    else:
        x = x/norm(x)
    H = uninitialized((2, numIter))
    w = A*x
    H[1, 0] = inner(w, x)
    w -= H[1, 0]*x
    # alpha[0] = inner(w, x)
    # w -= alpha[0]*x
    for m in range(1, numIter):
        # beta[m-1] = norm(w)
        # if abs(beta[m-1]) < 1e-10:
        #     break
        # xold = x
        # x = w/beta[m-1]
        H[0, m] = norm(w)
        if abs(H[0, m]) < 1e-10:
            H = H[:, :m]
            break
        xold = x
        x = w/H[0, m]

        # w = A*x-beta[m-1]*xold
        # alpha[m] = inner(w, x)
        # w -= alpha[m]*x
        w = A*x-H[0, m]*xold
        H[1, m] = inner(w, x)
        w -= H[1, m]*x
    return H

def lanczos2(A, M, Minv, x=None, numIter=5):
    z = uninitialized((A.shape[0]))
    inner = np.vdot
    if x is None:
        x = np.ones((A.shape[0]))
    x /= np.sqrt(inner(x, M*x))
    H = uninitialized((2, numIter))
    w = A*x
    H[1, 0] = inner(w, x)
    w -= H[1, 0]*(M*x)
    # alpha[0] = inner(w, x)
    # w -= alpha[0]*x
    for m in range(1, numIter):
        # beta[m-1] = norm(w)
        # if abs(beta[m-1]) < 1e-10:
        #     break
        # xold = x
        # x = w/beta[m-1]
        # z = np.linalg.solve(M.toarray(), w)
        Minv(w, z)
        H[0, m] = np.sqrt(inner(w, z))
        if abs(H[0, m]) < 1e-10:
            H = H[:, :m]
            break
        xold = x
        x = z/H[0, m]

        # w = A*x-beta[m-1]*xold
        # alpha[m] = inner(w, x)
        # w -= alpha[m]*x
        w = A*x-H[0, m]*(M*xold)
        H[1, m] = inner(w, x)
        w -= H[1, m]*(M*x)
    return H
