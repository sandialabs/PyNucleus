###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from . myTypes import REAL, COMPLEX
from . blas cimport assign, assignScaled, assign3, update, updateScaled, gemvF
from . blas import uninitialized
from . ip_norm cimport vector_t, complex_vector_t, ip_serial, norm_serial, ip_distributed, norm_distributed, wrapRealInnerToComplex, wrapRealNormToComplex
from . linalg import ichol_csr
from . linalg cimport (forward_solve_csc, backward_solve_csc,
                       forward_solve_sss_noInverse,
                       backward_solve_sss_noInverse)


cdef class solver:
    def __init__(self, LinearOperator A=None, INDEX_t num_rows=-1):
        self.initialized = False
        self.PLogger = FakePLogger()
        if A is not None:
            self.A = A
            self.num_rows = A.num_rows
        else:
            self.A = None
            assert num_rows >= 0, 'num_rows < 0'
            self.num_rows = num_rows

    cpdef void setup(self, LinearOperator A=None):
        raise NotImplementedError()

    def __call__(self, vector_t b, vector_t x):
        return self.solve(b, x)

    cdef int solve(self, vector_t b, vector_t x) except -1:
        assert self.initialized, 'Solver not initialized, need to call \'solver.setup\' first.'
        assert b.shape[0] == self.num_rows, \
            'RHS vector has size {}, solver expects {}'.format(b.shape[0],
                                                               self.num_rows)
        assert x.shape[0] == self.num_rows, \
            'x vector has size {}, solver expects {}'.format(x.shape[0],
                                                             self.num_rows)

    def asPreconditioner(self):
        return preconditioner(self)

    def __repr__(self):
        return str(self)


cdef class preconditioner(LinearOperator):
    def __init__(self, solver solOp, dict ctxAttrs={}):
        LinearOperator.__init__(self, solOp.num_rows, solOp.num_rows)
        self.solOp = solOp
        self.ctxAttrs = ctxAttrs

    cdef INDEX_t matvec(self,
                        vector_t x,
                        vector_t y) except -1:
        assert self.solOp.initialized, 'solOp not initialized'
        self.solOp.solve(x, y)
        return 0

    def __str__(self):
        return str(self.solOp)


cdef class noop_solver(solver):
    def __init__(self, LinearOperator A, INDEX_t num_rows=-1):
        solver.__init__(self, A, num_rows)

    cpdef void setup(self, LinearOperator A=None):
        self.initialized = True


cdef class lu_solver(solver):
    def __init__(self, LinearOperator A, INDEX_t num_rows=-1):
        solver.__init__(self, A, num_rows)

    cpdef void setup(self, LinearOperator A=None):
        cdef:
            INDEX_t i, j, explicitZeros, explicitZerosRow
            REAL_t[:, ::1] data
            LinearOperator B = None

        if A is not None:
            self.A = A

        if not isinstance(self.A, (SSS_LinearOperator,
                                   CSR_LinearOperator,
                                   Dense_LinearOperator)):
            if self.A.isSparse():
                B = self.A.to_csr_linear_operator()
            else:
                B = Dense_LinearOperator(np.ascontiguousarray(self.A.toarray()))
        else:
            B = self.A
        try_sparsification = False
        sparsificationThreshold = 0.9
        if isinstance(B, Dense_LinearOperator) and try_sparsification:
            explicitZeros = 0
            data = B.data
            for i in range(B.num_rows):
                explicitZerosRow = 0
                for j in range(B.num_columns):
                    if data[i, j] == 0.:
                        explicitZerosRow += 1
                explicitZeros += explicitZerosRow
                if not (explicitZerosRow > sparsificationThreshold*B.num_columns):
                    break
            if explicitZeros > sparsificationThreshold*B.num_rows*B.num_columns:
                print('Converting dense to sparse matrix, since {}% of entries are zero.'.format(100.*explicitZeros/REAL(B.num_rows*B.num_columns)))
                B = CSR_LinearOperator.from_dense(B)
        self.useTriangularSolveRoutines = False
        if isinstance(B, (SSS_LinearOperator,
                          CSR_LinearOperator)):
            from scipy.sparse.linalg import splu
            try:
                if isinstance(B, SSS_LinearOperator):
                    Ainv = splu(B.to_csc())
                else:
                    Ainv = splu(B.to_csr().tocsc())
            except RuntimeError:
                print(B, np.array(B.data))
                raise
            try:
                self.L = CSR_LinearOperator.from_csr(Ainv.L)
                self.U = CSR_LinearOperator.from_csr(Ainv.U)
                self.perm_r = Ainv.perm_r
                self.perm_c = Ainv.perm_c
                n = self.perm_c.shape[0]
                self.useTriangularSolveRoutines = True
                self.temp_mem = uninitialized((n), dtype=REAL)
            except AttributeError:
                self.Ainv = Ainv
        elif isinstance(B, Dense_LinearOperator):
            from scipy.linalg import lu_factor
            self.lu, self.perm = lu_factor(B.data)
        else:
            raise NotImplementedError('Cannot use operator of type "{}"'.format(type(B)))
        self.initialized = True

    cdef int solve(self, vector_t b, vector_t x) except -1:
        cdef:
            INDEX_t i, n
            INDEX_t[::1] perm_r, perm_c
            vector_t temp
        solver.solve(self, b, x)
        if self.useTriangularSolveRoutines:
            perm_r = self.perm_r
            perm_c = self.perm_c
            try:
                temp = self.temp_mem
                n = perm_c.shape[0]
                for i in range(n):
                    temp[perm_r[i]] = b[i]
                x[:] = 0.
                forward_solve_csc(self.L.indptr, self.L.indices, self.L.data,
                                  temp, x,
                                  unitDiagonal=True)
                temp[:] = 0.
                backward_solve_csc(self.U.indptr, self.U.indices, self.U.data, x, temp)
                for i in range(n):
                    x[i] = temp[perm_c[i]]
            except AttributeError:
                x[:] = self.Ainv.solve(np.array(b, copy=False, dtype=REAL))
        else:
            from scipy.linalg import lu_solve
            assign(x, b)
            lu_solve((self.lu, self.perm),
                     np.array(x, copy=False, dtype=REAL),
                     overwrite_b=True)
        return 1

    def __str__(self):
        return 'LU'


include "solver_pypardiso.pxi"
include "solver_chol.pxi"
include "solver_ichol.pxi"


cdef class ilu_solver(solver):
    def __init__(self, LinearOperator A=None, INDEX_t num_rows=-1):
        solver.__init__(self, A, num_rows)
        self.temp1 = uninitialized((self.num_rows), dtype=REAL)
        self.temp2 = uninitialized((self.num_rows), dtype=REAL)
        self.fill_factor = 1.

    cpdef void setup(self, LinearOperator A=None):
        from scipy.sparse.linalg import spilu
        Clu = spilu(A.to_csr().tocsc(), fill_factor=self.fill_factor)
        self.Lindices = Clu.L.indices
        self.Lindptr = Clu.L.indptr
        self.Ldata = Clu.L.data
        self.Uindices = Clu.U.indices
        self.Uindptr = Clu.U.indptr
        self.Udata = Clu.U.data
        self.perm_r = Clu.perm_r
        self.perm_c = Clu.perm_c
        self.initialized = True

    cdef int solve(self, vector_t b, vector_t x) except -1:
        solver.solve(self, b, x)
        cdef:
            INDEX_t i
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
        return 1

    def __str__(self):
        return 'Incomplete LU'


cdef class jacobi_solver(solver):
    def __init__(self, LinearOperator A=None, INDEX_t num_rows=-1):
        solver.__init__(self, A, num_rows)

    cpdef void setup(self, LinearOperator A=None):
        if A is not None:
            self.A = A
        self.invD = invDiagonal(self.A)
        self.initialized = True

    cdef int solve(self, vector_t b, vector_t x) except -1:
        solver.solve(self, b, x)
        self.invD.matvec(b, x)
        return 1

    def __str__(self):
        return 'Jacobi'


cdef class iterative_solver(solver):
    def __init__(self, LinearOperator A=None, INDEX_t num_rows=-1):
        solver.__init__(self, A, num_rows)
        self.residuals = []
        self.setNormInner(norm_serial(), ip_serial())
        self.maxIter = -1
        self.tolerance = 1e-5
        self.relativeTolerance = False
        self.x0 = None

    cpdef void setInitialGuess(self, vector_t x0=None):
        if x0 is not None:
            assert self.num_rows == x0.shape[0], \
                'x0 vector has size {}, solver expects {}'.format(x0.shape[0],
                                                                  self.num_rows)
            self.x0 = x0
        else:
            self.x0 = None

    cpdef void setNormInner(self, normBase norm, ipBase inner):
        self.norm = norm
        self.inner = inner

    cpdef void setOverlapNormInner(self, object overlaps, level=-1):
        self.setNormInner(norm_distributed(overlaps, level),
                          ip_distributed(overlaps, level))

    cpdef void setup(self, LinearOperator A=None):
        if A is not None:
            assert A.num_rows == self.num_rows, \
                'A has {} rows, but solver expects {}.'.format(A.num_rows, self.num_rows)
            self.A = A
        else:
            assert self.A is not None, 'A not set'
        self.r = uninitialized((self.num_rows), dtype=REAL)

    cdef int solve(self, vector_t b, vector_t x) except -1:
        cdef:
            REAL_t res
            vector_t r = self.r
            LinearOperator A = self.A
            normBase norm = self.norm

        solver.solve(self, b, x)

        if self.x0 is None:
            for i in range(self.num_rows):
                x[i] = 0.
        elif &x[0] != &self.x0[0]:
            assign(x, self.x0)

        if self.relativeTolerance:
            A.residual(x, b, r, simpleResidual=self.x0 is None)   # dist
            res = norm.eval(r, False)                             # ip(dist, dist)
            self.tol = self.tolerance*res
        else:
            self.tol = self.tolerance


cdef class krylov_solver(iterative_solver):
    def __init__(self, LinearOperator A=None, INDEX_t num_rows=-1):
        iterative_solver.__init__(self, A, num_rows)
        self.prec = None
        self.convMaster = None
        self.convClient = None

    cpdef void setup(self, LinearOperator A=None):
        iterative_solver.setup(self, A)
        if self.prec is not None and isinstance(self.prec, preconditioner) and (not self.prec.solOp.initialized or A is not None):
            self.prec.solOp.setup(self.A)

    cpdef void setPreconditioner(self, LinearOperator prec, BOOL_t left=True):
        self.prec = prec
        self.isLeftPrec = left

    cdef int solve(self, vector_t b, vector_t x) except -1:
        iterative_solver.solve(self, b, x)
        if not self.relativeTolerance:
            self.A.residual(x, b, self.r, simpleResidual=self.x0 is None)   # dist


cdef class cg_solver(krylov_solver):
    """
    Conjugate Gradient iteration.

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
    4*dim with preconditioner,
    3*dim without.
    """

    def __init__(self, LinearOperator A=None, INDEX_t num_rows=-1):
        krylov_solver.__init__(self, A)
        self.use2norm = False
        self.maxIter = 50

    cpdef void setup(self, LinearOperator A=None):
        krylov_solver.setup(self, A)
        self.temporaryMemory = uninitialized((2*self.num_rows), dtype=REAL)
        self.initialized = True

    cpdef void setPreconditioner(self, LinearOperator prec, BOOL_t left=True):
        krylov_solver.setPreconditioner(self, prec, left)
        self.precTemporaryMemory = uninitialized(self.num_rows, dtype=REAL)

    cdef int solve(self, vector_t b, vector_t x) except -1:
        krylov_solver.solve(self, b, x)

        cdef:
            REAL_t beta, betaOld, alpha, temp
            INDEX_t dim = self.num_rows
            vector_t r = self.r
            vector_t p = self.temporaryMemory[:dim]
            vector_t Ap = self.temporaryMemory[dim:2*dim]
            vector_t Br
            INDEX_t i, k = 0
            list residuals = []
            ipBase inner = self.inner
            normBase norm = self.norm
            LinearOperator A = self.A
            LinearOperator precond = None
            REAL_t tol = self.tol
            INDEX_t maxiter = self.maxIter
            BOOL_t use2norm = self.use2norm
            REAL_t convCrit = 0.

        if self.prec is not None:
            precond = self.prec
            Br = self.precTemporaryMemory
        else:
            Br = None

        # Krylov space spans whole solution space after dim-1 iterations
        # Don't do this, doesn't work for distributed problem
        # maxiter = min(maxiter, dim)

        if precond is None:
            assign(p, r)
            betaOld = inner.eval(r, p, True, True)                              # ip(dist, acc)
            convCrit = sqrt(betaOld)
        else:
            precond.matvec(r, p)                                                # acc
            betaOld = inner.eval(r, p, False, True)                             # ip(dist, acc)
            if use2norm:
                convCrit = norm.eval(r, False)                                  # ip(dist, dist)
            else:
                convCrit = sqrt(betaOld)
        residuals.append(convCrit)
        if convCrit <= tol:
            self.residuals = residuals
            return 0
        for i in range(maxiter):
            A.matvec(p, Ap)                                            # dist
            alpha = betaOld/inner.eval(p, Ap, True, False)             # ip(acc, dist)
            updateScaled(x, p, alpha)                                  # acc
            updateScaled(r, Ap, -alpha)                                # dist
            if k == 50:
                # recalculate residual to avoid rounding errors
                A.residual(x, b, r)                                    # dist
                k = 0
            if precond is None:
                beta = norm.eval(r, True)                                   # ip(dist, dist)
                convCrit = beta
                residuals.append(convCrit)
                if convCrit <= tol:
                    self.residuals = residuals
                    return i
                beta = beta**2
                temp = beta/betaOld
                assign3(p, p, temp, r, 1.0)                            # acc
            else:
                precond.matvec(r, Br)                                  # acc
                beta = inner.eval(r, Br, False, True)                       # ip(dist, acc)
                if use2norm:
                    convCrit = norm.eval(r, False)                     # ip(dist, dist)
                else:
                    convCrit = sqrt(beta)
                residuals.append(convCrit)
                if convCrit <= tol:
                    self.residuals = residuals
                    return i
                temp = beta/betaOld
                assign3(p, p, temp, Br, 1.0)                           # acc
            betaOld = beta
            k += 1
        self.residuals = residuals
        return maxiter

    def __str__(self):
        s = 'CG(tolerance={},relTol={},maxIter={},2-norm={})'.format(self.tolerance, self.relativeTolerance, self.maxIter, self.use2norm)
        if self.prec is not None:
            if self.isLeftPrec:
                return s+', left preconditioned by '+str(self.prec)
            else:
                return s+', right preconditioned by '+str(self.prec)
        else:
            return s


cdef class gmres_solver(krylov_solver):
    """
    GMRES iteration.

    In a distributed solve, we want:
    A:        accumulated to distributed
    Lprecond:  distributed to accumulated
    Rprecond:  distributed to accumulated
    b:        distributed
    x0:       accumulated
    x:        accumulated

    In the unpreconditioned distributed case, set Lprecond to accumulate.

    Memory requirement:
    dim * (maxiter+1) for Q
    (maxiter+1) * maxiter for H
    (4*maxiter + 2) + 2*dim for c,s,gamma,y and r, Ar
    dim * (maxiter+1) for Z if flexible
    """

    def __init__(self, LinearOperator A=None, INDEX_t num_rows=-1):
        krylov_solver.__init__(self, A)
        self.use2norm = False
        self.flexible = False
        self.restarts = 1

    cpdef void setup(self, LinearOperator A=None):
        krylov_solver.setup(self, A)
        self.Ar = uninitialized((self.num_rows), dtype=REAL)
        assert self.maxIter > 0, 'Need maxiter > 0'
        self.H = np.ones((self.maxIter+1, self.maxIter), dtype=REAL, order='F')
        # set first dim to 1, not 0, so that things work
        d = max(self.num_rows, 1)
        self.Q = uninitialized((d, self.maxIter+1), dtype=REAL, order='F')
        if self.flexible:
            self.Z = uninitialized((d, self.maxIter+1), dtype=REAL, order='F')
        self.c = uninitialized((self.maxIter), dtype=REAL)
        self.s = uninitialized((self.maxIter), dtype=REAL)
        self.gamma = uninitialized((self.maxIter+1), dtype=REAL)
        self.y = uninitialized((self.maxIter+1), dtype=REAL)
        self.initialized = True

    cpdef void setPreconditioner(self, LinearOperator prec, BOOL_t left=True):
        krylov_solver.setPreconditioner(self, prec, left)

    cdef int solve(self, vector_t b, vector_t x) except -1:
        krylov_solver.solve(self, b, x)

        cdef:
            int k, i = -1, j, dim = self.num_rows, l
            REAL_t eps = 1e-15, beta, temp, rho, sigma
            REAL_t[::1, :] Q = self.Q
            REAL_t[::1, :] H = self.H
            REAL_t[::1, :] Z = None
            REAL_t[::1] c = self.c, s = self.s, gamma = self.gamma
            vector_t y = self.y
            vector_t r = self.r
            vector_t Ar = self.Ar
            BOOL_t breakout = False
            BOOL_t converged
            BOOL_t doLprecond = self.isLeftPrec and self.prec is not None
            BOOL_t doRprecond = not self.isLeftPrec and self.prec is not None
            LinearOperator A = self.A
            LinearOperator Lprecond = None, Rprecond = None
            list residuals = []
            ipBase inner = self.inner
            normBase norm = self.norm
            REAL_t tol = self.tol
            INDEX_t maxiter = self.maxIter
            INDEX_t restarts = self.restarts
            BOOL_t flexible = self.flexible
            int allIter = 0
            convergenceMaster convMaster=self.convMaster
            convergenceClient convClient=self.convClient

        if doLprecond:
            Lprecond = self.prec
        if doRprecond:
            Rprecond = self.prec
        if flexible:
            Z = self.Z

        for k in range(restarts):
            if breakout:
                self.residuals = residuals
                return allIter
            A.matvec(x, Ar)                                                  # dist
            if doLprecond:
                assign3(Ar, Ar, -1.0, b, 1.0)                                # dist
                Lprecond.matvec(Ar, r)                                       # acc
                gamma[0] = norm.eval(r, True)
            else:
                assign3(r, b, 1.0, Ar, -1.0)                                 # dist
                gamma[0] = norm.eval(r, False)
            if len(residuals) == 0:
                residuals.append(abs(gamma[0]))
            converged = abs(gamma[0]) < tol
            if convMaster is not None:
                convMaster.setStatus(converged)
            if convClient is not None:
                converged = convClient.getStatus()
            if converged:
                self.residuals = residuals
                return allIter
            assignScaled(Q[:, 0], r, 1./gamma[0])                            # acc for Lprecond, dist for Rprecond
            for i in range(maxiter):
                ##############################
                # Arnoldi iteration
                assign(r, Q[:, i])                                           # acc for Lprecond, dist for Rprecond
                if not flexible:
                    if doLprecond:
                        A.matvec(r, Ar)                                      # dist
                        Lprecond.matvec(Ar, r)                               # acc
                    elif doRprecond:
                        Rprecond.matvec(r, Ar)                               # acc
                        A.matvec(Ar, r)                                      # dist
                    else:
                        A.matvec(r, Ar)
                        assign(r, Ar)
                else:
                    if doLprecond:
                        A.matvec(r, Z[:, i])                                 # dist
                        Lprecond.matvec(Z[:, i], r)                          # acc
                    elif doRprecond:
                        Rprecond.matvec(r, Z[:, i])                          # acc
                        A.matvec(Z[:, i], r)                                 # dist
                    else:
                        A.matvec(r, Z[:, i])
                        assign(r, Z[:, i])
                if doRprecond:
                    if dim > 0:
                        for j in range(i+1):
                            H[j, i] = inner.eval(Q[:, j], r, False, False)
                            updateScaled(r, Q[:, j], -H[j, i])               # dist
                    H[i+1, i] = norm.eval(r, False)
                else:
                    if dim > 0:
                        for j in range(i+1):
                            H[j, i] = inner.eval(Q[:, j], r, True, True)
                            updateScaled(r, Q[:, j], -H[j, i])               # acc
                    H[i+1, i] = norm.eval(r, True)
                converged = abs(H[i+1, i]) > eps
                if convMaster is not None:
                    convMaster.setStatus(converged)
                if convClient is not None:
                    converged = convClient.getStatus()
                if converged:
                    assignScaled(Q[:, i+1], r, 1./H[i+1, i])                 # acc for Lprecond, dist for Rprecond
                else:
                    breakout = True
                    break
                ##############################
                # Apply previous Givens rotations to last column of H
                for j in range(i):
                    rho = H[j, i]
                    sigma = H[j+1, i]
                    H[j, i] = c[j]*rho + s[j]*sigma
                    H[j+1, i] = -s[j]*rho + c[j]*sigma
                ##############################
                # determine new Givens rotation
                beta = sqrt(H[i, i]**2 + H[i+1, i]**2)
                c[i] = H[i, i]/beta
                s[i] = H[i+1, i]/beta
                ##############################
                # Apply new Givens rotation to H
                H[i, i] = beta
                # H[i+1, i] = 0.0
                ##############################
                # Apply new Givens rotation to rhs
                gamma[i+1] = -s[i]*gamma[i]
                gamma[i] = c[i]*gamma[i]
                residuals.append(abs(gamma[i+1]))
                converged = abs(gamma[i+1]) < tol
                if convMaster is not None:
                    convMaster.setStatus(converged)
                if convClient is not None:
                    converged = convClient.getStatus()
                if converged:
                    breakout = True
                    break
            allIter += i
            ##############################
            # perform back-solve for y
            for j in range(i, -1, -1):
                temp = gamma[j]
                for l in range(j+1, i+1):
                    temp -= H[j, l]*y[l]
                y[j] = temp/H[j, j]
            ##############################
            # update x
            if not flexible:
                gemvF(Q[:, :i+1], y[:i+1], r)                                # acc for Lprecond, dist for Rprecond
                if doRprecond:
                    Rprecond.matvec(r, Ar)                                   # acc
                    update(x, Ar)                                            # acc
                else:
                    update(x, r)                                             # acc
            else:
                gemvF(Z[:, :i+1], y[:i+1], r)                                # dist for Lprecond, acc for Rprecond
                assert not doLprecond                                        # TODO: figure out what to do here
                update(x, r)                                                 # acc
        self.residuals = residuals
        return allIter

    def __str__(self):
        s = 'GMRES(tolerance={},relTol={},maxIter={},restarts={},2-norm={},flexible={})'.format(self.tolerance, self.relativeTolerance, self.maxIter,
                                                                                                self.restarts, self.use2norm, self.flexible)
        if self.prec is not None:
            if self.isLeftPrec:
                return s+', left preconditioned by '+str(self.prec)
            else:
                return s+', right preconditioned by '+str(self.prec)
        else:
            return s


cdef class bicgstab_solver(krylov_solver):
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

    def __init__(self, LinearOperator A=None, INDEX_t num_rows=-1):
        krylov_solver.__init__(self, A)
        self.use2norm = True
        self.maxIter = 50

    cpdef void setup(self, LinearOperator A=None):
        krylov_solver.setup(self, A)
        self.r0 = uninitialized((self.num_rows), dtype=REAL)
        self.p = uninitialized((self.num_rows), dtype=REAL)
        self.s = uninitialized((self.num_rows), dtype=REAL)
        self.temp = uninitialized((self.num_rows), dtype=REAL)
        self.temp2 = uninitialized((self.num_rows), dtype=REAL)
        self.initialized = True

    cpdef void setPreconditioner(self, LinearOperator prec, BOOL_t left=True):
        krylov_solver.setPreconditioner(self, prec, left)
        self.p2 = uninitialized(self.num_rows, dtype=REAL)
        self.s2 = uninitialized(self.num_rows, dtype=REAL)

    cdef int solve(self, vector_t b, vector_t x) except -1:
        krylov_solver.solve(self, b, x)

        cdef:
            INDEX_t i, k, dim = self.num_rows
            vector_t r0 = self.r0
            vector_t r = self.r
            vector_t p = self.p
            vector_t p2
            vector_t s = self.s
            vector_t s2
            vector_t temp = self.temp
            vector_t temp2 = self.temp2
            REAL_t kappa, kappaNew, alpha, omega, beta, tt
            list residuals = []
            REAL_t resNorm
            ipBase inner = self.inner
            normBase norm = self.norm
            REAL_t tol = self.tol
            INDEX_t maxiter = self.maxIter
            BOOL_t use2norm = self.use2norm
            LinearOperator A = self.A
            LinearOperator precond

        if self.prec is not None:
            precond = self.prec
            p2 = self.p2
            s2 = self.s2
        else:
            precond = None
            p2 = p
            s2 = s

        if precond is not None:
            # need an accumulated vector for r0
            assign(p, r)
            precond.matvec(r, r0)
        else:
            for i in range(dim):
                p[i] = r0[i] = r[i]

        kappa = inner.eval(r, r0, False, True)
        resNorm = sqrt(kappa)
        residuals.append(resNorm)
        for k in range(maxiter):
            if precond is not None:
                precond.matvec(p, p2)
            A.matvec(p2, temp)
            tt = inner.eval(temp, r0, False, True)
            alpha = kappa / tt
            assign3(s, r, 1.0, temp, -alpha)
            if precond is not None:
                precond.matvec(s, s2)
            A.matvec(s2, temp2)
            omega = inner.eval(temp2, s, False, False) / norm.eval(temp2, False)**2
            for i in range(dim):
                x[i] += alpha*p2[i] + omega*s2[i]
            assign3(r, s, 1.0, temp2, -omega)
            if use2norm:
                resNorm = norm.eval(r, False)
                residuals.append(resNorm)
            else:
                raise NotImplementedError()
            if resNorm < tol:
                self.residuals = residuals
                return k
            kappaNew = inner.eval(r, r0, False, True)
            beta = kappaNew/kappa * alpha/omega
            kappa = kappaNew
            for i in range(dim):
                p[i] = r[i] + beta*(p[i] - omega*temp[i])
        self.residuals = residuals
        return maxiter

    def __str__(self):
        s = 'BiCGStab(tolerance={},relTol={},maxIter={},2-norm={})'.format(self.tolerance, self.relativeTolerance, self.maxIter, self.use2norm)
        if self.prec is not None:
            if self.isLeftPrec:
                return s+', left preconditioned by '+str(self.prec)
            else:
                return s+', right preconditioned by '+str(self.prec)
        else:
            return s


######################################################################


cdef class complex_solver:
    def __init__(self, ComplexLinearOperator A=None, INDEX_t num_rows=-1):
        self.initialized = False
        self.PLogger = FakePLogger()
        if A is not None:
            self.A = A
            self.num_rows = A.num_rows
        else:
            self.A = None
            assert num_rows >= 0, 'num_rows < 0'
            self.num_rows = num_rows

    cpdef void setup(self, ComplexLinearOperator A=None):
        raise NotImplementedError()

    def __call__(self, complex_vector_t b, complex_vector_t x):
        return self.solve(b, x)

    cdef int solve(self, complex_vector_t b, complex_vector_t x) except -1:
        assert self.initialized, 'Solver not initialized, need to call \'solver.setup\' first.'
        assert b.shape[0] == self.num_rows, \
            'RHS vector has size {}, solver expects {}'.format(b.shape[0],
                                                               self.num_rows)
        assert x.shape[0] == self.num_rows, \
            'x vector has size {}, solver expects {}'.format(x.shape[0],
                                                             self.num_rows)

    def asPreconditioner(self):
        return preconditioner(self)


cdef class complex_preconditioner(ComplexLinearOperator):
    def __init__(self, complex_solver solOp, dict ctxAttrs={}):
        ComplexLinearOperator.__init__(self, solOp.num_rows, solOp.num_rows)
        self.solOp = solOp
        self.ctxAttrs = ctxAttrs

    cdef INDEX_t matvec(self,
                        complex_vector_t x,
                        complex_vector_t y) except -1:
        assert self.solOp.initialized, 'solOp not initialized'
        self.solOp.solve(x, y)
        return 0

    def __str__(self):
        return str(self.solOp)


cdef class complex_lu_solver(complex_solver):
    def __init__(self, ComplexLinearOperator A, INDEX_t num_rows=-1):
        complex_solver.__init__(self, A, num_rows)

    cpdef void setup(self, ComplexLinearOperator A=None):
        if A is not None:
            self.A = A

        if isinstance(self.A, ComplexDense_LinearOperator):
            from scipy.linalg import lu_factor
            self.lu, self.perm = lu_factor(self.A.data)
        elif isinstance(self.A, (ComplexLinearOperator, HelmholtzShiftOperator)):
            from scipy.sparse.linalg import splu
            try:
                if isinstance(self.A, ComplexSSS_LinearOperator):
                    Ainv = splu(self.A.to_csc())
                else:
                    Ainv = splu(self.A.to_csr().tocsc())
            except RuntimeError:
                print(self.A, np.array(self.A.data))
                raise
            try:
                self.L = ComplexCSR_LinearOperator.from_csr(Ainv.L)
                self.U = ComplexCSR_LinearOperator.from_csr(Ainv.U)
                self.perm_r = Ainv.perm_r
                self.perm_c = Ainv.perm_c
                n = self.perm_c.shape[0]
                self.temp_mem = uninitialized((n), dtype=COMPLEX)
            except AttributeError:
                self.Ainv = Ainv
        elif isinstance(self.A, ComplexDense_LinearOperator):
            from scipy.linalg import lu_factor
            self.lu, self.perm = lu_factor(self.A.data)
        else:
            raise NotImplementedError('Cannot use operator of type "{}"'.format(type(self.A)))
        self.initialized = True

    cdef int solve(self, complex_vector_t b, complex_vector_t x) except -1:
        cdef:
            INDEX_t i, n
            INDEX_t[::1] perm_r, perm_c
            complex_vector_t temp
        complex_solver.solve(self, b, x)
        if isinstance(self.A, (ComplexSSS_LinearOperator, ComplexCSR_LinearOperator, HelmholtzShiftOperator)):
            perm_r = self.perm_r
            perm_c = self.perm_c
            try:
                temp = self.temp_mem
                n = perm_c.shape[0]
                for i in range(n):
                    temp[perm_r[i]] = b[i]
                x[:] = 0.
                forward_solve_csc(self.L.indptr, self.L.indices, self.L.data,
                                  temp, x,
                                  unitDiagonal=True)
                temp[:] = 0.
                backward_solve_csc(self.U.indptr, self.U.indices, self.U.data, x, temp)
                for i in range(n):
                    x[i] = temp[perm_c[i]]
            except AttributeError:
                x[:] = self.Ainv.solve(np.array(b, copy=False, dtype=COMPLEX))
        elif isinstance(self.A, ComplexDense_LinearOperator):
            from scipy.linalg import lu_solve
            assign(x, b)
            lu_solve((self.lu, self.perm),
                     np.array(x, copy=False, dtype=COMPLEX),
                     overwrite_b=True)
        else:
            raise NotImplementedError('Cannot use operator of type "{}"'.format(type(self.A)))
        return 1

    def __str__(self):
        return 'LU'


cdef class complex_iterative_solver(complex_solver):
    def __init__(self, ComplexLinearOperator A=None, INDEX_t num_rows=-1):
        complex_solver.__init__(self, A, num_rows)
        self.residuals = []
        self.setNormInner(wrapRealNormToComplex(norm_serial()),
                          wrapRealInnerToComplex(ip_serial()))
        self.maxIter = -1
        self.tolerance = 1e-5
        self.relativeTolerance = False
        self.x0 = None

    cpdef void setInitialGuess(self, complex_vector_t x0=None):
        if x0 is not None:
            assert self.num_rows == x0.shape[0], \
                'x0 vector has size {}, solver expects {}'.format(x0.shape[0],
                                                                  self.num_rows)
            self.x0 = x0
        else:
            self.x0 = None

    cpdef void setNormInner(self, complexNormBase norm, complexipBase inner):
        self.norm = norm
        self.inner = inner

    cpdef void setOverlapNormInner(self, object overlaps, level=-1):
        self.setNormInner(wrapRealNormToComplex(norm_distributed(overlaps, level)),
                          wrapRealInnerToComplex(ip_distributed(overlaps, level)))

    cpdef void setup(self, ComplexLinearOperator A=None):
        if A is not None:
            assert A.num_rows == self.num_rows, \
                'A has {} rows, but solver expects {}.'.format(A.num_rows, self.num_rows)
            self.A = A
        else:
            assert self.A is not None, 'A not set'
        self.r = uninitialized((self.num_rows), dtype=COMPLEX)

    cdef int solve(self, complex_vector_t b, complex_vector_t x) except -1:
        cdef:
            REAL_t res
            complex_vector_t r = self.r
            ComplexLinearOperator A = self.A
            complexNormBase norm = self.norm

        complex_solver.solve(self, b, x)

        if self.x0 is None:
            for i in range(self.num_rows):
                x[i] = 0.
        elif &x[0] != &self.x0[0]:
            assign(x, self.x0)

        if self.relativeTolerance:
            A.residual(x, b, r, simpleResidual=self.x0 is None)   # dist
            res = norm.eval(r, False)                             # ip(dist, dist)
            self.tol = self.tolerance*res
        else:
            self.tol = self.tolerance


cdef class complex_krylov_solver(complex_iterative_solver):
    def __init__(self, ComplexLinearOperator A=None, INDEX_t num_rows=-1):
        complex_iterative_solver.__init__(self, A, num_rows)
        self.prec = None
        self.convMaster = None
        self.convClient = None

    cpdef void setup(self, ComplexLinearOperator A=None):
        complex_iterative_solver.setup(self, A)
        if self.prec is not None and isinstance(self.prec, complex_preconditioner) and (not self.prec.solOp.initialized or A is not None):
            self.prec.solOp.setup(self.A)

    cpdef void setPreconditioner(self, ComplexLinearOperator prec, BOOL_t left=True):
        self.prec = prec
        self.isLeftPrec = left

    cdef int solve(self, complex_vector_t b, complex_vector_t x) except -1:
        complex_iterative_solver.solve(self, b, x)
        if not self.relativeTolerance:
            self.A.residual(x, b, self.r, simpleResidual=self.x0 is None)   # dist


cdef class complex_gmres_solver(complex_krylov_solver):
    """
    GMRES iteration.

    In a distributed solve, we want:
    A:        accumulated to distributed
    Lprecond:  distributed to accumulated
    Rprecond:  distributed to accumulated
    b:        distributed
    x0:       accumulated
    x:        accumulated

    In the unpreconditioned distributed case, set Lprecond to accumulate.

    Memory requirement:
    dim * (maxiter+1) for Q
    (maxiter+1) * maxiter for H
    (4*maxiter + 2) + 2*dim for c,s,gamma,y and r, Ar
    dim * (maxiter+1) for Z if flexible
    """

    def __init__(self, ComplexLinearOperator A=None, INDEX_t num_rows=-1):
        complex_krylov_solver.__init__(self, A)
        self.use2norm = False
        self.flexible = False
        self.restarts = 1

    cpdef void setup(self, ComplexLinearOperator A=None):
        complex_krylov_solver.setup(self, A)
        self.Ar = uninitialized((self.num_rows), dtype=COMPLEX)
        assert self.maxIter > 0, 'Need maxiter > 0'
        self.H = np.ones((self.maxIter+1, self.maxIter), dtype=COMPLEX, order='F')
        # set first dim to 1, not 0, so that things work
        d = max(self.num_rows, 1)
        self.Q = uninitialized((d, self.maxIter+1), dtype=COMPLEX, order='F')
        if self.flexible:
            self.Z = uninitialized((d, self.maxIter+1), dtype=COMPLEX, order='F')
        self.c = uninitialized((self.maxIter), dtype=COMPLEX)
        self.s = uninitialized((self.maxIter), dtype=COMPLEX)
        self.gamma = uninitialized((self.maxIter+1), dtype=COMPLEX)
        self.y = uninitialized((self.maxIter+1), dtype=COMPLEX)
        self.initialized = True

    cpdef void setPreconditioner(self, ComplexLinearOperator prec, BOOL_t left=True):
        complex_krylov_solver.setPreconditioner(self, prec, left)

    cdef int solve(self, complex_vector_t b, complex_vector_t x) except -1:
        complex_krylov_solver.solve(self, b, x)

        cdef:
            int k, i = -1, j, dim = self.num_rows, l
            REAL_t eps = 1e-15, beta
            COMPLEX_t temp, rho, sigma
            COMPLEX_t[::1, :] Q = self.Q
            COMPLEX_t[::1, :] H = self.H
            COMPLEX_t[::1, :] Z = None
            COMPLEX_t[::1] c = self.c, s = self.s, gamma = self.gamma
            complex_vector_t y = self.y
            complex_vector_t r = self.r
            complex_vector_t Ar = self.Ar
            BOOL_t breakout = False
            BOOL_t converged
            BOOL_t doLprecond = self.isLeftPrec and self.prec is not None
            BOOL_t doRprecond = not self.isLeftPrec and self.prec is not None
            ComplexLinearOperator A = self.A
            ComplexLinearOperator Lprecond = None, Rprecond = None
            list residuals = []
            complexipBase inner = self.inner
            complexNormBase norm = self.norm
            REAL_t tol = self.tol
            INDEX_t maxiter = self.maxIter
            INDEX_t restarts = self.restarts
            BOOL_t flexible = self.flexible
            int allIter = 0
            convergenceMaster convMaster=self.convMaster
            convergenceClient convClient=self.convClient

        if doLprecond:
            Lprecond = self.prec
        if doRprecond:
            Rprecond = self.prec
        if flexible:
            Z = self.Z

        for k in range(restarts):
            if breakout:
                self.residuals = residuals
                return allIter
            A.matvec(x, Ar)                                                  # dist
            if doLprecond:
                assign3(Ar, Ar, -1.0, b, 1.0)                                # dist
                Lprecond.matvec(Ar, r)                                       # acc
                gamma[0] = norm.eval(r, True)
            else:
                assign3(r, b, 1.0, Ar, -1.0)                                 # dist
                gamma[0] = norm.eval(r, False)
            if len(residuals) == 0:
                residuals.append(abs(gamma[0]))
            converged = abs(gamma[0]) < tol
            if convMaster is not None:
                convMaster.setStatus(converged)
            if convClient is not None:
                converged = convClient.getStatus()
            if converged:
                self.residuals = residuals
                return allIter
            assignScaled(Q[:, 0], r, 1./gamma[0])                            # acc for Lprecond, dist for Rprecond
            for i in range(maxiter):
                ##############################
                # Arnoldi iteration
                assign(r, Q[:, i])                                           # acc for Lprecond, dist for Rprecond
                if not flexible:
                    if doLprecond:
                        A.matvec(r, Ar)                                      # dist
                        Lprecond.matvec(Ar, r)                               # acc
                    elif doRprecond:
                        Rprecond.matvec(r, Ar)                               # acc
                        A.matvec(Ar, r)                                      # dist
                    else:
                        A.matvec(r, Ar)
                        assign(r, Ar)
                else:
                    if doLprecond:
                        A.matvec(r, Z[:, i])                                 # dist
                        Lprecond.matvec(Z[:, i], r)                          # acc
                    elif doRprecond:
                        Rprecond.matvec(r, Z[:, i])                          # acc
                        A.matvec(Z[:, i], r)                                 # dist
                    else:
                        A.matvec(r, Z[:, i])
                        assign(r, Z[:, i])
                if doRprecond:
                    if dim > 0:
                        for j in range(i+1):
                            H[j, i] = inner.eval(Q[:, j], r, False, False)
                            updateScaled(r, Q[:, j], -H[j, i])               # dist
                    H[i+1, i] = norm.eval(r, False)
                else:
                    if dim > 0:
                        for j in range(i+1):
                            H[j, i] = inner.eval(Q[:, j], r, True, True)
                            updateScaled(r, Q[:, j], -H[j, i])               # acc
                    H[i+1, i] = norm.eval(r, True)
                converged = abs(H[i+1, i]) > eps
                if convMaster is not None:
                    convMaster.setStatus(converged)
                if convClient is not None:
                    converged = convClient.getStatus()
                if converged:
                    assignScaled(Q[:, i+1], r, 1./H[i+1, i])                 # acc for Lprecond, dist for Rprecond
                else:
                    breakout = True
                    break
                ##############################
                # Apply previous Givens rotations to last column of H
                for j in range(i):
                    rho = H[j, i]
                    sigma = H[j+1, i]
                    H[j, i] = c[j]*rho + s[j]*sigma
                    H[j+1, i] = -s[j].conjugate()*rho + c[j].conjugate()*sigma
                ##############################
                # determine new Givens rotation
                beta = sqrt(abs(H[i, i])**2 + abs(H[i+1, i])**2)
                c[i] = H[i, i].conjugate()/beta
                s[i] = H[i+1, i].conjugate()/beta
                ##############################
                # Apply new Givens rotation to H
                H[i, i] = beta
                # H[i+1, i] = 0.0
                ##############################
                # Apply new Givens rotation to rhs
                gamma[i+1] = -s[i].conjugate()*gamma[i]
                gamma[i] = c[i]*gamma[i]
                residuals.append(abs(gamma[i+1]))
                converged = abs(gamma[i+1]) < tol
                if convMaster is not None:
                    convMaster.setStatus(converged)
                if convClient is not None:
                    converged = convClient.getStatus()
                if converged:
                    breakout = True
                    break
            allIter += i
            ##############################
            # perform back-solve for y
            for j in range(i, -1, -1):
                temp = gamma[j]
                for l in range(j+1, i+1):
                    temp -= H[j, l]*y[l]
                y[j] = temp/H[j, j]
            ##############################
            # update x
            if not flexible:
                gemvF(Q[:, :i+1], y[:i+1], r)                                # acc for Lprecond, dist for Rprecond
                if doRprecond:
                    Rprecond.matvec(r, Ar)                                   # acc
                    update(x, Ar)                                            # acc
                else:
                    update(x, r)                                             # acc
            else:
                gemvF(Z[:, :i+1], y[:i+1], r)                                # dist for Lprecond, acc for Rprecond
                assert not doLprecond                                        # TODO: figure out what to do here
                update(x, r)                                                 # acc
        self.residuals = residuals
        return allIter

    def __str__(self):
        s = 'GMRES(tolerance={},relTol={},maxIter={},restarts={},2-norm={},flexible={})'.format(self.tolerance, self.relativeTolerance, self.maxIter,
                                                                                                self.restarts, self.use2norm, self.flexible)
        if self.prec is not None:
            if self.isLeftPrec:
                return s+', left preconditioned by '+str(self.prec)
            else:
                return s+', right preconditioned by '+str(self.prec)
        else:
            return s
