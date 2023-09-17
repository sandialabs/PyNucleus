###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t
from . linear_operators cimport (LinearOperator,
                                 ComplexLinearOperator,
                                 CSR_LinearOperator,
                                 SSS_LinearOperator,
                                 Dense_LinearOperator,
                                 invDiagonal,
                                 ComplexCSR_LinearOperator,
                                 ComplexSSS_LinearOperator,
                                 ComplexDense_LinearOperator,
                                 HelmholtzShiftOperator)
from . ip_norm cimport ipBase, normBase, complexipBase, complexNormBase
from . convergence cimport (convergenceMaster, noOpConvergenceMaster,
                            convergenceClient, noOpConvergenceClient)
from . performanceLogger cimport PLogger, FakePLogger

include "config.pxi"

cdef class solver:
    cdef:
        public BOOL_t initialized
        public LinearOperator A
        public INDEX_t num_rows
        public FakePLogger PLogger
    cpdef void setup(self, LinearOperator A=*)
    cdef int solve(self, REAL_t[::1] b, REAL_t[::1] x) except -1


cdef class preconditioner(LinearOperator):
    cdef:
        public solver solOp
        public dict ctxAttrs


cdef class lu_solver(solver):
    cdef:
        INDEX_t[::1] perm_r, perm_c, perm
        CSR_LinearOperator L, U
        REAL_t[::1] temp_mem
        object Ainv, lu
        BOOL_t useTriangularSolveRoutines


cdef class pardiso_lu_solver(solver):
    cdef:
        INDEX_t[::1] perm
        object Ainv, lu
        object Asp


cdef class chol_solver(solver):
    cdef:
        object Ainv
        REAL_t[::1] Lflat, temp
        BOOL_t denseFactor


cdef class ichol_solver(solver):
    cdef:
        public INDEX_t[::1] indptr, indices
        public REAL_t[::1] data, diagonal, temp
        CSR_LinearOperator L


cdef class ilu_solver(solver):
    cdef:
        public INDEX_t[::1] Lindptr, Lindices
        public REAL_t[::1] Ldata
        public INDEX_t[::1] Uindptr, Uindices
        public REAL_t[::1] Udata
        public INDEX_t[::1] perm_c, perm_r
        REAL_t[::1] temp1
        REAL_t[::1] temp2
        public REAL_t fill_factor


cdef class jacobi_solver(solver):
    cdef:
        invDiagonal invD


cdef class iterative_solver(solver):
    cdef:
        public INDEX_t maxIter
        public REAL_t tolerance
        REAL_t tol
        public list residuals
        REAL_t[::1] x0
        REAL_t[::1] r
        public ipBase inner
        public normBase norm
        public BOOL_t relativeTolerance
    cpdef void setInitialGuess(self, REAL_t[::1] x0=*)
    cpdef void setNormInner(self, normBase norm, ipBase inner)
    cpdef void setOverlapNormInner(self, object overlaps, level=*)
    cpdef void setup(self, LinearOperator A=*)
    cdef int solve(self, REAL_t[::1] b, REAL_t[::1] x) except -1


cdef class krylov_solver(iterative_solver):
    cdef:
        public LinearOperator prec
        BOOL_t isLeftPrec
        public convergenceMaster convMaster
        public convergenceClient convClient
    cpdef void setup(self, LinearOperator A=*)
    cpdef void setPreconditioner(self, LinearOperator prec, BOOL_t left=*)
    cdef int solve(self, REAL_t[::1] b, REAL_t[::1] x) except -1


cdef class cg_solver(krylov_solver):
    cdef:
        REAL_t[::1] temporaryMemory
        REAL_t[::1] precTemporaryMemory
        public BOOL_t use2norm
    cpdef void setup(self, LinearOperator A=*)
    cpdef void setPreconditioner(self, LinearOperator prec, BOOL_t left=*)
    cdef int solve(self, REAL_t[::1] b, REAL_t[::1] x) except -1


cdef class gmres_solver(krylov_solver):
    cdef:
        REAL_t[::1] Ar
        public BOOL_t use2norm
        public BOOL_t flexible
        public INDEX_t restarts
        REAL_t[::1, :] Q
        REAL_t[::1, :] Z
        REAL_t[::1, :] H
        REAL_t[::1] c, s, gamma, y
    cpdef void setup(self, LinearOperator A=*)
    cpdef void setPreconditioner(self, LinearOperator prec, BOOL_t left=*)
    cdef int solve(self, REAL_t[::1] b, REAL_t[::1] x) except -1


cdef class bicgstab_solver(krylov_solver):
    cdef:
        REAL_t[::1] r0
        REAL_t[::1] p
        REAL_t[::1] p2
        REAL_t[::1] s
        REAL_t[::1] s2
        REAL_t[::1] temp
        REAL_t[::1] temp2
        public BOOL_t use2norm
    cpdef void setup(self, LinearOperator A=*)
    cpdef void setPreconditioner(self, LinearOperator prec, BOOL_t left=*)
    cdef int solve(self, REAL_t[::1] b, REAL_t[::1] x) except -1


IF USE_PYAMG:
    cdef class pyamg_solver(iterative_solver):
        cdef:
            object ml


cdef class complex_solver:
    cdef:
        public BOOL_t initialized
        ComplexLinearOperator A
        public INDEX_t num_rows
        public FakePLogger PLogger
    cpdef void setup(self, ComplexLinearOperator A=*)
    cdef int solve(self, COMPLEX_t[::1] b, COMPLEX_t[::1] x) except -1


cdef class complex_preconditioner(ComplexLinearOperator):
    cdef:
        public complex_solver solOp
        public dict ctxAttrs


cdef class complex_lu_solver(complex_solver):
    cdef:
        INDEX_t[::1] perm_r, perm_c, perm
        ComplexCSR_LinearOperator L, U
        COMPLEX_t[::1] temp_mem
        object Ainv, lu


cdef class complex_iterative_solver(complex_solver):
    cdef:
        public INDEX_t maxIter
        public REAL_t tolerance
        REAL_t tol
        public list residuals
        COMPLEX_t[::1] x0
        COMPLEX_t[::1] r
        public complexipBase inner
        public complexNormBase norm
        public BOOL_t relativeTolerance
    cpdef void setInitialGuess(self, COMPLEX_t[::1] x0=*)
    cpdef void setNormInner(self, complexNormBase norm, complexipBase inner)
    cpdef void setOverlapNormInner(self, object overlaps, level=*)
    cpdef void setup(self, ComplexLinearOperator A=*)
    cdef int solve(self, COMPLEX_t[::1] b, COMPLEX_t[::1] x) except -1


cdef class complex_krylov_solver(complex_iterative_solver):
    cdef:
        public ComplexLinearOperator prec
        BOOL_t isLeftPrec
        public convergenceMaster convMaster
        public convergenceClient convClient
    cpdef void setup(self, ComplexLinearOperator A=*)
    cpdef void setPreconditioner(self, ComplexLinearOperator prec, BOOL_t left=*)
    cdef int solve(self, COMPLEX_t[::1] b, COMPLEX_t[::1] x) except -1


cdef class complex_gmres_solver(complex_krylov_solver):
    cdef:
        COMPLEX_t[::1] Ar
        public BOOL_t use2norm
        public BOOL_t flexible
        public INDEX_t restarts
        COMPLEX_t[::1, :] Q
        COMPLEX_t[::1, :] Z
        COMPLEX_t[::1, :] H
        COMPLEX_t[::1] c, s, gamma, y
    cpdef void setup(self, ComplexLinearOperator A=*)
    cpdef void setPreconditioner(self, ComplexLinearOperator prec, BOOL_t left=*)
    cdef int solve(self, COMPLEX_t[::1] b, COMPLEX_t[::1] x) except -1
