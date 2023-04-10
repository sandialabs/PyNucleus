###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
cimport numpy as np
cimport cython
from scipy.sparse import csr_matrix
from PyNucleus_base.myTypes import INDEX, REAL, COMPLEX
from PyNucleus_base import uninitialized
from PyNucleus_base.blas cimport (update, updateScaled,
                                   assignScaled, assign3)
from PyNucleus_base.linear_operators cimport (Product_Linear_Operator,
                                               CSR_LinearOperator,
                                               ComplexCSR_LinearOperator,
                                               SSS_LinearOperator,
                                               TimeStepperLinearOperator)
from PyNucleus_base.linalg import estimateSpectralRadius
from PyNucleus_base.linalg import ILU_solver
import logging

LOGGER = logging.getLogger(__name__)


include "smoothers_REAL.pxi"
include "smoothers_COMPLEX.pxi"


######################################################################
# SOR preconditioner and smoother

# Assumes that the indices of A are ordered
cdef class sorPreconditioner(preconditioner):
    cdef:
        public LinearOperator A
        public REAL_t[::1] D
        INDEX_t[::1] A_indptr, A_indices
        REAL_t[::1] A_data, A_diagonal
        REAL_t[::1] temp
        REAL_t omega
        public BOOL_t presmoother_forwardSweep
        public BOOL_t postsmoother_forwardSweep
        public BOOL_t forwardSweep

    def __init__(self,
                 LinearOperator A,
                 REAL_t[::1] D,
                 REAL_t omega,
                 BOOL_t presmoother_forwardSweep,
                 BOOL_t postsmoother_forwardSweep):
        preconditioner.__init__(self, D.shape[0], D.shape[0])
        self.D = D
        if isinstance(A, (CSR_LinearOperator, csr_matrix, SSS_LinearOperator)):
            self.A = A
            self.A_indptr = self.A.indptr
            self.A_indices = self.A.indices
            self.A_data = self.A.data
        elif isinstance(A, SSS_LinearOperator):
            self.A = A
            self.A_indptr = self.A.indptr
            self.A_indices = self.A.indices
            self.A_data = self.A.data
            self.A_diagonal = self.A.diagonal
        elif isinstance(A, TimeStepperLinearOperator):
            assert isinstance(A.M, CSR_LinearOperator)
            assert isinstance(A.S, CSR_LinearOperator)
            assert A.M.nnz == A.S.nnz
            self.A = A
        else:
            self.A = A.to_csr_linear_operator()
            self.A_indptr = self.A.indptr
            self.A_indices = self.A.indices
            self.A_data = self.A.data
        self.omega = omega
        self.presmoother_forwardSweep = presmoother_forwardSweep
        self.postsmoother_forwardSweep = postsmoother_forwardSweep
        self.forwardSweep = False

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] result) except -1:
        cdef:
            INDEX_t i, j, k, jj
            REAL_t t
            REAL_t facM, facS
            INDEX_t[::1] Sindptr, Sindices
            REAL_t[::1] Sdata, Mdata

        result[:] = 0.0
        if isinstance(self.A, (CSR_LinearOperator, csr_matrix)):
            if self.forwardSweep:
                for i in range(self.num_rows):
                    t = x[i]
                    for j in range(self.A_indptr[i], self.A_indptr[i+1]):
                        k = self.A_indices[j]
                        if k >= i:
                            break
                        t -= self.A_data[j]*result[k]
                    result[i] = self.omega*t/self.D[i]
            else:
                for i in range(self.num_rows-1, -1, -1):
                    t = x[i]
                    for j in range(self.A_indptr[i+1]-1, self.A_indptr[i]-1, -1):
                        k = self.A_indices[j]
                        if k <= i:
                            break
                        t -= self.A_data[j]*result[k]
                    result[i] = self.omega*t/self.D[i]
        elif isinstance(self.A, SSS_LinearOperator):
            if self.forwardSweep:
                for i in range(self.num_rows):
                    t = x[i]
                    for jj in range(self.A_indptr[i], self.A_indptr[i+1]):
                        t -= self.A_data[jj]*result[self.A_indices[jj]]
                    result[i] = self.omega*t/self.A_diagonal[i]
            else:
                result[:] = 0.
                for i in range(self.num_rows-1, -1, -1):
                    result[i] = (x[i]-self.omega*result[i])/self.A_diagonal[i]
                    for jj in range(self.A_indptr[i], self.A_indptr[i+1]):
                        result[self.A_indices[jj]] += self.A_data[jj]*result[i]
        elif isinstance(self.A, TimeStepperLinearOperator):
            Sindptr = self.A.S.indptr
            Sindices = self.A.S.indices
            Sdata = self.A.S.data
            Mdata = self.A.M.data
            facS = self.A.facS
            facM = self.A.facM
            if self.forwardSweep:
                for i in range(self.num_rows):
                    t = x[i]
                    for j in range(Sindptr[i], Sindptr[i+1]):
                        k = Sindices[j]
                        if k >= i:
                            break
                        t -= (facS*Sdata[j]+facM*Mdata[j])*result[k]
                    result[i] = self.omega*t/self.D[i]
            else:
                for i in range(self.num_rows-1, -1, -1):
                    t = x[i]
                    for j in range(Sindptr[i+1]-1, Sindptr[i]-1, -1):
                        k = Sindices[j]
                        if k <= i:
                            break
                        t -= (facS*Sdata[j]+facM*Mdata[j])*result[k]
                    result[i] = self.omega*t/self.D[i]
        else:
            return -1
        return 0

    cdef void setPre(self):
        self.forwardSweep = self.presmoother_forwardSweep

    cdef void setPost(self):
        self.forwardSweep = self.postsmoother_forwardSweep


# Assumes that the indices of A are ordered
cdef class sorSmoother(separableSmoother):
    # Needs 2n temporary memory for residual and result of application
    # of preconditioner
    def __init__(self, A, D,
                 dict params,
                 temporaryMemory=None,
                 temporaryMemory2=None,
                 overlap=None):
        defaults = {'presmootherSweep': 'forward',
                    'postsmootherSweep': 'backward',
                    'omega': 1.0}
        defaults.update(params)
        super(sorSmoother, self).__init__(A, 1)
        self.preconditioner = sorPreconditioner(A, D,
                                                defaults['omega'],
                                                defaults['presmootherSweep'] == 'forward',
                                                defaults['postsmootherSweep'] == 'forward')
        self.omega = defaults['omega']


######################################################################
# SSOR preconditioner and smoother

# FIX: Not sure this really works
# Assumes that the indices of A are ordered
cdef class ssorPreconditioner(preconditioner):
    cdef:
        REAL_t[::1] D
        LinearOperator A
        INDEX_t[::1] A_indptr, A_indices
        REAL_t[::1] A_data
        REAL_t[::1] temp
        REAL_t omega

    def __init__(self,
                 LinearOperator A,
                 REAL_t[::1] D,
                 REAL_t omega):
        preconditioner.__init__(self, D.shape[0], D.shape[0])
        self.D = D
        self.A = A
        self.A_indptr = A.indptr
        self.A_indices = A.indices
        self.A_data = A.data
        self.omega = omega

    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] result) except -1:
        cdef:
            INDEX_t i, j, k
            REAL_t t

        if isinstance(self.A, (CSR_LinearOperator, csr_matrix)):
            result[:] = 0.0
            # forward sweep
            # x -> results
            for i in range(self.num_rows):
                t = x[i]
                for j in range(self.A_indptr[i], self.A_indptr[i+1]):
                    k = self.A_indices[j]
                    if k >= i:
                        break
                    t -= self.A_data[j]*result[k]
                result[i] = self.omega*t/self.D[i]
            # D/omega
            for i in range(self.num_rows):
                result[i] *= self.D[i]/self.omega
            # backward sweep
            # result -> results
            for i in range(self.num_rows-1, -1, -1):
                t = result[i]
                for j in range(self.A_indptr[i+1]-1, self.A_indptr[i]-1, -1):
                    k = self.A_indices[j]
                    if k <= i:
                        break
                    t -= self.A_data[j]*result[k]
                result[i] = self.omega*t/self.D[i]
            # correction
            for i in range(self.num_rows):
                result[i] *= (2-self.omega)
        else:
            return -1
        return 0


cdef class ssorSmoother(separableSmoother):
    # Needs 2n temporary memory for residual and result of application
    # of preconditioner
    def __init__(self, A, D,
                 dict params,
                 temporaryMemory=None,
                 overlap=None):
        defaults = {'omega': 1.0}
        defaults.update(params)
        preconditioner = ssorPreconditioner(A, D, defaults['omega'])
        super(ssorSmoother, self).__init__(A, preconditioner, params, temporaryMemory, overlap)
        self.omega = defaults['omega']


######################################################################
# Gauss-Seidel smoother

cdef class gaussSeidelSmoother(smoother):
    cdef:
        public REAL_t[::1] D
        BOOL_t presmoother_forwardSweep, postsmoother_forwardSweep
        INDEX_t presmoothingSteps, postsmoothingSteps
        sorPreconditioner prec
        REAL_t[::1] temporaryMemory, temporaryMemory2
        INDEX_t[::1] boundaryDofs

    def __init__(self, A,
                 REAL_t[::1] D,
                 dict params,
                 temporaryMemory=None,
                 temporaryMemory2=None,
                 overlap=None):
        defaults = {'presmootherSweep': 'forward',
                    'postsmootherSweep': 'backward',
                    'presmoothingSteps': 1,
                    'postsmoothingSteps': 1}
        defaults.update(params)
        super(gaussSeidelSmoother, self).__init__(A)
        self.overlap = overlap
        if isinstance(A, (SSS_LinearOperator, TimeStepperLinearOperator)) or overlap is not None:
            if overlap:
                self.boundaryDofs = self.overlap.Didx
            self.prec = sorPreconditioner(A, D, 1.,
                                          defaults['presmootherSweep'] == 'forward',
                                          defaults['postsmootherSweep'] == 'forward')
            if temporaryMemory is not None:
                self.temporaryMemory = temporaryMemory
            else:
                LOGGER.debug(('Allocating temporary memory for ' +
                              'Gauss-Seidel smoother ({} elements)').format(D.shape[0]))
                self.temporaryMemory = uninitialized((D.shape[0]), dtype=REAL)
            if temporaryMemory2 is not None:
                self.temporaryMemory2 = temporaryMemory2
            else:
                LOGGER.debug(('Allocating temporary memory for ' +
                              'Gauss-Seidel smoother ({} elements)').format(D.shape[0]))
                self.temporaryMemory2 = uninitialized((D.shape[0]), dtype=REAL)
        self.setD(D)
        self.presmoothingSteps = defaults['presmoothingSteps']
        self.postsmoothingSteps = defaults['postsmoothingSteps']
        self.presmoother_forwardSweep = (defaults['presmootherSweep'] ==
                                         'forward')
        self.postsmoother_forwardSweep = (defaults['postsmootherSweep'] ==
                                          'forward')

    def setD(self, REAL_t[::1] D):
        self.D = D

    cdef void eval(self,
                   REAL_t[::1] rhs,
                   REAL_t[::1] y,
                   BOOL_t postsmoother,
                   BOOL_t simpleResidual=False):
        # simpleResidual is ignored, because GS uses updated values
        cdef:
            INDEX_t num_rows = self.A.shape[0]
            INDEX_t[::1] A_indptr, A_indices
            REAL_t[::1] A_data
            INDEX_t i, j, steps, k, lv
            REAL_t t
            BOOL_t sweep

        if postsmoother:
            steps = self.postsmoothingSteps
            sweep = self.postsmoother_forwardSweep
        else:
            steps = self.presmoothingSteps
            sweep = self.presmoother_forwardSweep
        if isinstance(self.A, (CSR_LinearOperator, csr_matrix)) and self.overlap is None:
            A_indices = self.A.indices
            A_indptr = self.A.indptr
            A_data = self.A.data
            if sweep:
                for k in range(steps):
                    for i in range(num_rows):
                        t = rhs[i]
                        for j in range(A_indptr[i], A_indptr[i+1]):
                            t -= A_data[j]*y[A_indices[j]]
                        t += self.D[i]*y[i]
                        y[i] = t/self.D[i]
            else:
                for k in range(steps):
                    for i in range(num_rows-1, -1, -1):
                        t = rhs[i]
                        for j in range(A_indptr[i], A_indptr[i+1]):
                            t -= A_data[j]*y[A_indices[j]]
                        t += self.D[i]*y[i]
                        y[i] = t/self.D[i]
        elif isinstance(self.A, (SSS_LinearOperator, TimeStepperLinearOperator)) or self.overlap is not None:
            for k in range(steps):
                self.A.residual(y, rhs, self.temporaryMemory)
                if self.overlap:
                    self.overlap.accumulate(self.temporaryMemory)
                if sweep:
                    self.prec.setPost()
                else:
                    self.prec.setPre()
                self.prec(self.temporaryMemory, self.temporaryMemory2)
                # FIX: This is not quite correct
                #      I should do Jacobi on crosspoints of subdomains,
                #      then SOR on boundaries,
                #      and then SOR in the interior.
                #      Or at least Jacobi on crosspoints and boundary,
                #      and then SOR in the interior.
                #      Also, this is not very efficient.
                #      Also, this is does not take the sweep in account.
                if self.overlap:
                    # perform Jacobi on boundary dofs
                    for lv in self.boundaryDofs:
                        self.temporaryMemory2[lv] = self.temporaryMemory[lv]/self.prec.D[lv]
                update(y, self.temporaryMemory2)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return 'Gauss-Seidel ({} {} / {} {} sweeps)'.format(self.presmoothingSteps,
                                                            'forward' if self.presmoother_forwardSweep else 'backward',
                                                            self.postsmoothingSteps,
                                                            'forward' if self.postsmoother_forwardSweep else 'backward',)

cdef class iluPreconditioner(preconditioner):
    def __init__(self,
                 LinearOperator A,
                 **kwargs):
        preconditioner.__init__(self, A.shape[0], A.shape[0])
        self.A = A
        ILUS = ILU_solver(self.A.num_rows)
        if 'fill_factor' in kwargs:
            fill_factor = kwargs['fill_factor']
        else:
            fill_factor = 1.0
        ILUS.setup(self.A, fill_factor=fill_factor)
        self.preconditioner = ILUS.asPreconditioner()

    cdef INDEX_t matvec(self, REAL_t[::1] x, REAL_t[::1] y) except -1:
        self.preconditioner(x, y)
        return 0


cdef class iluSmoother(separableSmoother):
    def __init__(self,
                 LinearOperator A,
                 dict params,
                 np.ndarray[REAL_t, ndim=1] temporaryMemory=None,
                 overlap=None):
        defaults = {'fill_factor': 1.0}
        defaults.update(params)
        preconditioner = iluPreconditioner(A, **defaults)
        super(iluSmoother, self).__init__(A, preconditioner, params, temporaryMemory, overlap)


cdef class flexibleSmoother(separableSmoother):
    def __init__(self,
                 LinearOperator A,
                 dict params,
                 overlap=None):
        preconditioner = params['prec']
        super(flexibleSmoother, self).__init__(A, preconditioner, params, None, overlap)
