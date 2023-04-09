###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}smoother:
    def __init__(self, {SCALAR_label}LinearOperator A):
        self._A = A

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self,
                 {SCALAR}_t[::1] b,
                 {SCALAR}_t[::1] y,
                 BOOL_t postsmoother,
                 BOOL_t simpleResidual=False):
        self.eval(b, y, postsmoother, simpleResidual)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void eval(self,
                   {SCALAR}_t[::1] b,
                   {SCALAR}_t[::1] y,
                   BOOL_t postsmoother,
                   BOOL_t simpleResidual=False):
        raise NotImplementedError()

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, {SCALAR_label}LinearOperator A):
        self._A = A


cdef class {SCALAR_label}preconditioner({SCALAR_label}LinearOperator):
    def __init__(self, INDEX_t numRows, INDEX_t numColumns):
        {SCALAR_label}LinearOperator.__init__(self, numRows, numColumns)

    cdef void setPre(self):
        pass

    cdef void setPost(self):
        pass


cdef class {SCALAR_label}separableSmoother({SCALAR_label}smoother):
    def __init__(self,
                 {SCALAR_label}LinearOperator A,
                 {SCALAR_label}preconditioner P,
                 dict params,
                 np.ndarray[{SCALAR}_t, ndim=1] temporaryMemory=None,
                 algebraicOverlapManager overlap=None):
        defaults = {'presmoothingSteps': 1,
                    'postsmoothingSteps': 1}
        defaults.update(params)
        {SCALAR_label}smoother.__init__(self, A)
        self.overlap = overlap
        self.prec = P
        self.A = A
        if temporaryMemory is not None:
            self.temporaryMemory = temporaryMemory
        else:
            LOGGER.debug(('Allocating temporary memory for ' +
                          'smoother ({} elements)').format(A.shape[0]))
            self.temporaryMemory = uninitialized((A.shape[0]), dtype={SCALAR})
        self.temporaryMemory2 = uninitialized((A.shape[0]), dtype={SCALAR})
        self.presmoothingSteps = defaults['presmoothingSteps']
        self.postsmoothingSteps = defaults['postsmoothingSteps']

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, {SCALAR_label}LinearOperator A):
        self._A = A
        if self.overlap is not None:
            if isinstance(A, {SCALAR_label}CSR_LinearOperator):
                self._accA = {SCALAR_label}CSR_DistributedLinearOperator(A, self.overlap, doDistribute=False, keepDistributedResult=False)
            else:
                self._accA = {SCALAR_label}DistributedLinearOperator(A, self.overlap, doDistribute=False, keepDistributedResult=False)
        else:
            self._accA = A

    def setD(self, {SCALAR}_t[::1] D):
        self.prec.setD(D)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void eval(self,
                   {SCALAR}_t[::1] b,
                   {SCALAR}_t[::1] y,
                   BOOL_t postsmoother,
                   BOOL_t simpleResidual=False):
        cdef:
            INDEX_t k, steps
            {SCALAR}_t[::1] temp_mem = self.temporaryMemory
            {SCALAR}_t[::1] temp_mem2 = self.temporaryMemory2
        if postsmoother:
            steps = self.postsmoothingSteps
            self.prec.setPost()
        else:
            steps = self.presmoothingSteps
            self.prec.setPre()
        for k in range(steps):
            # In a distributed setup, b is distributed, x is accumulated,
            # the solution in x is accumulted.
            # Residual is distributed.
            # prec*residual is distributed.
            self._accA.residual(y, b, temp_mem, simpleResidual=simpleResidual)
            self.prec.matvec(temp_mem, temp_mem2)
            simpleResidual = False
            update(y, temp_mem2)


######################################################################
# Jacobi preconditioner and smoother

cdef class {SCALAR_label}jacobiPreconditioner({SCALAR_label}preconditioner):
    def __init__(self, {SCALAR}_t[::1] D, {SCALAR}_t omega):
        {SCALAR_label}preconditioner.__init__(self, D.shape[0], D.shape[0])
        self.omega = omega
        self.setD(D)

    def setD(self, {SCALAR}_t[::1] D):
        self.invD = self.omega/np.array(D, copy=False)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec(self, {SCALAR}_t[::1] x, {SCALAR}_t[::1] y) except -1:
        cdef:
            INDEX_t i
        for i in range(self.num_rows):
            y[i] = x[i]*self.invD[i]
        return 0


cdef class {SCALAR_label}jacobiSmoother({SCALAR_label}separableSmoother):
    # Needs n temporary memory for residual
    def __init__(self,
                 {SCALAR_label}LinearOperator A,
                 {SCALAR}_t[::1] D,
                 dict params,
                 np.ndarray[{SCALAR}_t, ndim=1] temporaryMemory=None,
                 overlap=None):
        defaults = {'omega': 2.0/3.0}
        defaults.update(params)
        preconditioner = {SCALAR_label}jacobiPreconditioner(D, defaults['omega'])
        {SCALAR_label}separableSmoother.__init__(self, A, preconditioner, params, temporaryMemory, overlap)

    def __repr__(self):
        return 'Jacobi ({}/{} sweeps, {:.3} damping)'.format(self.presmoothingSteps, self.postsmoothingSteps, self.prec.omega)


######################################################################
# Block Jacobi preconditioner and smoother

from PyNucleus_base.solvers cimport lu_solver, complex_lu_solver
from PyNucleus_base.linear_operators cimport sparseGraph


cdef class {SCALAR_label}blockJacobiPreconditioner({SCALAR_label}preconditioner):
    def __init__(self, {SCALAR_label}LinearOperator A, sparseGraph blocks, {SCALAR}_t omega):
        {SCALAR_label}preconditioner.__init__(self, A.num_rows, A.num_columns)
        self.omega = omega
        self.setD(A, blocks)

    def setD(self, {SCALAR_label}LinearOperator A, sparseGraph blocks):
        cdef:
            {SCALAR_label}CSR_LinearOperator D
        if isinstance(A, {SCALAR_label}CSR_LinearOperator):
            D = A.getBlockDiagonal(blocks)
        else:
            D = A.to_csr_linear_operator().getBlockDiagonal(blocks)
        D.scale(1./self.omega)
        self.invD = {SCALAR_label_lc_}lu_solver(D)
        self.invD.setup()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec(self, {SCALAR}_t[::1] x, {SCALAR}_t[::1] y) except -1:
        self.invD.solve(x, y)
        return 0


cdef class {SCALAR_label}blockJacobiSmoother({SCALAR_label}separableSmoother):
    # Needs n temporary memory for residual
    def __init__(self,
                 {SCALAR_label}LinearOperator A,
                 dict params,
                 np.ndarray[{SCALAR}_t, ndim=1] temporaryMemory=None,
                 overlap=None):
        defaults = {'omega': 2.0/3.0}
        defaults.update(params)
        preconditioner = {SCALAR_label}blockJacobiPreconditioner(A, defaults['blocks'], defaults['omega'])
        {SCALAR_label}separableSmoother.__init__(self, A, preconditioner, params, temporaryMemory, overlap)

    def __repr__(self):
        return 'Block Jacobi ({}/{} sweeps, {:.3} damping)'.format(self.presmoothingSteps, self.postsmoothingSteps, self.prec.omega)


######################################################################
# GMRES smoother

from PyNucleus_base.solvers cimport {SCALAR_label_lc_}gmres_solver
from PyNucleus_base.linear_operators cimport {SCALAR_label}diagonalOperator


cdef class {SCALAR_label}gmresSmoother({SCALAR_label}smoother):
    def __init__(self,
                 {SCALAR_label}LinearOperator A,
                 {SCALAR}_t[::1] D,
                 dict params,
                 algebraicOverlapManager overlap=None):
        defaults = {'presmoothingSteps': 10,
                    'postsmoothingSteps': 10}
        defaults.update(params)
        {SCALAR_label}smoother.__init__(self, A)
        self.solver = {SCALAR_label_lc_}gmres_solver(A)
        self.solver.setPreconditioner({SCALAR_label}diagonalOperator(1./np.array(D, copy=False)))
        self.solver.maxIter = defaults['presmoothingSteps']
        self.solver.restarts = 1
        self.solver.tolerance = 1e-12
        if overlap:
            self.solver.setOverlapNormInner(overlap)
        self.solver.setup()
        self.overlap = overlap
        self.A = A
        self.presmoothingSteps = defaults['presmoothingSteps']
        self.postsmoothingSteps = defaults['postsmoothingSteps']
        self.temporaryMemory = uninitialized((A.shape[0]), dtype={SCALAR})
        self.temporaryMemory2 = uninitialized((A.shape[0]), dtype={SCALAR})

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, {SCALAR_label}LinearOperator A):
        self._A = A
        if self.overlap is not None:
            if isinstance(A, {SCALAR_label}CSR_LinearOperator):
                self._accA = {SCALAR_label}CSR_DistributedLinearOperator(A, self.overlap, doDistribute=False, keepDistributedResult=False)
            else:
                self._accA = {SCALAR_label}DistributedLinearOperator(A, self.overlap, doDistribute=False, keepDistributedResult=False)
        else:
            self._accA = A

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void eval(self,
                   {SCALAR}_t[::1] b,
                   {SCALAR}_t[::1] y,
                   BOOL_t postsmoother,
                   BOOL_t simpleResidual=False):
        cdef:
            INDEX_t k, steps
            {SCALAR}_t[::1] temp_mem = self.temporaryMemory
            {SCALAR}_t[::1] temp_mem2 = self.temporaryMemory2
        if postsmoother:
            steps = self.postsmoothingSteps
            # self.prec.setPost()
        else:
            steps = self.presmoothingSteps
            # self.prec.setPre()
        for k in range(steps):
            # In a distributed setup, b is distributed, x is accumulated,
            # the solution in x is accumulted.
            # Residual is distributed.
            # prec*residual is distributed.
            self._A.residual(y, b, temp_mem, simpleResidual=simpleResidual)
            self.solver.solve(temp_mem, temp_mem2)
            simpleResidual = False
            update(y, temp_mem2)

    def __repr__(self):
        return str(self.solver)
