###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
cimport numpy as np
from tabulate import tabulate
from PyNucleus_base import INDEX, REAL, COMPLEX, uninitialized
from PyNucleus_base.performanceLogger cimport FakeTimer
from PyNucleus_base.ip_norm cimport (ip_serial, ip_distributed,
                                     norm_serial, norm_distributed)
from PyNucleus_base.blas cimport assign, update
from PyNucleus_base.solvers cimport {SCALAR_label_lc_}preconditioner
from PyNucleus_base.linear_operators cimport {SCALAR_label}LinearOperator, CSR_LinearOperator, wrapRealToComplexCSR

from . coarseSolvers cimport {SCALAR_label}coarseSolver
from . smoothers cimport ({SCALAR_label}smoother, {SCALAR_label}jacobiSmoother, {SCALAR_label}blockJacobiSmoother,
                          {SCALAR_label}gmresSmoother,
                          chebyshevSmoother,
                          iluSmoother, flexibleSmoother)
from . smoothers import (gaussSeidelSmoother,
                         sorSmoother, ssorSmoother)
from . hierarchies import hierarchyManager


cdef class {SCALAR_label}levelMemory:
    cdef:
        INDEX_t size
        BOOL_t coarsest, finest, allocated
        public {SCALAR}_t[::1] rhs
        public {SCALAR}_t[::1] sol
        public {SCALAR}_t[::1] temp
        public {SCALAR}_t[::1] D
        public {SCALAR_label}LinearOperator A
        public {SCALAR_label}LinearOperator R
        public {SCALAR_label}LinearOperator P
        public {SCALAR_label}smoother smoother
        public tuple smootherType

    def __init__(self, INDEX_t size, BOOL_t coarsest, BOOL_t finest):
        self.size = size
        self.coarsest = coarsest
        self.finest = finest
        self.rhs = None
        self.sol = None
        self.temp = None
        self.D = None
        self.A = None
        self.R = None
        self.P = None
        self.smoother = None
        self.allocated = False

    cdef void allocate(self):
        if not self.allocated:
            if self.coarsest:
                self.rhs = uninitialized(self.size, dtype={SCALAR})
                self.sol = uninitialized(self.size, dtype={SCALAR})
            elif self.finest:
                self.temp = uninitialized(self.size, dtype={SCALAR})
            else:
                self.rhs = uninitialized(self.size, dtype={SCALAR})
                self.sol = uninitialized(self.size, dtype={SCALAR})
                self.temp = uninitialized(self.size, dtype={SCALAR})
            self.allocated = True

    def __getitem__(self, str key):
        if key in ('A', 'R', 'P', 'D'):
            return getattr(self, key)
        else:
            raise NotImplementedError(key)

    def __setitem__(self, str key, {SCALAR_label}LinearOperator value):
        if key in ('A', 'R', 'P', 'D'):
            self.__setattr__(self, key, value)
        else:
            raise NotImplementedError(key)


######################################################################
# Multi-level solvers

cdef class {SCALAR_label}multigrid({SCALAR_label_lc_}iterative_solver):
    def __init__(self,
                 myHierarchyManager,
                 smoother=('jacobi', {'omega': 2.0/3.0}),
                 BOOL_t logging=False,
                 **kwargs):
        cdef:
            INDEX_t numLevels, length
            list levels
            dict lvlDict
            {SCALAR_label}levelMemory lvl
        if not isinstance(myHierarchyManager, hierarchyManager):
            myHierarchyManager = hierarchyManager.fromLevelList(myHierarchyManager, comm=None)
        self.hierarchyManager = myHierarchyManager
        self.PLogger = PLogger() if logging else FakePLogger()

        fineHierarchy = myHierarchyManager.builtHierarchies[len(myHierarchyManager.builtHierarchies) - 1]
        levels = fineHierarchy.getLevelList(recurse=False)

        numLevels = len(levels)
        self.levels = []
        lvlDict = levels[0]
        try:
            # A on level 0 might be a global matrix, that won't work in a parallel setting
            length = lvlDict['R'].shape[0]
        except KeyError:
            length = lvlDict['A'].shape[0]
        lvl = {SCALAR_label}levelMemory(length, True, False)
        lvl.A = lvlDict['A']
        self.levels.append(lvl)
        for lvlNo in range(1, numLevels):
            lvlDict = levels[lvlNo]
            length = lvlDict['A'].shape[0]
            lvl = {SCALAR_label}levelMemory(length, False, lvlNo == numLevels-1)
            lvl.A = lvlDict['A']
            if '{SCALAR}' == 'COMPLEX':
                if isinstance(lvlDict['R'], CSR_LinearOperator):
                    lvl.R = wrapRealToComplexCSR(lvlDict['R'])
                else:
                    lvl.R = lvlDict['R']
                if isinstance(lvlDict['P'], CSR_LinearOperator):
                    lvl.P = wrapRealToComplexCSR(lvlDict['P'])
                else:
                    lvl.P = lvlDict['P']
            else:
                lvl.R = lvlDict['R']
                lvl.P = lvlDict['P']
            self.levels.append(lvl)

        {SCALAR_label_lc_}iterative_solver.__init__(self, lvl.A)
        self.maxIter = 50

        if 'multilevelAlgebraicOverlapManager' in levels[len(levels)-1]:
            overlap = levels[numLevels-1]['multilevelAlgebraicOverlapManager']
            if overlap.comm.size == 1:
                overlap = None
        else:
            overlap = None
        self.overlap = overlap

        # set norm and inner product
        if overlap:
            self.setOverlapNormInner(self.overlap, numLevels-1)
            self.comm = self.overlap.comm

        self.cycle = V

        if not isinstance(smoother, list):
            smoother = [smoother]*numLevels
        else:
            assert len(smoother) == numLevels
        for lvlNo in range(1, numLevels):
            if not isinstance(smoother[lvlNo], tuple):
                self.levels[lvlNo].smootherType = (smoother[lvlNo], {})
            else:
                self.levels[lvlNo].smootherType = smoother[lvlNo]

    cpdef void setup(self, {SCALAR_label}LinearOperator A=None):
        cdef:
            INDEX_t lvlNo
            INDEX_t numLevels = len(self.levels)
            {SCALAR_label}levelMemory lvl
            tuple smoother

        {SCALAR_label_lc_}iterative_solver.setup(self, A)

        ########################################################################
        # set smoothers
        lvl = self.levels[0]
        lvl.allocate()
        for lvlNo in range(1, numLevels):
            lvl = self.levels[lvlNo]
            lvl.allocate()
            smoother = lvl.smootherType
            smoother[1]['overlaps'] = self.overlap
            # get matrix diagonals, accumulate if distributed
            if not smoother[0] in ('ilu', 'block_jacobi'):
                if self.overlap:
                    lvl.D = uninitialized((lvl.A.num_rows), dtype={SCALAR})
                    self.overlap.accumulate{SCALAR_label}(lvl.A.diagonal, lvl.D, level=lvlNo)
                else:
                    lvl.D = lvl.A.diagonal
            tempMem = np.array(lvl.temp, copy=False)
            if self.overlap is not None:
                lvlOverlap = self.overlap.levels[lvlNo]
            else:
                lvlOverlap = None
            if smoother[0] == 'jacobi':
                lvl.smoother = {SCALAR_label}jacobiSmoother(lvl.A, lvl.D, smoother[1], tempMem, overlap=lvlOverlap)
            elif smoother[0] == 'block_jacobi':
                lvl.smoother = {SCALAR_label}blockJacobiSmoother(lvl.A, smoother[1], tempMem, overlap=lvlOverlap)
            elif smoother[0] == 'gauss_seidel':
                lvl.smoother = gaussSeidelSmoother(lvl.A, lvl.D, smoother[1], tempMem, overlap=lvlOverlap)
            elif smoother[0] == 'sor':
                lvl.smoother = sorSmoother(lvl.A, lvl.D, smoother[1], tempMem, overlap=lvlOverlap)
            elif smoother[0] == 'ssor':
                lvl.smoother = ssorSmoother(lvl.A, lvl.D, smoother[1], tempMem, overlap=lvlOverlap)
            elif smoother[0] == 'chebyshev':
                lvl.smoother = chebyshevSmoother(lvl.A, lvl.D, smoother[1], tempMem, overlap=lvlOverlap)
            elif smoother[0] == 'gmres':
                lvl.smoother = {SCALAR_label}gmresSmoother(lvl.A, lvl.D, smoother[1], overlap=lvlOverlap)
            elif smoother[0] == 'ilu':
                lvl.smoother = iluSmoother(lvl.A, smoother[1], tempMem, overlap=lvlOverlap)
            elif smoother[0] == 'flexible':
                lvl.smoother = flexibleSmoother(lvl.A, smoother[1], overlap=lvlOverlap)
            else:
                raise NotImplementedError(smoother[0])

        ########################################################################
        # set coarse solver
        myHierarchyManager = self.hierarchyManager
        if len(myHierarchyManager.builtHierarchies) > 1:
            coarseHierarchyManager = myHierarchyManager.getSubManager()
            coarseHierarchy = coarseHierarchyManager.builtHierarchies[len(coarseHierarchyManager.builtHierarchies)-1]
            coarseSolverName = coarseHierarchy.params['solver']
            coarseSolverParams = coarseHierarchy.params['solver_params']
            if coarseSolverName.find('MG') >= 0:
                coarseSolverParams['smoother'] = smoother
            self.coarse_solver = {SCALAR_label}coarseSolver(coarseHierarchyManager, self.PLogger, coarseSolverName, **coarseSolverParams)
        else:
            fineHierarchy = myHierarchyManager.builtHierarchies[0]
            coarseSolverName = fineHierarchy.params['solver']
            from PyNucleus_base import solverFactory
            if solverFactory.isRegistered(coarseSolverName):
                self.coarse_solver = solverFactory.build(coarseSolverName, A=self.levels[0].A, setup=True)
            else:
                raise NotImplementedError("No coarse solver named \"{}\"".format(coarseSolverName))
        self.coarse_solver.setup()

        self.initialized = True

    cdef void solveOnLevel(self, int lvlNo, {SCALAR}_t[::1] b, {SCALAR}_t[::1] x,
                           BOOL_t simpleResidual=False):
        cdef:
            {SCALAR}_t[::1] res, correction, defect, solcg
            INDEX_t i
            str label = str(lvlNo)
            FakeTimer Timer = self.PLogger.Timer(label, manualDataEntry=True)
            {SCALAR_label}levelMemory lvl, lvlCoarse
        if lvlNo == 0:
            Timer.start()
            if isinstance(self.coarse_solver, {SCALAR_label_lc_}iterative_solver):
                self.coarse_solver.tolerance = self._tol
                self.coarse_solver.maxIter = 1
            self.coarse_solver.solve(b, x)
            Timer.end()
        else:
            Timer.start()
            lvl = self.levels[lvlNo]
            lvlCoarse = self.levels[lvlNo-1]
            solcg = lvlCoarse.sol
            defect = lvlCoarse.rhs
            res = lvl.temp
            correction = lvl.temp

            # apply presmoother to x, result in x
            lvl.smoother.eval(b, x, postsmoother=False,
                              simpleResidual=simpleResidual)

            # get residual in res -> temp
            lvl.A.residual(x, b, res)

            # restrict res -> temp to defect -> rhs
            lvl.R.matvec(res, defect)
            Timer.end()

            # solve on coarser level with rhs defect -> rhs into solcg -> sol
            solcg[:] = 0.0
            simpleResidual = True
            for i in range(self.cycle):
                self.solveOnLevel(lvlNo-1, defect, solcg,
                                  simpleResidual=simpleResidual)
                simpleResidual = False

            Timer.start()
            # prolong solcg -> sol to correction -> temp
            lvl.P.matvec(solcg, correction)
            # lvl.P.matvec_no_overwrite(solcg, x)

            # update fine grid solution
            update(x, correction)

            # apply postsmoother to x, result in x
            lvl.smoother.eval(b, x, postsmoother=True)
            Timer.end()
        Timer.enterData()

    def asPreconditioner(self, INDEX_t maxIter=1, CycleType cycle=V):
        return {SCALAR_label}multigridPreconditioner(self, cycle, maxIter)

    cdef int solve(self,
                   {SCALAR}_t[::1] b,
                   {SCALAR}_t[::1] x) except -1:
        {SCALAR_label_lc_}iterative_solver.solve(self, b, x)

        # For a distributed solve, b should be distributed, and x accumulated.
        # The solution vector is accumulated.
        cdef:
            INDEX_t iterNo, lvlNo
            INDEX_t numLevels = len(self.levels)
            REAL_t tol = self.tol
            INDEX_t maxiter = self.maxIter
            {SCALAR}_t[::1] res
            REAL_t n
            BOOL_t simpleResidual = False
            {SCALAR_label}LinearOperator A, R, P
            {SCALAR}_t[::1] b_fine
            {SCALAR}_t[::1] b_coarse, x_fine, x_coarse
            {SCALAR_label}levelMemory lvl
            CycleType cycle
            BOOL_t doFMG
            list residuals = []

        lvl = self.levels[numLevels-1]
        A = lvl.A
        res = lvl.temp

        # if isinstance(self.coarse_solver, coarseSolver_MG):
        #     self._tol = 0.
        # else:
        self._tol = tol

        doFMG = False
        if self.cycle in (FMG_V, FMG_W):
            doFMG = True
            cycle = self.cycle
            if self.cycle == FMG_V:
                self.cycle = V
            else:
                self.cycle = W

            # coarsen rhs to all levels
            b_fine = b
            lvl = self.levels[numLevels-1]
            for lvlNo in range(numLevels-2, -1, -1):
                R = lvl.R
                lvl = self.levels[lvlNo]
                b_coarse = lvl.rhs
                R.matvec(b_fine, b_coarse)
                b_fine = lvl.rhs

            # FMG cycle
            lvl = self.levels[0]
            for lvlNo in range(numLevels-2):
                b_coarse = lvl.rhs
                x_coarse = lvl.sol
                lvl = self.levels[lvlNo+1]
                x_fine = lvl.sol
                P = lvl.P
                self.solveOnLevel(lvlNo, b_coarse, x_coarse)
                P.matvec(x_coarse, x_fine)
            lvlNo = numLevels-2
            lvl = self.levels[lvlNo]
            b_coarse = lvl.rhs
            x_coarse = lvl.sol
            lvl = self.levels[lvlNo+1]
            P = lvl.P
            self.solveOnLevel(lvlNo, b_coarse, x_coarse)
            P.matvec(x_coarse, x)
            lvl.smoother.eval(b, x, postsmoother=True)

            iterNo = 1
        else:
            if self.x0 is None:
                simpleResidual = True
            iterNo = 0

        A.residual(x, b, res, simpleResidual)
        n = self.norm.eval(res, False)
        self.PLogger.addValue('residual', n)
        residuals.append(n)

        while (residuals[len(residuals)-1] > tol) and (iterNo < maxiter):
            iterNo += 1
            self.solveOnLevel(numLevels-1, b, x, simpleResidual)
            simpleResidual = False
            A.residual(x, b, res, False)
            n = self.norm.eval(res, False)
            self.PLogger.addValue('residual', n)
            residuals.append(n)
        if doFMG:
            self.cycle = cycle
        self.residuals = residuals
        return iterNo

    cpdef int solveFMG(self,
                       {SCALAR}_t[::1] b,
                       {SCALAR}_t[::1] x):
        cdef:
            CycleType cycle
        cycle = self.cycle
        self.cycle = FMG_V
        numIter = self.solve(b, x)
        self.cycle = cycle
        return numIter

    def __str__(self):
        cdef:
            INDEX_t numLevels = len(self.levels)
        if self.overlap is None:
            columns = []
            for lvlNo in range(numLevels-1, -1, -1):
                lvl = self.levels[lvlNo]
                A = lvl.A
                row = [lvlNo, '{:,}'.format(A.shape[0])]
                if hasattr(A, 'nnz'):
                    row.append('{:,}'.format(A.nnz))
                    row.append('{:,}'.format(A.nnz/A.shape[0]))
                else:
                    row.append('')
                    row.append('')
                row.append(str(lvl.smoother) if lvlNo > 0 else str(self.coarse_solver))
                columns.append(row)
            return tabulate(columns, headers=['level', 'unknowns', 'nnz', 'nnz/row', 'solver']) + '\n'
        else:
            num_dofs = [self.overlap.countDoFs(self.levels[lvlNo].A.shape[0], lvlNo)
                        for lvlNo in range(numLevels)]
            num_nnz = [self.overlap.comm.allreduce(self.levels[lvlNo].A.nnz)
                       for lvlNo in range(numLevels)]
            coarse_solver_descr = str(self.coarse_solver)
            if coarse_solver_descr.find('\n') >= 0:
                short_coarse_solver_descr = 'Coarse solver'
                coarse_solver_descr = '\n\n'+coarse_solver_descr
            else:
                short_coarse_solver_descr = coarse_solver_descr
                coarse_solver_descr = ''
            if self.overlap.comm.rank == 0:
                return tabulate([[lvlNo,
                                  '{:,}'.format(num_dofs[lvlNo]),
                                  '{:,}'.format(num_dofs[lvlNo]/self.overlap.comm.size),
                                  '{:,}'.format(num_nnz[lvlNo]),
                                  '{:,}'.format(num_nnz[lvlNo]/self.overlap.comm.size),
                                  '{:,}'.format(num_nnz[lvlNo]/num_dofs[lvlNo]),
                                  str(self.levels[lvlNo].smoother) if lvlNo > 0 else short_coarse_solver_descr]
                                 for lvlNo in range(numLevels-1, -1, -1)],
                                headers=['level', 'unknowns', 'unknowns/rank', 'nnz', 'nnz/rank', 'nnz/row', 'solver']) + coarse_solver_descr + '\n'
            else:
                return ''

    def iterationMatrix(self):
        n = self.num_rows
        M = np.zeros((n, n))
        rhs = np.zeros((n), dtype={SCALAR})
        maxiter = self.maxIter
        self.maxIter = 1
        zeroInitialGuess = self.x0 is None
        if not zeroInitialGuess:
            initGuess = np.array(self.x0, copy=True)
        for i, x0 in enumerate(np.eye(n)):
            x = np.zeros((n), dtype={SCALAR})
            self.setInitialGuess(x0)
            self.solve(b=rhs, x=x)
            M[:, i] = x
        self.maxIter = maxiter
        if zeroInitialGuess:
            self.setInitialGuess()
        else:
            self.setInitialGuess(initGuess)
        return M

    def operatorComplexity(self):
        return sum([lvl.A.nnz for lvl in self.levels])/self.levels[len(self.levels)-1].A.nnz


cdef class {SCALAR_label}multigridPreconditioner({SCALAR_label_lc_}preconditioner):
    cdef:
        {SCALAR_label}multigrid ml
        CycleType cycle
        INDEX_t numLevels, maxIter

    def __init__(self, {SCALAR_label}multigrid ml, CycleType cycle, INDEX_t maxIter=1):
        {SCALAR_label_lc_}preconditioner.__init__(self, ml)
        self.ml = ml
        self.cycle = cycle
        self.maxIter = maxIter
        self.numLevels = len(self.ml.levels)

    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        assert self.ml.initialized, 'solOp not initialized'
        cdef:
            BOOL_t simpleResidual = True
            INDEX_t iterNo
        self.cycle, self.ml.cycle = self.ml.cycle, self.cycle
        self.ml._tol = 1e-8
        y[:] = 0.
        for iterNo in range(self.maxIter):
            self.ml.solveOnLevel(self.numLevels-1, x, y, simpleResidual=simpleResidual)
            simpleResidual = False
        self.cycle, self.ml.cycle = self.ml.cycle, self.cycle
        return 0

    def __str__(self):
        return '{} iterations of {}-cycle\n{}'.format(self.maxIter, self.cycle, self.ml)
