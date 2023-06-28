###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus_base import solverFactory
from PyNucleus_base.utilsFem import (classWithComputedDependencies,
                                     problem,
                                     generates)
from PyNucleus_base.ip_norm import (ip_distributed_nonoverlapping,
                                    norm_distributed_nonoverlapping)
from PyNucleus_base.solvers import iterative_solver
from PyNucleus_base.linear_operators import Dense_LinearOperator
from PyNucleus_base.timestepping import CrankNicolson, ImplicitEuler
from PyNucleus_fem.factories import functionFactory
from PyNucleus_fem.functions import constant
from PyNucleus_fem.quadrature import simplexXiaoGimbutas
from PyNucleus_fem.DoFMaps import Product_DoFMap
from PyNucleus_multilevelSolver import hierarchyManager
from copy import copy
from . helpers import (multilevelDirichletCondition,
                       paramsForFractionalHierarchy)
from . fractionalOrders import singleVariableUnsymmetricFractionalOrder
from . kernelsCy import FRACTIONAL
from . nonlocalProblems import (DIRICHLET,
                                NEUMANN, HOMOGENEOUS_NEUMANN,
                                transientFractionalProblem)
import logging


class stationaryModelSolution(classWithComputedDependencies):
    def __init__(self, discretizedProblem, u, **kwargs):
        super().__init__()
        self.discretizedProblem = discretizedProblem
        self.u = u
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @generates('u_interp')
    def interpolateAnalyticSolution(self, u, analyticSolution):
        if analyticSolution is not None:
            self.u_interp = u.dm.interpolate(analyticSolution)
        else:
            self.u_interp = None

    @generates('u_interp_global')
    def interpolateGlobalAnalyticSolution(self, u_global, analyticSolution):
        if analyticSolution is not None:
            self.u_interp_global = u_global.dm.interpolate(analyticSolution)
        else:
            self.u_interp_global = None

    @generates('u_global')
    def buildGlobalSolution(self, u):
        comm = self.discretizedProblem._driver.comm
        if comm.size > 1:
            temp = self.discretizedProblem._driver.comm.reduce(self.discretizedProblem.A.lclP*u)
            if comm.rank == 0:
                u_global = self.discretizedProblem.dm.zeros()
                u_global.assign(temp)
            else:
                u_global = None
        else:
            u_global = self.u
        self.u_global = u_global

    @generates('u_augmented')
    def buildAugmentedSolution(self, u, dirichletData):
        if u.dm.num_boundary_dofs > 0:
            dmBC = u.dm.getComplementDoFMap()
            uBC = dmBC.interpolate(dirichletData)
            self.u_augmented = u.augmentWithBoundaryData(uBC)
        else:
            self.u_augmented = u

    @generates('L2_error')
    def computeL2error(self, u, u_interp, analyticSolution, exactL2Squared):
        if exactL2Squared is not None:
            if u.dm == self.discretizedProblem.dmInterior:
                M = self.discretizedProblem.massInterior
            elif u.dm == self.discretizedProblem.dm:
                M = self.discretizedProblem.mass
            else:
                M = u.dm.assembleMass()
            z = u.dm.assembleRHS(analyticSolution)
            self.L2_error = np.sqrt(abs(exactL2Squared - 2*z.inner(u) + u.inner(M*u)))
        else:
            self.L2_error = None

    @generates('rel_L2_error')
    def computeRelL2error(self, L2_error, exactL2Squared):
        if (L2_error is not None) and (exactL2Squared is not None):
            self.rel_L2_error = L2_error/np.sqrt(exactL2Squared)
        else:
            self.rel_L2_error = None

    @generates('Hs_error')
    def computeHserror(self, u, b, exactHsSquared):
        if exactHsSquared is not None:
            self.Hs_error = np.sqrt(abs(b.inner(u, False, True) - exactHsSquared))
        else:
            self.Hs_error = None

    @generates('rel_Hs_error')
    def computeRelHserror(self, Hs_error, exactHsSquared):
        if (Hs_error is not None) and (exactHsSquared is not None):
            self.rel_Hs_error = Hs_error/np.sqrt(exactHsSquared)
        else:
            self.rel_Hs_error = None

    @generates('L2_error_interp')
    def computeL2errorInterpolated(self, u, u_interp):
        if u_interp is not None:
            if u.dm == self.discretizedProblem.dmInterior:
                M = self.discretizedProblem.massInterior
            elif u.dm == self.discretizedProblem.dm:
                M = self.discretizedProblem.mass
            else:
                M = u.dm.assembleMass()
            self.L2_error_interp = np.sqrt((u-u_interp).inner(M*(u-u_interp), True, False))
        else:
            self.L2_error_interp = None

    @generates('rel_L2_error_interp')
    def computeRelL2errorInterpolated(self, u_interp, L2_error_interp):
        if L2_error_interp is not None:
            if u_interp.dm == self.discretizedProblem.dmInterior:
                M = self.discretizedProblem.massInterior
            elif u_interp.dm == self.discretizedProblem.dm:
                M = self.discretizedProblem.mass
            else:
                M = u_interp.dm.assembleMass()
            self.rel_L2_error_interp = L2_error_interp/np.sqrt(u_interp.inner(M*u_interp, True, False))
        else:
            self.rel_L2_error_interp = None

    @generates('Linf_error_interp')
    def computeLinferrorInterpolated(self, u, u_interp):
        if u_interp is not None:
            self.Linf_error_interp = np.absolute(u-u_interp).max()
        else:
            self.Linf_error_interp = None

    @generates('rel_Linf_error_interp')
    def computeRelLinferrorInterpolated(self, u_interp, Linf_error_interp):
        if Linf_error_interp is not None:
            self.rel_Linf_error_interp = Linf_error_interp/np.absolute(u_interp).max()
        else:
            self.rel_Linf_error_interp = None

    @generates('error')
    def buildErrorVector(self, u, u_interp):
        if u_interp is not None:
            errVec = u-u_interp
            if isinstance(errVec.dm, Product_DoFMap):
                self.error = errVec.dm.scalarDM.zeros()
                self.error.assign(np.sqrt(sum(errVec.getComponent(j).toarray()**2 for j in range(errVec.dm.numComponents))))
            else:
                self.error = errVec.dm.zeros()
                self.error.assign(np.absolute(errVec.toarray()))
        else:
            self.error = None

    @generates('deformedMesh')
    def getDeformedMesh(self, u_augmented):
        mesh = u_augmented.dm.mesh
        assert u_augmented.dm.scalarDM.num_dofs == mesh.num_vertices
        assert u_augmented.dm.scalarDM.num_boundary_dofs == 0
        c = mesh.vertices_as_array
        components = u_augmented.getComponents()
        processed = set()
        for cellNo in range(mesh.num_cells):
            for dofNo in range(u_augmented.dm.scalarDM.dofs_per_element):
                dof = u_augmented.dm.scalarDM.cell2dof_py(cellNo, dofNo)
                if dof not in processed:
                    vertex = mesh.cells[cellNo, dofNo]
                    for component in range(u_augmented.dm.numComponents):
                        c[vertex, component] += components[component][dof]
                    processed.add(dof)
        deformedMesh = mesh.copy()
        deformedMesh.vertices = c
        self.deformedMesh = deformedMesh

    def plotSolution(self):
        dim = self.u.dm.mesh.dim
        self.u.plot(label='numerical solution')
        if dim == 1 and self.u_interp is not None:
            import matplotlib.pyplot as plt
            self.u_interp.plot(label='analytic solution')
            plt.legend()
        elif dim in (2, 3):
            import matplotlib.pyplot as plt
            plt.gca().set_aspect('equal')

    def plotSolutionComponents(self, plotDefaults={}):
        from PyNucleus.fem.mesh import plotManager
        pm = plotManager(self.u.dm.scalarDM.mesh, self.u.dm.scalarDM,
                         defaults=plotDefaults)
        for c in range(self.u.dm.numComponents):
            pm.add(self.u.getComponent(c), label='u'+str(c))
        pm.plot()

    def reportErrors(self, group):
        if self.L2_error is not None:
            group.add('L2 error', self.L2_error, rTol=3e-2, aTol=1e-8)
        if self.rel_L2_error is not None:
            group.add('relative L2 error', self.rel_L2_error, rTol=3e-2, aTol=1e-8)
        if self.L2_error_interp is not None:
            group.add('L2 error interpolated', self.L2_error_interp, rTol=3e-2, aTol=1e-8)
        if self.rel_L2_error_interp is not None:
            group.add('relative interpolated L2 error', self.rel_L2_error_interp, rTol=3e-2, aTol=1e-8)
        if self.Linf_error_interp is not None:
            group.add('Linf error interpolated', self.Linf_error_interp, rTol=3e-2, aTol=1e-8)
        if self.rel_Linf_error_interp is not None:
            group.add('relative interpolated Linf error', self.rel_Linf_error_interp, rTol=3e-2, aTol=1e-8)
        if self.Hs_error is not None:
            group.add('Hs error', self.Hs_error, rTol=3e-2, aTol=1e-8)
        if self.rel_Hs_error is not None:
            group.add('relative Hs error', self.rel_Hs_error, rTol=3e-2, aTol=1e-8)

    def reportSolve(self, group):
        group.add('solver', self.discretizedProblem.solverType)
        if isinstance(self.discretizedProblem.solver, iterative_solver):
            group.add('iterations', self.iterations)
            group.add('implicit residual norm', self.residuals[-1])
            group.add('explicit residual norm', self.explicitResidualError)
            group.add('tolerance', self.tol)


class transientModelSolution(classWithComputedDependencies):
    def __init__(self, discretizedProblem, u, **kwargs):
        super().__init__()
        self.discretizedProblem = discretizedProblem
        self.u = u
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def getSingleTimeStepSolution(self, timeStep=None):
        if timeStep is None:
            timeStep = self.u.numVectors-1
        t = timeStep*self.dt
        return stationaryModelSolution(self,
                                       self.u[timeStep],
                                       t=t,
                                       k=timeStep,
                                       exactL2squared=self.discretizedProblem.continuumProblem.exactL2Squared(t) if self.discretizedProblem.continuumProblem.exactL2Squared is not None else None,
                                       analyticSolution=self.discretizedProblem.continuumProblem.analyticSolution(t) if self.discretizedProblem.continuumProblem.analyticSolution is not None else None,
                                       dirichletData=self.dirichletData(t) if self.dirichletData is not None else None)

    @generates('error')
    def buildErrorVector(self):
        self.error = self.getSingleTimeStepSolution().error

    @generates('L2_error')
    def computeL2error(self, u, analyticSolution, exactL2Squared, timesVector):
        if exactL2Squared is not None:
            if u.dm == self.discretizedProblem.dmInterior:
                M = self.discretizedProblem.massInterior
            elif u.dm == self.discretizedProblem.dm:
                M = self.discretizedProblem.mass
            else:
                M = u.dm.assembleMass()
            integral = 0.
            for k in range(timesVector.shape[0]):
                if k == 0:
                    fac = timesVector[k+1]-timesVector[k]
                elif k == timesVector.shape[0]-1:
                    fac = timesVector[k]-timesVector[k-1]
                else:
                    fac = timesVector[k+1]-timesVector[k-1]
                t = timesVector[k]
                z = u.dm.assembleRHS(analyticSolution(t))
                integral += fac*abs(exactL2Squared(t) - 2*z.inner(u[k]) + u[k].inner(M*u[k]))
            self.L2_error = np.sqrt(integral)
        else:
            self.L2_error = None

    @generates('final_L2_error')
    def computeFinalL2error(self, u, analyticSolution, exactL2Squared, finalTime):
        if exactL2Squared is not None:
            if u.dm == self.discretizedProblem.dmInterior:
                M = self.discretizedProblem.massInterior
            elif u.dm == self.discretizedProblem.dm:
                M = self.discretizedProblem.mass
            else:
                M = u.dm.assembleMass()
            z = u.dm.assembleRHS(analyticSolution(finalTime))
            numTimeSteps = u.numVectors-1
            integral = abs(exactL2Squared(finalTime) - 2*z.inner(u[numTimeSteps]) + u[numTimeSteps].inner(M*u[numTimeSteps]))
            self.final_L2_error = np.sqrt(integral)
        else:
            self.final_L2_error = None

    @generates('L2_norm')
    def computeL2norm(self, u, timesVector):
        if u.dm == self.discretizedProblem.dmInterior:
            M = self.discretizedProblem.massInterior
        elif u.dm == self.discretizedProblem.dm:
            M = self.discretizedProblem.mass
        else:
            M = u.dm.assembleMass()
        integral = 0.
        for k in range(timesVector.shape[0]):
            if k == 0:
                fac = timesVector[k+1]-timesVector[k]
            elif k == timesVector.shape[0]-1:
                fac = timesVector[k]-timesVector[k-1]
            else:
                fac = timesVector[k+1]-timesVector[k-1]
            integral += fac*abs(u[k].inner(M*u[k]))
        self.L2_norm = np.sqrt(integral)

    def plotSolution(self):
        if self.u.dm.mesh.dim == 1:
            import matplotlib.pyplot as plt
            self.u.plot(yvals=self.timesVector)
            ax = plt.gca()
            ax.set_xlabel('$x$')
            ax.set_ylabel('$t$')
        else:
            self.getSingleTimeStepSolution().plotSolution()

    def plotSolutionComponents(self):
        self.getSingleTimeStepSolution().plotSolutionComponents()

    def reportErrors(self, group):
        group.add('L^2(0,T; L^2(Omega)) norm', self.L2_norm, rTol=3e-2, aTol=1e-8)
        if self.L2_error is not None:
            group.add('L^2(0,T; L^2(Omega)) error', self.L2_error, rTol=3e-2, aTol=1e-8)
        if self.final_L2_error is not None:
            group.add('L^2(Omega) error at t=finalTime', self.final_L2_error, rTol=3e-2, aTol=1e-8)

    def reportSolve(self, group):
        pass


class discretizedNonlocalProblem(problem):
    def __init__(self, driver, continuumProblem):
        super().__init__(driver)
        self.continuumProblem = continuumProblem
        self.addRemote(self.continuumProblem)

    def setDriverArgs(self):
        self.setDriverFlag('solverType', acceptedValues=['cg-mg', 'gmres-mg', 'lu', 'mg', 'cg-jacobi', 'gmres-jacobi'], help='solver for the linear system')
        self.setDriverFlag('maxiter', 100, help='maximum number of iterations')
        self.setDriverFlag('tol', 1e-6, help='solver tolerance')
        self.setDriverFlag('quadType', acceptedValues=['auto', 'classical', 'general', 'adaptive', 'classical-refactored'])
        self.setDriverFlag('quadTypeBoundary', acceptedValues=['auto', 'classical', 'general', 'adaptive', 'classical-refactored'])
        self.setDriverFlag('matrixFormat', acceptedValues=['H2', 'sparse', 'sparsified', 'dense'], help='matrix format')
        self.setDriverFlag('debugAssemblyTimes', False)

    def processCmdline(self, params):
        matrixFormat = params['matrixFormat']
        kernelType = params['kernelType']
        if matrixFormat.upper() == 'H2' and kernelType != 'fractional':
            matrixFormat = 'sparse'
            params['matrixFormat'] = matrixFormat
        super().processCmdline(params)

    @generates(['hierarchy', 'bc', 'finalMesh', 'dm', 'dmBC', 'dmInterior', 'R_interior', 'P_interior', 'R_BC', 'P_BC'])
    def buildHierarchy(self, mesh, kernel, sArgs, rangedKernel, solverType, matrixFormat, tag, boundaryCondition, domainIndicator, fluxIndicator,
                       zeroExterior, noRef, eta, target_order, element, quadType, quadTypeBoundary):
        assert matrixFormat != 'H2' or (kernel.kernelType == FRACTIONAL), 'Hierarchical matrices are only implemented for fractional kernels'
        if rangedKernel is not None:
            hierarchy = self.directlyGetWithoutChecks('hierarchy')
            if hierarchy is not None:
                newHierarchy = []
                for lvl in range(len(hierarchy)):
                    newHierarchy.append({})
                    for key in hierarchy[lvl]:
                        newHierarchy[lvl][key] = hierarchy[lvl][key]
                newHierarchy[0]['sArgs'] = sArgs
                s = kernel.sValue
                for lvl in range(len(newHierarchy)):
                    if 'A' in newHierarchy[lvl]:
                        newHierarchy[lvl]['A'].set(s, 0)
                if hierarchy[0]['sArgs'] != newHierarchy[0]['sArgs']:
                    assert hierarchy != newHierarchy
                self.hierarchy = newHierarchy

                for prop in ['bc', 'finalMesh', 'dm', 'dmBC', 'dmInterior', 'R_interior', 'P_interior', 'R_BC', 'P_BC']:
                    setattr(self, prop, self.directlyGetWithoutChecks(prop))
                return

        with self.timer('hierarchy'):
            params = {}
            params['domain'] = mesh
            if rangedKernel is None:
                params['kernel'] = kernel
            else:
                params['kernel'] = rangedKernel
            params['solver'] = solverType
            params['tag'] = tag
            params['element'] = element
            params['boundaryCondition'] = boundaryCondition
            params['zeroExterior'] = zeroExterior
            params['target_order'] = target_order
            params['eta'] = eta
            params['keepMeshes'] = 'all'
            params['keepAllDoFMaps'] = True
            params['buildMass'] = True
            params['assemble'] = 'ALL' if solverType.find('mg') >= 0 else 'last'
            params['dense'] = matrixFormat == 'dense'
            params['matrixFormat'] = matrixFormat
            params['logging'] = True
            if self.debugAssemblyTimes:
                from PyNucleus.base.utilsFem import TimerManager
                tm = TimerManager(self.driver.logger, comm=self.driver.comm, memoryProfiling=False, loggingSubTimers=True)
                params['PLogger'] = tm.PLogger

            if quadType == 'auto':
                quadType = 'classical-refactored'
            params['quadType'] = quadType

            if quadTypeBoundary == 'auto':
                quadTypeBoundary = 'classical-refactored'
            params['quadTypeBoundary'] = quadTypeBoundary

            comm = self.driver.comm
            onRanks = [0]
            if comm is not None and comm.size > 1:
                onRanks = range(comm.size)
            hierarchies, connectors = paramsForFractionalHierarchy(noRef, params, onRanks)
            hM = hierarchyManager(hierarchies, connectors, params, comm)
            hM.setup()

        if boundaryCondition == DIRICHLET:
            bc = multilevelDirichletCondition(hM.getLevelList(), domainIndicator, fluxIndicator)
            hierarchy = bc.naturalLevels
            dmInterior = bc.naturalDoFMap
            dmBC = bc.dirichletDoFMap
        else:
            hierarchy = hM.getLevelList()
            bc = None
            dmInterior = hierarchy[-1]['DoFMap']
            dmBC = dmInterior.getComplementDoFMap()
        if rangedKernel is not None:
            hierarchy[0]['sArgs'] = sArgs
            s = kernel.sValue
            for lvl in range(len(hierarchy)):
                if 'A' in hierarchy[lvl]:
                    hierarchy[lvl]['A'].set(s, 0)
        self.dmBC = dmBC
        self.dm, self.R_interior, self.R_bc = dmInterior.getFullDoFMap(dmBC)
        self.P_interior = self.R_interior.transpose()
        self.P_bc = self.R_bc.transpose()

        self.finalMesh = hM['fine'].meshLevels[-1].mesh
        self.bc = bc
        self.dmInterior = dmInterior
        self.hierarchy = hierarchy
        assert 2*self.finalMesh.h < kernel.horizon.value, "h = {}, horizon = {}".format(self.finalMesh.h, kernel.horizon.value)

    @generates('mass')
    def buildMass(self, dm):
        self.mass = dm.assembleMass()

    @generates('massInterior')
    def buildMassInterior(self, dmInterior):
        self.massInterior = dmInterior.assembleMass()

    @generates('A')
    def getOperators(self, hierarchy):
        self.A = hierarchy[-1]['A']

    @generates('b')
    def buildRHS(self, rhs, dim, bc, dirichletData, boundaryCondition, solverType, dmInterior, hierarchy):
        self.b = dmInterior.assembleRHS(rhs, qr=simplexXiaoGimbutas(3, dim))
        if bc is not None:
            bc.setDirichletData(dirichletData)
            bc.applyRHScorrection(self.b)

        # pure Neumann condition -> project out nullspace
        if boundaryCondition in (NEUMANN, HOMOGENEOUS_NEUMANN):
            assert bc is None
            if solverType.find('mg') >= 0:
                hierarchy[0]['A'] = hierarchy[0]['A'] + Dense_LinearOperator.ones(*hierarchy[0]['A'].shape)
            else:
                hierarchy[-1]['A'] = hierarchy[-1]['A'] + Dense_LinearOperator.ones(*hierarchy[-1]['A'].shape)
            const = dmInterior.ones()
            self.b -= self.b.inner(const)/const.inner(const)*const

    @generates('solver')
    def buildSolver(self, solverType, tol, maxiter, hierarchy, kernel):
        from PyNucleus_base.solvers import iterative_solver
        if solverType[:2] == 'cg':
            assert kernel.symmetric, 'CG solver requires a symmetric matrix'
        solver = solverFactory.build(solverType, hierarchy=hierarchy)
        if isinstance(solver, iterative_solver):
            solver.tolerance = tol
            solver.maxIter = maxiter
            comm = self.driver.comm
            if comm is not None and comm.size > 1:
                solver.setNormInner(norm_distributed_nonoverlapping(comm),
                                    ip_distributed_nonoverlapping(comm))
        solver.setup()
        self.solver = solver

    @generates('modelSolution')
    def solve(self, b, bc, dm, dmInterior, P_interior, solver, boundaryCondition, analyticSolution, dirichletData, tol, maxiter):
        uInterior = dmInterior.zeros()
        with self.timer('solve {}'.format(self.__class__.__name__)):
            its = solver(b, uInterior)

        resError = (b-solver.A*uInterior).norm(False)

        if isinstance(solver, iterative_solver):
            assert its < maxiter, "Only reached residual error {} > tol = {} in {} iterations".format(resError, tol, its)
        # assert resError < tol, "Only reached residual error {} > tol = {}".format(resError, tol)

        # pure Neumann condition -> add nullspace components to match analytic solution
        if boundaryCondition in (NEUMANN, HOMOGENEOUS_NEUMANN) and analyticSolution is not None:
            uEx = dmInterior.interpolate(analyticSolution)
            const = dmInterior.ones()
            uInterior += (const.inner(uEx)-const.inner(uInterior))/const.inner(const) * const

        u = dm.empty()
        if boundaryCondition in (DIRICHLET, ):
            u.assign(bc.augmentDirichlet(uInterior))
        else:
            u.assign(P_interior*uInterior)

        data = {'iterations': its,
                'uInterior': uInterior,
                'explicitResidualError': resError,
                'b': b}
        if isinstance(solver, iterative_solver):
            data['tol'] = solver.tolerance
            data['maxIterations'] = solver.maxIter
            data['residuals'] = copy(solver.residuals)
            data['preconditionedResidualError'] = solver.residuals[-1]
        data['analyticSolution'] = analyticSolution
        data['exactL2Squared'] = None
        data['exactHsSquared'] = None
        data['dirichletData'] = dirichletData
        if hasattr(self.continuumProblem, 'exactHsSquared'):
            data['exactHsSquared'] = self.continuumProblem.exactHsSquared
        if hasattr(self.continuumProblem, 'exactL2Squared'):
            data['exactL2Squared'] = self.continuumProblem.exactL2Squared
        self.modelSolution = stationaryModelSolution(self, u, **data)

    def report(self, group):
        group.add('h', self.finalMesh.h)
        group.add('hmin', self.finalMesh.hmin)
        group.add('mesh quality', self.finalMesh.delta)
        group.add('DoFMap', str(self.dm))
        group.add('Interior DoFMap', str(self.dmInterior))
        group.add('Dirichlet DoFMap', str(self.dmBC))


class discretizedTransientProblem(discretizedNonlocalProblem):
    def __init__(self, driver, continuumProblem, keepAllTimeSteps=True):
        assert isinstance(continuumProblem, transientFractionalProblem), type(continuumProblem)
        super().__init__(driver, continuumProblem)
        self.addRemote(self.continuumProblem)
        self.keepAllTimeSteps = keepAllTimeSteps

    def setDriverArgs(self):
        super().setDriverArgs()

        self.setDriverFlag('timeStepperType', acceptedValues=['Crank-Nicolson', 'Implicit Euler'])
        self.setDriverFlag('theta', 0.5, help='Crank-Nicolson parameter')

        viz = self.driver.addGroup('viz')
        self.setDriverFlag('doMovie', False, help='Create a movie of the solution', group=viz)
        self.setDriverFlag('movieFrameStep', 10, group=viz)
        self.setDriverFlag('movieFolder', 'movie', group=viz)
        self.setDriverFlag('shading', acceptedValues=['gouraud', 'flat'], group=viz)

    def buildTransientHierarchy(self, hierarchy, alpha, beta):
        newHierarchy = []
        for lvl in range(len(hierarchy)):
            newHierarchy.append({})
            if 'M' in hierarchy[lvl]:
                newHierarchy[lvl]['A'] = alpha*hierarchy[lvl]['M']+beta*hierarchy[lvl]['A']
            for key in ['R', 'P']:
                if key in hierarchy[lvl]:
                    newHierarchy[lvl][key] = hierarchy[lvl][key]
        return newHierarchy

    def buildTransientSolver(self, solverType, tol, maxiter, hierarchy, alpha, beta):
        from PyNucleus_base.solvers import iterative_solver
        from PyNucleus_base.linear_operators import multiIntervalInterpolationOperator
        for lvl in range(len(hierarchy)):
            if ('A' in hierarchy[lvl]) and isinstance(hierarchy[lvl]['A'],
                                                      multiIntervalInterpolationOperator):
                assert hierarchy[lvl]['A'].getSelectedOp().derivative == 0, hierarchy[lvl]['A'].getSelectedOp().derivative
        transientHierarchy = self.buildTransientHierarchy(hierarchy, alpha, beta)
        solver = solverFactory.build(solverType, hierarchy=transientHierarchy)
        if isinstance(solver, iterative_solver):
            solver.tolerance = tol
            solver.maxIter = maxiter
            comm = self.driver.comm
            if comm is not None and comm.size > 1:
                solver.setNormInner(norm_distributed_nonoverlapping(comm),
                                    ip_distributed_nonoverlapping(comm))
        solver.setup()
        return solver

    def massApply(self, t, u, rhs):
        self.massInterior(u, rhs)

    def solverBuilder(self, t, alpha, beta):
        with self.timer('build solver {}'.format(self.__class__.__name__)):
            return self.buildTransientSolver(self.solverType, self.tol, self.maxiter, self.hierarchy, alpha, beta)

    def explicit(self, t, u, rhs):
        self.A(u, rhs)

    def massApply_adjoint(self, t, u, rhs):
        self.massInterior(u, rhs)
        rhs *= -1

    def solverBuilder_adjoint(self, t, alpha, beta):
        with self.timer('build adjoint solver {}'.format(self.__class__.__name__)):
            return self.buildTransientSolver(self.solverType, self.tol, self.maxiter, self.hierarchy, -alpha, beta)

    def explicit_adjoint(self, t, u, rhs):
        self.A(u, rhs)

    def forcingBuilder(self, t, rhs):
        force = self.rhs(t)
        if isinstance(force, constant) and force.value == 0:
            rhs.assign(0.)
        else:
            rhs.assign(self.dmInterior.assembleRHS(force, qr=self.qr))
        if self.dirichletData is not None:
            uBC = self.dmBC.interpolate(self.dirichletData(t))
            rhs -= self.A_BC*uBC

    @generates('qr')
    def buildQuadratureRule(self):
        self.qr = simplexXiaoGimbutas(order=3, dim=self.continuumProblem.dim)

    @generates(['dt', 'numTimeSteps'])
    def determineTimeSteps(self, finalMesh, finalTime, timeStepperType):
        if timeStepperType == 'Crank-Nicolson':
            dt = np.sqrt(finalMesh.h)
        elif timeStepperType == 'Implicit Euler':
            dt = finalMesh.h
        numTimeSteps = int(np.around(finalTime/dt))
        self.dt = finalTime/numTimeSteps
        self.numTimeSteps = numTimeSteps

    @generates('timesVector')
    def buildTimesVector(self, finalTime, numTimeSteps):
        self.timesVector = np.linspace(0, finalTime, numTimeSteps+1)

    @generates('stepper')
    def buildTimeStepper(self, timeStepperType, dt, dmInterior, theta, hierarchy):
        if timeStepperType == 'Crank-Nicolson':
            self.stepper = CrankNicolson(dmInterior,
                                         self.massApply,
                                         self.solverBuilder,
                                         self.forcingBuilder,
                                         self.explicit,
                                         theta=theta,
                                         dt=dt,
                                         explicitIslinearAndTimeIndependent=True)
        elif timeStepperType == 'Implicit Euler':
            self.stepper = ImplicitEuler(dmInterior,
                                         self.massApply,
                                         self.solverBuilder,
                                         self.forcingBuilder,
                                         self.explicit,
                                         dt=dt,
                                         explicitIslinearAndTimeIndependent=True)

    @generates('b')
    def buildRHS(self, dmInterior, numTimeSteps, timesVector, stepper):
        self.b = dmInterior.zeros(numTimeSteps)
        for k in range(numTimeSteps):
            t = timesVector[k]
            stepper.setRHS(t, stepper.dt, self.b[k])

    @generates('adjointStepper')
    def buildAdjointTimeStepper(self, timeStepperType, dt, dmInterior, theta, hierarchy):
        if timeStepperType == 'Crank-Nicolson':
            self.adjointStepper = CrankNicolson(dmInterior,
                                                self.massApply_adjoint,
                                                self.solverBuilder_adjoint,
                                                None,
                                                self.explicit_adjoint,
                                                theta=theta,
                                                dt=-dt,
                                                explicitIslinearAndTimeIndependent=True)
        elif timeStepperType == 'Implicit Euler':
            self.adjointStepper = ImplicitEuler(dmInterior,
                                                self.massApply_adjoint,
                                                self.solverBuilder_adjoint,
                                                None,
                                                self.explicit_adjoint,
                                                dt=-dt,
                                                explicitIslinearAndTimeIndependent=True)

    @generates(['initialSolution'])
    def setInitialCondition(self, dm, initial):
        if self.keepAllTimeSteps:
            self.initialSolution = dm.interpolate(initial)
        else:
            assert self.keepAllTimeSteps
            self.u = dm.interpolate(initial)

        if self.doMovie:
            if self._driver.isMaster:
                outputFolder = self.movieFolder
                movie_kwargs = {}
                if self.continuumProblem.dim == 2:
                    movie_kwargs['vmin'] = self.initialSolution.min()
                    movie_kwargs['vmax'] = self.initialSolution.max()
                    movie_kwargs['shading'] = self.shading
                self.mC = movieCreator(self.initialSolution, outputFolder)

    # def doSteps(self, stepCount=1):
    #     if stepCount <= 0:
    #         stepCount = self.numTimeSteps
    #     end = min(self.numTimeSteps, self.timeStep+stepCount)
    #     for i in range(self.timeStep, end):
    #         with self.timer('time step', level=logging.DEBUG):
    #             if self.keepAllTimeSteps:
    #                 self.u[i+1].assign(self.u[i])
    #                 u = self.u[i+1]
    #             else:
    #                 u = self.u
    #             uInterior = self.R_interior*u
    #             self.t = self.stepper(self.t, self.dt, uInterior)
    #             uBC = self.dmBC.interpolate(self.dirichletData(self.t))
    #             u.assign(self.P_interior*uInterior + self.P_bc*uBC)

    #         if self.doMovie and (i % self.movieFrameStep == self.movieFrameStep-1):
    #             mS = self.getModelSolution()
    #             u = mS.u_global
    #             if self._driver.isMaster:
    #                 self.mC.addFrame(u)
    #     self.timeStep = end
    #     if end == self.numTimeSteps:
    #         assert abs(self.t - self.continuumProblem.finalTime) < 1e-10, (self.t, self.continuumProblem.finalTime)
    #     if self.doMovie and (end == self.numTimeSteps) and self._driver.isMaster:
    #         self.mC.generateMovie()
    #     mS = self.getModelSolution()
    #     return mS

    @generates('modelSolution')
    def solve(self, numTimeSteps, dt, finalTime, timesVector, initialSolution, R_interior, stepper, dm, dmBC, dirichletData,
              P_interior, P_bc, b, exactL2Squared, analyticSolution):
        with self.timer('solve {}'.format(self.__class__.__name__)):
            t = 0.
            u = dm.zeros(numTimeSteps+1)
            u[0].assign(initialSolution)
            uInterior = R_interior*u[0]
            for i in range(numTimeSteps):
                with self.timer('time step', level=logging.DEBUG):
                    self.t = t = stepper(t, dt, uInterior, forcingVector=b[i])
                    if dirichletData is not None:
                        uBC = dmBC.interpolate(dirichletData(t))
                        u[i+1].assign(P_interior*uInterior + P_bc*uBC)
                    else:
                        u[i+1].assign(P_interior*uInterior)

                if self.doMovie and (i % self.movieFrameStep == self.movieFrameStep-1):
                    if self._driver.isMaster:
                        self.mC.addFrame(u[i+1])
            assert abs(t - finalTime) < 1e-10, (t, finalTime)
            if self.doMovie and self._driver.isMaster:
                self.mC.generateMovie()

        self.modelSolution = transientModelSolution(self,
                                                    u,
                                                    timesVector=timesVector,
                                                    dt=dt,
                                                    finalTime=finalTime,
                                                    exactL2Squared=exactL2Squared,
                                                    analyticSolution=analyticSolution,
                                                    dirichletData=dirichletData)

    @generates('adjointModelSolution')
    def solveAdjoint(self, numTimeSteps, dt, finalTime, timesVector, R_interior, adjointStepper, dm, dmInterior, P_interior, b):
        with self.timer('solve adjoint {}'.format(self.__class__.__name__)):
            t = finalTime
            p = dm.zeros(numTimeSteps+1)
            pInterior = dmInterior.zeros()
            for i in range(numTimeSteps, 0, -1):
                with self.timer('time step', level=logging.DEBUG):
                    t = adjointStepper(t, -dt, pInterior, forcingVector=b[i-1])
                    P_interior(pInterior, p[i])
            assert abs(t) < 1e-10, t

        self.adjointModelSolution = transientModelSolution(self,
                                                           p,
                                                           timeValues=timesVector,
                                                           dt=dt,
                                                           dirichletData=lambda t: functionFactory('constant', 0.))

    def report(self, group):
        super().report(group)
        group.add('dt', self.dt)
        group.add('numTimeSteps', self.numTimeSteps)
