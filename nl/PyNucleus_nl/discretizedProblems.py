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
from PyNucleus_base.timestepping import timestepperFactory
from PyNucleus_fem.factories import functionFactory
from PyNucleus_fem.functions import constant
from PyNucleus_fem.quadrature import simplexXiaoGimbutas
from PyNucleus_fem.DoFMaps import Product_DoFMap
from PyNucleus_multilevelSolver import hierarchyManager
from copy import copy
from . helpers import paramsForFractionalHierarchy
from . nonlocalProblems import (DIRICHLET,
                                NEUMANN, HOMOGENEOUS_NEUMANN,
                                transientFractionalProblem)
from . clusterMethodCy import H2Matrix, DistributedH2Matrix_globalData, DistributedH2Matrix_localData
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
                if hasattr(self.discretizedProblem.continuumProblem, 'mass_weight'):
                    M = u.dm.assembleMass(coefficient=self.discretizedProblem.continuumProblem.mass_weight)
                else:
                    M = u.dm.assembleMass()
            if hasattr(self.discretizedProblem.continuumProblem, 'mass_weight') and self.discretizedProblem.continuumProblem.mass_weight is not None:
                z = u.dm.assembleRHS(analyticSolution*self.discretizedProblem.continuumProblem.mass_weight)
            else:
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
    def computeHserror(self, uRestricted, b, exactHsSquared):
        if exactHsSquared is not None:
            assert b.dm == uRestricted.dm, (b.dm, uRestricted.dm)
            self.Hs_error = np.sqrt(abs(b.inner(uRestricted, False, True) - exactHsSquared))
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
        dim = self.u.dm.mesh.manifold_dim
        self.u.plot(label='numerical solution')
        if dim == 1 and self.u_interp is not None:
            import matplotlib.pyplot as plt
            self.u_interp.plot(label='analytic solution')
            plt.legend()
        elif dim in (2, 3):
            import matplotlib.pyplot as plt
            plt.gca().set_aspect('equal')

    def plotSolutionComponents(self, plotDefaults={}):
        from PyNucleus_fem.mesh import plotManager
        pm = plotManager(self.u.dm.scalarDM.mesh, self.u.dm.scalarDM,
                         defaults=plotDefaults)
        for c in range(self.u.dm.numComponents):
            pm.add(self.u.getComponent(c), label='u'+str(c))
        pm.plot()

    def plotRHS(self):
        self.uRestricted.dm.interpolate(self.rhs).plot(label='rhs')

    def exportVTK(self, filename):
        x = [self.u]
        labels = ['numerical_solution']
        if self.u_interp is not None:
            x.append(self.u_interp)
            labels.append('interpolated_analytic_solution')
        if self.error is not None:
            x.append(self.error)
            labels.append('error')
        self.u.dm.mesh.exportSolutionVTK(x, filename, labels=labels)

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
        group.add('iterations', self.iterations)
        if isinstance(self.discretizedProblem.solver, iterative_solver):
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
        driver.addToProcessHook(self.setTimerManager)

    def setDriverArgs(self):
        p = self.driver.addGroup('solver')
        self.setDriverFlag('solverType', acceptedValues=['cg-mg', 'gmres-mg', 'lu', 'mg',
                                                         'cg-jacobi', 'gmres-jacobi'], help='solver for the linear system', group=p)
        self.setDriverFlag('maxiter', 100, help='maximum number of iterations', group=p)
        self.setDriverFlag('tol', 1e-6, help='solver tolerance', group=p)

        p = self.driver.addGroup('assembly')
        self.setDriverFlag('quadType', acceptedValues=['auto', 'classical', 'general', 'adaptive', 'classical-refactored'], group=p)
        self.setDriverFlag('quadTypeBoundary', acceptedValues=['auto', 'classical', 'general', 'adaptive', 'classical-refactored'], group=p)
        self.setDriverFlag('matrixFormat', acceptedValues=['H2', 'sparse', 'sparsified', 'dense'], help='matrix format', group=p)
        self.setDriverFlag('debugAssemblyTimes', False, group=p)

    def setTimerManager(self, params):
        self._timer = self.driver.getTimer().getSubManager(logging.getLogger(__name__))

    @generates(['meshHierarchy', 'finalMesh',
                'dm', 'dmBC', 'dmInterior',
                'R_interior', 'P_interior',
                'R_bc', 'P_bc'])
    def buildMeshHierarchy(self, mesh, solverType, domainIndicator, fluxIndicator, noRef, element):
        with self.timer('hierarchy - meshes'):
            params = {}
            params['domain'] = mesh
            params['solver'] = solverType
            params['tag'] = domainIndicator+fluxIndicator
            params['element'] = element
            params['keepMeshes'] = 'all'
            params['keepAllDoFMaps'] = True
            params['buildMass'] = True
            params['assemble'] = 'restrictionProlongation' if solverType.find('mg') >= 0 else 'dofmap only last'
            params['logging'] = True
            if self.debugAssemblyTimes:
                from PyNucleus_base.utilsFem import TimerManager
                tm = TimerManager(self.driver.logger, comm=self.driver.comm, memoryProfiling=False, loggingSubTimers=True)
                params['PLogger'] = tm.PLogger

            comm = self.driver.comm
            onRanks = [0]
            if comm is not None and comm.size > 1:
                onRanks = range(comm.size)
            hierarchies, connectors = paramsForFractionalHierarchy(noRef, params, onRanks)
            hM = hierarchyManager(hierarchies, connectors, params, comm)
            hM.setup()
        self.meshHierarchy = hM
        self.finalMesh = hM['fine'].meshLevels[-1].mesh

        self.dmInterior = hM['fine'].algebraicLevels[-1].DoFMap
        self.dmBC = self.dmInterior.getComplementDoFMap()
        self.dm, self.R_interior, self.R_bc = self.dmInterior.getFullDoFMap(self.dmBC)
        self.P_interior = self.R_interior.transpose()
        self.P_bc = self.R_bc.transpose()

    @generates('hierarchy')
    def buildHierarchy(self,
                       meshHierarchy,
                       dm, dmBC, dmInterior,
                       kernel, sArgs, rangedKernel, solverType, matrixFormat, tag, boundaryCondition, domainIndicator, fluxIndicator,
                       zeroExterior, noRef, eta, target_order, element, quadType, quadTypeBoundary):
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

                # for prop in ['bc', 'finalMesh', 'dm', 'dmBC', 'dmInterior', 'R_interior', 'P_interior', 'R_bc', 'P_bc']:
                #     setattr(self, prop, self.directlyGetWithoutChecks(prop))
                return

        hM = meshHierarchy

        assemblyParams = {}
        if quadType == 'auto':
            quadType = 'classical-refactored'
        assemblyParams['quadType'] = quadType

        if quadTypeBoundary == 'auto':
            quadTypeBoundary = 'classical-refactored'
        assemblyParams['quadTypeBoundary'] = quadTypeBoundary
        if rangedKernel is None:
            assemblyParams['kernel'] = kernel
        else:
            assemblyParams['kernel'] = rangedKernel
        assemblyParams['boundaryCondition'] = boundaryCondition
        assemblyParams['zeroExterior'] = zeroExterior
        assemblyParams['target_order'] = target_order
        assemblyParams['eta'] = eta
        assemblyParams['dense'] = matrixFormat == 'dense'
        assemblyParams['matrixFormat'] = matrixFormat

        with self.timer('hierarchy - matrices'):
            from PyNucleus_multilevelSolver.levels import ASSEMBLY
            if solverType.find('mg') >= 0:
                for subHierarchy in hM.builtHierarchies:
                    for level in subHierarchy.algebraicLevels:
                        level.Timer = self.timer
                        level.PLogger = level.Timer.PLogger
                        assemblyParams['PLogger'] = level.PLogger
                        level.params.update(assemblyParams)
                        level.build(ASSEMBLY)
            else:
                level = hM.builtHierarchies[-1].algebraicLevels[-1]
                level.Timer = self.timer
                level.PLogger = level.Timer.PLogger
                assemblyParams['PLogger'] = level.PLogger
                level.params.update(assemblyParams)
                level.build(ASSEMBLY)

        hierarchy = hM.getLevelList()

        if rangedKernel is not None:
            hierarchy[0]['sArgs'] = sArgs
            s = kernel.sValue
            for lvl in range(len(hierarchy)):
                if 'A' in hierarchy[lvl]:
                    hierarchy[lvl]['A'].set(s, 0)

        self.hierarchy = hierarchy
        if kernel is not None:
            assert 2*self.finalMesh.h < kernel.max_horizon, ("Please choose horizon bigger than two mesh sizes. " +
                                                             "h = {}, horizon = {}").format(self.finalMesh.h, kernel.horizon.value)

    @generates('adjointHierarchy')
    def buildAdjointHierarchy(self, hierarchy):
        adjointHierarchy = []
        for lvl in range(len(hierarchy)):
            adjointHierarchy.append({})
            for label in hierarchy[lvl]:
                if label in ('A', 'S'):
                    adjointHierarchy[lvl][label] = hierarchy[lvl][label].T
                else:
                    adjointHierarchy[lvl][label] = hierarchy[lvl][label]
        self.adjointHierarchy = adjointHierarchy

    @generates('A_BC')
    def buildBCoperator(self, dmInterior, dmBC,
                        kernel, sArgs, rangedKernel, solverType, matrixFormat, tag, boundaryCondition,
                        zeroExterior, noRef, eta, target_order, element, quadType, quadTypeBoundary):
        if boundaryCondition == DIRICHLET:
            from . helpers import getFracLapl
            assemblyParams = {}
            if quadType == 'auto':
                quadType = 'classical-refactored'
            assemblyParams['quadType'] = quadType

            if quadTypeBoundary == 'auto':
                quadTypeBoundary = 'classical-refactored'
            assemblyParams['quadTypeBoundary'] = quadTypeBoundary
            if rangedKernel is None:
                assemblyParams['kernel'] = kernel
            else:
                assemblyParams['kernel'] = rangedKernel
            assemblyParams['boundaryCondition'] = boundaryCondition
            assemblyParams['zeroExterior'] = zeroExterior
            assemblyParams['target_order'] = target_order
            assemblyParams['eta'] = eta
            assemblyParams['dense'] = matrixFormat == 'dense'
            assemblyParams['matrixFormat'] = matrixFormat
            assemblyParams['tag'] = tag
            with self.timer('build BC operator'):
                self.A_BC = getFracLapl(dmInterior, dm2=dmBC, **assemblyParams)
        else:
            self.A_BC = None

    @generates('mass')
    def buildMass(self, dm):
        self.mass = dm.assembleMass()

    @generates('massInterior')
    def buildMassInterior(self, dmInterior):
        self.massInterior = dmInterior.assembleMass()

    @generates('A')
    def getOperators(self, hierarchy):
        self.A = hierarchy[-1]['A']

    @generates('A_derivative')
    def getDerivativeOperator(self, kernel, dmInterior, matrixFormat, eta, target_order):
        self.A_derivative = dmInterior.assembleNonlocal(kernel.getDerivativeKernel(derivative=1),
                                                        matrixFormat=matrixFormat, params={'eta': eta,
                                                                                           'target_order': target_order})

    @generates('b')
    def buildRHS(self, rhs, dim, A_BC, dmBC, dirichletData, boundaryCondition, solverType, dmInterior, hierarchy):
        self.b = dmInterior.assembleRHS(rhs, qr=simplexXiaoGimbutas(3, dim))

        if A_BC is not None:
            assert dmInterior.num_dofs == A_BC.num_rows
            assert dmBC.num_dofs == A_BC.num_columns
            if dmBC.num_dofs > 0:
                self.b -= A_BC*dmBC.interpolate(dirichletData)

        # pure Neumann condition -> project out nullspace
        if boundaryCondition in (NEUMANN, HOMOGENEOUS_NEUMANN):
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
            if kernel is not None:
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

    @generates('adjointSolver')
    def buildAdjointSolver(self, solverType, tol, maxiter, adjointHierarchy, kernel):
        from PyNucleus_base.solvers import iterative_solver
        if solverType[:2] == 'cg':
            if kernel is not None:
                assert kernel.symmetric, 'CG solver requires a symmetric matrix'
        solver = solverFactory.build(solverType, hierarchy=adjointHierarchy)
        if isinstance(solver, iterative_solver):
            solver.tolerance = tol
            solver.maxIter = maxiter
            comm = self.driver.comm
            if comm is not None and comm.size > 1:
                solver.setNormInner(norm_distributed_nonoverlapping(comm),
                                    ip_distributed_nonoverlapping(comm))
        solver.setup()
        self.adjointSolver = solver

    @generates('modelSolution')
    def solve(self, b, dm, dmInterior, dmBC, P_interior, P_bc, R_interior, solver, boundaryCondition, analyticSolution, dirichletData, tol, maxiter, rhs):
        uInterior = dmInterior.zeros()
        with self.timer('solve {}'.format(self.__class__.__name__)):
            its = solver(b, uInterior)

        resError = (b-solver.A*uInterior).norm(False)

        if isinstance(solver, iterative_solver):
            if its >= maxiter-1:
                self.driver.logger.warn("WARNING: Only reached residual error {} > tol = {} in {} iterations".format(resError, tol, its))

        # pure Neumann condition -> add nullspace components to match analytic solution
        if boundaryCondition in (NEUMANN, HOMOGENEOUS_NEUMANN) and analyticSolution is not None:
            uEx = dmInterior.interpolate(analyticSolution)
            const = dmInterior.ones()
            uInterior += (const.inner(uEx)-const.inner(uInterior))/const.inner(const) * const

        u = dm.empty()
        if boundaryCondition in (DIRICHLET, ):
            u.assign(P_interior*uInterior+P_bc*dmBC.interpolate(dirichletData))
        else:
            u.assign(P_interior*uInterior)

        data = {'iterations': its,
                'uInterior': uInterior,
                'uRestricted': uInterior.dm.fromArray(R_interior*u),
                'explicitResidualError': resError,
                'b': b,
                'rhs': rhs}
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

    @generates('adjointModelSolution')
    def adjointSolve(self, b, dm, dmInterior, P_interior, adjointSolver, tol, maxiter):
        uInterior = dmInterior.zeros()
        with self.timer('solve adjoint {}'.format(self.__class__.__name__)):
            its = adjointSolver(b, uInterior)

        resError = (b-adjointSolver.A*uInterior).norm(False)

        if isinstance(adjointSolver, iterative_solver):
            if its >= maxiter-1:
                self.driver.logger.warn("WARNING: Only reached residual error {} > tol = {} in {} iterations".format(resError, tol, its))

        u = dm.fromArray(P_interior*uInterior)

        data = {'iterations': its,
                'uInterior': uInterior,
                'explicitResidualError': resError,
                'b': b}
        if isinstance(adjointSolver, iterative_solver):
            data['tol'] = adjointSolver.tolerance
            data['maxIterations'] = adjointSolver.maxIter
            data['residuals'] = copy(adjointSolver.residuals)
            data['preconditionedResidualError'] = adjointSolver.residuals[-1]
        data['analyticSolution'] = None
        data['exactL2Squared'] = None
        data['exactHsSquared'] = None
        data['dirichletData'] = None
        self.adjointModelSolution = stationaryModelSolution(self, u, **data)

    def report(self, group):
        group.add('kernel', repr(self.continuumProblem.kernel))
        group.add('kernel expression', self.continuumProblem.kernel.getLongDescription())
        group.add('problem', self.continuumProblem.problemDescription)
        group.add('has analytic solution', self.continuumProblem.analyticSolution is not None)
        group.add('h', self.finalMesh.h)
        group.add('hmin', self.finalMesh.hmin)
        if self.continuumProblem.kernel is not None:
            group.add('horizon', self.continuumProblem.kernel.horizonValue)
        else:
            group.add('horizon', 0.0)
        group.add('mesh quality', self.finalMesh.delta)
        group.add('DoFMap', str(self.dm))
        group.add('Interior DoFMap', str(self.dmInterior))
        group.add('Dirichlet DoFMap', str(self.dmBC))
        group.add('matrix', str(self.A))
        if isinstance(self.A, (H2Matrix,
                               DistributedH2Matrix_globalData,
                               DistributedH2Matrix_localData)):
            for label, key in [('near field matrix', 'Anear'),
                               ('min cluster size', 'minSize'),
                               ('interpolation order', 'interpolation_order'),
                               ('numAssembledCellPairs', 'numAssembledCellPairs'),
                               ('numIntegrations', 'numIntegrations')]:
                group.add(label, self.hierarchy[-1]['Timer'].PLogger[key][-1])
        if isinstance(self.A, (Dense_LinearOperator,
                               H2Matrix,
                               DistributedH2Matrix_globalData,
                               DistributedH2Matrix_localData)):
            for label, key in [('useSymmetricCells', 'useSymmetricCells'),
                               ('useSymmetricLocalMatrix', 'useSymmetricLocalMatrix')]:
                group.add(label, self.hierarchy[-1]['Timer'].PLogger[key][0])
        group.add('matrix memory size', self.A.getMemorySize())


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

    def residual(self, t, u, ut, residual, coeff_A=1., coeff_B=1., coeff_g=1., coeff_residual=0., forcingVector=None, adjoint=False):
        if coeff_residual != 1.:
            residual *= coeff_residual

        if adjoint:
            coeff_A = -coeff_A

        if coeff_A != 0:
            temp = residual.copy()
            self.massInterior(ut, temp)
            temp *= coeff_A
            residual += temp

        if coeff_B != 0.:
            temp = residual.copy()
            self.A(u, temp, trans=adjoint)
            temp *= coeff_B
            residual += temp

        if coeff_g != 0.:
            temp = residual.copy()
            if forcingVector is None:
                force = self.rhs(t)
                if isinstance(force, constant) and force.value == 0:
                    temp.assign(0.)
                else:
                    temp.assign(self.dmInterior.assembleRHS(force, qr=self.qr))
                if self.dirichletData is not None:
                    uBC = self.dmBC.interpolate(self.dirichletData(t))
                    temp -= self.A_BC*uBC
            else:
                temp.assign(forcingVector)
            temp *= -coeff_g
            residual += temp

    def residual_adjoint(self, t, u, ut, residual, coeff_A=1., coeff_B=1., coeff_g=1., coeff_residual=0., forcingVector=None):
        self.residual(t, u, ut, residual, coeff_A, coeff_B, coeff_g, coeff_residual, forcingVector, adjoint=True)

    def solverBuilder(self, t, alpha, beta):
        with self.timer('build solver {}'.format(self.__class__.__name__)):
            return self.buildTransientSolver(self.solverType, self.tol, self.maxiter, self.hierarchy, alpha, beta)

    def solverBuilder_adjoint(self, t, alpha, beta):
        with self.timer('build adjoint solver {}'.format(self.__class__.__name__)):
            return self.buildTransientSolver(self.solverType, self.tol, self.maxiter, self.hierarchy, -alpha, beta)

    @generates('stepper')
    def buildTimeStepper(self, timeStepperType, dt, dmInterior, theta, hierarchy):
        kwargs = {}
        if timeStepperType == 'Crank-Nicolson':
            kwargs['theta'] = theta
        self.stepper = timestepperFactory(timeStepperType,
                                          dm=dmInterior,
                                          residual=self.residual,
                                          solverBuilder=self.solverBuilder,
                                          dt=dt,
                                          explicitIslinearAndTimeIndependent=True,
                                          **kwargs)

    @generates('b')
    def buildRHS(self, dmInterior, numTimeSteps, timesVector, stepper):
        self.b = dmInterior.zeros(numTimeSteps)
        for k in range(numTimeSteps):
            t = timesVector[k]
            stepper.setRHS(t, stepper.dt, self.b[k])

    @generates('adjointStepper')
    def buildAdjointTimeStepper(self, timeStepperType, dt, dmInterior, theta, hierarchy):
        kwargs = {}
        if timeStepperType == 'Crank-Nicolson':
            kwargs['theta'] = theta
        self.stepper = timestepperFactory(timeStepperType,
                                          dm=dmInterior,
                                          residual=self.residual_adjoint,
                                          solverBuilder=self.solverBuilder_adjoint,
                                          dt=dt,
                                          explicitIslinearAndTimeIndependent=True,
                                          **kwargs)

    @generates(['initialSolution'])
    def setInitialCondition(self, dm, initial):
        if self.keepAllTimeSteps:
            self.initialSolution = dm.interpolate(initial)
        else:
            assert self.keepAllTimeSteps
            self.u = dm.interpolate(initial)

        if self.doMovie:
            if self._driver.isMaster:
                from PyNucleus_base.plot_utils import movieCreator

                outputFolder = self.movieFolder
                movie_kwargs = {}
                if self.continuumProblem.dim == 2:
                    movie_kwargs['vmin'] = self.initialSolution.min()
                    movie_kwargs['vmax'] = self.initialSolution.max()
                    movie_kwargs['shading'] = self.shading
                self.mC = movieCreator(self.initialSolution, outputFolder)

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


