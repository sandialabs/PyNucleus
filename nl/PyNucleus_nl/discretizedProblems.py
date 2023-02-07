###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from PyNucleus_base import solverFactory
from PyNucleus_base.utilsFem import (classWithComputedDependencies,
                                     INVALID,
                                     problem,
                                     generates)
from PyNucleus_multilevelSolver import hierarchyManager
from PyNucleus_base.solvers import iterative_solver
from PyNucleus_base.linear_operators import Dense_LinearOperator
from PyNucleus_fem import simplexXiaoGimbutas
from PyNucleus_fem.DoFMaps import Product_DoFMap
from copy import copy
from . helpers import (multilevelDirichletCondition,
                       paramsForFractionalHierarchy)
from . kernelsCy import FRACTIONAL
from . nonlocalProblems import (DIRICHLET, HOMOGENEOUS_DIRICHLET,
                                NEUMANN, HOMOGENEOUS_NEUMANN)


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
    def computeL2error(self, u, u_interp, analyticSolution, exactL2squared):
        if exactL2squared is not None:
            if u.dm == self.discretizedProblem.dmInterior:
                M = self.discretizedProblem.massInterior
            elif u.dm == self.discretizedProblem.dm:
                M = self.discretizedProblem.mass
            else:
                M = u.dm.assembleMass()
            z = u.dm.assembleRHS(analyticSolution)
            self.L2_error = np.sqrt(abs(exactL2squared - 2*z.inner(u) + u.inner(M*u)))
        else:
            self.L2_error = None

    @generates('L2_error_interp')
    def computeL2errorInterpolated(self, u, u_interp):
        if u_interp is not None:
            if u.dm == self.discretizedProblem.dmInterior:
                M = self.discretizedProblem.massInterior
            elif u.dm == self.discretizedProblem.dm:
                M = self.discretizedProblem.mass
            else:
                M = u.dm.assembleMass()
            self.L2_error_interp = np.sqrt((u-u_interp).inner(M*(u-u_interp)))
        else:
            self.L2_error_interp = None

    @generates('error')
    def buildErrorVector(self, u, u_interp):
        if self.u_interp is not None:
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
        from PyNucleus.fem import plotManager
        pm = plotManager(self.u.dm.scalarDM.mesh, self.u.dm.scalarDM,
                         defaults=plotDefaults)
        for c in range(self.u.dm.numComponents):
            pm.add(self.u.getComponent(c), label='u'+str(c))
        pm.plot()

    def reportErrors(self, group):
        group.add('L2 error', self.L2_error, rTol=3e-2)
        group.add('L2 error interpolated', self.L2_error_interp, rTol=3e-2)

    def reportSolve(self, group):
        group.add('solver', self.discretizedProblem.solverType)
        if isinstance(self.discretizedProblem.solver, iterative_solver):
            group.add('iterations', self.iterations)
            group.add('implicit residual norm', self.residuals[-1])
            group.add('explicit residual norm', self.explicitResidualError)
            group.add('tolerance', self.tol)


class discretizedNonlocalProblem(problem):
    def __init__(self, driver, continuumProblem):
        super().__init__(driver)
        self.continuumProblem = continuumProblem
        self.addRemote(self.continuumProblem)

    def setDriverArgs(self):
        self.setDriverFlag('solverType', acceptedValues=['lu', 'mg', 'cg-mg', 'cg-jacobi'], help='solver for the linear system')
        self.setDriverFlag('maxiter', 100, help='maximum number of iterations')
        self.setDriverFlag('tol', 1e-6, help='solver tolerance')
        self.setDriverFlag('genKernel', False)
        self.setDriverFlag('dense', False, help='assemble into dense matrix format')

    def processCmdline(self, params):
        dense = params['dense']
        kernelType = params['kernelType']
        if kernelType != 'fractional':
            dense = True
            params['dense'] = dense
        super().processCmdline(params)

    @generates(['hierarchy', 'bc', 'finalMesh', 'dm', 'dmBC', 'dmInterior', 'R_interior', 'P_interior', 'R_BC', 'P_BC'])
    def buildHierarchy(self, mesh, kernel, rangedKernel, solverType, genKernel, dense, tag, boundaryCondition, domainIndicator, fluxIndicator, zeroExterior, noRef, eta, target_order, element):
        assert dense or (kernel.kernelType == FRACTIONAL), 'Hierarchical matrices are only implemented for fractional kernels'
        if rangedKernel is not None:
            hierarchy = self.directlyGetWithoutChecks('hierarchy')
            if hierarchy is not None:
                s = kernel.sValue
                for lvl in range(len(hierarchy)):
                    if 'A' in hierarchy[lvl]:
                        hierarchy[lvl]['A'].set(s)
                self.directlySetWithoutChecks('hierarchy', hierarchy)
                # self.setState('solver', INVALID)

                for prop in ['dm', 'dmBC', 'dmInterior', 'R_BC', 'P_BC', 'R_interior', 'P_interior', 'bc', 'finalMesh']:
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
            params['assemble'] = 'ALL' if solverType.find('mg') >= 0 else 'last'
            params['dense'] = dense
            params['logging'] = True
            params['genKernel'] = genKernel
            hierarchies, connectors = paramsForFractionalHierarchy(noRef, params)
            hM = hierarchyManager(hierarchies, connectors, params)
            hM.setup()

        if not boundaryCondition == HOMOGENEOUS_DIRICHLET:
            bc = multilevelDirichletCondition(hM.getLevelList(), domainIndicator, fluxIndicator)
            hierarchy = bc.naturalLevels
            dmInterior = bc.naturalDoFMap
            dmBC = bc.dirichletDoFMap
        else:
            hierarchy = hM.getLevelList()
            bc = None
            dmInterior = hierarchy[-1]['DoFMap']
            dmBC = dmInterior.getComplementDoFMap()
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
    def buildRHS(self, rhs, dim, bc, dirichletData, boundaryCondition, solverType, dmInterior):
        self.b = dmInterior.assembleRHS(rhs, qr=simplexXiaoGimbutas(3, dim))
        if bc is not None:
            bc.setDirichletData(dirichletData)
            bc.applyRHScorrection(self.b)

        # pure Neumann condition -> project out nullspace
        if boundaryCondition in (NEUMANN, HOMOGENEOUS_NEUMANN):
            assert bc.dirichletDoFMap.num_dofs == 0, bc.dirichletDoFMap
            if solverType.find('mg') >= 0:
                bc.naturalLevels[0]['A'] = bc.naturalLevels[0]['A'] + Dense_LinearOperator.ones(*bc.naturalLevels[0]['A'].shape)
            const = dmInterior.ones()
            self.b -= self.b.inner(const)/const.inner(const)*const

    @generates('solver')
    def buildSolver(self, solverType, tol, maxiter, hierarchy):
        if solverType.find('mg') >= 0:
            ml = solverFactory.build('mg', hierarchy=hierarchy, setup=True, tolerance=tol, maxIter=maxiter)
        if solverType == 'mg':
            solver = ml
        elif solverType == 'cg-mg':
            solver = solverFactory.build('cg', A=hierarchy[-1]['A'], setup=True, tolerance=tol, maxIter=maxiter)
            solver.setPreconditioner(ml.asPreconditioner())
        else:
            solver = solverFactory.build(solverType, A=hierarchy[-1]['A'], setup=True)
        self.solver = solver

    @generates('modelSolution')
    def solve(self, b, bc, dm, dmInterior, P_interior, solver, boundaryCondition, analyticSolution, dirichletData):
        uInterior = dmInterior.zeros()
        its = solver(b, uInterior)

        resError = (b-solver.A*uInterior).norm(False)

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
                'explicitResidualError': resError}
        if isinstance(solver, iterative_solver):
            data['tol'] = solver.tolerance
            data['maxIterations'] = solver.maxIter
            data['residuals'] = copy(solver.residuals)
        data['analyticSolution'] = analyticSolution
        data['exactL2squared'] = None
        data['dirichletData'] = dirichletData
        self.modelSolution = stationaryModelSolution(self, u, **data)

    def report(self, group):
        group.add('h', self.finalMesh.h)
        group.add('hmin', self.finalMesh.hmin)
        group.add('mesh quality', self.finalMesh.delta)
        group.add('DoFMap', str(self.dm))
        group.add('Interior DoFMap', str(self.dmInterior))
        group.add('Dirichlet DoFMap', str(self.dmBC))
