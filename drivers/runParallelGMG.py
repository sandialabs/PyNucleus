#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from __future__ import division, print_function
from mpi4py import MPI
import numpy as np
from PyNucleus.base import driver, solverFactory
from PyNucleus.fem import diffusionProblem
from PyNucleus.fem.mesh import accumulate2global
from PyNucleus.multilevelSolver import (V, FMG_V,
                                        hierarchyManager,
                                        inputConnector,
                                        
                                        paramsForMG)

d = driver(MPI.COMM_WORLD)
p = diffusionProblem(d)

d.add('checkSolution', False)
d.add('saveVTK', False)

d.add('doMG', True)
d.add('doFMG', True)
d.add('doCG', False)
d.add('doPCG', True)
d.add('doFMGPCG', True)
d.add('doBICGSTAB', False)
d.add('doPBICGSTAB', True)
d.add('doGMRES', False)
d.add('doPGMRES', True)
d.add('doFMGPGMRES', True)
d.add('commType', 'standard', acceptedValues=['oneSided', 'persistent'])

d.add('partitioner', 'regular')
d.add('partitionerParams', {})
d.add('debugOverlaps', False)

solver = d.addGroup('solver')
solver.add('smoother', 'jacobi', acceptedValues=['gauss_seidel', 'chebyshev'])
solver.add('doPCoarsen', False)
solver.add('maxiter', 50)

d.declareFigure('residuals', default=False)
d.declareFigure('spSolve')
d.declareFigure('spSolveError')
d.declareFigure('spSolveExactSolution')

params = d.process()

with d.timer('setup levels'):
    if d.doPCoarsen and d.element != 'P1':
         raise NotImplementedError()
    else:
        hierarchies, connectors = paramsForMG(p.noRef,
                                              range(d.comm.size),
                                              params, p.dim, p.element)
        connectors['input'] = {'type': inputConnector,
                               'params': {'domain': d.domain}}

    FINE = 'fine'
    hierarchies[-1]['label'] = FINE

    hM = hierarchyManager(hierarchies, connectors, params, d.comm)
    hM.setup()
    hM.display()

    subdomain = hM[FINE].meshLevels[-1].mesh
    DoFMap_fine = hM[FINE].algebraicLevels[-1].DoFMap
    overlaps = hM[FINE].multilevelAlgebraicOverlapManager
    h = hM[FINE].meshLevels[-1].mesh.global_h(overlaps.comm)
    hmin = hM[FINE].meshLevels[-1].mesh.global_hmin(overlaps.comm)
    tol = {'P1': 0.5*h**2,
           'P2': 0.001*h**3,
           'P3': 0.001*h**4}[d.element]
    tol = max(tol, 2e-9)

# assemble rhs on finest grid
with d.timer('Assemble rhs on finest grid'):
    rhs = DoFMap_fine.assembleRHS(p.rhsFun)
if p.boundaryCond:
    with d.timer('BC'):
        boundaryDoFMap = DoFMap_fine.getComplementDoFMap()
        boundary_data = boundaryDoFMap.interpolate(p.boundaryCond)
        A_boundary = DoFMap_fine.assembleStiffness(dm2=boundaryDoFMap)
        rhs -= A_boundary*boundary_data

with d.timer('Setup solver'):
    smootherParams = {'jacobi': {'presmoothingSteps': 2,
                                 'postsmoothingSteps': 2},
                      'gauss_seidel': {'presmoothingSteps': 1,
                                       'postsmoothingSteps': 1},
                      'chebyshev': {'degree': 3}}
    ml = solverFactory.build('mg',
                             hierarchy=hM,
                             smoother=(d.smoother, smootherParams[d.smoother]),
                             maxIter=d.maxiter, tolerance=tol,
                             setup=True)
info = d.addOutputGroup('info')
info.add('Subdomains', d.comm.size)
info.add('Refinement steps', p.noRef)
info.add('Elements', d.comm.allreduce(subdomain.num_cells))
info.add('DoFs', overlaps.countDoFs())
info.add('h', h)
info.add('hmin', hmin)
info.add('Tolerance', tol)
d.logger.info('\n' + str(info) + '\n')
d.logger.info('\n'+str(ml))
d.comm.Barrier()

x = DoFMap_fine.zeros()
r = DoFMap_fine.zeros()
A = hM[FINE].algebraicLevels[-1].A
acc = hM[FINE].algebraicLevels[-1].accumulateOperator
A.residual_py(x, rhs, r)
r0 = r.norm(False)


rate = d.addOutputGroup('rates', tested=True, aTol=1e-2)
its = d.addOutputGroup('iterations', tested=True)
res = d.addOutputGroup('residuals', tested=True, rTol=4e-1)
resHist = d.addOutputGroup('resHist', tested=True, aTol=5e-8)
errs = d.addOutputGroup('errors', tested=True, rTol=2.)

for cycle, label in [(V, 'MG'),
                     (FMG_V, 'FMG')]:
    if getattr(d, 'do'+label):
        ml.cycle = cycle
        with d.timer('Solve '+label):
            numIter = ml(rhs, x)
        residuals = ml.residuals
        A.residual_py(x, rhs, r)
        resNorm = r.norm(False)
        rate.add('Rate of convergence '+label, (resNorm/r0)**(1/numIter))
        its.add('Number of iterations '+label, numIter)
        res.add('Residual norm '+label, resNorm)
        resHist.add(label, residuals)

# set up cg
cg = solverFactory.build('cg', A=A, maxIter=d.maxiter, tolerance=tol, setup=True)
cg.setNormInner(ml.norm, ml.inner)
# set up gmres
gmres = solverFactory.build('gmres', A=A, maxIter=d.maxiter//5, restarts=5, tolerance=tol, setup=True)
gmres.setNormInner(ml.norm, ml.inner)
# set up bicgstab
bicgstab = solverFactory.build('bicgstab', A=A, maxIter=d.maxiter, tolerance=tol, setup=True)
bicgstab.setNormInner(ml.norm, ml.inner)

for solver, label in [
        (cg, 'CG'),
        (gmres, 'GMRES'),
        (bicgstab, 'BICGSTAB')]:
    if getattr(d, 'do'+label):
        solver.setPreconditioner(acc)
        solver.setInitialGuess()
        with d.timer('Solve '+label):
            numIter = solver(rhs, x)
        residuals = solver.residuals
        A.residual_py(x, rhs, r)
        resNorm = r.norm(False)
        rate.add('Rate of convergence '+label, (resNorm/r0)**(1/numIter))
        its.add('Number of iterations '+label, numIter)
        res.add('Residual norm '+label, resNorm)
        resHist.add(label, residuals)
    if getattr(d, 'doP'+label):
        solver.setPreconditioner(ml.asPreconditioner(cycle=V), False)
        solver.setInitialGuess()
        with d.timer('Solve P'+label):
            numIter = solver(rhs, x)
        residuals = solver.residuals
        A.residual_py(x, rhs, r)
        resNorm = r.norm(False)
        rate.add('Rate of convergence P'+label, (resNorm/r0)**(1/numIter), tested=False if label == 'BICGSTAB' else None)
        its.add('Number of iterations P'+label, numIter, aTol=1 if label == 'BICGSTAB' else None)
        res.add('Residual norm P'+label, resNorm)
        resHist.add('P'+label, residuals, tested=False if label == 'BICGSTAB' else None)


if d.saveVTK and p.boundaryCond:
    y = DoFMap_fine.augmentWithBoundaryData(x,
                                            boundary_data)
    subdomain.exportSolutionVTK(y, y.dm, '{}{}.vtk'.format(d.problem, d.comm.rank),
                                rank=d.comm.rank)

if d.doFMGPCG:
    ml.cycle = FMG_V
    ml.maxIter = 1
    cg.setPreconditioner(ml.asPreconditioner(cycle=V))
    with d.timer('Solve FMG-PCG'):
        ml(rhs, x)
        cg.setInitialGuess(x)
        numIter = cg(rhs, x)
    residuals = cg.residuals
    numIter += 1
    A.residual_py(x, rhs, r)
    resNorm = r.norm(False)
    rate.add('Rate of convergence FMG-PCG', (resNorm/r0)**(1/numIter))
    its.add('Number of iterations FMG-PCG', numIter)
    res.add('Residual norm FMG-PCG', resNorm)
    resHist.add('FMG-PCG', residuals)

if d.doFMGPGMRES:
    ml.cycle = FMG_V
    ml.maxIter = 1
    gmres.setPreconditioner(ml.asPreconditioner(cycle=V), False)
    with d.timer('Solve FMG-PGMRES'):
        ml(rhs, x)
        gmres.setInitialGuess(x)
        numIter = gmres(rhs, x)
    residuals = gmres.residuals
    numIter += 1
    A.residual_py(x, rhs, r)
    resNorm = r.norm(False)
    rate.add('Rate of convergence FMG-PGMRES', (resNorm/r0)**(1/numIter))
    its.add('Number of iterations FMG-PGMRES', numIter)
    res.add('Residual norm FMG-PGMRES', resNorm)
    resHist.add('FMG-PGMRES', residuals)

d.comm.Barrier()



if p.L2ex:
    if p.boundaryCond:
        d.logger.warning('L2 error is wrong for inhomogeneous BCs')
    with d.timer('Mass matrix'):
        M = DoFMap_fine.assembleMass(sss_format=d.symmetric)
    z = DoFMap_fine.assembleRHS(p.exactSolution)
    L2err = np.sqrt(np.absolute(x.inner(M*x, True, False) -
                                2*z.inner(x, False, True) +
                                p.L2ex))
    del z
    errs.add('L^2 error', L2err)
if p.H10ex:
    if p.boundaryCond:
        d.logger.warning('H^1_0 error is wrong for inhomogeneous BCs')
    H10err = np.sqrt(np.absolute(p.H10ex - rhs.inner(x, False, True)))
    errs.add('H^1_0 error', H10err)
d.logger.info('\n'+str(rate+its+res+errs))

if d.startPlot('residuals'):
    import matplotlib.pyplot as plt
    if d.doMG:
        plt.plot(resHist.MG, '-*', label='MG')
    if d.doFMG:
        plt.plot(resHist.FMG, '-.', label='FMG')
    if d.doCG:
        plt.plot(resHist.CG, '--', label='CG')
    if d.doPCG:
        plt.plot(resHist.PCG, '-*', label='MG-PCG')
    if d.doFMGPCG:
        plt.plot(resHist.FMGPCG, '-*', label='FMG-PCG')
    if d.doGMRES:
        plt.plot(resHist.GMRES, '.', label='GMRES')
    if d.doPGMRES:
        plt.plot(resHist.PGMRES, '-.', label='MG-GMRES')
    if d.doFMGPGMRES:
        plt.plot(resHist.FMGPGMRES, '-*', label='FMG-PGMRES')
    plt.yscale('log')
    plt.legend()

if d.checkSolution:
    interfaces = hM[FINE].meshLevels[-1].interfaces
    (global_mesh,
     global_solution,
     global_dm) = accumulate2global(subdomain, interfaces, DoFMap_fine, x,
                                    comm=d.comm)
    if d.isMaster:
        from scipy.sparse.linalg import spsolve
        from numpy.linalg import norm
        A = global_dm.assembleStiffness()
        rhs = global_dm.assembleRHS(p.rhsFun)
        if p.boundaryCond:
            global_boundaryDoFMap = global_dm.getComplementDoFMap()
            global_boundary_data = global_boundaryDoFMap.interpolate(p.boundaryCond)
            global_A_boundary = global_dm.assembleStiffness(dm2=global_boundaryDoFMap)
            rhs -= global_A_boundary*global_boundary_data
        with d.timer('SpSolver'):
            y = spsolve(A.to_csr(), rhs)
        if p.boundaryCond:
            sol_augmented, dm_augmented = global_dm.augmentWithBoundaryData(global_solution, global_boundary_data)
            global_mass = dm_augmented.assembleMass()
            global_z = dm_augmented.assembleRHS(p.exactSolution)
            L2err = np.sqrt(np.absolute(np.vdot(sol_augmented, global_mass*sol_augmented) -
                                        2*np.vdot(global_z, sol_augmented) +
                                        p.L2ex))
        else:
            global_mass = global_dm.assembleMass()
            global_z = global_dm.assembleRHS(p.exactSolution)
            L2err = np.sqrt(np.absolute(np.vdot(global_solution, global_mass*global_solution) -
                                        2*np.vdot(global_z, global_solution) +
                                        p.L2ex))
        errsSpSolve = d.addOutputGroup('errSpSolve')
        errsSpSolve.add('L2', L2err)
        errsSpSolve.add('2-norm', norm(global_solution-y, 2))
        errsSpSolve.add('max-norm', np.abs(global_solution-y).max())
        d.logger.info('\n'+str(errsSpSolve))
        if d.startPlot('spSolve'):
            import matplotlib.pyplot as plt
            global_solution.plot()
        if p.exactSolution and d.startPlot('spSolveExactSolution'):
            global_dm.interpolate(p.exactSolution).plot()
        if d.startPlot('spSolveError'):
            (global_solution-y).plot()
d.finish()
