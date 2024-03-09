#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from __future__ import division
import mpi4py.MPI as MPI
import numpy as np
from PyNucleus_base import driver, solverFactory
from PyNucleus_fem import diffusionProblem
from PyNucleus_multilevelSolver import (V, FMG_V,
                                        hierarchyManager,
                                        inputConnector,
                                        paramsForSerialMG)

comm = MPI.COMM_WORLD
assert comm.size == 1

d = driver()
p = diffusionProblem(d)

solver = d.addGroup('solver')
solver.add('amg', False)
solver.add('lu', False)
solver.add('chol', False)
solver.add('smoother', 'jacobi', acceptedValues=['gauss_seidel', 'sor', 'chebyshev', 'ilu'])
solver.add('maxiter', 50)

d.declareFigure('residuals', default=False)
d.declareFigure('solution', default=False)

params = d.process()

info = d.addOutputGroup('info')

with d.timer('setup levels'):
    if not params['amg']:
        hierarchies, connectors = paramsForSerialMG(p.noRef, params)
        connectors['input'] = {'type': inputConnector,
                               'params': {'domain': d.domain}}
        FINE = 'fine'
        hierarchies[-1]['label'] = FINE

        hM = hierarchyManager(hierarchies, connectors, params, comm)
        hM.setup()
        hM.display()

        levels = hM

        mesh = hM[FINE].meshLevels[-1].mesh
        DoFMap = hM[FINE].algebraicLevels[-1].DoFMap
    else:
        raise NotImplementedError()

if d.element == 'P1':
    tol = 0.5*mesh.h**2
elif d.element == 'P2':
    tol = 0.001*mesh.h**3
elif d.element == 'P3':
    tol = 0.001*mesh.h**4

with d.timer('RHS'):
    rhs = DoFMap.assembleRHS(p.rhsFun)
if p.boundaryCond:
    with d.timer('BC'):
        boundary_data = DoFMap.getBoundaryData(p.boundaryCond)
        levels[-1]['A'] = DoFMap.assembleStiffness(boundary_data, rhs,
                                                   sss_format=True,
                                                   reorder=d.reorder)

info.add('DoFs', rhs.shape[0])
info.add('element', params['element'])
info.add('Tol', tol)
d.logger.info('\n'+str(info))

smootherParams = {'jacobi': {'presmoothingSteps': 2,
                             'postsmoothingSteps': 2},
                  'gauss_seidel': {'presmoothingSteps': 1,
                                   'postsmoothingSteps': 1},
                  'sor': {},
                  'chebyshev': {'degree': 3},
                  'ilu': {}}
ml = solverFactory.build('mg', hierarchy=levels, smoother=(d.smoother, smootherParams[d.smoother]), maxIter=d.maxiter, tolerance=tol, setup=True)
d.logger.info('\n'+str(ml))

A = hM[FINE].algebraicLevels[-1].A
x = DoFMap.zeros()
r = DoFMap.zeros()
A.residual_py(x, rhs, r)
r0 = ml.norm(r, False)

rate = d.addOutputGroup('rates', tested=True, aTol=1e-2)
its = d.addOutputGroup('iterations', tested=True)
res = d.addOutputGroup('residuals', tested=True, rTol=3e-1)
resHist = d.addOutputGroup('resHist', tested=True, aTol=5e-8)
errors = d.addOutputGroup('errors', tested=True, rTol=2.)

for cycle, label in [(V, 'MG'),
                     (FMG_V, 'FMG')]:
    with d.timer('Solve MG'):
        ml.cycle = cycle
        numIter = ml(rhs, x)
        residuals = ml.residuals
    A.residual_py(x, rhs, r)
    resNorm = ml.norm(r, False)
    rate.add('Rate of convergence '+label, (resNorm/r0)**(1/numIter))
    its.add('Number of iterations '+label, numIter)
    res.add('Residual norm '+label, resNorm)
    resHist.add(label, residuals)

if p.boundaryCond:
    y = DoFMap.augmentWithBoundaryData(x, boundary_data)
    mesh.exportSolutionVTK(y, y.dm, 'fichera.vtk')

# set up cg
cg = solverFactory.build('cg', A=A, maxIter=d.maxiter, tolerance=tol, setup=True)
# set up gmres
gmres = solverFactory.build('gmres', A=A, maxIter=d.maxiter//5, restarts=5, tolerance=tol, setup=True)
# set up bicgstab
bicgstab = solverFactory.build('bicgstab', A=A, maxIter=d.maxiter, tolerance=tol, setup=True)

for solver, label in [(cg, 'CG'),
                      (gmres, 'GMRES'),
                      (bicgstab, 'BICGSTAB')]:
    with d.timer('Solve '+label):
        numIter = solver(rhs, x)
        residuals = solver.residuals
    A.residual_py(x, rhs, r)
    resNorm = ml.norm(r, False)
    rate.add('Rate of convergence '+label, (resNorm/r0)**(1/numIter))
    its.add('Number of iterations '+label, numIter)
    res.add('Residual norm '+label, resNorm)
    resHist.add(label, residuals)

    with d.timer('Solve P'+label):
        solver.setPreconditioner(ml.asPreconditioner(cycle=V))
        numIter = solver(rhs, x)
        residuals = solver.residuals
    A.residual_py(x, rhs, r)
    resNorm = ml.norm(r, False)
    rate.add('Rate of convergence P'+label, (resNorm/r0)**(1/numIter))
    its.add('Number of iterations P'+label, numIter)
    res.add('Residual norm P'+label, resNorm)
    resHist.add('P'+label, residuals)


if d.lu:
    # set up lu
    with d.timer('Setup LU'):
        lu = solverFactory.build('lu', A, setup=True)
    with d.timer('Solve LU'):
        lu(rhs, x)
    A.residual_py(x, rhs, r)
    resNorm = ml.norm(r, False)
    res.add('Residual norm LU', resNorm)

if d.chol:
    # set up cholesky
    with d.timer('Setup CHOL'):
        chol = solverFactory.build('chol', A, setup=True)
    with d.timer('Solve CHOL'):
        chol(rhs, x)
    A.residual_py(x, rhs, r)
    resNorm = ml.norm(r, False)
    res.add('Residual norm CHOL', resNorm)

del ml

if p.L2ex:
    with d.timer('Mass matrix'):
        M = DoFMap.assembleMass(sss_format=True)
    z = DoFMap.assembleRHS(p.exactSolution)
    L2err = np.sqrt(np.absolute(np.vdot(x, M*x) - 2*np.vdot(z, x) + p.L2ex))
    errors.add('L^2 error', L2err)
    errors.add('L^2 error constant', L2err/mesh.h**2)
if p.H10ex:
    H10err = np.sqrt(np.absolute(p.H10ex - np.vdot(rhs, x)))
    errors.add('H^1_0 error', H10err)
    errors.add('H^1_0 error constant', H10err/mesh.h)

d.logger.info('\n'+str(rate+its+res+errors))

if d.startPlot('residuals'):
    import matplotlib.pyplot as plt
    plt.plot(resHist.MG, '-*', label='MG')
    plt.plot(resHist.FMG, '-.', label='FMG')
    plt.plot(resHist.CG, '--', label='CG')
    plt.plot(resHist.PCG, '-*', label='MG-PCG')
    plt.plot(resHist.GMRES, '.', label='GMRES')
    plt.plot(resHist.PGMRES, '-o', label='MG-GMRES')
    plt.plot(resHist.BICGSTAB, '.', label='BICGSTAB')
    plt.plot(resHist.PBICGSTAB, '-o', label='MG-BICGSTAB')
    plt.yscale('log')
    plt.legend()

if d.startPlot('solution') and hasattr(mesh, 'plotFunction'):
    mesh.plotFunction(x)

d.finish()
