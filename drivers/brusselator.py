#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

###############################################################################
# Solves the fractional order Brusselator system
#
#           \partial_t U = -(-\Delta)^\alpha U + (B-1)*U + Q^2 V + B/Q * U**2 + 2*Q*U*V + U**2 * V
# \eta**2 * \partial_t V = -(-\Delta)^\beta  V - B*U     - Q^2 V - B/Q * U**2 - 2*Q*U*V - U**2 * V
#
# with zero flux conditions on U and V.
#
# s    = \beta / \alpha
# \eta = sqrt(D_X**s / D_Y)
# Q    = A * \eta
#
###############################################################################

from mpi4py import MPI
import numpy as np
from pathlib import Path
from PyNucleus_base import driver, solverFactory, timestepperFactory
from PyNucleus_base.solvers import iterative_solver
from PyNucleus_fem.mesh import plotManager
from PyNucleus_fem.femCy import assembleNonlinearity
from PyNucleus_multilevelSolver import hierarchyManager
from PyNucleus_nl.helpers import paramsForFractionalHierarchy
from PyNucleus_nl.nonlocalProblems import brusselatorProblem
import h5py

###############################################################################

d = driver(MPI.COMM_WORLD)
bP = brusselatorProblem(d)

d.add('timestepper', acceptedValues=['koto', 'euler_imex', 'ars3'])
d.add('dt', 0.01)
d.add('solver', acceptedValues=['cg-mg', 'cg-jacobi', 'lu'])
d.add('tol', 1e-6)
d.add('maxiter', 200)
d.add('nonlinear_tol', 1e-4)

d.add('dense', False)
d.add('forceRebuild', False)
d.add('restartFromCheckpoint', "")
d.add('outputStep', 10)

d.declareFigure('mesh')
d.declareFigure('solution')

params = d.process(override={'hdf5Output': 'auto'})

###############################################################################

with d.timer('Assemble matrices'):
    params['kernel'] = bP.kernelU
    params['domain'] = bP.mesh
    params['keepMeshes'] = 'all'
    params['keepAllDoFMaps'] = True
    params['buildMass'] = True
    params['assemble'] = 'ALL' if d.solver.find('mg') >= 0 else 'last'
    params['dense'] = d.dense
    params['doSave'] = True
    params['logging'] = True
    hierarchies, connectors = paramsForFractionalHierarchy(bP.noRef, params)
    hM = hierarchyManager(hierarchies, connectors, params)
    hM.setup()

    levelsU = hM.getLevelList()

    if bP.alpha == bP.beta:
        levelsV = levelsU
    else:
        raise NotImplementedError()

    mesh = levelsU[-1]['mesh']
    dm = levelsU[-1]['DoFMap']

if d.dt <= 0:
    d.dt = mesh.h**2

N = int(np.around(bP.T/d.dt))
d.dt = bP.T/N

info = d.addOutputGroup('info')
info.add('h', mesh.h)
info.add('hmin', mesh.hmin)
info.add('numDoFs', dm.num_dofs)
info.add('dt', d.dt)
info.add('N', N)
info.add('maxiter', d.maxiter)
info.add('tol', d.tol)
info.add('A', bP.A)
info.add('B', bP.B)
info.add('Dx', bP.Dx)
info.add('Dy', bP.Dy)
info.add('Q', bP.Q)
info.add('eta', bP.eta)
info.add('B_cr', bP.Bcr)
info.add('k_cr', bP.kcr)
d.logger.info('\n'+str(info))


###############################################################################
# step diffusion implicitly, nonlinearity explicitly

massU = levelsU[-1]['M']
scaledMassV = (bP.eta**2)*levelsV[-1]['M']
stiffnessU = levelsU[-1]['S']
stiffnessV = levelsV[-1]['S']


def residual(t, u, ut, residual, coeff_A=1., coeff_I=1., coeff_E=1., coeff_g=1., coeff_residual=0., forcingVector=None):
    if coeff_residual != 1.:
        residual *= coeff_residual

    if coeff_A != 0:
        temp = residual.copy()
        massU(ut[0], temp[0])
        scaledMassV(ut[1], temp[1])
        temp[0] *= coeff_A
        temp[1] *= coeff_A
        residual += temp

    if coeff_I != 0.:
        temp = residual.copy()
        stiffnessU(u[0], temp[0])
        stiffnessV(u[1], temp[1])
        temp *= coeff_I
        residual += temp

    if coeff_E != 0.:
        temp = assembleNonlinearity(dm.mesh,
                                    bP.nonlinearity,
                                    dm,
                                    u)
        temp *= -coeff_E
        residual += temp


def newHierarchy(levels, facM, facS, levels2=None, key1='M', key2='A'):
    if levels2 is None:
        levels2 = levels
    newLevels = []
    for i in range(len(levels)):
        newLevels.append({})
        if 'R' in levels[i]:
            newLevels[i]['R'] = levels[i]['R']
        if 'P' in levels[i]:
            newLevels[i]['P'] = levels[i]['P']
        newLevels[i]['A'] = facM*levels[i][key1] + facS*levels2[i][key2]
        if 'mesh' in levels[i]:
            newLevels[i]['mesh'] = levels[i]['mesh']
    return newLevels


def solverBuilder(t, alpha, beta):
    with d.timer('Setup solvers'):
        if beta == 0.:
            solverType = 'cg-jacobi'
        else:
            solverType = d.solver
        if solverType.find('mg') >= 0:
            levelsUMod = newHierarchy(levelsU, facM=alpha, facS=beta)
            levelsVMod = newHierarchy(levelsV, facM=bP.eta**2*alpha, facS=beta)
            solverU = solverFactory(solverType, hierarchy=levelsUMod, setup=True)
            solverV = solverFactory(solverType, hierarchy=levelsVMod, setup=True)
        else:
            solverU = solverFactory(solverType, A=alpha*massU+beta*stiffnessU, setup=True)
            solverV = solverFactory(solverType, A=alpha*scaledMassV+beta*stiffnessV, setup=True)

        if isinstance(solverU, iterative_solver):
            solverU.tolerance = d.tol
            solverU.maxIter = d.maxiter
        if isinstance(solverV, iterative_solver):
            solverV.tolerance = d.tol
            solverV.maxIter = d.maxiter

    def solve(rhs, sol):
        if isinstance(solverU, iterative_solver):
            solverU.setInitialGuess(sol[0])
        itsU = solverU(rhs[0], sol[0])
        if isinstance(solverU, iterative_solver):
            assert solverU.residuals[-1] < d.tol, solverU.residuals

        if isinstance(solverV, iterative_solver):
            solverV.setInitialGuess(sol[1])
        itsV = solverV(rhs[1], sol[1])
        if isinstance(solverV, iterative_solver):
            assert solverV.residuals[-1] < d.tol, solverV.residuals

        d.logger.info('Iterations {}: {}, {}'.format(solverType, itsU, itsV))

    return solve


timestepper = timestepperFactory(d.timestepper,
                                 dm=dm,
                                 residual=residual,
                                 solverBuilder=solverBuilder,
                                 numSystemVectors=bP.nonlinearity.numInputs)


###############################################################################

data = d.addOutputGroup('data')
data.add('dm', dm)
U = d.addOutputGroup('U')
V = d.addOutputGroup('V')

sol = dm.zeros(numVecs=2)

if d.restartFromCheckpoint != "" and Path(d.restartFromCheckpoint).exists():
    resultFile = h5py.File(str(d.restartFromCheckpoint), 'r')
    for i in resultFile['U']:
        sol[0].assign(np.array(resultFile['U'][str(i)]))
        sol[1].assign(np.array(resultFile['V'][str(i)]))
        U.add(i, sol[0])
        V.add(i, sol[1])

    I = max([int(i) for i in resultFile['U']])
    t = d.dt*I
    n = I+1
    resultFile.close()
    d.logger.info('Read state from HDF5')
    d.comm.Barrier()
else:
    n = 0
    t = 0.

    sol[0].assign(dm.project(bP.initial_U))
    sol[1].assign(dm.project(bP.initial_V))

d.logger.info('t={:.3} u in [{:.3}, {:.3}], v in [{:.3}, {:.3}]'.format(t,
                                                                        sol[0].min(), sol[0].max(),
                                                                        sol[1].min(), sol[1].max()))
for i in range(n, N):
    t, picardIts = timestepper.picardStep(t, d.dt, sol, tol=d.nonlinear_tol)
    d.logger.info('Picard iterations: {}'.format(picardIts))

    d.logger.info('t={:.3} u in [{:.3}, {:.3}], v in [{:.3}, {:.3}]'.format(t,
                                                                            sol[0].min(), sol[0].max(),
                                                                            sol[1].min(), sol[1].max()))

    if i % d.outputStep == 0:
        U.add(str(i), sol[0])
        V.add(str(i), sol[1])


if d.startPlot('mesh'):
    mesh.plot()

if d.startPlot('solution'):
    pM = plotManager(mesh, dm, useSubPlots=True, defaults={'shading': 'gouraud'})
    pM.add(sol[0], flat=True, label='u')
    pM.add(sol[1], flat=True, label='v')
    pM.plot()

d.finish()
