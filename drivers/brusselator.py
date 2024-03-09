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
# \eta**2 * \partial_t V = -(-\Delta)^\beta  U - B*U     - Q^2 V - B/Q * U**2 - 2*Q*U*V - U**2 * V
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
from PyNucleus_base import driver, solverFactory
from PyNucleus_base.linear_operators import TimeStepperLinearOperator
from PyNucleus_fem.femCy import assembleNonlinearity
from PyNucleus_multilevelSolver import hierarchyManager
from PyNucleus_nl import paramsForFractionalHierarchy
from PyNucleus_nl.nonlocalProblems import brusselatorProblem
from PyNucleus_base.timestepping import EulerIMEX, ARS3, koto
import h5py

###############################################################################

d = driver(MPI.COMM_WORLD)
bP = brusselatorProblem(d)

d.add('timestepper', acceptedValues=['koto', 'euler_imex', 'ars3'])
d.add('dt', 0.01)
d.add('solver', acceptedValues=['cg-mg', 'cg-jacobi'])
d.add('tol', 1e-6)
d.add('maxiter', 200)

d.add('dense', False)
d.add('forceRebuild', False)
d.add('restartFromCheckpoint', False)
d.add('outputStep', 10)

params = d.process()

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
massV = (bP.eta**2)*levelsV[-1]['M']
S_U = levelsU[-1]['S']
S_V = levelsV[-1]['S']

if d.timestepper == 'euler_imex':
    ts_class = EulerIMEX
elif d.timestepper == 'ars3':
    ts_class = ARS3
elif d.timestepper == 'koto':
    ts_class = koto


def explicit(t, sol, sol_new):
    sol_temp = assembleNonlinearity(mesh,
                                    bP.nonlinearity,
                                    dm,
                                    sol)
    sol_new[0].assign(sol_temp[0])
    sol_new[1].assign(sol_temp[1])


def implicit(t, sol, sol_new):
    S_U(sol[0], sol_new[0])
    sol_new[0] *= -1.
    S_V(sol[1], sol_new[1])
    sol_new[1] *= -1.


def mass(sol, sol_new):
    massU(sol[0], sol_new[0])
    massV(sol[1], sol_new[1])


with d.timer('Setup solvers'):
    if d.solver.find('mg') >= 0:
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
                newLevels[i]['A'] = TimeStepperLinearOperator(levels[i][key1],
                                                              levels2[i][key2],
                                                              facS, facM)
                if 'mesh' in levels[i]:
                    newLevels[i]['mesh'] = levels[i]['mesh']
            return newLevels

        levelsU = newHierarchy(levelsU, 1., ts_class.gamma*d.dt)
        levelsV = newHierarchy(levelsV, bP.eta**2, ts_class.gamma*d.dt)
        solverImplicitU = solverFactory(d.solver, hierarchy=levelsU, setup=True)
        solverImplicitV = solverFactory(d.solver, hierarchy=levelsV, setup=True)
    else:
        solverImplicitU = solverFactory(d.solver, A=massU+((ts_class.gamma*d.dt)*S_U), setup=True)
        solverImplicitV = solverFactory(d.solver, A=massV+((ts_class.gamma*d.dt)*S_V), setup=True)
    solverImplicitU.tolerance = d.tol
    solverImplicitU.maxIter = d.maxiter
    solverImplicitV.tolerance = d.tol
    solverImplicitV.maxIter = d.maxiter

    solverMassU = solverFactory('cg-jacobi', A=massU, setup=True)
    solverMassU.tolerance = d.tol
    solverMassU.maxIter = d.maxiter

    solverMassV = solverFactory('cg-jacobi', A=massV, setup=True)
    solverMassV.tolerance = d.tol
    solverMassV.maxIter = d.maxiter


def implicit_solve(t, rhs, sol, sol_new, dt):
    with d.timer('implicit_solve'):
        if dt == 0.:
            mass_solve(rhs, sol, sol_new)
            return

        solverImplicitU.setInitialGuess(sol[0])
        itsU = solverImplicitU(rhs[0], sol_new[0])
        assert solverImplicitU.residuals[-1] < d.tol, solverImplicitU.residuals

        solverImplicitV.setInitialGuess(sol[1])
        itsV = solverImplicitV(rhs[1], sol_new[1])
        assert solverImplicitV.residuals[-1] < d.tol, solverImplicitV.residuals
    d.logger.info('Iterations: {}, {}'.format(itsU, itsV))


def mass_solve(rhs, sol, sol_new):
    with d.timer('mass_solve'):
        solverMassU.setInitialGuess(sol[0])
        solverMassU(rhs[0], sol_new[0])
        assert solverMassU.residuals[-1] < d.tol, solverMassU.residuals

        solverMassV.setInitialGuess(sol[1])
        solverMassV(rhs[1], sol_new[1])
        assert solverMassV.residuals[-1] < d.tol, solverMassV.residuals


timestepper = ts_class(dm,
                       implicit, implicit_solve, explicit,
                       bP.nonlinearity.numInputs,
                       mass, mass_solve)

###############################################################################

filename = d.identifier+'.hdf5'

sol = dm.zeros(2)
if d.restartFromCheckpoint and Path(filename).exists():
    resultFile = h5py.File(str(filename), 'r')
    I = max([int(i) for i in resultFile['U']])
    sol[0].assign(np.array(resultFile['U'][str(I)]))
    sol[1].assign(np.array(resultFile['V'][str(I)]))
    t = d.dt*I
    n = I+1
    resultFile.close()
    d.logger.info('Read state from HDF5')
    d.comm.Barrier()
    if d.isMaster:
        resultFile = h5py.File(str(filename), 'a')
else:
    n = 0
    t = 0.
    if d.isMaster:
        resultFile = h5py.File(str(filename), 'w')
        resultFile.create_group('U')
        resultFile.create_group('V')

        resultFile.create_group('mesh')
        mesh.HDF5write(resultFile['mesh'])
        resultFile.create_group('dm')
        dm.HDF5write(resultFile['dm'])

    sol[0].assign(dm.project(bP.initial_U))
    sol[1].assign(dm.project(bP.initial_V))

d.logger.info('t={:.3} u in [{:.3}, {:.3}], v in [{:.3}, {:.3}]'.format(t,
                                                                        sol[0].min(), sol[0].max(),
                                                                        sol[1].min(), sol[1].max()))
for i in range(n, N):
    timestepper.step(sol, t, d.dt, sol)
    t += d.dt

    d.logger.info('t={:.3} u in [{:.3}, {:.3}], v in [{:.3}, {:.3}]'.format(t,
                                                                            sol[0].min(), sol[0].max(),
                                                                            sol[1].min(), sol[1].max()))

    if (i % d.outputStep == 0) and d.isMaster:
        resultFile['U'].create_dataset(str(i), data=sol[0].toarray())
        resultFile['V'].create_dataset(str(i), data=sol[1].toarray())
        resultFile.flush()
if d.isMaster:
    resultFile.close()

d.finish()
