###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus.base.utilsFem import runDriver
import os
import inspect
import pytest


def getPath():
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def idfunc(param):
    S = [str(p) for p in param]
    return '-'.join(S).replace('(', '').replace(')', '')


@pytest.fixture(scope='module', params=[
    ('interval', 'fractional', 'poly-Dirichlet', 'lu'),
    ('interval', 'fractional', 'poly-Neumann', 'lu'),
    ('interval', 'constant', 'poly-Dirichlet', 'lu'),
    ('interval', 'constant', 'poly-Neumann', 'lu'),
    ('interval', 'inverseDistance', 'poly-Dirichlet', 'lu'),
    ('interval', 'inverseDistance', 'poly-Neumann', 'lu'),
    ('square', 'fractional', 'poly-Dirichlet', 'cg-mg'),
    ('square', 'fractional', 'poly-Neumann', 'cg-mg'),
    ('square', 'constant', 'poly-Dirichlet', 'cg-mg'),
    ('square', 'constant', 'poly-Neumann', 'cg-mg'),
    ('square', 'inverseDistance', 'poly-Dirichlet', 'cg-mg'),
    ('square', 'inverseDistance', 'poly-Neumann', 'cg-mg'),
],
                ids=idfunc)
def runNonlocal_params(request):
    return request.param


@pytest.mark.slow
def testNonlocal(runNonlocal_params, extra):
    domain, kernel, problem, solver = runNonlocal_params
    base = getPath()+'/../'
    py = ['runNonlocal.py',
          '--domain', domain,
          '--kernelType', kernel,
          '--problem', problem,
          '--solver', solver]
    # if kernel != 'fractional':
    py += ['--matrixFormat', 'dense']
    path = base+'drivers'
    cacheDir = getPath()+'/'
    if problem == 'poly-Neumann' and domain == 'square':
        return pytest.skip('not implemented')
    runDriver(path, py, cacheDir=cacheDir, extra=extra)


@pytest.fixture(scope='module', params=[
    ('interval', 'const(0.25)', 'constant', 'P0', 'cg-mg', 'dense'),
    ('interval', 'const(0.25)', 'constant', 'P0', 'cg-mg', 'H2'),
    ('interval', 'const(0.25)', 'constant', 'P1', 'cg-mg', 'dense'),
    ('interval', 'const(0.25)', 'constant', 'P1', 'cg-mg', 'H2'),
    ('interval', 'const(0.25)', 'zeroFlux', 'P1', 'lu', 'H2'),
    ('interval', 'const(0.25)', 'knownSolution', 'P1', 'cg-jacobi', 'H2'),
    ('interval', 'const(0.75)', 'constant', 'P1', 'lu', 'dense'),
    ('interval', 'const(0.75)', 'constant', 'P1', 'lu', 'H2'),
    ('interval', 'const(0.75)', 'zeroFlux', 'P1', 'cg-jacobi', 'H2'),
    ('interval', 'const(0.75)', 'knownSolution', 'P1', 'cg-mg', 'H2'),
    ('interval', 'varconst(0.75)', 'constant', 'P1', 'cg-jacobi', 'dense'),
    ('interval', 'varconst(0.75)', 'constant', 'P1', 'cg-jacobi', 'H2'),
    ('interval', 'varconst(0.75)', 'zeroFlux', 'P1', 'cg-mg', 'H2'),
    ('interval', 'varconst(0.75)', 'knownSolution', 'P1', 'lu', 'H2'),
    ('interval', 'const(0.25)', 'constant', 'P2', 'cg-mg', 'dense'),
    ('interval', 'const(0.25)', 'constant', 'P2', 'cg-mg', 'H2'),
    ('interval', 'const(0.75)', 'constant', 'P2', 'cg-mg', 'dense'),
    ('interval', 'const(0.75)', 'constant', 'P2', 'cg-mg', 'H2'),
    ('interval', 'const(0.25)', 'constant', 'P3', 'cg-mg', 'dense'),
    ('interval', 'const(0.25)', 'constant', 'P3', 'cg-mg', 'H2'),
    ('interval', 'const(0.75)', 'constant', 'P3', 'cg-mg', 'dense'),
    ('interval', 'const(0.75)', 'constant', 'P3', 'cg-mg', 'H2'),
    #
    ('disc', 'const(0.25)', 'constant', 'P0', 'cg-mg', 'dense'),
    ('disc', 'const(0.25)', 'constant', 'P0', 'cg-mg', 'H2'),
    ('disc', 'const(0.25)', 'constant', 'P1', 'cg-mg', 'dense'),
    ('disc', 'const(0.25)', 'constant', 'P1', 'cg-mg', 'H2'),
    ('disc', 'const(0.75)', 'constant', 'P1', 'cg-mg', 'dense'),
    ('disc', 'const(0.75)', 'constant', 'P1', 'cg-mg', 'H2'),
],
                ids=idfunc)
def runFractional_params(request):
    return request.param


@pytest.mark.slow
def testFractional(runFractional_params, extra):
    domain, s, problem, element, solver, matrixFormat = runFractional_params
    base = getPath()+'/../'
    py = ['runFractional.py',
          '--domain', domain,
          '--s', s,
          '--problem', problem,
          '--element', element,
          '--solver', solver,
          '--matrixFormat', matrixFormat]
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, cacheDir=cacheDir, extra=extra)


@pytest.mark.slow
def testFractionalHeat(runFractional_params, extra):
    domain, s, problem, element, solver, matrixFormat = runFractional_params
    base = getPath()+'/../'
    py = ['runFractionalHeat.py',
          '--domain', domain,
          '--s', s,
          '--problem', problem,
          '--element', element,
          '--solver', solver,
          '--matrixFormat', matrixFormat]
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, cacheDir=cacheDir, extra=extra)


@pytest.mark.slow
def testVariableOrder(extra):
    base = getPath()+'/../'
    py = 'variableOrder.py'
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, cacheDir=cacheDir, extra=extra)


@pytest.fixture(scope='module', params=[
    ('interval', 'const(0.25)'),
    ('interval', 'const(0.75)'),
    ('interval', 'varconst(0.25)'),
    ('interval', 'varconst(0.75)'),
    ('square', 'const(0.25)'),
    ('square', 'const(0.75)'),
    ('square', 'varconst(0.25)'),
    ('square', 'varconst(0.75)'),
],
                ids=idfunc)
def runDistOp_params(request):
    return request.param


@pytest.mark.slow
def testMatvecs(runDistOp_params, extra):
    base = getPath()+'/../'
    domain, fractionalOrder = runDistOp_params
    if domain == 'interval':
        noRef = 6
    elif domain == 'square':
        noRef = 3
    py = ['testDistOp.py',
          '--horizon', 'inf',
          '--domain', domain,
          '--s', fractionalOrder,
          '--problem', 'constant',
          '--noRef', str(noRef),
          '--buildDense',
          '--buildH2',
          '--buildH2Reduced',
          '--buildDistributedH2Bcast',
          '--buildDistributedH2',
          '--doSolve']
    py += ['--no-write']
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, ranks=4, cacheDir=cacheDir, extra=extra)


