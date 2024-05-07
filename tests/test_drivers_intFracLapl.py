###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.utilsFem import runDriver
import os
import inspect
import pytest


def getPath():
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def idfunc(param):
    S = [str(p) for p in param]
    return '-'.join(S).replace('(', '').replace(')', '')


@pytest.fixture(scope='module', params=[
    ('interval', 'fractional', 'poly-Dirichlet', 'lu', 'dense'),
    ('interval', 'fractional', 'poly-Neumann', 'lu', 'dense'),
    ('interval', 'constant', 'poly-Dirichlet', 'lu', 'dense'),
    ('interval', 'constant', 'poly-Neumann', 'lu', 'dense'),
    ('interval', 'inverseDistance', 'poly-Dirichlet', 'lu', 'dense'),
    ('interval', 'inverseDistance', 'poly-Neumann', 'lu', 'dense'),
    ('square', 'fractional', 'poly-Dirichlet', 'cg-mg', 'dense'),
    ('square', 'fractional', 'poly-Neumann', 'cg-mg', 'dense'),
    ('square', 'constant', 'poly-Dirichlet', 'cg-mg', 'dense'),
    ('square', 'constant', 'poly-Neumann', 'cg-mg', 'dense'),
    ('square', 'inverseDistance', 'poly-Dirichlet', 'cg-mg', 'dense'),
    ('square', 'inverseDistance', 'poly-Neumann', 'cg-mg', 'dense'),
    ('interval', 'fractional', 'poly-Dirichlet', 'lu', 'H2'),
    ('interval', 'fractional', 'poly-Neumann', 'lu', 'H2'),
    ('interval', 'constant', 'poly-Dirichlet', 'lu', 'H2'),
    ('interval', 'constant', 'poly-Neumann', 'lu', 'H2'),
    ('interval', 'inverseDistance', 'poly-Dirichlet', 'lu', 'H2'),
    ('interval', 'inverseDistance', 'poly-Neumann', 'lu', 'H2'),
    ('interval', 'gaussian', 'gaussian', 'lu', 'H2', 'fullSpace'),
    ('interval', 'exponential', 'exponential', 'lu', 'H2', 'fullSpace'),
    ('square', 'fractional', 'poly-Dirichlet', 'cg-mg', 'H2'),
    ('square', 'fractional', 'poly-Neumann', 'cg-mg', 'H2'),
    ('square', 'constant', 'poly-Dirichlet', 'cg-mg', 'H2'),
    ('square', 'constant', 'poly-Neumann', 'cg-mg', 'H2'),
    ('square', 'inverseDistance', 'poly-Dirichlet', 'cg-mg', 'H2'),
    ('square', 'inverseDistance', 'poly-Neumann', 'cg-mg', 'H2'),
],
                ids=idfunc)
def runNonlocal_params(request):
    return request.param


@pytest.mark.slow
def testNonlocal(runNonlocal_params, extras):
    if len(runNonlocal_params) == 5:
        domain, kernel, problem, solver, matrixFormat = runNonlocal_params
        interaction = None
    else:
        domain, kernel, problem, solver, matrixFormat, interaction = runNonlocal_params
    base = getPath()+'/../'
    py = ['runNonlocal.py',
          '--domain', domain,
          '--kernelType', kernel,
          '--problem', problem,
          '--solver', solver,
          '--matrixFormat', matrixFormat]
    if kernel == 'exponential':
        py += ['--exponentialRate', str(8.0)]
    elif kernel == 'gaussian':
        py += ['--gaussianVariance', str(0.1)]
    if interaction is not None:
        py += ['--interaction', interaction]
        if interaction == 'fullSpace':
            py += ['--horizon', 'inf']
    # if kernel != 'fractional':
    path = base+'drivers'
    cacheDir = getPath()+'/'
    if problem == 'poly-Neumann' and domain == 'square':
        return pytest.skip('not implemented')
    runDriver(path, py, cacheDir=cacheDir, extra=extras)


@pytest.fixture(scope='module', params=[
    ('interval', 'const(0.25)', 'constant', 'P0', 'cg-mg', 'dense', 1),
    ('interval', 'const(0.25)', 'constant', 'P0', 'cg-mg', 'H2', 1),
    ('interval', 'const(0.25)', 'constant', 'P1', 'cg-mg', 'dense', 1),
    ('interval', 'const(0.25)', 'constant', 'P1', 'cg-mg', 'H2', 1),
    ('interval', 'const(0.25)', 'zeroFlux', 'P1', 'lu', 'H2', 1),
    ('interval', 'const(0.25)', 'knownSolution', 'P1', 'cg-jacobi', 'H2', 1),
    ('interval', 'const(0.75)', 'constant', 'P1', 'lu', 'dense', 1),
    ('interval', 'const(0.75)', 'constant', 'P1', 'lu', 'H2', 1),
    ('interval', 'const(0.75)', 'zeroFlux', 'P1', 'cg-jacobi', 'H2', 1),
    ('interval', 'const(0.75)', 'knownSolution', 'P1', 'cg-mg', 'H2', 1),
    ('interval', 'varconst(0.75)', 'constant', 'P1', 'cg-jacobi', 'dense', 1),
    ('interval', 'varconst(0.75)', 'constant', 'P1', 'cg-jacobi', 'H2', 1),
    ('interval', 'varconst(0.75)', 'zeroFlux', 'P1', 'cg-mg', 'H2', 1),
    ('interval', 'varconst(0.75)', 'knownSolution', 'P1', 'lu', 'H2', 1),
    ('interval', 'const(0.25)', 'constant', 'P2', 'cg-mg', 'dense', 1),
    ('interval', 'const(0.25)', 'constant', 'P2', 'cg-mg', 'H2', 1),
    ('interval', 'const(0.75)', 'constant', 'P2', 'cg-mg', 'dense', 1),
    ('interval', 'const(0.75)', 'constant', 'P2', 'cg-mg', 'H2', 1),
    ('interval', 'const(0.25)', 'constant', 'P3', 'cg-mg', 'dense', 1),
    ('interval', 'const(0.25)', 'constant', 'P3', 'cg-mg', 'H2', 1),
    ('interval', 'const(0.75)', 'constant', 'P3', 'cg-mg', 'dense', 1),
    ('interval', 'const(0.75)', 'constant', 'P3', 'cg-mg', 'H2', 1),
    #
    ('disc', 'const(0.25)', 'constant', 'P0', 'cg-mg', 'dense', 1),
    ('disc', 'const(0.25)', 'constant', 'P0', 'cg-mg', 'H2', 1),
    ('disc', 'const(0.25)', 'constant', 'P1', 'cg-mg', 'dense', 1),
    ('disc', 'const(0.25)', 'constant', 'P1', 'cg-mg', 'H2', 1),
    ('disc', 'const(0.75)', 'constant', 'P1', 'cg-mg', 'dense', 1),
    ('disc', 'const(0.75)', 'constant', 'P1', 'cg-mg', 'H2', 1),
],
                ids=idfunc)
def runFractional_params(request):
    return request.param


@pytest.mark.slow
def testFractional(runFractional_params, extras):
    domain, s, problem, element, solver, matrixFormat, ranks = runFractional_params
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
    if ranks == 1:
        ranks = None
    runDriver(path, py, cacheDir=cacheDir, extra=extras, ranks=ranks)


@pytest.mark.slow
def testFractionalHeat(runFractional_params, extras):
    domain, s, problem, element, solver, matrixFormat, ranks = runFractional_params
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
    if ranks == 1:
        ranks = None
    runDriver(path, py, cacheDir=cacheDir, extra=extras, ranks=ranks)


@pytest.mark.slow
def testVariableOrder(extras):
    base = getPath()+'/../'
    py = 'variableOrder.py'
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, cacheDir=cacheDir, extra=extras)


@pytest.fixture(scope='module', params=[
    ('interval', 'const(0.25)'),
    ('interval', 'const(0.75)'),
    ('interval', 'varconst(0.25)'),
    ('interval', 'varconst(0.75)'),
    ('interval', 'twoDomainNonSym(0.25,0.75)'),
    ('disc', 'const(0.25)'),
    ('disc', 'const(0.75)'),
    ('disc', 'varconst(0.25)'),
    ('disc', 'varconst(0.75)'),
    ('square', 'const(0.25)'),
    ('square', 'const(0.75)'),
    ('square', 'varconst(0.25)'),
    ('square', 'varconst(0.75)'),
    ('square', 'twoDomainNonSym(0.25,0.75)'),
],
                ids=idfunc)
def runDistOp_params(request):
    return request.param


@pytest.mark.slow
def testMatvecs(runDistOp_params, extras):
    base = getPath()+'/../'
    domain, fractionalOrder = runDistOp_params
    if domain == 'interval':
        noRef = 6
    elif domain == 'disc':
        noRef = 2
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
    runDriver(path, py, ranks=4, cacheDir=cacheDir, extra=extras)


@pytest.fixture(scope='module', params=[
    ('doubleInterval', 'fractional', 'fractional', '0.2', '0.4', '0.2', '0.2', 'exact-sin-variableSolJump-fluxJump'),
    ('doubleInterval', 'fractional', 'fractional', '0.2', '0.4', '0.2', '0.4', 'exact-sin-variableSolJump-fluxJump'),
    ('doubleInterval', 'indicator', 'indicator', '0.2', '0.4', '0.2', '0.2', 'exact-sin-variableSolJump-fluxJump'),
    ('doubleInterval', 'indicator', 'indicator', '0.2', '0.4', '0.2', '0.4', 'exact-sin-variableSolJump-fluxJump'),
    ('doubleInterval', 'indicator', 'fractional', '0.2', '0.4', '0.2', '0.2', 'exact-sin-variableSolJump-fluxJump'),
    ('doubleInterval', 'indicator', 'fractional', '0.2', '0.4', '0.2', '0.4', 'exact-sin-variableSolJump-fluxJump'),
    ('doubleSquare', 'fractional', 'fractional', '0.2', '0.4', '0.2', '0.2', 'sin-variableSolJump-fluxJump'),
    ('doubleSquare', 'fractional', 'fractional', '0.2', '0.4', '0.2', '0.4', 'sin-variableSolJump-fluxJump'),
    ('doubleSquare', 'indicator', 'indicator', '0.2', '0.4', '0.2', '0.2', 'sin-variableSolJump-fluxJump'),
    ('doubleSquare', 'indicator', 'indicator', '0.2', '0.4', '0.2', '0.4', 'sin-variableSolJump-fluxJump'),
    ('doubleSquare', 'indicator', 'fractional', '0.2', '0.4', '0.2', '0.2', 'sin-variableSolJump-fluxJump'),
    ('doubleSquare', 'indicator', 'fractional', '0.2', '0.4', '0.2', '0.4', 'sin-variableSolJump-fluxJump'),
],
                ids=idfunc)
def runNonlocalInterface_params(request):
    return request.param


@pytest.mark.slow
def testNonlocalInterface(runNonlocalInterface_params, extras):
    domain, kernel1, kernel2, s11, s22, horizon1, horizon2, problem = runNonlocalInterface_params
    s12 = s11
    s21 = s22
    base = getPath()+'/../'
    py = ['runNonlocalInterface.py',
          '--domain', domain,
          '--kernel1', kernel1,
          '--kernel2', kernel2,
          '--s11', s11,
          '--s12', s12,
          '--s21', s21,
          '--s22', s22,
          '--horizon1', horizon1,
          '--horizon2', horizon2,
          '--problem', problem]
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, cacheDir=cacheDir, extra=extras)
