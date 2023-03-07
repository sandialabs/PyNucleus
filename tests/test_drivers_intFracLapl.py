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


@pytest.fixture(scope='module', params=['interval', 'square'])
def domain(request):
    return request.param


@pytest.fixture(scope='module', params=['fractional', 'constant', 'inverseDistance'])
def kernel(request):
    return request.param


@pytest.fixture(scope='module', params=['poly-Dirichlet', 'poly-Neumann'])
def problem(request):
    return request.param


def testNonlocal(domain, kernel, problem, extra):
    base = getPath()+'/../'
    py = ['runNonlocal.py',
          '--domain', domain,
          '--kernelType', kernel,
          '--problem', problem]
    # if kernel != 'fractional':
    py += ['--matrixFormat', 'dense']
    path = base+'drivers'
    cacheDir = getPath()+'/'
    if problem == 'poly-Neumann' and domain == 'square':
        return pytest.skip('not implemented')
    runDriver(path, py, cacheDir=cacheDir, extra=extra)



@pytest.fixture(scope='module', params=['constant', 'zeroFlux'])
def fractional_1d_problem(request):
    return request.param


def testFractional(fractional_1d_problem, extra):
    base = getPath()+'/../'
    py = ['runFractional.py',
          '--domain', 'interval',
          '--problem', fractional_1d_problem]
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, cacheDir=cacheDir, extra=extra)



def testVariableOrder(extra):
    base = getPath()+'/../'
    py = 'variableOrder.py'
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, cacheDir=cacheDir, extra=extra)


@pytest.fixture(scope='module', params=['const(0.25)',
                                        'const(0.75)',
                                        'varconst(0.25)',
                                        'varconst(0.75)',
                                        'twoDomain(0.25,0.75,0.5,0.5)'])
def fractionalOrder(request):
    return request.param


def testMatvecs(domain, fractionalOrder, extra):
    base = getPath()+'/../'
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
    if problem == 'poly-Neumann' and domain == 'square':
        return pytest.skip('not implemented')
    runDriver(path, py, ranks=4, cacheDir=cacheDir, extra=extra)
