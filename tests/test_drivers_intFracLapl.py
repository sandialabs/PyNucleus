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


@pytest.fixture(scope='module', params=['fractional', 'indicator', 'peridynamic'])
def kernel(request):
    return request.param


@pytest.fixture(scope='module', params=['poly-Dirichlet', 'poly-Neumann'])
def problem(request):
    return request.param


def testNonlocal(domain, kernel, problem, extra):
    base = getPath()+'/../'
    py = ['runNonlocal.py',
          '--domain', domain,
          '--kernel', kernel,
          '--problem', problem]
    # if kernel != 'fractional':
    py += ['--dense']
    path = base+'drivers'
    cacheDir = getPath()+'/'
    if problem == 'poly-Neumann' and domain == 'square':
        return pytest.skip('not implemented')
    runDriver(path, py, cacheDir=cacheDir, extra=extra)



def testVariableOrder(extra):
    base = getPath()+'/../'
    py = 'variableOrder.py'
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, cacheDir=cacheDir, extra=extra)
