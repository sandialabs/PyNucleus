###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.utilsFem import runDriver
import pytest


@pytest.fixture()
def path(request):
    return request.path.parent


################################################################################
# multigrid

@pytest.fixture(scope='module', params=[1, 4])
def ranks(request):
    return request.param


@pytest.fixture(scope='module', params=['interval',
                                        pytest.param('square', marks=pytest.mark.slow),
                                        pytest.param('cube', marks=pytest.mark.slow)])
def domain(request):
    return request.param


@pytest.fixture(scope='module', params=['P1', 'P2', 'P3'])
def element(request):
    return request.param


@pytest.fixture(scope='module', params=[False, True])
def symmetric(request):
    return request.param


@pytest.mark.slow
def testGMG(path, extras):
    py = 'runSerialGMG.py'
    runDriver(path/'../drivers', py, cacheDir=path, extra=extras)


def testParallelGMG(ranks, domain, element, symmetric, path, extras):
    py = ['runParallelGMG.py',
          '--domain', domain,
          '--element', element]
    if symmetric:
        py.append('--symmetric')
    runDriver(path/'../drivers', py, ranks=ranks, cacheDir=path, relTol=3e-2, extra=extras)


################################################################################
# multigrid for Helmholtz

def testHelmholtz(ranks, domain, path, extras):
    py = ['runHelmholtz.py', '--domain', domain]
    runDriver(path/'../drivers', py, ranks=ranks, cacheDir=path, extra=extras)


################################################################################
# interface problem

@pytest.mark.parametrize("domain, noRef",
                         [
                             ('doubleInterval', 10),
                             ('doubleSquare', 5)
                         ])
def testInterface(domain, noRef, path, extras):
    py = ['interfaceProblem.py',
          '--domain', domain,
          '--noRef', str(noRef)]
    runDriver(path/'../drivers', py, ranks=1, cacheDir=path, relTol=5e-2, extra=extras)
