###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from mpi4py import MPI
from PyNucleus.base.utilsFem import runDriver
import os
import inspect
import pytest


def getPath():
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


################################################################################
# multigrid

@pytest.fixture(scope='module', params=[1, 4])
def ranks(request):
    return request.param


@pytest.fixture(scope='module', params=['interval', 'square', 'cube'])
def domain(request):
    return request.param


@pytest.fixture(scope='module', params=['P1', 'P2', 'P3'])
def element(request):
    return request.param


@pytest.fixture(scope='module', params=[False, True])
def symmetric(request):
    return request.param


def testGMG(extra):
    base = getPath()+'/../'
    py = 'runSerialGMG.py'
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, cacheDir=cacheDir, extra=extra)


def testParallelGMG(ranks, domain, element, symmetric, extra):
    base = getPath()+'/../'
    py = ['runParallelGMG.py',
          '--domain', domain,
          '--element', element]
    if symmetric:
        py.append('--symmetric')
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, ranks=ranks, cacheDir=cacheDir, relTol=3e-2, extra=extra)



################################################################################
# multigrid for Helmholtz

def testHelmholtz(ranks, domain, extra):
    base = getPath()+'/../'
    py = ['runHelmholtz.py', '--domain', domain]
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, ranks=ranks, cacheDir=cacheDir, extra=extra)



################################################################################
# interface problem

@pytest.fixture(scope='module', params=[('doubleInterval', 10),
                                        ('doubleSquare', 5)])
def domainNoRef(request):
    return request.param


def testInterface(domainNoRef, extra):
    domain, noRef = domainNoRef
    base = getPath()+'/../'
    py = ['interfaceProblem.py',
          '--domain', domain,
          '--noRef', str(noRef)]
    path = base+'drivers'
    cacheDir = getPath()+'/'
    runDriver(path, py, ranks=1, cacheDir=cacheDir, relTol=5e-2, extra=extra)
