###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from __future__ import division
import numpy as np
import numpy.linalg
from PyNucleus.fem.mesh import simpleInterval, circle
from PyNucleus.fem.DoFMaps import P1_DoFMap, P2_DoFMap
from PyNucleus.fem.functions import constant
from PyNucleus.nl.nonlocalAssembly import (assembleNonlocalOperator,
                                           nonlocalBuilder)
from PyNucleus.nl.clusterMethodCy import H2Matrix
from PyNucleus.base.myTypes import REAL
from scipy.special import gamma
from PyNucleus.nl.kernels import getFractionalKernel
from PyNucleus.nl.fractionalOrders import constFractionalOrder
from PyNucleus.nl.kernelNormalization import variableFractionalLaplacianScaling
import pytest
import logging
LOGGER = logging.getLogger(__name__)


######################################################################
# Test dense operators by checking Hs-error

def fracLapl(dim, s, errBnd, refinements, element, genKernel=False, cached=False):
    if dim == 1:
        mesh = simpleInterval(-1, 1)
    elif dim == 2:
        mesh = circle(10)
    for _ in range(refinements):
        mesh = mesh.refine()
    if element == 'P1':
        dm = P1_DoFMap(mesh, tag=0)
    elif element == 'P2':
        dm = P2_DoFMap(mesh, tag=0)
    if cached:
        raise NotImplementedError()
    else:
        A = assembleNonlocalOperator(mesh, dm, s, genKernel=genKernel).data
    fun = constant(1.)
    rhs = dm.assembleRHS(fun)
    u = np.linalg.solve(A, rhs)
    s = s.value
    if dim == 1:
        err = np.sqrt(abs(np.vdot(rhs, u) - 2**(-2*s)*np.pi/gamma(1/2+s)/gamma(s+3/2)))
    else:
        err = np.sqrt(abs(np.dot(rhs, u)-2*np.pi * 2**(-2*s)*gamma(1)/gamma(1+s)**2/2/(s+1)))
    msg = ''
    msg += '\nBound:      {}'.format(errBnd)
    msg += '\nAll:        {}'.format(err)
    LOGGER.info(msg)
    assert err < errBnd, '{} not smaller than {}'.format(err, errBnd)


@pytest.fixture(scope='module',
                params=[(1, constFractionalOrder(0.3), 'P1', 0.15),
                        (1, constFractionalOrder(0.7), 'P1', 0.1),
                        (2, constFractionalOrder(0.3), 'P1', 0.5),
                        (2, constFractionalOrder(0.7), 'P1', 0.35)],
                ids=['1-P1-0.3', '1-P1-0.7',
                     '2-P1-0.3', '2-P1-0.7'])
def setupExact(request):
    return request.param


def testFracLapl(setupExact):
    dim, s, element, errBnd = setupExact
    if dim == 1:
        refinements = 6
    else:
        refinements = 2
    fracLapl(dim, s, errBnd, refinements, element)


######################################################################
# Test scaling and diagonal of dense operators

def scaling(dim, s, horizon, refinements):
    if dim == 1:
        mesh = simpleInterval(-1, 1)
    else:
        mesh = circle(10)
    for _ in range(refinements):
        mesh = mesh.refine()
    dm = P1_DoFMap(mesh, tag=0)

    kernel1 = getFractionalKernel(mesh.dim, s, horizon)
    scaling = variableFractionalLaplacianScaling(True)
    kernel2 = getFractionalKernel(mesh.dim, s, horizon, scaling=scaling)
    print(kernel1, kernel2)
    zeroExterior = not np.isfinite(horizon.value)
    builder1 = nonlocalBuilder(dm, kernel1, zeroExterior=zeroExterior)
    builder2 = nonlocalBuilder(dm, kernel2, zeroExterior=zeroExterior)
    A = builder1.getDense().toarray()
    B = builder2.getDense().toarray()
    assert np.allclose(A, B)

    if horizon.value == np.inf:
        dA = builder1.getDiagonal()
        mA = np.absolute(A.diagonal()-dA.diagonal).max()
        rmA = np.absolute((A.diagonal()-dA.diagonal)/A.diagonal()).max()
        assert np.allclose(A.diagonal(), dA.diagonal, rtol=2e-3), 'Diagonal A does not match; max diff = {}, rel max diff={}'.format(mA, rmA)
        dB = builder2.getDiagonal()
        mB = np.absolute(B.diagonal()-dB.diagonal).max()
        rmB = np.absolute((B.diagonal()-dB.diagonal)/B.diagonal()).max()
        assert np.allclose(B.diagonal(), dB.diagonal, rtol=2e-3), 'Diagonal B does not match; max diff = {}, rel max diff={}'.format(mB, rmB)


@pytest.fixture(scope='module',
                params=[(1, constFractionalOrder(0.25), constant(np.inf)),
                        (1, constFractionalOrder(0.25), constant(1.)),
                        (1, constFractionalOrder(0.75), constant(np.inf)),
                        (1, constFractionalOrder(0.75), constant(1.)),
                        (2, constFractionalOrder(0.25), constant(np.inf)),
                        (2, constFractionalOrder(0.25), constant(1.)),
                        (2, constFractionalOrder(0.75), constant(np.inf)),
                        (2, constFractionalOrder(0.75), constant(1.))],
                ids=['1-0.25-inf', '1-0.25-1', '1-0.75-inf', '1-0.75-1',
                     '2-0.25-inf', '2-0.25-1', '2-0.75-inf', '2-0.75-1'])
def setupScaling(request):
    return request.param


def testScaling(setupScaling):
    dim, s, horizon = setupScaling
    if dim == 1:
        refinements = 6
    else:
        refinements = 2
    scaling(dim, s, horizon, refinements)


######################################################################
# Test H2 operators by comparing to dense operator

def h2(dim, s, refinements, element, errBnd, genKernel=False):
    if dim == 1:
        mesh = simpleInterval(-1, 1)
        eta = 1
        maxLevels = None
    elif dim == 2:
        mesh = circle(10)
        eta = 3
        maxLevels = 4
    # mesh = mesh.refine()
    for _ in range(refinements):
        mesh = mesh.refine()
    # mesh.sortVertices()
    if element == 'P1':
        DoFMap_fine = P1_DoFMap(mesh, tag=-1 if s.value < 0.5 else 0)
    elif element == 'P2':
        DoFMap_fine = P2_DoFMap(mesh, tag=-1 if s.value < 0.5 else 0)
    params = {}
    params['genKernel'] = genKernel
    params['eta'] = eta
    params['maxLevels'] = maxLevels
    kernel = getFractionalKernel(mesh.dim, s, constant(np.inf))
    builder = nonlocalBuilder(DoFMap_fine, kernel, params=params, zeroExterior=True)

    A_d = np.array(builder.getDense().data)
    A_h2 = builder.getH2()
    assert isinstance(A_h2, H2Matrix)
    LOGGER.info(str(A_h2))

    n = A_d.shape[0]
    Afar = np.zeros((n, n), dtype=REAL)
    for level in A_h2.Pfar:
        for c in A_h2.Pfar[level]:
            Afar[np.ix_(list(c.n1.dofs.toSet()), list(c.n2.dofs.toSet()))] = A_d[np.ix_(list(c.n1.dofs.toSet()), list(c.n2.dofs.toSet()))]
    Anear = A_d-Afar
    errNear = np.absolute(Anear-A_h2.Anear.toarray()).max()

    x = np.ones((A_d.shape[0]), dtype=REAL)

    y_d = np.dot(Afar, x)
    y_h2 = np.zeros_like(y_d)
    if len(A_h2.Pfar) > 0:
        A_h2.tree.upwardPass_py(x)
        A_h2.tree.resetCoefficientsDown_py()
        for level in A_h2.Pfar:
            for clusterPair in A_h2.Pfar[level]:
                n1, n2 = clusterPair.n1, clusterPair.n2
                clusterPair.apply(n2.coefficientsUp, n1.coefficientsDown)
        A_h2.tree.downwardPass_py(y_h2)
    errFar = np.absolute(y_d-y_h2).max()

    y_d = np.dot(A_d, x)
    y_h2 = A_h2*x
    errAll = np.absolute(y_d-y_h2).max()
    msg = ''
    msg += '\nBound:      {}'.format(errBnd)
    msg += '\nNear field: {}'.format(errNear)
    msg += '\nFar field:  {}'.format(errFar)
    msg += '\nAll:        {}'.format(errAll)
    LOGGER.info(msg)
    if errNear > errBnd:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            print(Anear-A_h2.Anear.toarray())
            plt.pcolormesh(np.absolute(Anear-A_h2.Anear.toarray()), norm=matplotlib.colors.LogNorm())
            plt.colorbar()
            plt.show()
        except ImportError:
            pass
    assert errNear < errBnd
    assert errFar < errBnd
    assert errAll < errBnd


def idfunc(param):
    S = [str(p) for p in param]
    return '-'.join(S)


@pytest.fixture(scope='module',
                params=[(1, constFractionalOrder(0.3), 1e-4, 'P1'),
                        (1, constFractionalOrder(0.7), 1e-2, 'P1'),
                        (2, constFractionalOrder(0.3), 1.2e-4, 'P1'),
                        (2, constFractionalOrder(0.7), 1e-2, 'P1')],
                ids=idfunc)
def setupH2(request):
    return request.param


def testH2(setupH2):
    dim, s, errBnd, element = setupH2
    if dim == 1:
        refinements = 6
    else:
        refinements = 3
    h2(dim, s, refinements, element, errBnd)
