###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from PyNucleus_nl import fractionalOrderFactory, kernelFactory
from PyNucleus_nl.fractionalOrders import (constFractionalOrder,
                                           variableConstFractionalOrder,
                                           smoothedLeftRightFractionalOrder)
from PyNucleus_fem import constant
from scipy.special import gamma, erf
from numpy import pi, exp, sqrt
from numpy.linalg import norm
import pytest


def idfunc(param):
    S = [str(p) for p in param]
    return '-'.join(S)



@pytest.fixture(scope='module', params=[
    (1, 'constant', 0.5, True),
    (1, 'constant', 0.5, False),
    (2, 'constant', 0.5, True),
    (2, 'constant', 0.5, False),
    (1, 'inverseDistance', 0.5, True),
    (1, 'inverseDistance', 0.5, False),
    (2, 'inverseDistance', 0.5, True),
    (2, 'inverseDistance', 0.5, False),
    (1, 'Gaussian', 0.5, True),
    (1, 'Gaussian', 0.5, False),
    (2, 'Gaussian', 0.5, True),
    (2, 'Gaussian', 0.5, False),
], ids=idfunc)
def integrableKernelParams(request):
    return request.param


def testIntegrableKernel(integrableKernelParams):
    dim, kernelType, horizon, normalized = integrableKernelParams
    if dim == 1:
        xy_values = [(np.array([0.1]), np.array([0.2])),
                     (np.array([0.1]), np.array([0.7]))]
    elif dim == 2:
        xy_values = [(np.array([0.1, 0.1]), np.array([0.2, 0.2])),
                     (np.array([0.1, 0.1]), np.array([0.7, 0.2]))]
    else:
        raise NotImplementedError()
    kernel = kernelFactory(kernelType, dim=dim, horizon=horizon, normalized=normalized)
    horizonValue = kernel.horizon.value

    if normalized:
        if kernelType == 'constant':
            if dim == 1:
                const = 3/horizonValue**3 * 0.5
            elif dim == 2:
                const = 8./pi/horizonValue**4 *0.5
            else:
                raise NotImplementedError()
        elif kernelType == 'inverseDistance':
            if dim == 1:
                const = 2./horizonValue**2 * 0.5
            elif dim == 2:
                const = 6./pi/horizonValue**3 *0.5
            else:
                raise NotImplementedError()
        elif kernelType == 'Gaussian':
            if dim == 1:
                const = 4.0/sqrt(pi)/(erf(3.0)-6.0*exp(-9.0)/sqrt(pi))/(horizonValue/3.0)**3 / 2.
            elif dim == 2:
                const = 4.0/pi/(1.0-10.0*exp(-9.0))/(horizonValue/3.0)**4 / 2.
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    else:
        const = 0.5

    for x, y in xy_values:

        if norm(x-y) < horizonValue:
            if kernelType == 'constant':
                ref = const
            elif kernelType == 'inverseDistance':
                ref = const/norm(x-y)
            elif kernelType == 'Gaussian':
                ref = const*exp(-(3.*norm(x-y)/horizonValue)**2)
            else:
                raise NotImplementedError()
        else:
            ref = 0.
        assert np.isclose(kernel(x, y), ref)


@pytest.fixture(scope='module', params=[
    (1, fractionalOrderFactory('const', 0.25), np.inf, True),
    (1, fractionalOrderFactory('const', 0.25), np.inf, False),
    (1, fractionalOrderFactory('const', 0.25), 0.5, True),
    (1, fractionalOrderFactory('const', 0.25), 0.5, False),
    (1, fractionalOrderFactory('const', 0.75), np.inf, True),
    (1, fractionalOrderFactory('const', 0.75), np.inf, False),
    (1, fractionalOrderFactory('const', 0.75), 0.5, True),
    (1, fractionalOrderFactory('const', 0.75), 0.5, False),
    (1, fractionalOrderFactory('varconst', 0.75), np.inf, True),
    (1, fractionalOrderFactory('varconst', 0.75), np.inf, False),
    (1, fractionalOrderFactory('varconst', 0.75), 0.5, True),
    (1, fractionalOrderFactory('varconst', 0.75), 0.5, False),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, True),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, False),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, True),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, False),
    (2, fractionalOrderFactory('const', 0.25), np.inf, True),
    (2, fractionalOrderFactory('const', 0.25), np.inf, False),
    (2, fractionalOrderFactory('const', 0.25), 0.5, True),
    (2, fractionalOrderFactory('const', 0.25), 0.5, False),
    (2, fractionalOrderFactory('const', 0.75), np.inf, True),
    (2, fractionalOrderFactory('const', 0.75), np.inf, False),
    (2, fractionalOrderFactory('const', 0.75), 0.5, True),
    (2, fractionalOrderFactory('const', 0.75), 0.5, False),
    (2, fractionalOrderFactory('varconst', 0.75), np.inf, True),
    (2, fractionalOrderFactory('varconst', 0.75), np.inf, False),
    (2, fractionalOrderFactory('varconst', 0.75), 0.5, True),
    (2, fractionalOrderFactory('varconst', 0.75), 0.5, False),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, True),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, False),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, True),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, False),
], ids=idfunc)
def fractionalKernelParams(request):
    return request.param


def testFractionalKernel(fractionalKernelParams):
    dim, s, horizon, normalized = fractionalKernelParams
    if dim == 1:
        xy_values = [(np.array([-0.1]), np.array([0.1])),
                     (np.array([0.1]), np.array([-0.1])),
                     (np.array([-0.1]), np.array([0.5]))]
    elif dim == 2:
        xy_values = [(np.array([-0.1, 0.1]), np.array([0.1, 0.2])),
                     (np.array([0.1, 0.1]), np.array([-0.1, 0.2])),
                     (np.array([-0.1, 0.1]), np.array([0.5, 0.2]))]
    else:
        raise NotImplementedError()
    kernel = kernelFactory('fractional', dim=dim, s=s, horizon=horizon, normalized=normalized)
    boundaryKernel = kernel.getBoundaryKernel()
    infHorizonKernel = kernel.getModifiedKernel(horizon=constant(np.inf))
    boundaryKernelInf = infHorizonKernel.getBoundaryKernel()

    for x, y in xy_values:

        sValue = s(x, y)

        if isinstance(s, (constFractionalOrder, variableConstFractionalOrder)):
            assert np.isclose(sValue, s.value)
        elif isinstance(s, smoothedLeftRightFractionalOrder):
            assert np.isclose(sValue, s.sFun(x))
            if x[0] < 0:
                assert np.isclose(sValue, 0.25)
            else:
                assert np.isclose(sValue, 0.75)
        else:
            assert False

        horizonValue = kernel.horizon.value

        if normalized:
            if dim == 1:
                if horizonValue < np.inf:
                    const = (2.-2*sValue) * pow(horizonValue**2, sValue-1.) * 0.5
                else:
                    const = 2.0**(2.0*sValue) * sValue * gamma(sValue+0.5)/sqrt(pi)/gamma(1.0-sValue) * 0.5
            elif dim == 2:
                if horizonValue < np.inf:
                    const = (2.-2*sValue)*pow(horizonValue**2, sValue-1.) * 2./pi * 0.5
                else:
                    const = 2.0**(2.0*sValue) * sValue * gamma(sValue+1.0)/pi/gamma(1.-sValue) * 0.5
        else:
            const = 0.5


        refInf = const/norm(x-y)**(dim+2*sValue)
        ref = refInf if (norm(x-y) < horizonValue) else 0.

        assert np.isclose(kernel(x, y), ref)

        # test boundary kernel, do not change horizon
        assert np.isclose(boundaryKernel(x, y), ref*norm(x-y)/sValue)

        # test kernel with infinite horizon
        assert np.isclose(infHorizonKernel(x, y), refInf)

        if (horizonValue < np.inf) and norm(x-y) > horizonValue:
            assert not np.isclose(ref, refInf)
            assert np.isclose(ref, 0.)
        else:
            assert np.isclose(ref, refInf)

        # test boundary kernel, infinite horizon
        assert np.isclose(boundaryKernelInf(x, y), refInf*norm(x-y)/sValue)
