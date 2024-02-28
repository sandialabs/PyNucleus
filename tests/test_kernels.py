###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus_nl import fractionalOrderFactory, kernelFactory, twoPointFunctionFactory
from PyNucleus_nl.twoPointFunctions import constantTwoPoint
from PyNucleus_nl.fractionalOrders import (constFractionalOrder,
                                           variableConstFractionalOrder,
                                           constantNonSymFractionalOrder,
                                           smoothedLeftRightFractionalOrder,
                                           feFractionalOrder)
from PyNucleus_fem.functions import constant
from scipy.special import gamma, erf, digamma, gammaincc, polygamma
from numpy import log
from numpy import pi, exp, sqrt
from numpy.linalg import norm
import pytest


def idfuncIntegrable(param):
    dim, kernelType, horizon, normalized = param
    return f'dim{dim}-kernelType{kernelType}horizon{horizon}-normalized{normalized}'


@pytest.fixture(scope='module', params=[
    # 1d kernels
    (1, 'constant', 0.5, True),
    (1, 'constant', 0.5, False),
    (1, 'inverseDistance', 0.5, True),
    (1, 'inverseDistance', 0.5, False),
    (1, 'Gaussian', 0.5, True),
    (1, 'Gaussian', 0.5, False),
    # 2d kernels
    (2, 'constant', 0.5, True),
    (2, 'constant', 0.5, False),
    (2, 'inverseDistance', 0.5, True),
    (2, 'inverseDistance', 0.5, False),
    (2, 'Gaussian', 0.5, True),
    (2, 'Gaussian', 0.5, False),
], ids=idfuncIntegrable)
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
    infHorizonKernel = kernel.getModifiedKernel(horizon=constant(np.inf))
    boundaryKernelInf = infHorizonKernel.getBoundaryKernel()

    horizonValue = kernel.horizon.value

    if normalized:
        if kernelType == 'constant':
            if dim == 1:
                const = 3/horizonValue**3 * 0.5
            elif dim == 2:
                const = 8./pi/horizonValue**4 * 0.5
            else:
                raise NotImplementedError()
        elif kernelType == 'inverseDistance':
            if dim == 1:
                const = 2./horizonValue**2 * 0.5
            elif dim == 2:
                const = 6./pi/horizonValue**3 * 0.5
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

        if kernelType == 'constant':
            refInf = const
        elif kernelType == 'inverseDistance':
            refInf = const/norm(x-y)
        elif kernelType == 'Gaussian':
            invD = (3/horizonValue)**2
            refInf = const*exp(-invD*norm(x-y)**2)
        else:
            raise NotImplementedError()

        if norm(x-y) < horizonValue:
            ref = refInf
        else:
            ref = 0.

        if kernelType == 'constant':
            refBoundary = refInf*(-1/dim)
        elif kernelType == 'inverseDistance':
            if dim == 1:
                refBoundary = refInf*(-log(norm(x-y)))
            else:
                refBoundary = refInf*(-1/(dim-1))
        elif kernelType == 'Gaussian':
            refBoundary = const*0.5*(invD*norm(x-y)**2)**(-dim/2) * gamma(dim/2) * gammaincc(dim/2, invD*norm(x-y)**2)
        else:
            raise NotImplementedError()

        # test kernel
        assert np.isclose(kernel(x, y), ref), (kernel(x, y), ref)

        # test boundary kernel
        assert np.isclose(boundaryKernelInf(x, y), 2*refBoundary*norm(x-y)), (boundaryKernelInf(x, y), 2*refBoundary*norm(x-y))

        # test that div_y (boundaryKernelInf(x,y) (x-y)/norm(x-y)) == 2*infHorizonKernel(x,y)
        eps = 1e-8
        div_fd = 0.
        for i in range(dim):
            yShifted = y.copy()
            yShifted[i] += eps
            div_fd += (boundaryKernelInf(x, yShifted) * (x-yShifted)[i]/norm(x-yShifted) - boundaryKernelInf(x, y) * (x-y)[i]/norm(x-y))/eps
        assert np.isclose(div_fd, 2*infHorizonKernel(x, y)), (div_fd, 2*infHorizonKernel(x, y))


def idfuncFractional(param):
    dim, s, horizon, normalized, phi, derivative = param
    return f'dim{dim}-s{s}-horizon{horizon}-normalized{normalized}-phi{phi}-derivative{derivative}'


from PyNucleus_fem import meshFactory, dofmapFactory
mesh1d = meshFactory('interval', a=-1, b=1, hTarget=1e-2)
dm1d = dofmapFactory('P1', mesh1d, -1)
mesh2d = meshFactory('disc', hTarget=1e-1, n=8)
dm2d = dofmapFactory('P1', mesh2d, -1)


@pytest.fixture(scope='module', params=[
    # 1d kernels
    (1, fractionalOrderFactory('const', 0.25), np.inf, True, None, 0),
    (1, fractionalOrderFactory('const', 0.25), np.inf, False, None, 0),
    (1, fractionalOrderFactory('const', 0.25), 0.5, True, None, 0),
    (1, fractionalOrderFactory('const', 0.25), 0.5, False, None, 0),
    (1, fractionalOrderFactory('const', 0.75), np.inf, True, None, 0),
    (1, fractionalOrderFactory('const', 0.75), np.inf, False, None, 0),
    (1, fractionalOrderFactory('const', 0.75), 0.5, True, None, 0),
    (1, fractionalOrderFactory('const', 0.75), 0.5, False, None, 0),
    (1, fractionalOrderFactory('varconst', 0.25), np.inf, True, None, 0),
    (1, fractionalOrderFactory('varconst', 0.25), np.inf, False, None, 0),
    (1, fractionalOrderFactory('varconst', 0.25), 0.5, True, None, 0),
    (1, fractionalOrderFactory('varconst', 0.25), 0.5, False, None, 0),
    (1, fractionalOrderFactory('varconst', 0.75), np.inf, True, None, 0),
    (1, fractionalOrderFactory('varconst', 0.75), np.inf, False, None, 0),
    (1, fractionalOrderFactory('varconst', 0.75), 0.5, True, None, 0),
    (1, fractionalOrderFactory('varconst', 0.75), 0.5, False, None, 0),
    (1, fractionalOrderFactory('constantNonSym', 0.25), np.inf, True, None, 0),
    (1, fractionalOrderFactory('constantNonSym', 0.25), np.inf, False, None, 0),
    (1, fractionalOrderFactory('constantNonSym', 0.25), 0.5, True, None, 0),
    (1, fractionalOrderFactory('constantNonSym', 0.25), 0.5, False, None, 0),
    (1, fractionalOrderFactory('constantNonSym', 0.75), np.inf, True, None, 0),
    (1, fractionalOrderFactory('constantNonSym', 0.75), np.inf, False, None, 0),
    (1, fractionalOrderFactory('constantNonSym', 0.75), 0.5, True, None, 0),
    (1, fractionalOrderFactory('constantNonSym', 0.75), 0.5, False, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, True, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, False, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, True, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, False, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, True, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, False, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, True, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, False, None, 0),
    # discretized fractional order
    (1, fractionalOrderFactory('const', 0.25, dm=dm1d), np.inf, True, None, 0),
    (1, fractionalOrderFactory('const', 0.25, dm=dm1d), np.inf, False, None, 0),
    (1, fractionalOrderFactory('const', 0.25, dm=dm1d), 0.5, True, None, 0),
    (1, fractionalOrderFactory('const', 0.25, dm=dm1d), 0.5, False, None, 0),
    (1, fractionalOrderFactory('const', 0.75, dm=dm1d), np.inf, True, None, 0),
    (1, fractionalOrderFactory('const', 0.75, dm=dm1d), np.inf, False, None, 0),
    (1, fractionalOrderFactory('const', 0.75, dm=dm1d), 0.5, True, None, 0),
    (1, fractionalOrderFactory('const', 0.75, dm=dm1d), 0.5, False, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25, dm=dm1d), np.inf, True, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25, dm=dm1d), np.inf, False, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25, dm=dm1d), 0.5, True, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25, dm=dm1d), 0.5, False, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75, dm=dm1d), np.inf, True, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75, dm=dm1d), np.inf, False, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75, dm=dm1d), 0.5, True, None, 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75, dm=dm1d), 0.5, False, None, 0),
    # now with a trivial phi
    (1, fractionalOrderFactory('const', 0.25), np.inf, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('const', 0.25), np.inf, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('const', 0.25), 0.5, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('const', 0.25), 0.5, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('const', 0.75), np.inf, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('const', 0.75), np.inf, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('const', 0.75), 0.5, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('const', 0.75), 0.5, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantSym', 0.25), np.inf, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantSym', 0.25), np.inf, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantSym', 0.25), 0.5, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantSym', 0.25), 0.5, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantSym', 0.75), np.inf, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantSym', 0.75), np.inf, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantSym', 0.75), 0.5, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantSym', 0.75), 0.5, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantNonSym', 0.25), np.inf, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantNonSym', 0.25), np.inf, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantNonSym', 0.25), 0.5, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantNonSym', 0.25), 0.5, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantNonSym', 0.75), np.inf, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantNonSym', 0.75), np.inf, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantNonSym', 0.75), 0.5, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('constantNonSym', 0.75), 0.5, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, False, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, True, twoPointFunctionFactory('const', 2.), 0),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, False, twoPointFunctionFactory('const', 2.), 0),
    # first derivative wrt s
    (1, fractionalOrderFactory('const', 0.25), np.inf, True, None, 1),
    (1, fractionalOrderFactory('const', 0.25), np.inf, False, None, 1),
    (1, fractionalOrderFactory('const', 0.25), 0.5, True, None, 1),
    (1, fractionalOrderFactory('const', 0.25), 0.5, False, None, 1),
    (1, fractionalOrderFactory('const', 0.75), np.inf, True, None, 1),
    (1, fractionalOrderFactory('const', 0.75), np.inf, False, None, 1),
    (1, fractionalOrderFactory('const', 0.75), 0.5, True, None, 1),
    (1, fractionalOrderFactory('const', 0.75), 0.5, False, None, 1),
    (1, fractionalOrderFactory('constantSym', 0.25), np.inf, True, None, 1),
    (1, fractionalOrderFactory('constantSym', 0.25), np.inf, False, None, 1),
    (1, fractionalOrderFactory('constantSym', 0.25), 0.5, True, None, 1),
    (1, fractionalOrderFactory('constantSym', 0.25), 0.5, False, None, 1),
    (1, fractionalOrderFactory('constantSym', 0.75), np.inf, True, None, 1),
    (1, fractionalOrderFactory('constantSym', 0.75), np.inf, False, None, 1),
    (1, fractionalOrderFactory('constantSym', 0.75), 0.5, True, None, 1),
    (1, fractionalOrderFactory('constantSym', 0.75), 0.5, False, None, 1),
    (1, fractionalOrderFactory('constantNonSym', 0.25), np.inf, True, None, 1),
    (1, fractionalOrderFactory('constantNonSym', 0.25), np.inf, False, None, 1),
    (1, fractionalOrderFactory('constantNonSym', 0.25), 0.5, True, None, 1),
    (1, fractionalOrderFactory('constantNonSym', 0.25), 0.5, False, None, 1),
    (1, fractionalOrderFactory('constantNonSym', 0.75), np.inf, True, None, 1),
    (1, fractionalOrderFactory('constantNonSym', 0.75), np.inf, False, None, 1),
    (1, fractionalOrderFactory('constantNonSym', 0.75), 0.5, True, None, 1),
    (1, fractionalOrderFactory('constantNonSym', 0.75), 0.5, False, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, True, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, False, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, True, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, False, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, True, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, False, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, True, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, False, None, 1),
    # second derivative wrt s
    (1, fractionalOrderFactory('const', 0.25), np.inf, True, None, 2),
    (1, fractionalOrderFactory('const', 0.25), np.inf, False, None, 2),
    (1, fractionalOrderFactory('const', 0.25), 0.5, True, None, 2),
    (1, fractionalOrderFactory('const', 0.25), 0.5, False, None, 2),
    (1, fractionalOrderFactory('const', 0.75), np.inf, True, None, 2),
    (1, fractionalOrderFactory('const', 0.75), np.inf, False, None, 2),
    (1, fractionalOrderFactory('const', 0.75), 0.5, True, None, 2),
    (1, fractionalOrderFactory('const', 0.75), 0.5, False, None, 2),
    (1, fractionalOrderFactory('constantSym', 0.25), np.inf, True, None, 2),
    (1, fractionalOrderFactory('constantSym', 0.25), np.inf, False, None, 2),
    (1, fractionalOrderFactory('constantSym', 0.25), 0.5, True, None, 2),
    (1, fractionalOrderFactory('constantSym', 0.25), 0.5, False, None, 2),
    (1, fractionalOrderFactory('constantSym', 0.75), np.inf, True, None, 2),
    (1, fractionalOrderFactory('constantSym', 0.75), np.inf, False, None, 2),
    (1, fractionalOrderFactory('constantSym', 0.75), 0.5, True, None, 2),
    (1, fractionalOrderFactory('constantSym', 0.75), 0.5, False, None, 2),
    (1, fractionalOrderFactory('constantNonSym', 0.25), np.inf, True, None, 2),
    (1, fractionalOrderFactory('constantNonSym', 0.25), np.inf, False, None, 2),
    (1, fractionalOrderFactory('constantNonSym', 0.25), 0.5, True, None, 2),
    (1, fractionalOrderFactory('constantNonSym', 0.25), 0.5, False, None, 2),
    (1, fractionalOrderFactory('constantNonSym', 0.75), np.inf, True, None, 2),
    (1, fractionalOrderFactory('constantNonSym', 0.75), np.inf, False, None, 2),
    (1, fractionalOrderFactory('constantNonSym', 0.75), 0.5, True, None, 2),
    (1, fractionalOrderFactory('constantNonSym', 0.75), 0.5, False, None, 2),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, True, None, 2),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, False, None, 2),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, True, None, 2),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, False, None, 2),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, True, None, 2),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, False, None, 2),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, True, None, 2),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, False, None, 2),
    # discretized fractional order
    (1, fractionalOrderFactory('const', 0.25, dm=dm1d), np.inf, True, None, 1),
    (1, fractionalOrderFactory('const', 0.25, dm=dm1d), np.inf, False, None, 1),
    (1, fractionalOrderFactory('const', 0.25, dm=dm1d), 0.5, True, None, 1),
    (1, fractionalOrderFactory('const', 0.25, dm=dm1d), 0.5, False, None, 1),
    (1, fractionalOrderFactory('const', 0.75, dm=dm1d), np.inf, True, None, 1),
    (1, fractionalOrderFactory('const', 0.75, dm=dm1d), np.inf, False, None, 1),
    (1, fractionalOrderFactory('const', 0.75, dm=dm1d), 0.5, True, None, 1),
    (1, fractionalOrderFactory('const', 0.75, dm=dm1d), 0.5, False, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25, dm=dm1d), np.inf, True, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25, dm=dm1d), np.inf, False, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25, dm=dm1d), 0.5, True, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25, dm=dm1d), 0.5, False, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75, dm=dm1d), np.inf, True, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75, dm=dm1d), np.inf, False, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75, dm=dm1d), 0.5, True, None, 1),
    (1, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75, dm=dm1d), 0.5, False, None, 1),
    ##################################################
    # 2d kernels
    (2, fractionalOrderFactory('const', 0.25), np.inf, True, None, 0),
    (2, fractionalOrderFactory('const', 0.25), np.inf, False, None, 0),
    (2, fractionalOrderFactory('const', 0.25), 0.5, True, None, 0),
    (2, fractionalOrderFactory('const', 0.25), 0.5, False, None, 0),
    (2, fractionalOrderFactory('const', 0.75), np.inf, True, None, 0),
    (2, fractionalOrderFactory('const', 0.75), np.inf, False, None, 0),
    (2, fractionalOrderFactory('const', 0.75), 0.5, True, None, 0),
    (2, fractionalOrderFactory('const', 0.75), 0.5, False, None, 0),
    (2, fractionalOrderFactory('constantSym', 0.25), np.inf, True, None, 0),
    (2, fractionalOrderFactory('constantSym', 0.25), np.inf, False, None, 0),
    (2, fractionalOrderFactory('constantSym', 0.25), 0.5, True, None, 0),
    (2, fractionalOrderFactory('constantSym', 0.25), 0.5, False, None, 0),
    (2, fractionalOrderFactory('constantSym', 0.75), np.inf, True, None, 0),
    (2, fractionalOrderFactory('constantSym', 0.75), np.inf, False, None, 0),
    (2, fractionalOrderFactory('constantSym', 0.75), 0.5, True, None, 0),
    (2, fractionalOrderFactory('constantSym', 0.75), 0.5, False, None, 0),
    (2, fractionalOrderFactory('constantNonSym', 0.25), np.inf, True, None, 0),
    (2, fractionalOrderFactory('constantNonSym', 0.25), np.inf, False, None, 0),
    (2, fractionalOrderFactory('constantNonSym', 0.25), 0.5, True, None, 0),
    (2, fractionalOrderFactory('constantNonSym', 0.25), 0.5, False, None, 0),
    (2, fractionalOrderFactory('constantNonSym', 0.75), np.inf, True, None, 0),
    (2, fractionalOrderFactory('constantNonSym', 0.75), np.inf, False, None, 0),
    (2, fractionalOrderFactory('constantNonSym', 0.75), 0.5, True, None, 0),
    (2, fractionalOrderFactory('constantNonSym', 0.75), 0.5, False, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, True, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, False, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, True, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, False, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, True, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, False, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, True, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, False, None, 0),
    # first derivative wrt s
    (2, fractionalOrderFactory('const', 0.25), np.inf, True, None, 1),
    (2, fractionalOrderFactory('const', 0.25), np.inf, False, None, 1),
    (2, fractionalOrderFactory('const', 0.25), 0.5, True, None, 1),
    (2, fractionalOrderFactory('const', 0.25), 0.5, False, None, 1),
    (2, fractionalOrderFactory('const', 0.75), np.inf, True, None, 1),
    (2, fractionalOrderFactory('const', 0.75), np.inf, False, None, 1),
    (2, fractionalOrderFactory('const', 0.75), 0.5, True, None, 1),
    (2, fractionalOrderFactory('const', 0.75), 0.5, False, None, 1),
    (2, fractionalOrderFactory('constantSym', 0.25), np.inf, True, None, 1),
    (2, fractionalOrderFactory('constantSym', 0.25), np.inf, False, None, 1),
    (2, fractionalOrderFactory('constantSym', 0.25), 0.5, True, None, 1),
    (2, fractionalOrderFactory('constantSym', 0.25), 0.5, False, None, 1),
    (2, fractionalOrderFactory('constantSym', 0.75), np.inf, True, None, 1),
    (2, fractionalOrderFactory('constantSym', 0.75), np.inf, False, None, 1),
    (2, fractionalOrderFactory('constantSym', 0.75), 0.5, True, None, 1),
    (2, fractionalOrderFactory('constantSym', 0.75), 0.5, False, None, 1),
    (2, fractionalOrderFactory('constantNonSym', 0.25), np.inf, True, None, 1),
    (2, fractionalOrderFactory('constantNonSym', 0.25), np.inf, False, None, 1),
    (2, fractionalOrderFactory('constantNonSym', 0.25), 0.5, True, None, 1),
    (2, fractionalOrderFactory('constantNonSym', 0.25), 0.5, False, None, 1),
    (2, fractionalOrderFactory('constantNonSym', 0.75), np.inf, True, None, 1),
    (2, fractionalOrderFactory('constantNonSym', 0.75), np.inf, False, None, 1),
    (2, fractionalOrderFactory('constantNonSym', 0.75), 0.5, True, None, 1),
    (2, fractionalOrderFactory('constantNonSym', 0.75), 0.5, False, None, 1),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, True, None, 1),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, False, None, 1),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, True, None, 1),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, False, None, 1),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, True, None, 1),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, False, None, 1),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, True, None, 1),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, False, None, 1),
    # second derivative wrt s
    (2, fractionalOrderFactory('const', 0.25), np.inf, True, None, 2),
    (2, fractionalOrderFactory('const', 0.25), np.inf, False, None, 2),
    (2, fractionalOrderFactory('const', 0.25), 0.5, True, None, 2),
    (2, fractionalOrderFactory('const', 0.25), 0.5, False, None, 2),
    (2, fractionalOrderFactory('const', 0.75), np.inf, True, None, 2),
    (2, fractionalOrderFactory('const', 0.75), np.inf, False, None, 2),
    (2, fractionalOrderFactory('const', 0.75), 0.5, True, None, 2),
    (2, fractionalOrderFactory('const', 0.75), 0.5, False, None, 2),
    (2, fractionalOrderFactory('constantSym', 0.25), np.inf, True, None, 2),
    (2, fractionalOrderFactory('constantSym', 0.25), np.inf, False, None, 2),
    (2, fractionalOrderFactory('constantSym', 0.25), 0.5, True, None, 2),
    (2, fractionalOrderFactory('constantSym', 0.25), 0.5, False, None, 2),
    (2, fractionalOrderFactory('constantSym', 0.75), np.inf, True, None, 2),
    (2, fractionalOrderFactory('constantSym', 0.75), np.inf, False, None, 2),
    (2, fractionalOrderFactory('constantSym', 0.75), 0.5, True, None, 2),
    (2, fractionalOrderFactory('constantSym', 0.75), 0.5, False, None, 2),
    (2, fractionalOrderFactory('constantNonSym', 0.25), np.inf, True, None, 2),
    (2, fractionalOrderFactory('constantNonSym', 0.25), np.inf, False, None, 2),
    (2, fractionalOrderFactory('constantNonSym', 0.25), 0.5, True, None, 2),
    (2, fractionalOrderFactory('constantNonSym', 0.25), 0.5, False, None, 2),
    (2, fractionalOrderFactory('constantNonSym', 0.75), np.inf, True, None, 2),
    (2, fractionalOrderFactory('constantNonSym', 0.75), np.inf, False, None, 2),
    (2, fractionalOrderFactory('constantNonSym', 0.75), 0.5, True, None, 2),
    (2, fractionalOrderFactory('constantNonSym', 0.75), 0.5, False, None, 2),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, True, None, 2),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, False, None, 2),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, True, None, 2),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, False, None, 2),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, True, None, 2),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, False, None, 2),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, True, None, 2),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, False, None, 2),
    # discretized fractional order
    (2, fractionalOrderFactory('const', 0.25, dm=dm2d), np.inf, True, None, 0),
    (2, fractionalOrderFactory('const', 0.25, dm=dm2d), np.inf, False, None, 0),
    (2, fractionalOrderFactory('const', 0.25, dm=dm2d), 0.5, True, None, 0),
    (2, fractionalOrderFactory('const', 0.25, dm=dm2d), 0.5, False, None, 0),
    (2, fractionalOrderFactory('const', 0.75, dm=dm2d), np.inf, True, None, 0),
    (2, fractionalOrderFactory('const', 0.75, dm=dm2d), np.inf, False, None, 0),
    (2, fractionalOrderFactory('const', 0.75, dm=dm2d), 0.5, True, None, 0),
    (2, fractionalOrderFactory('const', 0.75, dm=dm2d), 0.5, False, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25, dm=dm2d), np.inf, True, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25, dm=dm2d), np.inf, False, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25, dm=dm2d), 0.5, True, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25, dm=dm2d), 0.5, False, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75, dm=dm2d), np.inf, True, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75, dm=dm2d), np.inf, False, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75, dm=dm2d), 0.5, True, None, 0),
    (2, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75, dm=dm2d), 0.5, False, None, 0),
    ##################################################
    # 3d kernels
    (3, fractionalOrderFactory('const', 0.25), np.inf, True, None, 0),
    (3, fractionalOrderFactory('const', 0.25), np.inf, False, None, 0),
    (3, fractionalOrderFactory('const', 0.25), 0.5, True, None, 0),
    (3, fractionalOrderFactory('const', 0.25), 0.5, False, None, 0),
    (3, fractionalOrderFactory('const', 0.75), np.inf, True, None, 0),
    (3, fractionalOrderFactory('const', 0.75), np.inf, False, None, 0),
    (3, fractionalOrderFactory('const', 0.75), 0.5, True, None, 0),
    (3, fractionalOrderFactory('const', 0.75), 0.5, False, None, 0),
    (3, fractionalOrderFactory('constantSym', 0.25), np.inf, True, None, 0),
    (3, fractionalOrderFactory('constantSym', 0.25), np.inf, False, None, 0),
    (3, fractionalOrderFactory('constantSym', 0.25), 0.5, True, None, 0),
    (3, fractionalOrderFactory('constantSym', 0.25), 0.5, False, None, 0),
    (3, fractionalOrderFactory('constantSym', 0.75), np.inf, True, None, 0),
    (3, fractionalOrderFactory('constantSym', 0.75), np.inf, False, None, 0),
    (3, fractionalOrderFactory('constantSym', 0.75), 0.5, True, None, 0),
    (3, fractionalOrderFactory('constantSym', 0.75), 0.5, False, None, 0),
    (3, fractionalOrderFactory('constantNonSym', 0.25), np.inf, True, None, 0),
    (3, fractionalOrderFactory('constantNonSym', 0.25), np.inf, False, None, 0),
    (3, fractionalOrderFactory('constantNonSym', 0.25), 0.5, True, None, 0),
    (3, fractionalOrderFactory('constantNonSym', 0.25), 0.5, False, None, 0),
    (3, fractionalOrderFactory('constantNonSym', 0.75), np.inf, True, None, 0),
    (3, fractionalOrderFactory('constantNonSym', 0.75), np.inf, False, None, 0),
    (3, fractionalOrderFactory('constantNonSym', 0.75), 0.5, True, None, 0),
    (3, fractionalOrderFactory('constantNonSym', 0.75), 0.5, False, None, 0),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, True, None, 0),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, False, None, 0),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, True, None, 0),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, False, None, 0),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, True, None, 0),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, False, None, 0),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, True, None, 0),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, False, None, 0),
    # first derivative wrt s
    (3, fractionalOrderFactory('const', 0.25), np.inf, True, None, 1),
    (3, fractionalOrderFactory('const', 0.25), np.inf, False, None, 1),
    (3, fractionalOrderFactory('const', 0.25), 0.5, True, None, 1),
    (3, fractionalOrderFactory('const', 0.25), 0.5, False, None, 1),
    (3, fractionalOrderFactory('const', 0.75), np.inf, True, None, 1),
    (3, fractionalOrderFactory('const', 0.75), np.inf, False, None, 1),
    (3, fractionalOrderFactory('const', 0.75), 0.5, True, None, 1),
    (3, fractionalOrderFactory('const', 0.75), 0.5, False, None, 1),
    (3, fractionalOrderFactory('constantSym', 0.25), np.inf, True, None, 1),
    (3, fractionalOrderFactory('constantSym', 0.25), np.inf, False, None, 1),
    (3, fractionalOrderFactory('constantSym', 0.25), 0.5, True, None, 1),
    (3, fractionalOrderFactory('constantSym', 0.25), 0.5, False, None, 1),
    (3, fractionalOrderFactory('constantSym', 0.75), np.inf, True, None, 1),
    (3, fractionalOrderFactory('constantSym', 0.75), np.inf, False, None, 1),
    (3, fractionalOrderFactory('constantSym', 0.75), 0.5, True, None, 1),
    (3, fractionalOrderFactory('constantSym', 0.75), 0.5, False, None, 1),
    (3, fractionalOrderFactory('constantNonSym', 0.25), np.inf, True, None, 1),
    (3, fractionalOrderFactory('constantNonSym', 0.25), np.inf, False, None, 1),
    (3, fractionalOrderFactory('constantNonSym', 0.25), 0.5, True, None, 1),
    (3, fractionalOrderFactory('constantNonSym', 0.25), 0.5, False, None, 1),
    (3, fractionalOrderFactory('constantNonSym', 0.75), np.inf, True, None, 1),
    (3, fractionalOrderFactory('constantNonSym', 0.75), np.inf, False, None, 1),
    (3, fractionalOrderFactory('constantNonSym', 0.75), 0.5, True, None, 1),
    (3, fractionalOrderFactory('constantNonSym', 0.75), 0.5, False, None, 1),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, True, None, 1),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, False, None, 1),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, True, None, 1),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, False, None, 1),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, True, None, 1),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, False, None, 1),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, True, None, 1),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, False, None, 1),
    # second derivative wrt s
    (3, fractionalOrderFactory('const', 0.25), np.inf, True, None, 2),
    (3, fractionalOrderFactory('const', 0.25), np.inf, False, None, 2),
    (3, fractionalOrderFactory('const', 0.25), 0.5, True, None, 2),
    (3, fractionalOrderFactory('const', 0.25), 0.5, False, None, 2),
    (3, fractionalOrderFactory('const', 0.75), np.inf, True, None, 2),
    (3, fractionalOrderFactory('const', 0.75), np.inf, False, None, 2),
    (3, fractionalOrderFactory('const', 0.75), 0.5, True, None, 2),
    (3, fractionalOrderFactory('const', 0.75), 0.5, False, None, 2),
    (3, fractionalOrderFactory('constantSym', 0.25), np.inf, True, None, 2),
    (3, fractionalOrderFactory('constantSym', 0.25), np.inf, False, None, 2),
    (3, fractionalOrderFactory('constantSym', 0.25), 0.5, True, None, 2),
    (3, fractionalOrderFactory('constantSym', 0.25), 0.5, False, None, 2),
    (3, fractionalOrderFactory('constantSym', 0.75), np.inf, True, None, 2),
    (3, fractionalOrderFactory('constantSym', 0.75), np.inf, False, None, 2),
    (3, fractionalOrderFactory('constantSym', 0.75), 0.5, True, None, 2),
    (3, fractionalOrderFactory('constantSym', 0.75), 0.5, False, None, 2),
    (3, fractionalOrderFactory('constantNonSym', 0.25), np.inf, True, None, 2),
    (3, fractionalOrderFactory('constantNonSym', 0.25), np.inf, False, None, 2),
    (3, fractionalOrderFactory('constantNonSym', 0.25), 0.5, True, None, 2),
    (3, fractionalOrderFactory('constantNonSym', 0.25), 0.5, False, None, 2),
    (3, fractionalOrderFactory('constantNonSym', 0.75), np.inf, True, None, 2),
    (3, fractionalOrderFactory('constantNonSym', 0.75), np.inf, False, None, 2),
    (3, fractionalOrderFactory('constantNonSym', 0.75), 0.5, True, None, 2),
    (3, fractionalOrderFactory('constantNonSym', 0.75), 0.5, False, None, 2),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, True, None, 2),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), np.inf, False, None, 2),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, True, None, 2),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.75, 0.25), 0.5, False, None, 2),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, True, None, 2),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), np.inf, False, None, 2),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, True, None, 2),
    (3, fractionalOrderFactory('twoDomainNonSym', 0.25, 0.75), 0.5, False, None, 2),
], ids=idfuncFractional)
def fractionalKernelParams(request):
    return request.param


def testFractionalKernel(fractionalKernelParams):
    dim, s, horizon, normalized, phi, derivative = fractionalKernelParams
    if dim == 1:
        xy_values = [(np.array([-0.1]), np.array([0.1])),
                     (np.array([0.1]), np.array([-0.1])),
                     (np.array([-0.1]), np.array([0.5]))]
    elif dim == 2:
        xy_values = [(np.array([-0.1, 0.1]), np.array([0.1, 0.2])),
                     (np.array([0.1, 0.1]), np.array([-0.1, 0.2])),
                     (np.array([-0.1, 0.1]), np.array([0.5, 0.2]))]
    elif dim == 3:
        xy_values = [(np.array([-0.1, 0.1, 0.1]), np.array([0.1, 0.2, 0.2])),
                     (np.array([0.1, 0.1, 0.1]), np.array([-0.1, 0.2, 0.2])),
                     (np.array([-0.1, 0.1, 0.1]), np.array([0.5, 0.2, 0.2]))]
    else:
        raise NotImplementedError()

    kernel = kernelFactory('fractional', dim=dim, s=s, horizon=horizon, normalized=normalized, phi=phi)
    if derivative == 1:
        kernel = kernel.getDerivativeKernel()
    elif derivative == 2:
        kernel = kernel.getHessianKernel()
    boundaryKernel = kernel.getBoundaryKernel()
    infHorizonKernel = kernel.getModifiedKernel(horizon=constant(np.inf))
    boundaryKernelInf = infHorizonKernel.getBoundaryKernel()

    print(kernel)
    print(boundaryKernel)
    print(infHorizonKernel)
    print(boundaryKernelInf)

    for x, y in xy_values:

        sValue = s(x, y)
        if phi is not None:
            phiValue = phi(x, y)
        else:
            phiValue = 1.

        if isinstance(s, (constFractionalOrder, variableConstFractionalOrder, constantNonSymFractionalOrder)):
            assert np.isclose(sValue, s.value)
        elif isinstance(s, smoothedLeftRightFractionalOrder):
            assert np.isclose(sValue, s.sFun(x))
            if x[0] < 0:
                assert np.isclose(sValue, s.sFun.sl)
            else:
                assert np.isclose(sValue, s.sFun.sr)
        elif isinstance(s, feFractionalOrder):
            pass
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
            elif dim == 3:
                if horizonValue < np.inf:
                    const = (2.-2*sValue)*pow(horizonValue**2, sValue-1.) * 3*gamma(dim/2)/pow(pi, dim/2) * 0.5
                else:
                    const = 2.0**(2.0*sValue) * sValue * gamma(sValue+1.5)/pow(pi, 1.5)/gamma(1.-sValue) * 0.5
        else:
            const = 0.5

        if derivative == 0:
            refInf = const/norm(x-y)**(dim+2*sValue) * phiValue
        elif derivative == 1:
            refInf = const/norm(x-y)**(dim+2*sValue) * phiValue
            if normalized:
                if horizonValue < np.inf:
                    refInf *= (-log(norm(x-y)**2/horizonValue**2) - 1./(1.-sValue))
                else:
                    refInf *= (-log(norm(x-y)**2/4) + digamma(sValue+0.5*dim) + digamma(-sValue))
            else:
                refInf *= (-log(norm(x-y)**2))
        elif derivative == 2:
            refInf = const/norm(x-y)**(dim+2*sValue) * phiValue
            if normalized:
                if horizonValue < np.inf:
                    d2Cds2 = log(horizonValue**2)**2 - 2/(1-sValue)*log(horizonValue**2)
                    dCds = -1./(1.-sValue)+log(horizonValue**2)
                else:
                    d2Cds2 = (log(4.)+digamma(sValue+0.5*dim)+digamma(-sValue))**2 + polygamma(1, sValue+0.5*dim) - polygamma(1, -sValue)
                    dCds = log(4.)+digamma(sValue+0.5*dim)+digamma(-sValue)
            else:
                d2Cds2 = 0.
                dCds = 0.
            refInf *= (d2Cds2 - 2*dCds*log(norm(x-y)**2) + log(norm(x-y)**2)**2)
        else:
            raise NotImplementedError()
        ref = refInf if (norm(x-y) < horizonValue) else 0.
        if derivative == 1:
            refInfBoundary = const/norm(x-y)**(dim+2*sValue) * phiValue
            if normalized:
                if horizonValue == np.inf:
                    refInfBoundary *= (-log(norm(x-y)**2/4) + digamma(sValue+0.5*dim) + digamma(1.0-sValue))
                else:
                    refInfBoundary *= (-log(norm(x-y)**2/horizonValue**2) - 1./(1.-sValue) - 1./sValue)
            else:
                refInfBoundary *= (-log(norm(x-y)**2) - 1.0/sValue)
        elif derivative == 2:
            refInfBoundary = const/norm(x-y)**(dim+2*sValue) * phiValue
            refInfBoundary *= (d2Cds2 + 2./sValue**2 - 2.*dCds/sValue + (-2*dCds+2./sValue)*log(norm(x-y)**2) + log(norm(x-y)**2)**2)
        else:
            refInfBoundary = refInf
        refBoundary = refInfBoundary if (norm(x-y) < horizonValue) else 0.

        # test the kernel with potentially finite horizon
        assert np.isclose(kernel(x, y), ref), (kernel(x, y), ref)

        # test kernel with infinite horizon
        assert np.isclose(infHorizonKernel(x, y), refInf), (infHorizonKernel(x, y), refInf)

        if phi is not None and not isinstance(phi, constantTwoPoint):
            # Not implemented, as the boundary kernel will depend on phi
            continue

        # test boundary kernel with potentially finite horizon
        assert np.isclose(boundaryKernel(x, y), refBoundary*norm(x-y)/sValue), (boundaryKernel(x, y), refBoundary*norm(x-y)/sValue)

        # test boundary kernel with infinite horizon
        assert np.isclose(boundaryKernelInf(x, y), refInfBoundary*norm(x-y)/sValue), (boundaryKernelInf(x, y), refInfBoundary*norm(x-y)/sValue)

        # test that div_y (boundaryKernelInf(x,y) (x-y)/norm(x-y)) == 2*infHorizonKernel(x,y)
        eps = 1e-8
        div_fd = 0.
        for i in range(dim):
            yShifted = y.copy()
            yShifted[i] += eps
            div_fd += (boundaryKernelInf(x, yShifted) * (x-yShifted)[i]/norm(x-yShifted) - boundaryKernelInf(x, y) * (x-y)[i]/norm(x-y))/eps
        assert np.isclose(div_fd, 2*infHorizonKernel(x, y), rtol=1e-3), (div_fd, 2*infHorizonKernel(x, y))

        assert np.isclose(kernel.singularityValue, -dim-2*sValue), (kernel.singularityValue, -dim-2*sValue)
        assert np.isclose(boundaryKernel.singularityValue, 1-dim-2*sValue), (boundaryKernel.singularityValue, 1-dim-2*sValue)
        assert np.isclose(infHorizonKernel.singularityValue, -dim-2*sValue), (infHorizonKernel.singularityValue, -dim-2*sValue)
        assert np.isclose(boundaryKernelInf.singularityValue, 1-dim-2*sValue), (boundaryKernelInf.singularityValue, 1-dim-2*sValue)


from PyNucleus import dofmapFactory, fractionalOrderFactory, kernelFactory, meshFactory, REAL, functionFactory
from PyNucleus_nl import nonlocalBuilder


def test_discrete_s_const():
    """
    Compare operators for kernel with
    s = constantNonSym(0.75)
    and its finite element interpolation.
    """
    mesh = meshFactory('interval', a=-1, b=1, hTarget=1e-2)
    dmS = dofmapFactory('P1', mesh, -1)
    sFun = fractionalOrderFactory('constantNonSym', 0.75)
    sFE = fractionalOrderFactory('constantNonSym', 0.75, dm=dmS)
    kernelFun = kernelFactory('fractional', dim=mesh.dim, s=sFun)
    kernelFE = kernelFactory('fractional', dim=mesh.dim, s=sFE)
    dm = dofmapFactory('P1', mesh, 0)
    A_fun = dm.assembleNonlocal(kernelFun)
    A_fe = dm.assembleNonlocal(kernelFE)
    assert np.absolute((A_fun-A_fe).toarray()/A_fun.toarray()).max() == 0.


def test_discrete_leftRight():
    """
    Compare operators for kernel with
    s = twoDomainNonSym(0.25, 0.75)
    and its finite element interpolation.
    """
    mesh = meshFactory('interval', a=-1, b=1, hTarget=1e-2)
    dmS = dofmapFactory('P1', mesh, -1)
    sFun = fractionalOrderFactory('twoDomainNonSym', sl=0.25, sr=0.75, r=0.3)
    sFE = fractionalOrderFactory('twoDomainNonSym', sl=0.25, sr=0.75, r=0.3, dm=dmS)
    kernelFun = kernelFactory('fractional', dim=mesh.dim, s=sFun)
    kernelFE = kernelFactory('fractional', dim=mesh.dim, s=sFE)
    dm = dofmapFactory('P1', mesh, 0)
    A_fun = dm.assembleNonlocal(kernelFun)
    A_fe = dm.assembleNonlocal(kernelFE)
    print('max rel error', np.absolute((A_fun-A_fe).toarray()/A_fun.toarray()).max())
    x = dm.fromArray(np.random.randn(dm.num_dofs))
    a = x.inner(A_fun*x)
    b = x.inner(A_fe*x)
    print('apply abs error', abs(a-b))
    print('apply rel error', abs(a-b)/a)
    assert abs(a-b)/a < 1e-4
