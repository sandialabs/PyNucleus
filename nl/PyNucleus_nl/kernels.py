###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus_base.myTypes import INDEX, REAL
from PyNucleus_fem.functions import function, constant
from PyNucleus_fem.mesh import meshNd
from . twoPointFunctions import constantTwoPoint, inverseTwoPoint
from . interactionDomains import (interactionDomain,
                                  fullSpace,
                                  ball1_retriangulation,
                                  ball2_retriangulation,
                                  ballInf_retriangulation)
from . fractionalOrders import (fractionalOrderBase,
                                constFractionalOrder,
                                variableConstFractionalOrder,
                                singleVariableUnsymmetricFractionalOrder)
from . kernelNormalization import (constantFractionalLaplacianScaling,
                                   constantFractionalLaplacianScalingDerivative,
                                   variableFractionalLaplacianScaling,
                                   constantIntegrableScaling,
                                   variableIntegrableScaling,
                                   )
from . kernelsCy import (Kernel,
                         FractionalKernel,
                         RangedFractionalKernel,
                         FRACTIONAL,
                         PERIDYNAMIC,
                         LOGINVERSEDISTANCE,
                         GREENS_2D,
                         GREENS_3D,
                         MONOMIAL,
                         getKernelEnum)
from . operatorInterpolation import admissibleSet
import warnings


def _getDim(dim):
    if isinstance(dim, meshNd):
        return dim.dim
    elif isinstance(dim, (INDEX, int)):
        return dim
    else:
        raise NotImplementedError('Dim: {}'.format(dim))


def _getKernelType(kernel):
    if isinstance(kernel, str):
        kType = getKernelEnum(kernel)
    elif isinstance(kernel, int):
        kType = kernel
    else:
        raise NotImplementedError('Kernel type: {}'.format(kernel))
    return kType


def _getFractionalOrder(s):
    if isinstance(s, fractionalOrderBase):
        sFun = s
    elif isinstance(s, admissibleSet):
        sFun = s
    elif isinstance(s, tuple) and len(s) == 2:
        sFun = admissibleSet(s)
    elif isinstance(s, (REAL, float)):
        sFun = constFractionalOrder(s)
    else:
        raise NotImplementedError('Fractional order: {}'.format(s))
    return sFun


def _getHorizon(horizon):
    if isinstance(horizon, function):
        horizonFun = horizon
    elif isinstance(horizon, (REAL, float, int)):
        horizonFun = constant(horizon)
    elif horizon is None:
        horizonFun = constant(np.inf)
    else:
        raise NotImplementedError('Horizon: {}'.format(horizon))
    return horizonFun


def _getInteraction(interaction, horizon):
    if isinstance(interaction, interactionDomain):
        pass
    elif isinstance(horizon, constant) and horizon.value == np.inf:
        interaction = fullSpace()
    elif interaction is None:
        interaction = ball2_retriangulation(horizon)
    elif isinstance(interaction, str):
        if interaction == 'fullSpace':
            interaction = fullSpace()
        elif interaction == 'ball1':
            interaction = ball1_retriangulation(horizon)
        elif interaction == 'ball2':
            interaction = ball2_retriangulation(horizon)
        elif interaction == 'ballInf':
            interaction = ballInf_retriangulation(horizon)
        else:
            raise NotImplementedError('Interaction: {}'.format(interaction))
    else:
        raise NotImplementedError('Interaction: {}'.format(interaction))
    return interaction


def getFractionalKernel(dim,
                        s,
                        horizon=None,
                        interaction=None,
                        scaling=None,
                        normalized=True,
                        piecewise=True,
                        phi=None,
                        boundary=False,
                        derivative=0,
                        tempered=0.,
                        max_horizon=np.nan):
    dim_ = _getDim(dim)
    sFun = _getFractionalOrder(s)
    horizonFun = _getHorizon(horizon)
    interaction = _getInteraction(interaction, horizonFun)

    if isinstance(sFun, admissibleSet):
        kernel = RangedFractionalKernel(dim_, sFun, horizonFun, normalized=normalized, tempered=tempered)
    else:
        if scaling is None:
            if isinstance(sFun, constFractionalOrder) and isinstance(horizonFun, constant):
                if derivative == 0:
                    if normalized:
                        scaling = constantFractionalLaplacianScaling(dim, sFun.value, horizonFun.value, tempered)
                    else:
                        scaling = constantTwoPoint(0.5)
                else:
                    if piecewise:
                        warnings.warn('Derivative kernels cannot be piecewise. Switching to piecewise == False.')
                    piecewise = False
                    scaling = constantFractionalLaplacianScalingDerivative(dim, sFun.value, horizonFun.value, normalized, boundary, derivative, tempered)
            else:
                symmetric = sFun.symmetric and isinstance(horizonFun, constant)
                if piecewise and isinstance(sFun, singleVariableUnsymmetricFractionalOrder):
                    warnings.warn('Variable s kernels cannot be piecewise. Switching to piecewise == False.')
                    piecewise = False
                scaling = variableFractionalLaplacianScaling(symmetric, normalized, boundary, derivative)
            if boundary:
                if isinstance(sFun, (constFractionalOrder,
                                     variableConstFractionalOrder)):
                    fac = constantTwoPoint(1/sFun.value)
                else:
                    fac = inverseTwoPoint(sFun)
                if phi is not None:
                    phi = fac*phi
                else:
                    phi = fac
        kernel = FractionalKernel(dim_, sFun, horizonFun, interaction, scaling, phi, piecewise=piecewise, boundary=boundary,
                                  derivative=derivative, tempered=tempered, max_horizon=max_horizon)

    from . twoPointFunctions import parametrizedTwoPointFunction
    if isinstance(kernel.scaling, parametrizedTwoPointFunction):
        assert kernel.getParamPtrAddr() == kernel.scaling.getParamPtrAddr()
    if isinstance(kernel.interaction, parametrizedTwoPointFunction):
        assert kernel.getParamPtrAddr() == kernel.interaction.getParamPtrAddr()
    return kernel


def getIntegrableKernel(dim,
                        kernel,
                        horizon,
                        scaling=None,
                        interaction=None,
                        normalized=True,
                        piecewise=True,
                        phi=None,
                        boundary=False,
                        monomialPower=np.nan,
                        variance=1.,
                        exponentialRate=1.0,
                        a=1.,
                        max_horizon=np.nan):
    dim_ = _getDim(dim)
    kType = _getKernelType(kernel)
    horizonFun = _getHorizon(horizon)
    interaction = _getInteraction(interaction, horizonFun)

    if scaling is None:
        if normalized:
            if isinstance(horizonFun, constant):
                scaling = constantIntegrableScaling(kType, interaction, dim_, horizonFun.value, gaussian_variance=variance, exponentialRate=exponentialRate)
            else:
                scaling = variableIntegrableScaling(kType, interaction)
        else:
            scaling = constantTwoPoint(0.5)
    if (not scaling.symmetric) or (phi is not None and not phi.symmetric):
        piecewise = False
    return Kernel(dim_, kType=kType, horizon=horizonFun, interaction=interaction, scaling=scaling, phi=phi, piecewise=piecewise,
                  boundary=boundary, monomialPower=monomialPower, max_horizon=max_horizon, variance=variance, exponentialRate=exponentialRate, a=a)


def getKernel(dim,
              s=None,
              horizon=None,
              scaling=None,
              interaction=None,
              normalized=True,
              piecewise=True,
              phi=None,
              kernel=FRACTIONAL,
              boundary=False,
              max_horizon=np.nan,
              variance=1.,
              exponentialRate=1.0):
    kType = _getKernelType(kernel)
    if kType == FRACTIONAL:
        return getFractionalKernel(dim, s, horizon, interaction, scaling, normalized, piecewise, phi, boundary, max_horizon=max_horizon)
    else:
        return getIntegrableKernel(dim,
                                   kernel=kType,
                                   horizon=horizon,
                                   scaling=scaling,
                                   interaction=interaction,
                                   normalized=normalized,
                                   piecewise=piecewise, phi=phi,
                                   max_horizon=max_horizon,
                                   variance=variance,
                                   exponentialRate=exponentialRate)


