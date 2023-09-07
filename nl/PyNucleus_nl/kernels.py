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
                                  ball1,
                                  ball2,
                                  ballInf)
from . fractionalOrders import (fractionalOrderBase,
                                constFractionalOrder,
                                variableConstFractionalOrder,
                                singleVariableUnsymmetricFractionalOrder)
from . kernelNormalization import (constantFractionalLaplacianScaling,
                                   constantFractionalLaplacianScalingDerivative,
                                   variableFractionalLaplacianScaling,
                                   constantIntegrableScaling,
                                   )
from . kernelsCy import (Kernel,
                         ComplexKernel,
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
        interaction = ball2()
    elif isinstance(interaction, str):
        if interaction == 'fullSpace':
            interaction = fullSpace()
        elif interaction == 'ball1':
            interaction = ball1()
        elif interaction == 'ball2':
            interaction = ball2()
        elif interaction == 'ballInf':
            interaction = ballInf()
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
                        tempered=0.):
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
        kernel = FractionalKernel(dim_, sFun, horizonFun, interaction, scaling, phi, piecewise=piecewise, boundary=boundary, derivative=derivative, tempered=tempered)
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
                        monomialPower=np.nan):
    dim_ = _getDim(dim)
    kType = _getKernelType(kernel)
    horizonFun = _getHorizon(horizon)
    interaction = _getInteraction(interaction, horizonFun)

    if scaling is None:
        if normalized:
            if isinstance(horizonFun, constant):
                scaling = constantIntegrableScaling(kType, interaction, dim_, horizonFun.value)
            else:
                raise NotImplementedError()
        else:
            scaling = constantTwoPoint(0.5)
    return Kernel(dim_, kType=kType, horizon=horizonFun, interaction=interaction, scaling=scaling, phi=phi, piecewise=piecewise, boundary=boundary, monomialPower=monomialPower)


def getKernel(dim,
              s=None,
              horizon=None,
              scaling=None,
              interaction=None,
              normalized=True,
              piecewise=True,
              phi=None,
              kernel=FRACTIONAL,
              boundary=False):
    kType = _getKernelType(kernel)
    if kType == FRACTIONAL:
        return getFractionalKernel(dim, s, horizon, interaction, scaling, normalized, piecewise, phi, boundary)
    else:
        return getIntegrableKernel(dim,
                                   kernel=kType,
                                   horizon=horizon,
                                   scaling=scaling,
                                   interaction=interaction,
                                   normalized=normalized,
                                   piecewise=piecewise, phi=phi)


