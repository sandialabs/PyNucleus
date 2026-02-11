###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport (sin, cos, sqrt,
                        log,
                        pow,
                        exp,
                        M_PI as pi)
from scipy.special.cython_special cimport gammaincc, gamma, hankel1
from scipy.special import eval_legendre
import numpy as np
cimport numpy as np
from PyNucleus_base.myTypes import REAL, COMPLEX, INDEX
from PyNucleus_base import uninitialized
from PyNucleus_base.blas cimport mydot, assign
from PyNucleus_fem.functions cimport constant
from PyNucleus_fem.lookupFunction import (UniformLookup1D,
                                          Lookup1D)
from . interactionDomains cimport (interactionDomain,
                                   fullSpace,
                                   functionOfDistance,
                                   ball1_retriangulation,
                                   ball2_retriangulation,
                                   ballInf_retriangulation)
from . twoPointFunctions cimport (constantTwoPoint,
                                  productTwoPoint,
                                  inverseTwoPoint,
                                  productParametrizedTwoPoint)
from . fractionalOrders cimport (constFractionalOrder,
                                 fractionalOrderBase,
                                 variableFractionalOrder,
                                 variableConstFractionalOrder,
                                 singleVariableUnsymmetricFractionalOrder,
                                 piecewiseConstantFractionalOrder)
from . kernelNormalization cimport (constantFractionalLaplacianScaling,
                                    constantIntegrableScaling,
                                    variableFractionalLaplacianScaling,
                                    variableFractionalLaplacianScalingWithDifferentHorizon,
                                    variableIntegrableScaling,
                                    variableIntegrableScalingWithDifferentHorizon)
from . operatorInterpolation import admissibleSet
from copy import deepcopy


cdef inline REAL_t gammainc(REAL_t a, REAL_t x) noexcept:
    return gamma(a)*gammaincc(a, x)


cdef inline COMPLEX_t hankel10complex(REAL_t x) noexcept:
    return 1j*hankel1(0., x)


include "kernel_params.pxi"


def _getKernelType(kernel):
    if isinstance(kernel, str):
        kType = getKernelEnum(kernel)
    elif isinstance(kernel, int):
        kType = kernel
    else:
        raise NotImplementedError('Kernel type: {}'.format(kernel))
    return kType


cpdef INDEX_t _getDim(dim):
    from PyNucleus_fem.mesh import meshNd
    if isinstance(dim, meshNd):
        return dim.dim
    elif isinstance(dim, (INDEX, int)):
        return dim
    else:
        raise NotImplementedError('Dim: {}'.format(dim))


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


cpdef function _getHorizon(horizon):
    if isinstance(horizon, function):
        horizonFun = horizon
    elif isinstance(horizon, (REAL, float, int)):
        horizonFun = constant(horizon)
    elif horizon is None:
        horizonFun = constant(np.inf)
    else:
        raise NotImplementedError('Horizon: {}'.format(horizon))
    return horizonFun


cpdef interactionDomain _getInteraction(interaction, horizon):
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


def getKernelEnum(str kernelTypeString):
    if kernelTypeString.upper() == "FRACTIONAL":
        return FRACTIONAL
    elif kernelTypeString.upper() in ("INDICATOR", "CONSTANT"):
        return INDICATOR
    elif kernelTypeString.upper() in ("INVERSEDISTANCE", "INVERSEOFDISTANCE", "PERIDYNAMIC"):
        return PERIDYNAMIC
    elif kernelTypeString.upper() == "GAUSSIAN":
        return GAUSSIAN
    elif kernelTypeString.upper() == "EXPONENTIAL":
        return EXPONENTIAL
    elif kernelTypeString.upper() == "POLYNOMIAL":
        return POLYNOMIAL
    elif kernelTypeString.upper() == "LOGINVERSEDISTANCE":
        return LOGINVERSEDISTANCE
    elif kernelTypeString.upper() == "MONOMIAL":
        return MONOMIAL
    elif kernelTypeString.upper() == "GREENS_2D":
        return GREENS_2D
    elif kernelTypeString.upper() == "GREENS_3D":
        return GREENS_3D
    elif kernelTypeString.upper() == "MANIFOLD_FRACTIONAL":
        return MANIFOLD_FRACTIONAL
    else:
        raise NotImplementedError(kernelTypeString)


cdef REAL_t monomialKernel(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fKDIM)
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, inter, d2, singularityValue
    interaction.evalPtr(dim, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        singularityValue = getREAL(c_params, fEXPONENT)
        return C*pow(d2, 0.5*singularityValue)
    else:
        return 0.


cdef REAL_t monomialLogKernel(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fKDIM)
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, inter, d2, singularityValue, logPower
    interaction.evalPtr(dim, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        singularityValue = getREAL(c_params, fEXPONENT)
        logPower = getREAL(c_params, fLOG_SINGULARITY)
        return C*pow(d2, 0.5*singularityValue)*pow(-0.5*log(d2), logPower)
    else:
        return 0.


cdef REAL_t temperedMonomialKernel(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t singularityValue, C, lam, d2, inter
        INDEX_t dim = getINDEX(c_params, fKDIM)
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
    interaction.evalPtr(dim, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        singularityValue = getREAL(c_params, fEXPONENT)
        C = getREAL(c_params, fSCALING)
        lam = getREAL(c_params, fTEMPERED)
        return C*pow(d2, 0.5*singularityValue)*exp(-lam*sqrt(d2))
    else:
        return 0.


cdef REAL_t temperedMonomialKernelBoundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s, C, lam, d2, inter
        INDEX_t dim = getINDEX(c_params, fKDIM)
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
    interaction.evalPtr(dim, x, y, &inter)
    if inter != 0.:
        s = getREAL(c_params, fS)
        C = getREAL(c_params, fSCALING)
        lam = getREAL(c_params, fTEMPERED)
        d2 = interaction.dist2
        return C*pow(d2, 0.5-0.5*dim-s)*gammainc(-2*s, lam*sqrt(d2))
    else:
        return 0.


cdef REAL_t gaussianKernel(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fKDIM)
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, invD
        REAL_t d2, inter
    interaction.evalPtr(dim, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        invD = getREAL(c_params, fEXPONENTINVERSE)
        return C*exp(-d2*invD)
    else:
        return 0.


cdef REAL_t gaussianKernelBoundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fKDIM)
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, invD
        REAL_t d2, inter
    interaction.evalPtr(dim, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        invD = getREAL(c_params, fEXPONENTINVERSE)
        return -0.5*C*pow(d2*invD, -0.5*dim)*gammainc(0.5*dim, d2*invD)*sqrt(d2)
    else:
        return 0.


cdef REAL_t polynomialKernel(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fKDIM)
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, a
        REAL_t d2, inter
    interaction.evalPtr(dim, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        a = getREAL(c_params, fEXPONENTINVERSE)
        return C*(a**3*d2)/(a**2+d2)**2
    else:
        return 0.


cdef REAL_t polynomialKernel1Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, a
        REAL_t d2, inter
    interaction.evalPtr(1, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        a = getREAL(c_params, fEXPONENTINVERSE)
        return -0.5*C*(-a**2/(2*sqrt(d2)) + a**3/2/(a**2+d2))
    else:
        return 0.


cdef REAL_t logInverseDistanceKernel(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fKDIM)
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C
        REAL_t d2, inter
    interaction.evalPtr(dim, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        return -0.5*C*log(d2)
    else:
        return 0.


cdef COMPLEX_t greens2Dcomplex(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fKDIM)
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t lam = getREAL(c_params, fGREENS_LAMBDA)
        REAL_t inter
    interaction.evalPtr(dim, x, y, &inter)
    C = getREAL(c_params, fSCALING)
    return C*hankel10complex(lam*sqrt(interaction.d2))


cdef COMPLEX_t greens3Dcomplex(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fKDIM)
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C = getREAL(c_params, fSCALING)
        COMPLEX_t lam = getCOMPLEX(c_params, fGREENS_LAMBDA)
        REAL_t d, inter
    interaction.evalPtr(dim, x, y, &inter)
    C = getREAL(c_params, fSCALING)
    d = sqrt(interaction.d2)
    return C*exp(-lam.real*d)*(cos(-lam.imag*d)+1j*sin(-lam.imag*d))/d


include "kernels_REAL.pxi"
include "kernels_COMPLEX.pxi"
include "fractionalKernel.pxi"
