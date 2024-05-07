###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport (sin, cos, sqrt,
                        log,
                        fabs as abs, pow,
                        exp)
from scipy.special.cython_special cimport gammaincc, gamma, hankel1
import numpy as np
cimport numpy as np
from PyNucleus_base.myTypes import REAL
from PyNucleus_base import uninitialized
from PyNucleus_fem.functions cimport constant
from . interactionDomains cimport ball2_retriangulation, fullSpace
from . twoPointFunctions cimport (constantTwoPoint,
                                  productTwoPoint,
                                  inverseTwoPoint,
                                  productParametrizedTwoPoint)
from . fractionalOrders cimport (constFractionalOrder,
                                 variableFractionalOrder,
                                 piecewiseConstantFractionalOrder)
from . kernelNormalization cimport (constantFractionalLaplacianScaling,
                                    constantFractionalLaplacianScalingDerivative,
                                    variableFractionalLaplacianScaling,
                                    variableFractionalLaplacianScalingBoundary,
                                    variableFractionalLaplacianScalingWithDifferentHorizon,
                                    variableIntegrableScaling,
                                    variableIntegrableScalingWithDifferentHorizon)
from copy import deepcopy


cdef inline REAL_t gammainc(REAL_t a, REAL_t x):
    return gamma(a)*gammaincc(a, x)


cdef inline COMPLEX_t hankel10complex(REAL_t x):
    return 1j*hankel1(0., x)


include "kernel_params.pxi"


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
    else:
        raise NotImplementedError(kernelTypeString)


cdef REAL_t fracKernelFinite1D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s, C, d2, inter
        twoPointFunction interaction = <twoPointFunction>((<void**>(c_params+fINTERACTION))[0])
    interaction.evalPtr(1, x, y, &inter)
    if inter != 0.:
        s = getREAL(c_params, fS)
        C = getREAL(c_params, fSCALING)
        d2 = interaction.dist2
        return C*pow(d2, -0.5-s)
    else:
        return 0.


cdef REAL_t fracKernelFinite2D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s, C, d2, inter
        twoPointFunction interaction = <twoPointFunction>((<void**>(c_params+fINTERACTION))[0])
    interaction.evalPtr(2, x, y, &inter)
    if inter != 0.:
        s = getREAL(c_params, fS)
        C = getREAL(c_params, fSCALING)
        d2 = interaction.dist2
        return C*pow(d2, -1.-s)
    else:
        return 0.


cdef REAL_t fracKernelFinite3D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s, C, d2, inter
        twoPointFunction interaction = <twoPointFunction>((<void**>(c_params+fINTERACTION))[0])
    interaction.evalPtr(3, x, y, &inter)
    if inter != 0.:
        s = getREAL(c_params, fS)
        C = getREAL(c_params, fSCALING)
        d2 = interaction.dist2
        return C*pow(d2, -1.5-s)
    else:
        return 0.


cdef REAL_t fracKernelFinite1Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s, C, d2, inter
        twoPointFunction interaction = <twoPointFunction>((<void**>(c_params+fINTERACTION))[0])
    interaction.evalPtr(1, x, y, &inter)
    if inter != 0.:
        s = getREAL(c_params, fS)
        C = getREAL(c_params, fSCALING)
        d2 = interaction.dist2
        return C*pow(d2, -s)
    else:
        return 0.


cdef REAL_t fracKernelFinite2Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s, C, d2, inter
        twoPointFunction interaction = <twoPointFunction>((<void**>(c_params+fINTERACTION))[0])
    interaction.evalPtr(2, x, y, &inter)
    if inter != 0.:
        s = getREAL(c_params, fS)
        C = getREAL(c_params, fSCALING)
        d2 = interaction.dist2
        return C*pow(d2, -0.5-s)
    else:
        return 0.


cdef REAL_t fracKernelFinite3Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s, C, d2, inter
        twoPointFunction interaction = <twoPointFunction>((<void**>(c_params+fINTERACTION))[0])
    interaction.evalPtr(3, x, y, &inter)
    if inter != 0.:
        s = getREAL(c_params, fS)
        C = getREAL(c_params, fSCALING)
        d2 = interaction.dist2
        return C*pow(d2, -1.0-s)
    else:
        return 0.


cdef REAL_t fracKernelInfinite1D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fS)
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0])
    return C*pow(d2, -0.5-s)


cdef REAL_t fracKernelInfinite2D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fS)
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1])
    return C*pow(d2, -1.-s)


cdef REAL_t fracKernelInfinite3D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fS)
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2])
    return C*pow(d2, -1.5-s)


cdef REAL_t temperedFracKernelInfinite1D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fS)
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t lam = getREAL(c_params, fTEMPERED)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0])
    return C*pow(d2, -0.5-s)*exp(-lam*sqrt(d2))


cdef REAL_t temperedFracKernelInfinite2D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fS)
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t lam = getREAL(c_params, fTEMPERED)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1])
    return C*pow(d2, -1.-s)*exp(-lam*sqrt(d2))


cdef REAL_t temperedFracKernelInfinite3D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fS)
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t lam = getREAL(c_params, fTEMPERED)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2])
    return C*pow(d2, -1.5-s)*exp(-lam*sqrt(d2))


cdef REAL_t fracKernelInfinite1Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fS)
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0])
    return C*pow(d2, -s)


cdef REAL_t fracKernelInfinite2Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fS)
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1])
    return C*pow(d2, -0.5-s)


cdef REAL_t fracKernelInfinite3Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fS)
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2])
    return C*pow(d2, -1.0-s)


cdef REAL_t temperedFracKernelInfinite1Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fS)
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t lam = getREAL(c_params, fTEMPERED)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0])
    return C*pow(lam, 2*s)*pow(d2, -0.5)*gammainc(-2*s, lam*sqrt(d2))


cdef REAL_t temperedFracKernelInfinite2Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fS)
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t lam = getREAL(c_params, fTEMPERED)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1])
    return C*pow(lam, 2*s)*pow(d2, -1.0)*gammainc(-2*s, lam*sqrt(d2))


cdef REAL_t temperedFracKernelInfinite3Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fS)
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t lam = getREAL(c_params, fTEMPERED)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2])
    return C*pow(lam, 2*s)*pow(d2, -1.5)*gammainc(-2*s, lam*sqrt(d2))


cdef REAL_t indicatorKernel1D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        twoPointFunction interaction = <twoPointFunction>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, inter
    interaction.evalPtr(1, x, y, &inter)
    if inter != 0.:
        C = getREAL(c_params, fSCALING)
        return C
    else:
        return 0.


cdef REAL_t indicatorKernel2D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        twoPointFunction interaction = <twoPointFunction>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, inter
    interaction.evalPtr(2, x, y, &inter)
    if inter != 0.:
        C = getREAL(c_params, fSCALING)
        return C
    else:
        return 0.


cdef REAL_t indicatorKernel1Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        twoPointFunction interaction = <twoPointFunction>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, inter
    interaction.evalPtr(1, x, y, &inter)
    if inter != 0.:
        C = getREAL(c_params, fSCALING)
        return -C*2.0*sqrt((x[0]-y[0])*(x[0]-y[0]))
    else:
        return 0.


cdef REAL_t indicatorKernel2Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        twoPointFunction interaction = <twoPointFunction>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, inter
    interaction.evalPtr(2, x, y, &inter)
    if inter != 0.:
        C = getREAL(c_params, fSCALING)
        return -C*sqrt((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]))
    else:
        return 0.


cdef REAL_t peridynamicKernel1D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C
        REAL_t d2, inter
    interaction.evalPtr(1, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        return C/sqrt(d2)
    else:
        return 0.


cdef REAL_t peridynamicKernel2D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C
        REAL_t d2, inter
    interaction.evalPtr(2, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        return C/sqrt(d2)
    else:
        return 0.


cdef REAL_t peridynamicKernel3D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C
        REAL_t d2, inter
    interaction.evalPtr(3, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        return C/sqrt(d2)
    else:
        return 0.


cdef REAL_t peridynamicKernel1Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C
        REAL_t d, inter
    interaction.evalPtr(1, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        return -2.0*C*0.5*log(d2)
    else:
        return 0.


cdef REAL_t peridynamicKernel2Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, inter
    interaction.evalPtr(2, x, y, &inter)
    if inter != 0.:
        C = getREAL(c_params, fSCALING)
        return -2.0*C
    else:
        return 0.

cdef REAL_t gaussianKernel1D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, invD
        REAL_t d2, inter
    interaction.evalPtr(1, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        invD = getREAL(c_params, fEXPONENTINVERSE)
        return C*exp(-d2*invD)
    else:
        return 0.


cdef REAL_t gaussianKernel2D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, invD
        REAL_t d2, inter
    interaction.evalPtr(2, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        invD = getREAL(c_params, fEXPONENTINVERSE)
        return C*exp(-d2*invD)
    else:
        return 0.


cdef REAL_t gaussianKernel1Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, invD
        REAL_t d2, inter
    interaction.evalPtr(1, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        invD = getREAL(c_params, fEXPONENTINVERSE)
        return C*sqrt(1./(d2*invD))*gammainc(0.5, d2*invD)*sqrt(d2)
    else:
        return 0.


cdef REAL_t gaussianKernel2Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, invD
        REAL_t d2, inter
    interaction.evalPtr(2, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        invD = getREAL(c_params, fEXPONENTINVERSE)
        return C*(1./(d2*invD))*gammainc(1.0, d2*invD)*sqrt(d2)
    else:
        return 0.


cdef REAL_t exponentialKernel1D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, a
        REAL_t d2, inter
    interaction.evalPtr(1, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        a = getREAL(c_params, fEXPONENTINVERSE)
        return C*exp(-a*sqrt(d2))
    else:
        return 0.


cdef REAL_t exponentialKernel1Dboundary(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, a
        REAL_t d2, inter
    interaction.evalPtr(1, x, y, &inter)
    if inter != 0.:
        d2 = interaction.dist2
        C = getREAL(c_params, fSCALING)
        a = getREAL(c_params, fEXPONENTINVERSE)
        return 2.0*C*exp(-a*sqrt(d2))/a
    else:
        return 0.


cdef REAL_t polynomialKernel1D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        interactionDomain interaction = <interactionDomain>((<void**>(c_params+fINTERACTION))[0])
        REAL_t C, a
        REAL_t d2, inter
    interaction.evalPtr(1, x, y, &inter)
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
        return C*(-a**2/(2*sqrt(d2)) + a**3/2/(a**2+d2))
    else:
        return 0.


cdef REAL_t logInverseDistance2D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1])
    C = getREAL(c_params, fSCALING)
    return -0.5*C*log(d2)


cdef COMPLEX_t greens2Dcomplex(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t lam = getREAL(c_params, fGREENS_LAMBDA)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1])
    C = getREAL(c_params, fSCALING)
    return C*hankel10complex(lam*sqrt(d2))


cdef COMPLEX_t greens3Dcomplex(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t C = getREAL(c_params, fSCALING)
        COMPLEX_t lam = getCOMPLEX(c_params, fGREENS_LAMBDA)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2])
    C = getREAL(c_params, fSCALING)
    d2 = sqrt(d2)
    return C*exp(-lam.real*d2)*(cos(-lam.imag*d2)+1j*sin(-lam.imag*d2))/d2


cdef REAL_t monomial3D(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t C = getREAL(c_params, fSCALING)
        REAL_t singularityValue = getREAL(c_params, fSINGULARITY)
        REAL_t d2
    d2 = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2])
    C = getREAL(c_params, fSCALING)
    return C*pow(d2, 0.5*singularityValue)


cdef REAL_t updateAndEvalIntegrable(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fKDIM)
        REAL_t[::1] xA
        function horizonFun
        twoPointFunction scalingFun
        REAL_t horizon, C
        fun_t kernel = getFun(c_params, fEVAL)
        BOOL_t horizonFunNull = isNull(c_params, fHORIZONFUN)
        BOOL_t scalingFunNull = isNull(c_params, fSCALINGFUN)
    if not horizonFunNull or not scalingFunNull:
        xA = <REAL_t[:dim]> x
    if not horizonFunNull:
        horizonFun = <function>((<void**>(c_params+fHORIZONFUN))[0])
        horizon = horizonFun.eval(xA)
        setREAL(c_params, fHORIZON2, horizon*horizon)
    if not scalingFunNull:
        scalingFun = <twoPointFunction>((<void**>(c_params+fSCALINGFUN))[0])
        scalingFun.evalPtr(dim, x, y, &C)
        setREAL(c_params, fSCALING, C)
    return kernel(x, y, c_params)


cdef COMPLEX_t updateAndEvalIntegrableComplex(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fKDIM)
        REAL_t[::1] xA
        function horizonFun
        twoPointFunction scalingFun
        REAL_t horizon, C
        complex_fun_t kernel = getComplexFun(c_params, fEVAL)
        BOOL_t horizonFunNull = isNull(c_params, fHORIZONFUN)
        BOOL_t scalingFunNull = isNull(c_params, fSCALINGFUN)
    if not horizonFunNull or not scalingFunNull:
        xA = <REAL_t[:dim]> x
    if not horizonFunNull:
        horizonFun = <function>((<void**>(c_params+fHORIZONFUN))[0])
        horizon = horizonFun.eval(xA)
        setREAL(c_params, fHORIZON2, horizon*horizon)
    if not scalingFunNull:
        scalingFun = <twoPointFunction>((<void**>(c_params+fSCALINGFUN))[0])
        scalingFun.evalPtr(dim, x, y, &C)
        setREAL(c_params, fSCALING, C)
    return kernel(x, y, c_params)


cdef REAL_t updateAndEvalFractional(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fKDIM)
        REAL_t[::1] xA
        fractionalOrderBase sFun
        function horizonFun
        twoPointFunction scalingFun
        REAL_t s, horizon, C
        BOOL_t boundary
        fun_t kernel = getFun(c_params, fEVAL)

    if not isNull(c_params, fORDERFUN):
        sFun = <fractionalOrderBase>((<void**>(c_params+fORDERFUN))[0])
        sFun.evalPtr(dim, x, y, &s)
        setREAL(c_params, fS, s)
        boundary = getBOOL(c_params, fBOUNDARY)
        setREAL(c_params, fSINGULARITY, boundary-dim-2*s)
    if not isNull(c_params, fHORIZONFUN):
        xA = <REAL_t[:dim]> x
        horizonFun = <function>((<void**>(c_params+fHORIZONFUN))[0])
        horizon = horizonFun.eval(xA)
        setREAL(c_params, fHORIZON2, horizon*horizon)
    if not isNull(c_params, fSCALINGFUN):
        scalingFun = <twoPointFunction>((<void**>(c_params+fSCALINGFUN))[0])
        scalingFun.evalPtr(dim, x, y, &C)
        setREAL(c_params, fSCALING, C)
    return kernel(x, y, c_params)


cdef class Kernel(twoPointFunction):
    """A kernel functions that can be used to define a nonlocal operator."""

    def __init__(self, INDEX_t dim, kernelType kType, function horizon, interactionDomain interaction, twoPointFunction scaling, twoPointFunction phi, BOOL_t piecewise=True, BOOL_t boundary=False, INDEX_t valueSize=1, REAL_t max_horizon=np.nan, **kwargs):
        cdef:
            parametrizedTwoPointFunction parametrizedScaling
            int i

        self.c_kernel_params = PyMem_Malloc(NUM_KERNEL_PARAMS*OFFSET)
        for i in range(NUM_KERNEL_PARAMS):
            (<void**>(self.c_kernel_params+i*OFFSET))[0] = NULL

        self.dim = dim
        assert valueSize >= 1, "Creation of kernel with valueSize = {}".format(valueSize)
        self.valueSize = valueSize
        self.kernelType = kType
        self.piecewise = piecewise
        self.boundary = boundary

        setINDEX(self.c_kernel_params, fKDIM, dim)

        symmetric = isinstance(horizon, constant) and scaling.symmetric and interaction.symmetric and (phi is None or phi.symmetric)
        super(Kernel, self).__init__(symmetric, valueSize)

        if self.kernelType == INDICATOR:
            self.min_singularity = 0.
            self.max_singularity = 0.
            self.singularityValue = 0.
        elif self.kernelType == PERIDYNAMIC:
            self.min_singularity = -1.
            self.max_singularity = -1.
            self.singularityValue = -1.
        elif self.kernelType == GAUSSIAN:
            self.min_singularity = 0.
            self.max_singularity = 0.
            self.singularityValue = 0.
        elif self.kernelType == EXPONENTIAL:
            self.min_singularity = 0.
            self.max_singularity = 0.
            self.singularityValue = 0.
        elif self.kernelType == POLYNOMIAL:
            self.min_singularity = 0.
            self.max_singularity = 0.
            self.singularityValue = 0.
        elif self.kernelType == LOGINVERSEDISTANCE:
            self.min_singularity = 0.
            self.max_singularity = 0.
            self.singularityValue = 0.
        elif self.kernelType == MONOMIAL:
            monomialPower = kwargs.get('monomialPower', np.nan)
            self.min_singularity = monomialPower
            self.max_singularity = monomialPower
            self.singularityValue = monomialPower

        self.horizon = horizon
        self.variableHorizon = not isinstance(self.horizon, constant)
        if self.variableHorizon:
            self.horizonValue2 = np.nan
            self.finiteHorizon = True
            (<void**>(self.c_kernel_params+fHORIZONFUN))[0] = <void*>horizon
            self.max_horizon = max_horizon
        else:
            self.horizonValue = self.horizon.value
            self.max_horizon = self.horizon.value
            self.finiteHorizon = self.horizon.value != np.inf
            if self.kernelType == GAUSSIAN:
                if self.finiteHorizon:
                    setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
                else:
                    variance = kwargs.get('variance', 1.0)
                    setREAL(self.c_kernel_params, fEXPONENTINVERSE, 0.5/variance**self.dim)
            elif self.kernelType == EXPONENTIAL:
                exponentialRate = kwargs.get('exponentialRate', 1.0)
                setREAL(self.c_kernel_params, fEXPONENTINVERSE, exponentialRate)
            elif self.kernelType == POLYNOMIAL:
                a = kwargs.get('a', 1.0)
                setREAL(self.c_kernel_params, fEXPONENTINVERSE, a)

        self.interaction = interaction
        self.complement = self.interaction.complement
        (<void**>(self.c_kernel_params+fINTERACTION))[0] = <void*>self.interaction
        self.interaction.setParams(self.c_kernel_params)

        self.phi = phi
        if phi is not None:
            scaling = phi*scaling
        self.scaling = scaling
        self.variableScaling = not isinstance(self.scaling, (constantFractionalLaplacianScaling, constantTwoPoint))
        if self.variableScaling:
            if isinstance(self.scaling, parametrizedTwoPointFunction):
                parametrizedScaling = self.scaling
                parametrizedScaling.setParams(self.c_kernel_params)
            self.scalingValue = np.nan
            (<void**>(self.c_kernel_params+fSCALINGFUN))[0] = <void*>self.scaling
        else:
            self.scalingValue = self.scaling.value

        self.variable = self.variableHorizon or self.variableScaling

        if self.piecewise:
            if not self.boundary:
                if dim == 1:
                    if self.kernelType == INDICATOR:
                        self.kernelFun = indicatorKernel1D
                    elif self.kernelType == PERIDYNAMIC:
                        self.kernelFun = peridynamicKernel1D
                    elif self.kernelType == GAUSSIAN:
                        self.kernelFun = gaussianKernel1D
                    elif self.kernelType == EXPONENTIAL:
                        self.kernelFun = exponentialKernel1D
                    elif self.kernelType == POLYNOMIAL:
                        self.kernelFun = polynomialKernel1D
                elif dim == 2:
                    if self.kernelType == INDICATOR:
                        self.kernelFun = indicatorKernel2D
                    elif self.kernelType == PERIDYNAMIC:
                        self.kernelFun = peridynamicKernel2D
                    elif self.kernelType == GAUSSIAN:
                        self.kernelFun = gaussianKernel2D
                    elif self.kernelType == EXPONENTIAL:
                        raise NotImplementedError()
                    elif self.kernelType == POLYNOMIAL:
                        raise NotImplementedError()
                    elif self.kernelType == LOGINVERSEDISTANCE:
                        self.kernelFun = logInverseDistance2D
                elif dim == 3:
                    if self.kernelType == PERIDYNAMIC:
                        self.kernelFun = peridynamicKernel3D
                    elif self.kernelType == MONOMIAL:
                        self.kernelFun = monomial3D
                    elif self.kernelType == GAUSSIAN:
                        raise NotImplementedError()
                    elif self.kernelType == EXPONENTIAL:
                        raise NotImplementedError()
                    elif self.kernelType == POLYNOMIAL:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            else:
                if dim == 1:
                    if self.kernelType == INDICATOR:
                        self.kernelFun = indicatorKernel1Dboundary
                    elif self.kernelType == PERIDYNAMIC:
                        self.kernelFun = peridynamicKernel1Dboundary
                    elif self.kernelType == GAUSSIAN:
                        self.kernelFun = gaussianKernel1Dboundary
                    elif self.kernelType == EXPONENTIAL:
                        self.kernelFun = exponentialKernel1Dboundary
                    elif self.kernelType == POLYNOMIAL:
                        self.kernelFun = polynomialKernel1Dboundary
                elif dim == 2:
                    if self.kernelType == INDICATOR:
                        self.kernelFun = indicatorKernel2Dboundary
                    elif self.kernelType == PERIDYNAMIC:
                        self.kernelFun = peridynamicKernel2Dboundary
                    elif self.kernelType == GAUSSIAN:
                        self.kernelFun = gaussianKernel2Dboundary
                    elif self.kernelType == EXPONENTIAL:
                        raise NotImplementedError()
                    elif self.kernelType == POLYNOMIAL:
                        raise NotImplementedError()
                elif dim == 3:
                    if self.kernelType == GAUSSIAN:
                        raise NotImplementedError()
                    elif self.kernelType == EXPONENTIAL:
                        raise NotImplementedError()
                    elif self.kernelType == POLYNOMIAL:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
        else:
            self.kernelFun = updateAndEvalIntegrable

            if not self.boundary:
                if dim == 1:
                    if self.kernelType == INDICATOR:
                        setFun(self.c_kernel_params, fEVAL, indicatorKernel1D)
                    elif self.kernelType == PERIDYNAMIC:
                        setFun(self.c_kernel_params, fEVAL, peridynamicKernel1D)
                    elif self.kernelType == GAUSSIAN:
                        setFun(self.c_kernel_params, fEVAL, gaussianKernel1D)
                    elif self.kernelType == EXPONENTIAL:
                        setFun(self.c_kernel_params, fEVAL, exponentialKernel1D)
                    elif self.kernelType == POLYNOMIAL:
                        setFun(self.c_kernel_params, fEVAL, polynomialKernel1D)
                elif dim == 2:
                    if self.kernelType == INDICATOR:
                        setFun(self.c_kernel_params, fEVAL, indicatorKernel2D)
                    elif self.kernelType == PERIDYNAMIC:
                        setFun(self.c_kernel_params, fEVAL, peridynamicKernel2D)
                    elif self.kernelType == GAUSSIAN:
                        setFun(self.c_kernel_params, fEVAL, gaussianKernel2D)
                    elif self.kernelType == EXPONENTIAL:
                        raise NotImplementedError()
                    elif self.kernelType == POLYNOMIAL:
                        raise NotImplementedError()
                    elif self.kernelType == LOGINVERSEDISTANCE:
                        setFun(self.c_kernel_params, fEVAL, logInverseDistance2D)
                elif dim == 3:
                    if self.kernelType == PERIDYNAMIC:
                        setFun(self.c_kernel_params, fEVAL, peridynamicKernel3D)
                    elif self.kernelType == MONOMIAL:
                        setFun(self.c_kernel_params, fEVAL, monomial3D)
                    elif self.kernelType == GAUSSIAN:
                        raise NotImplementedError()
                    elif self.kernelType == EXPONENTIAL:
                        raise NotImplementedError()
                    elif self.kernelType == POLYNOMIAL:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            else:
                if dim == 1:
                    if self.kernelType == INDICATOR:
                        setFun(self.c_kernel_params, fEVAL, indicatorKernel1Dboundary)
                    elif self.kernelType == PERIDYNAMIC:
                        setFun(self.c_kernel_params, fEVAL, peridynamicKernel1Dboundary)
                    elif self.kernelType == GAUSSIAN:
                        setFun(self.c_kernel_params, fEVAL, gaussianKernel1Dboundary)
                    elif self.kernelType == EXPONENTIAL:
                        setFun(self.c_kernel_params, fEVAL, exponentialKernel1Dboundary)
                    elif self.kernelType == POLYNOMIAL:
                        setFun(self.c_kernel_params, fEVAL, polynomialKernel1Dboundary)
                elif dim == 2:
                    if self.kernelType == INDICATOR:
                        setFun(self.c_kernel_params, fEVAL, indicatorKernel2Dboundary)
                    elif self.kernelType == PERIDYNAMIC:
                        setFun(self.c_kernel_params, fEVAL, peridynamicKernel2Dboundary)
                    elif self.kernelType == GAUSSIAN:
                        setFun(self.c_kernel_params, fEVAL, gaussianKernel2Dboundary)
                    elif self.kernelType == EXPONENTIAL:
                        raise NotImplementedError()
                    elif self.kernelType == POLYNOMIAL:
                        raise NotImplementedError()
                elif dim == 3:
                    if self.kernelType == GAUSSIAN:
                        raise NotImplementedError()
                    elif self.kernelType == EXPONENTIAL:
                        raise NotImplementedError()
                    elif self.kernelType == POLYNOMIAL:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()

    def getParamPtrAddr(self):
        return <size_t>self.c_kernel_params

    @property
    def boundary(self):
        "The order of the boundary."
        return getBOOL(self.c_kernel_params, fBOUNDARY)

    @boundary.setter
    def boundary(self, BOOL_t boundary):
        setBOOL(self.c_kernel_params, fBOUNDARY, boundary)

    cdef BOOL_t getBoundary(self):
        return getBOOL(self.c_kernel_params, fBOUNDARY)

    cdef void setBoundary(self, BOOL_t boundary):
        setBOOL(self.c_kernel_params, fBOUNDARY, boundary)

    @property
    def singularityValue(self):
        "The order of the singularity."
        return getREAL(self.c_kernel_params, fSINGULARITY)

    @singularityValue.setter
    def singularityValue(self, REAL_t singularity):
        setREAL(self.c_kernel_params, fSINGULARITY, singularity)

    cdef REAL_t getSingularityValue(self):
        return getREAL(self.c_kernel_params, fSINGULARITY)

    cdef void setSingularityValue(self, REAL_t singularity):
        setREAL(self.c_kernel_params, fSINGULARITY, singularity)

    @property
    def horizonValue(self):
        "The value of the interaction horizon."
        return sqrt(getREAL(self.c_kernel_params, fHORIZON2))

    @horizonValue.setter
    def horizonValue(self, REAL_t horizon):
        setREAL(self.c_kernel_params, fHORIZON2, horizon**2)

    cdef REAL_t getHorizonValue(self):
        return sqrt(getREAL(self.c_kernel_params, fHORIZON2))

    cdef void setHorizonValue(self, REAL_t horizon):
        setREAL(self.c_kernel_params, fHORIZON2, horizon**2)

    @property
    def horizonValue2(self):
        return getREAL(self.c_kernel_params, fHORIZON2)

    cdef REAL_t getHorizonValue2(self):
        return getREAL(self.c_kernel_params, fHORIZON2)

    @horizonValue2.setter
    def horizonValue2(self, REAL_t horizon2):
        setREAL(self.c_kernel_params, fHORIZON2, horizon2)

    @property
    def scalingValue(self):
        "The value of the scaling factor."
        return getREAL(self.c_kernel_params, fSCALING)

    @scalingValue.setter
    def scalingValue(self, REAL_t scaling):
        setREAL(self.c_kernel_params, fSCALING, scaling)

    cdef REAL_t getScalingValue(self):
        return getREAL(self.c_kernel_params, fSCALING)

    cdef void setScalingValue(self, REAL_t scaling):
        setREAL(self.c_kernel_params, fSCALING, scaling)

    def getLongDescription(self):
        cdef:
            str descr = ''
            dict params = {}
        params['d'] = self.dim
        if self.finiteHorizon:
            params['\\delta'] = self.horizon
        if self.kernelType == INDICATOR:
            descr = ''
        elif self.kernelType == PERIDYNAMIC:
            descr = '\\frac{1}{|x-y|}'
        elif self.kernelType == FRACTIONAL:
            descr = '\\frac{1}{|x-y|^{d+2s}}'
        elif self.kernelType == GAUSSIAN:
            if self.finiteHorizon:
                exponentInverse = self.getKernelParam('exponentInverse')
                params['a'] = exponentInverse
                descr = '\\exp(-a |x-y|^2)'
            else:
                variance = self.getKernelParam('variance')
                params['\\sigma'] = variance
                descr = '\\exp(-0.5 |x-y|^2/\\sigma^{d})'
        elif self.kernelType == EXPONENTIAL:
            exponentialRate = self.getKernelParam('exponentialRate')
            params['a'] = exponentialRate
            descr = '\\exp(-a |x-y|)'
        elif self.kernelType == POLYNOMIAL:
            a = self.getKernelParam('a')
            params['a'] = a
            descr = '\\frac{a^3 |x-y|^2}{(a^2 + |x-y|^2)^2}'
        paramsStr = ', '.join([k+'='+str(v) for k, v in params.items()])
        return self.scaling.getLongDescription() + ' ' + descr + ' ' + self.interaction.getLongDescription() + ', ' + paramsStr

    def getKernelParams(self):
        params = []
        if self.kernelType == GAUSSIAN:
            if self.finiteHorizon:
                params.append('exponentInverse')
            else:
                params.append('variance')
        elif self.kernelType == EXPONENTIAL:
            params.append('exponentialRate')
        elif self.kernelType == POLYNOMIAL:
            params.append('a')
        elif self.kernelType == MONOMIAL:
            params.append('monomialPower')
        return params

    def getKernelParam(self, str param):
        if self.kernelType == GAUSSIAN:
            if self.finiteHorizon:
                if param == 'exponentInverse':
                    return getREAL(self.c_kernel_params, fEXPONENTINVERSE)
            else:
                if param == 'variance':
                    exponentInverse = getREAL(self.c_kernel_params, fEXPONENTINVERSE)
                    variance = 1.0/(2.0*exponentInverse)**(1.0/self.dim)
                    return variance
        elif self.kernelType == EXPONENTIAL:
            if param == 'exponentialRate':
                return getREAL(self.c_kernel_params, fEXPONENTINVERSE)
        elif self.kernelType == POLYNOMIAL:
            if param == 'a':
                return getREAL(self.c_kernel_params, fEXPONENTINVERSE)
        elif self.kernelType == MONOMIAL:
            if param == 'monomialPower':
                return self.getSingularityValue()
        raise NotImplementedError("Parameter not available: {}".format(param))

    cdef void evalParamsOnSimplices(self, REAL_t[::1] center1, REAL_t[::1] center2, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        # Set the horizon.
        if self.variableHorizon:
            self.horizonValue = self.horizon.eval(center1)

    cdef void evalParams(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t scalingValue
        if self.piecewise:
            if self.variableHorizon:
                self.horizonValue = self.horizon.eval(x)
                if self.kernelType == GAUSSIAN:
                    setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
                elif self.kernelType == POLYNOMIAL:
                    setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
            if self.variableScaling:
                self.scaling.evalPtr(x.shape[0], &x[0], &y[0], &scalingValue)
                self.scalingValue = scalingValue

    def evalParams_py(self, REAL_t[::1] x, REAL_t[::1] y):
        "Evaluate the kernel parameters."
        cdef:
            REAL_t scalingValue
        if self.piecewise:
            self.evalParams(x, y)
        else:
            if self.variableHorizon:
                self.horizonValue = self.horizon.eval(x)
                if self.kernelType == GAUSSIAN:
                    setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
                elif self.kernelType == POLYNOMIAL:
                    setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
            if self.variableScaling:
                self.scaling.evalPtr(x.shape[0], &x[0], &y[0], &scalingValue)
                self.scalingValue = scalingValue

    cdef void evalParamsPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t[::1] xA
            REAL_t scalingValue
        if self.piecewise:
            if self.variableHorizon:
                xA = <REAL_t[:dim]> x
                self.horizonValue = self.horizon.eval(xA)
                if self.kernelType == GAUSSIAN:
                    setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
                elif self.kernelType == POLYNOMIAL:
                    setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
            if self.variableScaling:
                self.scaling.evalPtr(dim, x, y, &scalingValue)
                self.scalingValue = scalingValue

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] vec):
        vec[0] = self.kernelFun(&x[0], &y[0], self.c_kernel_params)

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* vec):
        vec[0] = self.kernelFun(x, y, self.c_kernel_params)

    def __call__(self, REAL_t[::1] x, REAL_t[::1] y, BOOL_t callEvalParams=True):
        "Evaluate the kernel."
        if self.piecewise and callEvalParams:
            self.evalParams(x, y)
        return self.kernelFun(&x[0], &y[0], self.c_kernel_params)

    def eval_py(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] vec, BOOL_t callEvalParams=True):
        "Evaluate the kernel."
        if self.piecewise and callEvalParams:
            self.evalParams(x, y)
        self.eval(x, y, vec)

    def getModifiedKernel(self,
                          function horizon=None,
                          twoPointFunction scaling=None):
        cdef:
            Kernel newKernel
        if horizon is None:
            horizon = self.horizon
            interaction = self.interaction
        else:
            if scaling is None and isinstance(self.scaling, (variableFractionalLaplacianScaling,
                                                             variableIntegrableScaling)):
                scaling = self.scaling.getScalingWithDifferentHorizon()
            interaction = deepcopy(self.interaction)
        if scaling is None:
            scaling = self.scaling
        from . kernels import getKernel
        newKernel = getKernel(dim=self.dim, kernel=self.kernelType, horizon=horizon, interaction=interaction, scaling=scaling, piecewise=self.piecewise)
        setREAL(newKernel.c_kernel_params, fEXPONENTINVERSE, getREAL(self.c_kernel_params, fEXPONENTINVERSE))
        if isinstance(newKernel.scaling, parametrizedTwoPointFunction):
            assert newKernel.getParamPtrAddr() == newKernel.scaling.getParamPtrAddr()
        if isinstance(self.scaling, parametrizedTwoPointFunction):
            assert self.getParamPtrAddr() == self.scaling.getParamPtrAddr()
        return newKernel

    def getComplementKernel(self):
        "Get the complement kernel."
        raise NotImplementedError()
        from . kernels import getKernel
        newKernel = getKernel(dim=self.dim, kernel=self.kernelType, horizon=self.horizon, interaction=self.interaction.getComplement(), scaling=self.scaling, piecewise=self.piecewise)
        return newKernel

    def __repr__(self):
        extraInfo = {}
        for p in self.getKernelParams():
            extraInfo[p] = self.getKernelParam(p)
        if self.kernelType == INDICATOR:
            kernelName = 'indicator'
        elif self.kernelType == PERIDYNAMIC:
            kernelName = 'peridynamic'
        elif self.kernelType == GAUSSIAN:
            kernelName = 'Gaussian'
        elif self.kernelType == EXPONENTIAL:
            kernelName = 'exponential'
        elif self.kernelType == POLYNOMIAL:
            kernelName = 'polynomial'
        elif self.kernelType == LOGINVERSEDISTANCE:
            kernelName = 'logInverseDistance'
        elif self.kernelType == MONOMIAL:
            kernelName = 'monomial'
        else:
            raise NotImplementedError()
        if len(extraInfo) > 0:
            extraInfo = ', '.join([str(k)+'='+str(v) for k, v in extraInfo.items()])
            return "{}({}{}, {}, {}, {})".format(self.__class__.__name__, kernelName, '' if not self.boundary else '-boundary', repr(self.interaction), self.scaling, extraInfo)
        else:
            return "{}({}{}, {}, {})".format(self.__class__.__name__, kernelName, '' if not self.boundary else '-boundary', repr(self.interaction), self.scaling)

    def __reduce__(self):
        return ComplexKernel, (self.dim, self.kernelType, self.horizon, self.interaction, self.scaling, self.phi, self.piecewise, self.boundary, self.valueSize)

    def plot(self, x0=None):
        "Plot the kernel function."
        from matplotlib import ticker
        import matplotlib.pyplot as plt
        if x0 is None:
            x0 = np.zeros((self.dim), dtype=REAL)
        self.evalParams(x0, x0)
        if self.finiteHorizon:
            delta = self.horizonValue
        else:
            delta = 2.
        x = np.linspace(-1.1*delta, 1.1*delta, 201)
        if self.dim == 1:
            vals = np.zeros_like(x)
            for i in range(x.shape[0]):
                y = x0+np.array([x[i]], dtype=REAL)
                if np.linalg.norm(x0-y) > 1e-9 or self.singularityValue >= 0:
                    vals[i] = self(x0, y)
                else:
                    vals[i] = np.nan
            plt.plot(x, vals)
            plt.yscale('log')
            if not self.finiteHorizon:
                plt.xlim([x[0], x[x.shape[0]-1]])
            if self.singularityValue < 0:
                plt.ylim(top=np.nanmax(vals))
            plt.xlabel('$x-y$')
        elif self.dim == 2:
            X, Y = np.meshgrid(x, x)
            Z = np.zeros_like(X)
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    y = x0+np.array([x[i], x[j]], dtype=REAL)
                    if np.linalg.norm(x0-y) > 1e-9 or self.singularityValue >= 0:
                        Z[j, i] = self(x0, y)
                    else:
                        Z[j, i] = np.nan
            levels = np.logspace(np.log10(Z[np.absolute(Z)>0].min()),
                                 np.log10(Z[np.absolute(Z)>0].max()), 10)
            if levels[0] < levels[levels.shape[0]-1]:
                plt.contourf(X, Y, Z, locator=ticker.LogLocator(),
                             levels=levels)
            else:
                plt.contourf(X, Y, Z)
            plt.axis('equal')
            plt.colorbar()
            plt.xlabel('$y_1-x_1$')
            plt.ylabel('$y_2-x_2$')

    def getBoundaryKernel(self):
        "Get the boundary kernel. This is the kernel that corresponds to the elimination of a subdomain via Gauss theorem."
        cdef:
            Kernel newKernel
        scaling = deepcopy(self.scaling)
        if self.phi is not None:
            phi = deepcopy(self.phi)
        else:
            phi = None

        from . kernels import getIntegrableKernel
        newKernel = getIntegrableKernel(kernel=self.kernelType,
                                        dim=self.dim,
                                        horizon=deepcopy(self.horizon),
                                        interaction=None,
                                        scaling=scaling,
                                        phi=phi,
                                        piecewise=self.piecewise,
                                        boundary=True)
        setREAL(newKernel.c_kernel_params, fEXPONENTINVERSE, getREAL(self.c_kernel_params, fEXPONENTINVERSE))
        if isinstance(newKernel.scaling, parametrizedTwoPointFunction):
            assert newKernel.getParamPtrAddr() == newKernel.scaling.getParamPtrAddr()
        if isinstance(self.scaling, parametrizedTwoPointFunction):
            assert self.getParamPtrAddr() == self.scaling.getParamPtrAddr()
        return newKernel

    def __dealloc__(self):
        PyMem_Free(self.c_kernel_params)


cdef class ComplexKernel(ComplextwoPointFunction):
    """A kernel functions that can be used to define a nonlocal operator."""

    def __init__(self, INDEX_t dim, kernelType kType, function horizon, interactionDomain interaction, twoPointFunction scaling, twoPointFunction phi, BOOL_t piecewise=True, BOOL_t boundary=False, INDEX_t valueSize=1, REAL_t max_horizon=np.nan, **kwargs):
        cdef:
            parametrizedTwoPointFunction parametrizedScaling
            int i

        self.dim = dim
        self.valueSize = valueSize
        self.kernelType = kType
        self.piecewise = piecewise
        self.boundary = boundary

        self.c_kernel_params = PyMem_Malloc(NUM_KERNEL_PARAMS*OFFSET)
        for i in range(NUM_KERNEL_PARAMS):
            (<void**>(self.c_kernel_params+i*OFFSET))[0] = NULL
        setINDEX(self.c_kernel_params, fKDIM, dim)

        symmetric = isinstance(horizon, constant) and scaling.symmetric and interaction.symmetric and (phi is None or phi.symmetric)
        super(ComplexKernel, self).__init__(symmetric, valueSize)

        if self.kernelType == GREENS_2D:
            greensLambda = kwargs.get('greens2D_lambda', np.nan)
            setREAL(self.c_kernel_params, fGREENS_LAMBDA, -greensLambda.imag)
            self.min_singularity = 0.
            self.max_singularity = 0.
            self.singularityValue = 0.
        elif self.kernelType == GREENS_3D:
            greensLambda = kwargs.get('greens3D_lambda', np.nan)
            setCOMPLEX(self.c_kernel_params, fGREENS_LAMBDA, greensLambda)
            self.min_singularity = -1.
            self.max_singularity = -1.
            self.singularityValue = -1.

        self.horizon = horizon
        self.variableHorizon = not isinstance(self.horizon, constant)
        if self.variableHorizon:
            self.horizonValue2 = np.nan
            self.max_horizon = max_horizon
            self.finiteHorizon = True
            (<void**>(self.c_kernel_params+fHORIZONFUN))[0] = <void*>horizon
        else:
            self.horizonValue = self.horizon.value
            self.max_horizon = self.horizon.value
            self.finiteHorizon = self.horizon.value != np.inf
            if self.kernelType == GAUSSIAN:
                setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)

        self.interaction = interaction
        self.complement = self.interaction.complement
        (<void**>(self.c_kernel_params+fINTERACTION))[0] = <void*>self.interaction
        self.interaction.setParams(self.c_kernel_params)

        self.phi = phi
        if phi is not None:
            scaling = phi*scaling
        self.scaling = scaling
        self.variableScaling = not isinstance(self.scaling, (constantFractionalLaplacianScaling, constantTwoPoint))
        if self.variableScaling:
            if isinstance(self.scaling, parametrizedTwoPointFunction):
                parametrizedScaling = self.scaling
                parametrizedScaling.setParams(self.c_kernel_params)
            self.scalingValue = np.nan
            (<void**>(self.c_kernel_params+fSCALINGFUN))[0] = <void*>self.scaling
        else:
            self.scalingValue = self.scaling.value

        self.variable = self.variableHorizon or self.variableScaling

        if self.piecewise:
            if not self.boundary:
                if dim == 2:
                    if self.kernelType == GREENS_2D:
                        self.kernelFun = greens2Dcomplex
                elif dim == 3:
                    if self.kernelType == GREENS_3D:
                        self.kernelFun = greens3Dcomplex
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        else:
            self.kernelFun = updateAndEvalIntegrableComplex

            if not self.boundary:
                if dim == 2:
                    if self.kernelType == GREENS_2D:
                        setComplexFun(self.c_kernel_params, fEVAL, greens2Dcomplex)
                elif dim == 2:
                    if self.kernelType == GREENS_3D:
                        setComplexFun(self.c_kernel_params, fEVAL, greens3Dcomplex)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    @property
    def singularityValue(self):
        "The order of the singularity."
        return getREAL(self.c_kernel_params, fSINGULARITY)

    @singularityValue.setter
    def singularityValue(self, REAL_t singularity):
        setREAL(self.c_kernel_params, fSINGULARITY, singularity)

    cdef REAL_t getSingularityValue(self):
        return getREAL(self.c_kernel_params, fSINGULARITY)

    cdef void setSingularityValue(self, REAL_t singularity):
        setREAL(self.c_kernel_params, fSINGULARITY, singularity)

    @property
    def horizonValue(self):
        "The value of the interaction horizon."
        return sqrt(getREAL(self.c_kernel_params, fHORIZON2))

    @horizonValue.setter
    def horizonValue(self, REAL_t horizon):
        setREAL(self.c_kernel_params, fHORIZON2, horizon**2)

    cdef REAL_t getHorizonValue(self):
        return sqrt(getREAL(self.c_kernel_params, fHORIZON2))

    cdef void setHorizonValue(self, REAL_t horizon):
        setREAL(self.c_kernel_params, fHORIZON2, horizon**2)

    @property
    def horizonValue2(self):
        return getREAL(self.c_kernel_params, fHORIZON2)

    cdef REAL_t getHorizonValue2(self):
        return getREAL(self.c_kernel_params, fHORIZON2)

    @horizonValue2.setter
    def horizonValue2(self, REAL_t horizon2):
        setREAL(self.c_kernel_params, fHORIZON2, horizon2)

    @property
    def scalingValue(self):
        "The value of the scaling factor."
        return getREAL(self.c_kernel_params, fSCALING)

    @scalingValue.setter
    def scalingValue(self, REAL_t scaling):
        setREAL(self.c_kernel_params, fSCALING, scaling)

    cdef REAL_t getScalingValue(self):
        return getREAL(self.c_kernel_params, fSCALING)

    cdef void setScalingValue(self, REAL_t scaling):
        setREAL(self.c_kernel_params, fSCALING, scaling)

    cdef void evalParamsOnSimplices(self, REAL_t[::1] center1, REAL_t[::1] center2, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        # Set the horizon.
        if self.variableHorizon:
            self.horizonValue = self.horizon.eval(center1)

    cdef void evalParams(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t scalingValue
        if self.piecewise:
            if self.variableHorizon:
                self.setHorizonValue(self.horizon.eval(x))
                if self.kernelType == GAUSSIAN:
                    setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
            if self.variableScaling:
                self.scaling.evalPtr(x.shape[0], &x[0], &y[0], &scalingValue)
                self.scalingValue = scalingValue

    def evalParams_py(self, REAL_t[::1] x, REAL_t[::1] y):
        "Evaluate the kernel parameters."
        cdef:
            REAL_t scalingValue
        if self.piecewise:
            self.evalParams(x, y)
        else:
            if self.variableHorizon:
                self.setHorizonValue(self.horizon.eval(x))
                if self.kernelType == GAUSSIAN:
                    setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
            if self.variableScaling:
                self.scaling.evalPtr(x.shape[0], &x[0], &y[0], &scalingValue)
                self.scalingValue = scalingValue

    cdef void evalParamsPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t[::1] xA
            REAL_t scalingValue
        if self.piecewise:
            if self.variableHorizon:
                xA = <REAL_t[:dim]> x
                self.setHorizonValue(self.horizon.eval(xA))
                if self.kernelType == GAUSSIAN:
                    setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
            if self.variableScaling:
                self.scaling.evalPtr(dim, x, y, &scalingValue)
                self.scalingValue = scalingValue

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, COMPLEX_t[::1] value):
        value[0] = self.kernelFun(&x[0], &y[0], self.c_kernel_params)

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, COMPLEX_t* value):
        value[0] = self.kernelFun(x, y, self.c_kernel_params)

    def __call__(self, REAL_t[::1] x, REAL_t[::1] y, BOOL_t callEvalParams=True):
        "Evaluate the kernel."
        if self.piecewise and callEvalParams:
            self.evalParams(x, y)
        return self.kernelFun(&x[0], &y[0], self.c_kernel_params)

    def eval_py(self, REAL_t[::1] x, REAL_t[::1] y, COMPLEX_t[::1] vec, BOOL_t callEvalParams=True):
        "Evaluate the kernel."
        if self.piecewise and callEvalParams:
            self.evalParams(x, y)
        self.eval(x, y, vec)

    def getModifiedKernel(self,
                          function horizon=None,
                          twoPointFunction scaling=None):
        cdef:
            Kernel newKernel
        if horizon is None:
            horizon = self.horizon
            interaction = self.interaction
        else:
            if scaling is None and isinstance(self.scaling, (variableFractionalLaplacianScaling,
                                                             variableIntegrableScaling)):
                scaling = self.scaling.getScalingWithDifferentHorizon()
            interaction = type(self.interaction)()
        if scaling is None:
            scaling = self.scaling
        from . kernels import getKernel
        newKernel = getKernel(dim=self.dim, kernel=self.kernelType, horizon=horizon, interaction=interaction, scaling=scaling, piecewise=self.piecewise)
        setREAL(newKernel.c_kernel_params, fEXPONENTINVERSE, getREAL(self.c_kernel_params, fEXPONENTINVERSE))
        if isinstance(newKernel.scaling, parametrizedTwoPointFunction):
            assert newKernel.getParamPtrAddr() == newKernel.scaling.getParamPtrAddr()
        if isinstance(self.scaling, parametrizedTwoPointFunction):
            assert self.getParamPtrAddr() == self.scaling.getParamPtrAddr()
        return newKernel

    def getComplementKernel(self):
        "Get the complement kernel."
        raise NotImplementedError()
        from . kernels import getKernel
        newKernel = getKernel(dim=self.dim, kernel=self.kernelType, horizon=self.horizon, interaction=self.interaction.getComplement(), scaling=self.scaling, piecewise=self.piecewise)
        return newKernel

    def __repr__(self):
        if self.kernelType == GREENS_2D:
            kernelName = 'greens2D'
        if self.kernelType == GREENS_3D:
            kernelName = 'greens3D'
        else:
            raise NotImplementedError()
        return "{}({}{}, {}, {})".format(self.__class__.__name__, kernelName, '' if not self.boundary else '-boundary', repr(self.interaction), self.scaling)

    def __reduce__(self):
        return Kernel, (self.dim, self.kernelType, self.horizon, self.interaction, self.scaling, self.phi, self.piecewise, self.boundary, self.singularityValue)

    def plot(self, x0=None):
        "Plot the kernel function."
        from matplotlib import ticker
        import matplotlib.pyplot as plt
        if x0 is None:
            x0 = np.zeros((self.dim), dtype=REAL)
        self.evalParams(x0, x0)
        if self.finiteHorizon:
            delta = self.horizonValue
        else:
            delta = 2.
        x = np.linspace(-1.1*delta, 1.1*delta, 201)
        if self.dim == 1:
            vals = np.zeros_like(x)
            for i in range(x.shape[0]):
                y = x0+np.array([x[i]], dtype=REAL)
                if np.linalg.norm(x0-y) > 1e-9 or self.singularityValue >= 0:
                    vals[i] = self(x0, y)
                else:
                    vals[i] = np.nan
            plt.plot(x, vals)
            plt.yscale('log')
            if not self.finiteHorizon:
                plt.xlim([x[0], x[x.shape[0]-1]])
            if self.singularityValue < 0:
                plt.ylim(top=np.nanmax(vals))
            plt.xlabel('$x-y$')
        elif self.dim == 2:
            X, Y = np.meshgrid(x, x)
            Z = np.zeros_like(X)
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    y = x0+np.array([x[i], x[j]], dtype=REAL)
                    if np.linalg.norm(x0-y) > 1e-9 or self.singularityValue >= 0:
                        Z[i, j] = self(x0, y)
                    else:
                        Z[i, j] = np.nan
            levels = np.logspace(np.log10(Z[np.absolute(Z)>0].min()),
                                 np.log10(Z[np.absolute(Z)>0].max()), 10)
            if levels[0] < levels[levels.shape[0]-1]:
                plt.contourf(X, Y, Z, locator=ticker.LogLocator(),
                             levels=levels)
            else:
                plt.contourf(X, Y, Z)
            plt.axis('equal')
            plt.colorbar()
            plt.xlabel('$x_1-y_1$')
            plt.ylabel('$x_2-y_2$')

    def getBoundaryKernel(self):
        "Get the boundary kernel. This is the kernel that corresponds to the elimination of a subdomain via Gauss theorem."
        cdef:
            Kernel newKernel
        scaling = deepcopy(self.scaling)
        if self.phi is not None:
            phi = deepcopy(self.phi)
        else:
            phi = None

        from . kernels import getIntegrableKernel
        newKernel = getIntegrableKernel(kernel=self.kernelType,
                                        dim=self.dim,
                                        horizon=deepcopy(self.horizon),
                                        interaction=None,
                                        scaling=scaling,
                                        phi=phi,
                                        piecewise=self.piecewise,
                                        boundary=True)
        setREAL(newKernel.c_kernel_params, fEXPONENTINVERSE, getREAL(self.c_kernel_params, fEXPONENTINVERSE))
        if isinstance(newKernel.scaling, parametrizedTwoPointFunction):
            assert newKernel.getParamPtrAddr() == newKernel.scaling.getParamPtrAddr()
        if isinstance(self.scaling, parametrizedTwoPointFunction):
            assert self.getParamPtrAddr() == self.scaling.getParamPtrAddr()
        return newKernel

    def __dealloc__(self):
        PyMem_Free(self.c_kernel_params)


cdef class FractionalKernel(Kernel):
    """A kernel functions that can be used to define a fractional operator."""

    def __init__(self,
                 INDEX_t dim,
                 fractionalOrderBase s,
                 function horizon,
                 interactionDomain interaction,
                 twoPointFunction scaling,
                 twoPointFunction phi=None,
                 BOOL_t piecewise=True,
                 BOOL_t boundary=False,
                 INDEX_t derivative=0,
                 REAL_t tempered=0.,
                 REAL_t max_horizon=np.nan):
        if derivative == 0:
            valueSize = 1
        elif derivative == 1:
            valueSize = s.numParameters
        elif derivative == 2:
            valueSize = s.numParameters**2
            self.tempVec = uninitialized((s.numParameters), dtype=REAL)
        else:
            valueSize = 1

        super(FractionalKernel, self).__init__(dim, FRACTIONAL, horizon, interaction, scaling, phi, piecewise, boundary, valueSize, max_horizon)

        self.symmetric = s.symmetric and isinstance(horizon, constant) and scaling.symmetric and interaction.symmetric and (phi is None or phi.symmetric)
        self.derivative = derivative
        self.temperedValue = tempered

        self.s = s
        self.variableOrder = isinstance(self.s, variableFractionalOrder)
        self.variableSingularity = self.variableOrder
        if self.variableOrder:
            self.sValue = np.nan
            (<void**>(self.c_kernel_params+fORDERFUN))[0] = <void*>s
            self.singularityValue = np.nan
            if not self.boundary:
                self.min_singularity = -self.dim-2*self.s.min
                self.max_singularity = -self.dim-2*self.s.max
            else:
                self.min_singularity = 1.-self.dim-2*self.s.min
                self.max_singularity = 1.-self.dim-2*self.s.max
        else:
            self.sValue = self.s.value
            if not self.boundary:
                self.singularityValue = -self.dim-2*self.sValue
            else:
                self.singularityValue = 1.-self.dim-2*self.sValue
            self.min_singularity = self.singularityValue
            self.max_singularity = self.singularityValue

        self.variable = self.variableOrder or self.variableHorizon or self.variableScaling

        if self.piecewise:
            if isinstance(self.horizon, constant) and self.horizon.value == np.inf:
                if not self.boundary:
                    if tempered == 0.:
                        if dim == 1:
                            self.kernelFun = fracKernelInfinite1D
                        elif dim == 2:
                            self.kernelFun = fracKernelInfinite2D
                        elif dim == 3:
                            self.kernelFun = fracKernelInfinite3D
                        else:
                            raise NotImplementedError()
                    else:
                        if dim == 1:
                            self.kernelFun = temperedFracKernelInfinite1D
                        elif dim == 2:
                            self.kernelFun = temperedFracKernelInfinite2D
                        elif dim == 3:
                            self.kernelFun = temperedFracKernelInfinite3D
                        else:
                            raise NotImplementedError()
                else:
                    if tempered == 0.:
                        if dim == 1:
                            self.kernelFun = fracKernelInfinite1Dboundary
                        elif dim == 2:
                            self.kernelFun = fracKernelInfinite2Dboundary
                        elif dim == 3:
                            self.kernelFun = fracKernelInfinite3Dboundary
                        else:
                            raise NotImplementedError()
                    else:
                        if dim == 1:
                            self.kernelFun = temperedFracKernelInfinite1Dboundary
                        elif dim == 2:
                            self.kernelFun = temperedFracKernelInfinite2Dboundary
                        elif dim == 3:
                            self.kernelFun = temperedFracKernelInfinite3Dboundary
                        else:
                            raise NotImplementedError()
            else:
                if not self.boundary:
                    if dim == 1:
                        self.kernelFun = fracKernelFinite1D
                    elif dim == 2:
                        self.kernelFun = fracKernelFinite2D
                    elif dim == 3:
                        self.kernelFun = fracKernelFinite3D
                    else:
                        raise NotImplementedError()
                else:
                    if dim == 1:
                        self.kernelFun = fracKernelFinite1Dboundary
                    elif dim == 2:
                        self.kernelFun = fracKernelFinite2Dboundary
                    elif dim == 3:
                        self.kernelFun = fracKernelFinite3Dboundary
                    else:
                        raise NotImplementedError()
        else:
            self.kernelFun = updateAndEvalFractional

            if isinstance(self.horizon, constant) and self.horizon.value == np.inf:
                if not self.boundary:
                    if tempered == 0.:
                        if dim == 1:
                            setFun(self.c_kernel_params, fEVAL, fracKernelInfinite1D)
                        elif dim == 2:
                            setFun(self.c_kernel_params, fEVAL, fracKernelInfinite2D)
                        elif dim == 3:
                            setFun(self.c_kernel_params, fEVAL, fracKernelInfinite3D)
                        else:
                            raise NotImplementedError()
                    else:
                        if dim == 1:
                            setFun(self.c_kernel_params, fEVAL, temperedFracKernelInfinite1D)
                        elif dim == 2:
                            setFun(self.c_kernel_params, fEVAL, temperedFracKernelInfinite2D)
                        elif dim == 3:
                            setFun(self.c_kernel_params, fEVAL, temperedFracKernelInfinite3D)
                        else:
                            raise NotImplementedError()
                else:
                    if tempered == 0.:
                        if dim == 1:
                            setFun(self.c_kernel_params, fEVAL, fracKernelInfinite1Dboundary)
                        elif dim == 2:
                            setFun(self.c_kernel_params, fEVAL, fracKernelInfinite2Dboundary)
                        elif dim == 3:
                            setFun(self.c_kernel_params, fEVAL, fracKernelInfinite3Dboundary)
                        else:
                            raise NotImplementedError()
                    else:
                        if dim == 1:
                            setFun(self.c_kernel_params, fEVAL, temperedFracKernelInfinite1Dboundary)
                        elif dim == 2:
                            setFun(self.c_kernel_params, fEVAL, temperedFracKernelInfinite2Dboundary)
                        elif dim == 3:
                            setFun(self.c_kernel_params, fEVAL, temperedFracKernelInfinite3Dboundary)
                        else:
                            raise NotImplementedError()
            else:
                if not self.boundary:
                    if dim == 1:
                        setFun(self.c_kernel_params, fEVAL, fracKernelFinite1D)
                    elif dim == 2:
                        setFun(self.c_kernel_params, fEVAL, fracKernelFinite2D)
                    elif dim == 3:
                        setFun(self.c_kernel_params, fEVAL, fracKernelFinite3D)
                    else:
                        raise NotImplementedError()
                else:
                    if dim == 1:
                        setFun(self.c_kernel_params, fEVAL, fracKernelFinite1Dboundary)
                    elif dim == 2:
                        setFun(self.c_kernel_params, fEVAL, fracKernelFinite2Dboundary)
                    elif dim == 3:
                        setFun(self.c_kernel_params, fEVAL, fracKernelFinite3Dboundary)
                    else:
                        raise NotImplementedError()

    @property
    def sValue(self):
        "The value of the fractional order"
        return getREAL(self.c_kernel_params, fS)

    @sValue.setter
    def sValue(self, REAL_t s):
        setREAL(self.c_kernel_params, fS, s)

    cdef REAL_t getsValue(self):
        return getREAL(self.c_kernel_params, fS)

    cdef void setsValue(self, REAL_t s):
        setREAL(self.c_kernel_params, fS, s)

    @property
    def temperedValue(self):
        "The value of the tempering parameter"
        return getREAL(self.c_kernel_params, fTEMPERED)

    @temperedValue.setter
    def temperedValue(self, REAL_t tempered):
        setREAL(self.c_kernel_params, fTEMPERED, tempered)

    cdef REAL_t gettemperedValue(self):
        return getREAL(self.c_kernel_params, fTEMPERED)

    cdef void settemperedValue(self, REAL_t tempered):
        setREAL(self.c_kernel_params, fTEMPERED, tempered)

    cdef void evalParamsOnSimplices(self, REAL_t[::1] center1, REAL_t[::1] center2, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        # Set the max singularity and the horizon.
        cdef:
            REAL_t sValue, sValue2
        if self.variableOrder:
            if self.s.symmetric:
                self.s.evalPtr(center1.shape[0], &center1[0], &center2[0], &sValue)
            else:
                sValue = 0.
                self.s.evalPtr(center1.shape[0], &center1[0], &center2[0], &sValue2)
                sValue = max(sValue, sValue2)
                self.s.evalPtr(center1.shape[0], &center2[0], &center1[0], &sValue2)
                sValue = max(sValue, sValue2)
                for i in range(simplex1.shape[0]):
                    self.s.evalPtr(center1.shape[0], &simplex1[i, 0], &center2[0], &sValue2)
                    sValue = max(sValue, sValue2)
                for i in range(simplex2.shape[0]):
                    self.s.evalPtr(center1.shape[0], &simplex2[i, 0], &center1[0], &sValue2)
                    sValue = max(sValue, sValue2)
            if not self.boundary:
                self.setSingularityValue(-self.dim-2*sValue)
            else:
                self.setSingularityValue(1-self.dim-2*sValue)
        if self.variableHorizon:
            self.setHorizonValue(self.horizon.eval(center1))

    cdef void evalParams(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t sValue, scalingValue
        if self.piecewise:
            if self.variableOrder:
                self.s.evalPtr(x.shape[0], &x[0], &y[0], &sValue)
                if not self.boundary:
                    self.setSingularityValue(-self.dim-2*sValue)
                else:
                    self.setSingularityValue(1-self.dim-2*sValue)
                self.setsValue(sValue)
            if self.variableHorizon:
                self.setHorizonValue(self.horizon.eval(x))
            if self.variableScaling:
                self.scaling.evalPtr(x.shape[0], &x[0], &y[0], &scalingValue)
                self.setScalingValue(scalingValue)

    cdef void evalParamsPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t[::1] xA
            REAL_t sValue, scalingValue
        if self.piecewise:
            if self.variableOrder:
                self.s.evalPtr(dim, x, y, &sValue)
                if not self.boundary:
                    self.setSingularityValue(-self.dim-2*sValue)
                else:
                    self.setSingularityValue(1-self.dim-2*sValue)
                self.setsValue(sValue)
            if self.variableHorizon:
                xA = <REAL_t[:dim]> x
                self.setHorizonValue(self.horizon.eval(xA))
            if self.variableScaling:
                self.scaling.evalPtr(dim, x, y, &scalingValue)
                self.setScalingValue(scalingValue)

    def evalParams_py(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t sValue, scalingValue
        if self.piecewise:
            self.evalParams(x, y)
        else:
            if self.variableOrder:
                self.s.evalPtr(x.shape[0], &x[0], &y[0], &sValue)
                self.setsValue(sValue)
                if not self.boundary:
                    self.setSingularityValue(-self.dim-2*sValue)
                else:
                    self.setSingularityValue(1-self.dim-2*sValue)
            if self.variableHorizon:
                self.setHorizonValue(self.horizon.eval(x))
            if self.variableScaling:
                self.scaling.evalPtr(x.shape[0], &x[0], &y[0], & scalingValue)
                self.setScalingValue(scalingValue)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] vec):
        cdef:
            INDEX_t i, j, k
            REAL_t fac
        if self.derivative == 0:
            vec[0] = self.kernelFun(&x[0], &y[0], self.c_kernel_params)
        elif self.derivative == 1:
            fac = self.kernelFun(&x[0], &y[0], self.c_kernel_params)
            self.s.evalGrad(x, y, vec)
            for i in range(self.valueSize):
                vec[i] *= fac
        elif self.derivative == 2:
            fac = self.kernelFun(&x[0], &y[0], self.c_kernel_params)
            self.s.evalGrad(x, y, self.tempVec)
            k = 0
            for i in range(self.s.numParameters):
                for j in range(self.s.numParameters):
                    vec[k] = fac*self.tempVec[i]*self.tempVec[j]
                    k += 1

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* vec):
        cdef:
            INDEX_t i, j, k
            REAL_t fac
        if self.derivative == 0:
            vec[0] = self.kernelFun(x, y, self.c_kernel_params)
        elif self.derivative == 1:
            fac = self.kernelFun(x, y, self.c_kernel_params)
            self.s.evalGradPtr(dim, x, y, self.valueSize, vec)
            for i in range(self.valueSize):
                vec[i] *= fac
        elif self.derivative == 2:
            fac = self.kernelFun(&x[0], &y[0], self.c_kernel_params)
            self.s.evalGradPtr(dim, x, y, self.s.numParameters, &self.tempVec[0])
            k = 0
            for i in range(self.s.numParameters):
                for j in range(self.s.numParameters):
                    vec[k] = fac*self.tempVec[i]*self.tempVec[j]
                    k += 1

    def getModifiedKernel(self,
                          fractionalOrderBase s=None,
                          function horizon=None,
                          twoPointFunction scaling=None):
        if s is None:
            s = self.s
        else:
            if scaling is None and isinstance(self.scaling, variableFractionalLaplacianScaling):
                raise NotImplementedError()
        if horizon is None:
            horizon = self.horizon
            interaction = self.interaction
        else:
            if scaling is None and isinstance(self.scaling, variableFractionalLaplacianScaling):
                scaling = self.scaling.getScalingWithDifferentHorizon()
            elif scaling is None and isinstance(self.scaling, productParametrizedTwoPoint):
                if isinstance(self.scaling.f2, variableFractionalLaplacianScaling):
                    scaling = self.scaling.f1*self.scaling.f2.getScalingWithDifferentHorizon()
                elif isinstance(self.scaling.f1, variableFractionalLaplacianScaling):
                    scaling = self.scaling.f2.getScalingWithDifferentHorizon()*self.scaling.f2
                else:
                    raise NotImplementedError()
            interaction = deepcopy(self.interaction)
        if scaling is None:
            scaling = self.scaling
        from . kernels import getFractionalKernel
        newKernel = getFractionalKernel(dim=self.dim, s=s, horizon=horizon, interaction=interaction, scaling=scaling, piecewise=self.piecewise, boundary=self.boundary, derivative=self.derivative)
        if isinstance(newKernel.scaling, parametrizedTwoPointFunction):
            assert newKernel.getParamPtrAddr() == newKernel.scaling.getParamPtrAddr()
        if isinstance(self.scaling, parametrizedTwoPointFunction):
            assert self.getParamPtrAddr() == self.scaling.getParamPtrAddr()
        return newKernel

    def getBoundaryKernel(self):
        "Get the boundary kernel. This is the kernel that corresponds to the elimination of a subdomain via Gauss theorem."
        cdef:
            constantFractionalLaplacianScalingDerivative scal
            variableFractionalLaplacianScalingWithDifferentHorizon scalVarDiffHorizon
            variableFractionalLaplacianScaling scalVar

        s = deepcopy(self.s)
        if not self.variableOrder:
            fac = constantTwoPoint(1/s.value)
        else:
            fac = inverseTwoPoint(s)
        phi = fac

        if isinstance(self.scaling, constantFractionalLaplacianScalingDerivative):
            scal = self.scaling
            scaling = constantFractionalLaplacianScalingDerivative(scal.dim, scal.s, scal.horizon, scal.normalized, True, scal.derivative, scal.tempered)
        elif isinstance(self.scaling, variableFractionalLaplacianScalingWithDifferentHorizon):
            scalVarDiffHorizon = self.scaling
            scaling = variableFractionalLaplacianScalingWithDifferentHorizon(scalVarDiffHorizon.symmetric, scalVarDiffHorizon.normalized, True, scalVarDiffHorizon.derivative, scalVarDiffHorizon.horizonFun)
        elif isinstance(self.scaling, variableFractionalLaplacianScaling):
            scalVar = self.scaling
            scaling = variableFractionalLaplacianScaling(scalVar.symmetric, scalVar.normalized, True, scalVar.derivative)
        else:
            scaling = deepcopy(self.scaling)

        from . kernels import getFractionalKernel
        newKernel = getFractionalKernel(dim=self.dim,
                                        s=s,
                                        horizon=deepcopy(self.horizon),
                                        interaction=None,
                                        scaling=scaling,
                                        phi=phi,
                                        piecewise=self.piecewise,
                                        boundary=True,
                                        derivative=self.derivative)
        if isinstance(newKernel.scaling, parametrizedTwoPointFunction):
            assert newKernel.getParamPtrAddr() == newKernel.scaling.getParamPtrAddr(), (newKernel.getParamPtrAddr(), newKernel.scaling.getParamPtrAddr())
        if isinstance(self.scaling, parametrizedTwoPointFunction):
            assert self.getParamPtrAddr() == self.scaling.getParamPtrAddr(), (self.getParamPtrAddr(), self.scaling.getParamPtrAddr())
        return newKernel

    def getComplementKernel(self):
        from . kernels import getFractionalKernel
        newKernel = getFractionalKernel(dim=self.dim, s=self.s, horizon=self.horizon, interaction=self.interaction.getComplement(), scaling=self.scaling, piecewise=self.piecewise)
        return newKernel

    def getDerivativeKernel(self):
        cdef:
            constantFractionalLaplacianScaling scal
            variableFractionalLaplacianScaling scalVar
        if isinstance(self.scaling, constantFractionalLaplacianScaling):
            scal = self.scaling
            scaling = constantFractionalLaplacianScalingDerivative(scal.dim, scal.s, scal.horizon, True, False, 1, scal.tempered)
        elif isinstance(self.scaling, variableFractionalLaplacianScaling):
            scalVar = self.scaling
            scaling = variableFractionalLaplacianScaling(scalVar.symmetric, scalVar.normalized, scalVar.boundary, 1)
        elif isinstance(self.scaling, constantTwoPoint):
            scaling = constantFractionalLaplacianScalingDerivative(self.dim, self.sValue, np.nan, False, self.boundary, 1, 0.)
        else:
            raise NotImplementedError()
        return FractionalKernel(self.dim, self.s, self.horizon, self.interaction, scaling, self.phi, False, self.boundary, 1, self.temperedValue)

    def getHessianKernel(self):
        cdef:
            constantFractionalLaplacianScaling scal
            variableFractionalLaplacianScaling scalVar
        if isinstance(self.scaling, constantFractionalLaplacianScaling):
            scal = self.scaling
            scaling = constantFractionalLaplacianScalingDerivative(scal.dim, scal.s, scal.horizon, True, False, 2, scal.tempered)
        elif isinstance(self.scaling, constantTwoPoint):
            scaling = constantFractionalLaplacianScalingDerivative(self.dim, self.sValue, np.nan, False, self.boundary, 2, 0.)
        elif isinstance(self.scaling, variableFractionalLaplacianScaling):
            scalVar = self.scaling
            scaling = variableFractionalLaplacianScaling(scalVar.symmetric, scalVar.normalized, scalVar.boundary, 2)
        else:
            raise NotImplementedError()
        return FractionalKernel(self.dim, self.s, self.horizon, self.interaction, scaling, self.phi, False, self.boundary, 2, self.temperedValue)

    def __repr__(self):
        if self.temperedValue != 0.:
            return "kernel(tempered-fractional{}, {}, {}, {}, {})".format('' if not self.boundary else '-boundary', self.s, repr(self.interaction), self.scaling, self.temperedValue)
        else:
            return "kernel(fractional{}, {}, {}, {})".format('' if not self.boundary else '-boundary', self.s, repr(self.interaction), self.scaling)

    def __reduce__(self):
        return FractionalKernel, (self.dim, self.s, self.horizon, self.interaction, self.scaling, self.phi, self.piecewise, self.boundary, self.derivative, self.temperedValue)


cdef class RangedFractionalKernel(FractionalKernel):
    def __init__(self,
                 INDEX_t dim,
                 admissibleOrders,
                 function horizon,
                 BOOL_t normalized=True,
                 REAL_t tempered=0.,
                 REAL_t errorBound=-1.,
                 INDEX_t M_min=1, INDEX_t M_max=20,
                 REAL_t xi=0.):
        self.dim = dim
        assert admissibleOrders.numParams == 1, "Cannot handle {} params".format(admissibleOrders.numParams)
        self.admissibleOrders = admissibleOrders
        assert isinstance(horizon, constant)
        self.horizon = horizon
        if isinstance(horizon, constant) and horizon.value == np.inf:
            self.interaction = fullSpace()
        else:
            self.interaction = ball2_retriangulation()
        self.normalized = normalized
        self.tempered = tempered

        self.setOrder(admissibleOrders.getLowerBounds()[0])

        self.errorBound = errorBound
        self.M_min = M_min
        self.M_max = M_max
        self.xi = xi

    def setOrder(self, REAL_t s):
        assert self.admissibleOrders.isAdmissible(s)
        sFun = constFractionalOrder(s)
        dim = self.dim
        horizon = self.horizon
        interactionDomain = self.interaction
        tempered = self.tempered
        if self.normalized:
            scaling = constantFractionalLaplacianScaling(dim, sFun.value, horizon.value, tempered=tempered)
        else:
            scaling = constantTwoPoint(0.5)
        super(RangedFractionalKernel, self).__init__(dim, sFun, horizon, interactionDomain, scaling, tempered=tempered)

    def getFrozenKernel(self, REAL_t s):
        assert self.admissibleOrders.isAdmissible(s)
        sFun = constFractionalOrder(s)
        dim = self.dim
        horizon = self.horizon
        tempered = self.tempered
        interactionDomain = self.interaction
        if self.normalized:
            scaling = constantFractionalLaplacianScaling(dim, sFun.value, horizon.value, tempered=tempered)
        else:
            scaling = constantTwoPoint(0.5)
        return FractionalKernel(dim, sFun, horizon, interactionDomain, scaling)

    def __repr__(self):
        return 'ranged '+super(RangedFractionalKernel, self).__repr__()

    def __reduce__(self):
        return RangedFractionalKernel, (self.dim, self.admissibleOrders, self.horizon, self.normalized, self.tempered, self.errorBound, self.M_min, self.M_max, self.xi)


cdef class RangedVariableFractionalKernel(FractionalKernel):
    def __init__(self,
                 INDEX_t dim,
                 function blockIndicator,
                 admissibleOrders,
                 function horizon,
                 BOOL_t normalized=True,
                 REAL_t errorBound=-1.,
                 INDEX_t M_min=1, INDEX_t M_max=20,
                 REAL_t xi=0.):
        self.dim = dim
        self.blockIndicator = blockIndicator
        self.admissibleOrders = admissibleOrders

        assert isinstance(horizon, constant)
        self.horizon = horizon
        if isinstance(horizon, constant) and horizon.value == np.inf:
            self.interaction = fullSpace()
        else:
            self.interaction = ball2_retriangulation()
        self.normalized = normalized

        numBlocks = <INDEX_t>np.around(sqrt(admissibleOrders.numParams))
        assert numBlocks*numBlocks == admissibleOrders.numParams
        self.setOrder(admissibleOrders.getLowerBounds().reshape((numBlocks, numBlocks)))

        self.errorBound = errorBound
        self.M_min = M_min
        self.M_max = M_max
        self.xi = xi

    def setOrder(self, REAL_t[:, ::1] sVals):
        assert self.admissibleOrders.isAdmissible(np.array(sVals, copy=False).flatten())
        sFun = piecewiseConstantFractionalOrder(self.dim, self.blockIndicator, sVals)
        dim = self.dim
        horizon = self.horizon
        interactionDomain = self.interaction
        if self.normalized:
            scaling = variableFractionalLaplacianScaling(sFun.symmetric)
        else:
            scaling = constantTwoPoint(0.5)
        super(RangedVariableFractionalKernel, self).__init__(dim, sFun, horizon, interactionDomain, scaling)

    def getFrozenKernel(self, REAL_t[:, ::1] sVals):
        assert self.admissibleOrders.isAdmissible(np.array(sVals, copy=False).flatten())
        sFun = piecewiseConstantFractionalOrder(self.dim, self.blockIndicator, sVals)
        dim = self.dim
        horizon = self.horizon
        interactionDomain = self.interaction
        if self.normalized:
            scaling = variableFractionalLaplacianScaling(sFun.symmetric)
        else:
            scaling = constantTwoPoint(0.5)
        return FractionalKernel(dim, sFun, horizon, interactionDomain, scaling)

    def __repr__(self):
        return 'ranged '+super(RangedVariableFractionalKernel, self).__repr__()
