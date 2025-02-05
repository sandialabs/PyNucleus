###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

"""Defines normalizations for different types of kernels."""
import numpy as np
cimport numpy as np
from libc.math cimport (sqrt,
                        log,
                        fabs as abs, M_PI as pi, pow,
                        exp, erf)
from scipy.special.cython_special cimport psi as digamma
from scipy.special.cython_special cimport gamma as cgamma
from scipy.special import zeta

from PyNucleus_fem.functions cimport constant
from cpython.mem cimport PyMem_Malloc
from libc.string cimport memcpy
from . interactionDomains cimport (fullSpace,
                                   ball2_retriangulation, ball2_barycenter,
                                   ballInf_retriangulation, ballInf_barycenter,
                                   ellipse_retriangulation, ellipse_barycenter)

include "kernel_params.pxi"

cdef REAL_t inf = np.inf


cdef inline REAL_t gamma(REAL_t d):
    return cgamma(d)


cdef inline REAL_t polygamma(INDEX_t n, REAL_t d):
    return (-1.0)**(n+1) * cgamma(n+1.0) * zeta(n+1, d)


cdef class memoizedFun:
    def __init__(self):
        self.memory = dict()
        self.hit = 0
        self.miss = 0

    cdef REAL_t eval(self, REAL_t x):
        raise NotImplementedError()

    def __call__(self, REAL_t x):
        return self.eval(x)

    def stats(self):
        print(len(self.memory), self.hit, self.miss)


cdef class memoizedDigamma(memoizedFun):
    cdef REAL_t eval(self, REAL_t x):
        cdef REAL_t val
        try:
            val = self.memory[x]
            self.hit += 1
            return val
        except KeyError:
            self.miss += 1
            val = digamma(x)
            self.memory[x] = val
            return val


cdef class constantFractionalLaplacianScaling(constantTwoPoint):
    def __init__(self, INDEX_t dim, REAL_t s, REAL_t horizon, REAL_t tempered):
        self.dim = dim
        if 1. < s and s < 2.:
            s = s-1.
        self.s = s
        self.horizon = horizon
        self.tempered = tempered
        if (self.horizon <= 0.) or (self.s <= 0.) or (self.s >= 1.):
            value = np.nan
        else:
            if horizon < inf:
                value = (2.-2*s) * pow(horizon, 2*s-2.) * dim * gamma(0.5*dim)/pow(pi, 0.5*dim) * 0.5
            else:
                if (tempered == 0.) or (s == 0.5):
                    value = 2.0**(2.0*s) * s * gamma(s+0.5*dim)/pow(pi, 0.5*dim)/gamma(1.0-s) * 0.5
                else:
                    value = gamma(0.5*dim) / abs(gamma(-2*s))/pow(pi, 0.5*dim) * 0.5 * 0.5
        super(constantFractionalLaplacianScaling, self).__init__(value)

    def __reduce__(self):
        return constantFractionalLaplacianScaling, (self.dim, self.s, self.horizon, self.tempered)

    def __repr__(self):
        return '{}({},{} -> {})'.format(self.__class__.__name__, self.s, self.horizon, self.value)

    def getLongDescription(self):
        if self.horizon < inf:
            descr = '\\frac{(2-2s) horizon^{2s-2} d \\Gamma(d/2)}{\\pi^{d/2}}'
        else:
            if (self.tempered == 0.) or (self.s == 0.5):
                descr = '\\frac{2^{2s} s \\Gamma(s+d/2)}{\\pi^{d/2} \\Gamma(1-s)}'
            else:
                descr = '\\frac{\\Gamma(d/2)}{2 |\\Gamma(-2s)| \\pi^{d/2}}'
        return descr


cdef class constantFractionalLaplacianScalingDerivative(twoPointFunction):
    def __init__(self, INDEX_t dim, REAL_t s, REAL_t horizon, BOOL_t normalized, BOOL_t boundary, INDEX_t derivative, REAL_t tempered):
        super(constantFractionalLaplacianScalingDerivative, self).__init__(True, 1)

        self.dim = dim
        self.s = s
        self.horizon = horizon
        self.normalized = normalized
        self.boundary = boundary
        self.derivative = derivative
        self.tempered = tempered

        horizon2 = horizon**2
        self.horizon2 = horizon2

        if self.normalized:
            if horizon2 < inf:
                self.C = (2.-2*s) * pow(horizon2, s-1.) * dim*gamma(0.5*dim)/pow(pi, 0.5*dim) * 0.5
            else:
                if (tempered == 0.) or (s == 0.5):
                    self.C = 2.0**(2.0*s) * s * gamma(s+0.5*dim) * pow(pi, -0.5*dim) / gamma(1.0-s) * 0.5
                else:
                    self.C = gamma(0.5*dim) / abs(gamma(-2*s))/pow(pi, 0.5*dim) * 0.5 * 0.5
        else:
            self.C = 0.5

        if self.derivative == 1:
            if self.normalized:
                if horizon2 < inf:
                    if not self.boundary:
                        self.fac = -1./(1.-s) + log(horizon2)
                    else:
                        self.fac = -1./(1.-s) - 1./s + log(horizon2)
                else:
                    if not self.boundary:
                        self.fac = log(4.)+digamma(s+0.5*self.dim) + digamma(-s)
                    else:
                        self.fac = log(4.)+digamma(s+0.5*self.dim) + digamma(1.-s)
            else:
                if not self.boundary:
                    self.fac = 0.
                else:
                    self.fac = -1./s
        elif self.derivative == 2:
            if self.normalized:
                if horizon2 < inf:
                    if not self.boundary:
                        self.fac = -2./(1-s) * log(horizon2) + log(horizon2)**2
                        self.fac2 = -2.*(-1./(1.-s) + log(horizon2))
                    else:
                        self.fac = -2./(1-s) * log(horizon2) + log(horizon2)**2 + 2./s**2 - 2./s * (-1./(1.-s) + log(horizon2))
                        self.fac2 = -2.*(-1./(1.-s) + log(horizon2) - 1./s)
                else:
                    if not self.boundary:
                        self.fac = (log(4.)+digamma(s+0.5*self.dim) + digamma(-s))**2 + (polygamma(1, s+0.5*dim) - polygamma(1, -s))
                        self.fac2 = -2*(log(4.)+digamma(s+0.5*self.dim) + digamma(-s))
                    else:
                        self.fac = (log(4.)+digamma(s+0.5*self.dim) + digamma(1.-s))**2 + (polygamma(1, s+0.5*dim) - polygamma(1, 1.-s))
                        self.fac2 = -2.*(log(4.)+digamma(s+0.5*self.dim) + digamma(1.-s))
            else:
                if not self.boundary:
                    self.fac = 0.
                    self.fac2 = 0.
                else:
                    self.fac = 2./s**2
                    self.fac2 = 2./s
        else:
            raise NotImplementedError(self.derivative)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            REAL_t d2, logd2
            INDEX_t i

        if self.derivative == 0:
            value[0] = self.C
        elif self.derivative == 1:
            d2 = 0.
            for i in range(self.dim):
                d2 += (x[i]-y[i])*(x[i]-y[i])
            value[0] = self.C*(-log(d2) + self.fac)
        elif self.derivative == 2:
            d2 = 0.
            for i in range(self.dim):
                d2 += (x[i]-y[i])*(x[i]-y[i])
            logd2 = log(d2)
            value[0] = self.C*(logd2**2 + self.fac + self.fac2*logd2)
        else:
            raise NotImplementedError()

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t d2, logd2
            INDEX_t i

        if self.derivative == 0:
            value[0] = self.C
        elif self.derivative == 1:
            d2 = 0.
            for i in range(self.dim):
                d2 += (x[i]-y[i])*(x[i]-y[i])
            value[0] = self.C*(-log(d2) + self.fac)
        elif self.derivative == 2:
            d2 = 0.
            for i in range(self.dim):
                d2 += (x[i]-y[i])*(x[i]-y[i])
            logd2 = log(d2)
            value[0] = self.C*(logd2**2 + self.fac + self.fac2*logd2)
        else:
            raise NotImplementedError()

    def __reduce__(self):
        return constantFractionalLaplacianScalingDerivative, (self.dim, self.s, self.horizon, self.normalized, self.boundary, self.derivative, self.tempered)

    def __repr__(self):
        return "{}({},{} -> {})".format(self.__class__.__name__, self.s, self.horizon, self.fac)


cdef class constantIntegrableScaling(constantTwoPoint):
    def __init__(self, kernelType kType, interactionDomain interaction, INDEX_t dim, REAL_t horizon, REAL_t gaussian_variance=1.0, REAL_t exponentialRate=1.0):
        self.kType = kType
        self.dim = dim
        self.interaction = interaction
        self.horizon = horizon
        self.gaussian_variance = gaussian_variance
        self.exponentialRate = exponentialRate
        if self.horizon <= 0.:
            value = np.nan
        else:
            if kType == INDICATOR:
                if dim == 1:
                    value = 3./horizon**3 / 2.
                elif dim == 2:
                    if isinstance(self.interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                        value = 8./pi/horizon**4 / 2.
                    elif isinstance(self.interaction, (ballInf_retriangulation, ballInf_barycenter)):
                        value = 3./4./horizon**4 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif kType == PERIDYNAMIC:
                if dim == 1:
                    value = 2./horizon**2 / 2.
                elif dim == 2:
                    if isinstance(self.interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                        value = 6./pi/horizon**3 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif kType == GAUSSIAN:
                if dim == 1:
                    if horizon < inf:
                        # value = 4.0/sqrt(pi)/(horizon/3.)**3 / 2.
                        value = 4.0/sqrt(pi)/(erf(3.0)-6.0*exp(-9.0)/sqrt(pi))/(horizon/3.0)**3 / 2.
                    else:
                        value = 1.0/sqrt(2.0*pi*gaussian_variance) / 2.
                elif dim == 2:
                    if isinstance(self.interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                        # value = 4.0/pi/(horizon/3.0)**4 / 2.
                        value = 4.0/pi/(1.0-10.0*exp(-9.0))/(horizon/3.0)**4 / 2.
                    elif isinstance(self.interaction, fullSpace):
                        value = 1.0/(2.0*pi*gaussian_variance) / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif kType == EXPONENTIAL:
                if dim == 1:
                    if horizon < inf:
                        value = exponentialRate**3/(2.0-exp(-exponentialRate*horizon)*(2.0 + 2.0*exponentialRate*horizon + (exponentialRate*horizon)**2)) / 2.
                    else:
                        value = exponentialRate**3/2.0 / 2.
                else:
                    raise NotImplementedError()
            elif kType == POLYNOMIAL:
                value = 0.5
            else:
                raise NotImplementedError()
        super(constantIntegrableScaling, self).__init__(value)

    def __reduce__(self):
        return constantIntegrableScaling, (self.kType, self.interaction, self.dim, self.horizon, self.gaussian_variance, self.exponentialRate)

    def __repr__(self):
        return '{}({} -> {})'.format(self.__class__.__name__, self.horizon, self.value)

    def getLongDescription(self):
        descr = ''
        if self.kType == INDICATOR:
            if self.dim == 1:
                descr = '\\frac{3}{\\delta^3}'
            elif self.dim == 2:
                if isinstance(self.interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                    descr = '\\frac{8}{\\pi\\delta^4}'
                elif isinstance(self.interaction, (ballInf_retriangulation, ballInf_barycenter)):
                    descr = '\\frac{3}{4\\delta^4}'
        elif self.kType == PERIDYNAMIC:
            if self.dim == 1:
                descr = '\\frac{2}{\\delta^2}'
            if self.dim == 2:
                if isinstance(self.interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                    descr = '\\frac{6}{\\pi\\delta^3}'
        elif self.kType == GAUSSIAN:
            if self.horizon < inf:
                if self.dim == 1:
                    descr = '\\frac{4}{\\sqrt(\\pi) (\\operatorname{erf}(3)-6\\exp(-9)/\\sqrt(\\pi)) (\\delta/3)^3}'
                elif self.dim == 2:
                    descr = '\\frac{4.0}{\\pi (1-10\\exp(-9)) (\\delta/3)^4}'
            else:
                descr = '\\frac{1}{(2\\pi\\sigma)^{d/2}}'
        elif self.kType == EXPONENTIAL:
            if self.horizon < inf:
                descr = '\\frac{a^3}{2-exp(-a\\delta) (2+2a\\delta + (a\\delta)^2)}'
            else:
                descr = '\\frac{a^3}{2}'
        elif self.kType == POLYNOMIAL:
            descr = ''
        return descr


cdef class variableFractionalLaplacianScaling(parametrizedTwoPointFunction):
    def __init__(self, BOOL_t symmetric, BOOL_t normalized=True, BOOL_t boundary=False, INDEX_t derivative=0):
        super(variableFractionalLaplacianScaling, self).__init__(symmetric, 1)
        self.normalized = normalized
        self.boundary = boundary
        self.derivative = derivative
        self.digamma = memoizedDigamma()

    cdef void setParams(self, void *params):
        parametrizedTwoPointFunction.setParams(self, params)
        self.dim = getINDEX(self.params, fKDIM)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            REAL_t s = getREAL(self.params, fS)
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t C, d2, fac, fac2
            INDEX_t i

        if self.normalized:
            if horizon2 < inf:
                if self.dim == 1:
                    C = (2.-2*s) * pow(horizon2, s-1.) * 0.5
                elif self.dim == 2:
                    C = (2.-2*s) * pow(horizon2, s-1.) * 2./pi * 0.5
                elif self.dim == 3:
                    C = (2.-2*s) * pow(horizon2, s-1.) * 1.5/pi * 0.5
                else:
                    raise NotImplementedError()
            else:
                C = 2.0**(2.0*s) * s * gamma(s+0.5*self.dim) * pow(pi, -0.5*self.dim) / gamma(1.0-s) * 0.5
        else:
            C = 0.5

        if self.derivative == 0:
            value[0] = C
        elif self.derivative == 1:
            d2 = 0.
            for i in range(self.dim):
                d2 += (x[i]-y[i])*(x[i]-y[i])
            if self.normalized:
                if horizon2 < inf:
                    if not self.boundary:
                        value[0] = C*(-log(d2/horizon2) - 1./(1.-s))
                    else:
                        value[0] = C*(-log(d2/horizon2) - 1./(1.-s) - 1./s)
                else:
                    if not self.boundary:
                        value[0] = C*(-log(0.25*d2) + self.digamma.eval(s+0.5*self.dim) + self.digamma.eval(-s))
                    else:
                        value[0] = C*(-log(0.25*d2) + self.digamma.eval(s+0.5*self.dim) + self.digamma.eval(1.-s))
            else:
                if not self.boundary:
                    value[0] = C*(-log(d2))
                else:
                    value[0] = C*(-log(d2)-1./s)
        elif self.derivative == 2:
            d2 = 0.
            for i in range(self.dim):
                d2 += (x[i]-y[i])*(x[i]-y[i])
            logd2 = log(d2)
            if self.normalized:
                if horizon2 < inf:
                    loghorizon2 = log(horizon2)
                    if not self.boundary:
                        fac = -2./(1-s) * loghorizon2 + loghorizon2**2
                        fac2 = -2.*(-1./(1.-s) + log(horizon2))
                    else:
                        fac = -2./(1-s) * loghorizon2 + loghorizon2**2 + 2./s**2 - 2./s * (-1./(1.-s) + loghorizon2)
                        fac2 = -2.*(-1./(1.-s) + loghorizon2 - 1./s)
                else:
                    digamma1 = self.digamma(s+0.5*self.dim)
                    if not self.boundary:
                        digamma2 = self.digamma(-s)
                        fac = (log(4.)+digamma1 + digamma2)**2 + (polygamma(1, s+0.5*self.dim) - polygamma(1, -s))
                        fac2 = -2*(log(4.)+digamma1 + digamma2)
                    else:
                        digamma2 = self.digamma(1.-s)
                        fac = (log(4.)+digamma1 + digamma2)**2 + (polygamma(1, s+0.5*self.dim) - polygamma(1, 1.-s))
                        fac2 = -2.*(log(4.)+digamma1 + digamma2)
            else:
                if not self.boundary:
                    fac = 0.
                    fac2 = 0.
                else:
                    fac = 2./s**2
                    fac2 = 2./s

            value[0] = C*(logd2**2 + fac + fac2*logd2)
        else:
            raise NotImplementedError()

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t s = getREAL(self.params, fS)
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t C, d2, logd2, loghorizon2, digamma1, digamma2
            INDEX_t i

        if self.normalized:
            if horizon2 < inf:
                if self.dim == 1:
                    C = (2.-2*s) * pow(horizon2, s-1.) * 0.5
                elif self.dim == 2:
                    C = (2.-2*s) * pow(horizon2, s-1.) * 2./pi * 0.5
                elif self.dim == 3:
                    C = (2.-2*s) * pow(horizon2, s-1.) * 1.5/pi * 0.5
                else:
                    raise NotImplementedError()
            else:
                C = 2.0**(2.0*s) * s * gamma(s+0.5*self.dim) * pow(pi, -0.5*self.dim) / gamma(1.0-s) * 0.5
        else:
            C = 0.5

        if self.derivative == 0:
            value[0] = C
        elif self.derivative == 1:
            d2 = 0.
            for i in range(self.dim):
                d2 += (x[i]-y[i])*(x[i]-y[i])
            if self.normalized:
                if horizon2 < inf:
                    if not self.boundary:
                        value[0] = C*(-log(d2/horizon2) - 1./(1.-s))
                    else:
                        value[0] = C*(-log(d2/horizon2) - 1./(1.-s) - 1./s)
                else:
                    if not self.boundary:
                        value[0] = C*(-log(0.25*d2) + self.digamma.eval(s+0.5*self.dim) + self.digamma.eval(-s))
                    else:
                        value[0] = C*(-log(0.25*d2) + self.digamma.eval(s+0.5*self.dim) + self.digamma.eval(1.-s))
            else:
                if not self.boundary:
                    value[0] = C*(-log(d2))
                else:
                    value[0] = C*(-log(d2)-1./s)
        elif self.derivative == 2:
            d2 = 0.
            for i in range(self.dim):
                d2 += (x[i]-y[i])*(x[i]-y[i])
            logd2 = log(d2)
            if self.normalized:
                if horizon2 < inf:
                    loghorizon2 = log(horizon2)
                    if not self.boundary:
                        fac = -2./(1-s) * loghorizon2 + loghorizon2**2
                        fac2 = -2.*(-1./(1.-s) + log(horizon2))
                    else:
                        fac = -2./(1-s) * loghorizon2 + loghorizon2**2 + 2./s**2 - 2./s * (-1./(1.-s) + loghorizon2)
                        fac2 = -2.*(-1./(1.-s) + loghorizon2 - 1./s)
                else:
                    digamma1 = self.digamma(s+0.5*self.dim)
                    if not self.boundary:
                        digamma2 = self.digamma(-s)
                        fac = (log(4.)+digamma1 + digamma2)**2 + (polygamma(1, s+0.5*self.dim) - polygamma(1, -s))
                        fac2 = -2*(log(4.)+digamma1 + digamma2)
                    else:
                        digamma2 = self.digamma(1.-s)
                        fac = (log(4.)+digamma1 + digamma2)**2 + (polygamma(1, s+0.5*self.dim) - polygamma(1, 1.-s))
                        fac2 = -2.*(log(4.)+digamma1 + digamma2)
            else:
                if not self.boundary:
                    fac = 0.
                    fac2 = 0.
                else:
                    fac = 2./s**2
                    fac2 = 2./s

            value[0] = C*(logd2**2 + fac + fac2*logd2)
        else:
            raise NotImplementedError()

    def getScalingWithDifferentHorizon(self):
        cdef:
            variableFractionalLaplacianScalingWithDifferentHorizon scaling
            function horizonFun
            BOOL_t horizonFunNull = isNull(self.params, fHORIZONFUN)
        if not horizonFunNull:
            horizonFun = <function>((<void**>(self.params+fHORIZONFUN))[0])
        else:
            horizonFun = constant(sqrt(getREAL(self.params, fHORIZON2)))
        scaling = variableFractionalLaplacianScalingWithDifferentHorizon(self.symmetric, self.normalized, self.boundary, self.derivative, horizonFun)
        return scaling

    def __repr__(self):
        return 'variableFractionalLaplacianScaling'

    def __reduce__(self):
        return variableFractionalLaplacianScaling, (self.symmetric, self.normalized, self.boundary, self.derivative)

    def getLongDescription(self):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        if self.normalized:
            if horizon2 < inf:
                descr_C = '\\frac{(2-2s) horizon^{2s-2} d \\Gamma(d/2)}{\\pi^{d/2}}'
            else:
                descr_C = '\\frac{2^{2s} s \\Gamma(s+d/2)}{\\pi^{d/2} \\Gamma(1-s)}'
        else:
            descr_C = ''
        if self.derivative == 0:
            return descr_C
        else:
            descr = "MISSING"
        return descr


cdef class variableIntegrableScaling(parametrizedTwoPointFunction):
    def __init__(self, kernelType kType, interactionDomain interaction):
        super(variableIntegrableScaling, self).__init__(False, 1)
        self.kType = kType
        self.interaction = interaction

    cdef void setParams(self, void *params):
        parametrizedTwoPointFunction.setParams(self, params)
        self.dim = getINDEX(self.params, fKDIM)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)

        if horizon2 <= 0.:
            value[0] = np.nan
        else:
            if self.kType == INDICATOR:
                if self.dim == 1:
                    value[0] = 3./horizon2**1.5 / 2.
                elif self.dim == 2:
                    if isinstance(self.interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                        value[0] = 8./pi/horizon2**2 / 2.
                    elif isinstance(self.interaction, (ballInf_retriangulation, ballInf_barycenter)):
                        value[0] = 3./4./horizon2**2 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif self.kType == PERIDYNAMIC:
                if self.dim == 1:
                    value[0] = 2./horizon2 / 2.
                elif self.dim == 2:
                    if isinstance(self.interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                        value[0] = 6./pi/horizon2**1.5 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif self.kType == GAUSSIAN:
                if self.dim == 1:
                    # value[0] = 4.0/sqrt(pi)/(horizon/3.)**3 / 2.
                    value[0] = 4.0/sqrt(pi)/(erf(3.0)-6.0*exp(-9.0)/sqrt(pi))/(sqrt(horizon2)/3.0)**3 / 2.
                elif self.dim == 2:
                    if isinstance(self.interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                        # value[0] = 4.0/pi/(horizon/3.0)**4 / 2.
                        value[0] = 4.0/pi/(1.0-10.0*exp(-9.0))/(sqrt(horizon2)/3.0)**4 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)

        if horizon2 <= 0.:
            value[0] = np.nan
        else:
            if self.kType == INDICATOR:
                if self.dim == 1:
                    value[0] = 3./horizon2**1.5 / 2.
                elif self.dim == 2:
                    if isinstance(self.interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                        value[0] = 8./pi/horizon2**2 / 2.
                    elif isinstance(self.interaction, (ballInf_retriangulation, ballInf_barycenter)):
                        value[0] = 3./4./horizon2**2 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif self.kType == PERIDYNAMIC:
                if self.dim == 1:
                    value[0] = 2./horizon2 / 2.
                elif self.dim == 2:
                    if isinstance(self.interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                        value[0] = 6./pi/horizon2**1.5 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif self.kType == GAUSSIAN:
                if self.dim == 1:
                    # value[0] = 4.0/sqrt(pi)/(horizon/3.)**3 / 2.
                    value[0] = 4.0/sqrt(pi)/(erf(3.0)-6.0*exp(-9.0)/sqrt(pi))/(sqrt(horizon2)/3.0)**3 / 2.
                elif self.dim == 2:
                    if isinstance(self.interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                        # value[0] = 4.0/pi/(horizon/3.0)**4 / 2.
                        value[0] = 4.0/pi/(1.0-10.0*exp(-9.0))/(sqrt(horizon2)/3.0)**4 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    def getScalingWithDifferentHorizon(self):
        cdef:
            variableIntegrableScalingWithDifferentHorizon scaling
            function horizonFun
            BOOL_t horizonFunNull = isNull(self.params, fHORIZONFUN)
        if not horizonFunNull:
            horizonFun = <function>((<void**>(self.params+fHORIZONFUN))[0])
        else:
            horizonFun = constant(sqrt(getREAL(self.params, fHORIZON2)))
        scaling = variableIntegrableScalingWithDifferentHorizon(self.kType, self.interaction, horizonFun)
        return scaling

    def __repr__(self):
        return 'variableIntegrableScaling'

    def __reduce__(self):
        return variableIntegrableScaling, (self.kType, self.interaction)


######################################################################


cdef class variableFractionalLaplacianScalingWithDifferentHorizon(variableFractionalLaplacianScaling):
    def __init__(self, BOOL_t symmetric, BOOL_t normalized, BOOL_t boundary, INDEX_t derivative, function horizonFun):
        super(variableFractionalLaplacianScalingWithDifferentHorizon, self).__init__(symmetric, normalized, boundary, derivative)
        self.horizonFun = horizonFun

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            void* params
            void* paramsModified = PyMem_Malloc(NUM_KERNEL_PARAMS*OFFSET)
            REAL_t horizon, scalingValue
        horizon = self.horizonFun.eval(x)
        params = self.getParams()
        memcpy(paramsModified, params, NUM_KERNEL_PARAMS*OFFSET)
        setREAL(paramsModified, fHORIZON2, horizon**2)
        self.setParams(paramsModified)
        variableFractionalLaplacianScaling.evalPtr(self, x.shape[0], &x[0], &y[0], &scalingValue)
        self.setParams(params)
        value[0] = scalingValue

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            void* params
            void* paramsModified = PyMem_Malloc(NUM_KERNEL_PARAMS*OFFSET)
            REAL_t horizon, scalingValue
            REAL_t[::1] xA = <REAL_t[:dim]> x
        horizon = self.horizonFun.eval(xA)
        params = self.getParams()
        memcpy(paramsModified, params, NUM_KERNEL_PARAMS*OFFSET)
        setREAL(paramsModified, fHORIZON2, horizon**2)
        self.setParams(paramsModified)
        variableFractionalLaplacianScaling.evalPtr(self, dim, x, y, &scalingValue)
        self.setParams(params)
        value[0] = scalingValue

    def __reduce__(self):
        return variableFractionalLaplacianScalingWithDifferentHorizon, (self.symmetric, self.normalized, self.boundary, self.derivative, self.horizonFun)


cdef class variableIntegrableScalingWithDifferentHorizon(variableIntegrableScaling):
    def __init__(self, kernelType kType, interactionDomain interaction, function horizonFun):
        super(variableIntegrableScalingWithDifferentHorizon, self).__init__(kType, interaction)
        self.horizonFun = horizonFun

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            void* params
            void* paramsModified = PyMem_Malloc(NUM_KERNEL_PARAMS*OFFSET)
            REAL_t horizon, scalingValue
        horizon = self.horizonFun.eval(x)
        params = self.getParams()
        memcpy(paramsModified, params, NUM_KERNEL_PARAMS*OFFSET)
        setREAL(paramsModified, fHORIZON2, horizon**2)
        self.setParams(paramsModified)
        variableIntegrableScaling.evalPtr(self, x.shape[0], &x[0], &y[0], &scalingValue)
        self.setParams(params)
        value[0] = scalingValue

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            void* params
            void* paramsModified = PyMem_Malloc(NUM_KERNEL_PARAMS*OFFSET)
            REAL_t horizon, scalingValue
            REAL_t[::1] xA = <REAL_t[:dim]> x
        horizon = self.horizonFun.eval(xA)
        params = self.getParams()
        memcpy(paramsModified, params, NUM_KERNEL_PARAMS*OFFSET)
        setREAL(paramsModified, fHORIZON2, horizon**2)
        self.setParams(paramsModified)
        variableIntegrableScaling.evalPtr(self, dim, x, y, &scalingValue)
        self.setParams(params)
        value[0] = scalingValue

    def __reduce__(self):
        return variableIntegrableScalingWithDifferentHorizon, (self.kType, self.interaction, self.horizonFun)
