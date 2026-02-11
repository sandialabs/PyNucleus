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
from . zeta cimport zetaCy as zeta
from PyNucleus_base.myTypes import REAL
from PyNucleus_base.blas import uninitialized
from PyNucleus_fem.functions cimport constant
from cpython.mem cimport PyMem_Malloc
from libc.string cimport memcpy
from . interactionDomains cimport (fullSpace,
                                   ball2_retriangulation, ball2_barycenter,
                                   ballInf_retriangulation, ballInf_barycenter,
                                   ellipse_retriangulation, ellipse_barycenter)

include "kernel_params.pxi"

cdef REAL_t inf = np.inf


cdef inline REAL_t gamma(REAL_t d) noexcept:
    return cgamma(d)


cdef inline REAL_t polygamma(INDEX_t n, REAL_t d) noexcept:
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


cdef class memoizedGamma(memoizedFun):
    cdef REAL_t eval(self, REAL_t x):
        cdef REAL_t val
        try:
            val = self.memory[x]
            self.hit += 1
            return val
        except KeyError:
            self.miss += 1
            val = gamma(x)
            self.memory[x] = val
            return val


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


cdef class memoizedPolygamma1(memoizedFun):
    cdef REAL_t eval(self, REAL_t x):
        cdef REAL_t val
        try:
            val = self.memory[x]
            self.hit += 1
            return val
        except KeyError:
            self.miss += 1
            val = polygamma(1, x)
            self.memory[x] = val
            return val


cdef:
    memoizedDigamma mem_digamma = memoizedDigamma()
    memoizedPolygamma1 mem_polygamma = memoizedPolygamma1()


######################################################################

cdef inline void fractionalLaplacianScaling(INDEX_t dim, REAL_t s, REAL_t horizon2, BOOL_t normalized, BOOL_t boundary, INDEX_t derivative, REAL_t tempered, REAL_t* values):
    cdef:
        REAL_t C, d2, logd2, loghorizon2, digamma1, digamma2, fac, fac2, log4
        INDEX_t i

    if normalized:
        if horizon2 < inf:
            if dim == 1:
                C = (2.-2*s) * pow(horizon2, s-1.)
            elif dim == 2:
                C = (2.-2*s) * pow(horizon2, s-1.) * 2./pi
            elif dim == 3:
                C = (2.-2*s) * pow(horizon2, s-1.) * 1.5/pi
            else:
                raise NotImplementedError()
        else:
            C = 2.0**(2.0*s) * s * gamma(s+0.5*dim) * pow(pi, -0.5*dim) / gamma(1.0-s)
    else:
        C = 1.

    if boundary:
        C /= (-2.*s)

    if derivative == 0:
        values[0] = C
    elif derivative == 1:
        if normalized:
            if horizon2 < inf:
                if not boundary:
                    fac = -1./(1.-s) + log(horizon2)
                else:
                    fac = -1./(1.-s) - 1./s + log(horizon2)
            else:
                if not boundary:
                    fac = log(4.)+mem_digamma.eval(s+0.5*dim) + mem_digamma.eval(-s)
                else:
                    fac = log(4.)+mem_digamma.eval(s+0.5*dim) + mem_digamma.eval(1.-s)
        else:
            if not boundary:
                fac = 0.
            else:
                fac = -1./s
        values[0] = C*fac
        values[1] = 2.*C
    elif derivative == 2:
        if normalized:
            if horizon2 < inf:
                if not boundary:
                    fac = -2./(1-s) * log(horizon2) + log(horizon2)**2
                    fac2 = -2.*(-1./(1.-s) + log(horizon2))
                else:
                    fac = -2./(1-s) * log(horizon2) + log(horizon2)**2 + 2./s**2 - 2./s * (-1./(1.-s) + log(horizon2))
                    fac2 = -2.*(-1./(1.-s) + log(horizon2) - 1./s)
            else:
                if not boundary:
                    fac = (log(4.)+mem_digamma.eval(s+0.5*dim) + mem_digamma.eval(-s))**2 + (mem_polygamma.eval(s+0.5*dim) - mem_polygamma.eval(-s))
                    fac2 = -2*(log(4.)+mem_digamma(s+0.5*dim) + mem_digamma(-s))
                else:
                    fac = (log(4.)+mem_digamma.eval(s+0.5*dim) + mem_digamma.eval(1.-s))**2 + (mem_polygamma.eval(s+0.5*dim) - mem_polygamma.eval(1.-s))
                    fac2 = -2.*(log(4.)+mem_digamma.eval(s+0.5*dim) + mem_digamma.eval(1.-s))
        else:
            if not boundary:
                fac = 0.
                fac2 = 0.
            else:
                fac = 2./s**2
                fac2 = 2./s
        values[0] = C*fac
        values[1] = -2.0*C*fac2
        values[2] = 4.*C
    else:
        raise NotImplementedError(derivative)


cdef inline str fractionalLaplacianScalingDescription(INDEX_t dim, REAL_t s, REAL_t horizon2, BOOL_t normalized, BOOL_t boundary, INDEX_t derivative, REAL_t tempered, INDEX_t termNo):
    if normalized:
        if horizon2 < inf:
            descr = '\\frac{(2-2s) horizon^{2s-2} d \\Gamma(d/2)}{\\pi^{d/2}}'
        else:
            if (tempered == 0.) or (s == 0.5):
                descr = '\\frac{2^{2s} s \\Gamma(s+d/2)}{\\pi^{d/2} \\Gamma(1-s)}'
            else:
                descr = '\\frac{\\Gamma(d/2)}{2 |\\Gamma(-2s)| \\pi^{d/2}}'
    else:
        descr = ''
    if derivative == 0:
        return descr
    elif derivative == 1:
        if normalized:
            if horizon2 < inf:
                if not boundary:
                    fac = '(-\\frac{-1}{1-s} + \\log(horizon^{2}))'
                else:
                    fac = '(-\\frac{-1}{1-s} - \\frac{1}{s} + \\log(horizon^{2}))'
            else:
                if not boundary:
                    fac = '(\\log 4 + \\psi(s+d/2) + \\psi(-s))'
                else:
                    fac = '(\\log 4 + \\psi(s+d/2) + \\psi(1-s))'
        else:
            if not boundary:
                fac = '0'
            else:
                fac = '(\\frac{1}{2s^2})'
        if termNo == 0:
            return fac + ' * '+descr
        elif termNo == 1:
            return '2'+descr
        else:
            raise NotImplementedError()
    elif derivative == 2:
        return 'MISSING'+descr
    else:
        raise NotImplementedError()


cdef class constantFractionalLaplacianScaling(constantTwoPoint):
    def __init__(self, INDEX_t dim, REAL_t s, REAL_t horizon, BOOL_t normalized, BOOL_t boundary, INDEX_t derivative, REAL_t tempered, INDEX_t termNo):
        cdef:
            REAL_t C, fac, fac2
        self.dim = dim
        self.s = s
        self.horizon = horizon
        self.normalized = normalized
        self.boundary = boundary
        self.derivative = derivative
        self.tempered = tempered
        self.termNo = termNo
        self.values = uninitialized((self.derivative+1), dtype=REAL)
        assert 0 <= self.termNo <= self.derivative

        horizon2 = horizon**2
        fractionalLaplacianScaling(dim, s, horizon2, normalized, boundary, derivative, tempered, &self.values[0])
        if self.derivative > 0:
            value = 1.
        else:
            value = self.values[self.termNo]
        super(constantFractionalLaplacianScaling, self).__init__(value)

    def __reduce__(self):
        return constantFractionalLaplacianScaling, (self.dim, self.s, self.horizon, self.normalized, self.boundary, self.derivative, self.tempered, self.termNo)

    def __eq__(self, constantFractionalLaplacianScaling other):
        return (self.dim == other.dim) and (self.s == other.s) and (self.horizon == other.horizon) and (self.boundary == other.boundary) and (self.derivative == other.derivative) and (self.tempered == other.tempered) and (self.termNo == other.termNo)

    def getLongDescription(self):
        return fractionalLaplacianScalingDescription(self.dim, self.s, self.horizon**2, self.normalized, self.boundary, self.derivative, self.tempered, self.termNo)

    def __repr__(self):
        return '{}'.format(self.values[self.termNo])


cdef class variableFractionalLaplacianScaling(parametrizedTwoPointFunction):
    def __init__(self, BOOL_t symmetric, BOOL_t normalized, BOOL_t boundary, INDEX_t derivative, INDEX_t termNo):
        super(variableFractionalLaplacianScaling, self).__init__(symmetric, 1)
        self.normalized = normalized
        self.boundary = boundary
        self.derivative = derivative
        self.termNo = termNo
        self.values = uninitialized((self.derivative+1), dtype=REAL)
        assert 0 <= self.termNo <= self.derivative

    cdef void setParams(self, void *params):
        parametrizedTwoPointFunction.setParams(self, params)
        self.dim = getINDEX(self.params, fKDIM)

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t s = getREAL(self.params, fS)
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        fractionalLaplacianScaling(self.dim, s, horizon2, self.normalized, self.boundary, self.derivative, 0., &self.values[0])
        if self.derivative > 0:
            value[0] = 1.
        else:
            value[0] = self.values[self.termNo]

    def getScalingWithDifferentHorizon(self):
        cdef:
            variableFractionalLaplacianScalingWithDifferentHorizon scaling
            function horizonFun
            BOOL_t horizonFunNull = isNull(self.params, fHORIZONFUN)
        if not horizonFunNull:
            horizonFun = <function>((<void**>(self.params+fHORIZONFUN))[0])
        else:
            horizonFun = constant(sqrt(getREAL(self.params, fHORIZON2)))
        scaling = variableFractionalLaplacianScalingWithDifferentHorizon(self.symmetric, self.normalized, self.boundary, self.derivative, self.termNo, horizonFun)
        return scaling

    def __repr__(self):
        return 'variableFractionalLaplacianScaling(symmetric={},normalized={},boundary={},derivative={},termNo={})'.format(self.symmetric, self.normalized, self.boundary, self.derivative, self.termNo)

    def __reduce__(self):
        return variableFractionalLaplacianScaling, (self.symmetric, self.normalized, self.boundary, self.derivative, self.termNo)

    def __eq__(self, variableFractionalLaplacianScaling other):
        return (self.symmetric == other.symmetric) and (self.normalized == other.normalized) and (self.boundary == other.boundary) and (self.derivative == other.derivative) and (self.termNo == other.termNo)

    def getLongDescription(self):
        cdef:
            REAL_t s = getREAL(self.params, fS)
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        return fractionalLaplacianScalingDescription(self.dim, s, horizon2, self.normalized, self.boundary, self.derivative, 0., self.termNo)


cdef class variableFractionalLaplacianScalingWithDifferentHorizon(variableFractionalLaplacianScaling):
    def __init__(self, BOOL_t symmetric, BOOL_t normalized, BOOL_t boundary, INDEX_t derivative, INDEX_t termNo, function horizonFun):
        super(variableFractionalLaplacianScalingWithDifferentHorizon, self).__init__(symmetric, normalized, boundary, derivative, termNo)
        self.horizonFun = horizonFun

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
        return variableFractionalLaplacianScalingWithDifferentHorizon, (self.symmetric, self.normalized, self.boundary, self.derivative, self.termNo, self.horizonFun)


######################################################################

cdef inline REAL_t integrableScaling(kernelType kType, interactionDomain interaction, INDEX_t dim, REAL_t horizon, REAL_t gaussian_variance, REAL_t exponentialRate):
    cdef:
        REAL_t value
    if horizon <= 0.:
        value = np.nan
    else:
        if kType == INDICATOR:
            if dim == 1:
                value = 3./horizon**3
            elif dim == 2:
                if isinstance(interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                    value = 8./pi/horizon**4
                elif isinstance(interaction, (ballInf_retriangulation, ballInf_barycenter)):
                    value = 3./4./horizon**4
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        elif kType == PERIDYNAMIC:
            if dim == 1:
                value = 2./horizon**2
            elif dim == 2:
                if isinstance(interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                    value = 6./pi/horizon**3
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        elif kType == GAUSSIAN:
            if dim == 1:
                if horizon < inf:
                    # value = 4.0/sqrt(pi)/(horizon/3.)**3 / 2.
                    value = 4.0/sqrt(pi)/(erf(3.0)-6.0*exp(-9.0)/sqrt(pi))/(horizon/3.0)**3
                else:
                    value = 1.0/sqrt(2.0*pi*gaussian_variance)
            elif dim == 2:
                if isinstance(interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                    # value = 4.0/pi/(horizon/3.0)**4
                    value = 4.0/pi/(1.0-10.0*exp(-9.0))/(horizon/3.0)**4
                elif isinstance(interaction, fullSpace):
                    value = 1.0/(2.0*pi*gaussian_variance)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        elif kType == EXPONENTIAL:
            if dim == 1:
                if horizon < inf:
                    value = exponentialRate**3/(2.0-exp(-exponentialRate*horizon)*(2.0 + 2.0*exponentialRate*horizon + (exponentialRate*horizon)**2))
                else:
                    value = exponentialRate**3/2.0
            else:
                raise NotImplementedError()
        elif kType == POLYNOMIAL:
            value = 1.0
        else:
            raise NotImplementedError()
    return value


cdef inline str integrableScalingDescription(kernelType kType, interactionDomain interaction, INDEX_t dim, REAL_t horizon, REAL_t gaussian_variance, REAL_t exponentialRate):
    descr = ''
    if kType == INDICATOR:
        if dim == 1:
            descr = '\\frac{3}{\\delta^3}'
        elif dim == 2:
            if isinstance(interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                descr = '\\frac{8}{\\pi\\delta^4}'
            elif isinstance(interaction, (ballInf_retriangulation, ballInf_barycenter)):
                descr = '\\frac{3}{4\\delta^4}'
    elif kType == PERIDYNAMIC:
        if dim == 1:
            descr = '\\frac{2}{\\delta^2}'
        if dim == 2:
            if isinstance(interaction, (ball2_retriangulation, ball2_barycenter, ellipse_retriangulation, ellipse_barycenter)):
                descr = '\\frac{6}{\\pi\\delta^3}'
    elif kType == GAUSSIAN:
        if horizon < inf:
            if dim == 1:
                descr = '\\frac{4}{\\sqrt(\\pi) (\\operatorname{erf}(3)-6\\exp(-9)/\\sqrt(\\pi)) (\\delta/3)^3}'
            elif dim == 2:
                descr = '\\frac{4.0}{\\pi (1-10\\exp(-9)) (\\delta/3)^4}'
        else:
            descr = '\\frac{1}{(2\\pi\\sigma)^{d/2}}'
    elif kType == EXPONENTIAL:
        if horizon < inf:
            descr = '\\frac{a^3}{2-exp(-a\\delta) (2+2a\\delta + (a\\delta)^2)}'
        else:
            descr = '\\frac{a^3}{2}'
    elif kType == POLYNOMIAL:
        descr = ''
    return descr



cdef class constantIntegrableScaling(constantTwoPoint):
    def __init__(self, kernelType kType, interactionDomain interaction, INDEX_t dim, REAL_t horizon, REAL_t gaussian_variance=1.0, REAL_t exponentialRate=1.0):
        self.kType = kType
        self.dim = dim
        self.interaction = interaction
        self.horizon = horizon
        self.gaussian_variance = gaussian_variance
        self.exponentialRate = exponentialRate
        value = integrableScaling(kType, interaction, dim, horizon, gaussian_variance, exponentialRate)
        super(constantIntegrableScaling, self).__init__(value)

    def __reduce__(self):
        return constantIntegrableScaling, (self.kType, self.interaction, self.dim, self.horizon, self.gaussian_variance, self.exponentialRate)

    def __repr__(self):
        return '{}({} -> {})'.format(self.__class__.__name__, self.horizon, self.value)

    def getLongDescription(self):
        return integrableScalingDescription(self.kType, self.interaction, self.dim, self.horizon, self.gaussian_variance, self.exponentialRate)


cdef class variableIntegrableScaling(parametrizedTwoPointFunction):
    def __init__(self, kernelType kType, interactionDomain interaction):
        super(variableIntegrableScaling, self).__init__(False, 1)
        self.kType = kType
        self.interaction = interaction

    cdef void setParams(self, void *params):
        parametrizedTwoPointFunction.setParams(self, params)
        self.dim = getINDEX(self.params, fKDIM)

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t gaussian_variance = -1.0
            REAL_t exponentialRate = -1.0
        value[0] = integrableScaling(self.kType, self.interaction, self.dim, sqrt(horizon2), gaussian_variance, exponentialRate)

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

    def getLongDescription(self):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t gaussian_variance = -1.0
            REAL_t exponentialRate = -1.0
        return integrableScalingDescription(self.kType, self.interaction, self.dim, sqrt(horizon2), gaussian_variance, exponentialRate)


cdef class variableIntegrableScalingWithDifferentHorizon(variableIntegrableScaling):
    def __init__(self, kernelType kType, interactionDomain interaction, function horizonFun):
        super(variableIntegrableScalingWithDifferentHorizon, self).__init__(kType, interaction)
        self.horizonFun = horizonFun

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
