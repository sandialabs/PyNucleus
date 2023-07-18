###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

"""Defines normalizations for different types of kernels."""
import numpy as np
cimport numpy as np
from libc.math cimport (sin, cos, sinh, cosh, tanh, sqrt, atan, atan2,
                        log, ceil,
                        fabs as abs, M_PI as pi, pow,
                        exp, erf)
from scipy.special.cython_special cimport psi as digamma
from scipy.special.cython_special cimport gamma as cgamma

from PyNucleus_fem.functions cimport constant
from libc.stdlib cimport malloc
from libc.string cimport memcpy
from . interactionDomains cimport ball1, ball2, ballInf

include "kernel_params.pxi"

cdef REAL_t inf = np.inf


cdef inline REAL_t gamma(REAL_t d):
    return cgamma(d)


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
                value = (2.-2*s) * pow(horizon, 2*s-2.) * gamma(0.5*dim)/pow(pi, 0.5*dim) * 0.5
                if dim > 1:
                    value *= 2.
            else:
                if (tempered == 0.) or (s == 0.5):
                    value = 2.0**(2.0*s) * s * gamma(s+0.5*dim)/pow(pi, 0.5*dim)/gamma(1.0-s) * 0.5
                else:
                    value = gamma(0.5*dim) / abs(gamma(-2*s))/pow(pi, 0.5*dim) * 0.5 * 0.5
        super(constantFractionalLaplacianScaling, self).__init__(value)

    def __getstate__(self):
        return (self.dim, self.s, self.horizon, self.tempered)

    def __setstate__(self, state):
        constantFractionalLaplacianScaling.__init__(self, state[0], state[1], state[2], state[3])

    def __repr__(self):
        return '{}({},{} -> {})'.format(self.__class__.__name__, self.s, self.horizon, self.value)


cdef class constantFractionalLaplacianScalingDerivative(twoPointFunction):
    def __init__(self, INDEX_t dim, REAL_t s, REAL_t horizon, BOOL_t normalized, BOOL_t boundary, INDEX_t derivative, REAL_t tempered):
        super(constantFractionalLaplacianScalingDerivative, self).__init__(True)

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
                self.C = (2.-2*s) * pow(horizon2, s-1.) * gamma(0.5*dim)/pow(pi, 0.5*dim) * 0.5
                if dim > 1:
                    self.C *= 2.
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
                        self.fac = -1./(1.-s)
                    else:
                        self.fac = -1./(1.-s) - 1./s
                else:
                    if not self.boundary:
                        self.fac = digamma(s+0.5*self.dim) + digamma(-s)
                    else:
                        self.fac = digamma(s+0.5*self.dim) + digamma(1.-s)
            else:
                if not self.boundary:
                    self.fac = 0.
                else:
                    self.fac = -1./s
        else:
            raise NotImplementedError(self.derivative)

    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t d2
            INDEX_t i

        if self.derivative == 0:
            return self.C
        elif self.derivative == 1:
            d2 = 0.
            for i in range(self.dim):
                d2 += (x[i]-y[i])*(x[i]-y[i])
            if self.normalized:
                if self.horizon2 < inf:
                    return self.C*(-log(d2/self.horizon2) + self.fac)
                else:
                    return self.C*(-log(0.25*d2) + self.fac)
            else:
                return self.C*(-log(d2) + self.fac)
        else:
            raise NotImplementedError()

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t d2
            INDEX_t i

        if self.derivative == 0:
            return self.C
        elif self.derivative == 1:
            d2 = 0.
            for i in range(self.dim):
                d2 += (x[i]-y[i])*(x[i]-y[i])
            if self.normalized:
                if self.horizon2 < inf:
                    return self.C*(-log(d2/self.horizon2) + self.fac)
                else:
                    return self.C*(-log(0.25*d2) + self.fac)
            else:
                return self.C*(-log(d2) + self.fac)
        else:
            raise NotImplementedError()

    def __getstate__(self):
        return (self.dim, self.s, self.horizon, self.normalized, self.boundary, self.derivative, self.tempered)

    def __setstate__(self, state):
        constantFractionalLaplacianScalingDerivative.__init__(self, state[0], state[1], state[2], state[3], state[4], state[5], state[6])


cdef class constantIntegrableScaling(constantTwoPoint):
    def __init__(self, kernelType kType, interactionDomain interaction, INDEX_t dim, REAL_t horizon):
        self.kType = kType
        self.dim = dim
        self.interaction = interaction
        self.horizon = horizon
        if self.horizon <= 0.:
            value = np.nan
        else:
            if kType == INDICATOR:
                if dim == 1:
                    value = 3./horizon**3 / 2.
                elif dim == 2:
                    if isinstance(self.interaction, ball2):
                        value = 8./pi/horizon**4 / 2.
                    elif isinstance(self.interaction, ballInf):
                        value = 3./4./horizon**4 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif kType == PERIDYNAMIC:
                if dim == 1:
                    value = 2./horizon**2 / 2.
                elif dim == 2:
                    if isinstance(self.interaction, ball2):
                        value = 6./pi/horizon**3 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif kType == GAUSSIAN:
                if dim == 1:
                    # value = 4.0/sqrt(pi)/(horizon/3.)**3 / 2.
                    value = 4.0/sqrt(pi)/(erf(3.0)-6.0*exp(-9.0)/sqrt(pi))/(horizon/3.0)**3 / 2.
                elif dim == 2:
                    if isinstance(self.interaction, ball2):
                        # value = 4.0/pi/(horizon/3.0)**4 / 2.
                        value = 4.0/pi/(1.0-10.0*exp(-9.0))/(horizon/3.0)**4 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        super(constantIntegrableScaling, self).__init__(value)

    def __getstate__(self):
        return (self.kType, self.interaction, self.dim, self.horizon)

    def __setstate__(self, state):
        constantIntegrableScaling.__init__(self, state[0], state[1], state[2], state[3])

    def __repr__(self):
        return '{}({} -> {})'.format(self.__class__.__name__, self.horizon, self.value)


cdef class variableFractionalLaplacianScaling(parametrizedTwoPointFunction):
    def __init__(self, BOOL_t symmetric, BOOL_t normalized=True, BOOL_t boundary=False, INDEX_t derivative=0):
        super(variableFractionalLaplacianScaling, self).__init__(symmetric)
        self.normalized = normalized
        self.boundary = boundary
        self.derivative = derivative
        self.digamma = memoizedDigamma()

    cdef void setParams(self, void *params):
        parametrizedTwoPointFunction.setParams(self, params)
        self.dim = getINDEX(self.params, fKDIM)

    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t s = getREAL(self.params, fS)
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t C, d2
            INDEX_t i

        if self.normalized:
            if horizon2 < inf:
                if self.dim == 1:
                    C = (2.-2*s) * pow(horizon2, s-1.) * 0.5
                elif self.dim == 2:
                    C = (2.-2*s) * pow(horizon2, s-1.) * 2./pi * 0.5
                elif self.dim == 3:
                    C = (2.-2*s) * pow(horizon2, s-1.) * 1./pi * 0.5
                else:
                    raise NotImplementedError()
            else:
                C = 2.0**(2.0*s) * s * gamma(s+0.5*self.dim) * pow(pi, -0.5*self.dim) / gamma(1.0-s) * 0.5
        else:
            C = 0.5

        if self.derivative == 0:
            return C
        elif self.derivative == 1:
            d2 = 0.
            for i in range(self.dim):
                d2 += (x[i]-y[i])*(x[i]-y[i])
            if self.normalized:
                if horizon2 < inf:
                    if not self.boundary:
                        return C*(-log(d2/horizon2) - 1./(1.-s))
                    else:
                        return C*(-log(d2/horizon2) - 1./(1.-s) - 1./s)
                else:
                    if not self.boundary:
                        return C*(-log(0.25*d2) + self.digamma.eval(s+0.5*self.dim) + self.digamma.eval(-s))
                    else:
                        return C*(-log(0.25*d2) + self.digamma.eval(s+0.5*self.dim) + self.digamma.eval(1.-s))
            else:
                if not self.boundary:
                    return C*(-log(d2))
                else:
                    return C*(-log(d2)-1./s)
        else:
            raise NotImplementedError()

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t s = getREAL(self.params, fS)
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t C, d2
            INDEX_t i

        if self.normalized:
            if horizon2 < inf:
                if self.dim == 1:
                    C = (2.-2*s) * pow(horizon2, s-1.) * 0.5
                elif self.dim == 2:
                    C = (2.-2*s) * pow(horizon2, s-1.) * 2./pi * 0.5
                elif self.dim == 3:
                    C = (2.-2*s) * pow(horizon2, s-1.) * 1./pi * 0.5
                else:
                    raise NotImplementedError()
            else:
                C = 2.0**(2.0*s) * s * gamma(s+0.5*self.dim) * pow(pi, -0.5*self.dim) / gamma(1.0-s) * 0.5
        else:
            C = 0.5

        if self.derivative == 0:
            return C
        elif self.derivative == 1:
            d2 = 0.
            for i in range(self.dim):
                d2 += (x[i]-y[i])*(x[i]-y[i])
            if self.normalized:
                if horizon2 < inf:
                    if not self.boundary:
                        return C*(-log(d2/horizon2) - 1./(1.-s))
                    else:
                        return C*(-log(d2/horizon2) - 1./(1.-s) - 1./s)
                else:
                    if not self.boundary:
                        return C*(-log(0.25*d2) + self.digamma.eval(s+0.5*self.dim) + self.digamma.eval(-s))
                    else:
                        return C*(-log(0.25*d2) + self.digamma.eval(s+0.5*self.dim) + self.digamma.eval(1.-s))
            else:
                if not self.boundary:
                    return C*(-log(d2))
                else:
                    return C*(-log(d2)-1./s)

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

    def __getstate__(self):
        return (self.symmetric, self.normalized, self.boundary, self.derivative)

    def __setstate__(self, state):
        variableFractionalLaplacianScaling.__init__(self, state[0], state[1], state[2], state[3])


######################################################################


cdef class variableFractionalLaplacianScalingWithDifferentHorizon(variableFractionalLaplacianScaling):
    def __init__(self, BOOL_t symmetric, BOOL_t normalized, BOOL_t boundary, INDEX_t derivative, function horizonFun):
        super(variableFractionalLaplacianScalingWithDifferentHorizon, self).__init__(symmetric, normalized, boundary, derivative)
        self.horizonFun = horizonFun

    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            void* params
            void* paramsModified = malloc(NUM_KERNEL_PARAMS*OFFSET)
            REAL_t horizon
        horizon = self.horizonFun.eval(x)
        params = self.getParams()
        memcpy(paramsModified, params, NUM_KERNEL_PARAMS*OFFSET)
        setREAL(paramsModified, fHORIZON2, horizon**2)
        self.setParams(paramsModified)
        scalingValue = variableFractionalLaplacianScaling.eval(self, x, y)
        self.setParams(params)
        return scalingValue

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            void* params
            void* paramsModified = malloc(NUM_KERNEL_PARAMS*OFFSET)
            REAL_t horizon
            REAL_t[::1] xA = <REAL_t[:dim]> x
        horizon = self.horizonFun.eval(xA)
        params = self.getParams()
        memcpy(paramsModified, params, NUM_KERNEL_PARAMS*OFFSET)
        setREAL(paramsModified, fHORIZON2, horizon**2)
        self.setParams(paramsModified)
        scalingValue = variableFractionalLaplacianScaling.evalPtr(self, dim, x, y)
        self.setParams(params)
        return scalingValue

    def __getstate__(self):
        return (self.symmetric, self.normalized, self.boundary, self.derivative, self.horizonFun)

    def __setstate__(self, state):
        variableFractionalLaplacianScalingWithDifferentHorizon.__init__(self, state[0], state[1], state[2], state[3], state[4])
