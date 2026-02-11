###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log
from PyNucleus_base.myTypes import REAL
from PyNucleus_base.myTypes cimport INDEX_t
from PyNucleus_base.blas cimport uninitializedREAL
from scipy.integrate import quad
from scipy.special._orthogonal import _gen_roots_and_weights
from . quadrature cimport quadQuadratureRule
from itertools import product


cdef class Polynomial(function):
    cdef:
        public REAL_t[::1] coeffcients

    def __init__(self, coeffcients):
        super(Polynomial, self).__init__()
        if not isinstance(coeffcients, np.ndarray):
            coeffcients = np.array(coeffcients)
        self.coeffcients = coeffcients

    @property
    def degree(self):
        return self.coeffcients.shape[0]-1

    cdef REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t k
            REAL_t val = 0.
        for k in range(self.degree+1):
            val += self.coeffcients[k]*pow(x[0], k)
        return val

    def __repr__(self):
        s = []
        c = self.coeffcients[0]
        if abs(c) > 0:
            s.append('{}'.format(c))
        for i in range(1, self.degree+1):
            c = self.coeffcients[i]
            if abs(c) > 0:
                if c == 1.0:
                    s.append('x^{}'.format(i))
                else:
                    s.append('{}*x^{}'.format(c, i))
        return ' + '.join(s)

    def __mul__(self, other):
        if isinstance(other, Polynomial) and isinstance(self, Polynomial):
            newCoeffs = np.zeros((self.degree+other.degree+1))
            for i in range(self.degree+1):
                for j in range(other.degree+1):
                    newCoeffs[i+j] += self.coeffcients[i]*other.coeffcients[j]
            return Polynomial(newCoeffs)
        elif isinstance(other, float) and isinstance(self, Polynomial):
            return Polynomial(np.array(self.coeffcients)*other)
        elif isinstance(other, Polynomial) and isinstance(self, float):
            return Polynomial(np.array(other.coeffcients)*self)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        newCoefficients = np.zeros((max(self.degree, other.degree)+1))
        newCoefficients[:self.degree+1] += np.array(self.coeffcients)
        newCoefficients[:other.degree+1] += np.array(other.coeffcients)
        return Polynomial(newCoefficients)

    def __sub__(self, other):
        newCoefficients = np.zeros((max(self.degree, other.degree)+1))
        newCoefficients[:self.degree+1] += np.array(self.coeffcients)
        newCoefficients[:other.degree+1] -= np.array(other.coeffcients)
        return Polynomial(newCoefficients)

    def __rmul__(self, other):
        if isinstance(other, Polynomial) and isinstance(self, Polynomial):
            newCoeffs = np.zeros((self.degree+other.degree+1))
            for i in range(self.degree+1):
                for j in range(other.degree+1):
                    newCoeffs[i+j] += self.coeffcients[i]*other.coeffcients[j]
            return Polynomial(newCoeffs)
        else:
            return Polynomial(np.array(self.coeffcients)*other)

    def __truediv__(self, other):
        return Polynomial(np.array(self.coeffcients)/other)

    def diff(self):
        return Polynomial(np.array(self.coeffcients)[1:]*np.arange(1, self.degree+1))


cdef class Integrand:
    cdef:
        function f
        function g
        function weight
        REAL_t[::1] xA

    def __init__(self, function f, function g, function weight):
        self.f = f
        self.g = g
        self.weight = weight
        self.xA = np.empty((1), dtype=REAL)

    cdef REAL_t eval(self, REAL_t x):
        self.xA[0] = x
        return self.f.eval(self.xA)*self.g.eval(self.xA)*self.weight.eval(self.xA)

    def __call__(self, REAL_t x):
        return self.eval(x)


cdef class IP:
    def __init__(self, function weight, REAL_t a, REAL_t b):
        self.weight = weight
        self.a = a
        self.b = b

    def __call__(self, function f, function g):
        cdef:
            Integrand I
        I = Integrand(f, g, self.weight)
        return quad(I, self.a, self.b)[0]

    def __repr__(self):
        return "{} weighted inner product on ({}, {})".format(self.weight, self.a, self.b)


cdef class LegendreWeightShifted(function):
    cdef:
        REAL_t alpha
        REAL_t beta

    cdef REAL_t eval(self, REAL_t[::1] x):
        return 1.

    def __repr__(self):
        return '1'


cdef class LegendreShiftedIP(IP):
    def __init__(self):
        ip = LegendreWeightShifted()
        super(LegendreShiftedIP, self).__init__(ip, 0., 1.)


cdef class JacobiWeightShifted(function):
    cdef:
        REAL_t alpha
        REAL_t beta

    def __init__(self, REAL_t alpha, REAL_t beta):
        assert alpha > -1., "alpha needs to be > -1."
        assert beta > -1., "beta needs to be > -1."
        self.alpha = alpha
        self.beta = beta

    cdef REAL_t eval(self, REAL_t[::1] x):
        return pow(x[0], self.alpha)*pow(1.-x[0], self.beta)

    def __repr__(self):
        return "x**{} * (1-x)**{}".format(self.alpha, self.beta)


cdef class JacobiShiftedIP(IP):
    def __init__(self, REAL_t alpha, REAL_t beta):
        ip = JacobiWeightShifted(alpha, beta)
        super(JacobiShiftedIP, self).__init__(ip, 0., 1.)


cdef class LogJacobiWeightShifted(function):
    cdef:
        REAL_t alpha
        REAL_t beta
        REAL_t gamma

    def __init__(self, REAL_t alpha, REAL_t beta, REAL_t gamma):
        assert alpha > -1., "alpha needs to be > -1."
        assert beta > -1., "beta needs to be > -1."
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    cdef REAL_t eval(self, REAL_t[::1] x):
        return pow(x[0], self.alpha)*pow(1.-x[0], self.beta)*pow(-log(x[0]), self.gamma)

    def __repr__(self):
        return "x**{} * (1-x)**{} * (-log(x))**{}".format(self.alpha, self.beta, self.gamma)


cdef class LogJacobiShiftedIP(IP):
    def __init__(self, REAL_t alpha, REAL_t beta, REAL_t gamma):
        ip = LogJacobiWeightShifted(alpha, beta, gamma)
        super(LogJacobiShiftedIP, self).__init__(ip, 0., 1.)


class QuadRuleBuilder:
    def __init__(self, ip):
        self.ip = ip
        self.p = []
        self.normP = []
        self.a = []
        self.b = []

    def computeBasis(self, maxDegree):
        if len(self.p) == 0:
            qNew = Polynomial([1.])
            normQ = sqrt(self.ip(qNew, qNew))
            self.normP.append(normQ)
            self.p.append(qNew)

        x = Polynomial([0., 1.])
        while len(self.p) <= maxDegree:
            qNew = x*self.p[len(self.p)-1]
            self.a.append(self.ip(qNew, self.p[len(self.p)-1])/self.normP[len(self.normP)-1]**2)
            if len(self.p) >= 2:
                self.b.append(self.ip(qNew, self.p[len(self.p)-2])/self.normP[len(self.normP)-2]**2)
                qNew = qNew - self.a[len(self.a)-1]*self.p[len(self.p)-1] - self.b[len(self.b)-1]*self.p[len(self.p)-2]
            else:
                qNew = qNew - self.a[len(self.a)-1]*self.p[len(self.p)-1]
            normQ = sqrt(self.ip(qNew, qNew))
            self.normP.append(normQ)
            self.p.append(qNew)

    def build(self, INDEX_t maxDegree):
        self.computeBasis(maxDegree)

        a_s = np.array(self.a)
        b_s = np.array(self.b)

        def an_func(n):
            n = n.astype(int)
            return a_s[n]

        def bn_func(n):
            n = n.astype(int)-1
            return np.sqrt(b_s[n])

        def eval(n, x):
            poly = self.p[n]
            xA = np.empty((1), dtype=REAL)
            y = np.empty_like(x)
            for i in range(x.shape[0]):
                xA[0] = x[i]
                y[i] = poly(xA)
            return y

        def eval_diff(n, x):
            poly = self.p[n].diff()
            xA = np.empty((1), dtype=REAL)
            y = np.empty_like(x)
            for i in range(x.shape[0]):
                xA[0] = x[i]
                y[i] = poly(xA)
            return y

        one = Polynomial([1.])
        mu0 = self.ip(one, one)
        nodes, weights = _gen_roots_and_weights(maxDegree, mu0, an_func, bn_func, eval, eval_diff, False, False)
        qr = nodes, weights
        return qr
