###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from libc.math cimport (sin, cos, sinh, cosh, tanh, sqrt, atan2,
                        M_PI as pi, pow, exp, floor, log2, log)
from . zeta cimport zetaCy as zeta
from scipy.special.cython_special cimport psi as digamma
from scipy.special.cython_special cimport gamma as cgamma
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, BOOL_t
from PyNucleus_base.blas import uninitialized
from PyNucleus_fem.functions cimport function


cdef inline REAL_t polygamma(INDEX_t n, REAL_t d) noexcept:
    return (-1.0)**(n+1) * cgamma(n+1.0) * zeta(n+1, d)


cdef class solFractional(function):
    cdef:
        public REAL_t s
        public REAL_t fac
        REAL_t radius2
        INDEX_t dim
        public REAL_t L2norm

    def __init__(self, REAL_t s, INDEX_t dim, REAL_t radius=1.0):
        function.__init__(self)
        from scipy.special import gamma
        self.s = s
        self.dim = dim
        self.radius2 = radius**2
        self.fac = self.radius2**s * 2.**(-2.*s)*gamma(dim/2.)/gamma((dim+2.*s)/2.)/gamma(1.+s)
        if dim == 1:
            vol = 2.
        elif dim == 2:
            vol = 2.*pi
        elif dim == 3:
            vol = 4.*pi
        else:
            raise NotImplemented(dim)
        if dim == 1:
            # orthogonality of Jacobi polynomials
            P_ss0_P_ss0 = pow(2., 1+4*s) * gamma(1+2*s)**2 / gamma(1+4*s) / (1+4*s) / gamma(1.)
            ip = P_ss0_P_ss0
            self.L2norm = self.fac * pow(vol, 0.5)  * pow(radius, 0.5*dim) * sqrt(0.5 * ip)
        elif dim == 2:
            self.L2norm = self.fac * sqrt(vol/2./(1+2*s)) * radius
        elif dim == 3:
            P_ss1_P_ss1 = pow(2., 1+4*s) * gamma(1+2*s+1)**2 / gamma(1+4*s+1) / (1+4*s+2) / gamma(2.)
            ip = pow(s+1., -2.) * P_ss1_P_ss1
            self.L2norm = self.fac * pow(vol, 0.5)  * pow(radius, 0.5*dim) * sqrt(0.5 * ip)
        else:
            raise NotImplemented(dim)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t r2 = 0.
        cdef INDEX_t i
        for i in range(self.dim):
            r2 += x[i]**2
        if r2 <= self.radius2:
            return self.fac*pow(1.-r2/self.radius2, self.s)
        else:
            return 0.


from scipy.special import eval_jacobi as jacobi


cdef class rhsFractional1D(function):
    cdef:
        public REAL_t s
        REAL_t fac
        public INDEX_t n

    def __init__(self, REAL_t s, INDEX_t n):
        from scipy.special import gamma
        function.__init__(self)
        self.s = s
        self.n = n
        self.fac = 2.**(2.*s)*gamma(0.5+s+n)*gamma(1.+s+n)/gamma(1.+n)/gamma(0.5+n)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t r2 = 0.
        r2 = x[0]**2
        if r2 <= 1.:
            return self.fac * jacobi(self.n, self.s, -0.5, 2.*r2-1.)
        else:
            return 0.


cdef class solFractional1D(function):
    cdef:
        public REAL_t s
        public INDEX_t n

    def __init__(self, REAL_t s, INDEX_t n):
        function.__init__(self)
        self.s = s
        self.n = n

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t r2 = 0.
        r2 = x[0]**2
        if r2 <= 1.:
            return (1.-r2)**self.s * jacobi(self.n, self.s, -0.5, 2.*r2-1.)
        else:
            return 0.


cdef class rhsFractional2D(function):
    cdef:
        public REAL_t s
        public REAL_t angular_shift
        public INDEX_t l
        public INDEX_t n
        REAL_t fac

    def __init__(self, REAL_t s, INDEX_t l, INDEX_t n, REAL_t angular_shift=0.):
        function.__init__(self)
        from scipy.special import gamma
        self.s = s
        self.l = l
        self.n = n
        self.angular_shift = angular_shift
        self.fac = 2.**(2.*s)*gamma(1.+s+n)*gamma(1.+l+s+n)/gamma(1+n)/gamma(1.+l+n)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t r2 = 0., theta = atan2(x[1], x[0])
        r2 = x[0]**2+x[1]**2
        if r2 <= 1.:
            return self.fac*r2**(0.5*self.l)*cos(self.l*(theta+self.angular_shift))*jacobi(self.n, self.s, self.l, 2.*r2-1.)
        else:
            return 0.


cdef class solFractional2D(function):
    cdef:
        public REAL_t s
        public REAL_t angular_shift
        public INDEX_t l
        public INDEX_t n

    def __init__(self, REAL_t s, INDEX_t l, INDEX_t n, REAL_t angular_shift=0.):
        function.__init__(self)
        self.s = s
        self.l = l
        self.n = n
        self.angular_shift = angular_shift

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t r2 = 0., theta = atan2(x[1], x[0])
        r2 = x[0]**2+x[1]**2
        if r2 <= 1.:
            return (1.-r2)**self.s*r2**(0.5*self.l)*cos(self.l*(theta+self.angular_shift))*jacobi(self.n, self.s, self.l, 2.*r2-1.)
        else:
            return 0.


cdef class rhsFractional2Dcombination(function):
    cdef list functions

    def __init__(self, REAL_t s, params):
        function.__init__(self)
        self.functions = [rhsFractional2D(s, **p) for p in params]

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t val = 0.
            INDEX_t i
            function f
        for i in range(len(self.functions)):
            f = self.functions[i]
            val += f.eval(x)
        return val


cdef class solFractional2Dcombination(function):
    cdef list functions

    def __init__(self, REAL_t s, params):
        function.__init__(self)
        self.functions = [solFractional2D(s, **p) for p in params]

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t val = 0.
            INDEX_t i
            function f
        for i in range(len(self.functions)):
            f = self.functions[i]
            val += f.eval(x)
        return val


cdef class rhsTestFractional_U(function):
    cdef REAL_t t
    cdef function sol

    def __init__(self, REAL_t s, INDEX_t dim, REAL_t t, REAL_t radius=1.0):
        function.__init__(self)
        self.sol = solFractional(s, dim, radius)
        self.t = t

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t u = self.sol.eval(x)
        return cos(self.t)*u + (cos(self.t)**2-sin(self.t)**2)*u**2 + sin(self.t)


cdef class rhsTestFractional_V(function):
    cdef REAL_t t
    cdef function sol

    def __init__(self, REAL_t s, INDEX_t dim, REAL_t t, REAL_t radius=1.0):
        function.__init__(self)
        self.sol = solFractional(s, dim, radius)
        self.t = t

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t u = self.sol.eval(x)
        return -sin(self.t)*u + (-cos(self.t)**2+sin(self.t)**2)*u**2 + cos(self.t)


cdef class rhsFractionalBrusselator_U(function):
    cdef REAL_t t, B, Q, eta, radius2s
    cdef function solU, solV

    def __init__(self, REAL_t s1, REAL_t s2,
                 REAL_t B, REAL_t Q, REAL_t eta,
                 INDEX_t dim, REAL_t t, REAL_t radius=1.0):
        function.__init__(self)
        self.solU = solFractional(s1, dim, radius)
        self.solV = solFractional(s2, dim, radius)
        self.B = B
        self.Q = Q
        self.eta = eta
        self.t = t
        self.radius2s = radius**(2.*s1)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t u0 = self.solU.eval(x)*self.eta
            REAL_t v0 = self.solV.eval(x)/self.eta
            REAL_t s = sin(self.t)
            REAL_t c = cos(2.*self.t)
            REAL_t u = u0*s
            REAL_t v = v0*c
        return (cos(self.t)*u0) + s*self.eta/self.radius2s - ((self.B-1.)*u + self.Q**2*v + self.B/self.Q*u**2 + 2.*self.Q*u*v + u**2*v)


cdef class rhsFractionalBrusselator_V(function):
    cdef REAL_t t, B, Q, eta, radius2s
    cdef function solU, solV

    def __init__(self, REAL_t s1, REAL_t s2,
                 REAL_t B, REAL_t Q, REAL_t eta,
                 INDEX_t dim, REAL_t t, REAL_t radius=1.0):
        function.__init__(self)
        self.solU = solFractional(s1, dim, radius)
        self.solV = solFractional(s2, dim, radius)
        self.B = B
        self.Q = Q
        self.eta = eta
        self.t = t
        self.radius2s = radius**(2.*s2)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t u0 = self.solU.eval(x)*self.eta
            REAL_t v0 = self.solV.eval(x)/self.eta
            REAL_t s = sin(self.t)
            REAL_t c = cos(2.*self.t)
            REAL_t u = u0*s
            REAL_t v = v0*c
        return self.eta**2*(-2.*sin(2.*self.t)*v0) + c/self.eta/self.radius2s + (self.B*u + self.Q**2*v + self.B/self.Q*u**2 + 2.*self.Q*u*v + u**2*v)


cdef class solFractionalDerivative(function):
    cdef public REAL_t s
    cdef REAL_t fac, fac2, fac3, radius2
    cdef INDEX_t dim
    cdef INDEX_t derivative

    def __init__(self, REAL_t s, INDEX_t dim, REAL_t radius=1.0, INDEX_t derivative=1):
        function.__init__(self)
        from scipy.special import gamma
        self.s = s
        self.dim = dim
        self.radius2 = radius**2
        self.fac = self.radius2**s * 2.**(-2.*s)*gamma(dim/2.)/gamma((dim+2.*s)/2.)/gamma(1.+s)
        self.fac2 = log(0.25*self.radius2) - digamma(0.5*dim+s) - digamma(1+s)
        self.fac3 = -polygamma(1, 0.5*dim+s)-polygamma(1, 1+s)
        self.derivative = derivative
        assert self.derivative in (0, 1, 2)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t r2 = 0.
        cdef INDEX_t i
        for i in range(self.dim):
            r2 += x[i]**2
        if r2 < 1.:
            if self.derivative == 0:
                return self.fac*pow(1.-r2/self.radius2, self.s)
            elif self.derivative == 1:
                return (self.fac2+log(1.-r2/self.radius2)) * self.fac*pow(1.-r2/self.radius2, self.s)
            elif self.derivative == 2:
                return ((self.fac2 + log(1.-r2/self.radius2))**2 + self.fac3) * self.fac*pow(1.-r2/self.radius2, self.s)
        else:
            return 0.


cdef class solutionArbitraryOrder(function):
    """
    u(x) = (1-\\frac{|x|^2}{R^2})^{\\beta}
    """
    cdef:
        public INDEX_t dim
        public REAL_t beta
        public REAL_t radius2
        public INDEX_t derivative
        public REAL_t L2norm

    def __init__(self, INDEX_t dim, REAL_t beta, REAL_t radius=1., INDEX_t derivative=0):
        self.dim = dim
        self.beta = beta
        self.radius2 = radius**2
        if derivative == 0:
            # volume of ball with size=radius
            if self.dim == 1:
                vol = 2.*radius
            elif self.dim == 2:
                vol = pi*self.radius2
            else:
                raise NotImplementedError()

            self.L2norm = sqrt(vol*hyp2f1(0.5*dim, -2*beta, 0.5*dim+1, 1.))
        else:
            self.L2norm = 0.

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t r2 = 0.
            INDEX_t i
        if self.derivative == 0:
            for i in range(self.dim):
                r2 += x[i]**2
            r2 /= self.radius2
            if r2 < 1.:
                return pow(1.-r2, self.beta)
            else:
                return 0.
        else:
            return 0.


from scipy.special import hyp2f1

from . fractionalOrders cimport fractionalOrderBase
from . fractionalOrders import (constFractionalOrder,
                                variableConstFractionalOrder,
                                constantNonSymFractionalOrder,
                                singleVariableUnsymmetricFractionalOrder)
from scipy.special.cython_special cimport gamma as cgamma


cdef inline REAL_t gamma(REAL_t d) noexcept:
    return cgamma(d)


cdef class rhsArbitraryOrder(function):
    cdef:
        INDEX_t dim
        public REAL_t beta
        public fractionalOrderBase sFun
        public INDEX_t derivative
        REAL_t eps
        REAL_t[::1] dir, grad
        public BOOL_t normalized

    def __init__(self, INDEX_t dim, REAL_t beta, fractionalOrderBase sFun, INDEX_t derivative=0, eps=1e-6, REAL_t[::1] dir=None, BOOL_t normalized=True):
        self.dim = dim
        self.beta = beta
        assert isinstance(sFun, (constFractionalOrder, variableConstFractionalOrder,
                                 constantNonSymFractionalOrder, singleVariableUnsymmetricFractionalOrder))
        self.sFun = sFun
        self.derivative = derivative
        self.eps = eps
        self.dir = dir
        self.normalized = normalized
        assert derivative in (0, 1, 2), derivative
        if derivative == 1:
            assert self.dir.shape[0] == self.sFun.numParameters
            self.grad = uninitialized(self.sFun.numParameters)
        elif derivative == 2:
            assert self.dir.shape[0] == self.sFun.numParameters**2
            self.grad = uninitialized(self.sFun.numParameters)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t r2 = 0.
            REAL_t s, s2, s3, val, val2, val3
            INDEX_t i, j
            REAL_t dhyp_ds, d2hyp_ds2, C, dC_ds, d2C_ds2
            REAL_t df_ds, d2f_ds2, ds_dp, d2s_dp2

        self.sFun.evalPtr(self.dim, &x[0], &x[0], &s)

        for i in range(x.shape[0]):
            r2 += x[i]**2
        if self.derivative == 0:
            if self.normalized:
                C = 2**(2*s) * gamma(0.5*(self.dim+2*s)) * gamma(1.+self.beta) / gamma(0.5*self.dim) / gamma(self.beta-s+1.)
            else:
                C = gamma(1.+self.beta) / gamma(0.5*self.dim) / gamma(self.beta-s+1.) / (s * pow(pi, -0.5*self.dim) / gamma(1.0-s))
            return C * hyp2f1(0.5*(self.dim+2*s), s-self.beta, 0.5*self.dim, r2)
        elif self.derivative == 1:
            s2 = s-self.eps
            if self.normalized:
                C = 2**(2*s) * gamma(0.5*(self.dim+2*s)) * gamma(1.+self.beta) / gamma(0.5*self.dim) / gamma(self.beta-s+1.)
                dC_ds = (2.*log(2.) + digamma(0.5*(self.dim+2*s)) + digamma(self.beta-s+1.)) * C
            else:
                C = gamma(1.+self.beta) / gamma(0.5*self.dim) / gamma(self.beta-s+1.) / (s * pow(pi, -0.5*self.dim) / gamma(1.0-s))
                dC_ds = (digamma(self.beta-s+1.) - 1./s - digamma(1.0-s)) * C

            val = hyp2f1(0.5*(self.dim+2*s), s-self.beta, 0.5*self.dim, r2)
            val2 = hyp2f1(0.5*(self.dim+2*s2), s2-self.beta, 0.5*self.dim, r2)

            dhyp_ds = (val-val2)/self.eps
            df_ds = C*dhyp_ds + dC_ds*val

            self.sFun.evalGradPtr(self.dim, &x[0], &x[0], self.sFun.numParameters, &self.grad[0])
            ds_dp = 0.
            for i in range(self.sFun.numParameters):
                ds_dp += self.dir[i]*self.grad[i]
            return df_ds * ds_dp

        elif self.derivative == 2:
            s2 = s-self.eps
            s3 = s+self.eps
            if self.normalized:
                C = 2**(2*s) * gamma(0.5*(self.dim+2*s)) * gamma(1.+self.beta) / gamma(0.5*self.dim) / gamma(self.beta-s+1.)
                dC_ds = (2.*log(2.) + digamma(0.5*(self.dim+2*s)) + digamma(self.beta-s+1.)) * C
                d2C_ds2 = (polygamma(1, 0.5*(self.dim+2*s)) - polygamma(1, self.beta-s+1.) + (2.*log(2.) + digamma(0.5*(self.dim+2*s)) + digamma(self.beta-s+1.))**2) * C
            else:
                C = gamma(1.+self.beta) / gamma(0.5*self.dim) / gamma(self.beta-s+1.) / (s * pow(pi, -0.5*self.dim) / gamma(1.0-s))
                dC_ds = (digamma(self.beta-s+1.) - 1./s - digamma(1.0-s)) * C
                d2C_ds2 = (-polygamma(1, self.beta-s+1.) + 1/s**2 + polygamma(1, 1.0-s) +  (digamma(self.beta-s+1.) - 1./s - digamma(1.0-s))**2) * C

            val = hyp2f1(0.5*(self.dim+2*s), s-self.beta, 0.5*self.dim, r2)
            val2 = hyp2f1(0.5*(self.dim+2*s2), s2-self.beta, 0.5*self.dim, r2)
            val3 = hyp2f1(0.5*(self.dim+2*s3), s3-self.beta, 0.5*self.dim, r2)

            dhyp_ds = (val-val2)/self.eps
            d2hyp_ds2 = (-2*val+val2+val3)/self.eps/self.eps

            d2f_ds2 = C*d2hyp_ds2 + 2*dC_ds*dhyp_ds + d2C_ds2*val

            self.sFun.evalGradPtr(self.dim, &x[0], &x[0], self.sFun.numParameters, &self.grad[0])
            d2s_dp2 = 0.
            for i in range(self.sFun.numParameters):
                for j in range(self.sFun.numParameters):
                    d2s_dp2 += self.dir[self.sFun.numParameters*i+j]*self.grad[i]*self.grad[j]
            return d2f_ds2 * d2s_dp2
