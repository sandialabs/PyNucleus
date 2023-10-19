###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from libc.math cimport (sin, cos, sinh, cosh, tanh, sqrt, atan2,
                        M_PI as pi, pow, exp, floor, log2, log)
from scipy.special.cython_special cimport psi as digamma
import numpy as np
cimport numpy as np

from PyNucleus_base.myTypes import INDEX, REAL, COMPLEX, ENCODE
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, ENCODE_t
from PyNucleus_base import uninitialized
from PyNucleus_base.blas cimport norm

from . quadrature cimport sphericalQuadRule, sphericalQuadRule1D, sphericalQuadRule2D


cdef class function:
    def __call__(self, REAL_t[::1] x):
        return self.eval(x)

    cdef REAL_t eval(self, REAL_t[::1] x):
        pass

    def __add__(self, function other):
        if isinstance(self, mulFunction):
            if isinstance(other, mulFunction):
                return sumFunction(self.f, self.fac, other.f, other.fac)
            else:
                return sumFunction(self.f, self.fac, other, 1.)
        else:
            if isinstance(other, mulFunction):
                return sumFunction(self, 1., other.f, other.fac)
            else:
                return sumFunction(self, 1., other, 1.)

    def __sub__(self, function other):
        if isinstance(self, mulFunction):
            if isinstance(other, mulFunction):
                return sumFunction(self.f, self.fac, other.f, -other.fac)
            else:
                return sumFunction(self.f, self.fac, other, -1.)
        else:
            if isinstance(other, mulFunction):
                return sumFunction(self, 1., other.f, -other.fac)
            else:
                return sumFunction(self, 1., other, -1.)

    def __mul__(first, second):
        if isinstance(first, function) and isinstance(second, function):
            return prodFunction(first, second)
        elif isinstance(first, function):
            return mulFunction(first, second)
        elif isinstance(second, function):
            return mulFunction(second, first)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        if isinstance(self, mulFunction):
            return mulFunction(self.f, -self.fac)
        elif isinstance(self, sumFunction):
            return sumFunction(self.f1, -self.fac1, self.f2, -self.fac2)
        else:
            return mulFunction(self, -1.0)

    def __repr__(self):
        return '%s' % (self.__class__.__name__)


cdef class sumFunction(function):
    cdef:
        public function f1, f2
        public REAL_t fac1, fac2

    def __init__(self, function f1, REAL_t fac1, function f2, REAL_t fac2):
        self.f1 = f1
        self.fac1 = fac1
        self.f2 = f2
        self.fac2 = fac2

    cdef REAL_t eval(self, REAL_t[::1] x):
        return self.fac1*self.f1.eval(x)+self.fac2*self.f2.eval(x)

    def __repr__(self):
        return '{}*{}+{}*{}'.format(self.fac1, self.f1, self.fac2, self.f2)


cdef class mulFunction(function):
    cdef:
        public function f
        public REAL_t fac

    def __init__(self, function f, REAL_t fac):
        self.f = f
        self.fac = fac

    cdef REAL_t eval(self, REAL_t[::1] x):
        return self.fac*self.f.eval(x)

    def __repr__(self):
        return '{}*{}'.format(self.fac, self.f)


cdef class prodFunction(function):
    cdef:
        public function f1, f2

    def __init__(self, function f1, function f2):
        self.f1 = f1
        self.f2 = f2

    cdef REAL_t eval(self, REAL_t[::1] x):
        return self.f1.eval(x)*self.f2.eval(x)

    def __repr__(self):
        return '{}*{}'.format(self.f1, self.f2)


# FIX: This doesn't outperform the version without memory
#      Maybe I should create a binary tree on x, given that
#      I know the interval of values
cdef class _memoized_sin:
    cdef:
        dict memory
        int hit, miss

    def __init__(self):
        self.memory = dict()
        self.hit = 0
        self.miss = 0

    cdef inline REAL_t eval(self, REAL_t x):
        cdef REAL_t val
        try:
            val = self.memory[x]
            self.hit += 1
            return val
        except KeyError:
            self.miss += 1
            val = sin(x)
            self.memory[x] = val
            return val

    def stats(self):
        print(len(self.memory), self.hit, self.miss)


cdef _memoized_sin memoized_sin = _memoized_sin()


cdef class Lambda(function):
    cdef:
        object fun

    def __init__(self, fun):
        self.fun = fun

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return self.fun(x)


cdef class constant(function):
    def __init__(self, REAL_t value):
        self.value = value

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return self.value

    def __repr__(self):
        return '{}'.format(self.value)

    def __eq__(self, other):
        cdef:
            constant o
        if isinstance(other, constant):
            o = other
            return self.value == o.value
        else:
            return False



cdef class monomial(function):
    cdef:
        REAL_t[::1] exponent
        REAL_t factor

    def __init__(self, REAL_t[::1] exponent, REAL_t factor=1.):
        self.exponent = exponent
        self.factor = factor

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t i
            REAL_t s = self.factor
        for i in range(x.shape[0]):
            s *= pow(x[i], self.exponent[i])
        return s

    def __repr__(self):
        cdef:
            INDEX_t i
        s = ''
        for i in range(self.exponent.shape[0]):
            if self.exponent[i] != 0.:
                if len(s) > 0:
                    s += '*'
                if self.exponent[i] != 1.:
                    s += 'x_{}^{}'.format(i, self.exponent[i])
                else:
                    s += 'x_{}'.format(i)
        if self.factor != 1.:
            return str(self.factor) + s
        else:
            return s


cdef class _rhsFunSin1D(function):
    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return pi**2.0*sin(pi*x[0])


cdef class _solSin1D(function):
    cdef:
        REAL_t k

    def __init__(self, INDEX_t k=1):
        self.k = k*pi

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return sin(self.k*x[0])


cdef class _cos1D(function):
    cdef:
        REAL_t k

    def __init__(self, INDEX_t k=1):
        self.k = k*pi

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return cos(self.k*x[0])


cdef class _rhsFunSin2D(function):
    cdef:
        REAL_t k, l, fac

    def __init__(self, INDEX_t k=1, INDEX_t l=1):
        self.k = k*pi
        self.l = l*pi
        self.fac = self.k**2 + self.l**2

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return self.fac * sin(self.k*x[0])*sin(self.l*x[1])


cdef class _cos2D(function):
    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return cos(pi*x[0])*cos(pi*x[1])


cdef class _rhsCos2D(function):
    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return 2.0*pi**2*cos(pi*x[0])*cos(pi*x[1])


cdef class _grad_cos2d_n(function):
    cdef inline REAL_t eval(self, REAL_t[::1] x):
        if x[0] == 1.0:
            return -sin(1.0)*cos(x[1])
        elif x[1] == 1.0:
            return -sin(1.0)*cos(x[0])


cdef class _solSin2D(function):
    cdef:
        REAL_t k, l

    def __init__(self, INDEX_t k=1, INDEX_t l=1):
        self.k = k*pi
        self.l = l*pi

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return sin(self.k*x[0])*sin(self.l*x[1])


cdef class _rhsFunSin3D(function):
    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return 3.0*pi**2.0*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])


cdef class _rhsFunSin3D_memoized(function):
    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return 3.0*pi**2.0*memoized_sin.eval(pi*x[0])*memoized_sin.eval(pi*x[1])*memoized_sin.eval(pi*x[2])


cdef class _solSin3D(function):
    cdef:
        REAL_t k, l, m

    def __init__(self, INDEX_t k=1, INDEX_t l=1, INDEX_t m=1):
        self.k = k*pi
        self.l = l*pi
        self.m = m*pi

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return sin(self.k*x[0])*sin(self.l*x[1])*sin(self.m*x[2])


cdef class _rhsBoundaryLayer2D(function):
    cdef:
        public REAL_t radius, c

    def __init__(self, REAL_t radius=0.25, REAL_t c=100.0):
        self.radius = radius
        self.c = c

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t r, z
        r = sqrt((x[0]-0.5)**2.0 + (x[1]-0.5)**2.0)
        z = r**2.0 - self.radius**2.0
        return -4.0*self.c/cosh(self.c*z)**2.0 + 8.0*self.c**2.0*r**2.0*sinh(self.c*z)/cosh(self.c*z)**3.0


cdef class _solBoundaryLayer2D(function):
    cdef:
        public REAL_t radius, c

    def __init__(self, REAL_t radius=0.25, REAL_t c=100.0):
        self.radius = radius
        self.c = c

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t r, z
        r = sqrt((x[0]-0.5)**2.0 + (x[1]-0.5)**2.0)
        z = r**2.0 - self.radius**2.0
        return tanh(self.c*z)-1.0


cdef class _solCornerSingularity2D(function):
    cdef REAL_t twoThirds

    def __init__(self):
        self.twoThirds = 2.0/3.0

    cdef inline REAL_t eval(self, const REAL_t[::1] x):
        cdef:
            REAL_t y0, y1, theta
        y0, y1 = x[1]-1, -x[0]+1
        r = sqrt(y0**2.0 + y1**2.0)
        theta = np.arctan2(y1, y0)
        if theta < 0:
            theta += 2.0*pi
        return r**self.twoThirds*np.sin(self.twoThirds*theta)


cdef class rhsBoundarySingularity2D(function):
    cdef REAL_t alpha

    def __init__(self, REAL_t alpha):
        self.alpha = alpha

    cdef inline REAL_t eval(self, const REAL_t[::1] x):
        if x[0] > 0:
            return self.alpha*(1.-self.alpha)*pow(x[0], self.alpha-2.)
        else:
            return 1000.


cdef class solBoundarySingularity2D(function):
    cdef REAL_t alpha

    def __init__(self, REAL_t alpha):
        self.alpha = alpha

    cdef inline REAL_t eval(self, const REAL_t[::1] x):
        return x[0]**self.alpha


cdef class _rhsFichera(function):
    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return -0.75*pow((x[0]-1.0)**2.0+(x[1]-1.0)**2.0+(x[2]-1.0)**2.0, -0.75)


cdef class _solFichera(function):
    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return pow((x[0]-1.0)**2.0+(x[1]-1.0)**2.0+(x[2]-1.0)**2.0, 0.25)


cdef class rhsFunCos1DHeat(function):
    cdef REAL_t t

    def __init__(self, REAL_t t):
        function.__init__(self)
        self.t = t

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return (cos(self.t)+pi**2.0*sin(self.t))*cos(pi*x[0])


cdef class rhsFunSource1D(function):
    cdef REAL_t a, b

    def __init__(self, REAL_t a, REAL_t b):
        function.__init__(self)
        self.a = a
        self.b = b

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return (self.a <= x[0]) and (x[0] < self.b)


cdef class solCos1DHeat(function):
    cdef REAL_t t

    def __init__(self, REAL_t t):
        function.__init__(self)
        self.t = t

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return sin(self.t)*cos(pi*x[0])


cdef class rhsFunCos2DHeat(function):
    cdef REAL_t t

    def __init__(self, REAL_t t):
        function.__init__(self)
        self.t = t

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return (cos(self.t)+2.0*pi**2.0*sin(self.t))*cos(pi*x[0])*cos(pi*x[1])


cdef class rhsFunCos2DNonlinear(function):
    cdef REAL_t t, k

    def __init__(self, REAL_t t, REAL_t k=2.):
        function.__init__(self)
        self.t = t
        self.k = k

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return ((cos(self.t) +
                 2.0*pi**2.0*sin(self.t))*cos(pi*x[0])*cos(pi*x[1]) -
                (sin(self.t)*cos(pi*x[0])*cos(pi*x[1]))**self.k)


cdef class rhsFunCos2DNonlinear_U(function):
    cdef REAL_t t, k

    def __init__(self, REAL_t t, REAL_t k=2.):
        function.__init__(self)
        self.t = t
        self.k = k

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return ((cos(self.t) +
                 2.0*pi**2.0*sin(self.t))*cos(pi*x[0])*cos(pi*x[1]) +
                ((cos(self.t)*cos(pi*x[0])*cos(pi*x[1]))**self.k -
                 (sin(self.t)*cos(pi*x[0])*cos(pi*x[1]))**self.k))


cdef class rhsFunCos2DNonlinear_V(function):
    cdef REAL_t t, k

    def __init__(self, REAL_t t, REAL_t k=2.):
        function.__init__(self)
        self.t = t
        self.k = k

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return ((-sin(self.t) +
                 2.0*pi**2.0*cos(self.t))*cos(pi*x[0])*cos(pi*x[1]) +
                ((sin(self.t)*cos(pi*x[0])*cos(pi*x[1]))**self.k -
                 (cos(self.t)*cos(pi*x[0])*cos(pi*x[1]))**self.k))


cdef class solCos2DHeat(function):
    cdef REAL_t t

    def __init__(self, REAL_t t):
        function.__init__(self)
        self.t = t

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return sin(self.t)*cos(pi*x[0])*cos(pi*x[1])


cdef class rhsFunSource2D(function):
    cdef REAL_t[::1] a
    cdef REAL_t r2

    def __init__(self, REAL_t[::1] a, REAL_t r):
        function.__init__(self)
        self.a = a
        self.r2 = r**2

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return (x[0]-self.a[0])**2+(x[1]-self.a[1])**2 < self.r2


cdef class rhsTestGrayScott2D_U(function):
    cdef REAL_t k, F, Du, Dv, t

    def __init__(self, REAL_t k, REAL_t F, REAL_t Du, REAL_t Dv, REAL_t t):
        function.__init__(self)
        self.k = k
        self.F = F
        self.Du = Du
        self.Dv = Dv
        self.t = t

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t u, v
        u = sin(self.t)*cos(pi*x[0])*cos(pi*x[1])
        v = cos(self.t)*cos(pi*x[0])*cos(pi*x[1])
        return v+2*pi**2*self.Du*u+u*v**2-self.F*(1-u)


cdef class rhsTestGrayScott2D_V(function):
    cdef REAL_t k, F, Du, Dv, t

    def __init__(self, REAL_t k, REAL_t F, REAL_t Du, REAL_t Dv, REAL_t t):
        function.__init__(self)
        self.k = k
        self.F = F
        self.Du = Du
        self.Dv = Dv
        self.t = t

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t u, v
        u = sin(self.t)*cos(pi*x[0])*cos(pi*x[1])
        v = cos(self.t)*cos(pi*x[0])*cos(pi*x[1])
        return -u+2*pi**2*self.Dv*v-u*v**2+(self.k+self.F)*v


cdef class solFractional(function):
    cdef public REAL_t s
    cdef REAL_t fac, radius2
    cdef INDEX_t dim

    def __init__(self, REAL_t s, INDEX_t dim, REAL_t radius=1.0):
        function.__init__(self)
        from scipy.special import gamma
        self.s = s
        self.dim = dim
        self.radius2 = radius**2
        self.fac = self.radius2**s * 2.**(-2.*s)*gamma(dim/2.)/gamma((dim+2.*s)/2.)/gamma(1.+s)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t r2 = 0.
        cdef INDEX_t i
        for i in range(self.dim):
            r2 += x[i]**2
        if r2 <= self.radius2:
            return self.fac*pow(1.-r2/self.radius2, self.s)
        else:
            return 0.


cdef class solFractionalDerivative(function):
    cdef public REAL_t s
    cdef REAL_t fac, fac2, radius2
    cdef INDEX_t dim

    def __init__(self, REAL_t s, INDEX_t dim, REAL_t radius=1.0):
        function.__init__(self)
        from scipy.special import gamma
        self.s = s
        self.dim = dim
        self.radius2 = radius**2
        self.fac = self.radius2**s * 2.**(-2.*s)*gamma(dim/2.)/gamma((dim+2.*s)/2.)/gamma(1.+s)
        self.fac2 = log(0.25*self.radius2) - digamma(0.5*dim+s) - digamma(1+s)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t r2 = 0.
        cdef INDEX_t i
        for i in range(self.dim):
            r2 += x[i]**2
        if r2 <= self.radius2:
            return (self.fac2+log(1.-r2/self.radius2))*self.fac*pow(1.-r2/self.radius2, self.s)
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


cdef class simpleAnisotropy(function):
    cdef REAL_t epsilon

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        if x[0] < 0.5:
            return 1.0
        else:
            return self.epsilon


cdef class simpleAnisotropy2(function):
    cdef REAL_t epsilon

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        if (x[0] < 0.5) == (x[1] < 0.5):
            return 1.0
        else:
            return self.epsilon


cdef class inclusions(function):
    cdef REAL_t epsilon

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        if (x[0] % 0.4 > 0.2) and (x[1] % 0.4 > 0.2):
            return self.epsilon
        else:
            return 1.0


cdef class inclusionsHong(function):
    cdef REAL_t epsilon

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon/2.

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        if ((x[0]+1.+self.epsilon)**2+x[1]**2 < 1.) or ((x[0]-1.-self.epsilon)**2+x[1]**2 < 1.):
            return 0.1
        else:
            return 1.0


cdef inline REAL_t segmentRadius(REAL_t theta, REAL_t R, REAL_t theta1, REAL_t theta2, INDEX_t k):
    cdef:
        INDEX_t n
        REAL_t Rmid, thetamid
    n = <INDEX_t>(k*(theta-theta1)/(theta2-theta1))
    theta1, theta2 = theta1+n*(theta2-theta1)/k, theta1+(n+1)*(theta2-theta1)/k
    Rmid = R*cos((theta2-theta1)/2.0)
    thetamid = (theta1+theta2)/2.0
    return Rmid/cos(theta-thetamid)


cdef class motorPermeability(function):
    cdef:
        REAL_t thetaRotor, thetaRotor2, thetaStator, thetaCoil, epsilon
        REAL_t rRotorIn, rRotorOut, rStatorIn, rStatorOut, rCoilIn, rCoilOut
        INDEX_t nRotorIn, nRotorOut, nStatorIn, nStatorOut

    def __init__(self,
                 epsilon=1.0/5200.0,
                 thetaRotor=pi/12.0,
                 thetaCoil=pi/32.0,
                 rRotorIn=0.375,
                 rRotorOut=0.5,
                 rStatorIn=0.875,
                 rStatorOut=0.52,
                 rCoilIn=0.8,
                 rCoilOut=0.55,
                 nRotorOut=4,
                 nRotorIn=8,
                 nStatorOut=4,
                 nStatorIn=8):
        self.epsilon = epsilon
        self.thetaRotor = thetaRotor
        self.thetaCoil = thetaCoil
        self.rRotorIn = rRotorIn
        self.rRotorOut = rRotorOut
        self.rStatorIn = rStatorIn
        self.rStatorOut = rStatorOut
        self.rCoilIn = rCoilIn
        self.rCoilOut = rCoilOut
        # self.thetaRotor2 = np.arctan2(rRotorOut*sin(thetaRotor),sqrt(rRotorIn**2-rRotorOut**2*sin(thetaRotor)**2))
        # self.thetaStator = np.arctan2(rStatorOut*sin(thetaRotor),sqrt(rStatorIn**2-rStatorOut**2*sin(thetaRotor)**2))
        self.thetaRotor2 = atan2(rRotorOut*sin(thetaRotor), sqrt(rRotorIn**2-rRotorOut**2*sin(thetaRotor)**2))
        self.thetaStator = atan2(rStatorOut*sin(thetaRotor), sqrt(rStatorIn**2-rStatorOut**2*sin(thetaRotor)**2))
        self.nRotorIn = nRotorIn
        self.nRotorOut = nRotorOut
        self.nStatorIn = nStatorIn
        self.nStatorOut = nStatorOut

    cdef inline BOOL_t inRotor(self, REAL_t[::1] x):
        cdef:
            REAL_t r, theta, eps = 1e-6
            INDEX_t k
        r = sqrt(x[0]**2.0+x[1]**2.0)
        # theta = np.arctan2(x[1], x[0])
        theta = atan2(x[1], x[0])
        k = <INDEX_t>((theta+pi/4.0) // (pi/2.0))
        theta = abs(theta - (k * pi/2.0))
        if self.thetaRotor2 < theta:
            return r < segmentRadius(theta, self.rRotorIn, self.thetaRotor2, pi/2-self.thetaRotor2, self.nRotorIn)-eps
        if theta < self.thetaRotor:
            return r < segmentRadius(theta, self.rRotorOut, -self.thetaRotor, self.thetaRotor, self.nRotorOut)-eps
        y = r*sin(theta)
        return y < self.rRotorOut*sin(self.thetaRotor)-eps

    cdef inline BOOL_t inStator(self, REAL_t[::1] x):
        cdef:
            REAL_t r, theta, eps = 1e-6
            INDEX_t k

        r = sqrt(x[0]**2.0+x[1]**2.0)
        # theta = np.arctan2(x[1], x[0])
        theta = atan2(x[1], x[0])
        k = <INDEX_t>(theta // (pi/3.0))
        theta = abs(theta - pi/6.0 - k * pi/3.0)
        if theta > self.thetaRotor:
            return r > segmentRadius(theta, self.rStatorIn, self.thetaStator, pi/3.0-self.thetaStator, self.nStatorIn)+eps
        if theta < self.thetaStator:
            return r > segmentRadius(theta, self.rStatorOut, -self.thetaRotor, self.thetaRotor, self.nStatorOut)+eps

        y = r*sin(theta)
        if y < self.rStatorOut*sin(self.thetaRotor)-eps:
            return r > segmentRadius(theta, self.rStatorOut, -self.thetaRotor, self.thetaRotor, self.nStatorOut)+eps
        else:
            return r > segmentRadius(theta, self.rStatorIn, self.thetaStator, pi/3.0-self.thetaStator, self.nStatorIn)+eps

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        if self.inRotor(x):
            return self.epsilon
        if self.inStator(x):
            return self.epsilon
        return 1.0


cdef class _rhsMotor(function):
    cdef:
        REAL_t thetaRotor, thetaCoil
        REAL_t rRotorIn, rRotorOut, rStratorIn, rStratorOut, rCoilIn, rCoilOut

    def __init__(self,
                 thetaRotor=pi/12.0,
                 thetaCoil=pi/24.0,
                 rRotorIn=0.375,
                 rRotorOut=0.5,
                 rStratorIn=0.875,
                 rStratorOut=0.52,
                 rCoilIn=0.8,
                 rCoilOut=0.55):
        self.thetaRotor = thetaRotor
        self.thetaCoil = thetaCoil
        self.rRotorIn = rRotorIn
        self.rRotorOut = rRotorOut
        self.rStratorIn = rStratorIn
        self.rStratorOut = rStratorOut
        self.rCoilIn = rCoilIn
        self.rCoilOut = rCoilOut

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef REAL_t r, theta, s
        r = sqrt(x[0]**2.0+x[1]**2.0)
        # theta = atan2(x[1], x[0])
        if x[0] < 0:
            # theta = np.arctan2(x[1], -x[0])
            theta = atan2(x[1], -x[0])
            s = np.sign(theta)
            theta = abs(theta)
        else:
            # theta = np.arctan2(x[1], x[0])
            theta = atan2(x[1], x[0])
            s = -np.sign(theta)
            theta = abs(theta)
        if theta >= self.thetaCoil and theta <= self.thetaRotor:
            if r > self.rCoilOut and r < self.rCoilIn:
                # return 1.0*s
                return 0.
        elif theta >= 3.0*self.thetaRotor and theta <= 3.0*self.thetaRotor+self.thetaCoil:
            if r > self.rCoilOut and r < self.rCoilIn:
                # return s
                return 0.
        elif theta >= pi/2.0-self.thetaRotor-self.thetaCoil and theta <= pi/2.0-self.thetaRotor:
            if r > self.rCoilOut and r < self.rCoilIn:
                return 1.0*s
        return 0.


cdef class rhsMotor(function):
    cdef:
        list coilPairOn
        REAL_t dist1, dist2, rCoilOut, rCoilIn

    def __init__(self, coilPairOn=[0, 1, 2]):
        self.coilPairOn = coilPairOn
        self.dist1 = 0.16
        self.dist2 = 0.25
        self.rCoilIn = 0.8
        self.rCoilOut = 0.55

    cdef inline REAL_t eval(self, REAL_t[::1] z):
        cdef:
            REAL_t r, theta, x, y
            INDEX_t k

        r = sqrt(z[0]**2.0+z[1]**2.0)
        # theta = np.arctan2(z[1], z[0])
        theta = atan2(z[1], z[0])
        k = <INDEX_t>(theta // (pi/3.0))
        if k not in self.coilPairOn and k+3 not in self.coilPairOn:
            return 0.
        theta -= pi/6.0 + k * pi/3.0
        x, y = r*cos(theta), r*sin(theta)
        if self.dist1 < y < self.dist2 and self.rCoilOut < x < self.rCoilIn:
            return 1.0
        elif self.dist1 < -y < self.dist2 and self.rCoilOut < x < self.rCoilIn:
            return -1.0
        else:
            return 0.


cpdef function rhsHr(REAL_t r, INDEX_t dim, REAL_t scaling=1.):
    if r == 0.5:
        return constant(scaling)
    else:
        if dim == 1:
            return rhsHr1D(r, scaling)
        if dim == 2:
            return rhsHr2D(r, scaling)
        if dim == 3:
            return rhsHr3D(r, scaling)
        else:
            raise NotImplementedError()


cdef class rhsHr1D(function):
    cdef REAL_t beta, scaling

    def __init__(self, REAL_t r, REAL_t scaling=1.):
        self.beta = r-0.5
        self.scaling = scaling

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return self.scaling*pow(x[0]*(1.-x[0]), self.beta)


cdef class rhsHr2D(function):
    cdef REAL_t beta, scaling

    def __init__(self, REAL_t r, REAL_t scaling=1.):
        self.beta = r-0.5
        self.scaling = scaling

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return (self.scaling *
                pow(x[0]*(1.-x[0]), self.beta) *
                pow(x[1]*(1.-x[1]), self.beta))


cdef class rhsHr3D(function):
    cdef REAL_t beta, scaling

    def __init__(self, REAL_t r, REAL_t scaling=1.):
        self.beta = r-0.5
        self.scaling = scaling

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return (self.scaling *
                pow(x[0]*(1.-x[0]), self.beta) *
                pow(x[1]*(1.-x[1]), self.beta) *
                pow(x[2]*(1.-x[2]), self.beta))


cdef class rhsHr2Ddisk(function):
    cdef REAL_t beta, scaling

    def __init__(self, REAL_t r, REAL_t scaling=1.):
        self.beta = r-0.5
        self.scaling = scaling

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return (self.scaling *
                pow(min(1.-pow(x[0], 2)-pow(x[1], 2), 1.), self.beta))


cdef class logDiffusion1D(function):
    cdef:
        REAL_t[::1] c

    def __init__(self, REAL_t[::1] c):
        self.c = c

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t S
            INDEX_t i
        S = 0.
        for i in range(1, self.c.shape[0]+1):
            S += self.c[i-1]*sin(i*pi*x[0])
        return exp(S)


cdef class logDiffusion2D(function):
    cdef:
        REAL_t[:, ::1] c

    def __init__(self, REAL_t[:, ::1] c):
        self.c = c

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t S, sx
            INDEX_t i, j
        S = 0.
        for i in range(1, self.c.shape[0]+1):
            sx = sin(i*pi*x[0])
            for j in range(1, self.c.shape[1]+1):
                S += self.c[i-1, j-1] * sx*sin(j*pi*x[1])
        return exp(S)


cdef class fractalDiffusivity(function):
    cdef:
        REAL_t maxVal, offset

    def __init__(self, REAL_t maxVal, REAL_t offset):
        self.maxVal = maxVal
        self.offset = offset

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t a = self.maxVal
            INDEX_t i
        for i in range(x.shape[0]):
            if x[i] > 0.:
                a = min(2**(-floor(log2(x[i]+self.offset))), a)
        return a


cdef class expDiffusivity(function):
    cdef:
        REAL_t growth, frequency

    def __init__(self, REAL_t growth, REAL_t frequency):
        self.growth = growth
        self.frequency = frequency

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t r2 = 0.
            INDEX_t i
        for i in range(x.shape[0]):
            r2 += x[i]**2
        return exp(self.growth*r2)*(2+cos(self.frequency*r2))


######################################################################
# eigenfunctions for Laplacian on unit disc

cdef extern from "<math.h>" nogil:
    double jn(int n, double x)

from scipy.special import jn_zeros
jv = jn


cdef class eigfun_disc(function):
    cdef:
        INDEX_t k, l
        REAL_t a_lk, C
    def __init__(self, k, l):
        function.__init__(self)
        self.k = k
        self.l = l
        if l == 0:
            self.a_lk = jn_zeros(l, k+1)[k]
            self.C = 1.0/(sqrt(pi)*jv(l+1, self.a_lk))
        elif l > 0:
            self.a_lk = jn_zeros(l, k+1)[k]
            self.C = sqrt(2)/(sqrt(pi)*jv(l+1, self.a_lk))
        else:
            l = -l
            self.a_lk = jn_zeros(l, k+1)[k]
            self.C = sqrt(2)/(sqrt(pi)*jv(l+1, self.a_lk))

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef INDEX_t l
        if self.l == 0:
            return self.C * jv(self.l, self.a_lk*norm(x))
        elif self.l > 0:
            return self.C * jv(self.l, self.a_lk*norm(x)) * cos(self.l*atan2(x[1], x[0]))
        else:
            l = -self.l
            return self.C * jv(l, self.a_lk*norm(x)) * sin(l*atan2(x[1], x[0]))


cdef class eigfun_disc_deriv_x(function):
    cdef:
        INDEX_t k, l
        REAL_t a_lk, C
    def __init__(self, k, l):
        function.__init__(self)
        self.k = k
        self.l = l
        if l == 0:
            self.a_lk = jn_zeros(l, k+1)[k]
            self.C = 1.0/(sqrt(pi)*jv(l+1, self.a_lk)) * self.a_lk/2.
        elif l > 0:
            self.a_lk = jn_zeros(l, k+1)[k]
            self.C = sqrt(2)/(sqrt(pi)*jv(l+1, self.a_lk)) * self.a_lk/2.
        else:
            l = -l
            self.a_lk = jn_zeros(l, k+1)[k]
            self.C = sqrt(2)/(sqrt(pi)*jv(l+1, self.a_lk)) * self.a_lk/2.

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t l
            REAL_t theta, r, jm, jp
        theta = atan2(x[1], x[0])
        r = norm(x)
        if self.l == 0:
            jm = jv(self.l-1, self.a_lk*r)
            jp = jv(self.l+1, self.a_lk*r)
            return self.C * (jm-jp) * cos(theta)
        elif self.l > 0:
            jm = jv(self.l-1, self.a_lk*r)
            jp = jv(self.l+1, self.a_lk*r)
            return self.C * ((jm-jp) * cos(self.l*theta) * cos(theta) +
                             (jm+jp) * sin(self.l*theta) * sin(theta))
        else:
            l = -self.l
            jm = jv(l-1, self.a_lk*r)
            jp = jv(l+1, self.a_lk*r)
            return self.C * ((jm-jp) * sin(l*theta) * cos(theta) -
                             (jm+jp) * cos(l*theta) * sin(theta))


cdef class eigfun_disc_deriv_y(function):
    cdef:
        INDEX_t k, l
        REAL_t a_lk, C
    def __init__(self, k, l):
        function.__init__(self)
        self.k = k
        self.l = l
        if l == 0:
            self.a_lk = jn_zeros(l, k+1)[k]
            self.C = 1.0/(sqrt(pi)*jv(l+1, self.a_lk)) * self.a_lk/2.
        elif l > 0:
            self.a_lk = jn_zeros(l, k+1)[k]
            self.C = sqrt(2)/(sqrt(pi)*jv(l+1, self.a_lk)) * self.a_lk/2.
        else:
            l = -l
            self.a_lk = jn_zeros(l, k+1)[k]
            self.C = sqrt(2)/(sqrt(pi)*jv(l+1, self.a_lk)) * self.a_lk/2.

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t l
            REAL_t theta, r, jm, jp
        theta = atan2(x[1], x[0])
        r = norm(x)
        if self.l == 0:
            jm = jv(self.l-1, self.a_lk*r)
            jp = jv(self.l+1, self.a_lk*r)
            return self.C * (jm-jp) * cos(theta)
        elif self.l > 0:
            jm = jv(self.l-1, self.a_lk*r)
            jp = jv(self.l+1, self.a_lk*r)
            return self.C * ((jm-jp) * cos(self.l*theta) * sin(theta) -
                             (jm+jp) * sin(self.l*theta) * cos(theta))
        else:
            l = -self.l
            jm = jv(l-1, self.a_lk*r)
            jp = jv(l+1, self.a_lk*r)
            return self.C * ((jm-jp) * sin(l*theta) * sin(theta) -
                             (jm+jp) * cos(l*theta) * cos(theta))


cdef class radialIndicator(function):
    cdef:
        REAL_t radius
        BOOL_t centerIsOrigin
        REAL_t[::1] center

    def __init__(self, REAL_t radius, REAL_t[::1] center=None):
        self.radius = radius**2
        if center is None:
            self.centerIsOrigin = True
        else:
            self.centerIsOrigin = False
            self.center = center

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t r = 0.
            INDEX_t i
        if self.centerIsOrigin:
            for i in range(x.shape[0]):
                r += x[i]*x[i]
        else:
            for i in range(x.shape[0]):
                r += (x[i]-self.center[i])*(x[i]-self.center[i])
        return r <= self.radius

    def __eq__(self, other):
        cdef:
            radialIndicator o
            INDEX_t i
        if isinstance(other, radialIndicator):
            o = other
            if self.radius != o.radius:
                return False
            if self.centerIsOrigin != o.centerIsOrigin:
                return False
            for i in range(self.center.shape[0]):
                if not self.center[i] == o.center[i]:
                    return False
            return True
        else:
            return False



cdef class squareIndicator(function):
    cdef REAL_t[::1] a, b

    def __init__(self, REAL_t[::1] a, REAL_t[::1] b):
        self.a = a
        self.b = b

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t i
        for i in range(x.shape[0]):
            if (x[i] < self.a[i]) or (x[i] > self.b[i]):
                return False
        return True

    def __eq__(self, other):
        cdef:
            squareIndicator o
            INDEX_t i
        if isinstance(other, squareIndicator):
            o = other
            for i in range(self.a.shape[0]):
                if self.a[i] != o.a[i]:
                    return False
                if self.b[i] != o.b[i]:
                    return False
            return True
        else:
            return False


cdef class proj(function):
    cdef:
        function f
        REAL_t a, b
        function lower, upper
        BOOL_t lowerFun, upperFun

    def __init__(self, function f, tuple bounds):
        self.f = f
        self.lowerFun = isinstance(bounds[0], function)
        if self.lowerFun:
            self.lower = bounds[0]
        else:
            self.a = bounds[0]
        self.upperFun = isinstance(bounds[1], function)
        if self.upperFun:
            self.upper = bounds[1]
        else:
            self.b = bounds[1]

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t a, b
        if self.lowerFun:
            a = self.lower(x)
        else:
            a = self.a
        if self.upperFun:
            b = self.upper(x)
        else:
            b = self.b
        return max(a, min(b, self.f.eval(x)))


cdef class coordinate(function):
    cdef:
        INDEX_t i

    def __init__(self, INDEX_t i):
        self.i = i

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        return x[self.i]


cdef class indicatorFunctor(function):
    cdef:
        function indicator
        function f
        REAL_t threshold

    def __init__(self, function f, function indicator, REAL_t threshold=1e-9):
        self.f = f
        self.indicator = indicator
        self.threshold = threshold

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        if self.indicator(x) > self.threshold:
            return self.f(x)
        else:
            return 0.

    def __repr__(self):
        return '({} if {}>0)'.format(self.f, self.indicator)


cdef inline REAL_t evalPL1D(REAL_t[:, ::1] simplex, REAL_t[::1] uloc, REAL_t[::1] x):
    cdef:
        REAL_t l
    l = (x[0]-simplex[1, 0]) / (simplex[0, 0]-simplex[1, 0])
    return l*uloc[0] + (1.-l)*uloc[1]


cdef class lookupFunction1D(function):
    cdef:
        REAL_t[:, ::1] coords
        REAL_t[::1] vals
        list tree
        INDEX_t dim
        REAL_t[:, ::1] simplex
        REAL_t[::1] uloc

    def __init__(self, REAL_t[:, ::1] coords, REAL_t[::1] vals):
        self.coords = coords
        self.dim = coords.shape[1]
        self.vals = vals
        from scipy.spatial import cKDTree
        self.tree = [cKDTree(coords)]
        self.simplex = uninitialized((self.dim+1, self.dim), dtype=REAL)
        self.uloc = uninitialized((self.dim+1), dtype=REAL)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t i, j
            INDEX_t[::1] idx

        idx = self.tree[0].query(x, self.dim+1)[1].astype(INDEX)
        for i in range(self.dim+1):
            for j in range(self.dim):
                self.simplex[i, j] = self.coords[idx[i], j]
        for i in range(self.dim+1):
            self.uloc[i] = self.vals[idx[i]]
        return evalPL1D(self.simplex, self.uloc, x)


cdef class lookupFunctionTensor1DNew(function):
    cdef:
        REAL_t[:, ::1] coordsX
        REAL_t[::1] vals
        REAL_t[:, ::1] simplex
        REAL_t[::1] uloc
        INDEX_t N

    def __init__(self, REAL_t[:, ::1] coordsX, REAL_t[::1] vals, INDEX_t N):
        self.coordsX = coordsX
        self.vals = vals
        self.simplex = uninitialized((2, 1), dtype=REAL)
        self.uloc = uninitialized((2), dtype=REAL)
        self.N = N

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t i, idxX
        idxX = <INDEX_t>(self.N*x[0])
        for i in range(2):
            self.simplex[i, 0] = self.coordsX[idxX+i, 0]
        for i in range(2):
            self.uloc[i] = self.vals[idxX+i]
        return evalPL1D(self.simplex, self.uloc, x)


cdef inline REAL_t evalPLTensor2D(REAL_t[:, ::1] simplex, REAL_t[:, ::1] uloc, REAL_t[::1] x):
    cdef:
        REAL_t lX, lY
    lX = (x[0]-simplex[1, 0]) / (simplex[0, 0]-simplex[1, 0])
    lY = (x[1]-simplex[1, 1]) / (simplex[0, 1]-simplex[1, 1])
    return lX*lY*uloc[0, 0] + (1.-lX)*lY*uloc[1, 0] + lX*(1.-lY)*uloc[0, 1] + (1.-lX)*(1.-lY)*uloc[1, 1]


cdef inline REAL_t evalPLTensor3D(REAL_t[:, ::1] simplex, REAL_t[:, :, ::1] uloc, REAL_t[::1] x):
    cdef:
        REAL_t lX, lY, lZ
    lX = (x[0]-simplex[1, 0]) / (simplex[0, 0]-simplex[1, 0])
    lY = (x[1]-simplex[1, 1]) / (simplex[0, 1]-simplex[1, 1])
    lZ = (x[2]-simplex[1, 2]) / (simplex[0, 2]-simplex[1, 2])
    return (lX*lY*lZ*uloc[0, 0, 0] + (1.-lX)*lY*lZ*uloc[1, 0, 0] +
            lX*(1.-lY)*lZ*uloc[0, 1, 0] + (1.-lX)*(1.-lY)*lZ*uloc[1, 1, 0] +
            lX*lY*(1.-lZ)*uloc[0, 0, 1] + (1.-lX)*lY*(1.-lZ)*uloc[1, 0, 1] +
            lX*(1.-lY)*(1.-lZ)*uloc[0, 1, 1] + (1.-lX)*(1.-lY)*(1.-lZ)*uloc[1, 1, 1])


cdef class lookupFunctionTensor2D(function):
    cdef:
        REAL_t[:, ::1] coordsX, coordsY
        REAL_t[:, ::1] vals
        list trees
        REAL_t[:, ::1] simplex
        REAL_t[:, ::1] uloc
        REAL_t[::1] q

    def __init__(self, REAL_t[:, ::1] coordsX, REAL_t[:, ::1] coordsY, REAL_t[:, ::1] vals):
        self.coordsX = coordsX
        self.coordsY = coordsY
        self.vals = vals
        from scipy.spatial import cKDTree
        self.trees = [cKDTree(coordsX), cKDTree(coordsY)]
        self.simplex = uninitialized((2, 2), dtype=REAL)
        self.uloc = uninitialized((2, 2), dtype=REAL)
        self.q = uninitialized((1), dtype=REAL)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t i, j
            INDEX_t[::1] idxX, idxY
        self.q[0] = x[0]
        idxX = self.trees[0].query(self.q, 2)[1].astype(INDEX)
        self.q[0] = x[1]
        idxY = self.trees[1].query(self.q, 2)[1].astype(INDEX)
        for i in range(2):
            self.simplex[i, 0] = self.coordsX[idxX[i], 0]
            self.simplex[i, 1] = self.coordsY[idxY[i], 0]
        for i in range(2):
            for j in range(2):
                self.uloc[i, j] = self.vals[idxX[i], idxY[j]]
        return evalPLTensor2D(self.simplex, self.uloc, x)


cdef class lookupFunctionTensor2DNew(function):
    cdef:
        REAL_t[:, ::1] coordsX, coordsY
        REAL_t[:, ::1] vals
        REAL_t[:, ::1] simplex
        REAL_t[:, ::1] uloc
        INDEX_t N

    def __init__(self, REAL_t[:, ::1] coordsX, REAL_t[:, ::1] coordsY, REAL_t[:, ::1] vals, INDEX_t N):
        self.coordsX = coordsX
        self.coordsY = coordsY
        self.vals = vals
        self.simplex = uninitialized((2, 2), dtype=REAL)
        self.uloc = uninitialized((2, 2), dtype=REAL)
        self.N = N

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t i, j, idxX, idxY
        idxX = <INDEX_t>(self.N*x[0])
        idxY = <INDEX_t>(self.N*x[1])
        for i in range(2):
            self.simplex[i, 0] = self.coordsX[idxX+i, 0]
            self.simplex[i, 1] = self.coordsY[idxY+i, 0]
        for i in range(2):
            for j in range(2):
                self.uloc[i, j] = self.vals[idxX+i, idxY+j]
        return evalPLTensor2D(self.simplex, self.uloc, x)
        # assert np.isfinite(res), (np.array(x), idxX, idxY, np.array(self.simplex), np.array(self.uloc), self.N)
        # return res


cdef class lookupFunctionTensor2DNewSym(function):
    cdef:
        REAL_t[::1] coordsX, coordsY
        REAL_t[:, ::1] vals
        REAL_t[:, ::1] simplex
        REAL_t[:, ::1] uloc
        REAL_t[::1] xTemp
        INDEX_t N

    def __init__(self,
                 REAL_t[::1] coordsX,
                 REAL_t[::1] coordsY,
                 REAL_t[:, ::1] vals):
        self.coordsX = coordsX
        self.coordsY = coordsY
        self.vals = vals
        self.simplex = uninitialized((2, 2), dtype=REAL)
        self.uloc = uninitialized((2, 2), dtype=REAL)
        self.N = 2*vals.shape[0]-1
        assert self.coordsX.shape[0] == self.N
        assert self.coordsY.shape[0] == self.N
        self.xTemp = uninitialized((2), dtype=REAL)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t i, j, idxX, idxY, I, J

        for i in range(2):
            if x[i] > 0.5:
                self.xTemp[i] = 1.0-x[i]
            else:
                self.xTemp[i] = x[i]
        idxX = <INDEX_t>(self.N*self.xTemp[0])
        idxY = <INDEX_t>(self.N*self.xTemp[1])
        for i in range(2):
            self.simplex[i, 0] = self.coordsX[idxX+i]
            self.simplex[i, 1] = self.coordsY[idxY+i]
        if (idxX+1 < self.vals.shape[0]) and (idxY+1 < self.vals.shape[1]):
            for i in range(2):
                for j in range(2):
                    self.uloc[i, j] = self.vals[idxX+i, idxY+j]
        else:
            for i in range(2):
                if idxX+i >= self.vals.shape[0]:
                    I = self.vals.shape[0]-2
                else:
                    I = idxX+i
                for j in range(2):
                    if idxY+j >= self.vals.shape[1]:
                        J = self.vals.shape[1]-2
                    else:
                        J = idxY+j
                    self.uloc[i, j] = self.vals[I, J]
        return evalPLTensor2D(self.simplex, self.uloc, self.xTemp)


cdef class lookupFunctionTensor3D(function):
    cdef:
        REAL_t[:, ::1] coordsX, coordsY, coordsZ
        REAL_t[:, :, ::1] vals
        list trees
        REAL_t[:, ::1] simplex
        REAL_t[:, :, ::1] uloc
        REAL_t[::1] q

    def __init__(self,
                 REAL_t[:, ::1] coordsX,
                 REAL_t[:, ::1] coordsY,
                 REAL_t[:, ::1] coordsZ,
                 REAL_t[:, :, ::1] vals):
        self.coordsX = coordsX
        self.coordsY = coordsY
        self.coordsZ = coordsZ
        self.vals = vals
        from scipy.spatial import cKDTree
        self.trees = [cKDTree(coordsX), cKDTree(coordsY), cKDTree(coordsZ)]
        self.simplex = uninitialized((2, 3), dtype=REAL)
        self.uloc = uninitialized((2, 2, 2), dtype=REAL)
        self.q = uninitialized((1), dtype=REAL)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t i, j, k
            INDEX_t[::1] idxX, idxY, idxZ
        self.q[0] = x[0]
        idxX = self.trees[0].query(self.q, 2)[1].astype(INDEX)
        self.q[0] = x[1]
        idxY = self.trees[1].query(self.q, 2)[1].astype(INDEX)
        self.q[0] = x[2]
        idxZ = self.trees[2].query(self.q, 2)[1].astype(INDEX)
        for i in range(2):
            self.simplex[i, 0] = self.coordsX[idxX[i], 0]
            self.simplex[i, 1] = self.coordsY[idxY[i], 0]
            self.simplex[i, 2] = self.coordsZ[idxZ[i], 0]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    self.uloc[i, j, k] = self.vals[idxX[i], idxY[j], idxZ[k]]
        return evalPLTensor3D(self.simplex, self.uloc, x)


cdef class lookupFunctionTensor3DNew(function):
    cdef:
        REAL_t[:, ::1] coordsX, coordsY, coordsZ
        REAL_t[:, :, ::1] vals
        REAL_t[:, ::1] simplex
        REAL_t[:, :, ::1] uloc
        INDEX_t N

    def __init__(self,
                 REAL_t[:, ::1] coordsX,
                 REAL_t[:, ::1] coordsY,
                 REAL_t[:, ::1] coordsZ,
                 REAL_t[:, :, ::1] vals,
                 INDEX_t N):
        self.coordsX = coordsX
        self.coordsY = coordsY
        self.coordsZ = coordsZ
        self.vals = vals
        self.simplex = uninitialized((2, 3), dtype=REAL)
        self.uloc = uninitialized((2, 2, 2), dtype=REAL)
        self.N = N

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t i, j, k, idxX, idxY, idxZ
        idxX = <INDEX_t>(self.N*x[0])
        idxY = <INDEX_t>(self.N*x[1])
        idxZ = <INDEX_t>(self.N*x[2])
        for i in range(2):
            self.simplex[i, 0] = self.coordsX[idxX+i, 0]
            self.simplex[i, 1] = self.coordsY[idxY+i, 0]
            self.simplex[i, 2] = self.coordsZ[idxZ+i, 0]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    self.uloc[i, j, k] = self.vals[idxX+i, idxY+j, idxZ+k]
        return evalPLTensor3D(self.simplex, self.uloc, x)


cdef class lookupFunctionTensor3DNewSym(function):
    cdef:
        REAL_t[::1] coordsX, coordsY, coordsZ
        REAL_t[:, :, ::1] vals
        REAL_t[:, ::1] simplex
        REAL_t[:, :, ::1] uloc
        REAL_t[::1] xTemp
        INDEX_t N

    def __init__(self,
                 REAL_t[::1] coordsX,
                 REAL_t[::1] coordsY,
                 REAL_t[::1] coordsZ,
                 REAL_t[:, :, ::1] vals):
        self.coordsX = coordsX
        self.coordsY = coordsY
        self.coordsZ = coordsZ
        self.vals = vals
        self.simplex = uninitialized((2, 3), dtype=REAL)
        self.uloc = uninitialized((2, 2, 2), dtype=REAL)
        self.N = 2*vals.shape[0]-1
        assert self.coordsX.shape[0] == self.N
        assert self.coordsY.shape[0] == self.N
        assert self.coordsZ.shape[0] == self.N
        self.xTemp = uninitialized((3), dtype=REAL)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t i, j, k, idxX, idxY, idxZ, I, J, K

        for i in range(3):
            if x[i] > 0.5:
                self.xTemp[i] = 1.0-x[i]
            else:
                self.xTemp[i] = x[i]
        idxX = <INDEX_t>(self.N*self.xTemp[0])
        idxY = <INDEX_t>(self.N*self.xTemp[1])
        idxZ = <INDEX_t>(self.N*self.xTemp[2])
        for i in range(2):
            self.simplex[i, 0] = self.coordsX[idxX+i]
            self.simplex[i, 1] = self.coordsY[idxY+i]
            self.simplex[i, 2] = self.coordsZ[idxZ+i]
        if (idxX+1 < self.vals.shape[0]) and (idxY+1 < self.vals.shape[1]) and (idxZ+1 < self.vals.shape[2]):
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        self.uloc[i, j, k] = self.vals[idxX+i, idxY+j, idxZ+k]
        else:
            for i in range(2):
                if idxX+i >= self.vals.shape[0]:
                    I = self.vals.shape[0]-2
                else:
                    I = idxX+i
                for j in range(2):
                    if idxY+j >= self.vals.shape[1]:
                        J = self.vals.shape[1]-2
                    else:
                        J = idxY+j
                    for k in range(2):
                        if idxZ+k >= self.vals.shape[2]:
                            K = self.vals.shape[2]-2
                        else:
                            K = idxZ+k
                        self.uloc[i, j, k] = self.vals[I, J, K]
        return evalPLTensor3D(self.simplex, self.uloc, self.xTemp)


cdef class sphericalIntegral(function):
    cdef:
        sphericalQuadRule qr
        function f
        REAL_t[::1] y
        INDEX_t dim

    def __init__(self, function f, INDEX_t dim, REAL_t radius, INDEX_t numQuadNodes):
        self.f = f
        self.dim = dim
        if self.dim == 1:
            self.qr = sphericalQuadRule1D(radius)
        elif self.dim == 2:
            self.qr = sphericalQuadRule2D(radius, numQuadNodes)
        else:
            raise NotImplementedError()
        self.y = uninitialized((self.dim), dtype=REAL)

    cdef REAL_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t I = 0.
            INDEX_t k, j
        for k in range(self.qr.num_nodes):
            for j in range(self.dim):
                self.y[j] = x[j]+self.qr.vertexOffsets[k, j]
            I += self.qr.weights[k]*self.f.eval(self.y)
        return I


cdef class complexFunction:
    def __call__(self, REAL_t[::1] x):
        return self.eval(x)

    cdef COMPLEX_t eval(self, REAL_t[::1] x):
        pass

    def __add__(self, complexFunction other):
        if isinstance(self, complexMulFunction):
            if isinstance(other, complexMulFunction):
                return complexSumFunction(self.f, self.fac, other.f, other.fac)
            else:
                return complexSumFunction(self.f, self.fac, other, 1.)
        else:
            if isinstance(other, complexMulFunction):
                return complexSumFunction(self, 1., other.f, other.fac)
            else:
                return complexSumFunction(self, 1., other, 1.)

    def __sub__(self, complexFunction other):
        if isinstance(self, complexMulFunction):
            if isinstance(other, complexMulFunction):
                return complexSumFunction(self.f, self.fac, other.f, -other.fac)
            else:
                return complexSumFunction(self.f, self.fac, other, -1.)
        else:
            if isinstance(other, complexMulFunction):
                return complexSumFunction(self, 1., other.f, -other.fac)
            else:
                return complexSumFunction(self, 1., other, -1.)

    def __mul__(first, second):
        if isinstance(first, complexFunction):
            return complexMulFunction(first, second)
        elif isinstance(second, complexFunction):
            return complexMulFunction(second, first)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        if isinstance(self, complexMulFunction):
            return complexMulFunction(self.f, -self.fac)
        elif isinstance(self, complexSumFunction):
            return complexSumFunction(self.f1, -self.fac1, self.f2, -self.fac2)
        else:
            return complexMulFunction(self, -1.0)

    def __repr__(self):
        return '%s' % (self.__class__.__name__)


cdef class complexSumFunction(complexFunction):
    cdef:
        public complexFunction f1, f2
        public COMPLEX_t fac1, fac2

    def __init__(self, complexFunction f1, COMPLEX_t fac1, complexFunction f2, COMPLEX_t fac2):
        self.f1 = f1
        self.fac1 = fac1
        self.f2 = f2
        self.fac2 = fac2

    cdef COMPLEX_t eval(self, REAL_t[::1] x):
        return self.fac1*self.f1.eval(x)+self.fac2*self.f2.eval(x)


cdef class complexMulFunction(complexFunction):
    cdef:
        public complexFunction f
        public COMPLEX_t fac

    def __init__(self, complexFunction f, COMPLEX_t fac):
        self.f = f
        self.fac = fac

    cdef COMPLEX_t eval(self, REAL_t[::1] x):
        return self.fac*self.f.eval(x)


cdef class wrapRealToComplexFunction(complexFunction):
    cdef:
        function fun

    def __init__(self, function fun):
        self.fun = fun

    cdef COMPLEX_t eval(self, REAL_t[::1] x):
        return self.fun(x)


cdef class complexLambda(complexFunction):
    cdef:
        object fun

    def __init__(self, fun):
        self.fun = fun

    cdef inline COMPLEX_t eval(self, REAL_t[::1] x):
        return self.fun(x)


cdef class real(function):
    cdef:
        complexFunction fun

    def __init__(self, complexFunction fun):
        self.fun = fun

    cdef REAL_t eval(self, REAL_t[::1] x):
        return self.fun.eval(x).real


cdef class imag(function):
    cdef:
        complexFunction fun

    def __init__(self, complexFunction fun):
        self.fun = fun

    cdef REAL_t eval(self, REAL_t[::1] x):
        return self.fun.eval(x).imag



cdef class waveFunction(complexFunction):
    cdef:
        REAL_t[::1] waveVector

    def __init__(self, REAL_t[::1] waveVector):
        self.waveVector = waveVector

    cdef COMPLEX_t eval(self, REAL_t[::1] x):
        cdef:
            REAL_t s = 0.
            INDEX_t i
        for i in range(x.shape[0]):
            s = s+self.waveVector[i]*x[i]
        return cos(s)+1j*sin(s)


cdef class vectorFunction:
    def __init__(self, INDEX_t numComponents):
        self.rows = numComponents

    def __call__(self, REAL_t[::1] x):
        vals = uninitialized((self.rows), dtype=REAL)
        self.eval(x, vals)
        return vals

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] vals):
        raise NotImplementedError()

    def norm(self):
        return vectorNorm(self)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

    def __add__(self, vectorFunction other):
        if isinstance(self, mulVectorFunction):
            if isinstance(other, mulVectorFunction):
                return sumVectorFunction(self.f, self.fac, other.f, other.fac)
            else:
                return sumVectorFunction(self.f, self.fac, other, 1.)
        else:
            if isinstance(other, mulVectorFunction):
                return sumVectorFunction(self, 1., other.f, other.fac)
            else:
                return sumVectorFunction(self, 1., other, 1.)

    def __sub__(self, vectorFunction other):
        if isinstance(self, mulVectorFunction):
            if isinstance(other, mulVectorFunction):
                return sumVectorFunction(self.f, self.fac, other.f, -other.fac)
            else:
                return sumVectorFunction(self.f, self.fac, other, -1.)
        else:
            if isinstance(other, mulVectorFunction):
                return sumVectorFunction(self, 1., other.f, -other.fac)
            else:
                return sumVectorFunction(self, 1., other, -1.)

    def __mul__(first, second):
        if isinstance(first, vectorFunction) and isinstance(second, vectorFunction):
            return NotImplementedError()
        elif isinstance(first, vectorFunction) and isinstance(second, function):
            return mulVectorFunction(first, 1.0, second)
        elif isinstance(first, function) and isinstance(second, vectorFunction):
            return mulVectorFunction(second, 1.0, first)
        elif isinstance(first, vectorFunction):
            return mulVectorFunction(first, second)
        elif isinstance(second, vectorFunction):
            return mulVectorFunction(second, first)
        else:
            return NotImplemented


cdef class componentVectorFunction(vectorFunction):
    def __init__(self, list components):
        super(componentVectorFunction, self).__init__(len(components))
        self.components = components

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] vals):
        cdef:
            INDEX_t i
            function f
        for i in range(self.rows):
            f = self.components[i]
            vals[i] = f.eval(x)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ','.join([f.__repr__() for f in self.components]))

    def __getitem__(self, i):
        return self.components[i]


cdef class vectorNorm(function):
    cdef:
        vectorFunction vecFun
        REAL_t[::1] vals

    def __init__(self, vectorFunction vecFun):
        self.vecFun = vecFun
        self.vals = uninitialized((self.vecFun.rows), dtype=REAL)

    cdef REAL_t eval(self, REAL_t[::1] x):
        self.vecFun.eval(x, self.vals)
        return norm(self.vals)


cdef class sumVectorFunction(vectorFunction):
    cdef:
        public vectorFunction f1, f2
        public REAL_t fac1, fac2
        REAL_t[::1] temp

    def __init__(self, vectorFunction f1, REAL_t fac1, vectorFunction f2, REAL_t fac2):
        assert f1.rows == f2.rows
        super(sumVectorFunction, self).__init__(f1.rows)
        self.f1 = f1
        self.fac1 = fac1
        self.f2 = f2
        self.fac2 = fac2
        self.temp = uninitialized((self.f2.rows), dtype=REAL)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] vals):
        self.f1.eval(x, vals)
        self.f2.eval(x, self.temp)
        for i in range(self.rows):
            vals[i] = self.fac1*vals[i]+self.fac2*self.temp[i]

    def __repr__(self):
        return '{}*{}+{}*{}'.format(self.fac1, self.f1, self.fac2, self.f2)


cdef class mulVectorFunction(vectorFunction):
    cdef:
        public vectorFunction f
        public function g
        public REAL_t fac

    def __init__(self, vectorFunction f, REAL_t fac, function g=None):
        super(mulVectorFunction, self).__init__(f.rows)
        self.f = f
        self.g = g
        self.fac = fac

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] vals):
        cdef:
            INDEX_t i
            REAL_t v = 1.
        self.f.eval(x, vals)
        if self.g is not None:
            v = self.g.eval(x)
        for i in range(self.rows):
            vals[i] *= self.fac*v

    def __repr__(self):
        if self.g is None:
            return '{}*{}'.format(self.fac, self.f)
        elif abs(self.fac-1.) < 1e-10:
            return '{}*{}'.format(self.g, self.f)
        else:
            return '{}*{}*{}'.format(self.fac, self.g, self.f)


cdef class matrixFunction:
    def __init__(self, list components, BOOL_t symmetric):
        self.rows = len(components)
        self.columns = len(components[0])
        self.components = components
        self.symmetric = symmetric

    def __call__(self, REAL_t[::1] x):
        vals = uninitialized((self.rows, self.columns), dtype=REAL)
        self.eval(x, vals)
        return vals

    cdef void eval(self, REAL_t[::1] x, REAL_t[:, ::1] vals):
        cdef:
            INDEX_t i
            function f
        if self.symmetric:
            for i in range(self.rows):
                for j in range(i, self.columns):
                    f = self.components[i][j]
                    vals[j, i] = vals[i, j] = f.eval(x)
        else:
            for i in range(self.rows):
                for j in range(self.columns):
                    f = self.components[i][j]
                    vals[i, j] = f.eval(x)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ','.join([f.__repr__() for f in self.components]))

    def __getitem__(self, I):
        i, j = I
        return self.components[i][j]


cdef class periodicityFunctor(function):
    cdef:
        function f
        REAL_t[::1] period
        REAL_t[::1] y

    def __init__(self, function f, REAL_t[::1] period):
        self.f = f
        self.period = period
        self.y = uninitialized((period.shape[0]), dtype=REAL)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t i
        for i in range(self.period.shape[0]):
            self.y[i] = x[i] % self.period[i]
        return self.f(self.y)

    def __repr__(self):
        return '({} with period {})'.format(self.f, np.array(self.period))


cdef class shiftScaleFunctor(function):
    cdef:
        function f
        REAL_t[::1] scaling
        REAL_t[::1] shift, temp

    def __init__(self, function f, REAL_t[::1] shift, REAL_t[::1] scaling):
        self.f = f
        self.shift = shift
        self.scaling = scaling
        self.temp = uninitialized((shift.shape[0]), dtype=REAL)

    cdef inline REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t i
        for i in range(self.shift.shape[0]):
            self.temp[i] = self.scaling[i]*x[i]+self.shift[i]
        return self.f(self.temp)

    def __repr__(self):
        return '{}({}*x+{})'.format(self.f, np.array(self.scaling), np.array(self.shift))
