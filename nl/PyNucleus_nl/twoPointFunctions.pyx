###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

"""
Defines the base class for functions of two spatial variables, e.g. kernels, fractional orders and normalizations.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, atan
from PyNucleus_base.myTypes import INDEX, REAL, COMPLEX, ENCODE, BOOL
from PyNucleus_base.blas import uninitialized


cdef enum fixed_type:
    FIXED_X
    FIXED_Y
    DIAGONAL

include "twoPointFunctions_REAL.pxi"
include "twoPointFunctions_COMPLEX.pxi"


cdef class lambdaTwoPoint(twoPointFunction):
    cdef:
        object fun

    def __init__(self, fun, BOOL_t symmetric):
        super(lambdaTwoPoint, self).__init__(symmetric, 1)
        self.fun = fun

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        value[0] = self.fun(x, y)

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t[::1] xA =<REAL_t[:dim]> x
            REAL_t[::1] yA =<REAL_t[:dim]> y
        value[0] = self.fun(xA, yA)

    def __repr__(self):
        return 'Lambda({})'.format(self.fun)

    def __reduce__(self):
        return lambdaTwoPoint, (self.fun, self.symmetric)


cdef class matrixTwoPoint(twoPointFunction):
    def __init__(self, REAL_t[:, ::1] mat):
        self.mat = mat
        assert mat.shape[0] == mat.shape[1]
        symmetric = True
        for i in range(mat.shape[0]):
            for j in range(i, mat.shape[0]):
                if abs(mat[i, j]-mat[j, i]) > 1e-12:
                    symmetric = False
        super(matrixTwoPoint, self).__init__(symmetric, 1)
        self.n = np.zeros((mat.shape[0]), dtype=REAL)

    def __reduce__(self):
        return matrixTwoPoint, (self.mat)

    def __repr__(self):
        return '{}({},sym={})'.format(self.__class__.__name__, np.array(self.mat), self.symmetric)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            INDEX_t dim = x.shape[0]
            INDEX_t i, j
            REAL_t d = 0.
        for i in range(dim):
            self.n[i] = x[i] - y[i]
            d += self.n[i]**2
        d = sqrt(d)
        for i in range(dim):
            self.n[i] /= d
        d = 0.
        for i in range(dim):
            for j in range(dim):
                d += self.n[i]*self.mat[i, j]*self.n[j]
        value[0] = d

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            INDEX_t i, j
            REAL_t d = 0.
        for i in range(dim):
            self.n[i] = x[i] - y[i]
            d += self.n[i]**2
        if d > 0:
            d = sqrt(d)
            for i in range(dim):
                self.n[i] /= d
            d = 0.
            for i in range(dim):
                for j in range(dim):
                    d += self.n[i]*self.mat[i, j]*self.n[j]
            value[0] = d
            return
        value[0] = 1.


cdef class leftRightTwoPoint(twoPointFunction):
    def __init__(self, REAL_t ll, REAL_t rr, REAL_t lr=np.nan, REAL_t rl=np.nan, REAL_t interface=0.):
        if not np.isfinite(lr):
            lr = 0.5*(ll+rr)
        if not np.isfinite(rl):
            rl = 0.5*(ll+rr)
        super(leftRightTwoPoint, self).__init__(rl == lr, 1)
        self.ll = ll
        self.lr = lr
        self.rl = rl
        self.rr = rr
        self.interface = interface

    def __reduce__(self):
        return leftRightTwoPoint, (self.ll, self.rr, self.lr, self.rl, self.interface)

    def __repr__(self):
        return '{}(ll={},rr={},lr={},rl={},interface={},sym={})'.format(self.__class__.__name__, self.ll, self.rr, self.lr, self.rl, self.interface, self.symmetric)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        if x[0] < self.interface:
            if y[0] < self.interface:
                value[0] = self.ll
            else:
                value[0] = self.lr
        else:
            if y[0] < self.interface:
                value[0] = self.rl
            else:
                value[0] = self.rr

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        if x[0] < self.interface:
            if y[0] < self.interface:
                value[0] = self.ll
            else:
                value[0] = self.lr
        else:
            if y[0] < self.interface:
                value[0] = self.rl
            else:
                value[0] = self.rr


cdef class interfaceTwoPoint(twoPointFunction):
    def __init__(self, REAL_t horizon1, REAL_t horizon2, BOOL_t left, REAL_t interface=0.):
        super(interfaceTwoPoint, self).__init__(True, 1)
        self.horizon1 = horizon1
        self.horizon2 = horizon2
        self.left = left
        self.interface = interface

    def __reduce__(self):
        return interfaceTwoPoint, (self.horizon1, self.horizon2, self.left, self.interface)

    def __repr__(self):
        return '{}(horizon1={},horizon2={},left={},interface={})'.format(self.__class__.__name__, self.horizon1, self.horizon2, self.left, self.interface)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        self.evalPtr(x.shape[0], &x[0], &y[0], &value[0])

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        if dim == 1:
            if self.left:
                if ((x[0] <= self.interface) and (y[0] <= self.interface)):
                    value[0] = 1.
                elif ((x[0] > self.interface) and (y[0] > self.interface)):
                    value[0] = 0.
                elif ((x[0] <= self.interface-self.horizon2) and (y[0] > self.interface)):
                    value[0] = 1.
                elif ((x[0] > self.interface) and (y[0] <= self.interface-self.horizon2)):
                    value[0] = 1.
                else:
                    value[0] = 0.5
            else:
                if ((x[0] >= self.interface) and (y[0] >= self.interface)):
                    value[0] = 1.
                elif ((x[0] < self.interface) and (y[0] < self.interface)):
                    value[0] = 0.
                elif ((x[0] >= self.interface+self.horizon1) and (y[0] < self.interface)):
                    value[0] = 1.
                elif ((x[0] < self.interface) and (y[0] >= self.interface+self.horizon1)):
                    value[0] = 1.
                else:
                    value[0] = 0.5
        elif dim == 2:
            if self.left:
                if (x[0] <= self.interface) and ((x[1] > 0.) and (x[1] < 1.)):
                    if (y[0] <= self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        value[0] = 1.
                    elif (y[0] > self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        if x[0] <= self.interface-self.horizon2:
                            value[0] = 1.
                        else:
                            value[0] = 0.5
                    else:
                        value[0] = 1.
                elif (x[0] > self.interface) and ((x[1] > 0.) and (x[1] < 1.)):
                    if (y[0] <= self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        value[0] = 0.5
                    elif (y[0] > self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        value[0] = 0.
                    else:
                        value[0] = 0.
                else:
                    if (y[0] <= self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        value[0] = 1.
                    else:
                        value[0] = 0.
            else:
                if (x[0] >= self.interface) and ((x[1] > 0.) and (x[1] < 1.)):
                    if (y[0] >= self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        value[0] = 1.
                    elif (y[0] < self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        if x[0] >= self.interface+self.horizon1:
                            value[0] = 1.
                        else:
                            value[0] = 0.5
                    else:
                        value[0] = 1.
                elif (x[0] < self.interface) and ((x[1] > 0.) and (x[1] < 1.)):
                    if (y[0] >= self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        if y[0] <= self.interface+self.horizon1:
                            value[0] = 0.5
                        else:
                            value[0] = 1.
                    elif (y[0] < self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        value[0] = 0.
                    else:
                        value[0] = 0.
                else:
                    if (y[0] >= self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        value[0] = 1.
                    else:
                        value[0] = 0.


cdef class temperedTwoPoint(twoPointFunction):
    def __init__(self, REAL_t lambdaCoeff, INDEX_t dim):
        super(temperedTwoPoint, self).__init__(True, 1)
        self.lambdaCoeff = lambdaCoeff
        self.dim = dim

    def __reduce__(self):
        return temperedTwoPoint, (self.lambdaCoeff, self.dim)

    def __repr__(self):
        return '{}(lambda={})'.format(self.__class__.__name__, self.lambdaCoeff)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            INDEX_t i
            REAL_t r = 0.
        for i in range(self.dim):
            r += (x[i]-y[i])*(x[i]-y[i])
        value[0] = exp(-self.lambdaCoeff*sqrt(r))

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            INDEX_t i
            REAL_t r = 0.
        for i in range(dim):
            r += (x[i]-y[i])*(x[i]-y[i])
        value[0] = exp(-self.lambdaCoeff*sqrt(r))


cdef class tensorTwoPoint(twoPointFunction):
    cdef:
        INDEX_t i, j, dim

    def __init__(self, INDEX_t i, INDEX_t j, INDEX_t dim):
        super(tensorTwoPoint, self).__init__(True, 1)
        self.dim = dim
        self.i = i
        self.j = j

    def __reduce__(self):
        return tensorTwoPoint, (self.i, self.j, self.dim)

    def __repr__(self):
        return '{}(i={},j={})'.format(self.__class__.__name__, self.i, self.j)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            INDEX_t i
            REAL_t n2 = 0., ExE
        for i in range(self.dim):
            n2 += (x[i]-y[i])*(x[i]-y[i])
        if n2 > 0:
            ExE = (x[self.i]-y[self.i])*(x[self.j]-y[self.j])/n2
        else:
            ExE = 1.
        value[0] = ExE

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            INDEX_t i
            REAL_t n2 = 0., ExE
        for i in range(self.dim):
            n2 += (x[i]-y[i])*(x[i]-y[i])
        if n2 > 0:
            ExE = (x[self.i]-y[self.i])*(x[self.j]-y[self.j])/n2
        else:
            ExE = 1.
        value[0] = ExE


cdef class smoothedLeftRightTwoPoint(twoPointFunction):
    def __init__(self, REAL_t vl, REAL_t vr, REAL_t r=0.1, REAL_t slope=200.):
        super(smoothedLeftRightTwoPoint, self).__init__(False, 1)
        self.vl = vl
        self.vr = vr
        self.r = r
        self.slope = slope
        self.fac = 1./atan(r*slope)

    def __reduce__(self):
        return smoothedLeftRightTwoPoint, (self.vl, self.vr, self.r, self.slope)

    def __repr__(self):
        return '{}(vl={},vr={},r={},slope={})'.format(self.__class__.__name__, self.vl, self.vr, self.r, self.slope)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        if x[0] < -self.r:
            value[0] = self.vl
        elif x[0] > self.r:
            value[0] = self.vr
        value[0] = 0.5*(self.vl+self.vr)+0.5*(self.vr-self.vl)*atan(x[0]*self.slope) * self.fac

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        if x[0] < -self.r:
            value[0] = self.vl
        elif x[0] > self.r:
            value[0] = self.vr
        value[0] = 0.5*(self.vl+self.vr)+0.5*(self.vr-self.vl)*atan(x[0]*self.slope) * self.fac


cdef class unsymTwoPoint(twoPointFunction):
    def __init__(self, REAL_t l, REAL_t r):
        super(unsymTwoPoint, self).__init__(l == r, 1)
        self.l = l
        self.r = r

    def __reduce__(self):
        return unsymTwoPoint, (self.l, self.r)

    def __repr__(self):
        return '{}(l={},r={})'.format(self.__class__.__name__, self.l, self.r)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        if x[0] < y[0]:
            value[0] = self.l
        else:
            value[0] = self.r

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        if x[0] < y[0]:
            value[0] = self.l
        else:
            value[0] = self.r


cdef class inverseTwoPoint(twoPointFunction):
    def __init__(self, twoPointFunction f):
        super(inverseTwoPoint, self).__init__(f.symmetric, 1)
        self.f = f

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        self.f.eval(x, y, value)
        value[0] = 1./value[0]

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        self.f.evalPtr(dim, x, y, value)
        value[0] = 1./value[0]

    def __repr__(self):
        return '1/{}'.format(self.f)

    def __reduce__(self):
        return inverseTwoPoint, (self.f, )
