###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, exp, atan
from PyNucleus_base.myTypes import INDEX, REAL, ENCODE, BOOL


cdef class fixedTwoPointFunction(function):
    cdef:
        twoPointFunction f
        REAL_t[::1] point
        BOOL_t fixX

    def __init__(self, twoPointFunction f, REAL_t[::1] point, BOOL_t fixX):
        self.f = f
        self.point = point
        self.fixX = fixX

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x):
        if self.fixX:
            return self.f(self.point, x)
        else:
            return self.f(x, self.point)


cdef class twoPointFunction:
    def __init__(self, BOOL_t symmetric):
        self.symmetric = symmetric

    def __call__(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.eval(x, y)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        pass

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        pass

    def __getstate__(self):
        return self.symmetric

    def __setstate__(self, state):
        twoPointFunction.__init__(self, state)

    def fixedX(self, REAL_t[::1] x):
        return fixedTwoPointFunction(self, x, True)

    def fixedY(self, REAL_t[::1] y):
        return fixedTwoPointFunction(self, y, False)

    def plot(self, mesh, **kwargs):
        cdef:
            INDEX_t i, j
            REAL_t[:, ::1] S
            REAL_t[::1] x, y
        import matplotlib.pyplot as plt
        c = np.array(mesh.getCellCenters())
        if mesh.dim == 1:
            X, Y = np.meshgrid(c[:, 0], c[:, 0])
            x = np.empty((mesh.dim), dtype=REAL)
            y = np.empty((mesh.dim), dtype=REAL)
            S = np.zeros((mesh.num_cells, mesh.num_cells))
            for i in range(mesh.num_cells):
                for j in range(mesh.num_cells):
                    x[0] = X[i, j]
                    y[0] = Y[i, j]
                    S[i, j] = self.eval(x, y)
            plt.pcolormesh(X, Y, S, **kwargs)
            plt.colorbar()
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
        elif mesh.dim == 2:
            S = np.zeros(mesh.num_cells)
            for i in range(mesh.num_cells):
                S[i] = self(c[i, :], c[i, :])
            mesh.plotFunction(S, flat=True)
        else:
            raise NotImplementedError()

    def __mul__(self, twoPointFunction other):
        if isinstance(self, constantTwoPoint) and isinstance(other, constantTwoPoint):
            return constantTwoPoint(self.value*other.value)
        elif isinstance(self, parametrizedTwoPointFunction) or isinstance(other, parametrizedTwoPointFunction):
            return productParametrizedTwoPoint(self, other)
        else:
            return productTwoPoint(self, other)


cdef class lambdaTwoPoint(twoPointFunction):
    cdef:
        object fun

    def __init__(self, fun, BOOL_t symmetric):
        super(lambdaTwoPoint, self).__init__(symmetric)
        self.fun = fun

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.fun(x, y)

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t[::1] xA =<REAL_t[:dim]> x
            REAL_t[::1] yA =<REAL_t[:dim]> y
        return self.fun(xA, yA)

    def __repr__(self):
        return 'Lambda({})'.format(self.fun)

    def __getstate__(self):
        return (self.fun, self.symmetric)

    def __setstate__(self, state):
        lambdaTwoPoint.__init__(self, state[0], state[1])


cdef class productTwoPoint(twoPointFunction):
    def __init__(self, twoPointFunction f1, twoPointFunction f2):
        super(productTwoPoint, self).__init__(f1.symmetric and f2.symmetric)
        self.f1 = f1
        self.f2 = f2

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.f1.eval(x, y)*self.f2.eval(x, y)

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        return self.f1.evalPtr(dim, x, y)*self.f2.evalPtr(dim, x, y)

    def __repr__(self):
        return '{}*{}'.format(self.f1, self.f2)

    def __getstate__(self):
        return self.f1, self.f2

    def __setstate__(self, state):
        productTwoPoint.__init__(self, state[0], state[1])


cdef class constantTwoPoint(twoPointFunction):
    def __init__(self, REAL_t value):
        super(constantTwoPoint, self).__init__(True)
        self.value = value

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.value

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        return self.value

    def __repr__(self):
        return '{}'.format(self.value)

    def __getstate__(self):
        return self.value

    def __setstate__(self, state):
        constantTwoPoint.__init__(self, state)


cdef class matrixTwoPoint(twoPointFunction):
    def __init__(self, REAL_t[:, ::1] mat):
        self.mat = mat
        assert mat.shape[0] == mat.shape[1]
        symmetric = True
        for i in range(mat.shape[0]):
            for j in range(i, mat.shape[0]):
                if abs(mat[i, j]-mat[j, i]) > 1e-12:
                    symmetric = False
        super(matrixTwoPoint, self).__init__(symmetric)
        self.n = np.zeros((mat.shape[0]), dtype=REAL)

    def __getstate__(self):
        return (self.mat)

    def __setstate__(self, state):
        matrixTwoPoint.__init__(self, state)

    def __repr__(self):
        return '{}({},sym={})'.format(self.__class__.__name__, np.array(self.mat), self.symmetric)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
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
        return d

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
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
            return d
        return 1.


cdef class tensorTwoPoint(twoPointFunction):
    cdef:
        function f1, f2

    def __init__(self, function f1, function f2=None):
        self.f1 = f1
        if f2 is not None:
            self.f2 = f2
            super(tensorTwoPoint, self).__init__(False)
        else:
            self.f2 = f1
            super(tensorTwoPoint, self).__init__(True)

    def __getstate__(self):
        if self.symmetric:
            return self.f1
        else:
            return (self.f1, self.f2)

    def __setstate__(self, state):
        if isinstance(state, tuple):
            tensorTwoPoint.__init__(self, state[0], state[1])
        else:
            tensorTwoPoint.__init__(self, state)

    def __repr__(self):
        return '{}({},{},sym={})'.format(self.__class__.__name__, self.f1, self.f2, self.symmetric)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.f1(x)*self.f2(y)

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t[::1] xv = <REAL_t[:dim]>x
            REAL_t[::1] yv = <REAL_t[:dim]>y
        return self.f1(xv)*self.f2(yv)


cdef class leftRightTwoPoint(twoPointFunction):
    def __init__(self, REAL_t ll, REAL_t rr, REAL_t lr=np.nan, REAL_t rl=np.nan, REAL_t interface=0.):
        if not np.isfinite(lr):
            lr = 0.5*(ll+rr)
        if not np.isfinite(rl):
            rl = 0.5*(ll+rr)
        super(leftRightTwoPoint, self).__init__(rl == lr)
        self.ll = ll
        self.lr = lr
        self.rl = rl
        self.rr = rr
        self.interface = interface

    def __getstate__(self):
        return (self.ll, self.rr, self.lr, self.rl, self.interface)

    def __setstate__(self, state):
        leftRightTwoPoint.__init__(self, state[0], state[1], state[2], state[3], state[4])

    def __repr__(self):
        return '{}(ll={},rr={},lr={},rl={},interface={},sym={})'.format(self.__class__.__name__, self.ll, self.rr, self.lr, self.rl, self.interface, self.symmetric)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        if x[0] < self.interface:
            if y[0] < self.interface:
                return self.ll
            else:
                return self.lr
        else:
            if y[0] < self.interface:
                return self.rl
            else:
                return self.rr

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        if x[0] < self.interface:
            if y[0] < self.interface:
                return self.ll
            else:
                return self.lr
        else:
            if y[0] < self.interface:
                return self.rl
            else:
                return self.rr


cdef class interfaceTwoPoint(twoPointFunction):
    def __init__(self, REAL_t horizon1, REAL_t horizon2, BOOL_t left, REAL_t interface=0.):
        super(interfaceTwoPoint, self).__init__(True)
        self.horizon1 = horizon1
        self.horizon2 = horizon2
        self.left = left
        self.interface = interface

    def __getstate__(self):
        return (self.horizon1, self.horizon2, self.left, self.interface)

    def __setstate__(self, state):
        interfaceTwoPoint.__init__(self, state[0], state[1], state[2], state[3])

    def __repr__(self):
        return '{}(horizon1={},horizon2={},left={},interface={})'.format(self.__class__.__name__, self.horizon1, self.horizon2, self.left, self.interface)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.evalPtr(x.shape[0], &x[0], &y[0])

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        if dim == 1:
            if self.left:
                if ((x[0] <= self.interface) and (y[0] <= self.interface)):
                    return 1.
                elif ((x[0] > self.interface) and (y[0] > self.interface)):
                    return 0.
                elif ((x[0] <= self.interface-self.horizon2) and (y[0] > self.interface)):
                    return 1.
                elif ((x[0] > self.interface) and (y[0] <= self.interface-self.horizon2)):
                    return 1.
                else:
                    return 0.5
            else:
                if ((x[0] >= self.interface) and (y[0] >= self.interface)):
                    return 1.
                elif ((x[0] < self.interface) and (y[0] < self.interface)):
                    return 0.
                elif ((x[0] >= self.interface+self.horizon1) and (y[0] < self.interface)):
                    return 1.
                elif ((x[0] < self.interface) and (y[0] >= self.interface+self.horizon1)):
                    return 1.
                else:
                    return 0.5
        elif dim == 2:
            if self.left:
                if (x[0] <= self.interface) and ((x[1] > 0.) and (x[1] < 1.)):
                    if (y[0] <= self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        return 1.
                    elif (y[0] > self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        if x[0] <= self.interface-self.horizon2:
                            return 1.
                        else:
                            return 0.5
                    else:
                        return 1.
                elif (x[0] > self.interface) and ((x[1] > 0.) and (x[1] < 1.)):
                    if (y[0] <= self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        return 0.5
                    elif (y[0] > self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        return 0.
                    else:
                        return 0.
                else:
                    if (y[0] <= self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        return 1.
                    else:
                        return 0.
            else:
                if (x[0] >= self.interface) and ((x[1] > 0.) and (x[1] < 1.)):
                    if (y[0] >= self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        return 1.
                    elif (y[0] < self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        if x[0] >= self.interface+self.horizon1:
                            return 1.
                        else:
                            return 0.5
                    else:
                        return 1.
                elif (x[0] < self.interface) and ((x[1] > 0.) and (x[1] < 1.)):
                    if (y[0] >= self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        if y[0] <= self.interface+self.horizon1:
                            return 0.5
                        else:
                            return 1.
                    elif (y[0] < self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        return 0.
                    else:
                        return 0.
                else:
                    if (y[0] >= self.interface) and ((y[1] > 0.) and (y[1] < 1.)):
                        return 1.
                    else:
                        return 0.


cdef class temperedTwoPoint(twoPointFunction):
    def __init__(self, REAL_t lambdaCoeff, INDEX_t dim):
        super(temperedTwoPoint, self).__init__(True)
        self.lambdaCoeff = lambdaCoeff
        self.dim = dim

    def __getstate__(self):
        return (self.lambdaCoeff, self.dim)

    def __setstate__(self, state):
        temperedTwoPoint.__init__(self, state[0], state[1])

    def __repr__(self):
        return '{}(lambda={})'.format(self.__class__.__name__, self.lambdaCoeff)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            INDEX_t i
            REAL_t r = 0.
        for i in range(self.dim):
            r += (x[i]-y[i])*(x[i]-y[i])
        return exp(-self.lambdaCoeff*sqrt(r))

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            INDEX_t i
            REAL_t r = 0.
        for i in range(dim):
            r += (x[i]-y[i])*(x[i]-y[i])
        return exp(-self.lambdaCoeff*sqrt(r))


cdef class smoothedLeftRightTwoPoint(twoPointFunction):
    def __init__(self, REAL_t vl, REAL_t vr, REAL_t r=0.1, REAL_t slope=200.):
        super(smoothedLeftRightTwoPoint, self).__init__(False)
        self.vl = vl
        self.vr = vr
        self.r = r
        self.slope = slope
        self.fac = 1./atan(r*slope)

    def __getstate__(self):
        return (self.vl, self.vr, self.r, self.slope)

    def __setstate__(self, state):
        smoothedLeftRightTwoPoint.__init__(self, state[0], state[1], state[2], state[3])

    def __repr__(self):
        return '{}(vl={},vr={},r={},slope={})'.format(self.__class__.__name__, self.vl, self.vr, self.r, self.slope)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        if x[0] < -self.r:
            return self.vl
        elif x[0] > self.r:
            return self.vr
        return 0.5*(self.vl+self.vr)+0.5*(self.vr-self.vl)*atan(x[0]*self.slope) * self.fac

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        if x[0] < -self.r:
            return self.vl
        elif x[0] > self.r:
            return self.vr
        return 0.5*(self.vl+self.vr)+0.5*(self.vr-self.vl)*atan(x[0]*self.slope) * self.fac


cdef class parametrizedTwoPointFunction(twoPointFunction):
    def __init__(self, BOOL_t symmetric):
        super(parametrizedTwoPointFunction, self).__init__(symmetric)

    cdef void setParams(self, void *params):
        self.params = params

    cdef void* getParams(self):
        return self.params


cdef class productParametrizedTwoPoint(parametrizedTwoPointFunction):
    def __init__(self, twoPointFunction f1, twoPointFunction f2):
        super(productParametrizedTwoPoint, self).__init__(f1.symmetric and f2.symmetric)
        self.f1 = f1
        self.f2 = f2

    cdef void setParams(self, void *params):
        cdef:
            parametrizedTwoPointFunction f
        if isinstance(self.f1, parametrizedTwoPointFunction):
            f = self.f1
            f.setParams(params)
        if isinstance(self.f2, parametrizedTwoPointFunction):
            f = self.f2
            f.setParams(params)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.f1.eval(x, y)*self.f2.eval(x, y)

    def __repr__(self):
        return '{}*{}'.format(self.f1, self.f2)

    def __getstate__(self):
        return self.f1, self.f2

    def __setstate__(self, state):
        productParametrizedTwoPoint.__init__(self, state[0], state[1])
