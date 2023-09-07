###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}twoPointFunction:
    def __init__(self, BOOL_t symmetric):
        self.symmetric = symmetric

    def __call__(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.eval(x, y)

    cdef {SCALAR}_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        raise NotImplementedError()

    cdef {SCALAR}_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        raise NotImplementedError()

    def __getstate__(self):
        return self.symmetric

    def __setstate__(self, state):
        twoPointFunction.__init__(self, state)

    def fixedX(self, REAL_t[::1] x):
        return fixedTwoPointFunction(self, x, FIXED_X)

    def fixedY(self, REAL_t[::1] y):
        return fixedTwoPointFunction(self, y, FIXED_Y)

    def diagonal(self):
        return fixedTwoPointFunction(self, None, DIAGONAL)

    def plot(self, mesh, **kwargs):
        cdef:
            INDEX_t i, j
            {SCALAR}_t[:, ::1] S
            {SCALAR}_t[::1] S2
            REAL_t[::1] x, y
        import matplotlib.pyplot as plt
        c = np.array(mesh.getCellCenters())
        if mesh.dim == 1:
            X, Y = np.meshgrid(c[:, 0], c[:, 0])
            x = np.empty((mesh.dim), dtype={SCALAR})
            y = np.empty((mesh.dim), dtype={SCALAR})
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
            S2 = np.zeros(mesh.num_cells)
            for i in range(mesh.num_cells):
                S2[i] = self(c[i, :], c[i, :])
            mesh.plotFunction(S2, flat=True)
        else:
            raise NotImplementedError()

    def __mul__(self, {SCALAR_label}twoPointFunction other):
        if isinstance(self, {SCALAR_label}constantTwoPoint) and isinstance(other, {SCALAR_label}constantTwoPoint):
            return {SCALAR_label}constantTwoPoint(self.value*other.value)
        elif isinstance(self, {SCALAR_label}parametrizedTwoPointFunction) or isinstance(other, {SCALAR_label}parametrizedTwoPointFunction):
            return {SCALAR_label}productParametrizedTwoPoint(self, other)
        elif isinstance(self, {SCALAR_label}constantTwoPoint) and isinstance(other, (float, {SCALAR})):
            return {SCALAR_label}constantTwoPoint(self.value*other)
        elif isinstance(other, {SCALAR_label}constantTwoPoint) and isinstance(self, (float, {SCALAR})):
            return {SCALAR_label}constantTwoPoint(self*other.value)
        else:
            return {SCALAR_label}productTwoPoint(self, other)


cdef class {SCALAR_label}fixedTwoPointFunction({function_type}):
    cdef:
        {SCALAR_label}twoPointFunction f
        REAL_t[::1] point
        fixed_type fixedType

    def __init__(self, {SCALAR_label}twoPointFunction f, REAL_t[::1] point, fixed_type fixedType):
        self.f = f
        self.point = point
        self.fixedType = fixedType

    cdef {SCALAR}_t eval(self, REAL_t[::1] x):
        if self.fixedType == FIXED_X:
            return self.f(self.point, x)
        if self.fixedType == FIXED_Y:
            return self.f(x, self.point)
        else:
            return self.f(x, x)


cdef class {SCALAR_label}productTwoPoint({SCALAR_label}twoPointFunction):
    def __init__(self, {SCALAR_label}twoPointFunction f1, {SCALAR_label}twoPointFunction f2):
        super(productTwoPoint, self).__init__(f1.symmetric and f2.symmetric)
        self.f1 = f1
        self.f2 = f2

    cdef {SCALAR}_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.f1.eval(x, y)*self.f2.eval(x, y)

    cdef {SCALAR}_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        return self.f1.evalPtr(dim, x, y)*self.f2.evalPtr(dim, x, y)

    def __repr__(self):
        return '{}*{}'.format(self.f1, self.f2)

    def __getstate__(self):
        return self.f1, self.f2

    def __setstate__(self, state):
        {SCALAR_label}productTwoPoint.__init__(self, state[0], state[1])


cdef class {SCALAR_label}constantTwoPoint({SCALAR_label}twoPointFunction):
    def __init__(self, {SCALAR}_t value):
        super(constantTwoPoint, self).__init__(True)
        self.value = value

    cdef {SCALAR}_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.value

    cdef {SCALAR}_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        return self.value

    def __repr__(self):
        return '{}'.format(self.value)

    def __getstate__(self):
        return self.value

    def __setstate__(self, state):
        {SCALAR_label}constantTwoPoint.__init__(self, state)


cdef class {SCALAR_label}parametrizedTwoPointFunction({SCALAR_label}twoPointFunction):
    def __init__(self, BOOL_t symmetric):
        super({SCALAR_label}parametrizedTwoPointFunction, self).__init__(symmetric)

    cdef void setParams(self, void *params):
        self.params = params

    cdef void* getParams(self):
        return self.params


cdef class {SCALAR_label}productParametrizedTwoPoint({SCALAR_label}parametrizedTwoPointFunction):
    def __init__(self, {SCALAR_label}twoPointFunction f1, {SCALAR_label}twoPointFunction f2):
        super({SCALAR_label}productParametrizedTwoPoint, self).__init__(f1.symmetric and f2.symmetric)
        self.f1 = f1
        self.f2 = f2

    cdef void setParams(self, void *params):
        cdef:
            parametrizedTwoPointFunction f
        if isinstance(self.f1, {SCALAR_label}parametrizedTwoPointFunction):
            f = self.f1
            f.setParams(params)
        if isinstance(self.f2, {SCALAR_label}parametrizedTwoPointFunction):
            f = self.f2
            f.setParams(params)

    cdef {SCALAR}_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.f1.eval(x, y)*self.f2.eval(x, y)

    cdef {SCALAR}_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        return self.f1.evalPtr(dim, x, y)*self.f2.evalPtr(dim, x, y)

    def __repr__(self):
        return '{}*{}'.format(self.f1, self.f2)

    def __getstate__(self):
        return self.f1, self.f2

    def __setstate__(self, state):
        productParametrizedTwoPoint.__init__(self, state[0], state[1])
