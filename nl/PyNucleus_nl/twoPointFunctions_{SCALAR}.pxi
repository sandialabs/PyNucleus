###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}twoPointFunction:
    def __init__(self, BOOL_t symmetric, INDEX_t valueSize):
        self.symmetric = symmetric
        self.valueSize = valueSize

    def __call__(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            {SCALAR}_t[::1] value = uninitialized((self.valueSize), dtype={SCALAR})
        self.eval(x, y, value)
        if self.valueSize == 1:
            return value[0]
        else:
            return np.array(value, copy=False)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, {SCALAR}_t[::1] value):
        raise NotImplementedError()

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, {SCALAR}_t* value):
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
                    self.evalPtr(x.shape[0], &x[0], &y[0], &S[i, j])
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
        assert f.valueSize == 1
        self.f = f
        self.point = point
        self.fixedType = fixedType

    cdef {SCALAR}_t eval(self, REAL_t[::1] x):
        cdef:
            {SCALAR}_t val
        if self.fixedType == FIXED_X:
            self.f.evalPtr(x.shape[0], &self.point[0], &x[0], &val)
        elif self.fixedType == FIXED_Y:
            self.f.evalPtr(x.shape[0], &x[0], &self.point[0], &val)
        else:
            self.f.evalPtr(x.shape[0], &x[0], &x[0], &val)
        return val


cdef class {SCALAR_label}productTwoPoint({SCALAR_label}twoPointFunction):
    def __init__(self, {SCALAR_label}twoPointFunction f1, {SCALAR_label}twoPointFunction f2):
        assert f1.valueSize == 1
        assert f2.valueSize == 1
        super({SCALAR_label}productTwoPoint, self).__init__(f1.symmetric and f2.symmetric, 1)
        self.f1 = f1
        self.f2 = f2

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, {SCALAR}_t[::1] value):
        cdef:
            {SCALAR}_t val1, val2
        self.f1.evalPtr(x.shape[0], &x[0], &y[0], &val1)
        self.f2.evalPtr(x.shape[0], &x[0], &y[0], &val2)
        value[0] = val1*val2

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, {SCALAR}_t* value):
        cdef:
            {SCALAR}_t val1, val2
        self.f1.evalPtr(dim, x, y, &val1)
        self.f2.evalPtr(dim, x, y, &val2)
        value[0] = val1*val2

    def __repr__(self):
        return '{}*{}'.format(self.f1, self.f2)

    def __getstate__(self):
        return self.f1, self.f2

    def __setstate__(self, state):
        {SCALAR_label}productTwoPoint.__init__(self, state[0], state[1])


cdef class {SCALAR_label}constantTwoPoint({SCALAR_label}twoPointFunction):
    def __init__(self, {SCALAR}_t value):
        super({SCALAR_label}constantTwoPoint, self).__init__(True, 1)
        self.value = value

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, {SCALAR}_t[::1] value):
        value[0] = self.value

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, {SCALAR}_t* value):
        value[0] = self.value

    def __repr__(self):
        return '{}'.format(self.value)

    def __getstate__(self):
        return self.value

    def __setstate__(self, state):
        {SCALAR_label}constantTwoPoint.__init__(self, state)


cdef class {SCALAR_label}parametrizedTwoPointFunction({SCALAR_label}twoPointFunction):
    def __init__(self, BOOL_t symmetric, INDEX_t valueSize):
        super({SCALAR_label}parametrizedTwoPointFunction, self).__init__(symmetric, valueSize)

    cdef void setParams(self, void *params):
        self.params = params

    cdef void* getParams(self):
        return self.params


cdef class {SCALAR_label}productParametrizedTwoPoint({SCALAR_label}parametrizedTwoPointFunction):
    def __init__(self, {SCALAR_label}twoPointFunction f1, {SCALAR_label}twoPointFunction f2):
        assert f1.valueSize == 1
        assert f2.valueSize == 1
        super({SCALAR_label}productParametrizedTwoPoint, self).__init__(f1.symmetric and f2.symmetric, 1)
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

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, {SCALAR}_t[::1] value):
        cdef:
            {SCALAR}_t val1, val2
        self.f1.evalPtr(x.shape[0], &x[0], &y[0], &val1)
        self.f2.evalPtr(x.shape[0], &x[0], &y[0], &val2)
        value[0] = val1*val2

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, {SCALAR}_t* value):
        cdef:
            {SCALAR}_t val1, val2
        self.f1.evalPtr(dim, x, y, &val1)
        self.f2.evalPtr(dim, x, y, &val2)
        value[0] = val1*val2

    def __repr__(self):
        return '{}*{}'.format(self.f1, self.f2)

    def __getstate__(self):
        return self.f1, self.f2

    def __setstate__(self, state):
        productParametrizedTwoPoint.__init__(self, state[0], state[1])
