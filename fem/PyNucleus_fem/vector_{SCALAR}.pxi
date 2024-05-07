###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


cdef class {SCALAR_label_lc_}fe_vector:
    def __init__(self, {SCALAR}_t[::1] data, DoFMap dm):
        self.data = data
        self.dm = dm
        assert self.data.shape[0] == self.dm.num_dofs, "Mismatch in dimensions: {} != {}".format(self.data.shape[0], self.dm.num_dofs)

    def __getbuffer__(self, Py_buffer* info, int flags):
        info.buf = &self.data[0]
        info.len = self.data.shape[0]
        info.ndim = 1
        info.shape = self.data.shape
        info.strides = NULL
        info.suboffsets = NULL
        info.readonly = 0
        IF {IS_REAL}:
            info.itemsize = 8
            self.format = b"d"
        ELSE:
            info.itemsize = 16
            self.format = b"Zd"
        info.format = self.format

    @property
    def shape(self):
        return (self.data.shape[0], )

    @property
    def ndim(self):
        return 1

    @property
    def dtype(self):
        return {SCALAR}

    def __add__({SCALAR_label_lc_}fe_vector self, other):
        cdef:
            {SCALAR_label_lc_}fe_vector v, v3
            complex_fe_vector vc, v2c, v3c
            {SCALAR}_t[::1] v3d
        assert self.data.shape[0] == other.shape[0], "Mismatch in dimensions: {} != {}".format(self.data.shape[0], other.shape[0])
        if isinstance(other, {SCALAR_label_lc_}fe_vector):
            v = {SCALAR_label_lc_}fe_vector(np.empty((self.data.shape[0]), dtype={SCALAR}), self.dm)
            v3 = other
            assign3(v.data, self.data, 1.0, v3.data, 1.0)
            return v
        elif isinstance(other, np.ndarray) and other.dtype == {SCALAR}:
            v = {SCALAR_label_lc_}fe_vector(np.empty((self.data.shape[0]), dtype={SCALAR}), self.dm)
            v3d = other
            assign3(v.data, self.data, 1.0, v3d, 1.0)
            return v
        elif isinstance(other, complex_fe_vector):
            v2c = self.astype(COMPLEX)
            v3c = other
            vc = complex_fe_vector(np.empty((v2c.data.shape[0]), dtype=COMPLEX), v2c.dm)
            assign3(vc.data, v2c.data, 1.0, v3c.data, 1.0)
            return vc
        else:
            raise NotImplementedError()

    def __sub__({SCALAR_label_lc_}fe_vector self, other):
        cdef:
            {SCALAR_label_lc_}fe_vector v, v3
            complex_fe_vector vc, v2c, v3c
            {SCALAR}_t[::1] v3d
        assert self.data.shape[0] == other.shape[0], "Mismatch in dimensions: {} != {}".format(self.data.shape[0], other.shape[0])
        if isinstance(other, {SCALAR_label_lc_}fe_vector):
            v = {SCALAR_label_lc_}fe_vector(np.empty((self.data.shape[0]), dtype={SCALAR}), self.dm)
            v3 = other
            assign3(v.data, self.data, 1.0, v3.data, -1.0)
            return v
        elif isinstance(other, np.ndarray) and other.dtype == {SCALAR}:
            v = {SCALAR_label_lc_}fe_vector(np.empty((self.data.shape[0]), dtype={SCALAR}), self.dm)
            v3d = other
            assign3(v.data, self.data, 1.0, v3d, -1.0)
            return v
        elif isinstance(other, complex_fe_vector):
            v2c = self.astype(COMPLEX)
            v3c = other
            vc = complex_fe_vector(np.empty((v2c.data.shape[0]), dtype=COMPLEX), v2c.dm)
            assign3(vc.data, v2c.data, 1.0, v3c.data, -1.0)
            return vc
        else:
            raise NotImplementedError()

    def __iadd__({SCALAR_label_lc_}fe_vector self, {SCALAR}_t[::1] other):
        assert self.data.shape[0] == other.shape[0], "Mismatch in dimensions: {} != {}".format(self.data.shape[0], other.shape[0])
        assign3(self.data, self.data, 1.0, other, 1.0)
        return self

    def __isub__({SCALAR_label_lc_}fe_vector self, {SCALAR}_t[::1] other):
        assert self.data.shape[0] == other.shape[0], "Mismatch in dimensions: {} != {}".format(self.data.shape[0], other.shape[0])
        assign3(self.data, self.data, 1.0, other, -1.0)
        return self

    def __imul__({SCALAR_label_lc_}fe_vector self, {SCALAR}_t alpha):
        assignScaled(self.data, self.data, alpha)
        return self

    def __mul__(self, other):
        cdef:
            {SCALAR_label_lc_}fe_vector v1, v2
            complex_fe_vector v1c, v2c
            {SCALAR}_t alpha
            COMPLEX_t alphac
            INDEX_t i
        if isinstance(self, {SCALAR_label_lc_}fe_vector):
            if isinstance(other, (COMPLEX, complex)):
                v1c = self.astype(COMPLEX)
                alphac = other
                v2c = complex_fe_vector(np.empty((v1c.data.shape[0]), dtype=COMPLEX), v1c.dm)
                assignScaled(v2c.data, v1c.data, alphac)
                return v2c
            elif isinstance(other, {SCALAR_label_lc_}fe_vector):
                v1 = other
                v2 = {SCALAR_label_lc_}fe_vector(np.empty((v1.data.shape[0]), dtype={SCALAR}), v1.dm)
                for i in range(self.data.shape[0]):
                    v2.data[i] = self.data[i]*other.data[i]
                return v2
            else:
                v1 = self
                alpha = other
                v2 = {SCALAR_label_lc_}fe_vector(np.empty((v1.data.shape[0]), dtype={SCALAR}), v1.dm)
                assignScaled(v2.data, v1.data, alpha)
                return v2
        else:
            if isinstance(self, (COMPLEX, complex)):
                v1c = other.astype(COMPLEX)
                alphac = self
                v2c = complex_fe_vector(np.empty((v1c.data.shape[0]), dtype=COMPLEX), v1c.dm)
                assignScaled(v2c.data, v1c.data, alphac)
                return v2c
            else:
                v1 = other
                alpha = self
                v2 = {SCALAR_label_lc_}fe_vector(np.empty((v1.data.shape[0]), dtype={SCALAR}), v1.dm)
                assignScaled(v2.data, v1.data, alpha)
                return v2

    def __rmul__(self, other):
        return self.__mul__(other)

    def toarray(self, copy=False):
        return np.array(self.data, copy=copy)

    def assign(self, other):
        cdef:
            {SCALAR_label_lc_}fe_vector v
            {SCALAR}_t[::1] v2
        if isinstance(other, {SCALAR_label_lc_}fe_vector):
            v = other
            assert self.shape[0] == v.shape[0]
            assign(self.data, v.data)
        elif isinstance(other, {SCALAR}):
            for i in range(self.data.shape[0]):
                self.data[i] = other
        elif {IS_REAL} and isinstance(other, float):
            for i in range(self.data.shape[0]):
                self.data[i] = other
        else:
            v2 = other
            assert self.shape[0] == v2.shape[0]
            assign(self.data, v2)

    def astype(self, dtype):
        cdef:
            complex_fe_vector v
            INDEX_t i
        IF {IS_REAL}:
            if dtype == COMPLEX:
                v = complex_fe_vector(np.empty((self.data.shape[0]), dtype=COMPLEX), self.dm)
                for i in range(self.data.shape[0]):
                    v.data[i] = self.data[i]
                return v
            else:
                return self
        ELSE:
            if dtype == REAL:
                raise NotImplementedError()
            else:
                return self

    @property
    def real(self):
        cdef:
            fe_vector v
            INDEX_t i
        IF {IS_REAL}:
            return self
        ELSE:
            v = fe_vector(np.empty((self.data.shape[0]), dtype=REAL), self.dm)
            for i in range(self.data.shape[0]):
                v.data[i] = self.data[i].real
            return v

    @property
    def imag(self):
        cdef:
            fe_vector v
            INDEX_t i
        IF {IS_REAL}:
            v = fe_vector(np.zeros((self.data.shape[0]), dtype=REAL), self.dm)
            return v
        ELSE:
            v = fe_vector(np.empty((self.data.shape[0]), dtype=REAL), self.dm)
            for i in range(self.data.shape[0]):
                v.data[i] = self.data[i].imag
            return v

    def __repr__(self):
        if self.dm is not None:
            return '{SCALAR}fe_vector<{}>'.format(self.dm)
        else:
            return '{SCALAR}fe_vector'

    def __getitem__(self, INDEX_t i):
        return self.data[i]

    def __setitem__(self, INDEX_t i, {SCALAR}_t value):
        self.data[i] = value

    def plot(self, **kwargs):
        mesh = self.dm.mesh
        if isinstance(self.dm, P0_DoFMap):
            return mesh.plotFunction(self.toarray(), DoFMap=self.dm, **kwargs)
        elif isinstance(self.dm, Product_DoFMap) and self.dm.dim == 2:
            import matplotlib.pyplot as plt
            coords = self.dm.scalarDM.getDoFCoordinates()
            return plt.quiver(coords[:, 0], coords[:, 1], self.getComponent(0).toarray(), self.getComponent(1).toarray())
        elif isinstance(self.dm, Product_DoFMap) and self.dm.dim == 3:
            import matplotlib.pyplot as plt
            coords = self.dm.scalarDM.getDoFCoordinates()
            ax = plt.gcf().add_subplot(projection='3d')
            return plt.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
                              self.getComponent(0).toarray(), self.getComponent(1).toarray(), self.getComponent(2).toarray())
        else:
            y = self.linearPart()
            return mesh.plotFunction(y.toarray(), DoFMap=y.dm, **kwargs)

    def copy(self):
        cdef:
            {SCALAR_label_lc_}fe_vector v
        v = self.dm.empty()
        assign(v.data, self.data)
        return v

    def __reduce__(self):
        return {SCALAR_label_lc_}fe_vector, (np.array(self.data), self.dm)

    def __getattr__(self, name):
        if name == '__deepcopy__':
            raise AttributeError()
        return getattr(np.array(self.data, copy=False), name)

    cpdef REAL_t norm(self, BOOL_t acc=False, BOOL_t asynchronous=False):
        if self.dm.norm is not None:
            IF {IS_REAL}:
                return self.dm.norm.eval(self.data, acc)
            ELSE:
                return self.dm.complex_norm.eval(self.data, acc)
        else:
            raise AttributeError('\'Norm\' has not been set on the vectors DoFMap.')

    cpdef {SCALAR}_t inner(self, other, BOOL_t accSelf=False, BOOL_t accOther=False, BOOL_t asynchronous=False):
        if self.dm.inner is not None:
            IF {IS_REAL}:
                if isinstance(other, {SCALAR_label_lc_}fe_vector):
                    return self.dm.inner.eval(self.data, other.data, accSelf, accOther, asynchronous)
                else:
                    return self.dm.inner.eval(self.data, other, accSelf, accOther, asynchronous)
            ELSE:
                if isinstance(other, {SCALAR_label_lc_}fe_vector):
                    return self.dm.complex_inner.eval(self.data, other.data, accSelf, accOther, asynchronous)
                else:
                    return self.dm.complex_inner.eval(self.data, other, accSelf, accOther, asynchronous)
        else:
            raise AttributeError('\'Inner\' has not been set on the vectors DoFMap.')

    def linearPart(self):
        return self.dm.linearPart(self)[0]

    def getComponent(self, INDEX_t component):
        if isinstance(self.dm, Product_DoFMap):
            x = self.dm.scalarDM.zeros()
            R, _ = self.dm.getRestrictionProlongation(component)
            R(self, x)
            return x
        else:
            assert component == 0
            return self

    def getComponents(self):
        if isinstance(self.dm, Product_DoFMap):
            components = []
            for component in range(self.dm.numComponents):
                components.append(self.getComponent(component))
            return components
        else:
            return self

    def augmentWithBoundaryData(self, {SCALAR_label_lc_}fe_vector boundaryData):
        return self.dm.augmentWithBoundaryData(self, boundaryData)

    def exportVTK(self, filename, label):
        if isinstance(self.dm, P0_DoFMap):
            self.dm.mesh.exportSolutionVTK(x=[], filename=filename, labels=[], cell_data={label: [self.toarray()]})
        else:
            self.dm.mesh.exportSolutionVTK(x=[self], filename=filename, labels=[label])


cdef void {SCALAR_label_lc_}assign_2d({SCALAR}_t[:, ::1] y, const {SCALAR}_t[:, ::1] x):
    cdef:
        INDEX_t i, j
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[i, j] = x[i, j]


cdef void {SCALAR_label_lc_}assignScaled_2d({SCALAR}_t[:, ::1] y, const {SCALAR}_t[:, ::1] x, {SCALAR}_t alpha):
    cdef:
        INDEX_t i, j
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[i, j] = alpha*x[i, j]


cdef void {SCALAR_label_lc_}assign3_2d({SCALAR}_t[:, ::1] z, const {SCALAR}_t[:, ::1] x, {SCALAR}_t alpha, const {SCALAR}_t[:, ::1] y, {SCALAR}_t beta):
    cdef:
        INDEX_t i, j
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = alpha*x[i, j] + beta*y[i, j]


cdef class {SCALAR_label_lc_}multi_fe_vector:
    def __init__(self, {SCALAR}_t[:, ::1] data, DoFMap dm):
        self.data = data
        self.dm = dm

    def __getbuffer__(self, Py_buffer* info, int flags):
        info.buf = &self.data[0, 0]
        info.len = self.data.shape[0]
        info.ndim = 2
        info.shape = self.data.shape
        info.strides = NULL
        info.suboffsets = NULL
        info.readonly = 0
        IF {IS_REAL}:
            info.itemsize = 8
            self.format = b"d"
        ELSE:
            info.itemsize = 16
            self.format = b"Zd"
        info.format = self.format

    @property
    def shape(self):
        return (self.data.shape[0], self.data.shape[1])

    @property
    def ndim(self):
        return 2

    @property
    def numVectors(self):
        return self.data.shape[0]

    @property
    def dtype(self):
        return {SCALAR}

    def __add__({SCALAR_label_lc_}multi_fe_vector self, other):
        cdef:
            {SCALAR_label_lc_}multi_fe_vector v, v3
            complex_multi_fe_vector vc, v2c, v3c
            {SCALAR}_t[:, ::1] v3d
        if isinstance(other, {SCALAR_label_lc_}multi_fe_vector):
            v = {SCALAR_label_lc_}multi_fe_vector(np.empty((self.data.shape[0]), dtype={SCALAR}), self.dm)
            v3 = other
            {SCALAR_label_lc_}assign3_2d(v.data, self.data, 1.0, v3.data, 1.0)
            return v
        elif isinstance(other, np.ndarray) and other.dtype == {SCALAR}:
            v = {SCALAR_label_lc_}multi_fe_vector(np.empty((self.data.shape[0], self.data.shape[1]), dtype={SCALAR}), self.dm)
            v3d = other
            {SCALAR_label_lc_}assign3_2d(v.data, self.data, 1.0, v3d, 1.0)
            return v
        elif isinstance(other, complex_multi_fe_vector):
            v2c = self.astype(COMPLEX)
            v3c = other
            vc = complex_multi_fe_vector(np.empty((v2c.data.shape[0], v2c.data.shape[1]), dtype=COMPLEX), v2c.dm)
            complex_assign3_2d(vc.data, v2c.data, 1.0, v3c.data, 1.0)
            return vc
        else:
            raise NotImplementedError()

    def __sub__({SCALAR_label_lc_}multi_fe_vector self, other):
        cdef:
            {SCALAR_label_lc_}multi_fe_vector v, v3
            complex_multi_fe_vector vc, v2c, v3c
            {SCALAR}_t[:, ::1] v3d
        if isinstance(other, {SCALAR_label_lc_}multi_fe_vector):
            v = {SCALAR_label_lc_}multi_fe_vector(np.empty((self.data.shape[0], self.data.shape[1]), dtype={SCALAR}), self.dm)
            v3 = other
            {SCALAR_label_lc_}assign3_2d(v.data, self.data, 1.0, v3.data, -1.0)
            return v
        elif isinstance(other, np.ndarray) and other.dtype == {SCALAR}:
            v = {SCALAR_label_lc_}multi_fe_vector(np.empty((self.data.shape[0], self.data.shape[1]), dtype={SCALAR}), self.dm)
            v3d = other
            {SCALAR_label_lc_}assign3_2d(v.data, self.data, 1.0, v3d, -1.0)
            return v
        elif isinstance(other, complex_multi_fe_vector):
            v2c = self.astype(COMPLEX)
            v3c = other
            vc = complex_multi_fe_vector(np.empty((v2c.data.shape[0]), dtype=COMPLEX), v2c.dm)
            complex_assign3_2d(vc.data, v2c.data, 1.0, v3c.data, -1.0)
            return vc
        else:
            raise NotImplementedError()

    def __iadd__({SCALAR_label_lc_}multi_fe_vector self, {SCALAR}_t[:, ::1] other):
        {SCALAR_label_lc_}assign3_2d(self.data, self.data, 1.0, other, 1.0)
        return self

    def __isub__({SCALAR_label_lc_}multi_fe_vector self, {SCALAR}_t[:, ::1] other):
        {SCALAR_label_lc_}assign3_2d(self.data, self.data, 1.0, other, -1.0)
        return self

    def __imul__({SCALAR_label_lc_}multi_fe_vector self, {SCALAR}_t alpha):
        {SCALAR_label_lc_}assignScaled_2d(self.data, self.data, alpha)
        return self

    def __mul__(self, other):
        cdef:
            {SCALAR_label_lc_}multi_fe_vector v1, v2
            complex_multi_fe_vector v1c, v2c
            {SCALAR}_t alpha
            COMPLEX_t alphac
            INDEX_t i
        if isinstance(self, {SCALAR_label_lc_}multi_fe_vector):
            if isinstance(other, (COMPLEX, complex)):
                v1c = self.astype(COMPLEX)
                alphac = other
                v2c = complex_multi_fe_vector(np.empty((v1c.data.shape[0], v1c.data.shape[1]), dtype=COMPLEX), v1c.dm)
                complex_assignScaled_2d(v2c.data, v1c.data, alphac)
                return v2c
            elif isinstance(other, {SCALAR_label_lc_}multi_fe_vector):
                v1 = other
                v2 = {SCALAR_label_lc_}multi_fe_vector(np.empty((v1.data.shape[0], v1.data.shape[1]), dtype={SCALAR}), v1.dm)
                for i in range(self.data.shape[0]):
                    v2.data[i] = self.data[i]*other.data[i]
                return v2
            else:
                v1 = self
                alpha = other
                v2 = {SCALAR_label_lc_}multi_fe_vector(np.empty((v1.data.shape[0], v1.data.shape[1]), dtype={SCALAR}), v1.dm)
                {SCALAR_label_lc_}assignScaled_2d(v2.data, v1.data, alpha)
                return v2
        else:
            if isinstance(self, (COMPLEX, complex)):
                v1c = other.astype(COMPLEX)
                alphac = self
                v2c = complex_multi_fe_vector(np.empty((v1c.data.shape[0], v1c.data.shape[1]), dtype=COMPLEX), v1c.dm)
                complex_assignScaled_2d(v2c.data, v1c.data, alphac)
                return v2c
            else:
                v1 = other
                alpha = self
                v2 = {SCALAR_label_lc_}multi_fe_vector(np.empty((v1.data.shape[0], v1.data.shape[1]), dtype={SCALAR}), v1.dm)
                {SCALAR_label_lc_}assignScaled_2d(v2.data, v1.data, alpha)
                return v2

    def __rmul__(self, other):
        return self.__mul__(other)

    def scale(self, {SCALAR}_t[::1] other):
        cdef:
            INDEX_t i, j
            {SCALAR}_t alpha
        assert other.shape[0] == self.data.shape[0]
        for i in range(self.data.shape[0]):
            alpha = other[i]
            for j in range(self.data.shape[1]):
                self.data[i, j] *= alpha

    def scaledUpdate(self, {SCALAR}_t[::1] other, {SCALAR}_t[::1] scaling):
        cdef:
            INDEX_t i, j
            {SCALAR}_t alpha
        assert scaling.shape[0] == self.data.shape[0]
        assert other.shape[0] == self.data.shape[1]
        for i in range(self.data.shape[0]):
            alpha = scaling[i]
            for j in range(self.data.shape[1]):
                self.data[i, j] += alpha*other[j]

    def toarray(self, copy=False):
        return np.array(self.data, copy=copy)

    def assign(self, other):
        cdef:
            {SCALAR_label_lc_}multi_fe_vector v
            {SCALAR}_t[:, ::1] v2
            INDEX_t i, j
        if isinstance(other, {SCALAR_label_lc_}multi_fe_vector):
            v = other
            {SCALAR_label_lc_}assign_2d(self.data, v.data)
        elif isinstance(other, {SCALAR}):
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    self.data[i, j] = other
        elif {IS_REAL} and isinstance(other, float):
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    self.data[i, j] = other
        else:
            v2 = other
            {SCALAR_label_lc_}assign_2d(self.data, v2)

    def astype(self, dtype):
        cdef:
            complex_multi_fe_vector v
            INDEX_t i, j
        IF {IS_REAL}:
            if dtype == COMPLEX:
                v = complex_multi_fe_vector(np.empty((self.data.shape[0]), dtype=COMPLEX), self.dm)
                for i in range(self.data.shape[0]):
                    for j in range(self.data.shape[1]):
                        v.data[i, j] = self.data[i, j]
                return v
            else:
                return self
        ELSE:
            if dtype == REAL:
                raise NotImplementedError()
            else:
                return self

    @property
    def real(self):
        cdef:
            multi_fe_vector v
            INDEX_t i, j
        IF {IS_REAL}:
            return self
        ELSE:
            v = multi_fe_vector(np.empty((self.data.shape[0], self.data.shape[1]), dtype=REAL), self.dm)
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    v.data[i, j] = self.data[i, j].real
            return v

    @property
    def imag(self):
        cdef:
            multi_fe_vector v
            INDEX_t i, j
        IF {IS_REAL}:
            v = multi_fe_vector(np.zeros((self.data.shape[0]), dtype=REAL), self.dm)
            return v
        ELSE:
            v = multi_fe_vector(np.empty((self.data.shape[0]), dtype=REAL), self.dm)
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    v.data[i, j] = self.data[i, j].imag
            return v

    def __repr__(self):
        if self.dm is not None:
            return '{SCALAR}multi_fe_vector<{}>'.format(self.dm)
        else:
            return '{SCALAR}multi_fe_vector'

    def __getitem__(self, INDEX_t i):
        return {SCALAR_label_lc_}fe_vector(self.data[i, :], self.dm)

    def __setitem__(self, INDEX_t i, {SCALAR_label_lc_}fe_vector value):
        assign(self.data[i, :], value.data)

    def plot(self, **kwargs):
        mesh = self.dm.mesh
        if isinstance(self.dm, P0_DoFMap):
            return mesh.plotFunction(self.toarray(), DoFMap=self.dm, **kwargs)
        else:
            y = self.linearPart()
            return mesh.plotFunction(y.toarray(), DoFMap=y.dm, **kwargs)

    def copy(self):
        cdef:
            {SCALAR_label_lc_}multi_fe_vector v
        v = self.dm.empty(self.numVectors)
        {SCALAR_label_lc_}assign_2d(v.data, self.data)
        return v

    def __reduce__(self):
        return {SCALAR_label_lc_}multi_fe_vector, (np.array(self.data), self.dm)

    def __getattr__(self, name):
        if name == '__deepcopy__':
            raise AttributeError()
        return getattr(np.array(self.data, copy=False), name)

    def linearPart(self):
        return self.dm.linearPart(self)[0]
