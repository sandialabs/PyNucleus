###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}LinearOperator:
    def __init__(self, int num_rows, int num_columns):
        self.num_rows = num_rows
        self.num_columns = num_columns

    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        return -1

    cdef INDEX_t matvec_no_overwrite(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] y) except -1:
        return -1

    cdef INDEX_t matvec_multi(self,
                              {SCALAR}_t[:, ::1] x,
                              {SCALAR}_t[:, ::1] y) except -1:
        return -1

    def __call__(self,
                 {SCALAR}_t[::1] x,
                 {SCALAR}_t[::1] y,
                 BOOL_t no_overwrite=False):
        if no_overwrite:
            self.matvec_no_overwrite(x, y)
        else:
            self.matvec(x, y)

    def dot(self, {SCALAR}_t[::1] x):
        cdef:
            np.ndarray[{SCALAR}_t, ndim=1] y = np.zeros(self.num_rows,
                                                      dtype={SCALAR})
        self(x, y)
        return y

    def dotMV(self, {SCALAR}_t[:, ::1] x):
        cdef:
            np.ndarray[{SCALAR}_t, ndim=2] yMV
        if self.num_columns == x.shape[0]:
            yMV = np.zeros((self.num_rows, x.shape[1]), dtype={SCALAR})
            self.matvec_multi(x, yMV)
        elif self.num_columns == x.shape[1]:
            yMV = np.zeros((self.num_rows, x.shape[0]), dtype={SCALAR})
            self.matvec_multi(np.ascontiguousarray(np.array(x, copy=False).T), yMV)
            return np.ascontiguousarray(yMV.T)
        else:
            raise
        return yMV

    def __add__(self, x):
        if isinstance(x, {SCALAR_label}LinearOperator):
            if isinstance(self, {SCALAR_label}Multiply_Linear_Operator):
                if isinstance(x, {SCALAR_label}Multiply_Linear_Operator):
                    return {SCALAR_label}TimeStepperLinearOperator(self.A, x.A, x.factor, self.factor)
                else:
                    return {SCALAR_label}TimeStepperLinearOperator(self.A, x, 1.0, self.factor)
            else:
                if isinstance(x, {SCALAR_label}Multiply_Linear_Operator):
                    return {SCALAR_label}TimeStepperLinearOperator(self, x.A, x.factor)
                else:
                    return {SCALAR_label}TimeStepperLinearOperator(self, x, 1.0)
        elif isinstance(x, ComplexLinearOperator):
            return wrapRealToComplex(self)+x
        elif isinstance(self, ComplexLinearOperator):
            return self+wrapRealToComplex(x)
        else:
            raise NotImplementedError('Cannot add with {}'.format(x))

    def __sub__(self, x):
        return self + (-1.*x)

    def __mul__(self, x):
        cdef:
            np.ndarray[{SCALAR}_t, ndim=1] y
            {SCALAR}_t[::1] x_mv
        try:
            x_mv = x
            y = np.zeros((self.num_rows), dtype={SCALAR})
            self(x, y)
            return y
        except Exception as e:
            if isinstance(self, {SCALAR_label}LinearOperator) and isinstance(x, {SCALAR_label}LinearOperator):
                return {SCALAR_label}Product_Linear_Operator(self, x)
            elif isinstance(self, {SCALAR_label}LinearOperator) and hasattr(x, 'ndim') and x.ndim == 2:
                return self.dotMV(x)
            elif isinstance(self, {SCALAR_label}LinearOperator) and isinstance(x, (float, int, {SCALAR})):
                return {SCALAR_label}Multiply_Linear_Operator(self, x)
            elif isinstance(x, {SCALAR_label}LinearOperator) and isinstance(self, (float, int, {SCALAR})):
                return {SCALAR_label}Multiply_Linear_Operator(x, self)
            elif isinstance(x, complex):
                if isinstance(self, ComplexLinearOperator):
                    return {SCALAR_label}Multiply_Linear_Operator(self, COMPLEX(x))
                else:
                    return ComplexMultiply_Linear_Operator(wrapRealToComplex(self), COMPLEX(x))
            elif isinstance(x, COMPLEX):
                return ComplexMultiply_Linear_Operator(wrapRealToComplex(self), x)
            elif isinstance(self, LinearOperator) and hasattr(x, 'dtype') and x.dtype == COMPLEX:
                return wrapRealToComplex(self)*x
            else:
                raise NotImplementedError('Cannot multiply {} with {}:\n{}'.format(self, x, e))

    def __rmul__(self, x):
        if isinstance(x, {SCALAR}):
            return {SCALAR_label}Multiply_Linear_Operator(self, x)
        else:
            raise NotImplementedError('Cannot multiply with {}'.format(x))

    def __neg__(self):
        return {SCALAR_label}Multiply_Linear_Operator(self, -1.0)

    property shape:
        def __get__(self):
            return (self.num_rows, self.num_columns)

    cdef void residual(self,
                       {SCALAR}_t[::1] x,
                       {SCALAR}_t[::1] rhs,
                       {SCALAR}_t[::1] result,
                       BOOL_t simpleResidual=False):
        self._residual(x, rhs, result, simpleResidual)

    cdef void preconditionedResidual(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] rhs,
                                     {SCALAR}_t[::1] result,
                                     BOOL_t simpleResidual=False):
        self._preconditionedResidual(x, rhs, result, simpleResidual)

    cdef void _residual(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] rhs,
                        {SCALAR}_t[::1] result,
                        BOOL_t simpleResidual=False):
        cdef:
            INDEX_t i
        if not simpleResidual:
            self.matvec(x, result)
            assign3(result, result, -1.0, rhs, 1.0)
        else:
            assign(result, rhs)

    cdef void _preconditionedResidual(self,
                                      {SCALAR}_t[::1] x,
                                      {SCALAR}_t[::1] rhs,
                                      {SCALAR}_t[::1] result,
                                      BOOL_t simpleResidual=False):
        raise NotImplementedError()

    def residual_py(self,
                    {SCALAR}_t[::1] x,
                    {SCALAR}_t[::1] rhs,
                    {SCALAR}_t[::1] result,
                    BOOL_t simpleResidual=False):
        self.residual(x, rhs, result, simpleResidual)

    def isSparse(self):
        raise NotImplementedError()

    def to_csr(self):
        raise NotImplementedError()

    def to_dense(self):
        return Dense_LinearOperator(self.toarray())

    def toarray(self):
        return self.to_csr().toarray()

    def toLinearOperator(self):
        def matvec(x):
            if x.ndim == 1:
                return self*x
            elif x.ndim == 2 and x.shape[1] == 1:
                if x.flags.c_contiguous:
                    return self*x[:, 0]
                else:
                    y = np.zeros((x.shape[0]), dtype=x.dtype)
                    y[:] = x[:, 0]
                    return self*y
            else:
                raise NotImplementedError()

        from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
        return ScipyLinearOperator(shape=self.shape, matvec=matvec)

    def getDenseOpFromApply(self):
        cdef:
            INDEX_t i
            {SCALAR}_t[::1] x = np.zeros((self.shape[1]), dtype={SCALAR})
            {SCALAR}_t[::1, :] B = np.zeros(self.shape, dtype={SCALAR}, order='F')
        for i in range(self.shape[1]):
            if i > 0:
                x[i-1] = 0.
            x[i] = 1.
            self.matvec(x, B[:, i])
        return np.ascontiguousarray(B)

    @staticmethod
    def HDF5read(node):
        if node.attrs['type'] == 'csr':
            return CSR_LinearOperator.HDF5read(node)
        elif node.attrs['type'] == 'sss':
            return SSS_LinearOperator.HDF5read(node)
        elif node.attrs['type'] == 'split_csr':
            return split_CSR_LinearOperator.HDF5read(node)
        elif node.attrs['type'] == 'sparseGraph':
            return sparseGraph.HDF5read(node)
        elif node.attrs['type'] == 'restriction':
            return restrictionOp.HDF5read(node)
        elif node.attrs['type'] == 'prolongation':
            return prolongationOp.HDF5read(node)
        elif node.attrs['type'] == 'dense':
            return Dense_LinearOperator.HDF5read(node)
        elif node.attrs['type'] == 'diagonal':
            return diagonalOperator.HDF5read(node)
        else:
            raise NotImplementedError()

    def __getstate__(self):
        return

    def __setstate__(self, state):
        pass

    def __repr__(self):
        return '<%dx%d %s>' % (self.num_rows,
                               self.num_columns,
                               self.__class__.__name__)

    cdef void setEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        raise NotImplementedError()

    def setEntry_py(self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        self.setEntry(I, J, val)

    cdef void addToEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        raise NotImplementedError()

    def addToEntry_py(self, INDEX_t I, INDEX_t J, {SCALAR}_t val):
        self.addToEntry(I, J, val)

    cdef {SCALAR}_t getEntry(self, INDEX_t I, INDEX_t J):
        raise NotImplementedError()

    def getEntry_py(self, INDEX_t I, INDEX_t J):
        return self.getEntry(I, J)

    def get_diagonal(self):
        if self._diagonal is not None:
            return self._diagonal
        else:
            raise NotImplementedError()

    def set_diagonal(self, {SCALAR}_t[::1] diagonal):
        assert self.num_rows == diagonal.shape[0]
        self._diagonal = diagonal

    diagonal = property(fget=get_diagonal, fset=set_diagonal)


cdef class {SCALAR_label}TimeStepperLinearOperator({SCALAR_label}LinearOperator):
    def __init__(self,
                 {SCALAR_label}LinearOperator M,
                 {SCALAR_label}LinearOperator S,
                 {SCALAR}_t facS,
                 {SCALAR}_t facM=1.0):
        assert M.num_columns == S.num_columns
        assert M.num_rows == S.num_rows
        super({SCALAR_label}TimeStepperLinearOperator, self).__init__(M.num_rows,
                                                        M.num_columns)
        self.M = M
        self.S = S
        self.facM = facM
        self.facS = facS
        self.z = uninitialized((self.M.shape[0]), dtype={SCALAR})

    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        if self.facS != 0.:
            self.S.matvec(x, y)
            if self.facS != 1.0:
                scaleScalar(y, self.facS)
        if self.facM == 1.0:
            self.M.matvec_no_overwrite(x, y)
        else:
            self.M.matvec(x, self.z)
            assign3(y, y, 1.0, self.z, self.facM)
        return 0

    cdef INDEX_t matvec_no_overwrite(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        if self.facS == 1.0:
            self.S.matvec_no_overwrite(x, y)
        elif self.facS != 0.:
            self.S.matvec(x, self.z)
            assign3(y, y, 1.0, self.z, self.facS)
        if self.facM == 1.0:
            self.M.matvec_no_overwrite(x, y)
        elif self.facM != 0.:
            self.M.matvec(x, self.z)
            assign3(y, y, 1.0, self.z, self.facM)
        return 0

    def get_diagonal(self):
        return (self.facM*np.array(self.M.diagonal, copy=False) +
                self.facS*np.array(self.S.diagonal, copy=False))

    diagonal = property(fget=get_diagonal)

    def __repr__(self):
        if np.real(self.facS) >= 0:
            return '{}*{} + {}*{}'.format(self.facM, self.M, self.facS, self.S)
        else:
            return '{}*{} - {}*{}'.format(self.facM, self.M, -self.facS, self.S)

    def to_csr_linear_operator(self):
        if isinstance(self.S, {SCALAR_label}Dense_LinearOperator):
            return {SCALAR_label}Dense_LinearOperator(self.facM*self.M.toarray() + self.facS*self.S.toarray())
        else:
            B = self.facM*self.M.to_csr() + self.facS*self.S.to_csr()
            B.eliminate_zeros()
            return {SCALAR_label}CSR_LinearOperator(B.indices, B.indptr, B.data)

    def isSparse(self):
        return self.M.isSparse() and self.S.isSparse()

    def to_csr(self):
        cdef {SCALAR_label}CSR_LinearOperator csr
        csr = self.to_csr_linear_operator()
        return csr.to_csr()

    def toarray(self):
        return self.facM*self.M.toarray() + self.facS*self.S.toarray()

    def getnnz(self):
        return self.M.nnz+self.S.nnz

    nnz = property(fget=getnnz)

    def __mul__(self, x):
        if isinstance(self, {SCALAR_label}TimeStepperLinearOperator) and isinstance(x, ({SCALAR}, float, int)):
            return {SCALAR_label}TimeStepperLinearOperator(self.M, self.S, self.facS*x, self.facM*x)
        elif isinstance(x, {SCALAR_label}TimeStepperLinearOperator) and isinstance(self, ({SCALAR}, float, int)):
            return {SCALAR_label}TimeStepperLinearOperator(x.M, x.S, x.facS*self, x.facM*self)
        else:
            return super({SCALAR_label}TimeStepperLinearOperator, self).__mul__(x)


cdef class {SCALAR_label}Multiply_Linear_Operator({SCALAR_label}LinearOperator):
    def __init__(self,
                 {SCALAR_label}LinearOperator A,
                 {SCALAR}_t factor):
        super({SCALAR_label}Multiply_Linear_Operator, self).__init__(A.num_rows,
                                                       A.num_columns)
        self.A = A
        self.factor = factor

    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        self.A(x, y)
        scaleScalar(y, self.factor)
        return 0

    cdef INDEX_t matvec_no_overwrite(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] y) except -1:
        if self.factor != 0.:
            scaleScalar(y, 1./self.factor)
            self.A.matvec_no_overwrite(x, y)
            scaleScalar(y, self.factor)
        return 0

    def isSparse(self):
        return self.A.isSparse()

    def to_csr(self):
        return self.factor*self.A.to_csr()

    def to_csr_linear_operator(self):
        if isinstance(self.A, {SCALAR_label}Dense_LinearOperator):
            return {SCALAR_label}Dense_LinearOperator(self.factor*self.A.toarray())
        else:
            B = self.factor*self.A.to_csr()
            Bcsr = {SCALAR_label}CSR_LinearOperator(B.indices, B.indptr, B.data)
            Bcsr.num_rows = B.shape[0]
            Bcsr.num_columns = B.shape[1]
            return Bcsr

    def toarray(self):
        return self.factor*self.A.toarray()

    def __mul__(self, x):
        if isinstance(self, {SCALAR_label}Multiply_Linear_Operator) and isinstance(x, ({SCALAR}, float)):
            return {SCALAR_label}Multiply_Linear_Operator(self.A, self.factor*x)
        elif isinstance(x, {SCALAR_label}Multiply_Linear_Operator) and isinstance(self, ({SCALAR}, float)):
            return {SCALAR_label}Multiply_Linear_Operator(x.A, x.factor*self)
        elif isinstance(x, COMPLEX):
                return ComplexMultiply_Linear_Operator(wrapRealToComplex(self.A), self.factor*x)
        else:
            return super({SCALAR_label}Multiply_Linear_Operator, self).__mul__(x)

    def get_diagonal(self):
        return self.factor*np.array(self.A.diagonal, copy=False)

    diagonal = property(fget=get_diagonal)

    def __repr__(self):
        return '{}*{}'.format(self.factor, self.A)


cdef class {SCALAR_label}Product_Linear_Operator({SCALAR_label}LinearOperator):
    def __init__(self,
                 {SCALAR_label}LinearOperator A,
                 {SCALAR_label}LinearOperator B,
                 {SCALAR}_t[::1] temporaryMemory=None):
        assert A.num_columns == B.num_rows, '{} and {} are not compatible'.format(A.num_columns, B.num_rows)
        super({SCALAR_label}Product_Linear_Operator, self).__init__(A.num_rows,
                                                      B.num_columns)
        self.A = A
        self.B = B
        if temporaryMemory is not None:
            assert temporaryMemory.shape[0] == self.A.num_columns
            self.temporaryMemory = temporaryMemory
        else:
            self.temporaryMemory = uninitialized((self.A.num_columns), dtype={SCALAR})

    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        self.B(x, self.temporaryMemory)
        self.A(self.temporaryMemory, y)
        return 0

    cdef INDEX_t matvec_no_overwrite(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] y) except -1:
        self.B(x, self.temporaryMemory)
        self.A.matvec_no_overwrite(self.temporaryMemory, y)
        return 0

    cdef void _residual(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] rhs,
                        {SCALAR}_t[::1] result,
                        BOOL_t simpleResidual=False):
        self.B(x, self.temporaryMemory)
        self.A.residual(self.temporaryMemory, rhs, result, simpleResidual)

    cdef void preconditionedResidual(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] rhs,
                                     {SCALAR}_t[::1] result,
                                     BOOL_t simpleResidual=False):
        self.B.residual(x, rhs, self.temporaryMemory, simpleResidual)
        self.A(self.temporaryMemory, result)

    def isSparse(self):
        return self.A.isSparse() and self.B.isSparse()

    def to_csr(self):
        return self.A.to_csr().dot(self.B.to_csr())

    def toarray(self):
        if self.isSparse():
            return self.to_csr().toarray()
        elif self.A.isSparse():
            return self.A.to_csr() * self.B.toarray()
        elif self.B.isSparse():
            return self.A.toarray() * self.B.to_csr()
        return self.A.toarray().dot(self.B.toarray())

    def to_csr_linear_operator(self):
        if isinstance(self.A, {SCALAR_label}Dense_LinearOperator):
            return {SCALAR_label}Dense_LinearOperator(self.A.toarray().dot(self.facS*self.B.toarray()))
        else:
            B = self.A.to_csr().dot(self.B.to_csr())
            B.eliminate_zeros()
            Bcsr = {SCALAR_label}CSR_LinearOperator(B.indices, B.indptr, B.data)
            Bcsr.num_rows = B.shape[0]
            Bcsr.num_columns = B.shape[1]
            return Bcsr

    def __repr__(self):
        return '{}*{}'.format(self.A, self.B)
