###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}Kernel({SCALAR_label}twoPointFunction):
    """A kernel functions that can be used to define a nonlocal operator."""

    def __init__(self, INDEX_t dim, kernelType kType, function horizon, interactionDomain interaction, twoPointFunction scaling, twoPointFunction phi, BOOL_t piecewise=True, BOOL_t boundary=False, INDEX_t valueSize=1, REAL_t max_horizon=np.nan, INDEX_t manifold_dim=-1, **kwargs):
        cdef:
            parametrizedTwoPointFunction parametrizedScaling
            int i

        self.c_kernel_params = PyMem_Malloc(NUM_KERNEL_PARAMS*OFFSET)
        for i in range(NUM_KERNEL_PARAMS):
            (<void**>(self.c_kernel_params+i*OFFSET))[0] = NULL

        self.dim = dim
        if manifold_dim == -1:
            self.manifold_dim = dim
        else:
            self.manifold_dim = manifold_dim
        assert valueSize >= 1, "Creation of kernel with valueSize = {}".format(valueSize)
        self.valueSize = valueSize
        self.kernelType = kType
        self.piecewise = piecewise
        self.boundary = boundary

        setINDEX(self.c_kernel_params, fKDIM, dim)

        symmetric = scaling.symmetric and interaction.symmetric and (phi is None or phi.symmetric)
        super({SCALAR_label}Kernel, self).__init__(symmetric, valueSize)

        self.min_log_singularity = 0.
        self.max_log_singularity = 0.
        self.logSingularityValue = 0.

        if self.kernelType == INDICATOR:
            self.min_singularity = self.boundary
            self.max_singularity = self.boundary
            self.singularityValue = self.boundary
            setREAL(self.c_kernel_params, fEXPONENT, self.boundary)
        elif self.kernelType in (GAUSSIAN, EXPONENTIAL, POLYNOMIAL, LOGINVERSEDISTANCE):
            self.min_singularity = 0.
            self.max_singularity = 0.
            self.singularityValue = 0.
        elif self.kernelType == PERIDYNAMIC:
            setREAL(self.c_kernel_params, fEXPONENT, self.boundary-1.)
            self.min_singularity = self.boundary-1.
            self.max_singularity = self.boundary-1.
            self.singularityValue = self.boundary-1.
        elif self.kernelType == MONOMIAL:
            monomialPower = kwargs.get('monomialPower', np.nan)
            setREAL(self.c_kernel_params, fEXPONENT, self.boundary+monomialPower)
            self.min_singularity = self.boundary+self.manifold+monomialPower
            self.max_singularity = self.boundary+self.manifold+monomialPower
            self.singularityValue = self.boundary+self.manifold+monomialPower
        elif self.kernelType == GREENS_2D:
            greensLambda = kwargs.get('greens_lambda', np.nan)
            setREAL(self.c_kernel_params, fGREENS_LAMBDA, -greensLambda.imag)
            self.min_singularity = 0.
            self.max_singularity = 0.
            self.singularityValue = 0.
        elif self.kernelType == GREENS_3D:
            greensLambda = kwargs.get('greens_lambda', np.nan)
            setCOMPLEX(self.c_kernel_params, fGREENS_LAMBDA, greensLambda)
            self.min_singularity = -1.
            self.max_singularity = -1.
            self.singularityValue = -1.

        self.horizon = horizon
        self.variableHorizon = not isinstance(self.horizon, constant)
        if self.variableHorizon:
            (<void**>(self.c_kernel_params+fHORIZONFUN))[0] = <void*>horizon
            self.horizonValue2 = np.nan
            self.finiteHorizon = True
            self.max_horizon = max_horizon
        else:
            self.horizonValue = self.horizon.value
            self.max_horizon = self.horizon.value
            self.finiteHorizon = self.horizon.value != np.inf

        if self.kernelType == GAUSSIAN:
            if self.finiteHorizon:
                setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
            else:
                variance = kwargs.get('variance', 1.0)
                setREAL(self.c_kernel_params, fEXPONENTINVERSE, 0.5/variance**self.dim)
        elif self.kernelType == EXPONENTIAL:
            exponentialRate = kwargs.get('exponentialRate', 1.0)
            setREAL(self.c_kernel_params, fTEMPERED, exponentialRate)
        elif self.kernelType == POLYNOMIAL:
            a = kwargs.get('a', 1.0)
            setREAL(self.c_kernel_params, fEXPONENTINVERSE, a)

        self.interaction = interaction
        self.complement = self.interaction.complement
        (<void**>(self.c_kernel_params+fINTERACTION))[0] = <void*>self.interaction
        self.interaction.setParams(self.c_kernel_params)

        self.phi = phi
        self.scalingPrePhi = scaling
        if phi is not None:
            scaling = phi*scaling
        self.scaling = scaling
        self.variableScaling = not isinstance(self.scaling, constantTwoPoint)
        if self.variableScaling:
            (<void**>(self.c_kernel_params+fSCALINGFUN))[0] = <void*>self.scaling
            if isinstance(self.scaling, parametrizedTwoPointFunction):
                parametrizedScaling = self.scaling
                parametrizedScaling.setParams(self.c_kernel_params)
            self.scalingValue = np.nan
        else:
            self.scalingValue = self.scaling.value

        self.variableSingularity = False

        self.variable = self.variableHorizon or self.variableScaling or self.variableSingularity

        IF {IS_REAL}:
            if self.kernelType in (MONOMIAL, INDICATOR, PERIDYNAMIC):
                self.setKernelFun(monomialKernel)
            elif self.kernelType == GAUSSIAN:
                self.setKernelFun(gaussianKernel)
            elif self.kernelType == EXPONENTIAL:
                self.setKernelFun(temperedMonomialKernel)
            elif self.kernelType == LOGINVERSEDISTANCE:
                self.setKernelFun(logInverseDistanceKernel)
            elif self.kernelType == POLYNOMIAL:
                if dim == 1:
                    self.setKernelFun(polynomialKernel)
                else:
                    raise NotImplementedError()
        IF {IS_COMPLEX}:
            if self.kernelType == GREENS_2D:
                if dim == 2:
                    self.setKernelFun(greens2Dcomplex)
                else:
                    raise NotImplementedError()
            elif self.kernelType == GREENS_3D:
                if dim == 3:
                    self.setKernelFun(greens3Dcomplex)
                else:
                    raise NotImplementedError()

    @staticmethod
    def build(INDEX_t dim,
              kernel,
              horizon,
              twoPointFunction scaling=None,
              interaction=None,
              BOOL_t normalized=True,
              BOOL_t piecewise=True,
              twoPointFunction phi=None,
              REAL_t monomialPower=np.nan,
              REAL_t variance=1.,
              REAL_t exponentialRate=1.0,
              REAL_t exponentInverse=np.nan,
              REAL_t a=1.,
              REAL_t max_horizon=np.nan,
              INDEX_t termNo=-1,
              INDEX_t manifold_dim=-1):
        dim_ = _getDim(dim)
        kType = _getKernelType(kernel)
        horizonFun = _getHorizon(horizon)
        interaction = _getInteraction(interaction, horizonFun)

        if scaling is None:
            if normalized:
                if isinstance(horizonFun, constant):
                    scaling = constantIntegrableScaling(kType, interaction, dim_, horizonFun.value, gaussian_variance=variance, exponentialRate=exponentialRate)
                else:
                    scaling = variableIntegrableScaling(kType, interaction)
            else:
                scaling = constantTwoPoint(1.0)
        if (not scaling.symmetric) or (phi is not None and not phi.symmetric):
            piecewise = False
        return Kernel(dim_, kType=kType, horizon=horizonFun, interaction=interaction, scaling=scaling, phi=phi, piecewise=piecewise,
                      boundary=False, monomialPower=monomialPower, max_horizon=max_horizon, variance=variance, exponentialRate=exponentialRate, a=a, exponentInverse=exponentInverse,
                      manifold_dim=manifold_dim)

    cdef void setSimplices(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        pass

    def setNormal(self, REAL_t[::1] normal):
        assert self.n is not None and self.n.shape[0] == normal.shape[0]
        assign(self.n, normal)

    cdef setKernelFun(self, {SCALAR_label_lc_}kernel_fun_t kernelFun):
        self.kernelFun = kernelFun

    def getParamPtrAddr(self):
        return <size_t>self.c_kernel_params

    @property
    def is_integrable(self):
        dim = self.dim-self.boundary-self.manifold
        if dim > 0:
            return (dim+self.singularityValue > 0.)
        else:
            if self.kernelType == LOGINVERSEDISTANCE:
                return False
            else:
                return (self.singularityValue >= 0.)

    @property
    def manifold(self):
        return False

    @property
    def boundary(self):
        "The order of the boundary."
        return getBOOL(self.c_kernel_params, fBOUNDARY)

    @boundary.setter
    def boundary(self, BOOL_t boundary):
        setBOOL(self.c_kernel_params, fBOUNDARY, boundary)

    cdef BOOL_t getBoundary(self):
        return getBOOL(self.c_kernel_params, fBOUNDARY)

    cdef void setBoundary(self, BOOL_t boundary):
        setBOOL(self.c_kernel_params, fBOUNDARY, boundary)

    @property
    def singularityValue(self):
        "The order of the singularity."
        return getREAL(self.c_kernel_params, fSINGULARITY)

    @singularityValue.setter
    def singularityValue(self, REAL_t singularity):
        setREAL(self.c_kernel_params, fSINGULARITY, singularity)

    cdef REAL_t getSingularityValue(self):
        return getREAL(self.c_kernel_params, fSINGULARITY)

    cdef void setSingularityValue(self, REAL_t singularity):
        setREAL(self.c_kernel_params, fSINGULARITY, singularity)

    @property
    def logSingularityValue(self):
        "The order of the logarithmic singularity."
        return getREAL(self.c_kernel_params, fLOG_SINGULARITY)

    @logSingularityValue.setter
    def logSingularityValue(self, REAL_t log_singularity):
        setREAL(self.c_kernel_params, fLOG_SINGULARITY, log_singularity)

    cdef REAL_t getLogSingularityValue(self):
        return getREAL(self.c_kernel_params, fLOG_SINGULARITY)

    cdef void setLogSingularityValue(self, REAL_t log_singularity):
        setREAL(self.c_kernel_params, fLOG_SINGULARITY, log_singularity)

    @property
    def horizonValue(self):
        "The value of the interaction horizon."
        return sqrt(getREAL(self.c_kernel_params, fHORIZON2))

    @horizonValue.setter
    def horizonValue(self, REAL_t horizon):
        setREAL(self.c_kernel_params, fHORIZON2, horizon**2)

    cdef REAL_t getHorizonValue(self):
        return sqrt(getREAL(self.c_kernel_params, fHORIZON2))

    cdef void setHorizonValue(self, REAL_t horizon):
        setREAL(self.c_kernel_params, fHORIZON2, horizon**2)

    @property
    def horizonValue2(self):
        return getREAL(self.c_kernel_params, fHORIZON2)

    cdef REAL_t getHorizonValue2(self):
        return getREAL(self.c_kernel_params, fHORIZON2)

    @horizonValue2.setter
    def horizonValue2(self, REAL_t horizon2):
        setREAL(self.c_kernel_params, fHORIZON2, horizon2)

    @property
    def scalingValue(self):
        "The value of the scaling factor."
        return getREAL(self.c_kernel_params, fSCALING)

    @scalingValue.setter
    def scalingValue(self, REAL_t scaling):
        setREAL(self.c_kernel_params, fSCALING, scaling)

    cdef REAL_t getScalingValue(self):
        return getREAL(self.c_kernel_params, fSCALING)

    cdef void setScalingValue(self, REAL_t scaling):
        setREAL(self.c_kernel_params, fSCALING, scaling)

    def getLongDescription(self):
        cdef:
            str descr = ''
            dict params = {}
        params['d'] = self.dim
        if self.finiteHorizon:
            params['\\delta'] = self.horizon
        if self.kernelType == INDICATOR:
            descr = ''
        elif self.kernelType == PERIDYNAMIC:
            descr = '\\frac{1}{|x-y|}'
        elif self.kernelType == FRACTIONAL:
            descr = '\\frac{1}{|x-y|^{d+2s}}'
        elif self.kernelType == MANIFOLD_FRACTIONAL:
            descr = '(\\zeta(d+2s, d(x, y)) + \\zeta(d+2s, 1-d(x, y)))'
        elif self.kernelType == GAUSSIAN:
            if self.finiteHorizon:
                exponentInverse = self.getKernelParam('exponentInverse')
                params['a'] = exponentInverse
                descr = '\\exp(-a |x-y|^2)'
            else:
                variance = self.getKernelParam('variance')
                params['\\sigma'] = variance
                descr = '\\exp(-0.5 |x-y|^2/\\sigma^{d})'
        elif self.kernelType == EXPONENTIAL:
            exponentialRate = self.getKernelParam('exponentialRate')
            params['a'] = exponentialRate
            descr = '\\exp(-a |x-y|)'
        elif self.kernelType == POLYNOMIAL:
            a = self.getKernelParam('a')
            params['a'] = a
            descr = '\\frac{a^3 |x-y|^2}{(a^2 + |x-y|^2)^2}'
        paramsStr = ', '.join([k+'='+str(v) for k, v in params.items()])
        return self.scaling.getLongDescription() + ' ' + descr + ' ' + self.interaction.getLongDescription() + ', ' + paramsStr

    def getKernelParams(self):
        params = []
        if self.kernelType == GAUSSIAN:
            if self.finiteHorizon:
                params.append('exponentInverse')
            else:
                params.append('variance')
        elif self.kernelType == EXPONENTIAL:
            params.append('exponentialRate')
        elif self.kernelType == POLYNOMIAL:
            params.append('a')
        elif self.kernelType == MONOMIAL:
            params.append('monomialPower')
        return params

    def getKernelParam(self, str param):
        if self.kernelType == GAUSSIAN:
            if self.finiteHorizon:
                if param == 'exponentInverse':
                    return getREAL(self.c_kernel_params, fEXPONENTINVERSE)
            else:
                if param == 'variance':
                    exponentInverse = getREAL(self.c_kernel_params, fEXPONENTINVERSE)
                    variance = 1.0/(2.0*exponentInverse)**(1.0/self.dim)
                    return variance
        elif self.kernelType == EXPONENTIAL:
            if param == 'exponentialRate':
                return getREAL(self.c_kernel_params, fTEMPERED)
        elif self.kernelType == POLYNOMIAL:
            if param == 'a':
                return getREAL(self.c_kernel_params, fEXPONENTINVERSE)
        elif self.kernelType == MONOMIAL:
            if param == 'monomialPower':
                return getREAL(self.c_kernel_params, fEXPONENT)-self.boundary
        raise NotImplementedError("Parameter not available: {}".format(param))

    def getKernelKwargs(self):
        kwargs = {}
        for kernelParam in self.getKernelParams():
            kwargs[kernelParam] = self.getKernelParam(kernelParam)
        return kwargs

    cdef void evalParamsOnSimplicesPtr(self, INDEX_t dim, REAL_t* center1, REAL_t* center2, REAL_t* simplex1, REAL_t* simplex2):
        cdef:
            REAL_t[::1] center1A
        # Set the horizon.
        if self.variableHorizon:
            center1A = <REAL_t[:dim]> center1
            self.horizonValue = self.horizon.eval(center1A)

    def evalParams_py(self, REAL_t[::1] x, REAL_t[::1] y):
        "Evaluate the kernel parameters."
        cdef:
            REAL_t scalingValue
        self.evalParamsPtr(x.shape[0], &x[0], &y[0])

    cdef void updateParams(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t[::1] xA
            REAL_t scalingValue
        if self.variableHorizon:
            xA = <REAL_t[:dim]> x
            self.horizonValue = self.horizon.eval(xA)
            if self.kernelType == GAUSSIAN:
                setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
            elif self.kernelType == POLYNOMIAL:
                setREAL(self.c_kernel_params, fEXPONENTINVERSE, 1.0/(self.horizonValue/3.)**2)
        if self.variableScaling:
            self.scaling.evalPtr(dim, x, y, &scalingValue)
            self.scalingValue = scalingValue

    cdef void evalParamsPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t[::1] xA
            REAL_t scalingValue
        if self.piecewise:
            self.updateParams(dim, x, y)

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, {SCALAR}_t* vec):
        cdef:
            REAL_t[::1] xA
            REAL_t scalingValue
        if not self.piecewise:
            self.updateParams(dim, x, y)
        vec[0] = self.kernelFun(x, y, self.c_kernel_params)

    def __call__(self, REAL_t[::1] x, REAL_t[::1] y, BOOL_t callEvalParams=True):
        "Evaluate the kernel."
        cdef:
            np.ndarray[REAL_t, ndim = 1] vec
            np.ndarray[COMPLEX_t, ndim = 1] vecComplex

        assert x.shape[0] == y.shape[0], (x.shape[0], y.shape[0])
        self.evalParamsPtr(x.shape[0], &x[0], &y[0])
        if {IS_REAL}:
            vec = uninitialized((self.valueSize), dtype=REAL)
            self.evalPtr(x.shape[0], &x[0], &y[0], &vec[0])
            if self.valueSize == 1:
                return vec[0]
            else:
                return vec
        else:
            vecComplex = uninitialized((self.valueSize), dtype=COMPLEX)
            self.evalPtr(x.shape[0], &x[0], &y[0], &vecComplex[0])
            if self.valueSize == 1:
                return vecComplex[0]
            else:
                return vecComplex

    def eval_py(self, REAL_t[::1] x, REAL_t[::1] y, {SCALAR}_t[::1] vec, BOOL_t callEvalParams=True):
        "Evaluate the kernel."
        if self.piecewise and callEvalParams:
            self.evalParamsPtr(x.shape[0], &x[0], &y[0])
        self.evalPtr(x.shape[0], &x[0], &y[0], &vec[0])

    def _checkNewKernel(self, {SCALAR_label}Kernel newKernel):
        # checks
        if isinstance(newKernel.scaling, parametrizedTwoPointFunction):
            assert newKernel.getParamPtrAddr() == newKernel.scaling.getParamPtrAddr()
        if isinstance(self.scaling, parametrizedTwoPointFunction):
            assert self.getParamPtrAddr() == self.scaling.getParamPtrAddr()
        if isinstance(self.scaling, parametrizedTwoPointFunction) and isinstance(newKernel.scaling, parametrizedTwoPointFunction):
            assert self.scaling.getParamPtrAddr() != newKernel.scaling.getParamPtrAddr()

        if isinstance(newKernel.interaction, parametrizedTwoPointFunction):
            assert newKernel.getParamPtrAddr() == newKernel.interaction.getParamPtrAddr()
        if isinstance(self.interaction, parametrizedTwoPointFunction):
            assert self.interaction.getParamPtrAddr() == self.interaction.getParamPtrAddr()
        if isinstance(self.interaction, parametrizedTwoPointFunction) and isinstance(newKernel.interaction, parametrizedTwoPointFunction):
            assert self.interaction.getParamPtrAddr() != newKernel.interaction.getParamPtrAddr()

    def getModifiedKernel(self,
                          function horizon=None,
                          interactionDomain interaction=None,
                          twoPointFunction scaling=None,
                          twoPointFunction phi=None):
        cdef:
            Kernel newKernel
        if interaction is None:
            if horizon is None:
                horizon = deepcopy(self.horizon)
                interaction = deepcopy(self.interaction)
            else:
                if scaling is None and isinstance(self.scaling, (variableFractionalLaplacianScaling,
                                                                 variableIntegrableScaling)):
                    scaling = self.scaling.getScalingWithDifferentHorizon()
                if isinstance(horizon, constant) and horizon.value == np.inf:
                    interaction = fullSpace()
                else:
                    interaction = deepcopy(self.interaction)
        else:
            assert horizon is None
        if scaling is None and phi is None:
            scaling = deepcopy(self.scalingPrePhi)
            phi = deepcopy(self.phi)
        if scaling is None:
            scaling = deepcopy(self.scaling)
        kwargs = self.getKernelKwargs()
        if isinstance(self, {SCALAR_label}BoundaryKernel):
            newKernel = {SCALAR_label}BoundaryKernel.build(dim=self.dim, kernel=self.kernelType, horizon=horizon, interaction=interaction, scaling=scaling, phi=phi, piecewise=self.piecewise, manifold_dim=self.manifold_dim, **kwargs)
        else:
            newKernel = {SCALAR_label}Kernel.build(dim=self.dim, kernel=self.kernelType, horizon=horizon, interaction=interaction, scaling=scaling, phi=phi, piecewise=self.piecewise, manifold_dim=self.manifold_dim, **kwargs)
        setREAL(newKernel.c_kernel_params, fEXPONENTINVERSE, getREAL(self.c_kernel_params, fEXPONENTINVERSE))

        self._checkNewKernel(newKernel)
        return newKernel

    def getComplementKernel(self):
        "Get the complement kernel."
        raise NotImplementedError()
        newKernel = {SCALAR_label}Kernel.build(dim=self.dim, kernel=self.kernelType, horizon=self.horizon, interaction=self.interaction.getComplement(), scaling=self.scaling, piecewise=self.piecewise)
        self._checkNewKernel(newKernel)
        return newKernel

    def __repr__(self):
        extraInfo = {}
        for p in self.getKernelParams():
            extraInfo[p] = self.getKernelParam(p)
        if self.kernelType == INDICATOR:
            kernelName = 'indicator'
        elif self.kernelType == PERIDYNAMIC:
            kernelName = 'peridynamic'
        elif self.kernelType == GAUSSIAN:
            kernelName = 'Gaussian'
        elif self.kernelType == EXPONENTIAL:
            kernelName = 'exponential'
        elif self.kernelType == POLYNOMIAL:
            kernelName = 'polynomial'
        elif self.kernelType == LOGINVERSEDISTANCE:
            kernelName = 'logInverseDistance'
        elif self.kernelType == MONOMIAL:
            kernelName = 'monomial'
        elif self.kernelType == ERROR:
            kernelName = 'ERROR'
        else:
            raise NotImplementedError(self.kernelType)
        if len(extraInfo) > 0:
            extraInfo = ', '.join([str(k)+'='+str(v) for k, v in extraInfo.items()])
            return "{}({}, {}, {}, {})".format(self.__class__.__name__, kernelName, repr(self.interaction), self.scaling, extraInfo)
        else:
            return "{}({}, {}, {})".format(self.__class__.__name__, kernelName, repr(self.interaction), self.scaling)

    def __reduce__(self):
        kwargs = {}
        kwargs = self.getKernelKwargs()

        def helper(args, kwargs):
            return {SCALAR_label}Kernel(*args, **kwargs)

        return helper, ((self.dim, self.kernelType, self.horizon, self.interaction, self.scalingPrePhi, self.phi, self.piecewise, self.boundary, self.valueSize, self.max_horizon, self.manifold_dim), kwargs)

    def plot(self, REAL_t[::1] x0=None):
        "Plot the kernel function."
        from matplotlib import ticker
        import matplotlib.pyplot as plt
        if x0 is None:
            x0 = np.zeros((self.dim), dtype=REAL)
        self.evalParamsPtr(x0.shape[0], &x0[0], &x0[0])
        if self.finiteHorizon:
            delta = self.horizonValue
        else:
            delta = 2.
        x = np.linspace(-1.1*delta, 1.1*delta, 201)
        if self.dim == 1:
            vals = np.zeros((x.shape[0], self.valueSize))
            for i in range(x.shape[0]):
                y = x0+np.array([x[i]], dtype=REAL)
                if np.linalg.norm(x0-y) > 1e-9 or self.singularityValue >= 0:
                    vals[i, :] = self(x0, y)
                else:
                    vals[i, :] = np.nan
            for k in range(self.valueSize):
                plt.plot(x, vals[:, k])
            plt.yscale('log')
            if not self.finiteHorizon:
                plt.xlim([x[0], x[x.shape[0]-1]])
            if self.singularityValue < 0:
                plt.ylim(top=np.nanmax(vals))
            plt.xlabel('$x-y$')
        elif self.dim == 2:
            X, Y = np.meshgrid(x, x)
            Z = np.zeros_like(X)
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    y = x0+np.array([x[i], x[j]], dtype=REAL)
                    if np.linalg.norm(x0-y) > 1e-9 or self.singularityValue >= 0:
                        Z[j, i] = self(x0, y)
                    else:
                        Z[j, i] = np.nan
            levels = np.logspace(np.log10(Z[np.absolute(Z)>0].min()),
                                 np.log10(Z[np.absolute(Z)>0].max()), 10)
            if levels[0] < levels[levels.shape[0]-1]:
                plt.contourf(X, Y, Z, locator=ticker.LogLocator(),
                             levels=levels)
            else:
                plt.contourf(X, Y, Z)
            plt.axis('equal')
            plt.colorbar()
            plt.xlabel('$y_1-x_1$')
            plt.ylabel('$y_2-x_2$')

    def _evalKernel(self, REAL_t[::1] evaluation_nodes):
        """Evaluates the kernel at distances given by evaluation_nodes.
        This assumes that the kernel only depends on the distance between x and y."""
        cdef:
            REAL_t[::1] origin
            REAL_t[::1] y
            REAL_t node
            INDEX_t k
        origin = np.zeros((self.dim), dtype=REAL)
        y = np.zeros((self.dim), dtype=REAL)
        values = np.zeros((evaluation_nodes.shape[0]), dtype=REAL)
        if self.manifold:
            raise NotImplementedError(self.interaction)
        else:
            if self.boundary:
                self.n[0] = 1.
                for j in range(1, self.manifold_dim):
                     self.n[j] = 0.
            for k, node in enumerate(evaluation_nodes):
                y[0] = node
                values[k] = self(origin, y)
        assert np.all(np.isfinite(values[1:])), (np.array(evaluation_nodes), np.array(values))
        return values

    def _integrateKernel(self, REAL_t[::1] nodes):
        """Obtain a discrete representation of the boundary kernel via quadrature."""
        from scipy.integrate import quad

        cdef:
            INDEX_t k, numNodes, manifold_dim, dim
            REAL_t[::1] values
            REAL_t[::1] origin

        assert self.horizonValue == np.inf
        assert nodes[0] == 0.

        dim = self.dim
        manifold_dim = self.manifold_dim
        numNodes = nodes.shape[0]
        values = np.zeros((numNodes), dtype=REAL)

        # We compute
        #    values[i] = \int_{.}^{dist_i} gamma(t) (t/dist_i)^{manifold_dim-1} dt
        # where . just means that we do not care about the shift constant.
        #
        # We compute
        #  \int_{.}^{dist_1} gamma(t) t^{manifold_dim-1}
        # analytically and then obtain
        #  \int_{.}^{dist_j} gamma(t) t^{manifold_dim-1} = \int_{.}^{dist_1} gamma(t) t^{manifold_dim-1} + \sum_{k=2}^{j} \int_{dist_{k-1}}^{dist_k} gamma(t) t^{manifold_dim-1}
        # using numerical quadrature for all other intervals.

        if self.manifold:
            raise NotImplementedError()
        else:
            def pointAtDistance(t):
                p = np.zeros((dim), dtype=REAL)
                p[0] = t
                return p

        origin = pointAtDistance(0.)

        def integrand(t):
            return self(origin, pointAtDistance(t)) * t**(manifold_dim-1)

        # Assume that kernel is C*dist^singularity on [0, dist_1].
        self(origin, pointAtDistance(nodes[1]))
        if self.singularityValue+manifold_dim != 0.:
            values[1] = self.scalingValue * nodes[1]**(self.singularityValue+manifold_dim)/(self.singularityValue+manifold_dim)
        else:
            values[1] = self.scalingValue * log(nodes[1])

        for k in range(2, numNodes):
            values[k] = quad(integrand, nodes[k-1], nodes[k], points=[0.], epsabs=1e-10, limit=100)[0]

        values[0] = 0.
        for k in range(1, numNodes):
            values[k] += values[k-1]

        for k in range(numNodes):
            if nodes[k] > 0:
                values[k] *= nodes[k]**(1-manifold_dim)

        assert np.all(np.isfinite(values[1:])), np.array(values)
        return values

    def getCorrectedKernel(self, REAL_t[::1] interpolation_nodes, {SCALAR}_t[::1] interpolation_values, str kind='linear'):
        """Return a kernel of the form

        \\gamma(|x-y|) = I[k(|x-y|)/k_M(|x-y|)] k_M(|x-y|)

        where k_M is given by self and I[.] is the piecewise linear interpolation wrt interpolation_nodes.
        The values of k are given by interpolation_values, meaning that

        \\gamma(interpolation_nodes[j]) = interpolation_values[j]

        """
        cdef:
            {SCALAR}_t[::1] base_values, correction_factor
            INDEX_t k, numNodes
        numNodes = interpolation_nodes.shape[0]
        assert numNodes == interpolation_values.shape[0]
        assert np.min(interpolation_nodes) == 0., (np.min(interpolation_nodes), self.kernelType, self.singularityValue)
        assert np.all(np.array(interpolation_nodes)[1:] > np.array(interpolation_nodes)[:numNodes-1])
        # interpolate base kernel at nodes
        if self.valueSize == 1:
            base_values = self._evalKernel(interpolation_nodes)
        else:
            base_values2d = self._evalKernel(interpolation_nodes)
            base_values = np.ascontiguousarray(base_values2d[:, 0])
        if self.boundary:
            for k in range(base_values.shape[0]):
                base_values[k] = -base_values[k]
        assert np.all(np.isfinite(base_values[1:])), (np.array(interpolation_nodes), np.array(base_values))
        correction_factor = uninitialized((numNodes), dtype={SCALAR})
        for k in range(numNodes):
            if abs(base_values[k]) < 1e-15:
                correction_factor[k] = 1.0
            elif k == 0 and not (np.isfinite(interpolation_values[k]) and np.isfinite(base_values[k])):
                correction_factor[k] = interpolation_values[1]/base_values[1]
            else:
                correction_factor[k] = interpolation_values[k]/base_values[k]
        assert np.all(np.isfinite(correction_factor)), (np.array(interpolation_nodes), np.array(interpolation_values), np.array(base_values), np.array(correction_factor))
        if kind == 'linear':
            correction = UniformLookup1D(interpolation_nodes[0], interpolation_nodes[numNodes-1], correction_factor, outOfBoundsValue=np.nan)
        else:
            correction = Lookup1D(interpolation_nodes, correction_factor, outOfBoundsValue=np.nan, kind=kind)
        interaction = deepcopy(self.interaction)
        phi = functionOfDistance(interaction, correction)
        newKernel = self.getModifiedKernel(interaction=interaction, phi=phi)
        self._checkNewKernel(newKernel)
        return newKernel

    def interpolate(self, REAL_t[::1] interpolation_nodes, str kind='linear'):
        """Interpolate the kernel wrt interpolation_nodes. This can be useful if evaluation of the kernel is expensive."""
        interpolation_values = self._evalKernel(interpolation_nodes)
        base_kernel = {SCALAR_label}Kernel.build(kernel=MONOMIAL,
                                                 monomialPower=self.singularityValue,
                                                 dim=self.dim,
                                                 manifold_dim=self.manifold_dim,
                                                 horizon=deepcopy(self.horizon),
                                                 scaling=constantTwoPoint(self.scalingValue),
                                                 interaction=deepcopy(self.interaction))
        newKernel = base_kernel.getCorrectedKernel(interpolation_nodes, interpolation_values, kind)
        self._checkNewKernel(newKernel)
        return newKernel

    @property
    def isInterpolatedKernel(self):
        return isinstance(self.phi, functionOfDistance) and isinstance(self.phi.fun, (UniformLookup1D, Lookup1D))

    def getBoundaryKernel(self, REAL_t[::1] interpolation_nodes=None, str kind='linear'):
        "Get the boundary kernel. This is the kernel that corresponds to the elimination of a subdomain via Gauss theorem."
        cdef:
            Kernel newKernel
        assert not self.boundary
        if interpolation_nodes is None and self.isInterpolatedKernel:
            if isinstance(self.phi.fun, UniformLookup1D):
                interpolation_nodes = np.linspace(self.phi.fun.a, self.phi.fun.b, self.phi.fun.vals.shape[0])
                kind = 'linear'
            elif isinstance(self.phi.fun, Lookup1D):
                interpolation_nodes = self.phi.fun.x
                kind = self.phi.fun.kind
        if interpolation_nodes is None:
            scaling = deepcopy(self.scaling)
            if isinstance(self.phi, constantTwoPoint):
                phi = deepcopy(self.phi)
            elif self.phi is not None:
                newKernel = {SCALAR_label}BoundaryKernel.build(kernel=ERROR,
                                                               dim=self.dim,
                                                               horizon=deepcopy(self.horizon),
                                                               interaction=None,
                                                               scaling=scaling,
                                                               phi=deepcopy(self.phi),
                                                               piecewise=self.piecewise)
                return newKernel
            else:
                phi = None

            if self.kernelType in (MONOMIAL, INDICATOR, PERIDYNAMIC):
                if abs(self.singularityValue+self.dim) > 1e-9:
                    scaling *= constantTwoPoint(1.0/(self.singularityValue+self.dim))
                else:
                    # log distance kernel case
                    scaling *= constantTwoPoint(-1.0)
            elif self.kernelType == EXPONENTIAL:
                a = getREAL(self.c_kernel_params, fTEMPERED)
                scaling *= constantTwoPoint(-1.0/a)

            kwargs = {}
            if self.kernelType == MONOMIAL:
                kwargs['monomialPower'] = self.getKernelParam('monomialPower')
            newKernel = {SCALAR_label}BoundaryKernel.build(kernel=self.kernelType,
                                                           dim=self.dim,
                                                           horizon=deepcopy(self.horizon),
                                                           interaction=None,
                                                           scaling=scaling,
                                                           phi=phi,
                                                           piecewise=self.piecewise,
                                                           **kwargs)
            setREAL(newKernel.c_kernel_params, fEXPONENTINVERSE, getREAL(self.c_kernel_params, fEXPONENTINVERSE))
        else:
            if not self.variable or (self.variableScaling and isinstance(self.scalingPrePhi, constantTwoPoint) and isinstance(self.phi, functionOfDistance)):
                interpolation_values = self._integrateKernel(interpolation_nodes)
                base_boundary_kernel = BoundaryKernel.build(kernel=MONOMIAL, monomialPower=self.singularityValue, dim=self.dim, horizon=np.inf, normalized=False, interaction=deepcopy(self.interaction), manifold_dim=self.manifold_dim)
                if abs(self.dim+1+base_boundary_kernel.singularityValue) < 1e-9:
                    base_boundary_kernel = BoundaryKernel.build(kernel=LOGINVERSEDISTANCE, dim=self.dim, horizon=np.inf, normalized=False, interaction=deepcopy(self.interaction), manifold_dim=self.manifold_dim)
                newKernel = base_boundary_kernel.getCorrectedKernel(interpolation_nodes, interpolation_values, kind)
            else:
                newKernel = {SCALAR_label}BoundaryKernel.build(kernel=ERROR,
                                                               dim=self.dim,
                                                               horizon=deepcopy(self.horizon),
                                                               interaction=None,
                                                               scaling=deepcopy(self.scaling),
                                                               phi=deepcopy(self.phi),
                                                               piecewise=self.piecewise,
                                                               manifold_dim=self.manifold_dim)


        self._checkNewKernel(newKernel)
        return newKernel

    def __dealloc__(self):
        PyMem_Free(self.c_kernel_params)


cdef class {SCALAR_label}BoundaryKernel({SCALAR_label}Kernel):
    def __init__(self, INDEX_t dim, kernelType kType, function horizon, interactionDomain interaction, twoPointFunction scaling, twoPointFunction phi, BOOL_t piecewise=True, INDEX_t valueSize=1, REAL_t max_horizon=np.nan, INDEX_t manifold_dim=-1, **kwargs):
        super({SCALAR_label}BoundaryKernel, self).__init__(dim=dim,
                                                           kType=kType,
                                                           horizon=horizon,
                                                           interaction=interaction,
                                                           scaling=scaling,
                                                           phi=phi,
                                                           piecewise=piecewise,
                                                           boundary=True,
                                                           valueSize=valueSize,
                                                           max_horizon=max_horizon,
                                                           manifold_dim=manifold_dim,
                                                           **kwargs)
        IF {IS_REAL}:
            if self.kernelType in (INDICATOR, MONOMIAL, PERIDYNAMIC):
                if abs(dim-self.boundary+self.singularityValue) < 1e-9:
                    self.kernelType = LOGINVERSEDISTANCE
                    self.setKernelFun(logInverseDistanceKernel)
                else:
                    self.setKernelFun(monomialKernel)
            elif self.kernelType == GAUSSIAN:
                self.setKernelFun(gaussianKernelBoundary)
            elif self.kernelType == EXPONENTIAL:
                self.setKernelFun(temperedMonomialKernel)
            elif self.kernelType == POLYNOMIAL:
                if dim == 1:
                    self.setKernelFun(polynomialKernel1Dboundary)
                else:
                    raise NotImplementedError()
        self.isSphericalManifold = False
        self.n = uninitialized((self.manifold_dim), dtype=REAL)

    cdef void setSimplices(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            REAL_t val
        if not self.isSphericalManifold:
            if self.dim == 1:
                if 0.5*(simplex1[0, 0]+simplex1[1, 0]) > simplex2[0, 0]:
                    self.n[0] = -1.
                else:
                    self.n[0] = 1.
            elif self.dim == 2:
                # (x_1-x_0)^\perp
                self.n[0] = simplex2[1, 1] - simplex2[0, 1]
                self.n[1] = simplex2[0, 0] - simplex2[1, 0]
                val = 1./sqrt(mydot(self.n, self.n))
                self.n[0] *= val
                self.n[1] *= val
            elif self.dim == 3:
                self.n[0] = (simplex2[1, 1]-simplex2[0, 1])*(simplex2[2, 2]-simplex2[0, 2])-(simplex2[1, 2]-simplex2[0, 2])*(simplex2[2, 1]-simplex2[0, 1])
                self.n[1] = (simplex2[1, 2]-simplex2[0, 2])*(simplex2[2, 0]-simplex2[0, 0])-(simplex2[1, 0]-simplex2[0, 0])*(simplex2[2, 2]-simplex2[0, 2])
                self.n[2] = (simplex2[1, 0]-simplex2[0, 0])*(simplex2[2, 1]-simplex2[0, 1])-(simplex2[1, 1]-simplex2[0, 1])*(simplex2[2, 0]-simplex2[0, 0])
                val = 1./sqrt(mydot(self.n, self.n))
                self.n[0] *= val
                self.n[1] *= val
                self.n[2] *= val
        else:
            if self.manifold_dim == 1:
                self.n[0] = 1.

    @staticmethod
    def build(INDEX_t dim,
              kernel,
              horizon,
              twoPointFunction scaling=None,
              interaction=None,
              BOOL_t normalized=True,
              BOOL_t piecewise=True,
              twoPointFunction phi=None,
              REAL_t monomialPower=np.nan,
              REAL_t variance=1.,
              REAL_t exponentialRate=1.0,
              REAL_t exponentInverse=np.nan,
              REAL_t a=1.,
              REAL_t max_horizon=np.nan,
              INDEX_t manifold_dim=-1):
        dim_ = _getDim(dim)
        kType = _getKernelType(kernel)
        horizonFun = _getHorizon(horizon)
        interaction = _getInteraction(interaction, horizonFun)

        if scaling is None:
            if normalized:
                if isinstance(horizonFun, constant):
                    scaling = constantIntegrableScaling(kType, interaction, dim_, horizonFun.value, gaussian_variance=variance, exponentialRate=exponentialRate)
                else:
                    scaling = variableIntegrableScaling(kType, interaction)
            else:
                scaling = constantTwoPoint(1.0)
        if (not scaling.symmetric) or (phi is not None and not phi.symmetric):
            piecewise = False
        return BoundaryKernel(dim_, kType=kType, horizon=horizonFun, interaction=interaction, scaling=scaling, phi=phi, piecewise=piecewise,
                              monomialPower=monomialPower, max_horizon=max_horizon, variance=variance, exponentialRate=exponentialRate, a=a, exponentInverse=exponentInverse,
                              manifold_dim=manifold_dim)

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, {SCALAR}_t* vec):
        cdef:
            {SCALAR}_t temp
            INDEX_t i
        if not self.piecewise:
            self.updateParams(dim, x, y)
        temp = self.kernelFun(&x[0], &y[0], self.c_kernel_params)
        if not self.isSphericalManifold:
            if self.interaction.dist2 > 0:
                temp /= sqrt(self.interaction.dist2)
                vec[0] = 0.
                for i in range(dim):
                    vec[0] += temp * (x[i]-y[i])*self.n[i]
            else:
                for i in range(self.dim):
                    vec[i] = temp
        else:
            if self.interaction.dist2 > 0:
                vec[0] = temp
            else:
                for i in range(self.dim):
                    vec[i] = temp

    def getBoundaryKernel(self, REAL_t[::1] interpolation_nodes=None):
        raise NotImplementedError()


cdef class {SCALAR_label}ErrorKernel({SCALAR_label}Kernel):
    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, {SCALAR}_t* vec):
        raise Exception("Cannot evaluate {SCALAR_label}ErrorKernel")


cdef class {SCALAR_label}MultiSingularityKernel({SCALAR_label}twoPointFunction):
    def __init__(self, list kernels):
        cdef:
            {SCALAR_label}Kernel kernel
        assert len(kernels) >= 2
        symmetric = True
        valueSize = kernels[0].valueSize
        for kernel in kernels:
            symmetric &= kernel.symmetric
            assert kernel.valueSize == valueSize
        super().__init__(symmetric, valueSize)
        self.kernels = kernels
        self.tempVec = uninitialized((self.valueSize), dtype={SCALAR})

    @property
    def is_integrable(self):
        integrable = self.kernels[0].is_integrable
        for kernel in self.kernels[1:]:
            integrable &= kernel.is_integrable
        return integrable

    @property
    def horizon(self):
        horizon = self.kernels[0].horizon
        assert isinstance(horizon, constant)
        hv = horizon.value
        for kernel in self.kernels[1:]:
            horizon = kernel.horizon
            assert isinstance(horizon, constant)
            hv = max(hv, horizon.value)
        return constant(hv)

    @property
    def singularityValue(self):
        sv = self.kernels[0].singularityValue
        for kernel in self.kernels[1:]:
            sv = min(sv, kernel.singularityValue)
        return sv

    @property
    def max_horizon(self):
        mh = self.kernels[0].max_horizon
        for kernel in self.kernels[1:]:
            mh = min(mh, kernel.max_horizon)
        return mh

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, {SCALAR}_t* vec):
        cdef:
            INDEX_t i, k
            {SCALAR_label}Kernel kernel
            {SCALAR}_t fac
        for i in range(self.valueSize):
            vec[i] = 0.
        for k in range(len(self.kernels)):
            kernel = self.kernels[k]
            kernel.evalParamsPtr(dim, x, y)
            kernel.evalPtr(dim, x, y, &self.tempVec[0])
            fac = kernel.scaling.values[kernel.scaling.termNo]
            for i in range(self.valueSize):
                vec[i] += fac*self.tempVec[i]

    def setNormal(self, REAL_t[::1] normal):
        for k in range(len(self.kernels)):
            kernel = self.kernels[k]
            kernel.setNormal(normal)

    def getModifiedKernel(self,
                          function horizon=None,
                          interactionDomain interaction=None,
                          twoPointFunction scaling=None,
                          twoPointFunction phi=None):
        mkernels = []
        for kernel in self.kernels:
            mkernels.append(kernel.getModifiedKernel(horizon=horizon, interaction=interaction, scaling=scaling, phi=phi))
        return type(self)(mkernels)

    def getBoundaryKernel(self, REAL_t[::1] interpolation_nodes=None):
        bkernels = []
        for kernel in self.kernels:
            bkernels.append(kernel.getBoundaryKernel(interpolation_nodes))
        return type(self)(bkernels)

    def __repr__(self):
        return ' + '.join([repr(kernel) for kernel in self.kernels])

    def getLongDescription(self):
        return ' + '.join([k.getLongDescription() for k in self.kernels])

    def __reduce__(self):
        return {SCALAR_label}MultiSingularityKernel, (self.kernels, )
