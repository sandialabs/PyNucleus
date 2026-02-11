###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class FractionalKernel(Kernel):
    """A kernel function that can be used to define a fractional operator."""

    def __init__(self,
                 INDEX_t dim,
                 fractionalOrderBase s,
                 function horizon,
                 interactionDomain interaction,
                 twoPointFunction scaling,
                 twoPointFunction phi=None,
                 BOOL_t piecewise=True,
                 BOOL_t boundary=False,
                 INDEX_t derivative=0,
                 REAL_t tempered=0.,
                 REAL_t max_horizon=np.nan,
                 kernelType kType=FRACTIONAL,
                 INDEX_t manifold_dim=-1,
                 INDEX_t termNo=-1):
        if derivative == 0:
            valueSize = 1
        elif derivative == 1:
            valueSize = s.numParameters
            self.tempVec = uninitialized((s.numParameters), dtype=REAL)
        elif derivative == 2:
            valueSize = s.numParameters**2
            self.tempVec = uninitialized((s.numParameters), dtype=REAL)
        else:
            valueSize = 1
        self.termNo = termNo
        assert isinstance(scaling, (constantFractionalLaplacianScaling, variableFractionalLaplacianScaling, variableFractionalLaplacianScalingWithDifferentHorizon)), scaling
        assert scaling.termNo == self.termNo, (scaling.termNo, self.termNo)

        super(FractionalKernel, self).__init__(dim, kType, horizon, interaction, scaling, phi, piecewise, boundary, valueSize, max_horizon, manifold_dim=manifold_dim)

        self.symmetric = s.symmetric and scaling.symmetric and interaction.symmetric and (phi is None or phi.symmetric)
        self.derivative = derivative
        self.temperedValue = tempered

        self.s = s
        self.variableSingularity = isinstance(self.s, variableFractionalOrder)

        self.min_singularity = self.boundary+self.manifold-self.dim-2*self.s.min
        self.max_singularity = self.boundary+self.manifold-self.dim-2*self.s.max

        if self.variableSingularity:
            (<void**>(self.c_kernel_params+fORDERFUN))[0] = <void*>s
            self.sValue = np.nan
            self.singularityValue = np.nan
        else:
            self.sValue = self.s.value
            self.singularityValue = self.max_singularity
            setREAL(self.c_kernel_params, fEXPONENT, self.max_singularity)

        self.variable = self.variableSingularity or self.variableHorizon or self.variableScaling

        if piecewise and isinstance(s, singleVariableUnsymmetricFractionalOrder):
            self.piecewise = False

        if tempered == 0.:
            if derivative == 0:
                self.setKernelFun(monomialKernel)
            elif derivative == 1:
                if termNo == 1:
                    self.min_log_singularity = 1.
                    self.max_log_singularity = 1.
                    self.logSingularityValue = 1.
                    self.setKernelFun(monomialLogKernel)
                elif termNo == 0 or termNo == -1:
                    self.setKernelFun(monomialKernel)
                else:
                    raise NotImplementedError()
            elif derivative == 2:
                if termNo == 2:
                    self.min_log_singularity = 2.
                    self.max_log_singularity = 2.
                    self.logSingularityValue = 2.
                    self.setKernelFun(monomialLogKernel)
                elif termNo == 1:
                    self.min_log_singularity = 1.
                    self.max_log_singularity = 1.
                    self.logSingularityValue = 1.
                    self.setKernelFun(monomialLogKernel)
                elif termNo == 0 or termNo == -1:
                    self.setKernelFun(monomialKernel)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        else:
            if not boundary:
                self.setKernelFun(temperedMonomialKernel)
            else:
                self.setKernelFun(temperedMonomialKernelBoundary)

    @staticmethod
    def build(dim,
              s,
              horizon=None,
              interaction=None,
              twoPointFunction scaling=None,
              BOOL_t normalized=True,
              BOOL_t piecewise=True,
              twoPointFunction phi=None,
              INDEX_t derivative=0,
              REAL_t tempered=0.,
              REAL_t max_horizon=np.nan,
              BOOL_t manifold=False,
              INDEX_t termNo=0):
        dim_ = _getDim(dim)
        sFun = _getFractionalOrder(s)
        horizonFun = _getHorizon(horizon)
        interaction = _getInteraction(interaction, horizonFun)

        if isinstance(sFun, admissibleSet):
            kernel = RangedFractionalKernel(dim=dim_, admissibleOrders=sFun, horizon=horizonFun, normalized=normalized, tempered=tempered)
        else:
            if scaling is None:
                if isinstance(sFun, constFractionalOrder) and isinstance(horizonFun, constant):
                    scaling = constantFractionalLaplacianScaling(dim, sFun.value, horizonFun.value, normalized, False, derivative, tempered, termNo=termNo)
                else:
                    symmetric = sFun.symmetric and isinstance(horizonFun, constant)
                    if piecewise and isinstance(sFun, singleVariableUnsymmetricFractionalOrder):
                        piecewise = False
                    scaling = variableFractionalLaplacianScaling(symmetric, normalized, False, derivative, termNo)
            kernel = FractionalKernel(dim=dim_, s=sFun, horizon=horizonFun, interaction=interaction, scaling=scaling, phi=phi, piecewise=piecewise, boundary=False,
                                      derivative=derivative, tempered=tempered, max_horizon=max_horizon, termNo=termNo)

        from . twoPointFunctions import parametrizedTwoPointFunction
        if isinstance(kernel.scaling, parametrizedTwoPointFunction):
            assert kernel.getParamPtrAddr() == kernel.scaling.getParamPtrAddr()
        if isinstance(kernel.interaction, parametrizedTwoPointFunction):
            assert kernel.getParamPtrAddr() == kernel.interaction.getParamPtrAddr()
        return kernel

    cdef setKernelFun(self, kernel_fun_t kernelFun):
        self.kernelFun = kernelFun

    cdef void setSingularityValue(self, REAL_t singularity):
        setREAL(self.c_kernel_params, fSINGULARITY, singularity)
        setREAL(self.c_kernel_params, fEXPONENT, singularity)

    @property
    def variableOrder(self):
        return self.variableSingularity

    @property
    def sValue(self):
        "The value of the fractional order"
        return getREAL(self.c_kernel_params, fS)

    @sValue.setter
    def sValue(self, REAL_t s):
        setREAL(self.c_kernel_params, fS, s)

    cdef REAL_t getsValue(self):
        return getREAL(self.c_kernel_params, fS)

    cdef void setsValue(self, REAL_t s):
        setREAL(self.c_kernel_params, fS, s)

    @property
    def temperedValue(self):
        "The value of the tempering parameter"
        return getREAL(self.c_kernel_params, fTEMPERED)

    @temperedValue.setter
    def temperedValue(self, REAL_t tempered):
        setREAL(self.c_kernel_params, fTEMPERED, tempered)

    cdef REAL_t gettemperedValue(self):
        return getREAL(self.c_kernel_params, fTEMPERED)

    cdef void settemperedValue(self, REAL_t tempered):
        setREAL(self.c_kernel_params, fTEMPERED, tempered)

    cdef void evalParamsOnSimplicesPtr(self, INDEX_t dim, REAL_t* center1, REAL_t* center2, REAL_t* simplex1, REAL_t* simplex2):
        # Set the max singularity and the horizon.
        cdef:
            REAL_t sValue, sValue2
            REAL_t[::1] center1A
        if self.variableOrder:
            if self.s.symmetric:
                self.s.evalPtr(dim, center1, center2, &sValue)
            else:
                sValue = 0.
                self.s.evalPtr(dim, center1, center2, &sValue2)
                sValue = max(sValue, sValue2)
                self.s.evalPtr(dim, center2, center1, &sValue2)
                sValue = max(sValue, sValue2)
                for i in range(self.manifold_dim+1):
                    self.s.evalPtr(dim, &simplex1[i*self.dim], center2, &sValue2)
                    sValue = max(sValue, sValue2)
                for i in range(self.manifold_dim+1-self.getBoundary()):
                    self.s.evalPtr(dim, &simplex2[i*self.dim], center1, &sValue2)
                    sValue = max(sValue, sValue2)
            if not self.getBoundary():
                self.setSingularityValue(-self.dim-2*sValue)
            else:
                self.setSingularityValue(1-self.dim-2*sValue)
        if self.variableHorizon:
            center1A = <REAL_t[:dim]> center1
            self.setHorizonValue(self.horizon.eval(center1A))

    cdef void updateParams(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t[::1] xA
            REAL_t sValue, scalingValue
        if self.variableOrder:
            self.s.evalPtr(dim, x, y, &sValue)
            if not self.getBoundary():
                self.setSingularityValue(-self.dim-2*sValue)
            else:
                self.setSingularityValue(1-self.dim-2*sValue)
            self.setsValue(sValue)
        if self.variableHorizon:
            xA = <REAL_t[:dim]> x
            self.setHorizonValue(self.horizon.eval(xA))
        if self.variableScaling:
            self.scaling.evalPtr(dim, x, y, &scalingValue)
            self.setScalingValue(scalingValue)

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* vec):
        cdef:
            INDEX_t i, j, k
            REAL_t fac
            REAL_t[::1] xA
            REAL_t sValue, scalingValue
        if not self.piecewise:
            self.updateParams(dim, x, y)
        if self.derivative == 0:
            vec[0] = self.kernelFun(x, y, self.c_kernel_params)
        elif self.derivative == 1:
            fac = self.kernelFun(x, y, self.c_kernel_params)
            self.s.evalGradPtr(dim, x, y, self.valueSize, vec)
            for i in range(self.valueSize):
                vec[i] *= fac
        elif self.derivative == 2:
            fac = self.kernelFun(&x[0], &y[0], self.c_kernel_params)
            self.s.evalGradPtr(dim, x, y, self.s.numParameters, &self.tempVec[0])
            k = 0
            for i in range(self.s.numParameters):
                for j in range(self.s.numParameters):
                    vec[k] = fac*self.tempVec[i]*self.tempVec[j]
                    k += 1

    def getModifiedKernel(self,
                          fractionalOrderBase s=None,
                          function horizon=None,
                          interactionDomain interaction=None,
                          twoPointFunction scaling=None,
                          twoPointFunction phi=None):
        if s is None:
            s = self.s
        else:
            if scaling is None and isinstance(self.scalingPrePhi, variableFractionalLaplacianScaling):
                raise NotImplementedError()
        if interaction is None:
            if horizon is None:
                horizon = self.horizon
                interaction = self.interaction
            else:
                if scaling is None and isinstance(self.scalingPrePhi, variableFractionalLaplacianScaling):
                    scaling = self.scalingPrePhi.getScalingWithDifferentHorizon()
                    interaction = deepcopy(self.interaction)
        else:
            assert horizon is None
        if phi is None:
            phi = deepcopy(self.phi)
        if scaling is None:
            scaling = deepcopy(self.scalingPrePhi)
        newKernel = FractionalKernel.build(dim=self.dim, s=s, horizon=horizon, interaction=interaction, scaling=scaling, phi=phi, piecewise=self.piecewise, derivative=self.derivative, tempered=self.temperedValue, termNo=self.termNo)
        self._checkNewKernel(newKernel)
        return newKernel

    def getBoundaryKernel(self, REAL_t[::1] interpolation_nodes=None, str kind='linear'):
        "Get the boundary kernel. This is the kernel that corresponds to the elimination of a subdomain via Gauss theorem."
        cdef:
            constantFractionalLaplacianScaling scal
            variableFractionalLaplacianScaling scalVar
            variableFractionalLaplacianScalingWithDifferentHorizon scalVarDiffHorizon

        assert not self.boundary
        if interpolation_nodes is None:
            s = deepcopy(self.s)

            if isinstance(self.scalingPrePhi, constantFractionalLaplacianScaling):
                scal = self.scalingPrePhi
                scaling = constantFractionalLaplacianScaling(dim=scal.dim, s=scal.s, horizon=scal.horizon, normalized=scal.normalized, boundary=True, derivative=scal.derivative, tempered=scal.tempered, termNo=scal.termNo)
            elif isinstance(self.scalingPrePhi, variableFractionalLaplacianScalingWithDifferentHorizon):
                scalVarDiffHorizon = self.scalingPrePhi
                scaling = variableFractionalLaplacianScalingWithDifferentHorizon(symmetric=scalVarDiffHorizon.symmetric, normalized=scalVarDiffHorizon.normalized, boundary=True, derivative=scalVarDiffHorizon.derivative, termNo=scalVarDiffHorizon.termNo, horizonFun=scalVarDiffHorizon.horizonFun)
            elif isinstance(self.scalingPrePhi, variableFractionalLaplacianScaling):
                scalVar = self.scalingPrePhi
                scaling = variableFractionalLaplacianScaling(symmetric=scalVar.symmetric, normalized=scalVar.normalized, boundary=True, derivative=scalVar.derivative, termNo=scalVar.termNo)
            else:
                raise NotImplementedError(type(self.scalingPrePhi))

            newKernel = FractionalBoundaryKernel.build(dim=self.dim,
                                                       s=s,
                                                       horizon=deepcopy(self.horizon),
                                                       interaction=None,
                                                       scaling=scaling,
                                                       phi=deepcopy(self.phi),
                                                       piecewise=self.piecewise,
                                                       derivative=self.derivative,
                                                       tempered=self.temperedValue,
                                                       termNo=self.termNo)
        else:
            newKernel = super().getBoundaryKernel(interpolation_nodes, kind)
        self._checkNewKernel(newKernel)
        return newKernel

    def getComplementKernel(self):
        newKernel = FractionalKernel.build(dim=self.dim, s=self.s, horizon=self.horizon, interaction=self.interaction.getComplement(), scaling=self.scalingPrePhi, phi=self.phi, piecewise=self.piecewise, derivative=self.derivative, tempered=self.temperedValue, termNo=self.termNo)
        self._checkNewKernel(newKernel)
        return newKernel

    def getDerivative(self, INDEX_t derivative):
        cdef:
            constantFractionalLaplacianScaling scal
            variableFractionalLaplacianScaling scalVar
            list terms = []
            INDEX_t termNo
        assert not self.boundary
        for termNo in range(derivative+1):
            interaction = deepcopy(self.interaction)
            if isinstance(self.scaling, constantFractionalLaplacianScaling):
                scal = self.scaling
                scaling = constantFractionalLaplacianScaling(dim=scal.dim, s=scal.s, horizon=scal.horizon, normalized=scal.normalized, boundary=scal.boundary, derivative=derivative, tempered=scal.tempered, termNo=termNo)
            elif isinstance(self.scaling, variableFractionalLaplacianScaling):
                scalVar = self.scaling
                scaling = variableFractionalLaplacianScaling(symmetric=scalVar.symmetric, normalized=scalVar.normalized, boundary=scalVar.boundary, derivative=derivative, termNo=termNo)
            else:
                raise NotImplementedError(type(self.scaling))
            terms.append(FractionalKernel(dim=self.dim, s=self.s, horizon=self.horizon, interaction=interaction, scaling=scaling, phi=self.phi, piecewise=False, boundary=self.boundary, derivative=derivative, tempered=self.temperedValue, termNo=termNo))
            self._checkNewKernel(terms[termNo])
        return MultiSingularityFractionalKernel(terms)

    def getDerivativeKernel(self):
        return self.getDerivative(1)

    def getGradientKernel(self):
        return self.getDerivative(1)

    def getHessianKernel(self):
        return self.getDerivative(2)

    def __repr__(self):
        if self.temperedValue != 0.:
            return "{}({}, {}, {}, tempered {})".format(self.__class__.__name__, self.s, repr(self.interaction), self.scaling, self.temperedValue)
        else:
            if self.logSingularityValue == 0.:
                return "{}({}, {}, {})".format(self.__class__.__name__, self.s, repr(self.interaction), self.scaling)
            else:
                return "{}({}, logSingularity={}, {}, {})".format(self.__class__.__name__, self.s, self.logSingularityValue, repr(self.interaction), self.scaling)

    def getLongDescription(self):
        cdef:
            str descr = ''
            dict params = {}
        params['d'] = self.dim
        if self.temperedValue != 0.:
            descr = 'MISSING_DESCRIPTION'
        else:
            if self.logSingularityValue == 0.:
                descr = '\\frac{1}{|x-y|^{d+2s}}'
            else:
                descr = '\\frac{(-log(|x-y|))^{\\beta}}{|x-y|^{d+2s}}'
                params['\\beta'] = self.logSingularityValue
        paramsStr = ', '.join([k+'='+str(v) for k, v in params.items()])
        return self.scaling.getLongDescription() + ' ' + descr + ' ' + self.interaction.getLongDescription() + ', ' + paramsStr

    def __reduce__(self):
        return FractionalKernel, (self.dim, self.s, self.horizon, self.interaction, self.scalingPrePhi, self.phi, self.piecewise, self.boundary, self.derivative, self.temperedValue, self.max_horizon, self.kernelType, self.manifold_dim, self.termNo)

    def __eq__(self, FractionalKernel other):
        if other is None:
            return False
        return (self.dim == other.dim) and (self.s == other.s) and (self.horizon == other.horizon) and (self.interaction == other.interaction) and (self.scalingPrePhi == other.scalingPrePhi) and (self.phi == other.phi) and (self.piecewise == other.piecewise) and (self.boundary == other.boundary) and (self.derivative == other.derivative) and (self.temperedValue == other.temperedValue) and (self.max_horizon == other.max_horizon)


cdef class FractionalBoundaryKernel(FractionalKernel):
    def __init__(self,
                 INDEX_t dim,
                 fractionalOrderBase s,
                 function horizon,
                 interactionDomain interaction,
                 twoPointFunction scaling,
                 twoPointFunction phi=None,
                 BOOL_t piecewise=True,
                 INDEX_t derivative=0,
                 REAL_t tempered=0.,
                 REAL_t max_horizon=np.nan,
                 INDEX_t manifold_dim=-1,
                 INDEX_t termNo=-1):
        super(FractionalBoundaryKernel, self).__init__(dim=dim,
                                                       s=s,
                                                       horizon=horizon,
                                                       interaction=interaction,
                                                       scaling=scaling,
                                                       phi=phi,
                                                       piecewise=piecewise,
                                                       boundary=True,
                                                       derivative=derivative,
                                                       tempered=tempered,
                                                       max_horizon=max_horizon,
                                                       manifold_dim=manifold_dim,
                                                       termNo=termNo)
        if tempered == 0.:
            if derivative == 0:
                self.setKernelFun(monomialKernel)
            elif derivative == 1:
                if termNo == 1:
                    self.min_log_singularity = 1.
                    self.max_log_singularity = 1.
                    self.logSingularityValue = 1.
                    self.setKernelFun(monomialLogKernel)
                elif termNo == 0 or termNo == -1:
                    self.setKernelFun(monomialKernel)
                else:
                    raise NotImplementedError()
            elif derivative == 2:
                if termNo == 2:
                    self.min_log_singularity = 2.
                    self.max_log_singularity = 2.
                    self.logSingularityValue = 2.
                    self.setKernelFun(monomialLogKernel)
                elif termNo == 1:
                    self.min_log_singularity = 1.
                    self.max_log_singularity = 1.
                    self.logSingularityValue = 1.
                    self.setKernelFun(monomialLogKernel)
                elif termNo == 0 or termNo == -1:
                    self.setKernelFun(monomialKernel)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        else:
            self.setKernelFun(temperedMonomialKernelBoundary)
        self.isSphericalManifold = False
        self.n = uninitialized((self.dim), dtype=REAL)

    cdef void setSimplices(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            REAL_t val
        if self.isSphericalManifold:
            self.n[0] = 1.
        elif self.dim == 1:
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

    def __reduce__(self):
        return FractionalBoundaryKernel, (self.dim, self.s, self.horizon, self.interaction, self.scalingPrePhi, self.phi, self.piecewise, self.derivative, self.temperedValue, self.max_horizon, self.manifold_dim, self.termNo)

    @staticmethod
    def build(dim,
              s,
              horizon=None,
              interaction=None,
              twoPointFunction scaling=None,
              BOOL_t normalized=True,
              BOOL_t piecewise=True,
              twoPointFunction phi=None,
              INDEX_t derivative=0,
              REAL_t tempered=0.,
              REAL_t max_horizon=np.nan,
              BOOL_t manifold=False,
              INDEX_t termNo=-1):
        dim_ = _getDim(dim)
        sFun = _getFractionalOrder(s)
        horizonFun = _getHorizon(horizon)
        interaction = _getInteraction(interaction, horizonFun)

        if isinstance(sFun, admissibleSet):
            kernel = RangedFractionalKernel(dim=dim_, admissibleOrders=sFun, horizon=horizonFun, normalized=normalized, tempered=tempered)
        else:
            if scaling is None:
                if isinstance(sFun, constFractionalOrder) and isinstance(horizonFun, constant):
                    scaling = constantFractionalLaplacianScaling(dim=dim-manifold, s=sFun.value, horizon=horizonFun.value, normalized=normalized, boundary=True, derivative=derivative, tempered=tempered, termNo=termNo)
                else:
                    symmetric = sFun.symmetric and isinstance(horizonFun, constant)
                    if piecewise and isinstance(sFun, singleVariableUnsymmetricFractionalOrder):
                        piecewise = False
                    scaling = variableFractionalLaplacianScaling(symmetric, normalized, True, derivative)
            kernel = FractionalBoundaryKernel(dim_, sFun, horizonFun, interaction, scaling, phi=phi, piecewise=piecewise,
                                              derivative=derivative, tempered=tempered, max_horizon=max_horizon, termNo=termNo)

        from . twoPointFunctions import parametrizedTwoPointFunction
        if isinstance(kernel.scaling, parametrizedTwoPointFunction):
            assert kernel.getParamPtrAddr() == kernel.scaling.getParamPtrAddr()
        if isinstance(kernel.interaction, parametrizedTwoPointFunction):
            assert kernel.getParamPtrAddr() == kernel.interaction.getParamPtrAddr()
        return kernel

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* vec):
        cdef:
            REAL_t temp
            INDEX_t i, j, m, k
        if not self.piecewise:
            self.updateParams(dim, x, y)
        temp = self.kernelFun(x, y, self.c_kernel_params)
        temp /= sqrt(self.interaction.dist2)
        if self.derivative == 0:
            vec[0] = 0.
            for i in range(self.dim):
                vec[0] += temp * (x[i]-y[i])*self.n[i]
        elif self.derivative == 1:
            self.s.evalGradPtr(dim, x, y, self.s.numParameters, &vec[0])
            temp2 = 0.
            for j in range(self.dim):
                temp2 += (x[j]-y[j]) * self.n[j]
            for i in range(self.s.numParameters):
                vec[i] *= temp * temp2
        elif self.derivative == 2:
            self.s.evalGradPtr(dim, x, y, self.s.numParameters, &self.tempVec[0])
            temp2 = 0.
            for j in range(self.dim):
                temp2 += (x[j]-y[j]) * self.n[j]
            k = 0
            for i in range(self.s.numParameters):
                for m in range(self.s.numParameters):
                    vec[k] = temp*self.tempVec[i]*self.tempVec[m] * temp2
                    k += 1

    def getBoundaryKernel(self, REAL_t[::1] interpolation_nodes=None, str kind='linear'):
        raise NotImplementedError()


cdef class RangedFractionalKernel(FractionalKernel):
    def __init__(self,
                 INDEX_t dim,
                 admissibleOrders,
                 function horizon,
                 BOOL_t normalized=True,
                 REAL_t tempered=0.,
                 REAL_t errorBound=-1.,
                 INDEX_t M_min=1, INDEX_t M_max=20,
                 REAL_t xi=0.):
        self.dim = dim
        assert admissibleOrders.numParams == 1, "Cannot handle {} params".format(admissibleOrders.numParams)
        self.admissibleOrders = admissibleOrders
        assert isinstance(horizon, constant)
        self.horizon = horizon
        if isinstance(horizon, constant) and horizon.value == np.inf:
            self.interaction = fullSpace()
        else:
            self.interaction = ball2_retriangulation()
        self.normalized = normalized
        self.tempered = tempered

        self.setOrder(admissibleOrders.getLowerBounds()[0])

        self.errorBound = errorBound
        self.M_min = M_min
        self.M_max = M_max
        self.xi = xi

    def setOrder(self, REAL_t s):
        assert self.admissibleOrders.isAdmissible(s)
        sFun = constFractionalOrder(s)
        dim = self.dim
        horizon = self.horizon
        interactionDomain = self.interaction
        tempered = self.tempered
        scaling = constantFractionalLaplacianScaling(dim=dim, s=sFun.value, horizon=horizon.value, normalized=self.normalized, boundary=False, derivative=0, tempered=tempered, termNo=0)
        super(RangedFractionalKernel, self).__init__(dim, sFun, horizon, interactionDomain, scaling, tempered=tempered, termNo=0)

    def getFrozenKernel(self, REAL_t s):
        assert self.admissibleOrders.isAdmissible(s)
        sFun = constFractionalOrder(s)
        dim = self.dim
        horizon = self.horizon
        tempered = self.tempered
        interactionDomain = deepcopy(self.interaction)
        scaling = constantFractionalLaplacianScaling(dim=dim, s=sFun.value, horizon=horizon.value, normalized=self.normalized, boundary=False, derivative=0, tempered=tempered, termNo=0)
        return FractionalKernel(dim, sFun, horizon, interactionDomain, scaling, termNo=scaling.termNo)

    def __repr__(self):
        return 'ranged '+super(RangedFractionalKernel, self).__repr__()

    def __reduce__(self):
        return RangedFractionalKernel, (self.dim, self.admissibleOrders, self.horizon, self.normalized, self.tempered, self.errorBound, self.M_min, self.M_max, self.xi)


cdef class RangedVariableFractionalKernel(FractionalKernel):
    def __init__(self,
                 INDEX_t dim,
                 function blockIndicator,
                 admissibleOrders,
                 function horizon,
                 BOOL_t normalized=True,
                 REAL_t errorBound=-1.,
                 INDEX_t M_min=1, INDEX_t M_max=20,
                 REAL_t xi=0.):
        self.dim = dim
        self.blockIndicator = blockIndicator
        self.admissibleOrders = admissibleOrders

        assert isinstance(horizon, constant)
        self.horizon = horizon
        if isinstance(horizon, constant) and horizon.value == np.inf:
            self.interaction = fullSpace()
        else:
            self.interaction = ball2_retriangulation()
        self.normalized = normalized

        numBlocks = <INDEX_t>np.around(sqrt(admissibleOrders.numParams))
        assert numBlocks*numBlocks == admissibleOrders.numParams
        self.setOrder(admissibleOrders.getLowerBounds().reshape((numBlocks, numBlocks)))

        self.errorBound = errorBound
        self.M_min = M_min
        self.M_max = M_max
        self.xi = xi

    def setOrder(self, REAL_t[:, ::1] sVals):
        assert self.admissibleOrders.isAdmissible(np.array(sVals, copy=False).flatten())
        sFun = piecewiseConstantFractionalOrder(self.dim, self.blockIndicator, sVals)
        dim = self.dim
        horizon = self.horizon
        interactionDomain = self.interaction
        if self.normalized:
            scaling = variableFractionalLaplacianScaling(sFun.symmetric)
        else:
            scaling = constantTwoPoint(1.0)
        super(RangedVariableFractionalKernel, self).__init__(dim, sFun, horizon, interactionDomain, scaling)

    def getFrozenKernel(self, REAL_t[:, ::1] sVals):
        assert self.admissibleOrders.isAdmissible(np.array(sVals, copy=False).flatten())
        sFun = piecewiseConstantFractionalOrder(self.dim, self.blockIndicator, sVals)
        dim = self.dim
        horizon = self.horizon
        interactionDomain = self.interaction
        if self.normalized:
            scaling = variableFractionalLaplacianScaling(sFun.symmetric)
        else:
            scaling = constantTwoPoint(1.0)
        return FractionalKernel(dim, sFun, horizon, interactionDomain, scaling, termNo=scaling.termNo)

    def __repr__(self):
        return 'ranged '+super(RangedVariableFractionalKernel, self).__repr__()


cdef class MultiSingularityFractionalKernel(MultiSingularityKernel):
    def __init__(self, list kernels):
        cdef:
            FractionalKernel kernel
        super(MultiSingularityFractionalKernel, self).__init__(kernels)
        kernel = kernels[0]
        self.variableOrder = kernel.variableOrder
        self.derivative = kernel.derivative
