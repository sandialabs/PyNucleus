###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

            self.qrId = sQR.qr
            self.PHI_id = sQR.PHI3
        elif panel == COMMON_VERTEX:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:

                qr = singularityCancelationQuadRule1D(panel,
                                                      self.singularityCancelationIntegrandAcrossElements+singularityValue,
                                                      self.quad_order_diagonal,
                                                      2*dm_order)
                PHI = uninitialized((2*dofs_per_element - dofs_per_vertex,
                                     qr.num_nodes,
                                     2), dtype=REAL)

                for dof in range(dofs_per_vertex):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_y[0] = qr.nodes[2, i]
                        lcl_bary_y[1] = qr.nodes[3, i]
                        PHI[dof, i, 0] = sf.eval(lcl_bary_x)
                        PHI[dof, i, 1] = sf.eval(lcl_bary_y)

                for dof in range(dofs_per_vertex, dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_y[0] = qr.nodes[2, i]
                        lcl_bary_y[1] = qr.nodes[3, i]
                        PHI[dof, i, 0] = sf.eval(lcl_bary_x)
                        PHI[dof, i, 1] = 0
                        PHI[dofs_per_element+dof-dofs_per_vertex, i, 0] = 0
                        PHI[dofs_per_element+dof-dofs_per_vertex, i, 1] = sf.eval(lcl_bary_y)

                sQR = specialQuadRule(qr, PHI3=PHI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes), dtype={SCALAR})
                    self.temp2 = uninitialized((qr.num_nodes), dtype={SCALAR})
            self.qrVertex = sQR.qr
            self.PHI_vertex = sQR.PHI3
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))


cdef class {SCALAR_label}bem3D({SCALAR_label}bem):
    def __init__(self, {SCALAR_label}Kernel kernel, meshBase mesh, DoFMap DoFMap, num_dofs=None, manifold_dim2=-1, **kwargs):
        super({SCALAR_label}bem3D, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2, **kwargs)

    cdef REAL_t get_h_simplex(self, const REAL_t[:, ::1] simplex):
        cdef:
            INDEX_t i, j
            REAL_t hmax = 0., h2
        for i in range(2):
            for j in range(i+1, 3):
                h2 = ((simplex[j, 0]-simplex[i, 0])*(simplex[j, 0]-simplex[i, 0]) +
                      (simplex[j, 1]-simplex[i, 1])*(simplex[j, 1]-simplex[i, 1]) +
                      (simplex[j, 2]-simplex[i, 2])*(simplex[j, 2]-simplex[i, 2]))
                hmax = max(hmax, h2)
        return sqrt(hmax)

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        cdef:
            panelType panel, panel2
            REAL_t logdh1 = log(d/h1), logdh2 = log(d/h2)
            REAL_t c = (0.5*self.target_order+0.5)*log(self.num_dofs*self.H0**2) #-4.
            REAL_t logh1H0 = abs(log(h1/self.H0)), logh2H0 = abs(log(h2/self.H0))
            REAL_t loghminH0 = max(logh1H0, logh2H0)
            REAL_t s = max(-0.5*(self.kernel.getSingularityValue()+2), 0.)
        panel = <panelType>max(ceil((c + (s-1.)*logh2H0 + loghminH0 - s*logdh2) /
                                    (max(logdh1, 0) + 0.4)),
                               2)
        panel2 = <panelType>max(ceil((c + (s-1.)*logh1H0 + loghminH0 - s*logdh1) /
                                     (max(logdh2, 0) + 0.4)),
                                2)
        panel = max(panel, panel2)
        if self.distantQuadRulesPtr[panel] == NULL:
            self.addQuadRule_nonSym(panel)
        return panel

    cdef void getNearQuadRule(self, panelType panel):
        cdef:
            INDEX_t i
            REAL_t singularityValue = self.kernel.getSingularityValue()
            specialQuadRule sQR
            quadratureRule qr
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t dofs_per_edge = self.DoFMap.dofs_per_edge
            INDEX_t dofs_per_vertex = self.DoFMap.dofs_per_vertex
            INDEX_t dm_order = max(self.DoFMap.polynomialOrder, 1)
            shapeFunction sf
            INDEX_t manifold_dim = 2
            REAL_t lcl_bary_x[3]
            REAL_t lcl_bary_y[3]
            REAL_t[:, :, ::1] PHI
            INDEX_t dof

        if panel == COMMON_FACE:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:

                qr = singularityCancelationQuadRule2D(panel,
                                                      self.singularityCancelationIntegrandWithinElement+singularityValue,
                                                      self.quad_order_diagonal,
                                                      self.quad_order_diagonalV,
                                                      1)
                PHI = uninitialized((dofs_per_element, qr.num_nodes, 2), dtype=REAL)

                for dof in range(dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PHI[dof, i, 0] = sf.eval(lcl_bary_x)
                        PHI[dof, i, 1] = sf.eval(lcl_bary_y)

                sQR = specialQuadRule(qr, PHI3=PHI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes), dtype={SCALAR})
                    self.temp2 = uninitialized((qr.num_nodes), dtype={SCALAR})
            self.qrId = sQR.qr
            self.PHI_id = sQR.PHI3
        elif panel == COMMON_EDGE:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:

                qr = singularityCancelationQuadRule2D(panel,
                                                      self.singularityCancelationIntegrandAcrossElements+singularityValue,
                                                      self.quad_order_diagonal,
                                                      self.quad_order_diagonalV,
                                                      1)
                PHI = uninitialized((2*dofs_per_element - 2*dofs_per_vertex - dofs_per_edge,
                                     qr.num_nodes,
                                     2), dtype=REAL)

                for dof in range(2*dofs_per_vertex):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PHI[dof, i, 0] = sf.eval(lcl_bary_x)
                        PHI[dof, i, 1] = sf.eval(lcl_bary_y)

                for dof in range((manifold_dim+1)*dofs_per_vertex, (manifold_dim+1)*dofs_per_vertex+dofs_per_edge):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PHI[dof, i, 0] = sf.eval(lcl_bary_x)
                        PHI[dof, i, 1] = sf.eval(lcl_bary_y)

                for dof in range(2*dofs_per_vertex, (manifold_dim+1)*dofs_per_vertex):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PHI[dof, i, 0] = sf.eval(lcl_bary_x)
                        PHI[dof, i, 1] = 0
                        PHI[dofs_per_element+dof-2*dofs_per_vertex, i, 0] = 0
                        PHI[dofs_per_element+dof-2*dofs_per_vertex, i, 1] = sf.eval(lcl_bary_y)

                for dof in range((manifold_dim+1)*dofs_per_vertex+dofs_per_edge, dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PHI[dof, i, 0] = sf.eval(lcl_bary_x)
                        PHI[dof, i, 1] = 0
                        PHI[dofs_per_element+dof-2*dofs_per_vertex-dofs_per_edge, i, 0] = 0
                        PHI[dofs_per_element+dof-2*dofs_per_vertex-dofs_per_edge, i, 1] = sf.eval(lcl_bary_y)

                sQR = specialQuadRule(qr, PHI3=PHI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes), dtype={SCALAR})
                    self.temp2 = uninitialized((qr.num_nodes), dtype={SCALAR})
            self.qrEdge = sQR.qr
            self.PHI_edge = sQR.PHI3
        elif panel == COMMON_VERTEX:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:

                qr = singularityCancelationQuadRule2D(panel,
                                                      self.singularityCancelationIntegrandAcrossElements+singularityValue,
                                                      self.quad_order_diagonal,
                                                      self.quad_order_diagonalV,
                                                      1)
                PHI = uninitialized((2*dofs_per_element - dofs_per_vertex,
                                     qr.num_nodes,
                                     2), dtype=REAL)

                for dof in range(dofs_per_vertex):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PHI[dof, i, 0] = sf.eval(lcl_bary_x)
                        PHI[dof, i, 1] = sf.eval(lcl_bary_y)

                for dof in range(dofs_per_vertex, dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PHI[dof, i, 0] = sf.eval(lcl_bary_x)
                        PHI[dof, i, 1] = 0
                        PHI[dofs_per_element+dof-dofs_per_vertex, i, 0] = 0
                        PHI[dofs_per_element+dof-dofs_per_vertex, i, 1] = sf.eval(lcl_bary_y)

                sQR = specialQuadRule(qr, PHI3=PHI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes), dtype={SCALAR})
                    self.temp2 = uninitialized((qr.num_nodes), dtype={SCALAR})
            self.qrVertex = sQR.qr
            self.PHI_vertex = sQR.PHI3
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))


cdef class {SCALAR_label}bem2D_V({SCALAR_label}bem2D):
    """The local stiffness matrix

    .. math::

       \\int_{K_1}\\int_{K_2} 0.5 * [ u(x) v(y) + u(y) v(x) ] \\gamma(x,y) dy dx

    for the V operator.
    """
    def __init__(self,
                 {SCALAR_label}Kernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 quad_order_diagonal=None,
                 target_order=None,
                 num_dofs=None,
                 **kwargs):
        cdef:
            REAL_t smin, smax
        super({SCALAR_label}bem2D_V, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        # The integrand (excluding the kernel) cancels 2 orders of the singularity within an element.
        self.singularityCancelationIntegrandWithinElement = 0.
        # The integrand (excluding the kernel) cancels 2 orders of the
        # singularity across elements for continuous finite elements.
        if isinstance(DoFMap, P0_DoFMap):
            assert self.kernel.max_singularity > -1., "Discontinuous finite elements are not conforming for singularity order {} <= -2.".format(self.kernel.max_singularity)
            self.singularityCancelationIntegrandAcrossElements = 0.
        else:
            self.singularityCancelationIntegrandAcrossElements = 0.

        smin = max(-0.5*(self.kernel.min_singularity+1), 0.)
        smax = max(-0.5*(self.kernel.max_singularity+1), 0.)

        if target_order is None:
            # this is the desired local quadrature error
            target_order = self.DoFMap.polynomialOrder+1-smin
        self.target_order = target_order
        if quad_order_diagonal is None:
            # measured log(2 rho_2) = 0.43
            quad_order_diagonal = max(np.ceil(((target_order+2.)*log(self.num_dofs*self.H0) + (2.*smax-1.)*abs(log(self.hmin/self.H0)))/0.8), 2)
        self.quad_order_diagonal = quad_order_diagonal

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableOrder):
            self.getNearQuadRule(COMMON_EDGE)
            self.getNearQuadRule(COMMON_VERTEX)

    cdef void eval(self,
                   {SCALAR}_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, m, i, j, I, J
            REAL_t vol, vol1 = self.vol1, vol2 = self.vol2
            {SCALAR}_t val
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            quadratureRule qr
            REAL_t[:, :, ::1] PHI
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t dim = 2
            REAL_t x[2]
            REAL_t y[2]

        if panel >= 1:
            self.eval_distant_bem_V(contrib, panel, mask)
            return
        elif panel == COMMON_EDGE:
            qr = self.qrId
            PHI = self.PHI_id
        elif panel == COMMON_VERTEX:
            qr = self.qrVertex
            PHI = self.PHI_vertex
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

        vol = vol1*vol2
        for m in range(qr.num_nodes):
            for j in range(dim):
                x[j] = (simplex1[self.perm1[0], j]*qr.nodes[0, m] +
                        simplex1[self.perm1[1], j]*qr.nodes[1, m])
                y[j] = (simplex2[self.perm2[0], j]*qr.nodes[2, m] +
                        simplex2[self.perm2[1], j]*qr.nodes[3, m])
            self.temp[m] = qr.weights[m] * self.kernel.evalPtr(dim, &x[0], &y[0])

        contrib[:] = 0.
        for I in range(PHI.shape[0]):
            i = self.perm[I]
            for J in range(I, PHI.shape[0]):
                j = self.perm[J]
                if j < i:
                    k = 2*dofs_per_element*j-(j*(j+1) >> 1) + i
                else:
                    k = 2*dofs_per_element*i-(i*(i+1) >> 1) + j
                if mask[k]:
                    val = 0.
                    for m in range(qr.num_nodes):
                        val += self.temp[m] * (PHI[I, m, 0] * PHI[J, m, 1] + PHI[I, m, 1] * PHI[J, m, 0])
                    contrib[k] = 0.5*val*vol


cdef class {SCALAR_label}bem2D_K({SCALAR_label}bem2D):
    """The local stiffness matrix

    .. math::

       \\int_{K_1}\\int_{K_2} 0.5 * [ u(x) v(y) + u(y) v(x) ] \\gamma(x,y) dy dx

    for the V operator.
    """
    def __init__(self,
                 {SCALAR_label}Kernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 quad_order_diagonal=None,
                 target_order=None,
                 num_dofs=None,
                 **kwargs):
        cdef:
            REAL_t smin, smax
        super({SCALAR_label}bem2D_K, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        # The integrand (excluding the kernel) cancels 2 orders of the singularity within an element.
        self.singularityCancelationIntegrandWithinElement = 1.
        # The integrand (excluding the kernel) cancels 2 orders of the
        # singularity across elements for continuous finite elements.
        if isinstance(DoFMap, P0_DoFMap):
            assert self.kernel.max_singularity > -1., "Discontinuous finite elements are not conforming for singularity order {} <= -2.".format(self.kernel.max_singularity)
            self.singularityCancelationIntegrandAcrossElements = 1.
        else:
            self.singularityCancelationIntegrandAcrossElements = 1.

        smin = max(-0.5*(self.kernel.min_singularity+1), 0.)
        smax = max(-0.5*(self.kernel.max_singularity+1), 0.)

        if target_order is None:
            # this is the desired local quadrature error
            target_order = self.DoFMap.polynomialOrder+1-smin
        self.target_order = target_order
        if quad_order_diagonal is None:
            # measured log(2 rho_2) = 0.43
            quad_order_diagonal = max(np.ceil(((target_order+2.)*log(self.num_dofs*self.H0) + (2.*smax-1.)*abs(log(self.hmin/self.H0)))/0.8), 2)
        self.quad_order_diagonal = quad_order_diagonal

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableOrder):
            self.getNearQuadRule(COMMON_EDGE)
            self.getNearQuadRule(COMMON_VERTEX)

    cdef void eval(self,
                   {SCALAR}_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, m, i, j, I, J
            REAL_t vol, valReal, vol1 = self.vol1, vol2 = self.vol2
            {SCALAR}_t val
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            quadratureRule qr
            REAL_t[:, :, ::1] PHI
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t dim = 2
            REAL_t x[2]
            REAL_t y[2]
            REAL_t normW

        if panel >= 1:
            self.eval_distant_bem_K(contrib, panel, mask)
            return
        elif panel == COMMON_EDGE:
            qr = self.qrId
            PHI = self.PHI_id
        elif panel == COMMON_VERTEX:
            qr = self.qrVertex
            PHI = self.PHI_vertex
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

        # n is independent of x and y
        self.n1[0] = simplex1[1, 1] - simplex1[0, 1]
        self.n1[1] = simplex1[0, 0] - simplex1[1, 0]
        valReal = 1./sqrt(mydot(self.n1, self.n1))
        self.n1[0] *= valReal
        self.n1[1] *= valReal

        self.n2[0] = simplex2[1, 1] - simplex2[0, 1]
        self.n2[1] = simplex2[0, 0] - simplex2[1, 0]
        valReal = 1./sqrt(mydot(self.n2, self.n2))
        self.n2[0] *= valReal
        self.n2[1] *= valReal

        vol = vol1*vol2
        for m in range(qr.num_nodes):
            normW = 0.
            for j in range(dim):
                x[j] = (simplex1[self.perm1[0], j]*qr.nodes[0, m] +
                        simplex1[self.perm1[1], j]*qr.nodes[1, m])
                y[j] = (simplex2[self.perm2[0], j]*qr.nodes[2, m] +
                        simplex2[self.perm2[1], j]*qr.nodes[3, m])
                self.w[j] = y[j]-x[j]
                normW += self.w[j]**2
            normW = 1./sqrt(normW)
            for j in range(dim):
                self.w[j] *= normW

            val = qr.weights[m] * self.kernel.evalPtr(dim, &x[0], &y[0])
            self.temp[m] = val * mydot(self.n2, self.w)
            self.temp2[m] = - val * mydot(self.n1, self.w)

        contrib[:] = 0.
        for I in range(PHI.shape[0]):
            i = self.perm[I]
            for J in range(I, PHI.shape[0]):
                j = self.perm[J]
                if j < i:
                    k = 2*dofs_per_element*j-(j*(j+1) >> 1) + i
                else:
                    k = 2*dofs_per_element*i-(i*(i+1) >> 1) + j
                if mask[k]:
                    val = 0.
                    for m in range(qr.num_nodes):
                        val += self.temp[m] * PHI[I, m, 0] * PHI[J, m, 1] + self.temp2[m] * PHI[I, m, 1] * PHI[J, m, 0]
                    contrib[k] = 0.5*val*vol


cdef class {SCALAR_label}bem2D_K_prime({SCALAR_label}bem2D):
    """The local stiffness matrix

    .. math::

       \\int_{K_1}\\int_{K_2} 0.5 * [ u(x) v(y) + u(y) v(x) ] \\gamma(x,y) dy dx

    for the V operator.
    """
    def __init__(self,
                 {SCALAR_label}Kernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 quad_order_diagonal=None,
                 target_order=None,
                 num_dofs=None,
                 **kwargs):
        cdef:
            REAL_t smin, smax
        super({SCALAR_label}bem2D_K_prime, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        # The integrand (excluding the kernel) cancels 2 orders of the singularity within an element.
        self.singularityCancelationIntegrandWithinElement = 1.
        # The integrand (excluding the kernel) cancels 2 orders of the
        # singularity across elements for continuous finite elements.
        if isinstance(DoFMap, P0_DoFMap):
            assert self.kernel.max_singularity > -1., "Discontinuous finite elements are not conforming for singularity order {} <= -2.".format(self.kernel.max_singularity)
            self.singularityCancelationIntegrandAcrossElements = 1.
        else:
            self.singularityCancelationIntegrandAcrossElements = 1.

        smin = max(-0.5*(self.kernel.min_singularity+1), 0.)
        smax = max(-0.5*(self.kernel.max_singularity+1), 0.)

        if target_order is None:
            # this is the desired local quadrature error
            target_order = self.DoFMap.polynomialOrder+1-smin
        self.target_order = target_order
        if quad_order_diagonal is None:
            # measured log(2 rho_2) = 0.43
            quad_order_diagonal = max(np.ceil(((target_order+2.)*log(self.num_dofs*self.H0) + (2.*smax-1.)*abs(log(self.hmin/self.H0)))/0.8), 2)
        self.quad_order_diagonal = quad_order_diagonal

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableOrder):
            self.getNearQuadRule(COMMON_EDGE)
            self.getNearQuadRule(COMMON_VERTEX)

    cdef void eval(self,
                   {SCALAR}_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, m, i, j, I, J
            REAL_t vol, valReal, vol1 = self.vol1, vol2 = self.vol2
            {SCALAR}_t val
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            quadratureRule qr
            REAL_t[:, :, ::1] PHI
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t dim = 2
            REAL_t x[2]
            REAL_t y[2]
            REAL_t normW

        if panel >= 1:
            self.eval_distant_bem_K_prime(contrib, panel, mask)
            return
        elif panel == COMMON_EDGE:
            qr = self.qrId
            PHI = self.PHI_id
        elif panel == COMMON_VERTEX:
            qr = self.qrVertex
            PHI = self.PHI_vertex
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

        # n is independent of x and y
        self.n1[0] = simplex1[1, 1] - simplex1[0, 1]
        self.n1[1] = simplex1[0, 0] - simplex1[1, 0]
        valReal = 1./sqrt(mydot(self.n1, self.n1))
        self.n1[0] *= valReal
        self.n1[1] *= valReal

        self.n2[0] = simplex2[1, 1] - simplex2[0, 1]
        self.n2[1] = simplex2[0, 0] - simplex2[1, 0]
        valReal = 1./sqrt(mydot(self.n2, self.n2))
        self.n2[0] *= valReal
        self.n2[1] *= valReal

        vol = vol1*vol2
        for m in range(qr.num_nodes):
            normW = 0.
            for j in range(dim):
                x[j] = (simplex1[self.perm1[0], j]*qr.nodes[0, m] +
                        simplex1[self.perm1[1], j]*qr.nodes[1, m])
                y[j] = (simplex2[self.perm2[0], j]*qr.nodes[2, m] +
                        simplex2[self.perm2[1], j]*qr.nodes[3, m])
                self.w[j] = y[j]-x[j]
                normW += self.w[j]**2
            normW = 1./sqrt(normW)
            for j in range(dim):
                self.w[j] *= normW

            val = qr.weights[m] * self.kernel.evalPtr(dim, &x[0], &y[0])
            self.temp[m] = - val * mydot(self.n1, self.w)
            self.temp2[m] = val * mydot(self.n2, self.w)

        contrib[:] = 0.
        for I in range(PHI.shape[0]):
            i = self.perm[I]
            for J in range(I, PHI.shape[0]):
                j = self.perm[J]
                if j < i:
                    k = 2*dofs_per_element*j-(j*(j+1) >> 1) + i
                else:
                    k = 2*dofs_per_element*i-(i*(i+1) >> 1) + j
                if mask[k]:
                    val = 0.
                    for m in range(qr.num_nodes):
                        val += self.temp[m] * PHI[I, m, 0] * PHI[J, m, 1] + self.temp2[m] * PHI[I, m, 1] * PHI[J, m, 0]
                    contrib[k] = 0.5*val*vol


cdef class {SCALAR_label}bem3D_V({SCALAR_label}bem3D):
    """The local stiffness matrix

    .. math::

       \\int_{K_1}\\int_{K_2} 0.5 * [ u(x) v(y) + u(y) v(x) ] \\gamma(x,y) dy dx

    for the V operator.
    """
    def __init__(self,
                 {SCALAR_label}Kernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 quad_order_diagonal=None,
                 target_order=None,
                 num_dofs=None,
                 **kwargs):
        cdef:
            REAL_t smin, smax
        super({SCALAR_label}bem3D_V, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        # The integrand (excluding the kernel) cancels 2 orders of the singularity within an element.
        self.singularityCancelationIntegrandWithinElement = 0.
        # The integrand (excluding the kernel) cancels 2 orders of the
        # singularity across elements for continuous finite elements.
        if isinstance(DoFMap, P0_DoFMap):
            assert self.kernel.max_singularity > -3., "Discontinuous finite elements are not conforming for singularity order {} <= -3.".format(self.kernel.max_singularity)
            self.singularityCancelationIntegrandAcrossElements = 0.
        else:
            self.singularityCancelationIntegrandAcrossElements = 0.

        if target_order is None:
            # this is the desired local quadrature error
            # target_order = (2.-s)/self.dim
            target_order = 0.5
        self.target_order = target_order

        smax = max(-0.5*(self.kernel.max_singularity+2), 0.)
        if quad_order_diagonal is None:
            # measured log(2 rho_2) = 0.43
            quad_order_diagonal = max(np.ceil((target_order+1.+smax)/(0.43)*abs(np.log(self.hmin/self.H0))), 4)
            # measured log(2 rho_2) = 0.7
            quad_order_diagonalV = max(np.ceil((target_order+1.+smax)/(0.7)*abs(np.log(self.hmin/self.H0))), 4)
        else:
            quad_order_diagonalV = quad_order_diagonal
        self.quad_order_diagonal = quad_order_diagonal
        self.quad_order_diagonalV = quad_order_diagonalV

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableOrder):
            self.getNearQuadRule(COMMON_FACE)
            self.getNearQuadRule(COMMON_EDGE)
            self.getNearQuadRule(COMMON_VERTEX)

    cdef void eval(self,
                   {SCALAR}_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, m, i, j, I, J
            REAL_t vol, vol1 = self.vol1, vol2 = self.vol2
            {SCALAR}_t val
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            quadratureRule qr
            REAL_t[:, :, ::1] PHI
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t dim = 3
            REAL_t x[3]
            REAL_t y[3]

        if panel >= 1:
            self.eval_distant_bem_V(contrib, panel, mask)
            return
        elif panel == COMMON_FACE:
            qr = self.qrId
            PHI = self.PHI_id
        elif panel == COMMON_EDGE:
            qr = self.qrEdge
            PHI = self.PHI_edge
        elif panel == COMMON_VERTEX:
            qr = self.qrVertex
            PHI = self.PHI_vertex
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

        vol = 4.0*vol1*vol2
        for m in range(qr.num_nodes):
            for j in range(dim):
                x[j] = (simplex1[self.perm1[0], j]*qr.nodes[0, m] +
                        simplex1[self.perm1[1], j]*qr.nodes[1, m] +
                        simplex1[self.perm1[2], j]*qr.nodes[2, m])
                y[j] = (simplex2[self.perm2[0], j]*qr.nodes[3, m] +
                        simplex2[self.perm2[1], j]*qr.nodes[4, m] +
                        simplex2[self.perm2[2], j]*qr.nodes[5, m])
            self.temp[m] = qr.weights[m] * self.kernel.evalPtr(dim, &x[0], &y[0])

        contrib[:] = 0.
        for I in range(PHI.shape[0]):
            i = self.perm[I]
            for J in range(I, PHI.shape[0]):
                j = self.perm[J]
                if j < i:
                    k = 2*dofs_per_element*j-(j*(j+1) >> 1) + i
                else:
                    k = 2*dofs_per_element*i-(i*(i+1) >> 1) + j
                if mask[k]:
                    val = 0.
                    for m in range(qr.num_nodes):
                        val += self.temp[m] * (PHI[I, m, 0] * PHI[J, m, 1] + PHI[I, m, 1] * PHI[J, m, 0])
                    contrib[k] = 0.5*val*vol


cdef class {SCALAR_label}bem3D_K({SCALAR_label}bem3D):
    """The local stiffness matrix

    .. math::

       \\int_{K_1}\\int_{K_2} 0.5 * [ u(x) v(y) + u(y) v(x) ] \\gamma(x,y) dy dx

    for the V operator.
    """
    def __init__(self,
                 {SCALAR_label}Kernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 quad_order_diagonal=None,
                 target_order=None,
                 num_dofs=None,
                 **kwargs):
        cdef:
            REAL_t smin, smax
        super({SCALAR_label}bem3D_K, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        # The integrand (excluding the kernel) cancels 2 orders of the singularity within an element.
        self.singularityCancelationIntegrandWithinElement = 1.
        # The integrand (excluding the kernel) cancels 2 orders of the
        # singularity across elements for continuous finite elements.
        if isinstance(DoFMap, P0_DoFMap):
            assert self.kernel.max_singularity > -3., "Discontinuous finite elements are not conforming for singularity order {} <= -3.".format(self.kernel.max_singularity)
            self.singularityCancelationIntegrandAcrossElements = 1.
        else:
            self.singularityCancelationIntegrandAcrossElements = 1.

        if target_order is None:
            # this is the desired local quadrature error
            # target_order = (2.-s)/self.dim
            target_order = 0.5
        self.target_order = target_order

        smax = max(-0.5*(self.kernel.max_singularity+2), 0.)
        if quad_order_diagonal is None:
            # measured log(2 rho_2) = 0.43
            quad_order_diagonal = max(np.ceil((target_order+1.+smax)/(0.43)*abs(np.log(self.hmin/self.H0))), 4)
            # measured log(2 rho_2) = 0.7
            quad_order_diagonalV = max(np.ceil((target_order+1.+smax)/(0.7)*abs(np.log(self.hmin/self.H0))), 4)
        else:
            quad_order_diagonalV = quad_order_diagonal
        self.quad_order_diagonal = quad_order_diagonal
        self.quad_order_diagonalV = quad_order_diagonalV

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableOrder):
            self.getNearQuadRule(COMMON_FACE)
            self.getNearQuadRule(COMMON_EDGE)
            self.getNearQuadRule(COMMON_VERTEX)

    cdef void eval(self,
                   {SCALAR}_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, m, i, j, I, J
            REAL_t vol, valReal, vol1 = self.vol1, vol2 = self.vol2
            {SCALAR}_t val
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            quadratureRule qr
            REAL_t[:, :, ::1] PHI
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t dim = 3
            REAL_t x[3]
            REAL_t y[3]
            REAL_t normW

        if panel >= 1:
            self.eval_distant_bem_K(contrib, panel, mask)
            return
        elif panel == COMMON_FACE:
            qr = self.qrId
            PHI = self.PHI_id
        elif panel == COMMON_EDGE:
            qr = self.qrEdge
            PHI = self.PHI_edge
        elif panel == COMMON_VERTEX:
            qr = self.qrVertex
            PHI = self.PHI_vertex
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

        for j in range(dim):
            x[j] = simplex1[1, j]-simplex1[0, j]
        for j in range(dim):
            y[j] = simplex1[2, j]-simplex1[0, j]
        self.n1[0] = x[1]*y[2]-x[2]*y[1]
        self.n1[1] = x[2]*y[0]-x[0]*y[2]
        self.n1[2] = x[0]*y[1]-x[1]*y[0]
        valReal = 1./sqrt(mydot(self.n1, self.n1))
        self.n1[0] *= valReal
        self.n1[1] *= valReal
        self.n1[2] *= valReal

        for j in range(dim):
            x[j] = simplex2[1, j]-simplex2[0, j]
        for j in range(dim):
            y[j] = simplex2[2, j]-simplex2[0, j]
        self.n2[0] = x[1]*y[2]-x[2]*y[1]
        self.n2[1] = x[2]*y[0]-x[0]*y[2]
        self.n2[2] = x[0]*y[1]-x[1]*y[0]
        valReal = 1./sqrt(mydot(self.n2, self.n2))
        self.n2[0] *= valReal
        self.n2[1] *= valReal
        self.n2[2] *= valReal

        vol = 4.0*vol1*vol2
        for m in range(qr.num_nodes):
            normW = 0.
            for j in range(dim):
                x[j] = (simplex1[self.perm1[0], j]*qr.nodes[0, m] +
                        simplex1[self.perm1[1], j]*qr.nodes[1, m] +
                        simplex1[self.perm1[2], j]*qr.nodes[2, m])
                y[j] = (simplex2[self.perm2[0], j]*qr.nodes[3, m] +
                        simplex2[self.perm2[1], j]*qr.nodes[4, m] +
                        simplex2[self.perm2[2], j]*qr.nodes[5, m])
                self.w[j] = y[j]-x[j]
                normW += self.w[j]**2

            normW = 1./sqrt(normW)
            for j in range(dim):
                self.w[j] *= normW

            val = qr.weights[m] * self.kernel.evalPtr(dim, &x[0], &y[0])
            self.temp[m] = val * mydot(self.n2, self.w)
            self.temp2[m] = - val * mydot(self.n1, self.w)

        contrib[:] = 0.
        for I in range(PHI.shape[0]):
            i = self.perm[I]
            for J in range(I, PHI.shape[0]):
                j = self.perm[J]
                if j < i:
                    k = 2*dofs_per_element*j-(j*(j+1) >> 1) + i
                else:
                    k = 2*dofs_per_element*i-(i*(i+1) >> 1) + j
                if mask[k]:
                    val = 0.
                    for m in range(qr.num_nodes):
                        val += self.temp[m] * PHI[I, m, 0] * PHI[J, m, 1] + self.temp2[m] * PHI[I, m, 1] * PHI[J, m, 0]
                    contrib[k] = 0.5*val*vol


cdef class {SCALAR_label}bem3D_K_prime({SCALAR_label}bem3D):
    """The local stiffness matrix

    .. math::

       \\int_{K_1}\\int_{K_2} 0.5 * [ u(x) v(y) + u(y) v(x) ] \\gamma(x,y) dy dx

    for the V operator.
    """
    def __init__(self,
                 {SCALAR_label}Kernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 quad_order_diagonal=None,
                 target_order=None,
                 num_dofs=None,
                 **kwargs):
        cdef:
            REAL_t smin, smax
        super({SCALAR_label}bem3D_K_prime, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        # The integrand (excluding the kernel) cancels 2 orders of the singularity within an element.
        self.singularityCancelationIntegrandWithinElement = 1.
        # The integrand (excluding the kernel) cancels 2 orders of the
        # singularity across elements for continuous finite elements.
        if isinstance(DoFMap, P0_DoFMap):
            assert self.kernel.max_singularity > -3., "Discontinuous finite elements are not conforming for singularity order {} <= -3.".format(self.kernel.max_singularity)
            self.singularityCancelationIntegrandAcrossElements = 1.
        else:
            self.singularityCancelationIntegrandAcrossElements = 1.

        if target_order is None:
            # this is the desired local quadrature error
            # target_order = (2.-s)/self.dim
            target_order = 0.5
        self.target_order = target_order

        smax = max(-0.5*(self.kernel.max_singularity+2), 0.)
        if quad_order_diagonal is None:
            # measured log(2 rho_2) = 0.43
            quad_order_diagonal = max(np.ceil((target_order+1.+smax)/(0.43)*abs(np.log(self.hmin/self.H0))), 4)
            # measured log(2 rho_2) = 0.7
            quad_order_diagonalV = max(np.ceil((target_order+1.+smax)/(0.7)*abs(np.log(self.hmin/self.H0))), 4)
        else:
            quad_order_diagonalV = quad_order_diagonal
        self.quad_order_diagonal = quad_order_diagonal
        self.quad_order_diagonalV = quad_order_diagonalV

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableOrder):
            self.getNearQuadRule(COMMON_FACE)
            self.getNearQuadRule(COMMON_EDGE)
            self.getNearQuadRule(COMMON_VERTEX)

    cdef void eval(self,
                   {SCALAR}_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, m, i, j, I, J
            REAL_t vol, valReal, vol1 = self.vol1, vol2 = self.vol2
            {SCALAR}_t val
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            quadratureRule qr
            REAL_t[:, :, ::1] PHI
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t dim = 3
            REAL_t x[3]
            REAL_t y[3]
            REAL_t normW

        if panel >= 1:
            self.eval_distant_bem_K_prime(contrib, panel, mask)
            return
        elif panel == COMMON_FACE:
            qr = self.qrId
            PHI = self.PHI_id
        elif panel == COMMON_EDGE:
            qr = self.qrEdge
            PHI = self.PHI_edge
        elif panel == COMMON_VERTEX:
            qr = self.qrVertex
            PHI = self.PHI_vertex
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

        for j in range(dim):
            x[j] = simplex1[1, j]-simplex1[0, j]
        for j in range(dim):
            y[j] = simplex1[2, j]-simplex1[0, j]
        self.n1[0] = x[1]*y[2]-x[2]*y[1]
        self.n1[1] = x[2]*y[0]-x[0]*y[2]
        self.n1[2] = x[0]*y[1]-x[1]*y[0]
        valReal = 1./sqrt(mydot(self.n1, self.n1))
        self.n1[0] *= valReal
        self.n1[1] *= valReal
        self.n1[2] *= valReal

        for j in range(dim):
            x[j] = simplex2[1, j]-simplex2[0, j]
        for j in range(dim):
            y[j] = simplex2[2, j]-simplex2[0, j]
        self.n2[0] = x[1]*y[2]-x[2]*y[1]
        self.n2[1] = x[2]*y[0]-x[0]*y[2]
        self.n2[2] = x[0]*y[1]-x[1]*y[0]
        valReal = 1./sqrt(mydot(self.n2, self.n2))
        self.n2[0] *= valReal
        self.n2[1] *= valReal
        self.n2[2] *= valReal

        vol = 4.0*vol1*vol2
        for m in range(qr.num_nodes):
            normW = 0.
            for j in range(dim):
                x[j] = (simplex1[self.perm1[0], j]*qr.nodes[0, m] +
                        simplex1[self.perm1[1], j]*qr.nodes[1, m] +
                        simplex1[self.perm1[2], j]*qr.nodes[2, m])
                y[j] = (simplex2[self.perm2[0], j]*qr.nodes[3, m] +
                        simplex2[self.perm2[1], j]*qr.nodes[4, m] +
                        simplex2[self.perm2[2], j]*qr.nodes[5, m])
                self.w[j] = y[j]-x[j]
                normW += self.w[j]**2

            normW = 1./sqrt(normW)
            for j in range(dim):
                self.w[j] *= normW

            val = qr.weights[m] * self.kernel.evalPtr(dim, &x[0], &y[0])
            self.temp[m] = - val * mydot(self.n1, self.w)
            self.temp2[m] = val * mydot(self.n2, self.w)

        contrib[:] = 0.
        for I in range(PHI.shape[0]):
            i = self.perm[I]
            for J in range(I, PHI.shape[0]):
                j = self.perm[J]
                if j < i:
                    k = 2*dofs_per_element*j-(j*(j+1) >> 1) + i
                else:
                    k = 2*dofs_per_element*i-(i*(i+1) >> 1) + j
                if mask[k]:
                    val = 0.
                    for m in range(qr.num_nodes):
                        val += self.temp[m] * PHI[I, m, 0] * PHI[J, m, 1] + self.temp2[m] * PHI[I, m, 1] * PHI[J, m, 0]
                    contrib[k] = 0.5*val*vol


cdef class {SCALAR_label}boundaryIntegralSingleLayer({function_type}):
    cdef:
        {SCALAR_label_lc_}fe_vector u
        {SCALAR_label}twoPointFunction kernel
        REAL_t[:, ::1] simplex
        REAL_t[:, ::1] span
        simplexQuadratureRule qr
        REAL_t[:, ::1] PHI
        {SCALAR}_t[::1] fvals

    def __init__(self, {SCALAR_label_lc_}fe_vector u, {SCALAR_label}twoPointFunction kernel):
        self.u = u
        self.kernel = kernel
        dm = u.dm
        mesh = dm.mesh

        dim = mesh.dim
        dimManifold = mesh.manifold_dim

        self.simplex = uninitialized((dimManifold+1, dim), dtype=REAL)
        self.span = uninitialized((dimManifold, dim), dtype=REAL)
        self.qr = simplexXiaoGimbutas(dim=dim, manifold_dim=dimManifold, order=3)

        self.PHI = uninitialized((dm.dofs_per_element, self.qr.num_nodes), dtype=REAL)
        for i in range(dm.dofs_per_element):
            for j in range(self.qr.num_nodes):
                self.PHI[i, j] = dm.localShapeFunctions[i](np.ascontiguousarray(self.qr.nodes[:, j]))

        self.fvals = uninitialized((self.qr.num_nodes), dtype={SCALAR})

    cdef {SCALAR}_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t cellNo
            {SCALAR}_t val = 0.
            DoFMap dm = self.u.dm
            meshBase mesh = dm.mesh
            INDEX_t dim = mesh.dim
            INDEX_t manifold_dim = mesh.manifold_dim
            simplexQuadratureRule qr = self.qr
            INDEX_t num_quad_nodes = qr.num_nodes
            INDEX_t k, j, I, m
            REAL_t vol
            {SCALAR}_t[::1] u = self.u

        for cellNo in range(mesh.num_cells):
            mesh.getSimplex(cellNo, self.simplex)

            # Calculate volume
            vol = qr.getSimplexVolume(self.simplex)

            for j in range(num_quad_nodes):
                qr.tempVec[:] = 0.
                for k in range(manifold_dim+1):
                    for m in range(dim):
                        qr.tempVec[m] += qr.nodes[k, j] * self.simplex[k, m]
                self.fvals[j] = self.kernel.evalPtr(dim, &x[0], &qr.tempVec[0])

            for k in range(dm.dofs_per_element):
                I = dm.cell2dof(cellNo, k)
                if I < 0:
                    continue
                for j in range(num_quad_nodes):
                    val += vol*qr.weights[j]*self.fvals[j]*self.PHI[k, j]*u[I]
        return val


cdef class {SCALAR_label}boundaryIntegralDoubleLayer({function_type}):
    cdef:
        {SCALAR_label_lc_}fe_vector u
        {SCALAR_label}twoPointFunction kernel
        REAL_t[:, ::1] simplex
        REAL_t[:, ::1] span
        simplexQuadratureRule qr
        REAL_t[:, ::1] PHI
        {SCALAR}_t[::1] fvals
        REAL_t[::1] n2, w, x, y

    def __init__(self, {SCALAR_label_lc_}fe_vector u, twoPointFunction kernel):
        self.u = u
        self.kernel = kernel
        dm = u.dm
        mesh = dm.mesh

        dim = mesh.dim
        dimManifold = mesh.manifold_dim

        self.simplex = uninitialized((dimManifold+1, dim), dtype=REAL)
        self.span = uninitialized((dimManifold, dim), dtype=REAL)
        self.qr = simplexXiaoGimbutas(dim=dim, manifold_dim=dimManifold, order=3)

        self.PHI = uninitialized((dm.dofs_per_element, self.qr.num_nodes), dtype=REAL)
        for i in range(dm.dofs_per_element):
            for j in range(self.qr.num_nodes):
                self.PHI[i, j] = dm.localShapeFunctions[i](np.ascontiguousarray(self.qr.nodes[:, j]))

        self.fvals = uninitialized((self.qr.num_nodes), dtype={SCALAR})
        self.n2 = uninitialized((dim), dtype=REAL)
        self.w = uninitialized((dim), dtype=REAL)
        self.x = uninitialized((dim), dtype=REAL)
        self.y = uninitialized((dim), dtype=REAL)

    cdef {SCALAR}_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t cellNo
            {SCALAR}_t val = 0.
            DoFMap dm = self.u.dm
            meshBase mesh = dm.mesh
            INDEX_t dim = mesh.dim
            INDEX_t manifold_dim = mesh.manifold_dim
            simplexQuadratureRule qr = self.qr
            INDEX_t num_quad_nodes = qr.num_nodes
            INDEX_t k, j, I, m
            REAL_t vol, normW
            {SCALAR}_t[::1] u = self.u

        for cellNo in range(mesh.num_cells):
            mesh.getSimplex(cellNo, self.simplex)

            if dim == 2:
                self.n2[0] = self.simplex[1, 1] - self.simplex[0, 1]
                self.n2[1] = self.simplex[0, 0] - self.simplex[1, 0]
                normW = 1./sqrt(mydot(self.n2, self.n2))
                self.n2[0] *= normW
                self.n2[1] *= normW
            elif dim == 3:
                for j in range(dim):
                    self.x[j] = self.simplex[1, j]-self.simplex[0, j]
                for j in range(dim):
                    self.y[j] = self.simplex[2, j]-self.simplex[0, j]
                self.n2[0] = self.x[1]*self.y[2]-self.x[2]*self.y[1]
                self.n2[1] = self.x[2]*self.y[0]-self.x[0]*self.y[2]
                self.n2[2] = self.x[0]*self.y[1]-self.x[1]*self.y[0]
                normW = 1./sqrt(mydot(self.n2, self.n2))
                self.n2[0] *= normW
                self.n2[1] *= normW
                self.n2[2] *= normW

            # Calculate volume
            vol = qr.getSimplexVolume(self.simplex)

            for j in range(num_quad_nodes):
                qr.tempVec[:] = 0.
                for k in range(manifold_dim+1):
                    for m in range(dim):
                        qr.tempVec[m] += qr.nodes[k, j] * self.simplex[k, m]
                normW = 0.
                for m in range(dim):
                    self.w[m] = qr.tempVec[m]-x[m]
                    normW += self.w[m]**2
                normW = 1./sqrt(normW)
                for m in range(dim):
                    self.w[m] *= normW

                self.fvals[j] = self.kernel.evalPtr(dim, &x[0], &qr.tempVec[0]) * mydot(self.n2, self.w)

            for k in range(dm.dofs_per_element):
                I = dm.cell2dof(cellNo, k)
                if I < 0:
                    continue
                for j in range(num_quad_nodes):
                    val += vol*qr.weights[j]*self.fvals[j]*self.PHI[k, j]*u[I]
        return val
