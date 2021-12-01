###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes import INDEX, REAL
from PyNucleus_base import uninitialized, uninitialized_like
from PyNucleus_fem.meshCy cimport meshBase
from PyNucleus_fem.DoFMaps cimport DoFMap, P1_DoFMap, P0_DoFMap, shapeFunction

import numpy as np
cimport numpy as np
from libc.math cimport pow, sqrt, fabs as abs, log, ceil
cimport cython
from . nonlocalLaplacianBase import ALL

include "config.pxi"
include "panelTypes.pxi"

cdef INDEX_t MAX_INT = np.iinfo(INDEX).max



cdef class integrable1D(nonlocalLaplacian1D):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap DoFMap, num_dofs=None, manifold_dim2=-1, target_order=None, **kwargs):
        super(integrable1D, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2, **kwargs)

        assert isinstance(DoFMap, P1_DoFMap)

        if target_order is None:
            self.target_order = 3.0
        else:
            self.target_order = target_order
        quad_order_diagonal = None
        if quad_order_diagonal is None:
            alpha = self.kernel.singularityValue
            if alpha == 0.:
                quad_order_diagonal = max(ceil(self.target_order), 2)
            else:
                # measured log(2 rho_2) = 0.43
                quad_order_diagonal = max(np.ceil(((self.target_order+2.)*log(self.num_dofs*self.H0) + (-2.-self.kernel.max_singularity)*abs(log(self.hmin/self.H0)))/0.8), 2)
        self.quad_order_diagonal = quad_order_diagonal

        self.x = uninitialized((0, self.dim))
        self.y = uninitialized((0, self.dim))
        self.temp = uninitialized((0), dtype=REAL)
        self.temp2 = uninitialized((0), dtype=REAL)

        self.idx = uninitialized((3), dtype=INDEX)

        if not self.kernel.variable:
            self.getNearQuadRule(COMMON_EDGE)
            self.getNearQuadRule(COMMON_VERTEX)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        cdef:
            panelType panel, panel2
            REAL_t logdh1 = log(d/h1), logdh2 = log(d/h2)
            REAL_t alpha = self.kernel.getSingularityValue()
        if alpha == 0.:
            panel = <panelType>max(ceil(self.target_order), 2)
        else:
            panel = <panelType>max(ceil(((self.target_order+2.)*log(self.num_dofs*self.H0) + (-alpha-2.)*abs(log(h2/self.H0)) + (alpha+1.)*logdh2) /
                                        (max(logdh1, 0) + 0.8)),
                                   2)
            panel2 = <panelType>max(ceil(((self.target_order+2.)*log(self.num_dofs*self.H0) + (-alpha-2.)*abs(log(h1/self.H0)) + (alpha+1)*logdh1) /
                                         (max(logdh2, 0) + 0.8)),
                                    2)
            panel = max(panel, panel2)
        try:
            self.distantQuadRules[panel]
        except KeyError:
            self.addQuadRule(panel)
        return panel

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void addQuadRule(self, panelType panel):
        cdef:
            simplexQuadratureRule qr
            doubleSimplexQuadratureRule qr2
            REAL_t[:, ::1] PSI
            INDEX_t I, k, i, j
            shapeFunction sf
        qr = simplexXiaoGimbutas(panel, self.dim)
        qr2 = doubleSimplexQuadratureRule(qr, qr)
        PSI = uninitialized((2*self.DoFMap.dofs_per_element,
                             qr2.num_nodes), dtype=REAL)
        # phi_i(x) - phi_i(y) = phi_i(x) for i = 0,1
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(qr2.rule1.num_nodes):
                for j in range(qr2.rule2.num_nodes):
                    PSI[I, k] = sf(qr2.rule1.nodes[:, i])
                    k += 1
        # phi_i(x) - phi_i(y) = -phi_i(y) for i = 2,3
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(qr2.rule1.num_nodes):
                for j in range(qr2.rule2.num_nodes):
                    PSI[I+self.DoFMap.dofs_per_element, k] = -sf(qr2.rule2.nodes[:, j])
                    k += 1
        sQR = specialQuadRule(qr2, PSI)
        self.distantQuadRules[panel] = sQR
        self.distantQuadRulesPtr[panel] = <void*>(self.distantQuadRules[panel])

        if qr2.rule1.num_nodes > self.x.shape[0]:
            self.x = uninitialized((qr2.rule1.num_nodes, self.dim), dtype=REAL)
        if qr2.rule2.num_nodes > self.y.shape[0]:
            self.y = uninitialized((qr2.rule2.num_nodes, self.dim), dtype=REAL)
        if qr2.num_nodes > self.temp.shape[0]:
            self.temp = uninitialized((qr2.num_nodes), dtype=REAL)

    cdef void getNearQuadRule(self, panelType panel):
        cdef:
            INDEX_t i
            REAL_t alpha = self.kernel.getSingularityValue()
            REAL_t eta0, eta1
            specialQuadRule sQR0, sQR1

        if panel == COMMON_EDGE:
            try:
                sQR0 = self.specialQuadRules[(alpha, panel, 0)]
            except KeyError:
                qrId = GaussJacobi(((1, 1, 2+alpha),
                                    (1, 0, 0)))

                PSI_id = uninitialized((self.DoFMap.dofs_per_element, qrId.num_nodes), dtype=REAL)
                # COMMON_FACE panels
                for i in range(qrId.num_nodes):
                    eta0 = qrId.nodes[0, i]
                    eta1 = qrId.nodes[1, i]

                    # P0

                    # phi_1(x) = 1
                    # phi_1(y) = 1
                    # psi_1(x, y) = (phi_1(x)-phi_1(y))/(x-y) = 0

                    # P1

                    # phi_1(x) = 1-x
                    # phi_2(x) = x
                    # phi_1(y) = 1-y
                    # phi_2(y) = y
                    # psi_1(x, y) = (phi_1(x)-phi_1(y))/(x-y) = -1
                    # psi_2(x, y) = (phi_2(x)-phi_2(y))/(x-y) = 1

                    # x = 1-eta0+eta0*eta1
                    # PHI_id[0, 0, i] = 1.-x
                    # PHI_id[0, 1, i] = x

                    # y = eta0*eta1
                    # PHI_id[2, i] = 1.-y
                    # PHI_id[3, i] = y

                    PSI_id[0, i] = -1                      # ((1-x)-(1-y))/(1-eta0)
                    PSI_id[1, i] = 1                       # (x-y)/(1-eta0)
                sQR0 = specialQuadRule(qrId, PSI_id)
                self.specialQuadRules[(alpha, panel, 0)] = sQR0
                if qrId.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrId.num_nodes), dtype=REAL)
                    self.temp2 = uninitialized_like(self.temp)
            self.qrId = sQR0.qr
            self.PSI_id = sQR0.PSI
        elif panel == COMMON_VERTEX:
            try:
                sQR0 = self.specialQuadRules[(alpha, panel, 0)]
                sQR1 = self.specialQuadRules[(alpha, panel, 1)]
            except KeyError:

                qrVertex0 = GaussJacobi(((1, 3+alpha, 0),
                                         (self.quad_order_diagonal, 0, 0)))
                qrVertex1 = GaussJacobi(((self.quad_order_diagonal, 1, 0),
                                         (self.quad_order_diagonal, 0, 0)))

                PSI_vertex0 = uninitialized((2*self.DoFMap.dofs_per_element - self.DoFMap.dofs_per_vertex, qrVertex0.num_nodes), dtype=REAL)
                PSI_vertex1 = uninitialized((2*self.DoFMap.dofs_per_element - self.DoFMap.dofs_per_vertex, qrVertex1.num_nodes), dtype=REAL)

                # panels with common vertex
                # first integral
                for i in range(qrVertex0.num_nodes):
                    eta0 = qrVertex0.nodes[0, i]
                    eta1 = qrVertex0.nodes[1, i]

                    # x = eta0*eta1
                    # y = eta0*(1.-eta1)

                    # P0

                    # phi_1(x) = 1
                    # phi_2(x) = 0
                    # phi_1(y) = 0
                    # phi_2(y) = 1
                    # psi_1(x, y) = (phi_1(x)-phi_1(y))/(x-y) = 1/(x+y)  = 1/eta0
                    # psi_2(x, y) = (phi_2(x)-phi_2(y))/(x-y) = -1/(x+y) = -1/eta0

                    # => P0 should not be used for singular kernels, since phi_i(x)-phi_i(y) does not cancel any singular behavior

                    # P1

                    # x     y
                    # <- | ->
                    # [2 1 3]

                    # phi_1(x) = x
                    # phi_2(x) = 1-x
                    # phi_3(x) = 0
                    # phi_1(y) = 0
                    # phi_2(y) = 1-y
                    # phi_3(y) = y
                    # psi_1(x, y) = (phi_1(x)-phi_1(y))/(x-y) = (x)/(x+y)           = eta1
                    # psi_2(x, y) = (phi_2(x)-phi_2(y))/(x-y) = ((1-x)-(1-y))/(x+y) = 1-2*eta1
                    # psi_3(x, y) = (phi_2(x)-phi_2(y))/(x-y) = (-y)/(x+y)          = eta1-1

                    # x = eta0*eta1
                    # PHI_vertex0[0, i] = x
                    # PHI_vertex0[1, i] = 1.-x

                    # y = eta0*(1.-eta1)
                    # PHI_vertex0[2, i] = 1.-y
                    # PHI_vertex0[3, i] = y

                    PSI_vertex0[0, i] = eta1               # (x)/eta0
                    PSI_vertex0[1, i] = 1.-2.*eta1         # ((1-x)-(1-y))/eta0
                    PSI_vertex0[2, i] = eta1-1.            # (-y)/eta0
                # second integral
                for i in range(qrVertex1.num_nodes):
                    eta0 = qrVertex1.nodes[0, i]
                    eta1 = qrVertex1.nodes[1, i]
                    # x = 1-eta0+eta0*eta1
                    # PHI_vertex1[0, i] = x
                    # PHI_vertex1[1, i] = 1.-x

                    # y = 1.-eta0*eta1
                    # PHI_vertex1[2, i] = 1.-y
                    # PHI_vertex1[3, i] = y

                    PSI_vertex1[0, i] = 1.-eta0+eta0*eta1  # x
                    PSI_vertex1[1, i] = eta0*(1.-2.*eta1)  # (1-x)-(1-y)
                    PSI_vertex1[2, i] = eta0*eta1-1.       # -y

                sQR0 = specialQuadRule(qrVertex0, PSI_vertex0)
                sQR1 = specialQuadRule(qrVertex1, PSI_vertex1)
                self.specialQuadRules[(alpha, panel, 0)] = sQR0
                self.specialQuadRules[(alpha, panel, 1)] = sQR1
                if qrVertex0.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrVertex0.num_nodes), dtype=REAL)
                    self.temp2 = uninitialized_like(self.temp)
                if qrVertex1.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrVertex1.num_nodes), dtype=REAL)
                    self.temp2 = uninitialized_like(self.temp)
            self.qrVertex0 = sQR0.qr
            self.PSI_vertex0 = sQR0.PSI
            self.qrVertex1 = sQR1.qr
            self.PSI_vertex1 = sQR1.PSI
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

    def __repr__(self):
        return (super(integrable1D, self).__repr__() +
                'hmin:                          {:.3}\n'.format(self.hmin) +
                'H0:                            {:.3}\n'.format(self.H0) +
                'target order:                  {}\n'.format(self.target_order))

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J, t
            REAL_t vol, val, vol1 = self.vol1, vol2 = self.vol2
            REAL_t alpha = self.kernel.getSingularityValue()
            INDEX_t[::1] idx = self.idx
            doubleSimplexQuadratureRule qr2
            REAL_t[:, ::1] PSI
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            transformQuadratureRule qr0trans, qr1trans
            INDEX_t dofs_per_element, numQuadNodes0, numQuadNodes1
            REAL_t c1, c2, PSI_I, PSI_J
            REAL_t a_b1[2]
            REAL_t a_b2[2]
            REAL_t a_A1[2][2]
            REAL_t a_A2[2][2]
            REAL_t[::1] b1, b2
            REAL_t[:, ::1] A1, A2
            BOOL_t cutElements = False

        if self.kernel.finiteHorizon and panel >= 1 :
            # check if the horizon might cut the elements
            if self.kernel.interaction.relPos == CUT:
                cutElements = True
            if self.kernel.complement:
                cutElements = False
                # TODO: cutElements should be set to True, but
                #       need to figure out the element
                #       transformation.

        if panel == COMMON_EDGE:
            # # exact value:
            # val = scaling * vol1**(1.-2.*s)/(1.-s)/(3.-2.*s)
            # contrib[0] = 0.
            # contrib[1] = 0.
            # contrib[2] = 0.
            # contrib[3] = 0.
            # contrib[4] = 0.
            # contrib[5] = 0.
            # contrib[6] = 0.
            # contrib[7] = val
            # contrib[8] = -val
            # contrib[9] = val

            # factor 2 comes from symmetric contributions
            vol = self.kernel.getScalingValue() * 2.0*vol1**2

            contrib[:] = 0.
            # distance between x and y quadrature nodes
            for i in range(self.qrId.num_nodes):
                self.temp[i] = (simplex1[0, 0]*self.PSI_id[0, i] +
                                simplex1[1, 0]*self.PSI_id[1, i])**2
                self.temp[i] = self.qrId.weights[i]*pow(self.temp[i], 0.5*alpha)
            for I in range(2):
                for J in range(I, 2):
                    k = 4*I-(I*(I+1) >> 1) + J
                    if mask & (1 << k):
                        val = 0.
                        for i in range(self.qrId.num_nodes):
                            val += (self.temp[i] *
                                    self.PSI_id[I, i] *
                                    self.PSI_id[J, i])
                        contrib[k] += val*vol
        elif panel == COMMON_VERTEX:
            vol = self.kernel.getScalingValue() * vol1*vol2

            contrib[:] = 0.

            i = 0
            j = 0
            for k in range(4):
                if self.cells1[self.cellNo1, i] == self.cells2[self.cellNo2, j]:
                    break
                elif j == 1:
                    j = 0
                    i += 1
                else:
                    j += 1
            if i == 1 and j == 0:
                idx[0], idx[1], idx[2] = 0, 1, 2
                t = 2
            elif i == 0 and j == 1:
                idx[0], idx[1], idx[2] = 1, 0, 2
                t = 3
            else:
                raise IndexError('COMMON_VERTEX')

            # loop over all local DoFs
            for I in range(3):
                for J in range(I, 3):
                    i = 3*(I//t)+(I%t)
                    j = 3*(J//t)+(J%t)
                    if j < i:
                        i, j = j, i
                    k = 4*i-(i*(i+1) >> 1) + j
                    if mask & (1 << k):
                        val = 0.
                        for i in range(self.qrVertex0.num_nodes):
                            val += (self.qrVertex0.weights[i] *
                                    self.PSI_vertex0[idx[I], i] *
                                    self.PSI_vertex0[idx[J], i] *
                                    pow(vol1*self.PSI_vertex0[0, i]-vol2*self.PSI_vertex0[2, i], alpha))
                        for i in range(self.qrVertex1.num_nodes):
                            val += (self.qrVertex1.weights[i] *
                                    self.PSI_vertex1[idx[I], i] *
                                    self.PSI_vertex1[idx[J], i] *
                                    pow(vol1*self.PSI_vertex1[0, i]-vol2*self.PSI_vertex1[2, i], alpha))
                        contrib[k] += val*vol
        elif panel >= 1 and not cutElements:
            sQR = <specialQuadRule>(self.distantQuadRulesPtr[panel])
            qr2 = <doubleSimplexQuadratureRule>(sQR.qr)
            PSI = sQR.PSI
            qr2.rule1.nodesInGlobalCoords(simplex1, self.x)
            qr2.rule2.nodesInGlobalCoords(simplex2, self.y)
            k = 0
            for i in range(qr2.rule1.num_nodes):
                for j in range(qr2.rule2.num_nodes):
                    self.temp[k] = qr2.weights[k]*self.kernel.evalPtr(1,
                                                                      &self.x[i, 0],
                                                                      &self.y[j, 0])
                    k += 1

            vol = vol1*vol2
            k = 0
            for I in range(2*self.DoFMap.dofs_per_element):
                for J in range(I, 2*self.DoFMap.dofs_per_element):
                    if mask & (1 << k):
                        val = 0.
                        for i in range(qr2.num_nodes):
                            val += self.temp[i]*PSI[I, i]*PSI[J, i]
                        contrib[k] = val*vol
                    k += 1
        elif panel >= 1 and cutElements:
            sQR = <specialQuadRule>(self.distantQuadRulesPtr[panel])
            qr2 = <doubleSimplexQuadratureRule>(sQR.qr)
            if sQR.qrTransformed0 is not None:
                qr0trans = sQR.qrTransformed0
            else:
                qr0 = qr2.rule1
                qr0trans = transformQuadratureRule(qr0)
                sQR.qrTransformed0 = qr0trans
            if sQR.qrTransformed1 is not None:
                qr1trans = sQR.qrTransformed1
            else:
                qr1 = qr2.rule2
                qr1trans = transformQuadratureRule(qr1)
                sQR.qrTransformed1 = qr1trans
            numQuadNodes0 = qr0trans.num_nodes
            numQuadNodes1 = qr1trans.num_nodes

            contrib[:] = 0.

            vol = vol1*vol2
            dofs_per_element = self.DoFMap.dofs_per_element

            A1 = a_A1
            b1 = a_b1
            A2 = a_A2
            b2 = a_b2

            self.kernel.interaction.startLoopSubSimplices_Simplex(simplex1, simplex2)
            while self.kernel.interaction.nextSubSimplex_Simplex(A1, b1, &c1):
                qr0trans.setBaryTransform(A1, b1)
                qr0trans.nodesInGlobalCoords(simplex1, self.x)
                for i in range(qr0trans.num_nodes):
                    self.kernel.interaction.startLoopSubSimplices_Node(self.x[i, :], simplex2)
                    while self.kernel.interaction.nextSubSimplex_Node(A2, b2, &c2):
                        qr1trans.setBaryTransform(A2, b2)
                        qr1trans.nodesInGlobalCoords(simplex2, self.y)
                        for j in range(qr1trans.num_nodes):
                            val = qr0trans.weights[i]*qr1trans.weights[j]*self.kernel.evalPtr(1, &self.x[i, 0], &self.y[j, 0])
                            val *= c1 * c2 * vol
                            k = 0
                            for I in range(2*dofs_per_element):
                                if I < dofs_per_element:
                                    PSI_I = self.getLocalShapeFunction(I).evalStrided(&qr0trans.nodes[0, i], numQuadNodes0)
                                else:
                                    PSI_I = -self.getLocalShapeFunction(I-dofs_per_element).evalStrided(&qr1trans.nodes[0, j], numQuadNodes1)
                                for J in range(I, 2*dofs_per_element):
                                    if mask & (1 << k):
                                        if J < dofs_per_element:
                                            PSI_J = self.getLocalShapeFunction(J).evalStrided(&qr0trans.nodes[0, i], numQuadNodes0)
                                        else:
                                            PSI_J = -self.getLocalShapeFunction(J-dofs_per_element).evalStrided(&qr1trans.nodes[0, j], numQuadNodes1)
                                        contrib[k] += val * PSI_I*PSI_J
                                    k += 1

        else:
            print(np.array(simplex1), np.array(simplex2))
            raise NotImplementedError('Unknown panel type: {}'.format(panel))


cdef class integrable2D(nonlocalLaplacian2D):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap DoFMap, quad_order_diagonal=None, num_dofs=None, manifold_dim2=-1, target_order=None, **kwargs):
        super(integrable2D, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2, **kwargs)

        assert isinstance(DoFMap, P1_DoFMap)

        if target_order is None:
            self.target_order = 3.0
        else:
            self.target_order = target_order

        if quad_order_diagonal is None:
            alpha = self.kernel.singularityValue
            if alpha == 0.:
                quad_order_diagonal = max(ceil(self.target_order), 2)
                quad_order_diagonalV = max(ceil(self.target_order), 2)
            else:
                # measured log(2 rho_2) = 0.43
                quad_order_diagonal = max(np.ceil((self.target_order-0.5*alpha)/(0.43)*abs(np.log(self.hmin/self.H0))), 4)
                # measured log(2 rho_2) = 0.7
                quad_order_diagonalV = max(np.ceil((self.target_order-0.5*alpha)/(0.7)*abs(np.log(self.hmin/self.H0))), 4)
        else:
            quad_order_diagonalV = quad_order_diagonal
        self.quad_order_diagonal = quad_order_diagonal
        self.quad_order_diagonalV = quad_order_diagonalV

        self.x = uninitialized((0, self.dim))
        self.y = uninitialized((0, self.dim))
        self.temp = uninitialized((0), dtype=REAL)

        self.idx = uninitialized((3), dtype=INDEX)

        self.idx1 = uninitialized((self.dim+1), dtype=INDEX)
        self.idx2 = uninitialized((self.dim+1), dtype=INDEX)
        self.idx3 = uninitialized((2*(self.dim+1)), dtype=INDEX)
        self.idx4 = uninitialized(((2*self.DoFMap.dofs_per_element)*(2*self.DoFMap.dofs_per_element+1)//2), dtype=INDEX)

        if not self.kernel.variable:
            self.getNearQuadRule(COMMON_FACE)
            self.getNearQuadRule(COMMON_EDGE)
            self.getNearQuadRule(COMMON_VERTEX)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
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
            REAL_t alpha = self.kernel.getSingularityValue()
        if alpha == 0.:
            panel = <panelType>max(ceil(self.target_order), 2)
        else:
            panel = <panelType>max(ceil((c + 0.5*alpha*logh2H0 + loghminH0 - (1.-0.5*alpha)*logdh2) /
                                        (max(logdh1, 0) + 0.4)),
                                   2)
            panel2 = <panelType>max(ceil((c + 0.5*alpha*logh1H0 + loghminH0 - (1.-0.5*alpha)*logdh1) /
                                         (max(logdh2, 0) + 0.4)),
                                    2)
            panel = max(panel, panel2)
        if self.distantQuadRulesPtr[panel] == NULL:
            self.addQuadRule(panel)
        return panel

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void addQuadRule(self, panelType panel):
        cdef:
            simplexQuadratureRule qr0, qr1
            doubleSimplexQuadratureRule qr2
            specialQuadRule sQR
            REAL_t[:, ::1] PSI
            INDEX_t I, k, i, j
            INDEX_t numQuadNodes0, numQuadNodes1, dofs_per_element
            shapeFunction sf
        qr0 = simplexXiaoGimbutas(panel, self.dim)
        qr1 = qr0
        qr2 = doubleSimplexQuadratureRule(qr0, qr1)
        numQuadNodes0 = qr0.num_nodes
        numQuadNodes1 = qr1.num_nodes
        dofs_per_element = self.DoFMap.dofs_per_element
        PSI = uninitialized((2*dofs_per_element,
                             qr2.num_nodes), dtype=REAL)
        # phi_i(x) - phi_i(y) = phi_i(x) for i = 0,1,2
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    PSI[I, k] = sf(qr0.nodes[:, i])
                    k += 1
        # phi_i(x) - phi_i(y) = -phi_i(y) for i = 3,4,5
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    PSI[I+dofs_per_element, k] = -sf(qr1.nodes[:, j])
                    k += 1
        sQR = specialQuadRule(qr2, PSI)
        self.distantQuadRules[panel] = sQR
        self.distantQuadRulesPtr[panel] = <void*>(self.distantQuadRules[panel])

        if numQuadNodes0 > self.x.shape[0]:
            self.x = uninitialized((numQuadNodes0, self.dim), dtype=REAL)
        if numQuadNodes1 > self.y.shape[0]:
            self.y = uninitialized((numQuadNodes1, self.dim), dtype=REAL)
        if numQuadNodes0*numQuadNodes1 > self.temp.shape[0]:
            self.temp = uninitialized((numQuadNodes0*numQuadNodes1), dtype=REAL)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void getNearQuadRule(self, panelType panel):
        cdef:
            INDEX_t i
            REAL_t alpha = self.kernel.getSingularityValue()
            REAL_t eta0, eta1, eta2, eta3
            specialQuadRule sQR0, sQR1
            quadQuadratureRule qrId, qrEdge0, qrEdge1, qrVertex
            REAL_t[:, :, ::1] PSI_id, PSI_edge, PSI_vertex
        if panel == COMMON_FACE:
            try:
                sQR0 = self.specialQuadRules[(alpha, panel, 0)]
            except KeyError:
                # COMMON_FACE panels have 3 integral contributions.
                # Each integral is over a 1D domain.
                qrId = GaussJacobi(((1, 5+alpha, 0),
                                    (1, 4+alpha, 0),
                                    (1, 3+alpha, 0),
                                    (self.quad_order_diagonal, 0, 0)))
                PSI_id = uninitialized((3,
                                        self.DoFMap.dofs_per_element,
                                        qrId.num_nodes),
                                       dtype=REAL)
                for i in range(qrId.num_nodes):
                    eta0 = qrId.nodes[0, i]
                    eta1 = qrId.nodes[1, i]
                    eta2 = qrId.nodes[2, i]
                    eta3 = qrId.nodes[3, i]

                    PSI_id[0, 0, i] = -eta3
                    PSI_id[0, 1, i] = eta3-1.
                    PSI_id[0, 2, i] = 1.

                    PSI_id[1, 0, i] = -1.
                    PSI_id[1, 1, i] = 1.-eta3
                    PSI_id[1, 2, i] = eta3

                    PSI_id[2, 0, i] = eta3
                    PSI_id[2, 1, i] = -1.
                    PSI_id[2, 2, i] = 1.-eta3
                sQR0 = specialQuadRule(qrId, PSI3=PSI_id)
                self.specialQuadRules[(alpha, panel, 0)] = sQR0
                if qrId.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrId.num_nodes), dtype=REAL)
            self.qrId = sQR0.qr
            self.PSI_id = sQR0.PSI3
        elif panel == COMMON_EDGE:
            try:
                sQR0 = self.specialQuadRules[(alpha, panel, 0)]
                sQR1 = self.specialQuadRules[(alpha, panel, 1)]
            except KeyError:
                qrEdge0 = GaussJacobi(((1, 5+alpha, 0),
                                       (1, 4+alpha, 0),
                                       (self.quad_order_diagonal, 0, 0),
                                       (self.quad_order_diagonal, 0, 0)))
                qrEdge1 = GaussJacobi(((1, 5+alpha, 0),
                                       (1, 4+alpha, 0),
                                       (self.quad_order_diagonal, 1, 0),
                                       (self.quad_order_diagonal, 0, 0)))
                PSI_edge = uninitialized((5,
                                          2*self.DoFMap.dofs_per_element-2*self.DoFMap.dofs_per_vertex-self.DoFMap.dofs_per_edge,
                                          qrEdge0.num_nodes),
                                    dtype=REAL)
                for i in range(qrEdge0.num_nodes):
                    eta0 = qrEdge0.nodes[0, i]
                    eta1 = qrEdge0.nodes[1, i]
                    eta2 = qrEdge0.nodes[2, i]
                    eta3 = qrEdge0.nodes[3, i]

                    PSI_edge[0, 0, i] = -eta2
                    PSI_edge[0, 1, i] = 1.-eta3
                    PSI_edge[0, 2, i] = eta3
                    PSI_edge[0, 3, i] = eta2-1.

                    eta0 = qrEdge1.nodes[0, i]
                    eta1 = qrEdge1.nodes[1, i]
                    eta2 = qrEdge1.nodes[2, i]
                    eta3 = qrEdge1.nodes[3, i]

                    PSI_edge[1, 0, i] = -eta2*eta3
                    PSI_edge[1, 1, i] = eta2-1.
                    PSI_edge[1, 2, i] = 1.
                    PSI_edge[1, 3, i] = eta2*(eta3-1.)

                    PSI_edge[2, 0, i] = eta2
                    PSI_edge[2, 1, i] = eta2*eta3-1.
                    PSI_edge[2, 2, i] = 1.-eta2
                    PSI_edge[2, 3, i] = -eta2*eta3

                    PSI_edge[3, 0, i] = eta2*eta3
                    PSI_edge[3, 1, i] = 1.-eta2
                    PSI_edge[3, 2, i] = eta2*(1.-eta3)
                    PSI_edge[3, 3, i] = -1.

                    PSI_edge[4, 0, i] = eta2*eta3
                    PSI_edge[4, 1, i] = eta2-1.
                    PSI_edge[4, 2, i] = 1.-eta2*eta3
                    PSI_edge[4, 3, i] = -eta2

                sQR0 = specialQuadRule(qrEdge0, PSI3=PSI_edge)
                sQR1 = specialQuadRule(qrEdge1, PSI3=PSI_edge)
                self.specialQuadRules[(alpha, panel, 0)] = sQR0
                self.specialQuadRules[(alpha, panel, 1)] = sQR1
                if qrEdge0.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrEdge0.num_nodes), dtype=REAL)
                if qrEdge1.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrEdge1.num_nodes), dtype=REAL)
            self.qrEdge0 = sQR0.qr
            self.qrEdge1 = sQR1.qr
            self.PSI_edge = sQR0.PSI3
        elif panel == COMMON_VERTEX:
            try:
                sQR0 = self.specialQuadRules[(alpha, panel, 0)]
            except KeyError:
                qrVertex = GaussJacobi(((1, 5+alpha, 0),
                                        (self.quad_order_diagonalV, 0, 0),
                                        (self.quad_order_diagonalV, 1, 0),
                                        (self.quad_order_diagonalV, 0, 0)))
                PSI_vertex = uninitialized((2,
                                            2*self.DoFMap.dofs_per_element-self.DoFMap.dofs_per_vertex,
                                            qrVertex.num_nodes),
                                           dtype=REAL)
                for i in range(qrVertex.num_nodes):
                    eta0 = qrVertex.nodes[0, i]
                    eta1 = qrVertex.nodes[1, i]
                    eta2 = qrVertex.nodes[2, i]
                    eta3 = qrVertex.nodes[3, i]

                    PSI_vertex[0, 0, i] = eta2-1.
                    PSI_vertex[0, 1, i] = 1.-eta1
                    PSI_vertex[0, 2, i] = eta1
                    PSI_vertex[0, 3, i] = eta2*(eta3-1.)
                    PSI_vertex[0, 4, i] = -eta2*eta3

                    PSI_vertex[1, 0, i] = 1.-eta2
                    PSI_vertex[1, 1, i] = eta2*(1.-eta3)
                    PSI_vertex[1, 2, i] = eta2*eta3
                    PSI_vertex[1, 3, i] = eta1-1.
                    PSI_vertex[1, 4, i] = -eta1

                sQR0 = specialQuadRule(qrVertex, PSI3=PSI_vertex)
                self.specialQuadRules[(alpha, panel, 0)] = sQR0
                if qrVertex.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrVertex.num_nodes), dtype=REAL)
            self.qrVertex = sQR0.qr
            self.PSI_vertex = sQR0.PSI3
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

    def __repr__(self):
        return (super(integrable2D, self).__repr__() +
                'hmin:                          {:.3}\n'.format(self.hmin) +
                'H0:                            {:.3}\n'.format(self.H0) +
                'target order:                  {}\n'.format(self.target_order))

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J, l, m
            REAL_t vol, val, vol1 = self.vol1, vol2 = self.vol2
            specialQuadRule sQR
            doubleSimplexQuadratureRule qr2
            REAL_t[:, ::1] PSI
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            INDEX_t numQuadNodes, numQuadNodes0, numQuadNodes1, dofs_per_element
            REAL_t alpha = self.kernel.getSingularityValue()
            REAL_t scaling = self.kernel.getScalingValue()
            INDEX_t[::1] idx1, idx2, idx3, idx4
            BOOL_t cutElements = False
            REAL_t horizon2
            simplexQuadratureRule qr0, qr1
            transformQuadratureRule qr0trans, qr1trans
            INDEX_t numInside
            INDEX_t outside, inside1, inside2
            INDEX_t inside, outside1, outside2
            REAL_t vol3 = np.nan, vol4 = np.nan, d1, d2, c1, c2
            REAL_t PSI_I, PSI_J
            REAL_t a_b1[3]
            REAL_t a_b2[3]
            REAL_t a_A1[3][3]
            REAL_t a_A2[3][3]
            REAL_t[:, ::1] A1, A2
            REAL_t[::1] b1, b2
            BOOL_t a_ind[3]
            BOOL_t[::1] ind
            REAL_t temp

        if self.kernel.finiteHorizon and panel >= 1 :
            # check if the horizon might cut the elements
            if self.kernel.interaction.relPos == CUT:
                cutElements = True
            if self.kernel.complement:
                cutElements = False
                # TODO: cutElements should be set to True, but
                #       need to figure out the element
                #       transformation.

        contrib[:] = 0.

        if panel >= 1 and not cutElements:
            sQR = <specialQuadRule>(self.distantQuadRulesPtr[panel])
            qr2 = <doubleSimplexQuadratureRule>(sQR.qr)
            PSI = sQR.PSI
            qr0 = <simplexQuadratureRule>qr2.rule1
            qr1 = <simplexQuadratureRule>qr2.rule2
            numQuadNodes0 = qr0.num_nodes
            numQuadNodes1 = qr1.num_nodes
            qr0.nodesInGlobalCoords(simplex1, self.x)
            qr1.nodesInGlobalCoords(simplex2, self.y)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    self.temp[k] = (qr0.weights[i] *
                                    qr1.weights[j] *
                                    self.kernel.evalPtr(2,
                                                        &self.x[i, 0],
                                                        &self.y[j, 0]))
                    k += 1
            vol = vol1 * vol2
            # loop over all local DoFs
            k = 0
            for I in range(2*self.DoFMap.dofs_per_element):
                for J in range(I, 2*self.DoFMap.dofs_per_element):
                    if mask & (1 << k):
                        val = 0.
                        for l in range(numQuadNodes0*numQuadNodes1):
                            val += self.temp[l]*PSI[I, l]*PSI[J, l]
                        contrib[k] = val*vol
                    k += 1
        elif panel >= 1 and cutElements:
            sQR = <specialQuadRule>(self.distantQuadRulesPtr[panel])
            qr2 = <doubleSimplexQuadratureRule>(sQR.qr)
            qr0 = qr2.rule1
            qr1 = qr2.rule2
            if sQR.qrTransformed1 is not None:
                qr1trans = sQR.qrTransformed1
            else:
                qr1trans = transformQuadratureRule(qr1)
                sQR.qrTransformed1 = qr1trans
            numQuadNodes0 = qr0.num_nodes
            numQuadNodes1 = qr1.num_nodes

            horizon2 = self.kernel.getHorizonValue2()
            vol = vol1*vol2
            dofs_per_element = self.DoFMap.dofs_per_element

            A1 = a_A1
            A2 = a_A2
            b1 = a_b1
            b2 = a_b2

            # ind = a_ind
            # qr0.nodesInGlobalCoords(simplex1, self.x)
            # for i in range(qr0.num_nodes):
            #     numInside = 0
            #     for j in range(3):
            #         d2 = 0.
            #         for k in range(2):
            #             d2 += (simplex2[j, k]-self.x[i, k])**2
            #         ind[j] = (d2 <= horizon2)
            #         numInside += ind[j]
            #     if numInside == 0:
            #         continue
            #     elif numInside == 1:
            #         inside = 0
            #         while not ind[inside]:
            #             inside += 1
            #         outside1 = (inside+1)%3
            #         outside2 = (inside+2)%3
            #         c1 = findIntersection(self.x[i, :], simplex2[inside, :], simplex2[outside1, :], horizon2)
            #         c2 = findIntersection(self.x[i, :], simplex2[inside, :], simplex2[outside2, :], horizon2)
            #         A1[:, :] = 0.
            #         b1[:] = 0.
            #         A1[inside,inside] = c1+c2
            #         A1[inside,outside1] = c2
            #         A1[inside,outside2] = c1
            #         A1[outside1,outside1] = c1
            #         A1[outside2,outside2] = c2
            #         b1[inside] = 1-c1-c2
            #         vol3 = c1*c2
            #         qr1trans.setBaryTransform(A1, b1)
            #         qr1 = qr1trans
            #     elif numInside == 2:
            #         # outside = np.where(ind == False)[0][0]
            #         outside = 0
            #         while ind[outside]:
            #             outside += 1
            #         inside1 = (outside+1)%3
            #         inside2 = (outside+2)%3
            #         c1 = findIntersection(self.x[i,: ], simplex2[outside, :], simplex2[inside1, :], horizon2)
            #         c2 = findIntersection(self.x[i,: ], simplex2[outside, :], simplex2[inside2, :], horizon2)
            #         d1 = 0.
            #         d2 = 0.
            #         for k in range(2):
            #             d1 += (simplex2[outside, k]
            #                    + c1*(simplex2[inside1, k]-simplex2[outside, k])
            #                    - simplex2[inside2, k])**2
            #             d2 += (simplex2[outside, k]
            #                    + c2*(simplex2[inside2, k]-simplex2[outside, k])
            #                    - simplex2[inside1, k])
            #         A1[:, :] = 0.
            #         b1[:] = 0.
            #         A2[:, :] = 0.
            #         b2[:] = 0.

            #         if d1 < d2:
            #             A1[outside,outside] = 1-c1
            #             A1[inside1,inside1] = 1-c1
            #             A1[inside1,inside2] = -c1
            #             A1[inside2,inside2] = 1.
            #             b1[inside1] = c1
            #             vol3 = 1-c1

            #             A2[outside,outside] = 1-c2
            #             A2[inside2,inside2] = 1
            #             A2[inside2,outside] = c2
            #             A2[outside,inside1] = 1-c1
            #             A2[inside1,inside1] = c1
            #             vol4 = c1*(1-c2)
            #         else:
            #             A1[outside,outside] = 1-c2
            #             A1[inside2,inside2] = 1-c2
            #             A1[inside2,inside1] = -c2
            #             A1[inside1,inside1] = 1.
            #             b1[inside2] = c2
            #             vol3 = 1-c2

            #             A2[outside,outside] = 1-c1
            #             A2[inside1,inside1] = 1
            #             A2[inside1,outside] = c1
            #             A2[outside,inside2] = 1-c2
            #             A2[inside2,inside2] = c2
            #             vol4 = c2*(1-c1)

            #         qr1trans.setBaryTransform(A1, b1)
            #         qr1 = qr1trans
            #     else:
            #         qr1 = qr2.rule2
            #         vol3 = 1.

            #     qr1.nodesInGlobalCoords(simplex2, self.y)
            #     for j in range(qr1.num_nodes):
            #         val = qr0.weights[i]*qr1.weights[j]*self.kernel.evalPtr(2, &self.x[i, 0], &self.y[j, 0])
            #         val *= vol*vol3

            #         k = 0
            #         for I in range(2*dofs_per_element):
            #             if I < dofs_per_element:
            #                 PSI_I = self.getLocalShapeFunction(I).evalStrided(&qr0.nodes[0, i], numQuadNodes0)
            #             else:
            #                 PSI_I = -self.getLocalShapeFunction(I-dofs_per_element).evalStrided(&qr1.nodes[0, j], numQuadNodes1)
            #             for J in range(I, 2*dofs_per_element):
            #                 if mask & (1 << k):
            #                     if J < dofs_per_element:
            #                         PSI_J = self.getLocalShapeFunction(J).evalStrided(&qr0.nodes[0, i], numQuadNodes0)
            #                     else:
            #                         PSI_J = -self.getLocalShapeFunction(J-dofs_per_element).evalStrided(&qr1.nodes[0, j], numQuadNodes1)
            #                     contrib[k] += val * PSI_I*PSI_J
            #                 k += 1
            #     if numInside == 2:
            #         qr1trans.setBaryTransform(A2, b2)
            #         qr1.nodesInGlobalCoords(simplex2, self.y)
            #         for j in range(qr1.num_nodes):
            #             val = qr0.weights[i]*qr1.weights[j]*self.kernel.evalPtr(2, &self.x[i, 0], &self.y[j, 0])
            #             val *= vol*vol4

            #             k = 0
            #             for I in range(2*dofs_per_element):
            #                 if I < dofs_per_element:
            #                     PSI_I = self.getLocalShapeFunction(I).evalStrided(&qr0.nodes[0, i], numQuadNodes0)
            #                 else:
            #                     PSI_I = -self.getLocalShapeFunction(I-dofs_per_element).evalStrided(&qr1.nodes[0, j], numQuadNodes1)
            #                 for J in range(I, 2*dofs_per_element):
            #                     if mask & (1 << k):
            #                         if J < dofs_per_element:
            #                             PSI_J = self.getLocalShapeFunction(J).evalStrided(&qr0.nodes[0, i], numQuadNodes0)
            #                         else:
            #                             PSI_J = -self.getLocalShapeFunction(J-dofs_per_element).evalStrided(&qr1.nodes[0, j], numQuadNodes1)
            #                         contrib[k] += val * PSI_I*PSI_J
            #                     k += 1

            # contrib2 = np.zeros((contrib.shape[0]), dtype=REAL)
            # qr0trans = transformQuadratureRule(qr2.rule1, self.tempNodes1)
            if sQR.qrTransformed0 is not None:
                qr0trans = sQR.qrTransformed0
            else:
                qr0trans = transformQuadratureRule(qr0)
                sQR.qrTransformed0 = qr0trans
            # qr1trans = transformQuadratureRule(qr2.rule2)

            self.kernel.interaction.startLoopSubSimplices_Simplex(simplex1, simplex2)
            while self.kernel.interaction.nextSubSimplex_Simplex(A1, b1, &c1):
                qr0trans.setBaryTransform(A1, b1)
                qr0trans.nodesInGlobalCoords(simplex1, self.x)
                for i in range(qr0trans.num_nodes):
                    self.kernel.interaction.startLoopSubSimplices_Node(self.x[i, :], simplex2)
                    while self.kernel.interaction.nextSubSimplex_Node(A2, b2, &c2):
                        qr1trans.setBaryTransform(A2, b2)
                        qr1trans.nodesInGlobalCoords(simplex2, self.y)
                        for j in range(qr1trans.num_nodes):
                            val = qr0trans.weights[i]*qr1trans.weights[j]*self.kernel.evalPtr(2, &self.x[i, 0], &self.y[j, 0])
                            val *= c1 * c2 * vol
                            k = 0
                            for I in range(2*dofs_per_element):
                                if I < dofs_per_element:
                                    PSI_I = self.getLocalShapeFunction(I).evalStrided(&qr0trans.nodes[0, i], numQuadNodes0)
                                else:
                                    PSI_I = -self.getLocalShapeFunction(I-dofs_per_element).evalStrided(&qr1trans.nodes[0, j], numQuadNodes1)
                                for J in range(I, 2*dofs_per_element):
                                    if mask & (1 << k):
                                        if J < dofs_per_element:
                                            PSI_J = self.getLocalShapeFunction(J).evalStrided(&qr0trans.nodes[0, i], numQuadNodes0)
                                        else:
                                            PSI_J = -self.getLocalShapeFunction(J-dofs_per_element).evalStrided(&qr1trans.nodes[0, j], numQuadNodes1)
                                        contrib[k] += val * PSI_I*PSI_J
                                    k += 1

        elif panel == COMMON_FACE:
            # factor 2 comes from symmetric contributions
            vol = scaling*4.0*2.0*vol1**2

            # three different integrals
            numQuadNodes = self.qrId.num_nodes
            for l in range(3):
                # distance between x and y quadrature nodes
                for i in range(numQuadNodes):
                    temp = 0.
                    for j in range(2):
                        temp += (simplex1[0, j]*self.PSI_id[l, 0, i] +
                                 simplex1[1, j]*self.PSI_id[l, 1, i] +
                                 simplex1[2, j]*self.PSI_id[l, 2, i])**2
                    self.temp[i] = self.qrId.weights[i]*pow(temp, 0.5*alpha)
                # loop over all local DoFs
                for I in range(3):
                    for J in range(I, 3):
                        k = 6*I-(I*(I+1) >> 1) + J
                        if mask & (1 << k):
                            val = 0.
                            for i in range(numQuadNodes):
                                val += (self.temp[i] *
                                        self.PSI_id[l, I, i] *
                                        self.PSI_id[l, J, i])
                            contrib[k] += val*vol
        elif panel == COMMON_EDGE:
            # order so that common edge matches up and first triangle
            # is ordered in usual sense and second triangle in counter
            # sense

            idx1 = self.idx1
            idx2 = self.idx2
            idx3 = self.idx3
            idx4 = self.idx4

            k = 0
            for i in range(3):
                for j in range(3):
                    if self.cells1[self.cellNo1, i] == self.cells2[self.cellNo2, j]:
                        idx3[k] = i
                        idx4[k] = j
                        k += 1
                        break

            if idx3[0] > idx3[1]:
                idx3[1], idx3[0] = idx3[0], idx3[1]

            if idx3[0] == 0:
                if idx3[1] == 1:
                    idx1[0], idx1[1], idx1[2] = 0, 1, 2
                elif idx3[1] == 2:
                    idx1[0], idx1[1], idx1[2] = 2, 0, 1
                else:
                    raise NotImplementedError("Something went wrong for COMMON_EDGE 1")
            elif idx3[0] == 1 and idx3[1] == 2:
                idx1[0], idx1[1], idx1[2] = 1, 2, 0
            else:
                raise NotImplementedError("Something went wrong for COMMON_EDGE 1")

            if idx4[0] > idx4[1]:
                idx4[1], idx4[0] = idx4[0], idx4[1]

            if idx4[0] == 0:
                if idx4[1] == 1:
                    idx2[0], idx2[1], idx2[2] = 1, 0, 2
                elif idx4[1] == 2:
                    idx2[0], idx2[1], idx2[2] = 0, 2, 1
                else:
                    raise NotImplementedError("Something went wrong for COMMON_EDGE 2")
            elif idx4[0] == 1 and idx4[1] == 2:
                idx2[0], idx2[1], idx2[2] = 2, 1, 0
            else:
                raise NotImplementedError("Something went wrong for COMMON_EDGE 2")

            idx3[0], idx3[1], idx3[2], idx3[3] = idx1[0], idx1[1], idx1[2], 3+idx2[2]

            vol = scaling*4.0*vol1*vol2

            # loop over all local DoFs
            m = 0
            for I in range(4):
                for J in range(I, 4):
                    i = idx3[I]
                    j = idx3[J]
                    if j < i:
                        i, j = j, i
                    idx4[m] = 6*i-(i*(i+1) >> 1) + j
                    m += 1

            # five different integrals
            for l in range(5):
                if l == 0:
                    qrEdge = self.qrEdge0
                else:
                    qrEdge = self.qrEdge1
                numQuadNodes = qrEdge.num_nodes
                # distance between x and y quadrature nodes
                for i in range(numQuadNodes):
                    temp = 0.
                    for j in range(2):
                        temp += (simplex1[idx1[0], j]*self.PSI_edge[l, 0, i] +
                                 simplex1[idx1[1], j]*self.PSI_edge[l, 1, i] +
                                 simplex1[idx1[2], j]*self.PSI_edge[l, 2, i] +
                                 simplex2[idx2[2], j]*self.PSI_edge[l, 3, i])**2
                    self.temp[i] = qrEdge.weights[i]*pow(temp, 0.5*alpha)

                # loop over all local DoFs
                m = 0
                for I in range(4):
                    for J in range(I, 4):
                        k = idx4[m]
                        m += 1
                        if mask & (1 << k):
                            val = 0.
                            for i in range(numQuadNodes):
                                val += (self.temp[i] *
                                        self.PSI_edge[l, I, i] *
                                        self.PSI_edge[l, J, i])
                            contrib[k] += val*vol
        elif panel == COMMON_VERTEX:
            # Find vertex that matches
            i = 0
            j = 0
            while True:
                if self.cells1[self.cellNo1, i] == self.cells2[self.cellNo2, j]:
                    break
                if j == 2:
                    i += 1
                    j = 0
                else:
                    j += 1

            idx1 = self.idx1
            idx2 = self.idx2
            idx3 = self.idx3

            if i == 0:
                idx1[0], idx1[1], idx1[2] = 0, 1, 2
            elif i == 1:
                idx1[0], idx1[1], idx1[2] = 1, 2, 0
            else:
                idx1[0], idx1[1], idx1[2] = 2, 0, 1
            if j == 0:
                idx2[0], idx2[1], idx2[2] = 0, 1, 2
            elif j == 1:
                idx2[0], idx2[1], idx2[2] = 1, 2, 0
            else:
                idx2[0], idx2[1], idx2[2] = 2, 0, 1
            idx3[0], idx3[1], idx3[2], idx3[3], idx3[4] = idx1[0], idx1[1], idx1[2], 3+idx2[1], 3+idx2[2]

            # factor 4. comes from inverse sqare of volume of standard simplex
            vol = scaling*4.0*vol1*vol2

            # two different integrals
            numQuadNodes = self.qrVertex.num_nodes
            for l in range(2):
                # distance between x and y quadrature nodes
                for i in range(numQuadNodes):
                    temp = 0.
                    for j in range(2):
                        temp += (simplex1[idx1[0], j]*self.PSI_vertex[l, 0, i] +
                                 simplex1[idx1[1], j]*self.PSI_vertex[l, 1, i] +
                                 simplex1[idx1[2], j]*self.PSI_vertex[l, 2, i] +
                                 simplex2[idx2[1], j]*self.PSI_vertex[l, 3, i] +
                                 simplex2[idx2[2], j]*self.PSI_vertex[l, 4, i])**2
                    self.temp[i] = self.qrVertex.weights[i]*pow(temp, 0.5*alpha)

                # loop over all local DoFs
                for I in range(5):
                    for J in range(I, 5):
                        i = idx3[I]
                        j = idx3[J]
                        if j < i:
                            i, j = j, i
                        k = 6*i-(i*(i+1) >> 1) + j
                        if mask & (1 << k):
                            val = 0.
                            for i in range(numQuadNodes):
                                val += (self.temp[i] *
                                        self.PSI_vertex[l, I, i] *
                                        self.PSI_vertex[l, J, i])
                            contrib[k] += val*vol
        else:
            print(np.array(simplex1), np.array(simplex2))
            raise NotImplementedError('Unknown panel type: {}'.format(panel))
