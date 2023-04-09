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

    cdef void getNearQuadRule(self, panelType panel):
        cdef:
            INDEX_t i
            REAL_t alpha = self.kernel.getSingularityValue()
            REAL_t eta0, eta1
            specialQuadRule sQR0, sQR1

        if alpha == 0.:
            self.getNonSingularNearQuadRule(panel)
            return

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

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J, t
            REAL_t vol, val, vol1 = self.vol1, vol2 = self.vol2
            REAL_t alpha = self.kernel.getSingularityValue()
            INDEX_t[::1] idx = self.idx
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2

        if panel >= 1 or alpha == 0.:
            self.eval_distant(contrib, panel, mask)
            return

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

    cdef void getNearQuadRule(self, panelType panel):
        cdef:
            INDEX_t i
            REAL_t alpha = self.kernel.getSingularityValue()
            REAL_t eta0, eta1, eta2, eta3
            specialQuadRule sQR0, sQR1
            quadQuadratureRule qrId, qrEdge0, qrEdge1, qrVertex
            REAL_t[:, :, ::1] PSI_id, PSI_edge, PSI_vertex
        if alpha == 0.:
            self.getNonSingularNearQuadRule(panel)
            return
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

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J, l, m
            REAL_t vol, val, vol1 = self.vol1, vol2 = self.vol2
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            INDEX_t numQuadNodes
            REAL_t alpha = self.kernel.getSingularityValue()
            REAL_t scaling = self.kernel.getScalingValue()
            INDEX_t[::1] idx1, idx2, idx3, idx4
            REAL_t temp

        if panel >= 1 or alpha == 0.:
            self.eval_distant(contrib, panel, mask)
            return

        contrib[:] = 0.

        if panel == COMMON_FACE:
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

            # factor 4. comes from inverse square of volume of standard simplex
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
