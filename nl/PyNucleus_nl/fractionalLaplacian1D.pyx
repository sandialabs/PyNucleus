###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from libc.math cimport (log, ceil, fabs as abs, pow)
import numpy as np
cimport numpy as np

from PyNucleus_base.myTypes import INDEX, REAL
from PyNucleus_base import uninitialized, uninitialized_like
from PyNucleus_fem.meshCy cimport meshBase
from . nonlocalLaplacianBase import ALL
from PyNucleus_fem.quadrature cimport (simplexQuadratureRule,
                             transformQuadratureRule,
                             doubleSimplexQuadratureRule,
                             GaussJacobi,
                             simplexXiaoGimbutas)
from PyNucleus_fem.DoFMaps cimport DoFMap, P1_DoFMap, P2_DoFMap, P0_DoFMap, shapeFunction
include "panelTypes.pxi"

cdef INDEX_t MAX_INT = np.iinfo(INDEX).max


cdef class fractionalLaplacian1DZeroExterior(nonlocalLaplacian1D):
    def __init__(self, FractionalKernel kernel, meshBase mesh, DoFMap DoFMap, num_dofs=None, **kwargs):
        manifold_dim2 = mesh.dim-1
        super(fractionalLaplacian1DZeroExterior, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2=manifold_dim2, **kwargs)
        self.symmetricCells = False


cdef class fractionalLaplacian1D_P1(nonlocalLaplacian1D):
    """The local matrix

    .. math::

       \\int_{K_1}\\int_{K_2} (u(x)-u(y)) (v(x)-v(y)) \\gamma(x,y) dy dx

    for the 1D fractional Laplacian on P1 elements.
    """
    def __init__(self,
                 FractionalKernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 quad_order_diagonal=None,
                 target_order=None,
                 num_dofs=None,
                 **kwargs):
        cdef:
            REAL_t smin, smax
        assert isinstance(DoFMap, P1_DoFMap)
        super(fractionalLaplacian1D_P1, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        smin, smax = self.kernel.s.min, self.kernel.s.max

        if target_order is None:
            # this is the desired local quadrature error
            target_order = 2.-smin
        self.target_order = target_order
        if quad_order_diagonal is None:
            # measured log(2 rho_2) = 0.43
            quad_order_diagonal = max(np.ceil(((target_order+2.)*log(self.num_dofs*self.H0) + (2.*smax-1.)*abs(log(self.hmin/self.H0)))/0.8), 2)
        self.quad_order_diagonal = quad_order_diagonal

        self.x = uninitialized((0, self.dim))
        self.y = uninitialized((0, self.dim))
        self.temp = uninitialized((0), dtype=REAL)
        self.temp2 = uninitialized((0), dtype=REAL)

        self.idx = uninitialized((3), dtype=INDEX)
        self.distantPSI = {}

        if not self.kernel.variableOrder:
            self.getNearQuadRule(COMMON_EDGE)
            self.getNearQuadRule(COMMON_VERTEX)

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        cdef:
            panelType panel, panel2
            REAL_t logdh1 = log(d/h1), logdh2 = log(d/h2)
            REAL_t s = (<FractionalKernel>self.kernel).getsValue()
        panel = <panelType>max(ceil(((self.target_order+2.)*log(self.num_dofs*self.H0) + (2.*s-1.)*abs(log(h2/self.H0)) - 2.*s*logdh2) /
                                    (max(logdh1, 0) + 0.8)),
                               2)
        panel2 = <panelType>max(ceil(((self.target_order+2.)*log(self.num_dofs*self.H0) + (2.*s-1.)*abs(log(h1/self.H0)) - 2.*s*logdh1) /
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
            REAL_t s = (<FractionalKernel>self.kernel).getsValue()
            REAL_t eta0, eta1
            specialQuadRule sQR0

        if panel == COMMON_EDGE:
            try:
                sQR0 = self.specialQuadRules[(s, panel)]
            except KeyError:
                qrId = GaussJacobi(((1, 1, 1-2*s),
                                    (1, 0, 0)))

                PSI_id = uninitialized((self.DoFMap.dofs_per_element, qrId.num_nodes), dtype=REAL)
                # COMMON_FACE panels
                for i in range(qrId.num_nodes):
                    eta0 = qrId.nodes[0, i]
                    eta1 = qrId.nodes[1, i]

                    # x = 1-eta0+eta0*eta1
                    # PHI_id[0, 0, i] = 1.-x
                    # PHI_id[0, 1, i] = x

                    # y = eta0*eta1
                    # PHI_id[2, i] = 1.-y
                    # PHI_id[3, i] = y

                    PSI_id[0, i] = -1                      # ((1-x)-(1-y))/(1-eta0)
                    PSI_id[1, i] = 1                       # (x-y)/(1-eta0)
                sQR0 = specialQuadRule(qrId, PSI_id)
                self.specialQuadRules[(s, panel)] = sQR0
                if qrId.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrId.num_nodes), dtype=REAL)
                    self.temp2 = uninitialized_like(self.temp)
            self.qrId = sQR0.qr
            self.PSI_id = sQR0.PSI
        elif panel == COMMON_VERTEX:
            try:
                sQR0 = self.specialQuadRules[(s, panel)]
            except KeyError:

                qrVertex0 = GaussJacobi(((1, 2-2*s, 0),
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

                qrVertex = qrVertex0+qrVertex1
                sQR0 = specialQuadRule(qrVertex, np.hstack((PSI_vertex0, PSI_vertex1)))
                self.specialQuadRules[(s, panel)] = sQR0
                if qrVertex.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrVertex.num_nodes), dtype=REAL)
                    self.temp2 = uninitialized_like(self.temp)
            self.qrVertex = sQR0.qr
            self.PSI_vertex = sQR0.PSI
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

    def __repr__(self):
        return (super(fractionalLaplacian1D_P1, self).__repr__() +
                'hmin:                          {:.3}\n'.format(self.hmin) +
                'H0:                            {:.3}\n'.format(self.H0) +
                'target order:                  {}\n'.format(self.target_order) +
                'quad_order_diagonal:           {}\n'.format(self.quad_order_diagonal) +
                'quad_order_off_diagonal:       {}\n'.format(list(self.distantQuadRules.keys())))

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J, t
            REAL_t vol, val, vol1 = self.vol1, vol2 = self.vol2
            INDEX_t[::1] idx = self.idx
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            REAL_t s = (<FractionalKernel>self.kernel).getsValue()

        if panel >= 1:
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
                self.temp[i] = self.qrId.weights[i]*pow(self.temp[i], -0.5-s)
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
                        for i in range(self.qrVertex.num_nodes):
                            val += (self.qrVertex.weights[i] *
                                    self.PSI_vertex[idx[I], i] *
                                    self.PSI_vertex[idx[J], i] *
                                    pow(vol1*self.PSI_vertex[0, i]-vol2*self.PSI_vertex[2, i], -1.-2.*s))
                        contrib[k] += val*vol
        else:
            print(np.array(simplex1), np.array(simplex2))
            raise NotImplementedError('Unknown panel type: {}'.format(panel))


cdef class fractionalLaplacian1D_P1_boundary(fractionalLaplacian1DZeroExterior):
    def __init__(self,
                 FractionalKernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 quad_order_diagonal=None,
                 target_order=None,
                 num_dofs=None,
                 **kwargs):
        assert isinstance(DoFMap, P1_DoFMap)
        super(fractionalLaplacian1D_P1_boundary, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        if self.kernel.variableOrder:
            smin, smax = self.kernel.s.min, self.kernel.s.max
        else:
            smin, smax = self.kernel.sValue, self.kernel.sValue
        if target_order is None:
            # this is the desired local quadrature error
            target_order = 2.-smin
        self.target_order = target_order

        if quad_order_diagonal is None:
            # measured log(2 rho_2) = 0.4
            quad_order_diagonal = max(np.ceil(((target_order+1.)*log(self.num_dofs*self.H0)+(2.*smax-1.)*abs(log(self.hmin/self.H0)))/0.8), 2)
        self.quad_order_diagonal = quad_order_diagonal

        self.x = uninitialized((0, self.dim), dtype=REAL)
        self.temp = uninitialized((0), dtype=REAL)
        self.temp2 = uninitialized((0), dtype=REAL)
        self.distantPHI = {}

        if not self.kernel.variableOrder:
            self.getNearQuadRule(COMMON_VERTEX)

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        cdef:
            panelType panel, panel2
            REAL_t logdh1 = max(log(d/h1), 0.), logdh2 = max(log(d/h2), 0.)
            REAL_t s = self.kernel.sValue
            REAL_t h
        panel = <panelType>max(ceil(((self.target_order+1.)*log(self.num_dofs*self.H0) + (2.*s-1.)*abs(log(h2/self.H0)) - 2.*s*log(d/h2)) /
                                    (logdh1 + 0.8)),
                               2)
        panel2 = <panelType>max(ceil(((self.target_order+1.)*log(self.num_dofs*self.H0) + (2.*s-1.)*abs(log(h1/self.H0)) - 2.*s*log(d/h1)) /
                                     (logdh2 + 0.8)),
                                2)
        panel = max(panel, panel2)
        if self.kernel.finiteHorizon:
            # check if the horizon might cut the elements
            h = 0.5*max(h1, h2)
            if (d-h < self.kernel.horizonValue) and (self.kernel.horizonValue < d+h):
                panel *= 3
        try:
            self.distantQuadRules[panel]
        except KeyError:
            self.addQuadRule(panel)
        return panel

    cdef void addQuadRule(self, panelType panel):
        cdef:
            simplexQuadratureRule qr0
            REAL_t[:, ::1] PHI
            INDEX_t i
        qr0 = simplexXiaoGimbutas(panel, self.dim)
        self.distantQuadRules[panel] = qr0
        PHI = uninitialized((2, qr0.num_nodes), dtype=REAL)
        for i in range(qr0.num_nodes):
            PHI[0, i] = self.getLocalShapeFunction(0)(qr0.nodes[:, i])
            PHI[1, i] = self.getLocalShapeFunction(1)(qr0.nodes[:, i])
        self.distantPHI[panel] = PHI

        if qr0.num_nodes > self.x.shape[0]:
            self.x = uninitialized((qr0.num_nodes, self.dim), dtype=REAL)
        if qr0.num_nodes > self.temp.shape[0]:
            self.temp = uninitialized((qr0.num_nodes), dtype=REAL)
            self.temp2 = uninitialized((qr0.num_nodes), dtype=REAL)

    cdef void getNearQuadRule(self, panelType panel):
        cdef:
            INDEX_t i
            REAL_t s = self.kernel.sValue
            REAL_t eta
            specialQuadRule sQR0
        if panel == COMMON_VERTEX:
            try:
                sQR0 = self.specialQuadRules[(s, panel)]
            except KeyError:

                if s < 0.5:
                    qrVertex = GaussJacobi(((self.quad_order_diagonal, -2*s, 0), ))
                    PHI_vertex = uninitialized((2, qrVertex.num_nodes), dtype=REAL)
                    for i in range(qrVertex.num_nodes):
                        eta = qrVertex.nodes[0, i]
                        PHI_vertex[0, i] = 1.-eta
                        PHI_vertex[1, i] = eta
                else:
                    qrVertex = GaussJacobi(((self.quad_order_diagonal, 2-2*s, 0), ))
                    PHI_vertex = uninitialized((2, qrVertex.num_nodes), dtype=REAL)
                    for i in range(qrVertex.num_nodes):
                        eta = qrVertex.nodes[0, i]
                        PHI_vertex[0, i] = 1.  # unused
                        PHI_vertex[1, i] = 1.

                sQR0 = specialQuadRule(qrVertex, PHI=PHI_vertex)
                self.specialQuadRules[(s, panel)] = sQR0
                if qrVertex.num_nodes > self.x.shape[0]:
                    self.x = uninitialized((qrVertex.num_nodes, self.dim), dtype=REAL)
                if qrVertex.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrVertex.num_nodes), dtype=REAL)
                    self.temp2 = uninitialized((qrVertex.num_nodes), dtype=REAL)
            self.qrVertex = sQR0.qr
            self.PHI_vertex = sQR0.PHI
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

    def __repr__(self):
        return (super(fractionalLaplacian1D_P1_boundary, self).__repr__() +
                'hmin:                          {:.3}\n'.format(self.hmin) +
                'H0:                            {:.3}\n'.format(self.H0) +
                'target order:                  {}\n'.format(self.target_order) +
                'quad_order_diagonal:           {}\n'.format(self.quad_order_diagonal) +
                'quad_order_off_diagonal:       {}\n'.format(list(self.distantQuadRules.keys())))

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            REAL_t vol = self.vol1, val
            INDEX_t i, j, k, m
            INDEX_t[::1] idx = uninitialized((2), dtype=INDEX)
            simplexQuadratureRule qr
            REAL_t[:, ::1] PHI
            REAL_t[:, ::1] simplex = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2

        # Kernel:
        #  \Gamma(x,y) = n \dot (x-y) * C(d,s) / (2s) / |x-y|^{d+2s}
        # with inward normal n.
        #
        # Rewrite as
        #  \Gamma(x,y) = [ n \dot (x-y)/|x-y| ] * [ C(d,s) / (2s) / |x-y|^{d-1+2s} ]
        #
        # In 1D:
        #  n = (x-y)/|x-y|
        # so
        #  n \dot (x-y) / |x-y| = 1

        if panel == COMMON_VERTEX:
            # For s >= 0.5, we have also an exact expression
            # if s >= 0.5:
            #     if abs(simplex[0, 0]-simplex2[0, 0]) < 1e-12:
            #         contrib[0] = contrib[1] = 0.
            #         contrib[2] = scaling*(vol**(1.-2.*s)/s/(3.-2.*s))
            #     else:
            #         contrib[0] = scaling*(vol**(1.-2.*s)/s/(3.-2.*s))
            #         contrib[1] = contrib[2] = 0.
            # else:
            #     if abs(simplex[0, 0]-simplex2[0, 0]) < 1e-12:
            #         idx[0], idx[1] = 0, 1
            #     else:
            #         idx[0], idx[1] = 1, 0
            #     k = 0
            #     for i in range(2):
            #         for j in range(i, 2):
            #             s = 0.
            #             for m in range(self.qrVertex.num_nodes):
            #                 s += self.PHI_vertex[idx[i], m]*self.PHI_vertex[idx[j], m]*self.qrVertex.weights[m]
            #             s /= abs(simplex[1, 0]-simplex[0, 0])**(2.*s) * s
            #             contrib[k] = s*vol*scaling
            #             k += 1

            # sort common vertex first
            if abs(simplex[0, 0]-simplex2[0, 0]) < 1e-12:
                idx[0], idx[1] = 0, 1
            else:
                idx[0], idx[1] = 1, 0
            k = 0
            for i in range(2):
                for j in range(i, 2):
                    val = 0.
                    for m in range(self.qrVertex.num_nodes):
                        val += self.PHI_vertex[idx[i], m]*self.PHI_vertex[idx[j], m]*self.qrVertex.weights[m]
                    val *= self.kernel.evalPtr(1, &simplex[idx[0], 0], &simplex[idx[1], 0])
                    contrib[k] = val*vol
                    k += 1
        elif panel >= 1:
            qr = self.distantQuadRules[panel]
            PHI = self.distantPHI[panel]
            qr.nodesInGlobalCoords(simplex, self.x)
            for j in range(qr.num_nodes):
                self.temp[j] = self.kernel.evalPtr(1, &self.x[j, 0], &simplex2[0, 0])
            k = 0
            for i in range(2):
                for j in range(i, 2):
                    for m in range(qr.num_nodes):
                        self.temp2[m] = self.temp[m]*PHI[i, m]*PHI[j, m]
                    contrib[k] = qr.eval(self.temp2, vol)
                    k += 1
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))


cdef class fractionalLaplacian1D_P0(nonlocalLaplacian1D):
    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        return 1

    cdef void addQuadRule(self, panelType panel):
        pass

    cdef void getNearQuadRule(self, panelType panel):
        pass

    def __init__(self,
                 FractionalKernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 quad_order_diagonal=None,
                 target_order=None,
                 num_dofs=None):
        assert isinstance(DoFMap, P0_DoFMap)
        super(fractionalLaplacian1D_P0, self).__init__(kernel, mesh, DoFMap, num_dofs=num_dofs)

    def __repr__(self):
        return (super(fractionalLaplacian1D_P0, self).__repr__() +
                'hmin:                          {:.3}\n'.format(self.hmin) +
                'H0:                            {:.3}\n'.format(self.H0) +
                'target order:                  {}\n'.format(self.target_order) +
                'quad_order_diagonal:           {}\n'.format(self.quad_order_diagonal) +
                'quad_order_off_diagonal:       {}\n'.format(list(self.distantQuadRules.keys())))

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            REAL_t T
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            REAL_t s = self.kernel.sValue
            REAL_t scaling = self.kernel.scalingValue
        if panel == COMMON_EDGE:
            contrib[:] = 0.
        else:
            T = -((pow(abs(simplex1[1, 0]-simplex2[1, 0]), 1.-2.*s) -
                   pow(abs(simplex1[1, 0]-simplex2[0, 0]), 1.-2.*s) -
                   pow(abs(simplex1[0, 0]-simplex2[1, 0]), 1.-2.*s) +
                   pow(abs(simplex1[0, 0]-simplex2[0, 0]), 1.-2.*s)) /
                  ((2.*s)*(2.*s-1.)))

            contrib[0] = scaling*T
            contrib[1] = -scaling*T
            contrib[2] = scaling*T
        # else:
        #     print(np.array(simplex1), np.array(simplex2))
        #     raise NotImplementedError('Unknown panel type: {}'.format(panel))


cdef class fractionalLaplacian1D_P0_boundary(fractionalLaplacian1DZeroExterior):
    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        return 1

    cdef void addQuadRule(self, panelType panel):
        pass

    cdef void getNearQuadRule(self, panelType panel):
        pass


    def __init__(self,
                 FractionalKernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 quad_order_diagonal=None,
                 target_order=None,
                 num_dofs=None):
        assert isinstance(DoFMap, P0_DoFMap)
        super(fractionalLaplacian1D_P0_boundary, self).__init__(kernel, mesh, DoFMap, num_dofs=num_dofs)

    def __repr__(self):
        return (super(fractionalLaplacian1D_P0_boundary, self).__repr__() +
                'hmin:                          {:.3}\n'.format(self.hmin) +
                'H0:                            {:.3}\n'.format(self.H0) +
                'target order:                  {}\n'.format(self.target_order) +
                'quad_order_diagonal:           {}\n'.format(self.quad_order_diagonal) +
                'quad_order_off_diagonal:       {}\n'.format(list(self.distantQuadRules.keys())))

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            REAL_t vol = self.vol1, T
            REAL_t[:, ::1] simplex = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            REAL_t s = self.kernel.sValue
            REAL_t scaling = self.kernel.scalingValue
        T = -((pow(abs(simplex[1, 0]-simplex2[0, 0]), 1.-2.*s) -
               pow(abs(simplex[0, 0]-simplex2[0, 0]), 1.-2.*s)) /
              ((2.*s)*(2.*s-1.))) * (-1)**(simplex[0, 0] < simplex2[0, 0])*2.

        contrib[0] = scaling*T
