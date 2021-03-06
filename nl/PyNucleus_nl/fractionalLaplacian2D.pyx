###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from libc.math cimport (sqrt, log, ceil, fabs as abs, pow)
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc

from PyNucleus_base.myTypes import INDEX, REAL, BOOL
from PyNucleus_base import uninitialized, uninitialized_like
from PyNucleus_base.blas cimport mydot
from PyNucleus_fem.quadrature cimport (simplexQuadratureRule,
                             transformQuadratureRule,
                             doubleSimplexQuadratureRule, GaussJacobi,
                             simplexDuffyTransformation, simplexXiaoGimbutas)
from PyNucleus_fem.DoFMaps cimport DoFMap, P1_DoFMap, shapeFunction
from scipy.special import gamma
from . nonlocalLaplacianBase import ALL

include "panelTypes.pxi"

cdef INDEX_t MAX_INT = np.iinfo(INDEX).max


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef inline REAL_t findIntersection(REAL_t[::1] x, REAL_t[::1] y1, REAL_t[::1] y2, REAL_t horizon2):
    cdef:
        REAL_t nn = 0., p = 0., q = 0., A, B, c
        INDEX_t k
    for k in range(2):
        A = y2[k]-y1[k]
        B = y1[k]-x[k]
        nn += A**2
        p += A*B
        q += B**2
    nn = 1./nn
    p *= 2.*nn
    q = (q-horizon2)*nn
    A = -p*0.5
    B = sqrt(A**2-q)
    c = A+B
    if (c < 0) or (c > 1):
        c = A-B
    return c


cdef class fractionalLaplacian2DZeroExterior(nonlocalLaplacian2D):
    def __init__(self, FractionalKernel kernel, meshBase mesh, DoFMap DoFMap, num_dofs=None, **kwargs):
        manifold_dim2 = mesh.dim-1
        super(fractionalLaplacian2DZeroExterior, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2=manifold_dim2, **kwargs)
        self.symmetricCells = False


cdef class fractionalLaplacian2D_P1(nonlocalLaplacian2D):
    def __init__(self,
                 FractionalKernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 target_order=None,
                 quad_order_diagonal=None,
                 num_dofs=None,
                 **kwargs):
        assert isinstance(DoFMap, P1_DoFMap)
        super(fractionalLaplacian2D_P1, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        if target_order is None:
            # this is the desired local quadrature error
            # target_order = (2.-s)/self.dim
            target_order = 0.5
        self.target_order = target_order

        smax = self.kernel.s.max
        if quad_order_diagonal is None:
            # measured log(2 rho_2) = 0.43
            quad_order_diagonal = max(np.ceil((target_order+1.+smax)/(0.43)*abs(np.log(self.hmin/self.H0))), 4)
            # measured log(2 rho_2) = 0.7
            quad_order_diagonalV = max(np.ceil((target_order+1.+smax)/(0.7)*abs(np.log(self.hmin/self.H0))), 4)
        else:
            quad_order_diagonalV = quad_order_diagonal
        self.quad_order_diagonal = quad_order_diagonal
        self.quad_order_diagonalV = quad_order_diagonalV

        self.x = uninitialized((0, self.dim), dtype=REAL)
        self.y = uninitialized((0, self.dim), dtype=REAL)
        self.temp = uninitialized((0), dtype=REAL)
        self.idx1 = uninitialized((self.dim+1), dtype=INDEX)
        self.idx2 = uninitialized((self.dim+1), dtype=INDEX)
        self.idx3 = uninitialized((2*(self.dim+1)), dtype=INDEX)
        self.idx4 = uninitialized(((2*self.DoFMap.dofs_per_element)*(2*self.DoFMap.dofs_per_element+1)//2), dtype=INDEX)

        if not self.kernel.variableOrder:
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
            REAL_t s = (<FractionalKernel>self.kernel).getsValue()
        panel = <panelType>max(ceil((c + (s-1.)*logh2H0 + loghminH0 - s*logdh2) /
                                    (max(logdh1, 0) + 0.4)),
                               2)
        panel2 = <panelType>max(ceil((c + (s-1.)*logh1H0 + loghminH0 - s*logdh1) /
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
                    PSI[I, k] = sf.evalStrided(&qr0.nodes[0, i], numQuadNodes0)
                    k += 1
        # phi_i(x) - phi_i(y) = -phi_i(y) for i = 3,4,5
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    PSI[I+dofs_per_element, k] = -sf.evalStrided(&qr1.nodes[0, j], numQuadNodes1)
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
            REAL_t s = self.kernel.sValue
            REAL_t eta0, eta1, eta2, eta3
            specialQuadRule sQR0, sQR1
            quadQuadratureRule qrId, qrEdge0, qrEdge1, qrVertex
            REAL_t[:, :, ::1] PSI_id, PSI_edge, PSI_vertex
        if panel == COMMON_FACE:
            try:
                sQR0 = self.specialQuadRules[(s, panel, 0)]
            except KeyError:
                # COMMON_FACE panels have 3 integral contributions.
                # Each integral is over a 1D domain.
                qrId = GaussJacobi(((1, 3-2*s, 0),
                                    (1, 2-2*s, 0),
                                    (1, 1-2*s, 0),
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
                self.specialQuadRules[(s, panel, 0)] = sQR0
                if qrId.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrId.num_nodes), dtype=REAL)
            self.qrId = sQR0.qr
            self.PSI_id = sQR0.PSI3
        elif panel == COMMON_EDGE:
            try:
                sQR0 = self.specialQuadRules[(s, panel, 0)]
                sQR1 = self.specialQuadRules[(s, panel, 1)]
            except KeyError:
                qrEdge0 = GaussJacobi(((1, 3-2*s, 0),
                                       (1, 2-2*s, 0),
                                       (self.quad_order_diagonal, 0, 0),
                                       (self.quad_order_diagonal, 0, 0)))
                qrEdge1 = GaussJacobi(((1, 3-2*s, 0),
                                       (1, 2-2*s, 0),
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
                self.specialQuadRules[(s, panel, 0)] = sQR0
                self.specialQuadRules[(s, panel, 1)] = sQR1
                if qrEdge0.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrEdge0.num_nodes), dtype=REAL)
                if qrEdge1.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrEdge1.num_nodes), dtype=REAL)
            self.qrEdge0 = sQR0.qr
            self.qrEdge1 = sQR1.qr
            self.PSI_edge = sQR0.PSI3
        elif panel == COMMON_VERTEX:
            try:
                sQR0 = self.specialQuadRules[(s, panel, 0)]
            except KeyError:
                qrVertex = GaussJacobi(((1, 3-2*s, 0),
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
                self.specialQuadRules[(s, panel, 0)] = sQR0
                if qrVertex.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrVertex.num_nodes), dtype=REAL)
            self.qrVertex = sQR0.qr
            self.PSI_vertex = sQR0.PSI3
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

    def __repr__(self):
        return (super(fractionalLaplacian2D_P1, self).__repr__() +
                'hmin:                          {:.3}\n'.format(self.hmin) +
                'H0:                            {:.3}\n'.format(self.H0) +
                'target order:                  {}\n'.format(self.target_order) +
                'quad_order_diagonal:           {}\n'.format(self.quad_order_diagonal) +
                'quad_order_off_diagonal:       {}\n'.format(list(self.distantQuadRules.keys())))

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, l, m, I, J, k2
            REAL_t vol, val, temp
            REAL_t vol1 = self.vol1, vol2 = self.vol2
            INDEX_t[::1] idx1, idx2, idx3, idx4
            INDEX_t numQuadNodes, numQuadNodes0, numQuadNodes1, dofs_per_element
            specialQuadRule sQR
            doubleSimplexQuadratureRule qr2
            quadQuadratureRule qrEdge
            REAL_t[:, ::1] PSI
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            REAL_t s = (<FractionalKernel>self.kernel).getsValue()
            REAL_t scaling = self.kernel.getScalingValue()
            BOOL_t cutElements = False
            REAL_t horizon2
            simplexQuadratureRule qr0, qr1
            transformQuadratureRule qr1trans
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
            numQuadNodes0 = qr2.rule1.num_nodes
            numQuadNodes1 = qr2.rule2.num_nodes
            qr2.rule1.nodesInGlobalCoords(simplex1, self.x)
            qr2.rule2.nodesInGlobalCoords(simplex2, self.y)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    self.temp[k] = (qr2.weights[k] *
                                    self.kernel.evalPtr(2,
                                                        &self.x[i, 0],
                                                        &self.y[j, 0]))
                    k += 1
            vol = vol1 * vol2
            # loop over all local DoFs
            k = 0
            for I in range(6):
                for J in range(I, 6):
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
            qr1trans = transformQuadratureRule(qr1)
            numQuadNodes0 = qr0.num_nodes
            numQuadNodes1 = qr1.num_nodes

            horizon2 = self.kernel.getHorizonValue2()
            vol = vol1*vol2
            dofs_per_element = self.DoFMap.dofs_per_element

            A1 = a_A1
            A2 = a_A2
            b1 = a_b1
            b2 = a_b2

            ind = a_ind
            qr0.nodesInGlobalCoords(simplex1, self.x)
            for i in range(qr0.num_nodes):
                numInside = 0
                for j in range(3):
                    d2 = 0.
                    for k in range(2):
                        d2 += (simplex2[j, k]-self.x[i, k])**2
                    ind[j] = (d2 <= horizon2)
                    numInside += ind[j]
                if numInside == 0:
                    continue
                elif numInside == 1:
                    inside = 0
                    while not ind[inside]:
                        inside += 1
                    outside1 = (inside+1)%3
                    outside2 = (inside+2)%3
                    c1 = findIntersection(self.x[i, :], simplex2[inside, :], simplex2[outside1, :], horizon2)
                    c2 = findIntersection(self.x[i, :], simplex2[inside, :], simplex2[outside2, :], horizon2)
                    A1[:, :] = 0.
                    b1[:] = 0.
                    A1[inside,inside] = c1+c2
                    A1[inside,outside1] = c2
                    A1[inside,outside2] = c1
                    A1[outside1,outside1] = c1
                    A1[outside2,outside2] = c2
                    b1[inside] = 1-c1-c2
                    vol3 = c1*c2
                    qr1trans.setBaryTransform(A1, b1)
                    qr1 = qr1trans
                elif numInside == 2:
                    # outside = np.where(ind == False)[0][0]
                    outside = 0
                    while ind[outside]:
                        outside += 1
                    inside1 = (outside+1)%3
                    inside2 = (outside+2)%3
                    c1 = findIntersection(self.x[i,: ], simplex2[outside, :], simplex2[inside1, :], horizon2)
                    c2 = findIntersection(self.x[i,: ], simplex2[outside, :], simplex2[inside2, :], horizon2)
                    d1 = 0.
                    d2 = 0.
                    for k in range(2):
                        d1 += (simplex2[outside, k]
                               + c1*(simplex2[inside1, k]-simplex2[outside, k])
                               - simplex2[inside2, k])**2
                        d2 += (simplex2[outside, k]
                               + c2*(simplex2[inside2, k]-simplex2[outside, k])
                               - simplex2[inside1, k])
                    A1[:, :] = 0.
                    b1[:] = 0.
                    A2[:, :] = 0.
                    b2[:] = 0.

                    if d1 < d2:
                        A1[outside,outside] = 1-c1
                        A1[inside1,inside1] = 1-c1
                        A1[inside1,inside2] = -c1
                        A1[inside2,inside2] = 1.
                        b1[inside1] = c1
                        vol3 = 1-c1

                        A2[outside,outside] = 1-c2
                        A2[inside2,inside2] = 1
                        A2[inside2,outside] = c2
                        A2[outside,inside1] = 1-c1
                        A2[inside1,inside1] = c1
                        vol4 = c1*(1-c2)
                    else:
                        A1[outside,outside] = 1-c2
                        A1[inside2,inside2] = 1-c2
                        A1[inside2,inside1] = -c2
                        A1[inside1,inside1] = 1.
                        b1[inside2] = c2
                        vol3 = 1-c2

                        A2[outside,outside] = 1-c1
                        A2[inside1,inside1] = 1
                        A2[inside1,outside] = c1
                        A2[outside,inside2] = 1-c2
                        A2[inside2,inside2] = c2
                        vol4 = c2*(1-c1)

                    qr1trans.setBaryTransform(A1, b1)
                    qr1 = qr1trans
                else:
                    qr1 = qr2.rule2
                    vol3 = 1.

                qr1.nodesInGlobalCoords(simplex2, self.y)
                for j in range(qr1.num_nodes):
                    val = qr0.weights[i]*qr1.weights[j]*self.kernel.evalPtr(2, &self.x[i, 0], &self.y[j, 0])
                    val *= vol*vol3

                    k = 0
                    for I in range(6):
                        if I < dofs_per_element:
                            PSI_I = self.getLocalShapeFunction(I).evalStrided(&qr0.nodes[0, i], numQuadNodes0)
                        else:
                            PSI_I = -self.getLocalShapeFunction(I-dofs_per_element).evalStrided(&qr1.nodes[0, j], numQuadNodes1)
                        for J in range(I, 6):
                            if mask & (1 << k):
                                if J < dofs_per_element:
                                    PSI_J = self.getLocalShapeFunction(J).evalStrided(&qr0.nodes[0, i], numQuadNodes0)
                                else:
                                    PSI_J = -self.getLocalShapeFunction(J-dofs_per_element).evalStrided(&qr1.nodes[0, j], numQuadNodes1)
                                contrib[k] += val * PSI_I*PSI_J
                            k += 1
                if numInside == 2:
                    qr1trans.setBaryTransform(A2, b2)
                    qr1.nodesInGlobalCoords(simplex2, self.y)
                    for j in range(qr1.num_nodes):
                        val = qr0.weights[i]*qr1.weights[j]*self.kernel.evalPtr(2, &self.x[i, 0], &self.y[j, 0])
                        val *= vol*vol4

                        k = 0
                        for I in range(6):
                            if I < dofs_per_element:
                                PSI_I = self.getLocalShapeFunction(I).evalStrided(&qr0.nodes[0, i], numQuadNodes0)
                            else:
                                PSI_I = -self.getLocalShapeFunction(I-dofs_per_element).evalStrided(&qr1.nodes[0, j], numQuadNodes1)
                            for J in range(I, 6):
                                if mask & (1 << k):
                                    if J < dofs_per_element:
                                        PSI_J = self.getLocalShapeFunction(J).evalStrided(&qr0.nodes[0, i], numQuadNodes0)
                                    else:
                                        PSI_J = -self.getLocalShapeFunction(J-dofs_per_element).evalStrided(&qr1.nodes[0, j], numQuadNodes1)
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
                    self.temp[i] = self.qrId.weights[i]*pow(temp, -1.-s)
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
                    self.temp[i] = qrEdge.weights[i]*pow(temp, -1.-s)

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
                    self.temp[i] = self.qrVertex.weights[i]*pow(temp, -1.-s)

                # loop over all local DoFs
                for I in range(5):
                    i = idx3[I]
                    for J in range(I, 5):
                        j = idx3[J]
                        if j < i:
                            k = 6*j-(j*(j+1) >> 1) + i
                        else:
                            k = 6*i-(i*(i+1) >> 1) + j
                        if mask & (1 << k):
                            val = 0.
                            for k2 in range(numQuadNodes):
                                val += (self.temp[k2] *
                                        self.PSI_vertex[l, I, k2] *
                                        self.PSI_vertex[l, J, k2])
                            contrib[k] += val*vol
        else:
            raise NotImplementedError('Panel type unknown: {}'.format(panel))


cdef class fractionalLaplacian2D_P1_boundary(fractionalLaplacian2DZeroExterior):
    def __init__(self,
                 FractionalKernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 target_order=None,
                 quad_order_diagonal=None,
                 num_dofs=None,
                 **kwargs):
        assert isinstance(DoFMap, P1_DoFMap)
        super(fractionalLaplacian2D_P1_boundary, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        smax = self.kernel.s.max
        if target_order is None:
            # this is the desired global order wrt to the number of DoFs
            # target_order = (2.-s)/self.dim
            target_order = 0.5
        self.target_order = target_order
        self.distantPHI = {}

        if quad_order_diagonal is None:
            # measured log(2 rho_2) = 0.4
            quad_order_diagonal = max(np.ceil((target_order+0.5+smax)/(0.35)*abs(np.log(self.hmin/self.H0))), 2)
        self.quad_order_diagonal = quad_order_diagonal

        self.x = uninitialized((0, self.dim), dtype=REAL)
        self.y = uninitialized((0, self.dim), dtype=REAL)
        self.temp = uninitialized((0), dtype=REAL)

        self.n = uninitialized((self.dim), dtype=REAL)
        self.w = uninitialized((self.dim), dtype=REAL)

        self.idx1 = uninitialized((self.dim+1), dtype=INDEX)
        self.idx2 = uninitialized((self.dim), dtype=INDEX)

        if not self.kernel.variableOrder:
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
            REAL_t logdh1 = max(log(d/h1), 0.), logdh2 = max(log(d/h2), 0.)
            REAL_t logh1H0 = abs(log(h1/self.H0)), logh2H0 = abs(log(h2/self.H0))
            REAL_t loghminH0 = max(logh1H0, logh2H0)
            REAL_t s = self.kernel.sValue
            REAL_t h
        panel = <panelType>max(ceil(((0.5*self.target_order+0.25)*log(self.num_dofs*self.H0**2) + loghminH0 + (s-1.)*logh2H0 - s*logdh2) /
                                    (max(logdh1, 0) + 0.35)),
                               2)
        panel2 = <panelType>max(ceil(((0.5*self.target_order+0.25)*log(self.num_dofs*self.H0**2) + loghminH0 + (s-1.)*logh1H0 - s*logdh1) /
                                     (max(logdh2, 0) + 0.35)),
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

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void addQuadRule(self, panelType panel):
        cdef:
            simplexQuadratureRule qr0, qr1
            doubleSimplexQuadratureRule qr2
            REAL_t[:, ::1] PHI
            INDEX_t i, j, k, l
        qr0 = simplexXiaoGimbutas(panel, self.dim)
        qr1 = simplexDuffyTransformation(panel, self.dim, self.dim-1)
        qr2 = doubleSimplexQuadratureRule(qr0, qr1)
        self.distantQuadRules[panel] = qr2
        PHI = uninitialized((3, qr2.num_nodes), dtype=REAL)
        for i in range(3):
            for j in range(qr2.rule1.num_nodes):
                for k in range(qr2.rule2.num_nodes):
                    l = j*qr2.rule2.num_nodes+k
                    PHI[i, l] = self.getLocalShapeFunction(i)(qr2.rule1.nodes[:, j])
        self.distantPHI[panel] = PHI

        if qr2.rule1.num_nodes > self.x.shape[0]:
            self.x = uninitialized((qr2.rule1.num_nodes, self.dim), dtype=REAL)
        if qr2.rule2.num_nodes > self.y.shape[0]:
            self.y = uninitialized((qr2.rule2.num_nodes, self.dim), dtype=REAL)
        if qr2.num_nodes > self.temp.shape[0]:
            self.temp = uninitialized((qr2.num_nodes), dtype=REAL)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void getNearQuadRule(self, panelType panel):
        cdef:
            INDEX_t i
            REAL_t s = self.kernel.sValue
            REAL_t eta0, eta1, eta2, x, y
            specialQuadRule sQR0, sQR1
        if panel == COMMON_EDGE:
            try:
                sQR0 = self.specialQuadRules[(s, panel, 0)]
            except KeyError:
                if s < 0.5:
                    qrEdge = GaussJacobi(((2, -2.*s, 1.),
                                          (self.quad_order_diagonal, 0., 0.),
                                          (2, 0., 0.)))
                    PHI_edge = uninitialized((3, 3, qrEdge.num_nodes), dtype=REAL)
                    PSI_edge = uninitialized((3, 3, qrEdge.num_nodes), dtype=REAL)
                    for i in range(qrEdge.num_nodes):
                        eta0 = qrEdge.nodes[0, i]
                        eta1 = qrEdge.nodes[1, i]
                        eta2 = qrEdge.nodes[2, i]

                        # int 0
                        x = eta0 + (1.-eta0)*eta2
                        y = eta0*eta1

                        PHI_edge[0, 0, i] = 1.-x
                        PHI_edge[0, 1, i] = x-y
                        PHI_edge[0, 2, i] = y

                        PSI_edge[0, 0, i] = -1.
                        PSI_edge[0, 1, i] = 1.-eta1
                        PSI_edge[0, 2, i] = eta1

                        # int 1
                        x = eta0 + (1.-eta0)*eta2
                        y = eta0

                        PHI_edge[1, 0, i] = 1.-x
                        PHI_edge[1, 1, i] = x-y
                        PHI_edge[1, 2, i] = y

                        PSI_edge[1, 0, i] = -eta1
                        PSI_edge[1, 1, i] = eta1-1.
                        PSI_edge[1, 2, i] = 1.

                        # int 2
                        x = eta0*eta1 + (1.-eta0)*eta2
                        y = eta0*eta1

                        PHI_edge[2, 0, i] = 1.-x
                        PHI_edge[2, 1, i] = x-y
                        PHI_edge[2, 2, i] = y

                        PSI_edge[2, 0, i] = 1.-eta1
                        PSI_edge[2, 1, i] = -1.
                        PSI_edge[2, 2, i] = eta1
                else:
                    qrEdge = GaussJacobi(((2, 2.-2.*s, 1.),
                                          (self.quad_order_diagonal, 0., 0.),
                                          (2, 0., 0.)))
                    PHI_edge = uninitialized((3, 3, qrEdge.num_nodes), dtype=REAL)
                    PSI_edge = uninitialized((3, 3, qrEdge.num_nodes), dtype=REAL)
                    for i in range(qrEdge.num_nodes):
                        eta0 = qrEdge.nodes[0, i]
                        eta1 = qrEdge.nodes[1, i]
                        eta2 = qrEdge.nodes[2, i]

                        # int 0
                        x = eta0 + (1.-eta0)*eta2
                        y = eta1

                        PHI_edge[0, 0, i] = 0.
                        PHI_edge[0, 1, i] = 0.
                        PHI_edge[0, 2, i] = y

                        PSI_edge[0, 0, i] = -1.
                        PSI_edge[0, 1, i] = 1.-eta1
                        PSI_edge[0, 2, i] = eta1

                        # int 1
                        x = eta0 + (1.-eta0)*eta2
                        y = 1.

                        PHI_edge[1, 0, i] = 0.
                        PHI_edge[1, 1, i] = 0.
                        PHI_edge[1, 2, i] = y

                        PSI_edge[1, 0, i] = -eta1
                        PSI_edge[1, 1, i] = eta1-1.
                        PSI_edge[1, 2, i] = 1.

                        # int 2
                        x = eta0*eta1 + (1.-eta0)*eta2
                        y = eta1

                        PHI_edge[2, 0, i] = 0.
                        PHI_edge[2, 1, i] = 0.
                        PHI_edge[2, 2, i] = y

                        PSI_edge[2, 0, i] = 1.-eta1
                        PSI_edge[2, 1, i] = -1.
                        PSI_edge[2, 2, i] = eta1

                sQR0 = specialQuadRule(qrEdge, PSI3=PSI_edge, PHI3=PHI_edge)
                self.specialQuadRules[(s, panel, 0)] = sQR0
                if qrEdge.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrEdge.num_nodes), dtype=REAL)
            self.qrEdge = sQR0.qr
            self.PSI_edge = sQR0.PSI3
            self.PHI_edge = sQR0.PHI3
        elif panel == COMMON_VERTEX:
            try:
                sQR0 = self.specialQuadRules[(s, panel, 0)]
                sQR1 = self.specialQuadRules[(s, panel, 1)]
            except KeyError:
                qrVertex0 = GaussJacobi(((2, 1.0-2.0*s, 0),
                                         (self.quad_order_diagonal, 0, 0),
                                         (self.quad_order_diagonal, 0, 0)))
                qrVertex1 = GaussJacobi(((2, 1.0-2.0*s, 0),
                                         (self.quad_order_diagonal, 1.0, 0),
                                         (self.quad_order_diagonal, 0, 0)))
                PHI_vertex = uninitialized((2, 3, qrVertex0.num_nodes), dtype=REAL)
                PSI_vertex = uninitialized((2, 4, qrVertex0.num_nodes), dtype=REAL)
                for i in range(qrVertex0.num_nodes):
                    eta0 = qrVertex0.nodes[0, i]
                    eta1 = qrVertex0.nodes[1, i]
                    eta2 = qrVertex0.nodes[2, i]

                    # int 0
                    x = eta0
                    y = eta0*eta1

                    PHI_vertex[0, 0, i] = 1.-x
                    PHI_vertex[0, 1, i] = x-y
                    PHI_vertex[0, 2, i] = y

                    PSI_vertex[0, 0, i] = eta2-1.
                    PSI_vertex[0, 1, i] = 1.-eta1
                    PSI_vertex[0, 2, i] = eta1
                    PSI_vertex[0, 3, i] = -eta2

                    # int 1
                    eta0 = qrVertex1.nodes[0, i]
                    eta1 = qrVertex1.nodes[1, i]
                    eta2 = qrVertex1.nodes[2, i]

                    x = eta0*eta1
                    y = eta0*eta1*eta2

                    PHI_vertex[1, 0, i] = 1.-x
                    PHI_vertex[1, 1, i] = x-y
                    PHI_vertex[1, 2, i] = y

                    PSI_vertex[1, 0, i] = 1.-eta1
                    PSI_vertex[1, 1, i] = eta1*(1.-eta2)
                    PSI_vertex[1, 2, i] = eta1*eta2
                    PSI_vertex[1, 3, i] = -1.

                sQR0 = specialQuadRule(qrVertex0, PSI3=PSI_vertex, PHI3=PHI_vertex)
                sQR1 = specialQuadRule(qrVertex1, PSI3=PSI_vertex, PHI3=PHI_vertex)
                self.specialQuadRules[(s, panel, 0)] = sQR0
                self.specialQuadRules[(s, panel, 1)] = sQR1
                if qrVertex0.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrVertex0.num_nodes), dtype=REAL)
                if qrVertex1.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qrVertex1.num_nodes), dtype=REAL)
            self.qrVertex0 = sQR0.qr
            self.qrVertex1 = sQR1.qr
            self.PSI_vertex = sQR0.PSI3
            self.PHI_vertex = sQR0.PHI3
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

    def __repr__(self):
        return (super(fractionalLaplacian2D_P1_boundary, self).__repr__() +
                'hmin:                          {:.3}\n'.format(self.hmin) +
                'H0:                            {:.3}\n'.format(self.H0) +
                'target order:                  {}\n'.format(self.target_order) +
                'quad_order_diagonal:           {}\n'.format(self.quad_order_diagonal) +
                'quad_order_off_diagonal        {}\n'.format(list(self.distantQuadRules.keys())))

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            REAL_t vol1 = self.vol1, vol2 = self.vol2, vol
            INDEX_t l, i, j, k, m, I, J
            set K1, K2
            INDEX_t[::1] idx1 = self.idx1, idx2 = self.idx2
            doubleSimplexQuadratureRule qr2
            quadQuadratureRule qrVertex
            REAL_t[:, ::1] PHI
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            REAL_t s = self.kernel.sValue
            REAL_t scaling = self.kernel.scalingValue
        self.n[0] = simplex2[1, 1] - simplex2[0, 1]
        self.n[1] = simplex2[0, 0] - simplex2[1, 0]
        # F is same as vol2
        val = 1./sqrt(mydot(self.n, self.n))
        self.n[0] *= val
        self.n[1] *= val

        contrib[:] = 0.

        if panel == COMMON_EDGE:
            # find reordering of cell and edge so that the singularity
            # is on the first edge of the cell
            K1 = set()
            for i in range(3):
                for j in range(2):
                    if simplex1[i, 0] == simplex2[j, 0] and simplex1[i, 1] == simplex2[j, 1]:
                        K1.add(i)
            if K1 == set([0, 1]):
                idx1[0], idx1[1], idx1[2] = 0, 1, 2
            elif K1 == set([1, 2]):
                idx1[0], idx1[1], idx1[2] = 1, 2, 0
            elif K1 == set([2, 0]):
                idx1[0], idx1[1], idx1[2] = 2, 0, 1
            else:
                raise NotImplementedError("Something went wrong for COMMON_EDGE")

            vol = -scaling*2.0*vol1*vol2/s

            # We need to calculate 3 integrals
            for l in range(3):
                for i in range(self.qrEdge.num_nodes):
                    for j in range(2):
                        self.w[j] = (simplex1[idx1[0], j]*self.PSI_edge[l, 0, i] +
                                     simplex1[idx1[1], j]*self.PSI_edge[l, 1, i] +
                                     simplex1[idx1[2], j]*self.PSI_edge[l, 2, i])
                    self.temp[i] = self.qrEdge.weights[i] * mydot(self.n, self.w) * pow(mydot(self.w, self.w), -1.-s)
                for I in range(3):
                    for J in range(I, 3):
                        val = 0.
                        for i in range(self.qrEdge.num_nodes):
                            val += (self.temp[i] *
                                    self.PHI_edge[l, I, i] *
                                    self.PHI_edge[l, J, i])
                        i = idx1[I]
                        j = idx1[J]
                        if j < i:
                            i, j = j, i
                        k = 4*i-(i*(i+1))//2 + j-i
                        contrib[k] += val*vol
        elif panel == COMMON_VERTEX:
            K1 = set()
            K2 = set()
            i = 0
            j = 0
            while True:
                if simplex1[i, 0] == simplex2[j, 0] and simplex1[i, 1] == simplex2[j, 1]:
                    break
                if j == 1:
                    i += 1
                    j = 0
                else:
                    j += 1
            if i == 0:
                idx1[0], idx1[1], idx1[2] = 0, 1, 2
            elif i == 1:
                idx1[0], idx1[1], idx1[2] = 1, 2, 0
            else:
                idx1[0], idx1[1], idx1[2] = 2, 0, 1

            if j == 0:
                idx2[0], idx2[1] = 0, 1
            else:
                idx2[0], idx2[1] = 1, 0

            vol = -scaling*2.0*vol1*vol2/s

            for l in range(2):
                if l == 0:
                    qrVertex = self.qrVertex0
                else:
                    qrVertex = self.qrVertex1

                for i in range(qrVertex.num_nodes):
                    for j in range(2):
                        self.w[j] = (simplex1[idx1[0], j]*self.PSI_vertex[l, 0, i] +
                                     simplex1[idx1[1], j]*self.PSI_vertex[l, 1, i] +
                                     simplex1[idx1[2], j]*self.PSI_vertex[l, 2, i] +
                                     simplex2[idx2[1], j]*self.PSI_vertex[l, 3, i])
                    self.temp[i] = qrVertex.weights[i] * mydot(self.n, self.w) * pow(mydot(self.w, self.w), -1.-s)
                for I in range(3):
                    for J in range(I, 3):
                        val = 0.
                        for i in range(qrVertex.num_nodes):
                            val += (self.temp[i] *
                                    self.PHI_vertex[l, I, i] *
                                    self.PHI_vertex[l, J, i])
                        i = idx1[I]
                        j = idx1[J]
                        if j < i:
                            i, j = j, i
                        k = 4*i-(i*(i+1))//2 + j-i
                        contrib[k] += val*vol
        elif panel >= 1:
            qr2 = self.distantQuadRules[panel]
            PHI = self.distantPHI[panel]
            qr2.rule1.nodesInGlobalCoords(simplex1, self.x)
            qr2.rule2.nodesInGlobalCoords(simplex2, self.y)
            for k in range(qr2.rule1.num_nodes):
                for m in range(qr2.rule2.num_nodes):
                    for j in range(2):
                        self.w[j] = self.y[m, j]-self.x[k, j]
                    i = k*qr2.rule2.num_nodes+m
                    self.temp[i] = qr2.weights[i] * mydot(self.n, self.w) * pow(mydot(self.w, self.w), -1.-s)
            vol = scaling*vol1*vol2/s
            k = 0
            for i in range(3):
                for j in range(i, 3):
                    val = 0.
                    for m in range(qr2.num_nodes):
                        val += self.temp[m]*PHI[i, m]*PHI[j, m]
                    contrib[k] = val*vol
                    k += 1
        else:
            raise NotImplementedError('Panel type unknown: {}.'.format(panel))


