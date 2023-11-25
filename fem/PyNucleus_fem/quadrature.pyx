###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes import REAL
from PyNucleus_base import uninitialized
from PyNucleus_base.blas cimport uninitializedREAL
from libc.math cimport sin, cos, M_PI as pi
import numpy as np
from modepy import XiaoGimbutasSimplexQuadrature
from modepy.tools import unit_to_barycentric


cdef class quadratureRule:
    def __init__(self,
                 REAL_t[:, ::1] nodes,  # in barycentric coordinates
                 REAL_t[::1] weights,
                 INDEX_t dim,
                 manifold_dim=None):
        assert nodes.shape[1] == weights.shape[0]
        self.dim = dim
        self.nodes = nodes
        self.num_nodes = self.nodes.shape[1]
        if manifold_dim is None:
            self.manifold_dim = self.dim
        else:
            self.manifold_dim = manifold_dim
        self.weights = weights

    cdef inline REAL_t eval(self,
                            const REAL_t[::1] fun_vals,
                            const REAL_t vol):
        cdef:
            INDEX_t i
            REAL_t I
        I = 0.
        for i in range(self.num_nodes):
            I += self.weights[i]*fun_vals[i]
        return I*vol


cdef class simplexQuadratureRule(quadratureRule):
    def __init__(self,
                 REAL_t[:, ::1] nodes,  # in barycentric coordinates
                 REAL_t[::1] weights,
                 INDEX_t dim,
                 manifold_dim=None):
        quadratureRule.__init__(self, nodes, weights, dim, manifold_dim)
        if self.manifold_dim == 0:
            self.volume = volume0D
        elif self.dim == 1 and self.manifold_dim == 1:
            self.volume = volume1Dnew
        elif self.dim == 2 and self.manifold_dim == 1:
            self.volume = volume1D_in_2D
        elif self.dim == 2 and self.manifold_dim == 2:
            self.volume = volume2Dnew
        elif self.dim == 3 and self.manifold_dim == 3:
            self.volume = volume3D
        elif self.dim == 3 and self.manifold_dim == 2:
            self.volume = volume2D_in_3D
        else:
            raise NotImplementedError('dim={}'.format(self.dim))
        self.span = uninitialized((self.manifold_dim, self.dim), dtype=REAL)
        self.tempVec = uninitializedREAL((self.dim, ))

    def __add__(self, simplexQuadratureRule other):
        assert self.dim == other.dim
        assert self.manifold_dim == other.manifold_dim
        return simplexQuadratureRule(np.hstack((self.nodes, other.nodes)),
                                     np.concatenate((self.weights, other.weights)),
                                     self.dim, self.manifold_dim)

    cdef inline void nodesInGlobalCoords(self,
                                         const REAL_t[:, ::1] simplexVertices,
                                         REAL_t[:, ::1] coords):
        cdef:
            INDEX_t i, k, m
            REAL_t temp
        coords[:] = 0.
        for k in range(self.manifold_dim+1):
            for i in range(self.num_nodes):
                temp = self.nodes[k, i]
                for m in range(self.dim):
                    coords[i, m] += temp * simplexVertices[k, m]

    def nodesInGlobalCoords_py(self,
                               const REAL_t[:, ::1] simplexVertices,
                               REAL_t[:, ::1] coords):
        self.nodesInGlobalCoords(simplexVertices, coords)

    cdef inline void nodesInGlobalCoordsReorderer(self,
                                                  const REAL_t[:, ::1] simplexVertices,
                                                  REAL_t[:, ::1] coords,
                                                  const INDEX_t[::1] idx):
        cdef:
            INDEX_t i, k, m, kk
            REAL_t temp
        coords[:] = 0.
        for k in range(self.manifold_dim+1):
            for i in range(self.num_nodes):
                temp = self.nodes[k, i]
                kk = idx[k]
                for m in range(self.dim):
                    coords[i, m] += temp * simplexVertices[kk, m]

    def getAllNodesInGlobalCoords(self, meshBase mesh):
        cdef:
            REAL_t[:, ::1] simplex = uninitializedREAL((mesh.dim+1, mesh.dim))
            REAL_t[:, ::1] coords = uninitializedREAL((mesh.num_cells*self.num_nodes, mesh.dim))
        for cellNo in range(mesh.num_cells):
            mesh.getSimplex(cellNo, simplex)
            self.nodesInGlobalCoords(simplex, coords[self.num_nodes*cellNo:self.num_nodes*(cellNo+1), :])
        return np.array(coords, copy=False)

    cpdef void evalFun(self,
                       function fun,
                       const REAL_t[:, ::1] simplexVertices,
                       REAL_t[::1] fun_vals):
        cdef:
            INDEX_t i, k, m
        for i in range(self.num_nodes):
            self.tempVec[:] = 0.
            for k in range(self.manifold_dim+1):
                for m in range(self.dim):
                    self.tempVec[m] += self.nodes[k, i] * simplexVertices[k, m]
            fun_vals[i] = fun.eval(self.tempVec)

    cpdef void evalVectorFun(self,
                             vectorFunction fun,
                             const REAL_t[:, ::1] simplexVertices,
                             REAL_t[:, ::1] fun_vals):
        cdef:
            INDEX_t i, k, m
        for i in range(self.num_nodes):
            self.tempVec[:] = 0.
            for k in range(self.manifold_dim+1):
                for m in range(self.dim):
                    self.tempVec[m] += self.nodes[k, i] * simplexVertices[k, m]
            fun.eval(self.tempVec, fun_vals[i, :])

    cpdef void evalComplexFun(self,
                              complexFunction fun,
                              const REAL_t[:, ::1] simplexVertices,
                              COMPLEX_t[::1] fun_vals):
        cdef:
            INDEX_t i, k, m
        for i in range(self.num_nodes):
            self.tempVec[:] = 0.
            for k in range(self.manifold_dim+1):
                for m in range(self.dim):
                    self.tempVec[m] += self.nodes[k, i] * simplexVertices[k, m]
            fun_vals[i] = fun.eval(self.tempVec)

    cdef REAL_t getSimplexVolume(self,
                                 const REAL_t[:, ::1] simplexVertices):
        cdef INDEX_t k, j
        for k in range(self.manifold_dim):
            for j in range(self.dim):
                self.span[k, j] = simplexVertices[k+1, j]-simplexVertices[0, j]
        return self.volume(self.span)

    def getSimplexVolume_py(self,
                            const REAL_t[:, ::1] simplexVertices):
        return self.getSimplexVolume(simplexVertices)

    def integrate(self,
                  function fun,
                  const REAL_t[:, ::1] simplexVertices):
        cdef:
            REAL_t vol
            REAL_t[::1] fun_vals = uninitializedREAL((self.num_nodes))
        self.evalFun(fun, simplexVertices, fun_vals)
        vol = self.getSimplexVolume(simplexVertices)
        return self.eval(fun_vals, vol)


cdef class transformQuadratureRule(simplexQuadratureRule):
    def __init__(self, simplexQuadratureRule qr):
        nodes = uninitializedREAL((qr.nodes.shape[0],
                                   qr.nodes.shape[1]))
        weights = qr.weights
        super(transformQuadratureRule, self).__init__(nodes, weights, qr.dim, qr.manifold_dim)
        self.qr = qr

    cpdef void setLinearBaryTransform(self, REAL_t[:, ::1] A):
        b = np.zeros((A.shape[0]), dtype=REAL)
        self.setAffineBaryTransform(A, b)

    cpdef void setAffineBaryTransform(self, REAL_t[:, ::1] A, REAL_t[::1] b):
        self.A = A
        self.b = b
        self.compute()

    cdef void compute(self):
        cdef:
            INDEX_t k, j, i
        for k in range(self.manifold_dim+1):
            for i in range(self.num_nodes):
                self.nodes[k, i] = self.b[k]
        for k in range(self.manifold_dim+1):
            for j in range(self.manifold_dim+1):
                for i in range(self.num_nodes):
                    self.nodes[k, i] += self.A[k, j]*self.qr.nodes[j, i]


cdef class doubleSimplexQuadratureRule(quadratureRule):
    def __init__(self,
                 simplexQuadratureRule rule1,
                 simplexQuadratureRule rule2):
        cdef:
            INDEX_t i, j, k
            REAL_t[::1] weights

        self.rule1 = rule1
        self.rule2 = rule2
        nodes = uninitialized((0, rule1.num_nodes*rule2.num_nodes), dtype=REAL)
        weights = uninitializedREAL((rule1.num_nodes*rule2.num_nodes, ))
        k = 0
        for i in range(rule1.num_nodes):
            for j in range(rule2.num_nodes):
                weights[k] = rule1.weights[i]*rule2.weights[j]
                k += 1
        quadratureRule.__init__(self,
                                nodes, weights,
                                rule1.dim+rule2.dim,
                                rule1.manifold_dim+rule2.manifold_dim)

    # cdef inline REAL_t eval(self,
    #                           const REAL_t[::1] fun_vals,
    #                           const REAL_t vol):
    #     cdef:
    #         INDEX_t i, j, k = 0
    #         REAL_t I = 0.
    #     for i in range(self.rule1.num_nodes):
    #         for j in range(self.rule2.num_nodes):
    #             I += self.rule1.weights[i]*self.rule2.weights[j]*fun_vals[k]
    #             k += 1
    #     return I*vol

    cpdef void evalFun(self,
                       function fun,
                       const REAL_t[:, ::1] simplexVertices1,
                       const REAL_t[:, ::1] simplexVertices2,
                       REAL_t[::1] fun_vals):
        cdef:
            INDEX_t i, j, k, m, l
            INDEX_t dim1 = self.rule1.dim
            INDEX_t dim2 = self.rule2.dim
            REAL_t[::1] x = uninitializedREAL((dim1+dim2))
        l = 0
        for i in range(self.rule1.num_nodes):
            for j in range(self.rule2.num_nodes):
                x[:] = 0.
                for k in range(dim1+1):
                    for m in range(dim1):
                        x[m] += self.rule1.nodes[k, i] * simplexVertices1[k, m]
                for k in range(dim2+1):
                    for m in range(dim2):
                        x[dim1+m] += self.rule2.nodes[k, j] * simplexVertices2[k, m]
                fun_vals[l] = fun(x)
                l += 1

    def integrate(self,
                  function fun,
                  const REAL_t[:, ::1] simplexVertices1,
                  const REAL_t[:, ::1] simplexVertices2):
        cdef:
            REAL_t vol
            REAL_t[::1] fun_vals = uninitializedREAL((self.rule1.num_nodes*self.rule2.num_nodes))

        self.evalFun(fun, simplexVertices1, simplexVertices2, fun_vals)
        vol = self.rule1.getSimplexVolume(simplexVertices1)*self.rule2.getSimplexVolume(simplexVertices2)
        return self.eval(fun_vals, vol)


quad_point2D_order2 = np.array([[0.5, 0.0, 0.5],
                                [0.5, 0.5, 0.0],
                                [0.0, 0.5, 0.5]], dtype=REAL)
weights2D_order2 = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=REAL)

cdef:
    REAL_t a1 = (6.0-sqrt(15.0))/21.0, a2 = (6.0+sqrt(15.0))/21.0
    REAL_t c1 = a1*(2.0*a1-1.0), c2 = a2*(2.0*a2-1.0)
    REAL_t d1 = (4.0*a1-1.0)*(2.0*a1-1.0), d2 = (4.0*a2-1.0)*(2.0*a2-1.0)
    REAL_t e1 = 4.0*a1**2, e2 = 4.0*a2**2
    REAL_t f1 = 4.0*a1*(1.0-2.0*a1), f2 = 4.0*a2*(1.0-2.0*a2)
    REAL_t w1 = (155.0-sqrt(15.0))/1200.0, w2 = (155.0+sqrt(15.0))/1200.0
quad_point2D_order5 = np.array([[1.0/3.0, a1, a1, 1.0-2.0*a1, a2, a2, 1.0-2.0*a2],
                                [1.0/3.0, a1, 1.0-2.0*a1, a1, a2, 1.0-2.0*a2, a2],
                                [1.0/3.0, 1.0-2.0*a1, a1, a1, 1.0-2.0*a2, a2, a2]], dtype=REAL)
weights2D_order5 = np.array([9.0/40.0, w1, w1, w1, w2, w2, w2], dtype=REAL)

quad_point3D_order3 = np.array([[0.25, 0.5,     1.0/6.0, 1.0/6.0, 1.0/6.0],
                                [0.25, 1.0/6.0, 0.5,     1.0/6.0, 1.0/6.0],
                                [0.25, 1.0/6.0, 1.0/6.0, 0.5,     1.0/6.0],
                                [0.25, 1.0/6.0, 1.0/6.0, 1.0/6.0, 0.5]], dtype=REAL)
weights3D_order3 = np.array([-0.8, 9.0/20.0, 9.0/20.0, 9.0/20.0, 9.0/20.0], dtype=REAL)


cdef class Gauss1D(simplexQuadratureRule):
    def __init__(self, order):
        k = (order+1)//2
        if 2*k-1 != order:
            raise NotImplementedError()
        from scipy.special import p_roots
        nodesT, weights = p_roots(k)
        nodes = uninitializedREAL((2, nodesT.shape[0]))
        for i in range(nodesT.shape[0]):
            nodes[0, i] = (nodesT[i]+1.)/2.
            nodes[1, i] = 1. - nodes[0, i]
        weights /= 2.
        super(Gauss1D, self).__init__(nodes, weights, 1)
        self.order = order


cdef class Gauss2D(simplexQuadratureRule):
    def __init__(self, INDEX_t order):
        if order == 2:
            super(Gauss2D, self).__init__(quad_point2D_order2,
                                          weights2D_order2,
                                          2)
        elif order == 5:
            super(Gauss2D, self).__init__(quad_point2D_order5,
                                          weights2D_order5,
                                          2)
        else:
            raise NotImplementedError()
        self.order = order


cdef class Gauss3D(simplexQuadratureRule):
    def __init__(self, INDEX_t order):
        if order == 3:
            super(Gauss3D, self).__init__(quad_point3D_order3,
                                          weights3D_order3,
                                          3)
        else:
            raise NotImplementedError()
        self.order = order


cdef class quadQuadratureRule(quadratureRule):
    def __init__(self, REAL_t[:, ::1] nodes, REAL_t[::1] weights):
        quadratureRule.__init__(self,
                                nodes, weights,
                                nodes.shape[0])
        if self.dim == 1:
            self.volume = volume1Dnew
        elif self.dim == 2:
            self.volume = volume2Dnew
        elif self.dim == 3:
            self.volume = volume3D
        else:
            # raise NotImplementedError()
            pass

    def __add__(self, quadQuadratureRule other):
        assert self.dim == other.dim
        return quadQuadratureRule(np.hstack((self.nodes, other.nodes)),
                                  np.concatenate((self.weights, other.weights)))

    # cpdef REAL_t eval(self,
    #                     REAL_t[::1] fun_vals,
    #                     REAL_t vol):
    #     cdef:
    #         INDEX_t i
    #         REAL_t I
    #     I = 0.
    #     for i in range(self.num_nodes):
    #         I += self.weights[i]*fun_vals[i]
    #     return I*vol

    cpdef void nodesInGlobalCoords(self,
                                   const REAL_t[:, ::1] quadVertices,
                                   REAL_t[:, ::1] coords):
        cdef:
            INDEX_t i, k, m
        coords[:] = 0.
        for i in range(self.num_nodes):
            for k in range(self.dim):
                for m in range(self.dim):
                    coords[i, m] += quadVertices[0, m] + self.nodes[i, k] * (quadVertices[k+1, m]-quadVertices[0, m])

    cpdef void evalFun(self,
                       function fun,
                       const REAL_t[:, ::1] quadVertices,
                       REAL_t[::1] fun_vals):
        cdef:
            INDEX_t i, k, m
            REAL_t[::1] x = uninitializedREAL((self.dim, ))
        for i in range(self.num_nodes):
            x[:] = 0.
            for k in range(self.dim):
                for m in range(self.dim):
                    x[m] += quadVertices[0, m] + self.nodes[i, k] * (quadVertices[k+1, m]-quadVertices[0, m])
                    fun_vals[i] = fun(x)

    cpdef REAL_t getQuadVolume(self,
                               const REAL_t[:, ::1] quadVertices):
        cdef:
            INDEX_t k, j
            REAL_t[:, ::1] span = uninitializedREAL((self.dim, self.dim))
            REAL_t vol
        for k in range(self.dim):
            for j in range(self.dim):
                span[k, j] = quadVertices[k+1, j]-quadVertices[0, j]
        vol = self.volume(span)
        if self.dim == 2:
            vol *= 2.
        elif self.dim == 3:
            vol *= 6.
        return vol

    def integrate(self,
                  function fun,
                  const REAL_t[:, ::1] quadVertices):
        cdef:
            REAL_t[::1] fun_vals = uninitializedREAL((self.num_nodes, ))
        self.evalFun(fun, quadVertices, fun_vals)
        vol = self.getQuadVolume(quadVertices)
        return self.eval(fun_vals, vol)


from itertools import product

cdef class Gauss(quadQuadratureRule):
    def __init__(self, order, dim):
        k = (order+1)//2
        if 2*k-1 != order:
            print('Incrementing order in Gauss quadrature rule, only odd orders are available.')
            k += 1
        from scipy.special import p_roots
        nodes1D, weights1D = p_roots(k)
        nodes = uninitializedREAL((dim, nodes1D.shape[0]**dim))
        weights = uninitializedREAL((nodes1D.shape[0]**dim))
        nodes1D = (nodes1D+1.)/2.
        weights1D /= 2.
        k = 0
        for idx in product(*([range(nodes1D.shape[0])]*dim)):
            for m in range(dim):
                nodes[m, k] = nodes1D[idx[m]]
            weights[k] = np.prod([weights1D[h] for h in idx])
            k += 1
        super(Gauss, self).__init__(nodes, weights)
        self.order = order


cdef class GaussJacobi(quadQuadratureRule):
    def __init__(self, order_weight_exponents):
        from scipy.special import js_roots
        nodes1D = []
        weights1D = []
        dim = len(order_weight_exponents)
        self.orders = []
        for order, alpha, beta in order_weight_exponents:
            k = (order+1)//2
            if 2*k-1 != order:
                # print('Incrementing order in Gauss-Jacobi quadrature rule, only odd orders are available.')
                k += 1
            self.orders.append(2*k-1)
            alpha = alpha+1
            beta = beta+alpha
            n1D, w1D = js_roots(k, beta, alpha)
            nodes1D.append(n1D)
            weights1D.append(w1D)
        nodes = uninitializedREAL((dim, np.prod([n1D.shape[0] for n1D in nodes1D])))
        weights = np.ones((nodes.shape[1]), dtype=REAL)
        k = 0
        for idx in product(*([range(n1D.shape[0]) for n1D in nodes1D])):
            for m in range(dim):
                nodes[m, k] = nodes1D[m][idx[m]]
                weights[k] *= weights1D[m][idx[m]]
            k += 1
        super(GaussJacobi, self).__init__(nodes, weights)
        self.order = order


cdef class simplexDuffyTransformation(simplexQuadratureRule):
    def __init__(self, order, dim, manifold_dim=None):
        cdef:
            list orders
            INDEX_t i, j, k
        if manifold_dim is None:
            manifold_dim = dim
        if manifold_dim == 0:
            nodes = np.ones((1, 1), dtype=REAL)
            weights = np.ones((1), dtype=REAL)
            super(simplexDuffyTransformation, self).__init__(nodes, weights, dim, manifold_dim)
            self.orders = [100]
            return
        weight_exponents = [(order+manifold_dim-d-1, 0, manifold_dim-d-1) for d in range(manifold_dim)]
        qr = GaussJacobi(weight_exponents)
        orders = qr.orders
        nodes = uninitializedREAL((manifold_dim+1, qr.num_nodes))
        for i in range(qr.num_nodes):
            for j in range(manifold_dim-1, -1, -1):
                nodes[j+1, i] = qr.nodes[j, i]
                for k in range(j):
                    nodes[j+1, i] *= (1.-qr.nodes[k, i])
            nodes[0, i] = 1.
            for j in range(manifold_dim):
                nodes[0, i] -= nodes[j+1, i]
        # adjust for volume of reference element
        if manifold_dim == 1:
            pass
        elif manifold_dim == 2:
            for i in range(qr.num_nodes):
                qr.weights[i] *= 2.
        elif manifold_dim == 3:
            for i in range(qr.num_nodes):
                qr.weights[i] *= 6.
        else:
            raise NotImplementedError('dim={}'.format(manifold_dim))
        super(simplexDuffyTransformation, self).__init__(nodes, qr.weights, dim, manifold_dim)
        self.orders = orders


cdef class simplexXiaoGimbutas(simplexQuadratureRule):
    def __init__(self, order, dim, manifold_dim=None):
        if manifold_dim is None:
            manifold_dim = dim

        if manifold_dim in (0, 1, ):
            qr = simplexDuffyTransformation(order, dim, manifold_dim)
            super(simplexXiaoGimbutas, self).__init__(qr.nodes, qr.weights,
                                                      dim, manifold_dim)
        else:
            qr = XiaoGimbutasSimplexQuadrature(order, manifold_dim)
            nodes = unit_to_barycentric(qr.nodes)
            num_nodes = nodes.shape[1]
            # adjust for volume of reference element
            if manifold_dim == 2:
                for i in range(num_nodes):
                    qr.weights[i] *= 0.5
            elif manifold_dim == 3:
                for i in range(num_nodes):
                    qr.weights[i] *= 0.75
            else:
                raise NotImplementedError('dim={}'.format(manifold_dim))
            self.order = qr.exact_to
            super(simplexXiaoGimbutas, self).__init__(nodes, qr.weights,
                                                      dim, manifold_dim)


cdef class sphericalQuadRule:
    def __init__(self, REAL_t[:, ::1] vertexOffsets, REAL_t[::1] weights):
        assert vertexOffsets.shape[0] == weights.shape[0]
        self.vertexOffsets = vertexOffsets
        self.weights = weights
        self.num_nodes = self.weights.shape[0]


cdef class sphericalQuadRule1D(sphericalQuadRule):
    def __init__(self, REAL_t radius):
        vertexOffsets = uninitializedREAL((2, 1))
        vertexOffsets[0, 0] = -radius
        vertexOffsets[1, 0] = radius
        weights = np.ones((2), dtype=REAL)
        sphericalQuadRule.__init__(self, vertexOffsets, weights)


cdef class sphericalQuadRule2D(sphericalQuadRule):
    def __init__(self, REAL_t radius, INDEX_t numQuadNodes):
        vertexOffsets = uninitializedREAL((numQuadNodes, 2))
        for i in range(numQuadNodes):
            angle = 2*pi*i/numQuadNodes
            vertexOffsets[i, 0] = radius*cos(angle)
            vertexOffsets[i, 1] = radius*sin(angle)
        weights = (2*pi*radius/numQuadNodes)*np.ones((numQuadNodes), dtype=REAL)
        sphericalQuadRule.__init__(self, vertexOffsets, weights)


cdef class simplexJaskowiecSukumar(simplexQuadratureRule):
    def __init__(self, order, dim, manifold_dim=None):
        cdef:
            REAL_t[:, ::1] nodes, bary_nodes
            INDEX_t i, j
        if manifold_dim is None:
            manifold_dim = dim

        if manifold_dim == 3:
            from . js_data import schemes

            nodes = np.array(schemes[order]['points'], dtype=REAL)
            weights = np.array(schemes[order]['weights'], dtype=REAL)
            num_nodes = nodes.shape[0]
            bary_nodes = uninitialized((manifold_dim+1, num_nodes), dtype=REAL)
            for i in range(num_nodes):
                bary_nodes[0, i] = 1.
                for j in range(1, manifold_dim+1):
                    bary_nodes[j, i] = nodes[i, j-1]
                    bary_nodes[0, i] -= nodes[i, j-1]
            self.order = order
            super(simplexJaskowiecSukumar, self).__init__(bary_nodes, weights,
                                                          dim, manifold_dim)
        else:
            raise NotImplementedError()
