###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from libc.math cimport (sqrt, log, ceil, fabs as abs, pow)
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc

from PyNucleus_base.myTypes import INDEX, REAL, BOOL
from PyNucleus_base import uninitialized, uninitialized_like
from PyNucleus_base.blas cimport mydot
from PyNucleus_fem.quadrature cimport (simplexQuadratureRule,
                                       transformQuadratureRule,
                                       doubleSimplexQuadratureRule, GaussJacobi,
                                       simplexXiaoGimbutas)
from PyNucleus_fem.DoFMaps cimport DoFMap, P0_DoFMap, P1_DoFMap, shapeFunction
from PyNucleus_nl.fractionalOrders cimport constFractionalOrder
# from . nonlocalLaplacianBase import ALL

include "kernel_params.pxi"
include "panelTypes.pxi"

cdef:
    MASK_t ALL
ALL.set()


cdef class fractionalLaplacian2DZeroExterior(nonlocalLaplacian2D):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap DoFMap, num_dofs=None, **kwargs):
        manifold_dim2 = mesh.dim-1
        super(fractionalLaplacian2DZeroExterior, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2=manifold_dim2, **kwargs)
        self.symmetricCells = False


cdef class singularityCancelationQuadRule2D(quadratureRule):
    def __init__(self, panelType panel,
                 REAL_t singularity,
                 INDEX_t quad_order_diagonal,
                 INDEX_t quad_order_diagonalV,
                 INDEX_t quad_order_regular):
        cdef:
            INDEX_t i
            REAL_t eta0, eta1, eta2, eta3, x1, x2, y1, y2
            quadratureRule qrId, qrEdge0, qrEdge1, qrVertex
            INDEX_t dim = 2
            REAL_t lcl_bary_x[3]
            REAL_t lcl_bary_y[3]
            REAL_t[:, ::1] bary, bary_x, bary_y
            REAL_t[::1] weights
            INDEX_t offset

        if panel == COMMON_FACE:
            # We obtain 6 subdomains from splitting the integral,
            # but pairs of 2 are symmetric wrt to exchange of K_1
            # and K_2. So we get 3 integrals, each with a weight
            # 2.

            #  Jacobian = eta0**3 * eta1**2 * eta2

            # We factor out (eta0 * eta1 * eta2) from each PSI and
            # (eta0 * eta1 * eta2)**singularity from the kernel.

            # Differences of basis functions one the same element
            # cancel one singularity order i.e.
            # singularityCancelationFromContinuity = 1.
            qrId = GaussJacobi(((1, 3+singularity, 0),
                                (1, 2+singularity, 0),
                                (1, 1+singularity, 0),
                                (quad_order_diagonal, 0, 0)))

            bary = uninitialized((2*dim+2,
                                  3*qrId.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            weights = uninitialized((3*qrId.num_nodes), dtype=REAL)

            # integral 0
            offset = 0
            for i in range(qrId.num_nodes):
                eta0 = qrId.nodes[0, i]
                eta1 = qrId.nodes[1, i]
                eta2 = qrId.nodes[2, i]
                eta3 = qrId.nodes[3, i]

                x1 = eta0
                x2 = eta0*eta1*(1-eta2+eta2*eta3)
                y1 = eta0*(1-eta1*eta2)
                y2 = eta0*eta1*(1-eta2)

                lcl_bary_x[0] = 1-x1
                lcl_bary_x[1] = x1-x2
                lcl_bary_x[2] = x2

                lcl_bary_y[0] = 1-y1
                lcl_bary_y[1] = y1-y2
                lcl_bary_y[2] = y2

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]
                bary_y[2, offset+i] = lcl_bary_y[2]

                weights[offset+i] = 2.0*qrId.weights[i]*(eta0*eta1*eta2)**(-singularity)

            # integral 1
            offset = qrId.num_nodes
            for i in range(qrId.num_nodes):
                eta0 = qrId.nodes[0, i]
                eta1 = qrId.nodes[1, i]
                eta2 = qrId.nodes[2, i]
                eta3 = qrId.nodes[3, i]

                x1 = eta0
                x2 = eta0*eta1
                y1 = eta0*(1-eta1*eta2*eta3)
                y2 = eta0*eta1*(1-eta2)

                lcl_bary_x[0] = 1-x1
                lcl_bary_x[1] = x1-x2
                lcl_bary_x[2] = x2

                lcl_bary_y[0] = 1-y1
                lcl_bary_y[1] = y1-y2
                lcl_bary_y[2] = y2

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]
                bary_y[2, offset+i] = lcl_bary_y[2]

                weights[offset+i] = 2.0*qrId.weights[i]*(eta0*eta1*eta2)**(-singularity)

            # integral 2
            offset = 2*qrId.num_nodes
            for i in range(qrId.num_nodes):
                eta0 = qrId.nodes[0, i]
                eta1 = qrId.nodes[1, i]
                eta2 = qrId.nodes[2, i]
                eta3 = qrId.nodes[3, i]

                x1 = eta0
                x2 = eta0*eta1*(1-eta2)
                y1 = eta0*(1-eta1*eta2*eta3)
                y2 = eta0*eta1*(1-eta2*eta3)

                lcl_bary_x[0] = 1-x1
                lcl_bary_x[1] = x1-x2
                lcl_bary_x[2] = x2

                lcl_bary_y[0] = 1-y1
                lcl_bary_y[1] = y1-y2
                lcl_bary_y[2] = y2

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]
                bary_y[2, offset+i] = lcl_bary_y[2]

                weights[offset+i] = 2.0*qrId.weights[i]*(eta0*eta1*eta2)**(-singularity)

            super(singularityCancelationQuadRule2D, self).__init__(bary, weights, dim+1)
        elif panel == COMMON_EDGE:
            # We obtain 4 subdomains from splitting the integral.

            #  Jacobian0,1 = eta0**3 * eta1**2
            #  Jacobian2,3 = eta0**3 * eta1**2 * eta2

            # We factor out (eta0 * eta1) from each PSI and
            # (eta0 * eta1)**singularity from the kernel.

            qrEdge0 = GaussJacobi(((1, 3+singularity, 0),
                                   (1, 2+singularity, 0),
                                   (quad_order_diagonal, 0, 0),
                                   (quad_order_diagonal, 0, 0)))
            qrEdge1 = GaussJacobi(((1, 3+singularity, 0),
                                   (1, 2+singularity, 0),
                                   (quad_order_diagonal, 1, 0),
                                   (quad_order_diagonal, 0, 0)))

            bary = uninitialized((2*dim+2,
                                  2*(qrEdge0.num_nodes+qrEdge1.num_nodes)), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            weights = uninitialized((2*(qrEdge0.num_nodes+qrEdge1.num_nodes)), dtype=REAL)

            # integral 0
            offset = 0
            for i in range(qrEdge0.num_nodes):
                eta0 = qrEdge0.nodes[0, i]
                eta1 = qrEdge0.nodes[1, i]
                eta2 = qrEdge0.nodes[2, i]
                eta3 = qrEdge0.nodes[3, i]

                x1 = eta0*(1-eta1*eta2)
                x2 = eta0*eta1*(1-eta2)
                y1 = eta0
                y2 = eta0*eta1*eta3

                lcl_bary_x[0] = 1-x1
                lcl_bary_x[1] = x1-x2
                lcl_bary_x[2] = x2

                lcl_bary_y[0] = 1-y1
                lcl_bary_y[1] = y1-y2
                lcl_bary_y[2] = y2

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]
                bary_y[2, offset+i] = lcl_bary_y[2]

                weights[offset+i] = qrEdge0.weights[i] * (eta0*eta1)**(-singularity)

            # integral 1
            offset = qrEdge0.num_nodes
            for i in range(qrEdge0.num_nodes):
                eta0 = qrEdge0.nodes[0, i]
                eta1 = qrEdge0.nodes[1, i]
                eta2 = qrEdge0.nodes[2, i]
                eta3 = qrEdge0.nodes[3, i]

                x1 = eta0
                x2 = eta0*eta1*eta3
                y1 = eta0*(1-eta1*eta2)
                y2 = eta0*eta1*(1-eta2)

                lcl_bary_x[0] = 1-x1
                lcl_bary_x[1] = x1-x2
                lcl_bary_x[2] = x2

                lcl_bary_y[0] = 1-y1
                lcl_bary_y[1] = y1-y2
                lcl_bary_y[2] = y2

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]
                bary_y[2, offset+i] = lcl_bary_y[2]

                weights[offset+i] = qrEdge0.weights[i] * (eta0*eta1)**(-singularity)

            # integral 2
            offset = 2*qrEdge0.num_nodes
            for i in range(qrEdge1.num_nodes):
                eta0 = qrEdge1.nodes[0, i]
                eta1 = qrEdge1.nodes[1, i]
                eta2 = qrEdge1.nodes[2, i]
                eta3 = qrEdge1.nodes[3, i]

                x1 = eta0*(1-eta1*eta2*eta3)
                x2 = eta0*eta1*eta2*(1-eta3)
                y1 = eta0
                y2 = eta0*eta1

                lcl_bary_x[0] = 1-x1
                lcl_bary_x[1] = x1-x2
                lcl_bary_x[2] = x2

                lcl_bary_y[0] = 1-y1
                lcl_bary_y[1] = y1-y2
                lcl_bary_y[2] = y2

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]
                bary_y[2, offset+i] = lcl_bary_y[2]

                weights[offset+i] = qrEdge1.weights[i] * (eta0*eta1)**(-singularity)

            # integral 3
            offset = 2*qrEdge0.num_nodes+qrEdge1.num_nodes
            for i in range(qrEdge1.num_nodes):
                eta0 = qrEdge1.nodes[0, i]
                eta1 = qrEdge1.nodes[1, i]
                eta2 = qrEdge1.nodes[2, i]
                eta3 = qrEdge1.nodes[3, i]

                x1 = eta0
                x2 = eta0*eta1
                y1 = eta0*(1-eta1*eta2*eta3)
                y2 = eta0*eta1*eta2*(1-eta3)

                lcl_bary_x[0] = 1-x1
                lcl_bary_x[1] = x1-x2
                lcl_bary_x[2] = x2

                lcl_bary_y[0] = 1-y1
                lcl_bary_y[1] = y1-y2
                lcl_bary_y[2] = y2

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]
                bary_y[2, offset+i] = lcl_bary_y[2]

                weights[offset+i] = qrEdge1.weights[i] * (eta0*eta1)**(-singularity)

            super(singularityCancelationQuadRule2D, self).__init__(bary, weights, 2*dim)
        elif panel == COMMON_VERTEX:
            # We obtain 2 subdomains from splitting the integral.

            # Jacobian = eta0**3

            # We factor out eta0 from each PSI and
            # eta0**singularity from the kernel.

            qrVertex = GaussJacobi(((1, 3+singularity, 0),
                                    (quad_order_diagonalV, 0, 0),
                                    (quad_order_diagonalV, 1, 0),
                                    (quad_order_diagonalV, 0, 0)))
            bary = uninitialized((2*dim+2,
                                  2*qrVertex.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            weights = uninitialized((2*qrVertex.num_nodes), dtype=REAL)

            offset = 0
            for i in range(qrVertex.num_nodes):
                eta0 = qrVertex.nodes[0, i]
                eta1 = qrVertex.nodes[1, i]
                eta2 = qrVertex.nodes[2, i]
                eta3 = qrVertex.nodes[3, i]

                x1 = eta0
                x2 = eta0*eta1
                y1 = eta0*eta2
                y2 = eta0*eta2*eta3

                lcl_bary_x[0] = 1-x1
                lcl_bary_x[1] = x1-x2
                lcl_bary_x[2] = x2

                lcl_bary_y[0] = 1-y1
                lcl_bary_y[1] = y1-y2
                lcl_bary_y[2] = y2

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]
                bary_y[2, offset+i] = lcl_bary_y[2]

                weights[offset+i] = qrVertex.weights[i] * eta0**(-singularity)

            offset = qrVertex.num_nodes
            for i in range(qrVertex.num_nodes):
                eta0 = qrVertex.nodes[0, i]
                eta1 = qrVertex.nodes[1, i]
                eta2 = qrVertex.nodes[2, i]
                eta3 = qrVertex.nodes[3, i]

                x1 = eta0*eta2
                x2 = eta0*eta2*eta3
                y1 = eta0
                y2 = eta0*eta1

                lcl_bary_x[0] = 1-x1
                lcl_bary_x[1] = x1-x2
                lcl_bary_x[2] = x2

                lcl_bary_y[0] = 1-y1
                lcl_bary_y[1] = y1-y2
                lcl_bary_y[2] = y2

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]
                bary_y[2, offset+i] = lcl_bary_y[2]

                weights[offset+i] = qrVertex.weights[i] * eta0**(-singularity)

            super(singularityCancelationQuadRule2D, self).__init__(bary, weights, 2*dim+1)


cdef class singularityCancelationQuadRule2D_boundary(quadratureRule):
    def __init__(self, panelType panel,
                 REAL_t singularity,
                 INDEX_t quad_order_diagonal,
                 INDEX_t quad_order_regular):
        cdef:
            INDEX_t i, offset
            REAL_t eta0, eta1, eta2
            quadratureRule qrEdge0, qrEdge1, qrEdge2, qrVertex0, qrVertex1
            INDEX_t dim = 2
            REAL_t lcl_bary_x[3]
            REAL_t lcl_bary_y[2]
            REAL_t[:, ::1] bary, bary_x, bary_y
            REAL_t[::1] weights

        if panel == COMMON_EDGE:
            qrEdge0 = qrEdge1 = qrEdge2 = GaussJacobi(((quad_order_regular, 1.+singularity, 1.),
                                                       (quad_order_diagonal, 0., 0.),
                                                       (quad_order_diagonal, 0., 0.)))

            bary = uninitialized((2*dim+1,
                                  qrEdge0.num_nodes+
                                  qrEdge1.num_nodes+
                                  qrEdge2.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            weights = uninitialized((qrEdge0.num_nodes+
                                     qrEdge1.num_nodes+
                                     qrEdge2.num_nodes), dtype=REAL)

            # int 0
            offset = 0
            for i in range(qrEdge0.num_nodes):
                eta0 = qrEdge0.nodes[0, i]
                eta1 = qrEdge0.nodes[1, i]
                eta2 = qrEdge0.nodes[2, i]

                lcl_bary_x[0] = 1-eta0-(1-eta0)*eta2
                lcl_bary_x[1] = eta0+(1-eta0)*eta2-eta0*eta1
                lcl_bary_x[2] = eta0*eta1

                lcl_bary_y[0] = 1-eta2*(1-eta0)
                lcl_bary_y[1] = eta2*(1-eta0)

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]

                weights[offset+i] = qrEdge0.weights[i] * eta0**(-singularity)

            # int 1
            offset = qrEdge0.num_nodes
            for i in range(qrEdge1.num_nodes):
                eta0 = qrEdge1.nodes[0, i]
                eta1 = qrEdge1.nodes[1, i]
                eta2 = qrEdge1.nodes[2, i]

                lcl_bary_x[0] = 1-eta0-eta2+eta0*eta2
                lcl_bary_x[1] = eta2-eta0*eta2
                lcl_bary_x[2] = eta0

                lcl_bary_y[0] = 1-eta2+eta0*eta2+eta0*eta1-eta0
                lcl_bary_y[1] = eta2-eta0*eta2-eta0*eta1+eta0

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]

                weights[offset+i] = qrEdge1.weights[i] * eta0**(-singularity)

            # int 2
            offset = qrEdge0.num_nodes+qrEdge1.num_nodes
            for i in range(qrEdge2.num_nodes):
                eta0 = qrEdge2.nodes[0, i]
                eta1 = qrEdge2.nodes[1, i]
                eta2 = qrEdge2.nodes[2, i]

                lcl_bary_x[0] = 1-eta2+eta0*eta2-eta0*eta1
                lcl_bary_x[1] = eta2-eta0*eta2
                lcl_bary_x[2] = eta0*eta1

                lcl_bary_y[0] = 1-eta2+eta0*eta2-eta0
                lcl_bary_y[1] = eta2-eta0*eta2+eta0

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]

                weights[offset+i] = qrEdge2.weights[i] * eta0**(-singularity)

            super(singularityCancelationQuadRule2D_boundary, self).__init__(bary, weights, 2*dim+1)
        elif panel == COMMON_VERTEX:
            qrVertex0 = GaussJacobi(((quad_order_regular, 2.0+singularity, 0),
                                     (quad_order_diagonal, 0, 0),
                                     (quad_order_diagonal, 0, 0)))
            qrVertex1 = GaussJacobi(((quad_order_regular, 2.0+singularity, 0),
                                     (quad_order_diagonal, 1, 0),
                                     (quad_order_diagonal, 0, 0)))
            bary = uninitialized((2*dim+1,
                                  qrVertex0.num_nodes+
                                  qrVertex1.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            weights = uninitialized((qrVertex0.num_nodes+
                                     qrVertex1.num_nodes), dtype=REAL)

            # int 0
            offset = 0
            for i in range(qrVertex0.num_nodes):
                eta0 = qrVertex0.nodes[0, i]
                eta1 = qrVertex0.nodes[1, i]
                eta2 = qrVertex0.nodes[2, i]

                lcl_bary_x[0] = 1-eta0
                lcl_bary_x[1] = eta0*(1-eta1)
                lcl_bary_x[2] = eta0*eta1

                lcl_bary_y[0] = 1-eta0*eta2
                lcl_bary_y[1] = eta0*eta2

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]

                weights[offset+i] = qrVertex0.weights[i] * eta0**(-singularity)

            # int 1
            offset = qrVertex0.num_nodes
            for i in range(qrVertex1.num_nodes):
                eta0 = qrVertex1.nodes[0, i]
                eta1 = qrVertex1.nodes[1, i]
                eta2 = qrVertex1.nodes[2, i]

                lcl_bary_x[0] = 1-eta0*eta1
                lcl_bary_x[1] = eta0*eta1*(1-eta2)
                lcl_bary_x[2] = eta0*eta1*eta2

                lcl_bary_y[0] = 1-eta0
                lcl_bary_y[1] = eta0

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]
                bary_x[2, offset+i] = lcl_bary_x[2]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]

                weights[offset+i] = qrVertex1.weights[i] * eta0**(-singularity)

            super(singularityCancelationQuadRule2D_boundary, self).__init__(bary, weights, 2*dim+1)


cdef class fractionalLaplacian2D(nonlocalLaplacian2D):
    """The local stiffness matrix

    .. math::

       \\int_{K_1}\\int_{K_2} (u(x)-u(y)) (v(x)-v(y)) \\gamma(x,y) dy dx

    for the symmetric 2D nonlocal Laplacian.
    """
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 target_order=None,
                 quad_order_diagonal=None,
                 num_dofs=None,
                 **kwargs):
        super(fractionalLaplacian2D, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        # The integrand (excluding the kernel) cancels 2 orders of the singularity within an element.
        self.singularityCancelationIntegrandWithinElement = 2.
        # The integrand (excluding the kernel) cancels 2 orders of the
        # singularity across elements for continuous finite elements.
        if isinstance(DoFMap, P0_DoFMap):
            assert self.kernel.max_singularity > -3., "Discontinuous finite elements are not conforming for singularity order {} <= -3.".format(self.kernel.max_singularity)
            self.singularityCancelationIntegrandAcrossElements = 0.
        else:
            self.singularityCancelationIntegrandAcrossElements = 2.

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
            self.addQuadRule(panel)
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
            INDEX_t dim = 2
            REAL_t lcl_bary_x[3]
            REAL_t lcl_bary_y[3]
            REAL_t[:, ::1] PSI
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
                PSI = uninitialized((dofs_per_element, qr.num_nodes), dtype=REAL)

                for dof in range(dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PSI[dof, i] = (sf.eval(lcl_bary_x)-sf.eval(lcl_bary_y))

                sQR = specialQuadRule(qr, PSI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes), dtype=REAL)
            self.qrId = sQR.qr
            self.PSI_id = sQR.PSI
        elif panel == COMMON_EDGE:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:

                qr = singularityCancelationQuadRule2D(panel,
                                                      self.singularityCancelationIntegrandAcrossElements+singularityValue,
                                                      self.quad_order_diagonal,
                                                      self.quad_order_diagonalV,
                                                      1)
                PSI = uninitialized((2*dofs_per_element - 2*dofs_per_vertex - dofs_per_edge,
                                     qr.num_nodes), dtype=REAL)

                for dof in range(2*dofs_per_vertex):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PSI[dof, i] = (sf.eval(lcl_bary_x)-sf.eval(lcl_bary_y))

                for dof in range((dim+1)*dofs_per_vertex, (dim+1)*dofs_per_vertex+dofs_per_edge):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PSI[dof, i] = (sf.eval(lcl_bary_x)-sf.eval(lcl_bary_y))

                for dof in range(2*dofs_per_vertex, (dim+1)*dofs_per_vertex):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PSI[dof, i] = sf.eval(lcl_bary_x)
                        PSI[dofs_per_element+dof-2*dofs_per_vertex, i] = -sf.eval(lcl_bary_y)

                for dof in range((dim+1)*dofs_per_vertex+dofs_per_edge, dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PSI[dof, i] = sf.eval(lcl_bary_x)
                        PSI[dofs_per_element+dof-2*dofs_per_vertex-dofs_per_edge, i] = -sf.eval(lcl_bary_y)

                sQR = specialQuadRule(qr, PSI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes), dtype=REAL)
            self.qrEdge = sQR.qr
            self.PSI_edge = sQR.PSI
        elif panel == COMMON_VERTEX:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:

                qr = singularityCancelationQuadRule2D(panel,
                                                      self.singularityCancelationIntegrandAcrossElements+singularityValue,
                                                      self.quad_order_diagonal,
                                                      self.quad_order_diagonalV,
                                                      1)
                PSI = uninitialized((2*dofs_per_element - dofs_per_vertex,
                                     qr.num_nodes), dtype=REAL)

                for dof in range(dofs_per_vertex):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PSI[dof, i] = (sf.eval(lcl_bary_x)-sf.eval(lcl_bary_y))

                for dof in range(dofs_per_vertex, dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        PSI[dof, i] = sf.eval(lcl_bary_x)
                        PSI[dofs_per_element+dof-dofs_per_vertex, i] = -sf.eval(lcl_bary_y)

                sQR = specialQuadRule(qr, PSI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes), dtype=REAL)
            self.qrVertex = sQR.qr
            self.PSI_vertex = sQR.PSI
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

    def __repr__(self):
        return (super(fractionalLaplacian2D, self).__repr__() +
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
            INDEX_t k, m, i, j, I, J, dofs_per_element, dim = 2
            REAL_t vol, val
            quadratureRule qr
            REAL_t[:, ::1] PSI
            REAL_t x[2]
            REAL_t y[2]

        if panel >= 1:
            self.eval_distant(contrib, panel, mask)
            return
        elif panel == COMMON_FACE:
            qr = self.qrId
            PSI = self.PSI_id
        elif panel == COMMON_EDGE:
            qr = self.qrEdge
            PSI = self.PSI_edge
        elif panel == COMMON_VERTEX:
            qr = self.qrVertex
            PSI = self.PSI_vertex
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

        vol = 4.0*self.vol1*self.vol2

        dofs_per_element = self.DoFMap.dofs_per_element

        # Evaluate the kernel on the quadrature nodes
        for m in range(qr.num_nodes):
            for j in range(dim):
                x[j] = (self.simplex1[self.perm1[0], j]*qr.nodes[0, m] +
                        self.simplex1[self.perm1[1], j]*qr.nodes[1, m] +
                        self.simplex1[self.perm1[2], j]*qr.nodes[2, m])
                y[j] = (self.simplex2[self.perm2[0], j]*qr.nodes[3, m] +
                        self.simplex2[self.perm2[1], j]*qr.nodes[4, m] +
                        self.simplex2[self.perm2[2], j]*qr.nodes[5, m])
            self.temp[m] = qr.weights[m] * self.kernel.evalPtr(dim, &x[0], &y[0])

        # "perm" maps from dofs on the reordered simplices (matching
        # vertices first) to the dofs in the usual ordering.
        contrib[:] = 0.
        for I in range(PSI.shape[0]):
            i = self.perm[I]
            for J in range(I, PSI.shape[0]):
                j = self.perm[J]
                # We are assembling the upper trinagular part of the
                # symmetric (2*dofs_per_element)**2 local stiffness
                # matrix. This computes the flattened index.
                if j < i:
                    k = 2*dofs_per_element*j-(j*(j+1) >> 1) + i
                else:
                    k = 2*dofs_per_element*i-(i*(i+1) >> 1) + j
                # Check if that entry has been requested.
                if mask[k]:
                    val = 0.
                    for m in range(qr.num_nodes):
                        val += self.temp[m] * PSI[I, m] * PSI[J, m]
                    contrib[k] = val*vol


cdef class fractionalLaplacian2D_nonsym(fractionalLaplacian2D):
    """The local stiffness matrix

    .. math::

       \\int_{K_1}\\int_{K_2} [ u(x) \\gamma(x,y) - u(y) \\gamma(y,x) ] [ v(x)-v(y) ] dy dx

    for the 2D non-symmetric nonlocal Laplacian.
    """
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
            INDEX_t dim = 2
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
                    self.temp = uninitialized((qr.num_nodes), dtype=REAL)
                    self.temp2 = uninitialized((qr.num_nodes), dtype=REAL)
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

                for dof in range((dim+1)*dofs_per_vertex, (dim+1)*dofs_per_vertex+dofs_per_edge):
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

                for dof in range(2*dofs_per_vertex, (dim+1)*dofs_per_vertex):
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

                for dof in range((dim+1)*dofs_per_vertex+dofs_per_edge, dofs_per_element):
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
                    self.temp = uninitialized((qr.num_nodes), dtype=REAL)
                    self.temp2 = uninitialized((qr.num_nodes), dtype=REAL)
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
                    self.temp = uninitialized((qr.num_nodes), dtype=REAL)
                    self.temp2 = uninitialized((qr.num_nodes), dtype=REAL)
            self.qrVertex = sQR.qr
            self.PHI_vertex = sQR.PHI3
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, m, i, j, I, J, dofs_per_element, dim = 2
            REAL_t vol, val
            quadratureRule qr
            REAL_t[:, :, ::1] PHI
            REAL_t x[2]
            REAL_t y[2]

        if panel >= 1:
            self.eval_distant_nonsym(contrib, panel, mask)
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

        vol = 4.0*self.vol1*self.vol2

        dofs_per_element = self.DoFMap.dofs_per_element

        # Evaluate the kernel on the quadrature nodes
        for m in range(qr.num_nodes):
            for j in range(dim):
                x[j] = (self.simplex1[self.perm1[0], j]*qr.nodes[0, m] +
                        self.simplex1[self.perm1[1], j]*qr.nodes[1, m] +
                        self.simplex1[self.perm1[2], j]*qr.nodes[2, m])
                y[j] = (self.simplex2[self.perm2[0], j]*qr.nodes[3, m] +
                        self.simplex2[self.perm2[1], j]*qr.nodes[4, m] +
                        self.simplex2[self.perm2[2], j]*qr.nodes[5, m])
            self.temp[m] = qr.weights[m] * self.kernel.evalPtr(dim, &x[0], &y[0])
            self.temp2[m] = qr.weights[m] * self.kernel.evalPtr(dim, &y[0], &x[0])

        # "perm" maps from dofs on the reordered simplices (matching
        # vertices first) to the dofs in the usual ordering.
        contrib[:] = 0.
        for I in range(PHI.shape[0]):
            i = self.perm[I]
            for J in range(PHI.shape[0]):
                j = self.perm[J]
                k = i*(2*dofs_per_element)+j
                # Check if that entry has been requested.
                if mask[k]:
                    val = 0.
                    for m in range(qr.num_nodes):
                        val += (self.temp[m] * PHI[I, m, 0] - self.temp2[m] * PHI[I, m, 1]) * (PHI[J, m, 0] - PHI[J, m, 1])
                    contrib[k] = val*vol


cdef class fractionalLaplacian2D_boundary(fractionalLaplacian2DZeroExterior):
    """The local stiffness matrix

    .. math::

       \\int_{K} u(x) v(x) \\int_{e} \\Gamma(x,y) dy dx

    """
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 target_order=None,
                 quad_order_diagonal=None,
                 num_dofs=None,
                 **kwargs):
        super(fractionalLaplacian2D_boundary, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        smax = max(0.5*(-self.kernel.max_singularity-1.), 0.)
        if target_order is None:
            # this is the desired global order wrt to the number of DoFs
            # target_order = (2.-s)/self.dim
            target_order = 0.5
        self.target_order = target_order

        if quad_order_diagonal is None:
            # measured log(2 rho_2) = 0.4
            quad_order_diagonal = max(np.ceil((target_order+0.5+smax)/(0.35)*abs(np.log(self.hmin/self.H0))), 2)
        self.quad_order_diagonal = quad_order_diagonal

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableOrder):
            self.getNearQuadRule(COMMON_EDGE)
            self.getNearQuadRule(COMMON_VERTEX)

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        cdef:
            panelType panel, panel2
            REAL_t logdh1 = max(log(d/h1), 0.), logdh2 = max(log(d/h2), 0.)
            REAL_t logh1H0 = abs(log(h1/self.H0)), logh2H0 = abs(log(h2/self.H0))
            REAL_t loghminH0 = max(logh1H0, logh2H0)
            REAL_t s = max(0.5*(-self.kernel.getSingularityValue()-1.), 0.)
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
            self.addQuadRule_boundary(panel)
        return panel

    cdef void getNearQuadRule(self, panelType panel):
        cdef:
            INDEX_t i
            REAL_t singularityValue = self.kernel.getSingularityValue()
            INDEX_t dof
            quadratureRule qr
            specialQuadRule sQR
            REAL_t[:, ::1] PHI
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            shapeFunction sf
            REAL_t lcl_bary_x[3]
        if panel == COMMON_EDGE:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:
                if singularityValue > -2.+1e-3:
                    qr = singularityCancelationQuadRule2D_boundary(panel, singularityValue, self.quad_order_diagonal, self.quad_order_diagonal)
                else:
                    qr = singularityCancelationQuadRule2D_boundary(panel, 2.+singularityValue, self.quad_order_diagonal, self.quad_order_diagonal)
                PHI = uninitialized((dofs_per_element, qr.num_nodes), dtype=REAL)

                for dof in range(dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        PHI[dof, i] = sf.eval(lcl_bary_x)

                sQR = specialQuadRule(qr, PHI=PHI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes), dtype=REAL)
            self.qrEdge = sQR.qr
            self.PHI_edge2 = sQR.PHI

        elif panel == COMMON_VERTEX:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:
                qr = singularityCancelationQuadRule2D_boundary(panel, singularityValue, self.quad_order_diagonal, self.quad_order_diagonal)
                PHI = uninitialized((dofs_per_element, qr.num_nodes), dtype=REAL)

                for dof in range(dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        PHI[dof, i] = sf.eval(lcl_bary_x)

                sQR = specialQuadRule(qr, PHI=PHI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes), dtype=REAL)
            self.qrVertex = sQR.qr
            self.PHI_vertex2 = sQR.PHI

    def __repr__(self):
        return (super(fractionalLaplacian2D_boundary, self).__repr__() +
                'hmin:                          {:.3}\n'.format(self.hmin) +
                'H0:                            {:.3}\n'.format(self.H0) +
                'target order:                  {}\n'.format(self.target_order) +
                'quad_order_diagonal:           {}\n'.format(self.quad_order_diagonal) +
                'quad_order_off_diagonal        {}\n'.format(list(self.distantQuadRules.keys())))

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            REAL_t vol1 = self.vol1, vol2 = self.vol2, vol, val
            INDEX_t i, j, k, I, J, m
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            REAL_t normW
            quadratureRule qr
            REAL_t[:, ::1] PHI
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t dim = 2
            REAL_t x[2]
            REAL_t y[2]

        if panel >= 1:
            self.eval_distant_boundary(contrib, panel, mask)
            return

        # Kernel:
        #  \Gamma(x,y) = n \dot (x-y) * C(d,s) / (2s) / |x-y|^{d+2s}
        # with inward normal n.
        #
        # Rewrite as
        #  \Gamma(x,y) = [ n \dot (x-y)/|x-y| ] * [ C(d,s) / (2s) / |x-y|^{d-1+2s} ]
        #                                         \--------------------------------/
        #                                                 |
        #                                           boundaryKernel

        # n is independent of x and y
        self.n[0] = simplex2[1, 1] - simplex2[0, 1]
        self.n[1] = simplex2[0, 0] - simplex2[1, 0]
        # F is same as vol2
        val = 1./sqrt(mydot(self.n, self.n))
        self.n[0] *= val
        self.n[1] *= val

        contrib[:] = 0.

        if panel == COMMON_EDGE:
            qr = self.qrEdge
            PHI = self.PHI_edge2
        elif panel == COMMON_VERTEX:
            qr = self.qrVertex
            PHI = self.PHI_vertex2
        else:
            raise NotImplementedError('Panel type unknown: {}.'.format(panel))

        vol = -2.0*vol1*vol2

        for m in range(qr.num_nodes):
            normW = 0.
            for j in range(dim):
                x[j] = (simplex1[self.perm1[0], j]*qr.nodes[0, m] +
                        simplex1[self.perm1[1], j]*qr.nodes[1, m] +
                        simplex1[self.perm1[2], j]*qr.nodes[2, m])
                y[j] = (simplex2[self.perm2[0], j]*qr.nodes[3, m] +
                        simplex2[self.perm2[1], j]*qr.nodes[4, m])
                self.w[j] = x[j]-y[j]
                normW += self.w[j]**2
            normW = 1./sqrt(normW)
            for j in range(dim):
                self.w[j] *= normW
            self.temp[m] = qr.weights[m] * mydot(self.n, self.w) * self.kernel.evalPtr(dim, &x[0], &y[0])
        for I in range(dofs_per_element):
            i = self.perm[I]
            for J in range(I, dofs_per_element):
                j = self.perm[J]
                if j < i:
                    k = dofs_per_element*j-(j*(j+1) >> 1) + i
                else:
                    k = dofs_per_element*i-(i*(i+1) >> 1) + j
                if mask[k]:
                    val = 0.
                    for m in range(qr.num_nodes):
                        val += self.temp[m] * PHI[I, m] * PHI[J, m]
                    contrib[k] = val*vol


