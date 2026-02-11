###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from libc.math cimport sqrt, log, ceil, fabs as abs
import numpy as np
cimport numpy as np

from PyNucleus_base.myTypes import INDEX, REAL
from PyNucleus_base import uninitialized
from PyNucleus_base.blas cimport mydot
from PyNucleus_fem.quadrature cimport (transformQuadratureRule,
                                       doubleSimplexQuadratureRule,
                                       GaussJacobi, LogGaussJacobi)
from PyNucleus_fem.DoFMaps cimport DoFMap, P0_DoFMap, P1_DoFMap, shapeFunction
from PyNucleus_nl.fractionalOrders cimport constFractionalOrder

include "kernel_params.pxi"
include "panelTypes.pxi"

cdef:
    MASK_t ALL
ALL.set()


cdef class fractionalLaplacian2DZeroExterior(nonlocalLaplacian2D):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap dm, num_dofs=None, **kwargs):
        manifold_dim2 = mesh.manifold_dim-1
        super(fractionalLaplacian2DZeroExterior, self).__init__(kernel, mesh, dm, num_dofs, manifold_dim2=manifold_dim2, **kwargs)
        self.symmetricCells = False
        self.symmetricLocalMatrix = True


cdef class singularityCancelationQuadRule2D(singularityCancelationQuadRule):
    def __init__(self, panelType panel,
                 REAL_t singularity,
                 REAL_t log_singularity,
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
            REAL_t[::1] singularPart
            INDEX_t offset

        if panel == COMMON_FACE:
            # We obtain 6 subdomains from splitting the integral,
            # but pairs of 2 are symmetric wrt to exchange of K_1
            # and K_2. So we get 3 integrals, each with a weight
            # 2.

            #  Jacobian = eta0**3 * eta1**2 * eta2

            # We factor out (eta0 * eta1 * eta2) from each PSI and
            # (eta0 * eta1 * eta2)**singularity from the kernel.

            if log_singularity == 0:
                qrId = GaussJacobi(((1, 3+singularity, 0),
                                    (1, 2+singularity, 0),
                                    (1, 1+singularity, 0),
                                    (quad_order_diagonal, 0, 0)))
            elif log_singularity == 1:
                qr1000 = LogGaussJacobi(((1, 3+singularity, 0, 1),
                                         (1, 2+singularity, 0, 0),
                                         (1, 1+singularity, 0, 0),
                                         (quad_order_diagonal, 0, 0, 0)))
                qr0100 = LogGaussJacobi(((1, 3+singularity, 0, 0),
                                         (1, 2+singularity, 0, 1),
                                         (1, 1+singularity, 0, 0),
                                         (quad_order_diagonal, 0, 0, 0)))
                qr0010 = LogGaussJacobi(((1, 3+singularity, 0, 0),
                                         (1, 2+singularity, 0, 0),
                                         (1, 1+singularity, 0, 1),
                                         (quad_order_diagonal, 0, 0, 0)))
                qrId = qr1000+qr0100+qr0010
            elif log_singularity == 2:
                qr2000 = LogGaussJacobi(((1, 3+singularity, 0, 2),
                                         (1, 2+singularity, 0, 0),
                                         (1, 1+singularity, 0, 0),
                                         (quad_order_diagonal, 0, 0, 0)))
                qr1100 = LogGaussJacobi(((1, 3+singularity, 0, 1),
                                         (1, 2+singularity, 0, 1),
                                         (1, 1+singularity, 0, 0),
                                         (quad_order_diagonal, 0, 0, 0)))
                qr0200 = LogGaussJacobi(((1, 3+singularity, 0, 0),
                                         (1, 2+singularity, 0, 2),
                                         (1, 1+singularity, 0, 0),
                                         (quad_order_diagonal, 0, 0, 0)))
                qr0110 = LogGaussJacobi(((1, 3+singularity, 0, 0),
                                         (1, 2+singularity, 0, 1),
                                         (1, 1+singularity, 0, 1),
                                         (quad_order_diagonal, 0, 0, 0)))
                qr0020 = LogGaussJacobi(((1, 3+singularity, 0, 0),
                                         (1, 2+singularity, 0, 0),
                                         (1, 1+singularity, 0, 2),
                                         (quad_order_diagonal, 0, 0, 0)))
                qr1010 = LogGaussJacobi(((1, 3+singularity, 0, 1),
                                         (1, 2+singularity, 0, 0),
                                         (1, 1+singularity, 0, 1),
                                         (quad_order_diagonal, 0, 0, 0)))
                for i in range(qr1100.num_nodes):
                    qr1100.weights[i] *= 2.
                for i in range(qr0110.num_nodes):
                    qr0110.weights[i] *= 2.
                for i in range(qr1010.num_nodes):
                    qr1010.weights[i] *= 2.
                qrId = qr2000+qr0200+qr0020+qr1100+qr0110+qr1010
            else:
                raise NotImplementedError(log_singularity)

            bary = uninitialized((2*dim+2,
                                  3*qrId.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            weights = uninitialized((3*qrId.num_nodes), dtype=REAL)
            singularPart = uninitialized((3*qrId.num_nodes), dtype=REAL)

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

                singularPart[offset+i] = eta0*eta1*eta2
                weights[offset+i] = 2.0*qrId.weights[i]*singularPart[offset+i]**(-singularity)

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

                singularPart[offset+i] = eta0*eta1*eta2
                weights[offset+i] = 2.0*qrId.weights[i]*singularPart[offset+i]**(-singularity)

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

                singularPart[offset+i] = eta0*eta1*eta2
                weights[offset+i] = 2.0*qrId.weights[i]*singularPart[offset+i]**(-singularity)

            super(singularityCancelationQuadRule2D, self).__init__(bary, weights, dim+1)
        elif panel == COMMON_EDGE:
            # We obtain 4 subdomains from splitting the integral.

            #  Jacobian0,1 = eta0**3 * eta1**2
            #  Jacobian2,3 = eta0**3 * eta1**2 * eta2

            # We factor out (eta0 * eta1) from each PSI and
            # (eta0 * eta1)**singularity from the kernel.

            if log_singularity == 0:
                qrEdge0 = GaussJacobi(((1, 3+singularity, 0),
                                       (1, 2+singularity, 0),
                                       (quad_order_diagonal, 0, 0),
                                       (quad_order_diagonal, 0, 0)))
                qrEdge1 = GaussJacobi(((1, 3+singularity, 0),
                                       (1, 2+singularity, 0),
                                       (quad_order_diagonal, 1, 0),
                                       (quad_order_diagonal, 0, 0)))
            elif log_singularity == 1:
                qrEdge0_1000 = LogGaussJacobi(((1, 3+singularity, 0, 1),
                                               (1, 2+singularity, 0, 0),
                                               (quad_order_diagonal, 0, 0, 0),
                                               (quad_order_diagonal, 0, 0, 0)))
                qrEdge0_0100 = LogGaussJacobi(((1, 3+singularity, 0, 0),
                                               (1, 2+singularity, 0, 1),
                                               (quad_order_diagonal, 0, 0, 0),
                                               (quad_order_diagonal, 0, 0, 0)))
                qrEdge0 = qrEdge0_1000+qrEdge0_0100
                qrEdge1_1000 = LogGaussJacobi(((1, 3+singularity, 0, 1),
                                               (1, 2+singularity, 0, 0),
                                               (quad_order_diagonal, 1, 0, 0),
                                               (quad_order_diagonal, 0, 0, 0)))
                qrEdge1_0100 = LogGaussJacobi(((1, 3+singularity, 0, 0),
                                               (1, 2+singularity, 0, 1),
                                               (quad_order_diagonal, 1, 0, 0),
                                               (quad_order_diagonal, 0, 0, 0)))
                qrEdge1 = qrEdge1_1000+qrEdge1_0100
            elif log_singularity == 2:
                qrEdge0_2000 = LogGaussJacobi(((1, 3+singularity, 0, 1),
                                               (1, 2+singularity, 0, 0),
                                               (quad_order_diagonal, 0, 0, 0),
                                               (quad_order_diagonal, 0, 0, 0)))
                qrEdge0_1100 = LogGaussJacobi(((1, 3+singularity, 0, 1),
                                               (1, 2+singularity, 0, 1),
                                               (quad_order_diagonal, 0, 0, 0),
                                               (quad_order_diagonal, 0, 0, 0)))
                qrEdge0_0200 = LogGaussJacobi(((1, 3+singularity, 0, 0),
                                               (1, 2+singularity, 0, 2),
                                               (quad_order_diagonal, 0, 0, 0),
                                               (quad_order_diagonal, 0, 0, 0)))
                for i in range(qrEdge0_1100.num_nodes):
                    qrEdge0_1100.weights[i] *= 2.
                qrEdge0 = qrEdge0_2000+qrEdge0_1100+qrEdge0_0200
                qrEdge1_2000 = LogGaussJacobi(((1, 3+singularity, 0, 2),
                                               (1, 2+singularity, 0, 0),
                                               (quad_order_diagonal, 1, 0, 0),
                                               (quad_order_diagonal, 0, 0, 0)))
                qrEdge1_1100 = LogGaussJacobi(((1, 3+singularity, 0, 1),
                                               (1, 2+singularity, 0, 1),
                                               (quad_order_diagonal, 1, 0, 0),
                                               (quad_order_diagonal, 0, 0, 0)))
                qrEdge1_0200 = LogGaussJacobi(((1, 3+singularity, 0, 0),
                                               (1, 2+singularity, 0, 2),
                                               (quad_order_diagonal, 1, 0, 0),
                                               (quad_order_diagonal, 0, 0, 0)))
                for i in range(qrEdge1_1100.num_nodes):
                    qrEdge1_1100.weights[i] *= 2.
                qrEdge1 = qrEdge1_2000+qrEdge1_1100+qrEdge1_0200
            else:
                raise NotImplementedError(log_singularity)

            bary = uninitialized((2*dim+2,
                                  2*(qrEdge0.num_nodes+qrEdge1.num_nodes)), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            weights = uninitialized((2*(qrEdge0.num_nodes+qrEdge1.num_nodes)), dtype=REAL)
            singularPart = uninitialized((2*(qrEdge0.num_nodes+qrEdge1.num_nodes)), dtype=REAL)

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

                singularPart[offset+i] = eta0*eta1
                weights[offset+i] = qrEdge0.weights[i] * singularPart[offset+i]**(-singularity)

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

                singularPart[offset+i] = eta0*eta1
                weights[offset+i] = qrEdge0.weights[i] * singularPart[offset+i]**(-singularity)

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

                singularPart[offset+i] = eta0*eta1
                weights[offset+i] = qrEdge1.weights[i] * singularPart[offset+i]**(-singularity)

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

                singularPart[offset+i] = eta0*eta1
                weights[offset+i] = qrEdge1.weights[i] * singularPart[offset+i]**(-singularity)

            super(singularityCancelationQuadRule2D, self).__init__(bary, weights, 2*dim)
        elif panel == COMMON_VERTEX:
            # We obtain 2 subdomains from splitting the integral.

            # Jacobian = eta0**3

            # We factor out eta0 from each PSI and
            # eta0**singularity from the kernel.

            if log_singularity == 0:
                qrVertex = GaussJacobi(((1, 3+singularity, 0),
                                        (quad_order_diagonalV, 0, 0),
                                        (quad_order_diagonalV, 1, 0),
                                        (quad_order_diagonalV, 0, 0)))
            else:
                qrVertex = LogGaussJacobi(((1, 3+singularity, 0, log_singularity),
                                           (quad_order_diagonalV, 0, 0, 0),
                                           (quad_order_diagonalV, 1, 0, 0),
                                           (quad_order_diagonalV, 0, 0, 0)))
            bary = uninitialized((2*dim+2,
                                  2*qrVertex.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            weights = uninitialized((2*qrVertex.num_nodes), dtype=REAL)
            singularPart = uninitialized((2*qrVertex.num_nodes), dtype=REAL)

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

                singularPart[offset+i] = eta0
                weights[offset+i] = qrVertex.weights[i] * singularPart[offset+i]**(-singularity)

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

                singularPart[offset+i] = eta0
                weights[offset+i] = qrVertex.weights[i] * singularPart[offset+i]**(-singularity)

            super(singularityCancelationQuadRule2D, self).__init__(bary, weights, 2*dim+1)
        self.singularPart = singularPart


cdef class singularityCancelationQuadRule2D_boundary(singularityCancelationQuadRule):
    def __init__(self, panelType panel,
                 REAL_t singularity,
                 REAL_t log_singularity,
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
            REAL_t[::1] singularPart

        if panel == COMMON_EDGE:
            if log_singularity == 0:
                qrEdge0 = qrEdge1 = qrEdge2 = GaussJacobi(((quad_order_regular, 1.+singularity, 1.),
                                                           (quad_order_diagonal, 0., 0.),
                                                           (quad_order_diagonal, 0., 0.)))
            else:
                qrEdge0 = qrEdge1 = qrEdge2 = LogGaussJacobi(((quad_order_regular, 1.+singularity, 1., log_singularity),
                                                              (quad_order_diagonal, 0., 0., 0.),
                                                              (quad_order_diagonal, 0., 0., 0.)))

            bary = uninitialized((2*dim+1,
                                  qrEdge0.num_nodes+
                                  qrEdge1.num_nodes+
                                  qrEdge2.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            weights = uninitialized((qrEdge0.num_nodes+
                                     qrEdge1.num_nodes+
                                     qrEdge2.num_nodes), dtype=REAL)
            singularPart = uninitialized((qrEdge0.num_nodes+
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

                singularPart[offset+i] = eta0
                weights[offset+i] = qrEdge0.weights[i] * singularPart[offset+i]**(-singularity)

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

                singularPart[offset+i] = eta0
                weights[offset+i] = qrEdge1.weights[i] * singularPart[offset+i]**(-singularity)

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

                singularPart[offset+i] = eta0
                weights[offset+i] = qrEdge2.weights[i] * singularPart[offset+i]**(-singularity)

            super(singularityCancelationQuadRule2D_boundary, self).__init__(bary, weights, 2*dim+1)
        elif panel == COMMON_VERTEX:
            if log_singularity == 0:
                qrVertex0 = GaussJacobi(((quad_order_regular, 2.0+singularity, 0),
                                         (quad_order_diagonal, 0, 0),
                                         (quad_order_diagonal, 0, 0)))
                qrVertex1 = GaussJacobi(((quad_order_regular, 2.0+singularity, 0),
                                         (quad_order_diagonal, 1, 0),
                                         (quad_order_diagonal, 0, 0)))
            else:
                qrVertex0 = LogGaussJacobi(((quad_order_regular, 2.0+singularity, 0, 1),
                                            (quad_order_diagonal, 0, 0, 0),
                                            (quad_order_diagonal, 0, 0, 0)))
                qrVertex1 = LogGaussJacobi(((quad_order_regular, 2.0+singularity, 0, 1),
                                            (quad_order_diagonal, 1, 0, 0),
                                            (quad_order_diagonal, 0, 0, 0)))
            bary = uninitialized((2*dim+1,
                                  qrVertex0.num_nodes+
                                  qrVertex1.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            weights = uninitialized((qrVertex0.num_nodes+
                                     qrVertex1.num_nodes), dtype=REAL)
            singularPart = uninitialized((qrVertex0.num_nodes+
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

                singularPart[offset+i] = eta0
                weights[offset+i] = qrVertex0.weights[i] * singularPart[offset+i]**(-singularity)

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

                singularPart[offset+i] = eta0
                weights[offset+i] = qrVertex1.weights[i] * singularPart[offset+i]**(-singularity)

            super(singularityCancelationQuadRule2D_boundary, self).__init__(bary, weights, 2*dim+1)
        self.singularPart = singularPart


cdef class fractionalLaplacian2D(nonlocalLaplacian2D):
    """The local stiffness matrix

    .. math::

       0.5 \\int_{K_1}\\int_{K_2} (u(x)-u(y)) (v(x)-v(y)) \\gamma(x,y) dy dx

    for the symmetric 2D nonlocal Laplacian.
    """
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap dm,
                 target_order=None,
                 quad_order_diagonal=None,
                 num_dofs=None,
                 **kwargs):
        super(fractionalLaplacian2D, self).__init__(kernel, mesh, dm, num_dofs, **kwargs)
        self.setKernel(kernel, quad_order_diagonal, target_order)
        self.symmetricCells = True

    cpdef void setKernel(self, Kernel kernel, quad_order_diagonal=None, target_order=None):
        self.kernel = kernel

        # The integrand (excluding the kernel) cancels 2 orders of the singularity within an element.
        self.singularityCancelationIntegrandWithinElement = 2.
        # The integrand (excluding the kernel) cancels 2 orders of the
        # singularity across elements for continuous finite elements.
        if isinstance(self.DoFMap, P0_DoFMap):
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

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableSingularity):
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
            REAL_t c = (0.5*self.target_order+0.5)*log(self.num_dofs*self.H0**2)  # -4.
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
            REAL_t log_singularityValue = self.kernel.getLogSingularityValue()
            specialQuadRule sQR
            singularityCancelationQuadRule qr
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
            REAL_t phi_x = 0., phi_y = 0.

        if panel == COMMON_FACE:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:

                qr = singularityCancelationQuadRule2D(panel,
                                                      self.singularityCancelationIntegrandWithinElement+singularityValue,
                                                      log_singularityValue,
                                                      self.quad_order_diagonal,
                                                      self.quad_order_diagonalV,
                                                      1)
                qr.scaleWeights(4.0)
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
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PSI[dof, i] = phi_x-phi_y

                sQR = specialQuadRule(qr, PSI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes, self.kernel.valueSize), dtype=REAL)
            self.qrFace = sQR.qr
            self.PSI_face = sQR.PSI
        elif panel == COMMON_EDGE:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:

                qr = singularityCancelationQuadRule2D(panel,
                                                      self.singularityCancelationIntegrandAcrossElements+singularityValue,
                                                      log_singularityValue,
                                                      self.quad_order_diagonal,
                                                      self.quad_order_diagonalV,
                                                      1)
                qr.scaleWeights(4.0)
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
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PSI[dof, i] = phi_x-phi_y

                for dof in range((dim+1)*dofs_per_vertex, (dim+1)*dofs_per_vertex+dofs_per_edge):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PSI[dof, i] = phi_x-phi_y

                for dof in range(2*dofs_per_vertex, (dim+1)*dofs_per_vertex):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PSI[dof, i] = phi_x
                        PSI[dofs_per_element+dof-2*dofs_per_vertex, i] = -phi_y

                for dof in range((dim+1)*dofs_per_vertex+dofs_per_edge, dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PSI[dof, i] = phi_x
                        PSI[dofs_per_element+dof-2*dofs_per_vertex-dofs_per_edge, i] = -phi_y

                sQR = specialQuadRule(qr, PSI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes, self.kernel.valueSize), dtype=REAL)
            self.qrEdge = sQR.qr
            self.PSI_edge = sQR.PSI
        elif panel == COMMON_VERTEX:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:

                qr = singularityCancelationQuadRule2D(panel,
                                                      self.singularityCancelationIntegrandAcrossElements+singularityValue,
                                                      log_singularityValue,
                                                      self.quad_order_diagonal,
                                                      self.quad_order_diagonalV,
                                                      1)
                qr.scaleWeights(4.0)
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
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PSI[dof, i] = phi_x-phi_y

                for dof in range(dofs_per_vertex, dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PSI[dof, i] = phi_x
                        PSI[dofs_per_element+dof-dofs_per_vertex, i] = -phi_y

                sQR = specialQuadRule(qr, PSI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes, self.kernel.valueSize), dtype=REAL)
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
                   REAL_t[:, ::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        if panel >= 1:
            self.eval_distant_sym(contrib, panel, mask)
        else:
            self.eval_near_sym(contrib, panel, mask)


cdef class fractionalLaplacian2D_nonsym(fractionalLaplacian2D):
    """The local stiffness matrix

    .. math::

       0.5 \\int_{K_1}\\int_{K_2} [ u(x) \\gamma(x,y) - u(y) \\gamma(y,x) ] [ v(x)-v(y) ] dy dx

    for the 2D non-symmetric nonlocal Laplacian.
    """
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap dm,
                 target_order=None,
                 quad_order_diagonal=None,
                 num_dofs=None,
                 **kwargs):
        super(fractionalLaplacian2D_nonsym, self).__init__(kernel, mesh, dm, num_dofs, **kwargs)
        self.symmetricLocalMatrix = False
        self.symmetricCells = False

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        cdef:
            panelType panel, panel2
            REAL_t logdh1 = log(d/h1), logdh2 = log(d/h2)
            REAL_t c = (0.5*self.target_order+0.5)*log(self.num_dofs*self.H0**2)  # -4.
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
            REAL_t log_singularityValue = self.kernel.getLogSingularityValue()
            specialQuadRule sQR
            singularityCancelationQuadRule qr
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
            REAL_t phi_x = 0., phi_y = 0.

        if panel == COMMON_FACE:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:

                qr = singularityCancelationQuadRule2D(panel,
                                                      self.singularityCancelationIntegrandWithinElement+singularityValue,
                                                      log_singularityValue,
                                                      self.quad_order_diagonal,
                                                      self.quad_order_diagonalV,
                                                      1)
                qr.scaleWeights(4.0)
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
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PHI[dof, i, 0] = phi_x
                        PHI[dof, i, 1] = phi_y

                sQR = specialQuadRule(qr, PHI3=PHI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes, self.kernel.valueSize), dtype=REAL)
                    self.temp2 = uninitialized((qr.num_nodes, self.kernel.valueSize), dtype=REAL)
            self.qrFace = sQR.qr
            self.PHI_face = sQR.PHI3
        elif panel == COMMON_EDGE:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:

                qr = singularityCancelationQuadRule2D(panel,
                                                      self.singularityCancelationIntegrandAcrossElements+singularityValue,
                                                      log_singularityValue,
                                                      self.quad_order_diagonal,
                                                      self.quad_order_diagonalV,
                                                      1)
                qr.scaleWeights(4.0)
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
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PHI[dof, i, 0] = phi_x
                        PHI[dof, i, 1] = phi_y

                for dof in range((dim+1)*dofs_per_vertex, (dim+1)*dofs_per_vertex+dofs_per_edge):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PHI[dof, i, 0] = phi_x
                        PHI[dof, i, 1] = phi_y

                for dof in range(2*dofs_per_vertex, (dim+1)*dofs_per_vertex):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PHI[dof, i, 0] = phi_x
                        PHI[dof, i, 1] = 0
                        PHI[dofs_per_element+dof-2*dofs_per_vertex, i, 0] = 0
                        PHI[dofs_per_element+dof-2*dofs_per_vertex, i, 1] = phi_y

                for dof in range((dim+1)*dofs_per_vertex+dofs_per_edge, dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PHI[dof, i, 0] = phi_x
                        PHI[dof, i, 1] = 0
                        PHI[dofs_per_element+dof-2*dofs_per_vertex-dofs_per_edge, i, 0] = 0
                        PHI[dofs_per_element+dof-2*dofs_per_vertex-dofs_per_edge, i, 1] = phi_y

                sQR = specialQuadRule(qr, PHI3=PHI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes, self.kernel.valueSize), dtype=REAL)
                    self.temp2 = uninitialized((qr.num_nodes, self.kernel.valueSize), dtype=REAL)
            self.qrEdge = sQR.qr
            self.PHI_edge = sQR.PHI3
        elif panel == COMMON_VERTEX:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:

                qr = singularityCancelationQuadRule2D(panel,
                                                      self.singularityCancelationIntegrandAcrossElements+singularityValue,
                                                      log_singularityValue,
                                                      self.quad_order_diagonal,
                                                      self.quad_order_diagonalV,
                                                      1)
                qr.scaleWeights(4.0)
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
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PHI[dof, i, 0] = phi_x
                        PHI[dof, i, 1] = phi_y

                for dof in range(dofs_per_vertex, dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        lcl_bary_y[0] = qr.nodes[3, i]
                        lcl_bary_y[1] = qr.nodes[4, i]
                        lcl_bary_y[2] = qr.nodes[5, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PHI[dof, i, 0] = phi_x
                        PHI[dof, i, 1] = 0
                        PHI[dofs_per_element+dof-dofs_per_vertex, i, 0] = 0
                        PHI[dofs_per_element+dof-dofs_per_vertex, i, 1] = phi_y

                sQR = specialQuadRule(qr, PHI3=PHI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes, self.kernel.valueSize), dtype=REAL)
                    self.temp2 = uninitialized((qr.num_nodes, self.kernel.valueSize), dtype=REAL)
            self.qrVertex = sQR.qr
            self.PHI_vertex = sQR.PHI3
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

    cdef void eval(self,
                   REAL_t[:, ::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        if panel >= 1:
            self.eval_distant_nonsym(contrib, panel, mask)
        else:
            self.eval_near_nonsym(contrib, panel, mask)


cdef class fractionalLaplacian2D_nonsym2(fractionalLaplacian2D_nonsym):
    """The local stiffness matrix

    .. math::

       0.5 \\int_{K_1}\\int_{K_2} [ u(x) - u(y) ] [ v(x)-v(y) ] \\gamma(x,y) dy dx

    for the 2D non-symmetric nonlocal Laplacian.
    """
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap dm,
                 target_order=None,
                 quad_order_diagonal=None,
                 num_dofs=None,
                 **kwargs):
        super(fractionalLaplacian2D_nonsym2, self).__init__(kernel, mesh, dm, target_order, quad_order_diagonal, num_dofs, **kwargs)

    cdef void eval(self,
                   REAL_t[:, ::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        if panel >= 1:
            self.eval_distant_nonsym2(contrib, panel, mask)
        else:
            self.eval_near_nonsym2(contrib, panel, mask)


cdef class fractionalLaplacian2D_boundary(fractionalLaplacian2DZeroExterior):
    """The local stiffness matrix

    .. math::

       \\int_{K}\\int_{e} [ u(x) v(x) n_{y} \\cdot \\Gamma(x,y) dy dx

    for the 2D nonlocal Laplacian.
    """
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap dm,
                 target_order=None,
                 quad_order_diagonal=None,
                 num_dofs=None,
                 **kwargs):
        super(fractionalLaplacian2D_boundary, self).__init__(kernel, mesh, dm, num_dofs, **kwargs)
        self.setKernel(kernel, quad_order_diagonal, target_order)

    cpdef void setKernel(self, Kernel kernel, quad_order_diagonal=None, target_order=None):
        self.kernel = kernel

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

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableSingularity):
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
            REAL_t log_singularityValue = self.kernel.getLogSingularityValue()
            INDEX_t dof
            singularityCancelationQuadRule qr
            specialQuadRule sQR
            REAL_t[:, ::1] PHI
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            shapeFunction sf
            REAL_t lcl_bary_x[3]
            REAL_t phi_x = 0.
        if panel == COMMON_EDGE:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:
                if singularityValue > -2.+1e-3:
                    qr = singularityCancelationQuadRule2D_boundary(panel, singularityValue, log_singularityValue, self.quad_order_diagonal, self.quad_order_diagonal)
                else:
                    qr = singularityCancelationQuadRule2D_boundary(panel, 2.+singularityValue, log_singularityValue, self.quad_order_diagonal, self.quad_order_diagonal)
                qr.scaleWeights(2.0)
                PHI = uninitialized((dofs_per_element, qr.num_nodes), dtype=REAL)

                for dof in range(dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        PHI[dof, i] = phi_x

                sQR = specialQuadRule(qr, PHI=PHI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes, self.kernel.valueSize), dtype=REAL)
            self.qrEdge = sQR.qr
            self.PHI_edge2 = sQR.PHI

        elif panel == COMMON_VERTEX:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:
                qr = singularityCancelationQuadRule2D_boundary(panel, singularityValue, log_singularityValue, self.quad_order_diagonal, self.quad_order_diagonal)
                qr.scaleWeights(2.0)
                PHI = uninitialized((dofs_per_element, qr.num_nodes), dtype=REAL)

                for dof in range(dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_x[2] = qr.nodes[2, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        PHI[dof, i] = phi_x

                sQR = specialQuadRule(qr, PHI=PHI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes, self.kernel.valueSize), dtype=REAL)
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
                   REAL_t[:, ::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        if panel >= 1:
            self.eval_distant_boundary(contrib, panel, mask)
        else:
            self.eval_near_boundary(contrib, panel, mask)
