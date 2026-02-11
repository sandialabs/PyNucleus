###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from libc.math cimport sqrt, log, ceil, fabs as abs
import numpy as np
cimport numpy as np

from PyNucleus_base.myTypes import REAL
from PyNucleus_base import uninitialized
from PyNucleus_base.blas cimport mydot
from PyNucleus_fem.meshCy cimport meshBase
from PyNucleus_fem.quadrature cimport GaussJacobi, LogGaussJacobi
from PyNucleus_fem.DoFMaps cimport DoFMap, P1_DoFMap, P0_DoFMap, shapeFunction

include "kernel_params.pxi"
include "panelTypes.pxi"


cdef:
    MASK_t ALL
ALL.set()


cdef class fractionalLaplacian1DZeroExterior(nonlocalLaplacian1D):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap dm, num_dofs=None, **kwargs):
        manifold_dim2 = mesh.manifold_dim-1
        super(fractionalLaplacian1DZeroExterior, self).__init__(kernel, mesh, dm, num_dofs, manifold_dim2=manifold_dim2, **kwargs)
        self.symmetricCells = False
        self.symmetricLocalMatrix = True


cdef class singularityCancelationQuadRule1D(singularityCancelationQuadRule):
    def __init__(self, panelType panel, REAL_t singularity, REAL_t log_singularity, INDEX_t quad_order_diagonal, INDEX_t quad_order_regular):
        cdef:
            INDEX_t i
            REAL_t eta0, eta1, x, y
            quadratureRule qrId, qrVertex
            INDEX_t dim = 1
            REAL_t lcl_bary_x[2]
            REAL_t lcl_bary_y[2]
            REAL_t[:, ::1] bary, bary_x, bary_y
            REAL_t[::1] weights
            INDEX_t offset
            REAL_t[::1] singularPart

        if panel == COMMON_EDGE:
            if log_singularity == 0:
                qrId = GaussJacobi(((quad_order_regular, 1+singularity, 0),
                                    (quad_order_regular, 0+singularity, 0)))
            elif log_singularity == 1:
                qr10 = LogGaussJacobi(((quad_order_regular, 1+singularity, 0, 1),
                                       (quad_order_regular, 0+singularity, 0, 0)))
                qr01 = LogGaussJacobi(((quad_order_regular, 1+singularity, 0, 0),
                                       (quad_order_regular, 0+singularity, 0, 1)))
                qrId = qr10+qr01
            elif log_singularity == 2:
                qr20 = LogGaussJacobi(((quad_order_regular, 1+singularity, 0, 2),
                                       (quad_order_regular, 0+singularity, 0, 0)))
                qr11 = LogGaussJacobi(((quad_order_regular, 1+singularity, 0, 1),
                                       (quad_order_regular, 0+singularity, 0, 1)))
                qr02 = LogGaussJacobi(((quad_order_regular, 1+singularity, 0, 0),
                                       (quad_order_regular, 0+singularity, 0, 2)))
                for i in range(qr11.num_nodes):
                    qr11.weights[i] *= 2.
                qrId = qr20+qr11+qr02
            else:
                raise NotImplementedError(log_singularity)

            bary = uninitialized((2*dim+2,
                                  qrId.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            singularPart = uninitialized((qrId.num_nodes), dtype=REAL)
            weights = uninitialized((qrId.num_nodes), dtype=REAL)
            # COMMON_FACE panels
            for i in range(qrId.num_nodes):
                eta0 = qrId.nodes[0, i]
                eta1 = qrId.nodes[1, i]

                x = eta0*(1-eta1)
                y = eta0

                lcl_bary_x[0] = 1-x
                lcl_bary_x[1] = x

                lcl_bary_y[0] = 1-y
                lcl_bary_y[1] = y

                bary_x[0, i] = lcl_bary_x[0]
                bary_x[1, i] = lcl_bary_x[1]

                bary_y[0, i] = lcl_bary_y[0]
                bary_y[1, i] = lcl_bary_y[1]

                singularPart[i] = eta0*eta1
                weights[i] = 2.0*qrId.weights[i]*singularPart[i]**(-singularity)

            super(singularityCancelationQuadRule1D, self).__init__(bary, weights, 2*dim+2)
        elif panel == COMMON_VERTEX:
            if log_singularity == 0:
                qrVertex = GaussJacobi(((quad_order_regular, 1+singularity, 0),
                                        (quad_order_diagonal, 0, 0)))
            else:
                qrVertex = LogGaussJacobi(((quad_order_regular, 1+singularity, 0, log_singularity),
                                           (quad_order_diagonal, 0, 0, 0)))

            bary = uninitialized((2*dim+2,
                                  2*qrVertex.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]

            singularPart = uninitialized((2*qrVertex.num_nodes), dtype=REAL)
            weights = uninitialized((2*qrVertex.num_nodes), dtype=REAL)

            # panels with common vertex
            # first integral
            offset = 0
            for i in range(qrVertex.num_nodes):
                eta0 = qrVertex.nodes[0, i]
                eta1 = qrVertex.nodes[1, i]

                x = eta0*eta1
                y = eta0

                lcl_bary_x[0] = 1-x
                lcl_bary_x[1] = x

                lcl_bary_y[0] = 1-y
                lcl_bary_y[1] = y

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]

                singularPart[offset+i] = eta0
                weights[offset+i] = qrVertex.weights[i] * singularPart[offset+i]**(-singularity)

            # second integral
            offset = qrVertex.num_nodes
            for i in range(qrVertex.num_nodes):
                eta0 = qrVertex.nodes[0, i]
                eta1 = qrVertex.nodes[1, i]

                x = eta0
                y = eta0*eta1

                lcl_bary_x[0] = 1-x
                lcl_bary_x[1] = x

                lcl_bary_y[0] = 1-y
                lcl_bary_y[1] = y

                bary_x[0, offset+i] = lcl_bary_x[0]
                bary_x[1, offset+i] = lcl_bary_x[1]

                bary_y[0, offset+i] = lcl_bary_y[0]
                bary_y[1, offset+i] = lcl_bary_y[1]

                singularPart[offset+i] = eta0
                weights[offset+i] = qrVertex.weights[i] * singularPart[offset+i]**(-singularity)
            super(singularityCancelationQuadRule1D, self).__init__(bary, weights, 2*dim)
        else:
            raise NotImplementedError(panel)

        self.singularPart = singularPart


cdef class singularityCancelationQuadRule1D_boundary(singularityCancelationQuadRule):
    def __init__(self, panelType panel, REAL_t singularity, REAL_t log_singularity, INDEX_t quad_order_diagonal, INDEX_t quad_order_regular):
        cdef:
            INDEX_t i
            REAL_t eta
            quadratureRule qrVertex
            INDEX_t dim = 1
            REAL_t lcl_bary_x[2]
            REAL_t lcl_bary_y[1]
            REAL_t[:, ::1] bary, bary_x, bary_y
            REAL_t[::1] weights
            REAL_t[::1] singularPart

        if panel == COMMON_VERTEX:
            if log_singularity == 0:
                qrVertex = GaussJacobi(((quad_order_diagonal, singularity, 0), ))
            else:
                qrVertex = LogGaussJacobi(((quad_order_diagonal, singularity, 0, log_singularity), ))

            bary = uninitialized((2*dim+1,
                                  qrVertex.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            singularPart = uninitialized((qrVertex.num_nodes), dtype=REAL)
            weights = uninitialized((qrVertex.num_nodes), dtype=REAL)

            for i in range(qrVertex.num_nodes):
                eta = qrVertex.nodes[0, i]

                lcl_bary_x[0] = 1-eta
                lcl_bary_x[1] = eta

                lcl_bary_y[0] = 1

                bary_x[0, i] = lcl_bary_x[0]
                bary_x[1, i] = lcl_bary_x[1]

                bary_y[0, i] = lcl_bary_y[0]

                singularPart[i] = eta
                weights[i] = qrVertex.weights[i] * singularPart[i]**(-singularity)
            super(singularityCancelationQuadRule1D_boundary, self).__init__(bary, weights, 2*dim+1)
        self.singularPart = singularPart


cdef class fractionalLaplacian1D(nonlocalLaplacian1D):
    """The local stiffness matrix

    .. math::

       0.5 \\int_{K_1}\\int_{K_2} (u(x)-u(y)) (v(x)-v(y)) \\gamma(x,y) dy dx

    for the symmetric 1D nonlocal Laplacian.
    """
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap dm,
                 quad_order_diagonal=None,
                 target_order=None,
                 num_dofs=None,
                 **kwargs):
        super(fractionalLaplacian1D, self).__init__(kernel, mesh, dm, num_dofs, **kwargs)
        self.setKernel(kernel, quad_order_diagonal, target_order)
        self.symmetricCells = True

    cpdef void setKernel(self, Kernel kernel, quad_order_diagonal=None, target_order=None):
        cdef:
            REAL_t smin, smax
        self.kernel = kernel

        # The integrand (excluding the kernel) cancels 2 orders of the singularity within an element.
        self.singularityCancelationIntegrandWithinElement = 2.
        # The integrand (excluding the kernel) cancels 2 orders of the
        # singularity across elements for continuous finite elements.
        if isinstance(self.DoFMap, P0_DoFMap):
            assert self.kernel.max_singularity > -2., "Discontinuous finite elements are not conforming for singularity order {} <= -2.".format(self.kernel.max_singularity)
            self.singularityCancelationIntegrandAcrossElements = 0.
        else:
            self.singularityCancelationIntegrandAcrossElements = 2.

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

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableSingularity):
            self.getNearQuadRule(COMMON_EDGE)
            self.getNearQuadRule(COMMON_VERTEX)

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        cdef:
            panelType panel, panel2
            REAL_t logdh1 = log(d/h1), logdh2 = log(d/h2)
            REAL_t s = max(-0.5*(self.kernel.getSingularityValue()+1), 0.)
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
            REAL_t singularityValue = self.kernel.getSingularityValue()
            REAL_t log_singularityValue = self.kernel.getLogSingularityValue()
            specialQuadRule sQR
            quadratureRule qr
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t dofs_per_vertex = self.DoFMap.dofs_per_vertex
            INDEX_t dm_order = max(self.DoFMap.polynomialOrder, 1)
            shapeFunction sf
            REAL_t lcl_bary_x[2]
            REAL_t lcl_bary_y[2]
            REAL_t[:, ::1] PSI
            INDEX_t dof
            REAL_t phi_x = 0., phi_y = 0.

        if panel == COMMON_EDGE:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:
                qr = singularityCancelationQuadRule1D(panel,
                                                      self.singularityCancelationIntegrandWithinElement+singularityValue,
                                                      log_singularityValue,
                                                      self.quad_order_diagonal,
                                                      2*dm_order)
                PSI = uninitialized((dofs_per_element, qr.num_nodes), dtype=REAL)

                for dof in range(dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_y[0] = qr.nodes[2, i]
                        lcl_bary_y[1] = qr.nodes[3, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PSI[dof, i] = phi_x-phi_y

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
                qr = singularityCancelationQuadRule1D(panel,
                                                      self.singularityCancelationIntegrandAcrossElements+singularityValue,
                                                      log_singularityValue,
                                                      self.quad_order_diagonal,
                                                      2*dm_order)
                PSI = uninitialized((2*dofs_per_element - dofs_per_vertex,
                                     qr.num_nodes), dtype=REAL)

                for dof in range(dofs_per_vertex):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_y[0] = qr.nodes[2, i]
                        lcl_bary_y[1] = qr.nodes[3, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PSI[dof, i] = phi_x-phi_y

                for dof in range(dofs_per_vertex, dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_y[0] = qr.nodes[2, i]
                        lcl_bary_y[1] = qr.nodes[3, i]
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
        return (super(fractionalLaplacian1D, self).__repr__() +
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


cdef class fractionalLaplacian1D_nonsym(fractionalLaplacian1D):
    """The local stiffness matrix

    .. math::

       0.5 \\int_{K_1}\\int_{K_2} [ u(x) \\gamma(x,y) - u(y) \\gamma(y,x) ] [ v(x)-v(y) ] dy dx

    for the non-symmetric 1D nonlocal Laplacian.
    """
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap dm,
                 **kwargs):
        super(fractionalLaplacian1D_nonsym, self).__init__(kernel, mesh, dm, **kwargs)
        self.symmetricLocalMatrix = False
        self.symmetricCells = False

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        cdef:
            panelType panel, panel2
            REAL_t logdh1 = log(d/h1), logdh2 = log(d/h2)
            REAL_t s = max(-0.5*(self.kernel.getSingularityValue()+1), 0.)
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
            self.addQuadRule_nonSym(panel)
        return panel

    cdef void getNearQuadRule(self, panelType panel):
        cdef:
            INDEX_t i
            REAL_t singularityValue = self.kernel.getSingularityValue()
            REAL_t log_singularityValue = self.kernel.getLogSingularityValue()
            specialQuadRule sQR
            quadratureRule qr
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t dofs_per_vertex = self.DoFMap.dofs_per_vertex
            INDEX_t dm_order = max(self.DoFMap.polynomialOrder, 1)
            shapeFunction sf
            REAL_t lcl_bary_x[2]
            REAL_t lcl_bary_y[2]
            REAL_t[:, :, ::1] PHI
            INDEX_t dof
            REAL_t phi_x = 0., phi_y = 0.

        if panel == COMMON_EDGE:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:
                qr = singularityCancelationQuadRule1D(panel,
                                                      self.singularityCancelationIntegrandWithinElement+singularityValue,
                                                      log_singularityValue,
                                                      self.quad_order_diagonal,
                                                      2*dm_order)
                PHI = uninitialized((dofs_per_element,
                                     qr.num_nodes,
                                     2), dtype=REAL)

                for dof in range(dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_y[0] = qr.nodes[2, i]
                        lcl_bary_y[1] = qr.nodes[3, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PHI[dof, i, 0] = phi_x
                        PHI[dof, i, 1] = phi_y

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
                qr = singularityCancelationQuadRule1D(panel,
                                                      self.singularityCancelationIntegrandAcrossElements+singularityValue,
                                                      log_singularityValue,
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
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PHI[dof, i, 0] = phi_x
                        PHI[dof, i, 1] = phi_y

                for dof in range(dofs_per_vertex, dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        lcl_bary_y[0] = qr.nodes[2, i]
                        lcl_bary_y[1] = qr.nodes[3, i]
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


cdef class fractionalLaplacian1D_nonsym2(fractionalLaplacian1D_nonsym):
    """The local stiffness matrix

    .. math::

       0.5 \\int_{K_1}\\int_{K_2} [ u(x) - u(y) ] [ v(x)-v(y) ] \\gamma(x,y) dy dx

    for the non-symmetric 1D nonlocal Laplacian.
    """
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap dm,
                 target_order=None,
                 quad_order_diagonal=None,
                 num_dofs=None,
                 **kwargs):
        super(fractionalLaplacian1D_nonsym2, self).__init__(kernel, mesh, dm, target_order, quad_order_diagonal, num_dofs, **kwargs)

    cdef void eval(self,
                   REAL_t[:, ::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        if panel >= 1:
            self.eval_distant_nonsym2(contrib, panel, mask)
        else:
            self.eval_near_nonsym2(contrib, panel, mask)


cdef class fractionalLaplacian1D_boundary(fractionalLaplacian1DZeroExterior):
    """The local stiffness matrix

    .. math::

       \\int_{K}\\int_{e} u(x) v(x) n_{y} \\cdot \\Gamma(x,y) dy dx

    for the 1D nonlocal Laplacian.
    """
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap dm,
                 quad_order_diagonal=None,
                 target_order=None,
                 num_dofs=None,
                 **kwargs):
        super(fractionalLaplacian1D_boundary, self).__init__(kernel, mesh, dm, num_dofs, **kwargs)
        self.setKernel(kernel, quad_order_diagonal, target_order)

    cpdef void setKernel(self, Kernel kernel, quad_order_diagonal=None, target_order=None):
        self.kernel = kernel

        smin = max(0.5*(-self.kernel.min_singularity), 0.)
        smax = max(0.5*(-self.kernel.max_singularity), 0.)
        if target_order is None:
            # this is the desired local quadrature error
            target_order = self.DoFMap.polynomialOrder+1-smin
        self.target_order = target_order

        if quad_order_diagonal is None:
            # measured log(2 rho_2) = 0.4
            quad_order_diagonal = max(np.ceil(((target_order+1.)*log(self.num_dofs*self.H0)+(2.*smax-1.)*abs(log(self.hmin/self.H0)))/0.8), 2)
        self.quad_order_diagonal = quad_order_diagonal

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableSingularity):
            self.getNearQuadRule(COMMON_VERTEX)

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        cdef:
            panelType panel, panel2
            REAL_t logdh1 = max(log(d/h1), 0.), logdh2 = max(log(d/h2), 0.)
            REAL_t s = max(0.5*(-self.kernel.getSingularityValue()-1.), 0.)
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
            self.addQuadRule_boundary(panel)
        return panel

    cdef void getNearQuadRule(self, panelType panel):
        cdef:
            INDEX_t i
            REAL_t singularityValue = self.kernel.getSingularityValue()
            REAL_t log_singularityValue = self.kernel.getLogSingularityValue()
            specialQuadRule sQR
            quadratureRule qr
            INDEX_t dof
            REAL_t[:, ::1] PHI
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            REAL_t lcl_bary_x[2]
            shapeFunction sf
            REAL_t phi_x = 0.
        if panel == COMMON_VERTEX:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:
                if singularityValue > -1.+1e-3:
                    qr = singularityCancelationQuadRule1D_boundary(panel, singularityValue, log_singularityValue, self.quad_order_diagonal, 1)
                else:
                    qr = singularityCancelationQuadRule1D_boundary(panel, 2.+singularityValue, log_singularityValue, self.quad_order_diagonal, 1)
                PHI = uninitialized((dofs_per_element, qr.num_nodes), dtype=REAL)

                for dof in range(dofs_per_element):
                    sf = self.getLocalShapeFunction(dof)
                    for i in range(qr.num_nodes):
                        lcl_bary_x[0] = qr.nodes[0, i]
                        lcl_bary_x[1] = qr.nodes[1, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        PHI[dof, i] = phi_x

                sQR = specialQuadRule(qr, PHI=PHI)
                self.specialQuadRules[(singularityValue, panel)] = sQR
                if qr.num_nodes > self.temp.shape[0]:
                    self.temp = uninitialized((qr.num_nodes, self.kernel.valueSize), dtype=REAL)
            self.qrVertex = sQR.qr
            self.PHI_vertex2 = sQR.PHI
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

    def __repr__(self):
        return (super(fractionalLaplacian1D_boundary, self).__repr__() +
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
            self.eval_distant_boundary(contrib, panel, mask)
        else:
            self.eval_near_boundary(contrib, panel, mask)
