###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from libc.math cimport log, ceil, fabs as abs
import numpy as np
cimport numpy as np

from PyNucleus_base.myTypes import REAL
from PyNucleus_base import uninitialized
from PyNucleus_fem.meshCy cimport meshBase
from PyNucleus_fem.quadrature cimport GaussJacobi
from PyNucleus_fem.DoFMaps cimport DoFMap, P1_DoFMap, P0_DoFMap, shapeFunction

include "kernel_params.pxi"
include "panelTypes.pxi"


cdef:
    MASK_t ALL
ALL.set()


cdef class fractionalLaplacian1DZeroExterior(nonlocalLaplacian1D):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap dm, num_dofs=None, **kwargs):
        manifold_dim2 = mesh.dim-1
        super(fractionalLaplacian1DZeroExterior, self).__init__(kernel, mesh, dm, num_dofs, manifold_dim2=manifold_dim2, **kwargs)
        self.symmetricCells = False
        self.symmetricLocalMatrix = True


cdef class singularityCancelationQuadRule1D(quadratureRule):
    def __init__(self, panelType panel, REAL_t singularity, INDEX_t quad_order_diagonal, INDEX_t quad_order_regular):
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

        if panel == COMMON_EDGE:
            # Differences of basis functions one the same element
            # cancel one singularity order i.e.
            # singularityCancelationFromContinuity = 1.
            qrId = GaussJacobi(((quad_order_regular, 1+singularity, 0),
                                (quad_order_regular, 0+singularity, 0)))

            bary = uninitialized((2*dim+2,
                                  qrId.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
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

                weights[i] = 2.0*qrId.weights[i]*(eta0*eta1)**(-singularity)

            super(singularityCancelationQuadRule1D, self).__init__(bary, weights, 2*dim+2)
        elif panel == COMMON_VERTEX:
            qrVertex = GaussJacobi(((quad_order_regular, 1+singularity, 0),
                                    (quad_order_diagonal, 0, 0)))

            bary = uninitialized((2*dim+2,
                                  2*qrVertex.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]

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

                weights[offset+i] = qrVertex.weights[i] * eta0**(-singularity)

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

                weights[offset+i] = qrVertex.weights[i] * eta0**(-singularity)

            super(singularityCancelationQuadRule1D, self).__init__(bary, weights, 2*dim)


cdef class singularityCancelationQuadRule1D_boundary(quadratureRule):
    def __init__(self, panelType panel, REAL_t singularity, INDEX_t quad_order_diagonal, INDEX_t quad_order_regular):
        cdef:
            INDEX_t i
            REAL_t eta
            quadratureRule qrVertex
            INDEX_t dim = 1
            REAL_t lcl_bary_x[2]
            REAL_t lcl_bary_y[1]
            REAL_t[:, ::1] bary, bary_x, bary_y
            REAL_t[::1] weights

        if panel == COMMON_VERTEX:
            qrVertex = GaussJacobi(((quad_order_diagonal, singularity, 0), ))

            bary = uninitialized((2*dim+1,
                                  qrVertex.num_nodes), dtype=REAL)
            bary_x = bary[:dim+1, :]
            bary_y = bary[dim+1:, :]
            weights = uninitialized((qrVertex.num_nodes), dtype=REAL)

            for i in range(qrVertex.num_nodes):
                eta = qrVertex.nodes[0, i]

                lcl_bary_x[0] = 1-eta
                lcl_bary_x[1] = eta

                lcl_bary_y[0] = 1

                bary_x[0, i] = lcl_bary_x[0]
                bary_x[1, i] = lcl_bary_x[1]

                bary_y[0, i] = lcl_bary_y[0]

                weights[i] = qrVertex.weights[i] * eta**(-singularity)
            super(singularityCancelationQuadRule1D_boundary, self).__init__(bary, weights, 2*dim+1)


cdef class fractionalLaplacian1D(nonlocalLaplacian1D):
    """The local stiffness matrix

    .. math::

       \\int_{K_1}\\int_{K_2} (u(x)-u(y)) (v(x)-v(y)) \\gamma(x,y) dy dx

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

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableOrder):
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
            self.qrId = sQR.qr
            self.PSI_id = sQR.PSI
        elif panel == COMMON_VERTEX:
            try:
                sQR = self.specialQuadRules[(singularityValue, panel)]
            except KeyError:
                qr = singularityCancelationQuadRule1D(panel,
                                                      self.singularityCancelationIntegrandAcrossElements+singularityValue,
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
        cdef:
            INDEX_t k, m, i, j, I, J, l
            REAL_t vol, vol1 = self.vol1, vol2 = self.vol2, val
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            quadratureRule qr
            REAL_t[:, ::1] PSI
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t dim = 1
            INDEX_t valueSize = self.kernel.valueSize
            REAL_t x[1]
            REAL_t y[1]

        if panel >= 1:
            self.eval_distant(contrib, panel, mask)
            return
        elif panel == COMMON_EDGE:
            qr = self.qrId
            PSI = self.PSI_id
        elif panel == COMMON_VERTEX:
            qr = self.qrVertex
            PSI = self.PSI_vertex
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))

        vol = vol1*vol2

        for m in range(qr.num_nodes):
            for j in range(dim):
                x[j] = (simplex1[self.perm1[0], j]*qr.nodes[0, m] +
                        simplex1[self.perm1[1], j]*qr.nodes[1, m])
                y[j] = (simplex2[self.perm2[0], j]*qr.nodes[2, m] +
                        simplex2[self.perm2[1], j]*qr.nodes[3, m])
            self.kernel.evalPtr(dim,
                                &x[0],
                                &y[0],
                                &self.vec[0])
            for l in range(valueSize):
                self.temp[m, l] = qr.weights[m] * self.vec[l]

        contrib[:, :] = 0.
        for I in range(PSI.shape[0]):
            i = self.perm[I]
            for J in range(I, PSI.shape[0]):
                j = self.perm[J]
                if j < i:
                    k = 2*dofs_per_element*j-(j*(j+1) >> 1) + i
                else:
                    k = 2*dofs_per_element*i-(i*(i+1) >> 1) + j
                if mask[k]:
                    for l in range(valueSize):
                        val = 0.
                        for m in range(qr.num_nodes):
                            val += self.temp[m, l] * PSI[I, m] * PSI[J, m]
                        contrib[k, l] = val*vol


cdef class fractionalLaplacian1D_nonsym(fractionalLaplacian1D):
    """The local stiffness matrix

    .. math::

       \\int_{K_1}\\int_{K_2} [ u(x) \\gamma(x,y) - u(y) \\gamma(y,x) ] [ v(x)-v(y) ] dy dx

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
        super(fractionalLaplacian1D_nonsym, self).__init__(kernel, mesh, dm, num_dofs, **kwargs)
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
        cdef:
            INDEX_t k, m, i, j, I, J, l
            REAL_t vol, vol1 = self.vol1, vol2 = self.vol2, val
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            quadratureRule qr
            REAL_t[:, :, ::1] PHI
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t dim = 1
            REAL_t x[1]
            REAL_t y[1]
            INDEX_t valueSize = self.kernel.valueSize

        if panel >= 1:
            self.eval_distant_nonsym(contrib, panel, mask)
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
            self.kernel.evalPtr(dim, &x[0], &y[0], &self.vec[0])
            self.kernel.evalPtr(dim, &y[0], &x[0], &self.vec2[0])
            for l in range(valueSize):
                self.temp[m, l] = qr.weights[m] * self.vec[l]
                self.temp2[m, l] = qr.weights[m] * self.vec2[l]

        contrib[:, :] = 0.
        for I in range(PHI.shape[0]):
            i = self.perm[I]
            for J in range(PHI.shape[0]):
                j = self.perm[J]
                k = i*(2*dofs_per_element)+j
                if mask[k]:
                    for l in range(valueSize):
                        val = 0.
                        for m in range(qr.num_nodes):
                            val += (self.temp[m, l] * PHI[I, m, 0] - self.temp2[m, l] * PHI[I, m, 1]) * (PHI[J, m, 0] - PHI[J, m, 1])
                        contrib[k, l] = val*vol


cdef class fractionalLaplacian1D_boundary(fractionalLaplacian1DZeroExterior):
    """The local stiffness matrix

    .. math::

       \\int_{K}\\int_{e} [ u(x) v(x) n_{y} \\cdot \\frac{x-y}{|x-y|} \\Gamma(x,y) dy dx

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

        if (self.kernel.kernelType != FRACTIONAL) or (not self.kernel.variableOrder):
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
                    qr = singularityCancelationQuadRule1D_boundary(panel, singularityValue, self.quad_order_diagonal, 1)
                else:
                    qr = singularityCancelationQuadRule1D_boundary(panel, 2.+singularityValue, self.quad_order_diagonal, 1)
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
            self.PHI_vertex = sQR.PHI
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
        cdef:
            REAL_t vol = self.vol1, val
            INDEX_t i, j, k, m, l
            quadratureRule qr
            REAL_t[:, ::1] PHI
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            INDEX_t dim = 1
            REAL_t x[1]
            REAL_t y[1]
            INDEX_t dofs_per_element = self.DoFMap.dofs_per_element
            INDEX_t valueSize = self.kernel.valueSize

        # Kernel:
        #  \Gamma(x,y) = n \dot (x-y) * C(d,s) / (2s) / |x-y|^{d+2s}
        # with inward normal n.
        #
        # Rewrite as
        #  \Gamma(x,y) = [ n \dot (x-y)/|x-y| ] * [ C(d,s) / (2s) / |x-y|^{d-1+2s} ]
        #                                         \--------------------------------/
        #                                                 |
        #                                           boundaryKernel
        #
        # In 1D:
        #  n = (x-y)/|x-y|
        # so
        #  n \dot (x-y) / |x-y| = 1

        if panel >= 1:
            self.eval_distant_boundary(contrib, panel, mask)
        elif panel == COMMON_VERTEX:
            qr = self.qrVertex
            PHI = self.PHI_vertex

            for m in range(qr.num_nodes):
                for j in range(dim):
                    x[j] = (simplex1[self.perm1[0], j]*qr.nodes[0, m] +
                            simplex1[self.perm1[1], j]*qr.nodes[1, m])
                    y[j] = simplex2[self.perm2[0], j]*qr.nodes[2, m]
                self.kernel.evalPtr(dim, &x[0], &y[0], &self.vec[0])
                for l in range(valueSize):
                    self.temp[m, l] = qr.weights[m] * self.vec[l]

            contrib[:, :] = 0.

            for I in range(dofs_per_element):
                i = self.perm[I]
                for J in range(I, dofs_per_element):
                    j = self.perm[J]
                    if j < i:
                        k = dofs_per_element*j-(j*(j+1) >> 1) + i
                    else:
                        k = dofs_per_element*i-(i*(i+1) >> 1) + j
                    if mask[k]:
                        for l in range(valueSize):
                            val = 0.
                            for m in range(qr.num_nodes):
                                val += self.temp[m, l] * PHI[I, m] * PHI[J, m]
                            contrib[k, l] = val*vol
        else:
            raise NotImplementedError('Unknown panel type: {}'.format(panel))


