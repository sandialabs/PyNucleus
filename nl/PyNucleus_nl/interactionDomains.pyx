###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

"""Defines different types of interaction domains."""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, M_PI as pi, cos, sin
import warnings
from PyNucleus_base.myTypes import REAL
from PyNucleus_base import uninitialized
from PyNucleus_fem.meshCy cimport getBarycentricCoords2D, volume1Dsimplex, volume2Dsimplex, meshBase
from PyNucleus_fem.functions cimport constant, constantMatrixFunction, sqrtAffineFunction

include "kernel_params.pxi"


cdef REAL_t inf = np.inf


cdef class interactionDomain(parametrizedTwoPointFunction):
    """Base class for all interaction domains."""

    def __init__(self, function horizonFun, BOOL_t isComplement, BOOL_t symmetric):
        super(interactionDomain, self).__init__(symmetric, 1)
        self.horizonFun = horizonFun
        self.complement = isComplement
        maxNumSubSimplices = 3
        maxDim = 2
        self.A_Simplex = uninitialized((maxNumSubSimplices, maxDim+1, maxDim+1), dtype=REAL)
        self.b_Simplex = uninitialized((maxNumSubSimplices, maxDim+1), dtype=REAL)
        self.vol_Simplex = uninitialized((maxNumSubSimplices), dtype=REAL)
        self.A_Node = uninitialized((maxNumSubSimplices, maxDim+1, maxDim+1), dtype=REAL)
        self.vol_Node = uninitialized((maxNumSubSimplices), dtype=REAL)

        self.intervals1 = uninitialized((4), dtype=REAL)
        self.intervals2 = uninitialized((3), dtype=REAL)

        self.specialOffsets = uninitialized((0, 2), dtype=REAL)

    def getComplement(self):
        raise NotImplementedError()

    cdef BOOL_t isInside(self, REAL_t[::1] x, REAL_t[::1] y):
        "Returns true if y is in the interaction region around x."
        raise NotImplementedError()

    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        "Returns the relative position of two simplices."
        raise NotImplementedError()

    def getRelativePosition_py(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        "Returns the relative position of two simplices."
        return self.getRelativePosition(simplex1, simplex2)

    cdef void startLoopSubSimplices_Simplex(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        raise NotImplementedError()

    cdef BOOL_t nextSubSimplex_Simplex(self, REAL_t[:, ::1] A, REAL_t[::1] b, REAL_t *vol):
        "Get the affine transformation onto the next sub-simplex for the outer integration."
        cdef:
            INDEX_t dim, i, j
            REAL_t l, r, v0, v1
        if self.iter_Simplex == self.iterEnd_Simplex:
            return False
        dim = getINDEX(self.params, fKDIM)
        if dim == 1:
            l = self.intervals1[self.iter_Simplex]
            r = self.intervals1[self.iter_Simplex+1]
            if r-l <= 0:
                self.iter_Simplex += 1
                return self.nextSubSimplex_Simplex(A, b, vol)
            v0 = self.intervals1[0]
            v1 = self.intervals1[3]
            A[0, 0] = r-l
            A[0, 1] = 0.
            A[1, 0] = 0.
            A[1, 1] = r-l
            b[0] = v1-r
            b[1] = l-v0
            vol[0] = r-l
            self.iter_Simplex += 1
            return True
        elif dim == 2:
            for i in range(3):
                b[i] = self.b_Simplex[self.iter_Simplex, i]
                for j in range(3):
                    A[i, j] = self.A_Simplex[self.iter_Simplex, i, j]
            vol[0] = self.vol_Simplex[self.iter_Simplex]
            self.iter_Simplex += 1
            return True

    cdef void startLoopSubSimplices_Node(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2):
        raise NotImplementedError()

    cdef BOOL_t nextSubSimplex_Node(self, REAL_t[:, ::1] A, REAL_t *vol):
        "Get the affine transformation onto the next sub-simplex for the inner integration."
        cdef:
            INDEX_t dim, i, j
        if self.iter_Node == self.iterEnd_Node:
            return False
        dim = getINDEX(self.params, fKDIM)
        for i in range(dim+1):
            for j in range(dim+1):
                A[i, j] = self.A_Node[self.iter_Node, i, j]
        vol[0] = self.vol_Node[self.iter_Node]
        self.iter_Node += 1
        return True

    def startLoopSubSimplices_Simplex_py(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        self.startLoopSubSimplices_Simplex(simplex1, simplex2)

    def nextSubSimplex_Simplex_py(self, REAL_t[:, ::1] A, REAL_t[::1] b, REAL_t[::1] vol):
        return self.nextSubSimplex_Simplex(A, b, &vol[0])

    def startLoopSubSimplices_Node_py(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2):
        self.startLoopSubSimplices_Node(node1, simplex2)

    def nextSubSimplex_Node_py(self, REAL_t[:, ::1] A, REAL_t[::1] vol):
        return self.nextSubSimplex_Node(A, &vol[0])

    def get_Surface(self, REAL_t[::1] node1):
        raise NotImplementedError()

    def plot_Surface(self, REAL_t[::1] node1):
        "Plot the surface of the interaction neighborhood."
        import matplotlib.pyplot as plt
        z = self.get_Surface(node1)
        plt.plot(z[:, 0], z[:, 1])

    def plot_Simplex(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2, BOOL_t plotSurfaces=True):
        import matplotlib.pyplot as plt
        cdef:
            INDEX_t dim = getINDEX(self.params, fKDIM)
            REAL_t vol = 0.

        if dim == 2:
            def plotSimplex(simplex, fillColor=None):
                if fillColor is not None:
                    plt.fill(np.concatenate((simplex[:, 0], [simplex[0, 0]])),
                             np.concatenate((simplex[:, 1], [simplex[0, 1]])), fillColor)
                else:
                    plt.plot(np.concatenate((simplex[:, 0], [simplex[0, 0]])),
                             np.concatenate((simplex[:, 1], [simplex[0, 1]])), c='k')

            if plotSurfaces:
                for j in range(dim+1):
                    self.plot_Surface(simplex2[j, :])
            self.startLoopSubSimplices_Simplex(simplex1, simplex2)
            A = uninitialized((dim+1, dim+1), dtype=REAL)
            b = uninitialized((dim+1, 1), dtype=REAL)
            while self.nextSubSimplex_Simplex(A, b[:, 0], &vol):
                subSimplex = A.T.dot(simplex1)+b.T.dot(simplex1)
                plotSimplex(subSimplex, 'r')

    def plot_Node(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2, BOOL_t plotMeshSimplex=True):
        import matplotlib.pyplot as plt
        cdef:
            INDEX_t dim = getINDEX(self.params, fKDIM)
            REAL_t vol = -1.
            volSubSimplex = 0.

        if dim == 1:
            def plotSimplex(simplex, fillColor=None):
                if fillColor is not None:
                    plt.plot([simplex[0, 0], simplex[1, 0]], [0., 0.], c=fillColor)
        elif dim == 2:
            def plotSimplex(simplex, fillColor=None):
                if fillColor is not None:
                    plt.fill(np.concatenate((simplex[:, 0], [simplex[0, 0]])),
                             np.concatenate((simplex[:, 1], [simplex[0, 1]])), fillColor)
                    plt.plot(np.concatenate((simplex[:, 0], [simplex[0, 0]])),
                             np.concatenate((simplex[:, 1], [simplex[0, 1]])), c='gray')
                else:
                    plt.plot(np.concatenate((simplex[:, 0], [simplex[0, 0]])),
                             np.concatenate((simplex[:, 1], [simplex[0, 1]])), c='k')
        else:
            raise NotImplementedError()

        if plotMeshSimplex:
            plotSimplex(simplex2)
        self.startLoopSubSimplices_Node(node1, simplex2)
        if self.identityMapping:
            fillColor = 'b'
        else:
            fillColor = 'r'
        k = 0
        volSimplex = 0.
        if dim == 1:
            volSimplex = volume1Dsimplex(simplex2)
        elif dim == 2:
            volSimplex = volume2Dsimplex(simplex2)
        A = uninitialized((dim+1, dim+1), dtype=REAL)
        while self.nextSubSimplex_Node(A, &vol):
            subSimplex = A.T.dot(simplex2)
            if dim == 1:
                volSubSimplex = volume1Dsimplex(subSimplex)
            elif dim == 2:
                volSubSimplex = volume2Dsimplex(subSimplex)
            if abs(volSubSimplex-vol*volSimplex) > 1e-10:
                print(k, volSubSimplex, vol*volSimplex)
            plotSimplex(subSimplex, fillColor)
            k += 1

    def volume_Node(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t dim = getINDEX(self.params, fKDIM)
            REAL_t vol = -1.
            REAL_t volSum = 0.

        A = uninitialized((dim+1, dim+1), dtype=REAL)
        self.startLoopSubSimplices_Node(node1, simplex2)
        while self.nextSubSimplex_Node(A, &vol):
            volSum += vol
        if dim == 1:
            volSum *= volume1Dsimplex(simplex2)
        elif dim == 2:
            volSum *= volume2Dsimplex(simplex2)
        return volSum

    def plot_Node_Mesh(self, REAL_t[::1] node1, meshBase mesh):
        import matplotlib.pyplot as plt
        cdef:
            INDEX_t dim = getINDEX(self.params, fKDIM), k
            REAL_t vol = -1.
            REAL_t[:, ::1] simplex2 = uninitialized((mesh.manifold_dim+1, mesh.dim), dtype=REAL)

        if dim == 1:
            def plotSimplex(simplex, fillColor=None):
                if fillColor is not None:
                    plt.plot([simplex[0, 0], simplex[1, 0]], [0., 0.], c=fillColor)
        elif dim == 2:
            def plotSimplex(simplex, fillColor=None):
                if fillColor is not None:
                    plt.fill(np.concatenate((simplex[:, 0], [simplex[0, 0]])),
                             np.concatenate((simplex[:, 1], [simplex[0, 1]])), fillColor)
                    plt.plot(np.concatenate((simplex[:, 0], [simplex[0, 0]])),
                             np.concatenate((simplex[:, 1], [simplex[0, 1]])), c='gray')
                else:
                    plt.plot(np.concatenate((simplex[:, 0], [simplex[0, 0]])),
                             np.concatenate((simplex[:, 1], [simplex[0, 1]])), c='k')
        else:
            raise NotImplementedError()

        A = uninitialized((dim+1, dim+1), dtype=REAL)
        for cellNo in range(mesh.num_cells):
            mesh.getSimplex(cellNo, simplex2)
            self.startLoopSubSimplices_Node(node1, simplex2)
            if self.identityMapping:
                fillColor = 'b'
            else:
                fillColor = 'r'
            k = 0
            while self.nextSubSimplex_Node(A, &vol):
                subSimplex = A.T.dot(simplex2)
                plotSimplex(subSimplex, fillColor)
                k += 1

        # plot center and surface
        if dim == 1:
            plt.scatter([node1[0]], [0.])
        elif dim == 2:
            plt.scatter([node1[0]], [node1[1]])
            self.plot_Surface(node1)

    def volume_Node_Mesh(self, REAL_t[::1] node1, meshBase mesh):
        cdef:
            REAL_t[:, ::1] simplex2 = uninitialized((mesh.manifold_dim+1, mesh.dim), dtype=REAL)
            REAL_t vol = 0.
            INDEX_t cellNo
        for cellNo in range(mesh.num_cells):
            mesh.getSimplex(cellNo, simplex2)
            vol += self.volume_Node(node1, simplex2)
        return vol

    def plot_Simplex_Mesh(self, REAL_t[:, ::1] simplex2, meshBase mesh):
        import matplotlib.pyplot as plt
        cdef:
            INDEX_t simplexNo1
            REAL_t[:, ::1] simplex1 = uninitialized((mesh.manifold_dim+1, mesh.dim), dtype=REAL)
        for simplexNo1 in range(mesh.num_cells):
            mesh.getSimplex(simplexNo1, simplex1)
            if self.getRelativePosition(simplex1, simplex2) in (0, 2):
                self.plot_Simplex(simplex1, simplex2, False)
        for vertexNo1 in range(simplex1.shape[0]):
            self.plot_Surface(simplex2[vertexNo1,:])
        plt.fill(np.concatenate((simplex2[:, 0], [simplex2[0, 0]])),
                 np.concatenate((simplex2[:, 1], [simplex2[0, 1]])), 'b', zorder=1)

    def __reduce__(self):
        return interactionDomain, (self.horizonFun, self.complement, self.symmetric)

    def getLongDescription(self):
        return ''


cdef class barycenterDomain(interactionDomain):
    def __init__(self, function horizonFun, BOOL_t isComplement, BOOL_t symmetric):
        super(barycenterDomain, self).__init__(horizonFun, isComplement, symmetric)
        self.A_Simplex[0, :, :] = 0.
        self.A_Simplex[0, 0, 0] = 1
        self.A_Simplex[0, 1, 1] = 1
        self.A_Simplex[0, 2, 2] = 1
        self.b_Simplex[0, :] = 0.
        self.vol_Simplex[0] = 1.

        self.A_Node[0, :, :] = 0.
        self.A_Node[0, 0, 0] = 1.
        self.A_Node[0, 1, 1] = 1.
        self.A_Node[0, 2, 2] = 1.
        self.vol_Node[0] = 1.

        self.identityMapping = True

    cdef void startLoopSubSimplices_Simplex(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            REAL_t bary[2]
            INDEX_t j, noVertex1, noVertex2
        for j in range(simplex2.shape[1]):
            bary[j] = 0.
        for noVertex2 in range(simplex2.shape[0]):
            for j in range(simplex2.shape[1]):
                bary[j] += simplex2[noVertex2, j]
        for j in range(simplex2.shape[1]):
            bary[j] /= simplex2.shape[0]

        self.iter_Simplex = 0
        self.iterEnd_Simplex = 0
        for noVertex1 in range(simplex1.shape[0]):
            if self.isInside(simplex1[noVertex1, :], bary):
                self.iterEnd_Simplex = 1
                break

    cdef void startLoopSubSimplices_Node(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2):
        cdef:
            REAL_t bary[2]
            INDEX_t j, noVertex2
        for j in range(simplex2.shape[1]):
            bary[j] = 0.
        for noVertex2 in range(simplex2.shape[0]):
            for j in range(simplex2.shape[1]):
                bary[j] += simplex2[noVertex2, j]
        for j in range(simplex2.shape[1]):
            bary[j] /= simplex2.shape[0]

        self.iter_Node = 0
        self.iterEnd_Node = 0
        if self.isInside(node1, bary):
            self.iterEnd_Node = 1


cdef class retriangulationDomain(interactionDomain):
    def __init__(self, function horizonFun, BOOL_t isComplement, BOOL_t symmetric):
        super(retriangulationDomain, self).__init__(horizonFun, isComplement, symmetric)

    cdef computeSpecialPoints(self):
        pass

    cdef INDEX_t findIntersections(self, REAL_t[::1] x, REAL_t[:, ::1] simplex, INDEX_t start, INDEX_t end, REAL_t[::1] intersections):
        "Find intersections between surface of interaction around x and line segment between vertices start and end of simplex."
        raise NotImplementedError()

    cdef void startLoopSubSimplices_Simplex(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        "Start the outer loop over sub-simplices of simplex1 for interactions with simplex2."
        cdef:
            INDEX_t dim = getINDEX(self.params, fKDIM)
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t horizon = sqrt(horizon2)
            REAL_t vol1, invVol1
            BOOL_t lr
            BOOL_t insideIJ[3][3]
            BOOL_t insideI[3]
            BOOL_t isInside
            INDEX_t numInside
            INDEX_t outside, inside1, inside2
            INDEX_t inside, outside1, outside2
            REAL_t d1, d2
            INDEX_t i, k, j
            REAL_t c1, c2
            REAL_t intersections[2]
        if dim == 1:
            lr = simplex1[0, 0] < simplex2[0, 0]
            vol1 = abs(simplex1[0, 0]-simplex1[1, 0])
            invVol1 = 1./vol1
            if lr:
                self.intervals1[0] = simplex1[0, 0]*invVol1
                self.intervals1[1] = max(simplex1[0, 0], simplex2[0, 0]-horizon)*invVol1
                self.intervals1[2] = min(simplex1[1, 0], simplex2[1, 0]-horizon)*invVol1
                self.intervals1[3] = simplex1[1, 0]*invVol1
                self.iter_Simplex = 1
                self.iterEnd_Simplex = 3
            else:
                self.intervals1[0] = simplex1[0, 0]*invVol1
                self.intervals1[1] = max(simplex1[0, 0], simplex2[0, 0]+horizon)*invVol1
                self.intervals1[2] = min(simplex1[1, 0], simplex2[1, 0]+horizon)*invVol1
                self.intervals1[3] = simplex1[1, 0]*invVol1
                self.iter_Simplex = 0
                self.iterEnd_Simplex = 2
        elif dim == 2:
            # self.b_Simplex[0, :] = 0.
            # self.A_Simplex[0, :, :] = 0
            # self.A_Simplex[0, 0, 0] = 1
            # self.A_Simplex[0, 1, 1] = 1
            # self.A_Simplex[0, 2, 2] = 1
            # self.vol_Simplex[0] = 1
            # self.iter_Simplex = 0
            # self.iterEnd_Simplex = 1

            self.identityMapping = False
            numInside = 0
            for i in range(3):
                isInside = False
                for k in range(3):
                    insideIJ[i][k] = self.isInside(simplex1[i, :], simplex2[k, :])
                    isInside |= insideIJ[i][k]
                insideI[i] = isInside
                numInside += isInside
            if numInside == 0:
                raise NotImplementedError()
            elif numInside == 1:
                inside = 0
                while not insideI[inside]:
                    inside += 1
                outside1 = (inside+1) % 3
                outside2 = (inside+2) % 3
                c1 = 0
                c2 = 0
                for j in range(3):
                    if insideIJ[inside][j]:
                        self.findIntersections(simplex2[j, :], simplex1, inside, outside1, intersections)
                        c1 = max(c1, intersections[0])
                        self.findIntersections(simplex2[j, :], simplex1, inside, outside2, intersections)
                        c2 = max(c2, intersections[0])
                self.iter_Simplex = 0
                if c1*c2 > 0:
                    self.A_Simplex[0, :, :] = 0.
                    self.b_Simplex[0, :] = 0.
                    self.A_Simplex[0, inside, inside] = c1+c2
                    self.A_Simplex[0, inside, outside1] = c2
                    self.A_Simplex[0, inside, outside2] = c1
                    self.A_Simplex[0, outside1, outside1] = c1
                    self.A_Simplex[0, outside2, outside2] = c2
                    self.b_Simplex[0, inside] = 1-c1-c2
                    self.vol_Simplex[0] = c1*c2

                    self.iterEnd_Simplex = 1
                else:
                    self.iterEnd_Simplex = 0
            elif numInside == 2:
                outside = 0
                while insideI[outside]:
                    outside += 1
                inside1 = (outside+1) % 3
                inside2 = (outside+2) % 3
                c1 = 1
                c2 = 1
                for j in range(3):
                    if insideIJ[inside1][j]:
                        self.findIntersections(simplex2[j, :], simplex1, outside, inside1, intersections)
                        c1 = min(c1, intersections[0])
                    if insideIJ[inside2][j]:
                        self.findIntersections(simplex2[j, :], simplex1, outside, inside2, intersections)
                        c2 = min(c2, intersections[0])
                d1 = 0.
                d2 = 0.
                for k in range(2):
                    d1 += (simplex2[outside, k]
                           + c1*(simplex2[inside1, k]-simplex2[outside, k])
                           - simplex2[inside2, k])**2
                    d2 += (simplex2[outside, k]
                           + c2*(simplex2[inside2, k]-simplex2[outside, k])
                           - simplex2[inside1, k])
                self.A_Simplex[:, :, :] = 0.
                self.b_Simplex[:, :] = 0.

                self.iter_Simplex = 0
                self.iterEnd_Simplex = 0
                if d1 < d2:
                    if 1-c1 > 0:
                        self.A_Simplex[self.iterEnd_Simplex, outside, outside] = 1-c1
                        self.A_Simplex[self.iterEnd_Simplex, inside1, inside1] = 1-c1
                        self.A_Simplex[self.iterEnd_Simplex, inside1, inside2] = -c1
                        self.A_Simplex[self.iterEnd_Simplex, inside2, inside2] = 1.
                        self.b_Simplex[self.iterEnd_Simplex, inside1] = c1
                        self.vol_Simplex[self.iterEnd_Simplex] = 1-c1
                        self.iterEnd_Simplex += 1

                    if c1*(1-c2) > 0.:
                        self.A_Simplex[self.iterEnd_Simplex, outside, outside] = 1-c2
                        self.A_Simplex[self.iterEnd_Simplex, inside2, inside2] = 1
                        self.A_Simplex[self.iterEnd_Simplex, inside2, outside] = c2
                        self.A_Simplex[self.iterEnd_Simplex, outside, inside1] = 1-c1
                        self.A_Simplex[self.iterEnd_Simplex, inside1, inside1] = c1
                        self.vol_Simplex[self.iterEnd_Simplex] = c1*(1-c2)
                        self.iterEnd_Simplex += 1
                else:
                    if 1-c2 > 0:
                        self.A_Simplex[self.iterEnd_Simplex, outside, outside] = 1-c2
                        self.A_Simplex[self.iterEnd_Simplex, inside2, inside2] = 1-c2
                        self.A_Simplex[self.iterEnd_Simplex, inside2, inside1] = -c2
                        self.A_Simplex[self.iterEnd_Simplex, inside1, inside1] = 1.
                        self.b_Simplex[self.iterEnd_Simplex, inside2] = c2
                        self.vol_Simplex[self.iterEnd_Simplex] = 1-c2
                        self.iterEnd_Simplex += 1

                    if c2*(1-c1) > 0.:
                        self.A_Simplex[self.iterEnd_Simplex, outside, outside] = 1-c1
                        self.A_Simplex[self.iterEnd_Simplex, inside1, inside1] = 1
                        self.A_Simplex[self.iterEnd_Simplex, inside1, outside] = c1
                        self.A_Simplex[self.iterEnd_Simplex, outside, inside2] = 1-c2
                        self.A_Simplex[self.iterEnd_Simplex, inside2, inside2] = c2
                        self.vol_Simplex[self.iterEnd_Simplex] = c2*(1-c1)
                        self.iterEnd_Simplex += 1
            else:
                self.b_Simplex[0, :] = 0.
                self.A_Simplex[0, :, :] = 0
                self.A_Simplex[0, 0, 0] = 1
                self.A_Simplex[0, 1, 1] = 1
                self.A_Simplex[0, 2, 2] = 1
                self.vol_Simplex[0] = 1
                self.iter_Simplex = 0
                self.iterEnd_Simplex = 1
                self.identityMapping = True
        else:
            raise NotImplementedError()

    cdef void startLoopSubSimplices_Node(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2):
        "Start the inner loop over sub-simplices of simplex2 for interactions with node1."
        # The A_Node matrices contains the barycentric coordinates of the nodes of the subsimplices.
        # the volume of this transformation is given by the determinant of A.
        cdef:
            INDEX_t dim = getINDEX(self.params, fKDIM)
            INDEX_t numInside
            BOOL_t ind[3]
            INDEX_t j
            REAL_t d1, d2
            INDEX_t inside, outside1, outside2
            INDEX_t outside, inside1, inside2
            REAL_t c1, c2
            INDEX_t v0, v1, v2
            INDEX_t numIntersections
            BOOL_t haveSpecialPoint
            REAL_t[2] specialPoint
            REAL_t[3] bary
            REAL_t[2] intersections

        self.iter_Node = 0
        self.iterEnd_Node = 0
        self.identityMapping = False

        numInside = 0
        for j in range(dim+1):
            ind[j] = self.isInside(node1, simplex2[j, :])
            numInside += ind[j]

        if dim == 1:
            if numInside == 0:
                numIntersections = self.findIntersections(node1, simplex2, 0, 1, intersections)

                if numIntersections == 2:
                    self.A_Node[self.iterEnd_Node, :, :] = 0.
                    self.A_Node[self.iterEnd_Node, 0, 0] = 1-intersections[0]
                    self.A_Node[self.iterEnd_Node, 1, 0] = intersections[0]
                    self.A_Node[self.iterEnd_Node, 1, 1] = intersections[1]
                    self.A_Node[self.iterEnd_Node, 0, 1] = 1.-intersections[1]
                    self.vol_Node[self.iterEnd_Node] = intersections[1]-intersections[0]
                    self.iterEnd_Node += 1

            elif numInside == 1:
                inside = 0
                while not ind[inside]:
                    inside += 1
                outside = (inside+1) % (dim+1)

                self.findIntersections(node1, simplex2, inside, outside, intersections)

                self.A_Node[self.iterEnd_Node, :, :] = 0.
                self.A_Node[self.iterEnd_Node, inside, inside] = 1.
                self.A_Node[self.iterEnd_Node, outside, outside] = intersections[0]
                self.A_Node[self.iterEnd_Node, inside, outside] = 1.-intersections[0]
                self.vol_Node[self.iterEnd_Node] = intersections[0]
                self.iterEnd_Node += 1
            else:
                self.A_Node[0, :, :] = 0.
                self.A_Node[0, 0, 0] = 1.
                self.A_Node[0, 1, 1] = 1.
                self.vol_Node[0] = 1.
                self.iterEnd_Node += 1
                self.identityMapping = True

        elif dim == 2:

            haveSpecialPoint = False
            if numInside < dim+1:
                self.computeSpecialPoints()
                for j in range(self.specialOffsets.shape[0]):
                    for k in range(dim):
                        specialPoint[k] = node1[k]+self.specialOffsets[j, k]
                    getBarycentricCoords2D(simplex2, specialPoint, bary)
                    if (bary[0] >= 0.) and (bary[1] >= 0.) and (bary[2] >= 0.):
                        haveSpecialPoint = True
                        break

            if numInside == 0:
                if haveSpecialPoint:
                    for j in range(dim+1):
                        v0 = j
                        v1 = (j+1) % (dim+1)
                        v2 = (j+2) % (dim+1)
                        numIntersections = self.findIntersections(node1, simplex2, v0, v1, intersections)
                        if numIntersections > 0:
                            self.A_Node[self.iterEnd_Node, :, :] = 0.
                            self.A_Node[self.iterEnd_Node, v0, v0] = 1-intersections[0]
                            self.A_Node[self.iterEnd_Node, v1, v0] = intersections[0]
                            self.A_Node[self.iterEnd_Node, v0, v1] = 1-intersections[1]
                            self.A_Node[self.iterEnd_Node, v1, v1] = intersections[1]
                            self.A_Node[self.iterEnd_Node, v0, v2] = bary[v0]
                            self.A_Node[self.iterEnd_Node, v1, v2] = bary[v1]
                            self.A_Node[self.iterEnd_Node, v2, v2] = bary[v2]
                            self.vol_Node[self.iterEnd_Node] = bary[v2]*(intersections[1]-intersections[0])
                            self.iterEnd_Node += 1
                            break
                # else: There can be a nonzero intersection, but we ignore it

            elif numInside == 1:
                inside = 0
                while not ind[inside]:
                    inside += 1
                outside1 = (inside+1) % (dim+1)
                outside2 = (inside+2) % (dim+1)
                self.findIntersections(node1, simplex2, inside, outside1, intersections)
                c1 = intersections[0]
                self.findIntersections(node1, simplex2, inside, outside2, intersections)
                c2 = intersections[0]
                numIntersections = self.findIntersections(node1, simplex2, outside1, outside2, intersections)

                if numIntersections == 0:
                    self.A_Node[self.iterEnd_Node, :, :] = 0.
                    self.A_Node[self.iterEnd_Node, inside, inside] = 1
                    self.A_Node[self.iterEnd_Node, inside, outside1] = 1-c1
                    self.A_Node[self.iterEnd_Node, outside1, outside1] = c1
                    self.A_Node[self.iterEnd_Node, outside2, outside2] = c2
                    self.A_Node[self.iterEnd_Node, inside, outside2] = 1-c2
                    self.vol_Node[self.iterEnd_Node] = c1*c2
                    self.iterEnd_Node += 1

                    if haveSpecialPoint:
                        self.A_Node[self.iterEnd_Node, :, :] = 0.
                        self.A_Node[self.iterEnd_Node, inside, inside] = bary[inside]
                        self.A_Node[self.iterEnd_Node, outside1, inside] = bary[outside1]
                        self.A_Node[self.iterEnd_Node, outside2, inside] = bary[outside2]
                        self.A_Node[self.iterEnd_Node, inside, outside1] = 1-c1
                        self.A_Node[self.iterEnd_Node, inside, outside2] = 1-c2
                        self.A_Node[self.iterEnd_Node, outside1, outside1] = c1
                        self.A_Node[self.iterEnd_Node, outside2, outside2] = c2
                        self.vol_Node[self.iterEnd_Node] = bary[outside1]*c2+bary[outside2]*c1-c1*c2
                        self.iterEnd_Node += 1
                else:
                    self.A_Node[self.iterEnd_Node, :, :] = 0.
                    self.A_Node[self.iterEnd_Node, inside, inside] = 1
                    self.A_Node[self.iterEnd_Node, outside1, outside1] = c1
                    self.A_Node[self.iterEnd_Node, inside, outside1] = 1-c1
                    self.A_Node[self.iterEnd_Node, outside2, outside2] = intersections[0]
                    self.A_Node[self.iterEnd_Node, outside1, outside2] = 1-intersections[0]
                    self.vol_Node[self.iterEnd_Node] = c1*intersections[0]
                    self.iterEnd_Node += 1

                    self.A_Node[self.iterEnd_Node, :, :] = 0.
                    self.A_Node[self.iterEnd_Node, inside, inside] = 1
                    self.A_Node[self.iterEnd_Node, outside1, outside1] = 1-intersections[0]
                    self.A_Node[self.iterEnd_Node, outside2, outside1] = intersections[0]
                    self.A_Node[self.iterEnd_Node, outside1, outside2] = 1-intersections[1]
                    self.A_Node[self.iterEnd_Node, outside2, outside2] = intersections[1]
                    self.vol_Node[self.iterEnd_Node] = intersections[1]-intersections[0]
                    self.iterEnd_Node += 1

                    self.A_Node[self.iterEnd_Node, :, :] = 0.
                    self.A_Node[self.iterEnd_Node, inside, inside] = 1
                    self.A_Node[self.iterEnd_Node, outside1, outside1] = 1-intersections[1]
                    self.A_Node[self.iterEnd_Node, outside2, outside1] = intersections[1]
                    self.A_Node[self.iterEnd_Node, outside2, outside2] = c2
                    self.A_Node[self.iterEnd_Node, inside, outside2] = 1-c2
                    self.vol_Node[self.iterEnd_Node] = c2*(1-intersections[1])
                    self.iterEnd_Node += 1

            elif numInside == 2:
                outside = 0
                while ind[outside]:
                    outside += 1
                inside1 = (outside+1) % 3
                inside2 = (outside+2) % 3
                self.findIntersections(node1, simplex2, outside, inside1, intersections)
                c1 = intersections[0]
                self.findIntersections(node1, simplex2, outside, inside2, intersections)
                c2 = intersections[0]

                # compute lengths of two possible ways to cut the element
                d1 = 0.
                d2 = 0.
                for k in range(dim):
                    d1 += (simplex2[inside2, k]-(c1*simplex2[inside1, k]+(1-c1)*simplex2[outside, k]))**2
                    d2 += (simplex2[inside1, k]-(c2*simplex2[inside2, k]+(1-c2)*simplex2[outside, k]))**2

                if d1 < d2:
                    self.A_Node[self.iterEnd_Node, :, :] = 0.
                    self.A_Node[self.iterEnd_Node, inside2, inside2] = 1
                    self.A_Node[self.iterEnd_Node, outside, outside] = 1-c2
                    self.A_Node[self.iterEnd_Node, inside2, outside] = c2
                    self.A_Node[self.iterEnd_Node, inside1, inside1] = c1
                    self.A_Node[self.iterEnd_Node, outside, inside1] = 1-c1
                    self.vol_Node[self.iterEnd_Node] = c1*(1-c2)
                    self.iterEnd_Node += 1

                    self.A_Node[self.iterEnd_Node, :, :] = 0.
                    self.A_Node[self.iterEnd_Node, inside1, inside1] = 1
                    self.A_Node[self.iterEnd_Node, inside2, inside2] = 1
                    self.A_Node[self.iterEnd_Node, outside, outside] = 1-c1
                    self.A_Node[self.iterEnd_Node, inside1, outside] = c1
                    self.vol_Node[self.iterEnd_Node] = 1-c1
                    self.iterEnd_Node += 1
                else:
                    self.A_Node[self.iterEnd_Node, :, :] = 0.
                    self.A_Node[self.iterEnd_Node, inside1, inside1] = 1
                    self.A_Node[self.iterEnd_Node, inside2, inside2] = c2
                    self.A_Node[self.iterEnd_Node, outside, inside2] = 1-c2
                    self.A_Node[self.iterEnd_Node, outside, outside] = 1-c1
                    self.A_Node[self.iterEnd_Node, inside1, outside] = c1
                    self.vol_Node[self.iterEnd_Node] = c2*(1-c1)
                    self.iterEnd_Node += 1

                    self.A_Node[self.iterEnd_Node, :, :] = 0.
                    self.A_Node[self.iterEnd_Node, inside1, inside1] = 1
                    self.A_Node[self.iterEnd_Node, inside2, inside2] = 1
                    self.A_Node[self.iterEnd_Node, outside, outside] = 1-c2
                    self.A_Node[self.iterEnd_Node, inside2, outside] = c2
                    self.vol_Node[self.iterEnd_Node] = 1-c2
                    self.iterEnd_Node += 1

                if haveSpecialPoint:
                    self.A_Node[self.iterEnd_Node, :, :] = 0.
                    self.A_Node[self.iterEnd_Node, outside, inside1] = 1-c1
                    self.A_Node[self.iterEnd_Node, inside1, inside1] = c1
                    self.A_Node[self.iterEnd_Node, outside, outside] = bary[outside]
                    self.A_Node[self.iterEnd_Node, inside1, outside] = bary[inside1]
                    self.A_Node[self.iterEnd_Node, inside2, outside] = bary[inside2]
                    self.A_Node[self.iterEnd_Node, inside2, inside2] = c2
                    self.A_Node[self.iterEnd_Node, outside, inside2] = 1-c2
                    self.vol_Node[self.iterEnd_Node] = np.linalg.det(self.A_Node[self.iterEnd_Node, :, :])
                    self.iterEnd_Node += 1

            else:
                self.A_Node[0, :, :] = 0.
                self.A_Node[0, 0, 0] = 1.
                self.A_Node[0, 1, 1] = 1.
                self.A_Node[0, 2, 2] = 1.
                self.vol_Node[0] = 1.
                self.iterEnd_Node += 1
                self.identityMapping = True

        else:
            raise NotImplementedError()


cdef class fullSpace(interactionDomain):
    """Full space interaction domain, i.e. infinite interaction horizon."""

    def __init__(self):
        super(fullSpace, self).__init__(constant(np.inf), False, True)

    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        return INTERACT

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            INDEX_t i
            REAL_t d2 = 0.
        for i in range(x.shape[0]):
            d2 += (x[i]-y[i])**2
        self.dist2 = d2
        value[0] = 1.

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            INDEX_t i
            REAL_t d2 = 0.
        for i in range(dim):
            d2 += (x[i]-y[i])**2
        self.dist2 = d2
        value[0] = 1.

    def __repr__(self):
        dim = getINDEX(self.params, fKDIM)
        return 'R^{}'.format(dim)

    def __reduce__(self):
        return fullSpace, ()


cdef class ball2_retriangulation(retriangulationDomain):
    """l2 ball interaction domain"""
    def __init__(self, function horizonFun):
        super(ball2_retriangulation, self).__init__(horizonFun, False, True)

    def getComplement(self):
        return ball2Complement(self.horizonFun)

    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t i, k, j
            INDEX_t noSimplex1 = simplex1.shape[0]
            INDEX_t noSimplex2 = simplex2.shape[0]
            REAL_t d2
            REAL_t dmin2 = inf
            REAL_t dmax2 = 0.
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            INDEX_t dim = getINDEX(self.params, fKDIM)
        for i in range(noSimplex1):
            for k in range(noSimplex2):
                d2 = 0.
                for j in range(dim):
                    d2 += (simplex1[i, j] - simplex2[k, j])**2
                dmin2 = min(dmin2, d2)
                dmax2 = max(dmax2, d2)
        if dmin2 >= horizon2:
            self.relPos = REMOTE
        elif dmax2 <= horizon2:
            self.relPos = INTERACT
        else:
            self.relPos = CUT
        return self.relPos

    cdef BOOL_t isInside(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            BOOL_t isInside
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            INDEX_t j
            REAL_t d2 = 0.
        for j in range(x.shape[0]):
            d2 += (x[j] - y[j])**2
        isInside = (d2 <= horizon2)
        return isInside

    cdef INDEX_t findIntersections(self, REAL_t[::1] x, REAL_t[:, ::1] simplex, INDEX_t start, INDEX_t end, REAL_t[::1] intersections):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t nn = 0., p = 0., q = 0., A, B, c
            INDEX_t k
            INDEX_t numIntersections = 0
            INDEX_t dim = getINDEX(self.params, fKDIM)
        for k in range(dim):
            A = simplex[end, k]-simplex[start, k]
            B = simplex[start, k]-x[k]
            nn += A**2
            p += A*B
            q += B**2
        nn = 1./nn
        p *= 2.*nn
        q = (q-horizon2)*nn
        A = -p*0.5
        B = sqrt(A**2-q)
        c = A-B
        if (c >= 0) and (c <= 1):
            intersections[numIntersections] = c
            numIntersections += 1
        c = A+B
        if (c >= 0) and (c <= 1):
            intersections[numIntersections] = c
            numIntersections += 1
        return numIntersections

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            REAL_t d2 = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            d2 += (x[i]-y[i])**2
        self.dist2 = d2
        if d2 <= horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t d2 = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(dim):
            d2 += (x[i]-y[i])**2
        self.dist2 = d2
        if d2 <= horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    def __repr__(self):
        return '|x-y|_2 <= {}'.format(self.horizonFun)

    def get_Surface(self, REAL_t[::1] node1):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t horizon = sqrt(horizon2)
        t = np.linspace(0, 2*pi, 101)
        return np.vstack((node1[0]+horizon*np.cos(t), node1[1]+horizon*np.sin(t))).T

    def __reduce__(self):
        return ball2_retriangulation, (self.horizonFun, )

    def getLongDescription(self):
        return '\\chi_{|x-y|_2<\\delta}'


cdef class ball2_barycenter(barycenterDomain):
    """l2 ball interaction domain"""
    def __init__(self, function horizonFun):
        super(ball2_barycenter, self).__init__(horizonFun, False, True)

    def getComplement(self):
        return ball2Complement(self.horizonFun)

    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t i, k, j
            INDEX_t noSimplex1 = simplex1.shape[0]
            INDEX_t noSimplex2 = simplex2.shape[0]
            REAL_t d2
            REAL_t dmin2 = inf
            REAL_t dmax2 = 0.
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            INDEX_t dim = getINDEX(self.params, fKDIM)
        for i in range(noSimplex1):
            for k in range(noSimplex2):
                d2 = 0.
                for j in range(dim):
                    d2 += (simplex1[i, j] - simplex2[k, j])**2
                dmin2 = min(dmin2, d2)
                dmax2 = max(dmax2, d2)
        if dmin2 >= horizon2:
            self.relPos = REMOTE
        elif dmax2 <= horizon2:
            self.relPos = INTERACT
        else:
            self.relPos = CUT
        return self.relPos

    cdef BOOL_t isInside(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            BOOL_t isInside
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            INDEX_t j
            REAL_t d2 = 0.
        for j in range(x.shape[0]):
            d2 += (x[j] - y[j])**2
        isInside = (d2 <= horizon2)
        return isInside

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            REAL_t d2 = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            d2 += (x[i]-y[i])**2
        self.dist2 = d2
        if d2 <= horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t d2 = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(dim):
            d2 += (x[i]-y[i])**2
        self.dist2 = d2
        if d2 <= horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    def __repr__(self):
        return '|x-y|_2 <= {}'.format(self.horizonFun)

    def get_Surface(self, REAL_t[::1] node1):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t horizon = sqrt(horizon2)
        t = np.linspace(0, 2*pi, 101)
        return np.vstack((node1[0]+horizon*np.cos(t), node1[1]+horizon*np.sin(t))).T

    def __reduce__(self):
        return ball2_barycenter, (self.horizonFun, )

    def getLongDescription(self):
        return '\\chi_{|x-y|_2<\\delta}'


cdef class ballInf_retriangulation(retriangulationDomain):
    """l-inf ball interaction domain"""

    def __init__(self, function horizonFun):
        super(ballInf_retriangulation, self).__init__(horizonFun, False, True)

    cdef void setParams(self, void *params):
        interactionDomain.setParams(self, params)
        dim = getINDEX(self.params, fKDIM)
        if dim == 2:
            self.specialOffsets = np.zeros((4, dim), dtype=REAL)

    cdef computeSpecialPoints(self):
        cdef:
            INDEX_t dim
            REAL_t horizon2, horizon
        dim = getINDEX(self.params, fKDIM)
        if dim == 2:
            horizon2 = getREAL(self.params, fHORIZON2)
            horizon = sqrt(horizon2)
            self.specialOffsets[0, 0] = horizon
            self.specialOffsets[0, 1] = horizon
            self.specialOffsets[1, 0] = -horizon
            self.specialOffsets[1, 1] = horizon
            self.specialOffsets[2, 0] = -horizon
            self.specialOffsets[2, 1] = -horizon
            self.specialOffsets[3, 0] = horizon
            self.specialOffsets[3, 1] = -horizon

    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t i, k, j
            INDEX_t noSimplex1 = simplex1.shape[0]
            INDEX_t noSimplex2 = simplex2.shape[0]
            REAL_t d2
            REAL_t dmin2 = inf
            REAL_t dmax2 = 0.
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            INDEX_t dim = getINDEX(self.params, fKDIM)
        for i in range(noSimplex1):
            for k in range(noSimplex2):
                d2 = 0.
                for j in range(dim):
                    d2 = max(d2, (simplex1[i, j] - simplex2[k, j])**2)
                dmin2 = min(dmin2, d2)
                dmax2 = max(dmax2, d2)
        if dmin2 >= horizon2:
            self.relPos = REMOTE
        elif dmax2 <= horizon2:
            self.relPos = INTERACT
        else:
            self.relPos = CUT
        return self.relPos

    cdef BOOL_t isInside(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t s = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            s = max(s, (x[i]-y[i])**2)
        return s <= horizon2

    cdef INDEX_t findIntersections(self, REAL_t[::1] x, REAL_t[:, ::1] simplex, INDEX_t start, INDEX_t end, REAL_t[::1] intersections):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t horizon = sqrt(horizon2)
            REAL_t lam
            INDEX_t k, j
            BOOL_t isInside
            INDEX_t numIntersections = 0
            REAL_t d
            INDEX_t dim = getINDEX(self.params, fKDIM)

        for k in range(dim):
            d = simplex[start, k]-simplex[end, k]
            if abs(d) > 1e-12:
                lam = (horizon-(simplex[end, k]-x[k]))/d
                if (0 <= lam) and (lam <= 1):
                    isInside = True
                    for j in range(2):
                        isInside &= (x[j]-(lam*simplex[start, j]+(1-lam)*simplex[end, j]))**2 <= horizon2+1e-12
                    if isInside:
                        intersections[numIntersections] = 1-lam
                        numIntersections += 1
                lam = (-horizon-(simplex[end, k]-x[k]))/d
                if (0 <= lam) and (lam <= 1):
                    isInside = True
                    for j in range(2):
                        isInside &= (x[j]-(lam*simplex[start, j]+(1-lam)*simplex[end, j]))**2 <= horizon2+1e-12
                    if isInside:
                        intersections[numIntersections] = 1-lam
                        numIntersections += 1
        if numIntersections == 2:
            if intersections[0] > intersections[1]:
                intersections[0], intersections[1] = intersections[1], intersections[0]
        return numIntersections

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            REAL_t d2 = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            d2 = max(d2, (x[i]-y[i])**2)
        self.dist2 = d2
        if d2 <= horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t d2 = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(dim):
            d2 = max(d2, (x[i]-y[i])**2)
        self.dist2 = d2
        if d2 <= horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    def __repr__(self):
        return '|x-y|_inf <= {}'.format(self.horizonFun)

    def get_Surface(self, REAL_t[::1] node1):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t horizon = sqrt(horizon2)
        return np.array([[node1[0]+horizon, node1[0]+horizon, node1[0]-horizon, node1[0]-horizon, node1[0]+horizon],
                         [node1[1]-horizon, node1[1]+horizon, node1[1]+horizon, node1[1]-horizon, node1[1]-horizon]]).T

    def __reduce__(self):
        return ballInf_retriangulation, (self.horizonFun, )

    def getLongDescription(self):
        return '\\chi_{|x-y|_\infty<\\delta}'


cdef class ballInf_barycenter(barycenterDomain):
    """l-inf ball interaction domain"""

    def __init__(self, function horizonFun):
        super(ballInf_barycenter, self).__init__(horizonFun, False, True)

    cdef void setParams(self, void *params):
        cdef:
            INDEX_t dim
            REAL_t horizon2, horizon
        interactionDomain.setParams(self, params)
        dim = getINDEX(self.params, fKDIM)
        if dim == 2:
            horizon2 = getREAL(self.params, fHORIZON2)
            horizon = sqrt(horizon2)
            self.specialOffsets = np.zeros((4, dim), dtype=REAL)
            self.specialOffsets[0, 0] = horizon
            self.specialOffsets[0, 1] = horizon
            self.specialOffsets[1, 0] = -horizon
            self.specialOffsets[1, 1] = horizon
            self.specialOffsets[2, 0] = -horizon
            self.specialOffsets[2, 1] = -horizon
            self.specialOffsets[3, 0] = horizon
            self.specialOffsets[3, 1] = -horizon

    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t i, k, j
            INDEX_t noSimplex1 = simplex1.shape[0]
            INDEX_t noSimplex2 = simplex2.shape[0]
            REAL_t d2
            REAL_t dmin2 = inf
            REAL_t dmax2 = 0.
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            INDEX_t dim = getINDEX(self.params, fKDIM)
        for i in range(noSimplex1):
            for k in range(noSimplex2):
                d2 = 0.
                for j in range(dim):
                    d2 = max(d2, (simplex1[i, j] - simplex2[k, j])**2)
                dmin2 = min(dmin2, d2)
                dmax2 = max(dmax2, d2)
        if dmin2 >= horizon2:
            self.relPos = REMOTE
        elif dmax2 <= horizon2:
            self.relPos = INTERACT
        else:
            self.relPos = CUT
        return self.relPos

    cdef BOOL_t isInside(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t s = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            s = max(s, (x[i]-y[i])**2)
        return s <= horizon2

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            REAL_t d2 = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            d2 = max(d2, (x[i]-y[i])**2)
        self.dist2 = d2
        if d2 <= horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t d2 = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(dim):
            d2 = max(d2, (x[i]-y[i])**2)
        self.dist2 = d2
        if d2 <= horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    def __repr__(self):
        return '|x-y|_inf <= {}'.format(self.horizonFun)

    def get_Surface(self, REAL_t[::1] node1):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t horizon = sqrt(horizon2)
        return np.array([[node1[0]+horizon, node1[0]+horizon, node1[0]-horizon, node1[0]-horizon, node1[0]+horizon],
                         [node1[1]-horizon, node1[1]+horizon, node1[1]+horizon, node1[1]-horizon, node1[1]-horizon]]).T

    def __reduce__(self):
        return ballInf_barycenter, (self.horizonFun, )

    def getLongDescription(self):
        return '\\chi_{|x-y|_\infty<\\delta}'


cdef class ball2Complement(retriangulationDomain):
    def __init__(self, function horizonFun):
        super(ball2Complement, self).__init__(horizonFun, True, True)

    cdef void setParams(self, void *params):
        cdef:
            INDEX_t dim, k
        interactionDomain.setParams(self, params)
        warnings.warn('cut elements are currently not implemented for \'ball2Complement\', expect quadrature errors')
        dim = getINDEX(self.params, fKDIM)
        self.A_Simplex = np.zeros((1, dim+1, dim+1), dtype=REAL)
        for k in range(dim+1):
            self.A_Simplex[0, k, k] = 1.
        self.b_Simplex = np.zeros((1, dim+1), dtype=REAL)
        self.vol_Simplex = np.ones((1), dtype=REAL)
        self.A_Node = np.zeros((1, dim+1, dim+1), dtype=REAL)
        for k in range(dim+1):
            self.A_Node[0, k, k] = 1.
        self.vol_Node = np.ones((1), dtype=REAL)

    def getComplement(self):
        return ball2_retriangulation(self.horizonFun)

    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t i, k, j
            INDEX_t noSimplex1 = simplex1.shape[0]
            INDEX_t noSimplex2 = simplex2.shape[0]
            REAL_t d2
            REAL_t dmin2 = inf
            REAL_t dmax2 = 0.
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            INDEX_t dim = getINDEX(self.params, fKDIM)
        for i in range(noSimplex1):
            for k in range(noSimplex2):
                d2 = 0.
                for j in range(dim):
                    d2 += (simplex1[i, j] - simplex2[k, j])**2
                dmin2 = min(dmin2, d2)
                dmax2 = max(dmax2, d2)
        if dmin2 >= horizon2:
            self.relPos = INTERACT
        elif dmax2 <= horizon2:
            self.relPos = REMOTE
        else:
            self.relPos = CUT
        return self.relPos

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            REAL_t s = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            s += (x[i]-y[i])**2
        self.dist2 = s
        if s > horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t s = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(dim):
            s += (x[i]-y[i])**2
        self.dist2 = s
        if s > horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    def __repr__(self):
        return '|x-y|_2 > {}'.format(self.horizonFun)

    def getLongDescription(self):
        return '\\chi_{|x-y|_2>\\delta}'


cdef class linearTransformInteraction(interactionDomain):
    def __init__(self, function horizonFun, interactionDomain baseInteraction, matrixFunction transform):
        self.callTransform = not isinstance(transform, constantMatrixFunction)
        super(linearTransformInteraction, self).__init__(horizonFun, False, not self.callTransform)
        self.baseInteraction = baseInteraction
        self.transform = transform

        dim = transform.rows
        self.vec = uninitialized((dim), dtype=REAL)
        self.vec2 = uninitialized((dim), dtype=REAL)

        self.A = uninitialized((dim, dim), dtype=REAL)

        if not self.callTransform:
            self.transform.eval(self.vec, self.A)
        else:
            self.A_Simplex[0, :, :] = 0.
            self.A_Simplex[0, 0, 0] = 1
            self.A_Simplex[0, 1, 1] = 1
            self.A_Simplex[0, 2, 2] = 1
            self.b_Simplex[0, :] = 0.
            self.vol_Simplex[0] = 1.

        self.simplex1 = uninitialized((dim+1, dim), dtype=REAL)
        self.simplex2 = uninitialized((dim+1, dim), dtype=REAL)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        if self.callTransform:
            self.transform.eval(x, self.A)
        self.transformVectorForward(&x[0], &y[0], &self.vec2[0])
        self.baseInteraction.eval(x, self.vec2, value)
        self.dist2 = self.baseInteraction.dist2

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        if self.callTransform:
            self.transform.evalPtr(dim, x, &self.A[0, 0])
        self.transformVectorForward(x, y, &self.vec2[0])
        self.baseInteraction.evalPtr(dim, x, &self.vec2[0], value)
        self.dist2 = self.baseInteraction.dist2

    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t vertexNo1, vertexNo2, numInside = 0, numOutside = 0
            BOOL_t doExit = False
        for vertexNo1 in range(simplex1.shape[0]):
            if self.callTransform:
                self.transform.eval(simplex1[vertexNo1, :], self.A)
            self.transformSimplexForwardPoint(simplex1[vertexNo1, :], simplex2, self.simplex2)
            for vertexNo2 in range(self.simplex2.shape[0]):
                if self.baseInteraction.isInside(simplex1[vertexNo1, :], self.simplex2[vertexNo2, :]):
                    numInside += 1
                else:
                    numOutside += 1
                if numInside > 0 and numOutside > 0:
                    doExit = True
                    break
            if doExit:
                break
        if numOutside == 0:
            return INTERACT
        elif numInside == 0:
            return REMOTE
        else:
            return CUT

    cdef BOOL_t isInside(self, REAL_t[::1] x, REAL_t[::1] y):
        if self.callTransform:
            self.transform.eval(x, self.A)
        self.transformVectorForward(&x[0], &y[0], &self.vec2[0])
        return self.baseInteraction.isInside(x, self.vec2)

    cdef void transformVectorForward(self, REAL_t* x, REAL_t* y, REAL_t* new_y):
        # Transforms:
        # y -> A*(y-x) + x
        cdef:
            INDEX_t i, j
        for i in range(self.A.shape[0]):
            new_y[i] = x[i]
            for j in range(self.A.shape[1]):
                new_y[i] += self.A[i, j]*(y[j]-x[j])

    cdef void transformSimplexForwardPoint(self, REAL_t[::1] x, REAL_t[:, ::1] simplex, REAL_t[:, ::1] new_simplex):
        cdef:
            INDEX_t i, j, k
        for i in range(simplex.shape[0]):
            for k in range(simplex.shape[1]):
                new_simplex[i, k] = x[k]
                for j in range(simplex.shape[1]):
                    new_simplex[i, k] += (simplex[i, j]-x[j])*self.A[k, j]

    cdef void transformSimplexForward(self, REAL_t[:, ::1] simplex, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t i, j, k
        for i in range(simplex.shape[0]):
            for k in range(simplex.shape[1]):
                simplex2[i, k] = 0.
                for j in range(simplex.shape[1]):
                    simplex2[i, k] += simplex[i, j]*self.A[k, j]

    cdef void startLoopSubSimplices_Node(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2):
        if self.callTransform:
            self.transform.eval(node1, self.A)
        self.transformSimplexForwardPoint(node1, simplex2, self.simplex2)
        self.baseInteraction.startLoopSubSimplices_Node(node1, self.simplex2)
        self.identityMapping = self.baseInteraction.identityMapping

    cdef BOOL_t nextSubSimplex_Node(self, REAL_t[:, ::1] A, REAL_t * vol):
        return self.baseInteraction.nextSubSimplex_Node(A, vol)

    cdef void startLoopSubSimplices_Simplex(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t noVertex1, noVertex2
            BOOL_t doExit = False
        if self.callTransform:
            self.iter_Simplex = 0
            self.iterEnd_Simplex = 0
            for noVertex1 in range(simplex1.shape[0]):
                self.transform.eval(simplex1[noVertex1, :], self.A)
                for noVertex2 in range(simplex2.shape[0]):
                    if self.isInside(simplex1[noVertex1, :], simplex2[noVertex2, :]):
                        self.iterEnd_Simplex = 1
                        doExit = True
                        break
                if doExit:
                    break
        else:
            self.transformSimplexForward(simplex1, self.simplex1)
            self.transformSimplexForward(simplex2, self.simplex2)
            self.baseInteraction.startLoopSubSimplices_Simplex(self.simplex1, self.simplex2)
            self.identityMapping = self.baseInteraction.identityMapping

    cdef BOOL_t nextSubSimplex_Simplex(self, REAL_t[:, ::1] A, REAL_t[::1] b, REAL_t *vol):
        if self.callTransform:
            return interactionDomain.nextSubSimplex_Simplex(self, A, b, vol)
        else:
            return self.baseInteraction.nextSubSimplex_Simplex(A, b, vol)

    def get_Surface(self, REAL_t[::1] node1):
        cdef:
            INDEX_t dim = getINDEX(self.params, fKDIM)
        if self.callTransform:
            self.transform.eval(node1, self.A)
        z = self.baseInteraction.get_Surface(np.zeros((dim), dtype=REAL))
        z = np.array(z)@np.linalg.inv(self.A).T
        z += np.array(node1)
        return z


cdef class ellipseTransform(matrixFunction):
    cdef:
        function a, b, theta

    def __init__(self, function a, function b, function theta):
        self.a = a
        self.b = b
        self.theta = theta
        self.rows = 2
        self.columns = 2

    cdef void eval(self, REAL_t[::1] x, REAL_t[:, ::1] vals):
        cdef:
            REAL_t inv_a = 1./self.a.eval(x)
            REAL_t inv_b = 1./self.b.eval(x)
            REAL_t theta = self.theta.eval(x)
            REAL_t c = cos(theta)
            REAL_t s = sin(theta)

        vals[0, 0] = c*inv_a
        vals[0, 1] = -s*inv_a
        vals[1, 0] = s*inv_b
        vals[1, 1] = c*inv_b

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* vals):
        cdef:
            REAL_t inv_a = 1./self.a.evalPtr(dim, x)
            REAL_t inv_b = 1./self.b.evalPtr(dim, x)
            REAL_t theta = self.theta.evalPtr(dim, x)
            REAL_t c = cos(theta)
            REAL_t s = sin(theta)

        vals[0] = c*inv_a
        vals[1] = -s*inv_a
        vals[2] = s*inv_b
        vals[3] = c*inv_b


cdef class ellipse_retriangulation(linearTransformInteraction):
    """Ellipse interaction domain"""
    def __init__(self, function horizonFun, function a, function b, function theta):
        assert ((isinstance(a, constant) and a.value == 1.) or
                (isinstance(b, constant) and b.value == 1.)), "One of the two axes must be equal to 1."
        self.a = a
        self.b = b
        self.theta = theta
        base = ball2_retriangulation(horizonFun)
        if isinstance(a, constant) and isinstance(b, constant) and isinstance(theta, constant):
            transformation = constantMatrixFunction(np.array([[np.cos(theta.value)/a.value, -np.sin(theta.value)/a.value],
                                                              [np.sin(theta.value)/b.value, np.cos(theta.value)/b.value]]))
        else:
            transformation = ellipseTransform(a, b, theta)
        super(ellipse_retriangulation, self).__init__(horizonFun, base, transformation)

    cdef void setParams(self, void *params):
        linearTransformInteraction.setParams(self, params)
        self.baseInteraction.setParams(params)

    def __repr__(self):
        return 'ellipse(x,y) <= {}'.format(self.horizonFun)

    def __reduce__(self):
        return ellipse_retriangulation, (self.horizonFun, self.a, self.b, self.theta)


cdef class ellipse_barycenter(linearTransformInteraction):
    """Ellipse interaction domain"""
    def __init__(self, function horizonFun, function a, function b, function theta):
        assert ((isinstance(a, constant) and a.value == 1.) or
                (isinstance(b, constant) and b.value == 1.)), "One of the two axes must be equal to 1."
        self.a = a
        self.b = b
        self.theta = theta
        base = ball2_barycenter(horizonFun)
        transformation = ellipseTransform(a, b, theta)
        super(ellipse_barycenter, self).__init__(horizonFun, base, transformation)

    cdef void setParams(self, void *params):
        linearTransformInteraction.setParams(self, params)
        self.baseInteraction.setParams(params)

    def __repr__(self):
        return 'ellipse(x,y) <= {}'.format(self.horizonFun)

    def __reduce__(self):
        return ellipse_barycenter, (self.horizonFun, self.a, self.b, self.theta)


cdef class ball1_retriangulation(linearTransformInteraction):
    "l1 ball interaction domain"
    def __init__(self, function horizonFun):
        base = ballInf_retriangulation(horizonFun)
        A = np.array([[1., -1.], [1., 1.]], dtype=REAL)
        super(ball1_retriangulation, self).__init__(horizonFun, base, constantMatrixFunction(A))

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            REAL_t d2 = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            d2 += abs(x[i]-y[i])
        d2 *= d2
        self.dist2 = d2
        if d2 <= horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t d2 = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(dim):
            d2 += abs(x[i]-y[i])
        d2 *= d2
        self.dist2 = d2
        if d2 <= horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    cdef void setParams(self, void *params):
        linearTransformInteraction.setParams(self, params)
        self.baseInteraction.setParams(params)

    def __repr__(self):
        return '|x-y|_1 <= {}'.format(self.horizonFun)

    def __reduce__(self):
        return ball1_retriangulation, (self.horizonFun, )

    def getLongDescription(self):
        return '\\chi_{|x-y|_1<\\delta}'


cdef class ball1_barycenter(linearTransformInteraction):
    "l1 ball interaction domain"
    def __init__(self, function horizonFun):
        base = ballInf_barycenter(horizonFun)
        A = np.array([[1., -1.], [1., 1.]], dtype=REAL)
        super(ball1_barycenter, self).__init__(horizonFun, base, constantMatrixFunction(A))

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            REAL_t d2 = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            d2 += abs(x[i]-y[i])
        d2 *= d2
        self.dist2 = d2
        if d2 <= horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            REAL_t d2 = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(dim):
            d2 += abs(x[i]-y[i])
        d2 *= d2
        self.dist2 = d2
        if d2 <= horizon2:
            value[0] = 1.
        else:
            value[0] = 0.

    cdef void setParams(self, void *params):
        linearTransformInteraction.setParams(self, params)
        self.baseInteraction.setParams(params)

    def __repr__(self):
        return '|x-y|_1 <= {}'.format(self.horizonFun)

    def __reduce__(self):
        return ball1_barycenter, (self.horizonFun, )

    def getLongDescription(self):
        return '\\chi_{|x-y|_1<\\delta}'


cdef class ball2_dilation_barycenter(barycenterDomain):
    def __init__(self, sqrtAffineFunction horizonFun):
        super(ball2_dilation_barycenter, self).__init__(horizonFun, False, False)
        self.c = horizonFun.c
        self.d = np.linalg.norm(horizonFun.w)
        self.w = np.array(horizonFun.w)/self.d
        self.wT = np.array([self.w[1], -self.w[0]], dtype=REAL)
        self.mat = np.array([[self.w[0], self.w[1]],
                             [self.wT[0], self.wT[1]]], dtype=REAL)
        self.tempSimplex1 = uninitialized((3, 2), dtype=REAL)
        self.tempSimplex2 = uninitialized((3, 2), dtype=REAL)

    cdef void setParams(self, void *params):
        cdef:
            INDEX_t dim
            REAL_t horizon2, horizon
        interactionDomain.setParams(self, params)
        dim = getINDEX(self.params, fKDIM)
        if dim == 2:
            horizon2 = getREAL(self.params, fHORIZON2)
            horizon = sqrt(horizon2)
            z = uninitialized((2), dtype=REAL)
            z[0] = self.w[1]
            z[1] = -self.w[0]
            z /= np.linalg.norm(z)
            self.specialOffsets = np.zeros((2, dim), dtype=REAL)
            self.specialOffsets[0, 0] = horizon*z[0]
            self.specialOffsets[0, 1] = horizon*z[1]
            self.specialOffsets[1, 0] = -horizon*z[0]
            self.specialOffsets[1, 1] = -horizon*z[1]

    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t i, k, j
            INDEX_t noSimplex1 = simplex1.shape[0]
            INDEX_t noSimplex2 = simplex2.shape[0]
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            INDEX_t dim = getINDEX(self.params, fKDIM)
            INDEX_t numInside = 0, numOutside = 0
            REAL_t[2] coeff
        for i in range(noSimplex1):
            for j in range(dim):
                self.tempSimplex1[i, j] = 0.
                self.tempSimplex2[i, j] = 0.
                for k in range(dim):
                    self.tempSimplex1[i, j] += self.mat[j, k]*simplex1[i, k]
                    self.tempSimplex2[i, j] += self.mat[j, k]*simplex2[i, k]

        for i in range(noSimplex1):
            for k in range(noSimplex2):
                for j in range(dim):
                    coeff[j] = self.tempSimplex2[k, j]-self.tempSimplex1[i, j]

                if coeff[0]**2+coeff[1]**2 <= horizon2 + min(2*coeff[0]*self.d, 0.):
                    numInside += 1
                else:
                    numOutside += 1
                if numInside > 0 and numOutside > 0:
                    self.relPos = CUT
                    return self.relPos
        if numInside == 0:
            self.relPos = REMOTE
        else:
            self.relPos = INTERACT
        return self.relPos

    cdef BOOL_t isInside(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            INDEX_t i, j
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t[2] coeff

        for i in range(x.shape[0]):
            coeff[i] = 0.
            for j in range(x.shape[0]):
                coeff[i] += self.mat[i, j]*(y[j]-x[j])
        return coeff[0]**2+coeff[1]**2 <= horizon2 + min(2*coeff[0]*self.d, 0.)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            INDEX_t i, j
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t[2] coeff

        for i in range(x.shape[0]):
            coeff[i] = 0.
            for j in range(x.shape[0]):
                coeff[i] += self.mat[i, j]*(y[j]-x[j])

        if coeff[0]**2+coeff[1]**2 <= horizon2 + min(2*coeff[0]*self.d, 0.):
            value[0] = 1.
        else:
            value[0] = 0.

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            INDEX_t i, j
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t[2] coeff

        for i in range(dim):
            coeff[i] = 0.
            for j in range(dim):
                coeff[i] += self.mat[i, j]*(y[j]-x[j])

        if coeff[0]**2+coeff[1]**2 <= horizon2 + min(2*coeff[0]*self.d, 0.):
            value[0] = 1.
        else:
            value[0] = 0.

    def __repr__(self):
        return '|x-y|_2 <= {}'.format(self.horizonFun)

    def get_Surface(self, REAL_t[::1] node1):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t horizon = sqrt(horizon2)
            INDEX_t n = 30, i, j
            REAL_t[::1] t

        surf = uninitialized((2*n+3, node1.shape[0]), dtype=REAL)

        t = np.linspace(-0.5*pi, 0.5*pi, n+2)
        for i in range(n+2):
            for j in range(2):
                surf[i, j] = node1[j] + horizon*(np.cos(t[i])*self.w[j] + np.sin(t[i])*self.wT[j])

        horizon2 = sqrt(horizon**2+self.d**2)
        t = np.linspace(np.arccos(-self.d/horizon2),
                        2*np.pi-np.arccos(-self.d/horizon2),
                        n+2)[1:]
        for i in range(n+1):
            for j in range(2):
                surf[n+2+i, j] = node1[j]+self.d*self.w[j] + horizon2*(np.cos(t[i])*self.w[j] + np.sin(t[i])*self.wT[j])
        return surf

    def __reduce__(self):
        return ballInf_retriangulation, (self.horizonFun, )


cdef class ball2_dilation_retriangulation(retriangulationDomain):
    def __init__(self, sqrtAffineFunction horizonFun):
        super(ball2_dilation_retriangulation, self).__init__(horizonFun, False, False)
        self.c = horizonFun.c
        self.d = np.linalg.norm(horizonFun.w)
        self.w = np.array(horizonFun.w)/self.d
        self.wT = np.array([self.w[1], -self.w[0]], dtype=REAL)
        self.mat = np.array([[self.w[0], self.w[1]],
                             [self.wT[0], self.wT[1]]], dtype=REAL)
        self.tempSimplex1 = uninitialized((3, 2), dtype=REAL)
        self.tempSimplex2 = uninitialized((3, 2), dtype=REAL)

        self.A_Simplex[0, :, :] = 0.
        self.A_Simplex[0, 0, 0] = 1
        self.A_Simplex[0, 1, 1] = 1
        self.A_Simplex[0, 2, 2] = 1
        self.b_Simplex[0, :] = 0.
        self.vol_Simplex[0] = 1.

    cdef void setParams(self, void *params):
        cdef:
            INDEX_t dim
        interactionDomain.setParams(self, params)
        dim = getINDEX(self.params, fKDIM)
        if dim == 2:
            self.specialOffsets = np.zeros((2, dim), dtype=REAL)

    cdef computeSpecialPoints(self):
        cdef:
            INDEX_t dim
            REAL_t horizon2, horizon
        dim = getINDEX(self.params, fKDIM)
        if dim == 2:
            horizon2 = getREAL(self.params, fHORIZON2)
            horizon = sqrt(horizon2)
            self.specialOffsets = np.zeros((2, dim), dtype=REAL)
            self.specialOffsets[0, 0] = horizon*self.wT[0]
            self.specialOffsets[0, 1] = horizon*self.wT[1]
            self.specialOffsets[1, 0] = -horizon*self.wT[0]
            self.specialOffsets[1, 1] = -horizon*self.wT[1]

    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t i, k, j
            INDEX_t noSimplex1 = simplex1.shape[0]
            INDEX_t noSimplex2 = simplex2.shape[0]
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            INDEX_t dim = getINDEX(self.params, fKDIM)
            INDEX_t numInside = 0, numOutside = 0
            REAL_t[2] coeff
        for i in range(noSimplex1):
            for j in range(dim):
                self.tempSimplex1[i, j] = 0.
                self.tempSimplex2[i, j] = 0.
                for k in range(dim):
                    self.tempSimplex1[i, j] += self.mat[j, k]*simplex1[i, k]
                    self.tempSimplex2[i, j] += self.mat[j, k]*simplex2[i, k]

        for i in range(noSimplex1):
            for k in range(noSimplex2):
                for j in range(dim):
                    coeff[j] = self.tempSimplex2[k, j]-self.tempSimplex1[i, j]

                if coeff[0]**2+coeff[1]**2 <= horizon2 + min(2*coeff[0]*self.d, 0.):
                    numInside += 1
                else:
                    numOutside += 1
                if numInside > 0 and numOutside > 0:
                    self.relPos = CUT
                    return self.relPos
        if numInside == 0:
            self.relPos = REMOTE
        else:
            self.relPos = INTERACT
        return self.relPos

    cdef BOOL_t isInside(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            INDEX_t i, j
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t[2] coeff

        for i in range(x.shape[0]):
            coeff[i] = 0.
            for j in range(x.shape[0]):
                coeff[i] += self.mat[i, j]*(y[j]-x[j])
        return coeff[0]**2+coeff[1]**2 <= horizon2 + min(2*coeff[0]*self.d, 0.)

    cdef INDEX_t findIntersections(self, REAL_t[::1] x, REAL_t[:, ::1] simplex, INDEX_t start, INDEX_t end, REAL_t[::1] intersections):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t nn = 0., p = 0., q = 0., A, B, c
            INDEX_t k
            REAL_t[2] y
            REAL_t[2] coeff
            INDEX_t numIntersections = 0
            INDEX_t dim = getINDEX(self.params, fKDIM)
            BOOL_t origBallActive, startInside, endInside

        coeff[0] = 0.
        coeff[1] = 0.
        for k in range(dim):
            coeff[0] += self.mat[0, k]*(simplex[start, k]-x[k])
            coeff[1] += self.mat[0, k]*(simplex[end, k]-x[k])

        origBallActive = True
        if coeff[0] >= 0 and coeff[1] >= 0:
            origBallActive = True
        elif coeff[0] <= 0 and coeff[1] <= 0:
            origBallActive = False
        else:
            startInside = self.isInside(x, simplex[start, :])
            endInside = self.isInside(x, simplex[end, :])
            if startInside and not endInside:
                if coeff[1] >= 0:
                    origBallActive = True
                else:
                    origBallActive = False
            elif not startInside and endInside:
                if coeff[0] >= 0:
                    origBallActive = True
                else:
                    origBallActive = False
            else:
                return 0

        if origBallActive:
            for k in range(dim):
                y[k] = x[k]
        else:
            for k in range(dim):
                y[k] = x[k]+self.d*self.w[k]
            horizon2 += self.d**2

        for k in range(dim):
            A = simplex[end, k]-simplex[start, k]
            B = simplex[start, k]-y[k]
            nn += A**2
            p += A*B
            q += B**2
        nn = 1./nn
        p *= 2.*nn
        q = (q-horizon2)*nn
        A = -p*0.5
        B = sqrt(A**2-q)
        c = A-B
        if (c >= 0) and (c <= 1):
            intersections[numIntersections] = c
            numIntersections += 1
        c = A+B
        if (c >= 0) and (c <= 1):
            intersections[numIntersections] = c
            numIntersections += 1
        return numIntersections

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] value):
        cdef:
            INDEX_t i, j
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t[2] coeff

        for i in range(x.shape[0]):
            coeff[i] = 0.
            for j in range(x.shape[0]):
                coeff[i] += self.mat[i, j]*(y[j]-x[j])

        if coeff[0]**2+coeff[1]**2 <= horizon2 + min(2*coeff[0]*self.d, 0.):
            value[0] = 1.
        else:
            value[0] = 0.

    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, REAL_t* value):
        cdef:
            INDEX_t i, j
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t[2] coeff

        for i in range(dim):
            coeff[i] = 0.
            for j in range(dim):
                coeff[i] += self.mat[i, j]*(y[j]-x[j])

        if coeff[0]**2+coeff[1]**2 <= horizon2 + min(2*coeff[0]*self.d, 0.):
            value[0] = 1.
        else:
            value[0] = 0.

    def __repr__(self):
        return '|x-y|_2 <= {}'.format(self.horizonFun)

    def get_Surface(self, REAL_t[::1] node1):
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t horizon = sqrt(horizon2)
            INDEX_t n = 30, i, j
            REAL_t[::1] t

        surf = uninitialized((2*n+3, node1.shape[0]), dtype=REAL)

        t = np.linspace(-0.5*pi, 0.5*pi, n+2)
        for i in range(n+2):
            for j in range(2):
                surf[i, j] = node1[j] + horizon*(np.cos(t[i])*self.w[j] + np.sin(t[i])*self.wT[j])

        horizon2 = sqrt(horizon**2+self.d**2)
        t = np.linspace(np.arccos(-self.d/horizon2),
                        2*np.pi-np.arccos(-self.d/horizon2),
                        n+2)[1:]
        for i in range(n+1):
            for j in range(2):
                surf[n+2+i, j] = node1[j]+self.d*self.w[j] + horizon2*(np.cos(t[i])*self.w[j] + np.sin(t[i])*self.wT[j])
        return surf

    def __reduce__(self):
        return ballInf_retriangulation, (self.horizonFun, )

    cdef void startLoopSubSimplices_Simplex(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t noVertex1, noVertex2
            BOOL_t doExit = False
        self.iter_Simplex = 0
        self.iterEnd_Simplex = 0
        for noVertex1 in range(simplex1.shape[0]):
            for noVertex2 in range(simplex2.shape[0]):
                if self.isInside(simplex1[noVertex1, :], simplex2[noVertex2, :]):
                    self.iterEnd_Simplex = 1
                    doExit = True
                    break
            if doExit:
                break
