###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################



cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, M_PI as pi, pow
import warnings
from PyNucleus_base.myTypes import REAL

include "kernel_params.pxi"


cdef REAL_t inf = np.inf


cdef class interactionDomain(parametrizedTwoPointFunction):
    def __init__(self, BOOL_t isComplement):
        super(interactionDomain, self).__init__(True)
        self.complement = isComplement

    def getComplement(self):
        raise NotImplementedError()

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        raise NotImplementedError()

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void startLoopSubSimplices_Simplex(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        raise NotImplementedError()

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef BOOL_t nextSubSimplex_Simplex(self, REAL_t[:, ::1] A, REAL_t[::1] b, REAL_t * vol):
        raise NotImplementedError()

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void startLoopSubSimplices_Node(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2):
        raise NotImplementedError()

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef BOOL_t nextSubSimplex_Node(self, REAL_t[:, ::1] A, REAL_t[::1] b, REAL_t * vol):
        raise NotImplementedError()

    def startLoopSubSimplices_Simplex_py(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        self.startLoopSubSimplices_Simplex(simplex1, simplex2)

    def nextSubSimplex_Simplex_py(self, REAL_t[:, ::1] A, REAL_t[::1] b, REAL_t[::1] vol):
        return self.nextSubSimplex_Simplex(A, b, &vol[0])

    def startLoopSubSimplices_Node_py(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2):
        self.startLoopSubSimplices_Node(node1, simplex2)

    def nextSubSimplex_Node_py(self, REAL_t[:, ::1] A, REAL_t[::1] b, REAL_t[::1] vol):
        return self.nextSubSimplex_Node(A, b, &vol[0])

    def plot_Simplex(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        import matplotlib.pyplot as plt
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t horizon = sqrt(horizon2)
            INDEX_t dim = getINDEX(self.params, fKDIM)
            REAL_t vol

        if dim == 2:
            def plotSimplex(simplex, doFill=False):
                if doFill:
                    plt.fill(np.concatenate((simplex[:, 0], [simplex[0, 0]])),
                             np.concatenate((simplex[:, 1], [simplex[0, 1]])))
                else:
                    plt.plot(np.concatenate((simplex[:, 0], [simplex[0, 0]])),
                             np.concatenate((simplex[:, 1], [simplex[0, 1]])))

            A = np.empty((dim+1, dim+1), dtype=REAL)
            b = np.empty((dim+1, 1), dtype=REAL)
            plotSimplex(simplex1)
            plotSimplex(simplex2)
            t = np.linspace(0, 2*pi, 101)
            for j in range(dim+1):
                plt.plot(simplex2[j, 0]+horizon*np.cos(t), simplex2[j, 1]+horizon*np.sin(t))
            self.startLoopSubSimplices_Simplex(simplex1, simplex2)
            while self.nextSubSimplex_Simplex(A, b[:, 0], &vol):
                plotSimplex(A.T.dot(simplex1)+b.T.dot(simplex1), True)
            c = np.vstack((simplex1, simplex2))
            mins = c.min(axis=0)
            maxs = c.max(axis=0)
            plt.axis('equal')
            plt.xlim([mins[0], maxs[0]])
            plt.ylim([mins[1], maxs[1]])

    def plot_Node(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2):
        import matplotlib.pyplot as plt
        cdef:
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t horizon = sqrt(horizon2)
            INDEX_t dim = getINDEX(self.params, fKDIM)
            REAL_t vol

        if dim == 2:
            def plotSimplex(simplex, doFill=False):
                if doFill:
                    plt.fill(np.concatenate((simplex[:, 0], [simplex[0, 0]])),
                             np.concatenate((simplex[:, 1], [simplex[0, 1]])))
                else:
                    plt.plot(np.concatenate((simplex[:, 0], [simplex[0, 0]])),
                             np.concatenate((simplex[:, 1], [simplex[0, 1]])))

            A = np.empty((dim+1, dim+1), dtype=REAL)
            b = np.empty((dim+1, 1), dtype=REAL)
            # plotSimplex(simplex1)
            plotSimplex(simplex2)
            plt.scatter(node1[0], node1[1])
            t = np.linspace(0, 2*pi, 101)
            plt.plot(node1[0]+horizon*np.cos(t), node1[1]+horizon*np.sin(t))
            self.startLoopSubSimplices_Node(node1, simplex2)
            while self.nextSubSimplex_Node(A, b[:, 0], &vol):
                plotSimplex(A.T.dot(simplex2)+b.T.dot(simplex2), True)
            c = np.vstack((node1, simplex2))
            mins = c.min(axis=0)
            maxs = c.max(axis=0)
            plt.axis('equal')
            plt.xlim([mins[0], maxs[0]])
            plt.ylim([mins[1], maxs[1]])


cdef class fullSpace(interactionDomain):
    def __init__(self):
        super(fullSpace, self).__init__(False)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
        return INTERACT

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return 1.

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        return 1.

    def __repr__(self):
        dim = getINDEX(self.params, fKDIM)
        return 'R^{}'.format(dim)


cdef class ball1(interactionDomain):
    def __init__(self):
        super(ball1, self).__init__(False)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
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
                    d2 += abs(simplex1[i, j] - simplex2[k, j])
                d2 *= d2**2
                dmin2 = min(dmin2, d2)
                dmax2 = max(dmax2, d2)
        if dmin2 >= horizon2:
            self.relPos = REMOTE
        elif dmax2 <= horizon2:
            self.relPos = INTERACT
        else:
            self.relPos = CUT
        return self.relPos

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t s = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            s += abs(x[i]-y[i])
        s = s*s
        if s <= horizon2:
            return 1.
        else:
            return 0.

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t s = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(dim):
            s += abs(x[i]-y[i])
        s = s*s
        if s <= horizon2:
            return 1.
        else:
            return 0.

    def __repr__(self):
        horizon2 = getREAL(self.params, fHORIZON2)
        return '|x-y|_1 <= {}'.format(sqrt(horizon2))


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


cdef class ball2(interactionDomain):
    def __init__(self):
        super(ball2, self).__init__(False)
        self.intervals1 = np.empty((4), dtype=REAL)
        self.intervals2 = np.empty((3), dtype=REAL)
        self.A_Simplex = np.empty((2, 3, 3), dtype=REAL)
        self.b_Simplex = np.empty((2, 3), dtype=REAL)
        self.vol_Simplex = np.empty((2), dtype=REAL)
        self.A_Node = np.empty((2, 3, 3), dtype=REAL)
        self.b_Node = np.empty((2, 3), dtype=REAL)
        self.vol_Node = np.empty((2), dtype=REAL)

    def getComplement(self):
        return ball2Complement()

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
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

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void startLoopSubSimplices_Simplex(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2):
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
        if dim == 1:
            lr = simplex1[0, 0] < simplex2[0, 0]
            vol1 = abs(simplex1[0, 0]-simplex1[1, 0])
            invVol1 = 1./vol1
            if lr:
                self.intervals1[0] = simplex1[0,0]*invVol1
                self.intervals1[1] = max(simplex1[0,0], simplex2[0,0]-horizon)*invVol1
                self.intervals1[2] = min(simplex1[1,0], simplex2[1,0]-horizon)*invVol1
                self.intervals1[3] = simplex1[1,0]*invVol1
                self.iter_Simplex = 1
                self.iterEnd_Simplex = 3
            else:
                self.intervals1[0] = simplex1[0,0]*invVol1
                self.intervals1[1] = max(simplex1[0,0], simplex2[0,0]+horizon)*invVol1
                self.intervals1[2] = min(simplex1[1,0], simplex2[1,0]+horizon)*invVol1
                self.intervals1[3] = simplex1[1,0]*invVol1
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

            numInside = 0
            for i in range(3):
                isInside = False
                for k in range(3):
                    d2 = 0.
                    for j in range(2):
                        d2 += (simplex1[i, j] - simplex2[k, j])**2
                    insideIJ[i][k] = d2 <= horizon2
                    isInside |= insideIJ[i][k]
                insideI[i] = isInside
                numInside += isInside
            if numInside == 0:
                raise NotImplementedError()
            elif numInside == 1:
                inside = 0
                while not insideI[inside]:
                    inside += 1
                outside1 = (inside+1)%3
                outside2 = (inside+2)%3
                c1 = 0
                c2 = 0
                for j in range(3):
                    if insideIJ[inside][j]:
                        c1 = max(c1, findIntersection(simplex2[j, :], simplex1[inside, :], simplex1[outside1, :], horizon2))
                        c2 = max(c2, findIntersection(simplex2[j, :], simplex1[inside, :], simplex1[outside2, :], horizon2))
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
                inside1 = (outside+1)%3
                inside2 = (outside+2)%3
                c1 = 1
                c2 = 1
                for j in range(3):
                    if insideIJ[inside1][j]:
                        c1 = min(c1, findIntersection(simplex2[j, :], simplex1[outside, :], simplex1[inside1, :], horizon2))
                    if insideIJ[inside2][j]:
                        c2 = min(c2, findIntersection(simplex2[j, :], simplex1[outside, :], simplex1[inside2, :], horizon2))
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

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef BOOL_t nextSubSimplex_Simplex(self, REAL_t[:, ::1] A, REAL_t[::1] b, REAL_t *vol):
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

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void startLoopSubSimplices_Node(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2):
        cdef:
            INDEX_t dim = getINDEX(self.params, fKDIM)
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
            REAL_t horizon = sqrt(horizon2)
            REAL_t vol2, invVol2
            BOOL_t lr
            INDEX_t numInside
            BOOL_t ind[3]
            INDEX_t j
            REAL_t d1, d2
            INDEX_t inside, outside1, outside2
            INDEX_t outside, inside1, inside2
            REAL_t c1, c2
        if dim == 1:
            lr = node1[0] < simplex2[0, 0]
            vol2 = abs(simplex2[0, 0]-simplex2[1, 0])
            invVol2 = 1./vol2
            if lr:
                self.intervals2[0] = simplex2[0, 0] * invVol2
                self.intervals2[1] = min(simplex2[1, 0], node1[0]+horizon) * invVol2
                self.intervals2[2] = simplex2[1, 0] * invVol2
                self.iter_Node = 0
                self.iterEnd_Node = 1
            else:
                self.intervals2[0] = simplex2[0, 0] * invVol2
                self.intervals2[1] = max(simplex2[0, 0], node1[0]-horizon) * invVol2
                self.intervals2[2] = simplex2[1, 0] * invVol2
                self.iter_Node = 1
                self.iterEnd_Node = 2
        elif dim == 2:
            numInside = 0
            for j in range(3):
                d2 = 0.
                for k in range(2):
                    d2 += (simplex2[j, k]-node1[k])**2
                ind[j] = (d2 <= horizon2)
                numInside += ind[j]
            if numInside == 0:
                self.iter_Node = 0
                self.iterEnd_Node = 0
            elif numInside == 1:
                inside = 0
                while not ind[inside]:
                    inside += 1
                outside1 = (inside+1)%3
                outside2 = (inside+2)%3
                c1 = findIntersection(node1, simplex2[inside, :], simplex2[outside1, :], horizon2)
                c2 = findIntersection(node1, simplex2[inside, :], simplex2[outside2, :], horizon2)

                self.iter_Node = 0
                if c1+c2 > 0:
                    self.A_Node[0, :, :] = 0.
                    self.b_Node[0, :] = 0.
                    self.A_Node[0, inside, inside] = c1+c2
                    self.A_Node[0, inside, outside1] = c2
                    self.A_Node[0, inside, outside2] = c1
                    self.A_Node[0, outside1, outside1] = c1
                    self.A_Node[0, outside2, outside2] = c2
                    self.b_Node[0, inside] = 1-c1-c2
                    self.vol_Node[0] = c1*c2

                    self.iterEnd_Node = 1
                else:
                    self.iterEnd_Node = 0

            elif numInside == 2:
                outside = 0
                while ind[outside]:
                    outside += 1
                inside1 = (outside+1)%3
                inside2 = (outside+2)%3
                c1 = findIntersection(node1, simplex2[outside, :], simplex2[inside1, :], horizon2)
                c2 = findIntersection(node1, simplex2[outside, :], simplex2[inside2, :], horizon2)
                d1 = 0.
                d2 = 0.
                for k in range(2):
                    d1 += (simplex2[outside, k]
                           + c1*(simplex2[inside1, k]-simplex2[outside, k])
                           - simplex2[inside2, k])**2
                    d2 += (simplex2[outside, k]
                           + c2*(simplex2[inside2, k]-simplex2[outside, k])
                           - simplex2[inside1, k])
                self.A_Node[:, :, :] = 0.
                self.b_Node[:, :] = 0.

                self.iter_Node = 0
                self.iterEnd_Node = 0
                if d1 < d2:
                    if 1-c1 > 0:
                        self.A_Node[self.iterEnd_Node, outside, outside] = 1-c1
                        self.A_Node[self.iterEnd_Node, inside1, inside1] = 1-c1
                        self.A_Node[self.iterEnd_Node, inside1, inside2] = -c1
                        self.A_Node[self.iterEnd_Node, inside2, inside2] = 1.
                        self.b_Node[self.iterEnd_Node, inside1] = c1
                        self.vol_Node[self.iterEnd_Node] = 1-c1
                        self.iterEnd_Node += 1

                    if c1*(1-c2) > 0.:
                        self.A_Node[self.iterEnd_Node, outside, outside] = 1-c2
                        self.A_Node[self.iterEnd_Node, inside2, inside2] = 1
                        self.A_Node[self.iterEnd_Node, inside2, outside] = c2
                        self.A_Node[self.iterEnd_Node, outside, inside1] = 1-c1
                        self.A_Node[self.iterEnd_Node, inside1, inside1] = c1
                        self.vol_Node[self.iterEnd_Node] = c1*(1-c2)
                        self.iterEnd_Node += 1
                else:
                    if 1-c2 > 0:
                        self.A_Node[self.iterEnd_Node, outside, outside] = 1-c2
                        self.A_Node[self.iterEnd_Node, inside2, inside2] = 1-c2
                        self.A_Node[self.iterEnd_Node, inside2, inside1] = -c2
                        self.A_Node[self.iterEnd_Node, inside1, inside1] = 1.
                        self.b_Node[self.iterEnd_Node, inside2] = c2
                        self.vol_Node[self.iterEnd_Node] = 1-c2
                        self.iterEnd_Node += 1

                    if c2*(1-c1) > 0.:
                        self.A_Node[self.iterEnd_Node, outside, outside] = 1-c1
                        self.A_Node[self.iterEnd_Node, inside1, inside1] = 1
                        self.A_Node[self.iterEnd_Node, inside1, outside] = c1
                        self.A_Node[self.iterEnd_Node, outside, inside2] = 1-c2
                        self.A_Node[self.iterEnd_Node, inside2, inside2] = c2
                        self.vol_Node[self.iterEnd_Node] = c2*(1-c1)
                        self.iterEnd_Node += 1
            else:
                self.A_Node[0, :, :] = 0.
                self.b_Node[0, :] = 0.
                self.A_Node[0, 0, 0] = 1.
                self.A_Node[0, 1, 1] = 1.
                self.A_Node[0, 2, 2] = 1.
                self.vol_Node[0] = 1.
                self.iter_Node = 0
                self.iterEnd_Node = 1

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef BOOL_t nextSubSimplex_Node(self, REAL_t[:, ::1] A, REAL_t[::1] b, REAL_t *vol):
        cdef:
            INDEX_t dim, i, j
            REAL_t l, r, v0, v1
        if self.iter_Node == self.iterEnd_Node:
            return False
        dim = getINDEX(self.params, fKDIM)
        if dim == 1:
            l = self.intervals2[self.iter_Node]
            r = self.intervals2[self.iter_Node+1]
            if r-l <= 0:
                self.iter_Node += 1
                return self.nextSubSimplex_Node(A, b, vol)
            v0 = self.intervals2[0]
            v1 = self.intervals2[2]
            A[0, 0] = r-l
            A[0, 1] = 0.
            A[1, 0] = 0.
            A[1, 1] = r-l
            b[0] = v1-r
            b[1] = l-v0
            vol[0] = r-l
            self.iter_Node += 1
            return True
        elif dim == 2:
            for i in range(3):
                b[i] = self.b_Node[self.iter_Node, i]
                for j in range(3):
                    A[i, j] = self.A_Node[self.iter_Node, i, j]
            vol[0] = self.vol_Node[self.iter_Node]
            self.iter_Node += 1
            return True

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t s = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            s += (x[i]-y[i])**2
        if s <= horizon2:
            return 1.
        else:
            return 0.

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t s = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(dim):
            s += (x[i]-y[i])**2
        if s <= horizon2:
            return 1.
        else:
            return 0.

    def __repr__(self):
        horizon2 = getREAL(self.params, fHORIZON2)
        return '|x-y|_2 <= {}'.format(sqrt(horizon2))

    def __setstate__(self, state):
        ball2.__init__(self)


cdef class ballInf(interactionDomain):
    def __init__(self):
        super(ballInf, self).__init__(False)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
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

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t s = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            s = max(s, (x[i]-y[i])**2)
        if s <= horizon2:
            return 1.
        else:
            return 0.

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t s = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(dim):
            s = max(s, (x[i]-y[i])**2)
        if s <= horizon2:
            return 1.
        else:
            return 0.

    def __repr__(self):
        horizon2 = getREAL(self.params, fHORIZON2)
        return '|x-y|_inf <= {}'.format(sqrt(horizon2))


cdef class ball2Complement(interactionDomain):
    def __init__(self):
        super(ball2Complement, self).__init__(True)

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
        self.b_Node = np.zeros((1, dim+1), dtype=REAL)
        self.vol_Node = np.ones((1), dtype=REAL)

    def getComplement(self):
        return ball2()

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
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

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t s = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(x.shape[0]):
            s += (x[i]-y[i])**2
        if s > horizon2:
            return 1.
        else:
            return 0.

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t s = 0.
            INDEX_t i
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        for i in range(dim):
            s += (x[i]-y[i])**2
        if s > horizon2:
            return 1.
        else:
            return 0.

    def __repr__(self):
        horizon2 = getREAL(self.params, fHORIZON2)
        return '|x-y|_2 > {}'.format(sqrt(horizon2))


cdef class ellipse(interactionDomain):
    cdef:
        public REAL_t aFac2
        public REAL_t bFac2

    def __init__(self, REAL_t aFac, REAL_t bFac):
        super(ellipse, self).__init__()
        assert 0 < aFac <= 1.
        assert 0 < bFac <= 1.
        self.aFac2 = aFac**2
        self.bFac2 = bFac**2

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t s
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        s = (x[0]-y[0])**2/(self.aFac2*horizon2) + (x[1]-y[1])**2/(self.bFac2*horizon2)
        if s <= 1.:
            return 1.
        else:
            return 0.

    @cython.cdivision(True)
    cdef REAL_t evalPtr(ellipse self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t s
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)
        s = (x[0]-y[0])**2/(self.aFac2*horizon2) + (x[1]-y[1])**2/(self.bFac2*horizon2)
        if s <= 1.:
            return 1.
        else:
            return 0.

    def __getstate__(self):
        return (sqrt(self.aFac2), sqrt(self.bFac2))

    def __setstate__(self, state):
        ellipse.__init__(self, state[0], state[1])
