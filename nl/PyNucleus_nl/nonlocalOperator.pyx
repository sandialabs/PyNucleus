###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
cimport numpy as np
from libc.math cimport (sin, cos, sinh, cosh, tanh, sqrt, atan2,
                        log, ceil,
                        fabs as abs, M_PI as pi, pow,
                        tgamma as gamma)
from scipy.special import factorial
from PyNucleus_base.myTypes import INDEX, REAL, COMPLEX, ENCODE, BOOL
from PyNucleus_base import uninitialized
from PyNucleus_base.blas cimport mydot
from libc.stdlib cimport malloc, free
from . interactionDomains cimport CUT

cdef:
    MASK_t ALL
ALL.set()

include "panelTypes.pxi"

cdef INDEX_t MAX_INT = np.iinfo(INDEX).max
cdef REAL_t inf = np.inf


cdef inline void getSimplexAndCenter(const INDEX_t[:, ::1] cells,
                                     const REAL_t[:, ::1] vertices,
                                     const INDEX_t cellIdx,
                                     REAL_t[:, ::1] simplex,
                                     REAL_t[::1] center):
    cdef:
        INDEX_t dim = vertices.shape[1]
        INDEX_t manifold_dim = cells.shape[1]-1
        INDEX_t m, k, l
        REAL_t v, fac = 1./(manifold_dim+1)
    center[:] = 0.
    for m in range(manifold_dim+1):
        k = cells[cellIdx, m]
        for l in range(dim):
            v = vertices[k, l]
            simplex[m, l] = v
            center[l] += v
    for l in range(dim):
        center[l] *= fac


cdef class PermutationIndexer:
    def __init__(self, N):
        cdef:
            INDEX_t i
        self.N = N

        self.onesCountLookup = np.zeros((1 << N) - 1, dtype=INDEX)
        for i in range((1 << N) - 1):
            self.onesCountLookup[i] = i.bit_count()

        self.factorials = np.zeros((N), dtype=INDEX)
        for i in range(N):
            self.factorials[i] = factorial(N-1-i, True)
        self.lehmer = np.zeros(N, dtype=INDEX)

    cdef INDEX_t rank(self, INDEX_t[::1] perm):
        cdef:
            INDEX_t seen, i, numOnes, index

        seen = (1 << (self.N-1-perm[0]))
        self.lehmer[0] = perm[0]

        for i in range(1, self.N):
            seen |= (1 << (self.N-1-perm[i]))
            numOnes = self.onesCountLookup[seen >> (self.N-perm[i])]
            self.lehmer[i] = perm[i]-numOnes

        index = 0
        for i in range(self.N):
            index += self.lehmer[i]*self.factorials[i]
        return index

    def rank_py(self, INDEX_t[::1] perm):
        return self.rank(perm)



include "nonlocalOperator_REAL.pxi"
include "nonlocalOperator_COMPLEX.pxi"


cdef class specialQuadRule:
    def __init__(self,
                 quadratureRule qr,
                 REAL_t[:, ::1] PSI=None,
                 REAL_t[:, :, ::1] PSI3=None,
                 REAL_t[:, ::1] PHI=None,
                 REAL_t[:, :, ::1] PHI3=None):
        self.qr = qr
        if PSI is not None:
            self.PSI = PSI
        if PSI3 is not None:
            self.PSI3 = PSI3
        if PHI is not None:
            self.PHI = PHI
        if PHI3 is not None:
            self.PHI3 = PHI3


cdef panelType MAX_PANEL = 120


cdef class nonlocalLaplacian1D(nonlocalOperator):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap DoFMap, num_dofs=None, manifold_dim2=-1, **kwargs):
        super(nonlocalLaplacian1D, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2, **kwargs)

    cdef REAL_t get_h_simplex(self, const REAL_t[:, ::1] simplex):
        cdef:
            REAL_t h2
        h2 = abs(simplex[1, 0]-simplex[0, 0])
        return h2

    cdef REAL_t get_h_surface_simplex(self, const REAL_t[:, ::1] simplex):
        return 1.

    cdef INDEX_t getCellPairIdentifierSize(self):
        return 3

    cdef void computeCellPairIdentifierBase(self, INDEX_t[::1] ID, INDEX_t * perm):
        # use h1, h2 and midpoint distance as identifier
        cdef:
            REAL_t h1, h2, d

        h1 = self.simplex1[1, 0]-self.simplex1[0, 0]
        h2 = self.simplex2[1, 0]-self.simplex2[0, 0]
        d = self.center2[0]-self.center1[0]

        perm[0] = (d < 0)
        perm[0] += (h1 < 0) << 1
        perm[0] += (h2 < 0) << 3

        h1 = abs(h1)
        h2 = abs(h2)
        d = abs(d)

        ID[0] = <INDEX_t>(MAX_INT*d*self.dMaxInv)
        ID[1] = <INDEX_t>(MAX_INT*h1*self.h1MaxInv)
        ID[2] = <INDEX_t>(MAX_INT*h2*self.h2MaxInv)


cdef class nonlocalLaplacian2D(nonlocalOperator):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap DoFMap, num_dofs=None, manifold_dim2=-1, **kwargs):
        super(nonlocalLaplacian2D, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2, **kwargs)

    cdef REAL_t get_h_simplex(self, const REAL_t[:, ::1] simplex):
        cdef:
            INDEX_t i, j
            REAL_t hmax = 0., h2
        for i in range(2):
            for j in range(i+1, 3):
                h2 = (simplex[j, 0]-simplex[i, 0])*(simplex[j, 0]-simplex[i, 0]) + (simplex[j, 1]-simplex[i, 1])*(simplex[j, 1]-simplex[i, 1])
                hmax = max(hmax, h2)
        return sqrt(hmax)

    cdef REAL_t get_h_surface_simplex(self, const REAL_t[:, ::1] simplex):
        cdef:
            INDEX_t k
            REAL_t h2
        h2 = 0.
        for k in range(2):
            h2 += (simplex[1, k]-simplex[0, k])**2
        return sqrt(h2)

    cdef INDEX_t getCellPairIdentifierSize(self):
        return 9

    cdef void computeCellPairIdentifierBase(self, INDEX_t[::1] ID, INDEX_t * perm):
        cdef:
            REAL_t d, d1, d2
            REAL_t v00, v01, v10, v11, v20, v21
            REAL_t c00, c01, c10, c11, c20, c21
            INDEX_t rot
            BOOL_t permCells

        d1 = self.center2[0]-self.center1[0]
        d2 = self.center2[1]-self.center1[1]

        d = sqrt(d1*d1 + d2*d2)
        ID[0] = <INDEX_t>(MAX_INT*d*self.dMaxInv)

        if d < 1e-9:
            d1 = 1.
            d2 = 0.
            permCells = False
            perm[0] = False
        else:
            permCells = (d1 < 0)
            perm[0] = permCells
            d = 1.0/d
            d1 *= d
            d2 *= d
            if permCells:
                d1 = -d1
                d2 = -d2

        # 1st simplex

        if not permCells:
            v00 = self.simplex1[0, 0]-self.center1[0]
            v01 = self.simplex1[0, 1]-self.center1[1]
            v10 = self.simplex1[1, 0]-self.center1[0]
            v11 = self.simplex1[1, 1]-self.center1[1]
            v20 = self.simplex1[2, 0]-self.center1[0]
            v21 = self.simplex1[2, 1]-self.center1[1]
        else:
            v00 = self.simplex2[0, 0]-self.center2[0]
            v01 = self.simplex2[0, 1]-self.center2[1]
            v10 = self.simplex2[1, 0]-self.center2[0]
            v11 = self.simplex2[1, 1]-self.center2[1]
            v20 = self.simplex2[2, 0]-self.center2[0]
            v21 = self.simplex2[2, 1]-self.center2[1]

        c00 = v00*d1 + v01*d2
        c10 = v10*d1 + v11*d2
        c20 = v20*d1 + v21*d2

        d1, d2 = -d2, d1

        c01 = v00*d1 + v01*d2
        c11 = v10*d1 + v11*d2
        c21 = v20*d1 + v21*d2

        d1, d2 = d2, -d1

        if c00 != c10:
            if c00 > c10:
                if c00 != c20:
                    if c00 > c20:
                        rot = 0
                    else:
                        rot = 2
                elif c01 > c21:
                    rot = 0
                else:
                    rot = 2
            elif c10 != c20:
                if c10 > c20:
                    rot = 1
                else:
                    rot = 2
            elif c11 > c21:
                rot = 1
            else:
                rot = 2
        elif c01 > c11:
            if c00 > c20:
                rot = 0
            else:
                rot = 2
        elif c10 > c20:
            rot = 1
        else:
            rot = 2

        if not permCells:
            perm[0] += (rot << 1)
        else:
            perm[0] += (rot << 3)

        if rot == 0:
            pass
        elif rot == 1:
            c00, c10 = c10, c20
            c01, c11 = c11, c21
        else:
            c00, c10 = c20, c00
            c01, c11 = c21, c01

        ID[1] = <INDEX_t>(MAX_INT*c00*self.h1MaxInv)
        ID[2] = <INDEX_t>(MAX_INT*c10*self.h1MaxInv)

        ID[3] = <INDEX_t>(MAX_INT*c01*self.h1MaxInv)
        ID[4] = <INDEX_t>(MAX_INT*c11*self.h1MaxInv)

        # 2nd simplex

        if not permCells:
            v00 = self.simplex2[0, 0]-self.center2[0]
            v01 = self.simplex2[0, 1]-self.center2[1]
            v10 = self.simplex2[1, 0]-self.center2[0]
            v11 = self.simplex2[1, 1]-self.center2[1]
            v20 = self.simplex2[2, 0]-self.center2[0]
            v21 = self.simplex2[2, 1]-self.center2[1]
        else:
            v00 = self.simplex1[0, 0]-self.center1[0]
            v01 = self.simplex1[0, 1]-self.center1[1]
            v10 = self.simplex1[1, 0]-self.center1[0]
            v11 = self.simplex1[1, 1]-self.center1[1]
            v20 = self.simplex1[2, 0]-self.center1[0]
            v21 = self.simplex1[2, 1]-self.center1[1]


        c00 = v00*d1 + v01*d2
        c10 = v10*d1 + v11*d2
        c20 = v20*d1 + v21*d2

        d1, d2 = -d2, d1

        c01 = v00*d1 + v01*d2
        c11 = v10*d1 + v11*d2
        c21 = v20*d1 + v21*d2

        if c00 != c10:
            if c00 > c10:
                if c00 != c20:
                    if c00 > c20:
                        rot = 0
                    else:
                        rot = 2
                elif c01 > c21:
                    rot = 0
                else:
                    rot = 2
            elif c10 != c20:
                if c10 > c20:
                    rot = 1
                else:
                    rot = 2
            elif c11 > c21:
                rot = 1
            else:
                rot = 2
        elif c01 > c11:
            if c00 > c20:
                rot = 0
            else:
                rot = 2
        elif c10 > c20:
            rot = 1
        else:
            rot = 2

        if not permCells:
            perm[0] += (rot << 3)
        else:
            perm[0] += (rot << 1)

        if rot == 0:
            pass
        elif rot == 1:
            c00, c10 = c10, c20
            c01, c11 = c11, c21
        else:
            c00, c10 = c20, c00
            c01, c11 = c21, c01

        ID[5] = <INDEX_t>(MAX_INT*c00*self.h2MaxInv)
        ID[6] = <INDEX_t>(MAX_INT*c10*self.h2MaxInv)

        ID[7] = <INDEX_t>(MAX_INT*c01*self.h2MaxInv)
        ID[8] = <INDEX_t>(MAX_INT*c11*self.h2MaxInv)


