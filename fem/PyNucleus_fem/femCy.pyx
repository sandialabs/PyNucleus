###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from libc.math cimport (sin, cos, sinh, cosh, tanh, sqrt, atan2, pow)
import numpy as np
cimport numpy as np
cimport cython

from PyNucleus_base.myTypes import INDEX, REAL, COMPLEX, ENCODE, BOOL
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, ENCODE_t
from PyNucleus_base import uninitialized
from PyNucleus_base.ip_norm cimport mydot, vector_t, complex_vector_t
from . meshCy cimport (meshBase,
                       vertices_t, cells_t,
                       vectorProduct,
                       volume1D, volume1Dnew,
                       volume1D_in_2D,
                       volume2Dnew,
                       volume3D, volume3Dnew,
                       volume2D_in_3D,
                       sortEdge, sortFace,
                       decode_edge,
                       encode_edge)
from . mesh import NO_BOUNDARY
from PyNucleus_base.linear_operators cimport (LinearOperator,
                                               CSR_LinearOperator,
                                               SSS_LinearOperator)
from PyNucleus_base.sparsityPattern cimport sparsityPattern
from . DoFMaps cimport (P0_DoFMap, P1_DoFMap, P2_DoFMap, P3_DoFMap,

                        DoFMap,
                        vectorShapeFunction,
                        fe_vector, complex_fe_vector,
                        multi_fe_vector)
from . quadrature cimport simplexQuadratureRule, Gauss1D, Gauss2D, Gauss3D, simplexXiaoGimbutas
from . functions cimport function, complexFunction, vectorFunction, matrixFunction
from . simplexMapper cimport simplexMapper



cdef class local_matrix_t:
    def __init__(self, INDEX_t dim):
        self.dim = dim
        self.needsCellInfo = False
        self.cell = uninitialized((dim+1), dtype=INDEX)
        self.additiveAssembly = True

    def __call__(self,
                 REAL_t[:, ::1] simplex,
                 REAL_t[::1] contrib):
        return self.eval(simplex, contrib)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void eval(self,
                   REAL_t[:, ::1] simplex,
                   REAL_t[::1] contrib):
        pass

    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cdef void setCell(self,
                      INDEX_t[::1] cell):
        cdef:
            INDEX_t i
        for i in range(self.dim+1):
            self.cell[i] = cell[i]


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline REAL_t simplexVolume1D(const REAL_t[:, ::1] simplex,
                                   REAL_t[:, ::1] temp):
    # temp needs to bed of size 0x1
    # Calculate volume
    return abs(simplex[1, 0]-simplex[0, 0])


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline REAL_t simplexVolume2D(const REAL_t[:, ::1] simplex,
                                   REAL_t[:, ::1] temp):
    # temp needs to bed of size 2x2
    cdef:
        INDEX_t j

    # Calculate volume
    for j in range(2):
        temp[0, j] = simplex[1, j]-simplex[0, j]
        temp[1, j] = simplex[2, j]-simplex[0, j]
    return volume2Dnew(temp)


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline REAL_t simplexVolume1Din2D(const REAL_t[:, ::1] simplex,
                                       REAL_t[:, ::1] temp):
    # temp needs to bed of size 1x2
    # Calculate volume
    temp[0, 0] = simplex[1, 0]-simplex[0, 0]
    temp[0, 1] = simplex[1, 1]-simplex[0, 1]
    return volume1D_in_2D(temp)


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline REAL_t simplexVolume3D(const REAL_t[:, ::1] simplex,
                                   REAL_t[:, ::1] temp):
    # temp needs to be 4x3
    cdef:
        INDEX_t j

    # Calculate volume
    for j in range(3):
        temp[0, j] = simplex[1, j]-simplex[0, j]  # v01
        temp[1, j] = simplex[2, j]-simplex[0, j]  # v02
        temp[2, j] = simplex[3, j]-simplex[0, j]  # v03
    return volume3Dnew(temp[:3, :], temp[3, :])


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline REAL_t simplexVolume2Din3D(const REAL_t[:, ::1] simplex,
                                       REAL_t[:, ::1] temp):
    cdef:
        INDEX_t j
    # temp needs to bed of size 2x3
    # Calculate volume
    for j in range(3):
        temp[0, j] = simplex[1, j]-simplex[0, j]
        temp[1, j] = simplex[2, j]-simplex[0, j]
    return volume2D_in_3D(temp[0, :], temp[1, :])


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline void coeffProducts1D(const REAL_t[:, ::1] simplex,
                                 REAL_t vol,
                                 vectorFunction coeff,
                                 REAL_t[::1] innerProducts,
                                 REAL_t[:, ::1] temp):
    # innerProducts needs to be of size 2
    # temp needs to bed of size 2x1
    cdef:
        INDEX_t i
        REAL_t fac = 0.5
    temp[1, 0] = 0.
    for i in range(2):
        temp[1, 0] += simplex[i, 0]
    temp[1, 0] *= fac
    coeff.eval(temp[1, :], temp[0, :])

    # inner product of barycentric gradients
    innerProducts[0] = (-1.*temp[0, 0])/vol
    innerProducts[1] = (1.*temp[0, 0])/vol


# TODO: double check
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline REAL_t simplexVolumeAndProducts1D(const REAL_t[:, ::1] simplex,
                                                REAL_t[::1] innerProducts,
                                                REAL_t[:, ::1] temp):
    # innerProducts needs to be of size 2
    # temp needs to bed of size 2x1
    cdef:
        INDEX_t i
        REAL_t fac = 0.5, vol
    vol = abs(simplex[1, 0]-simplex[0, 0])

    # inner product of barycentric gradients
    innerProducts[0] = vol**2
    innerProducts[1] = -vol**2
    innerProducts[2] = vol**2
    return vol


cdef class simplexComputations:
    cdef:
        REAL_t[:, ::1] simplex

    cdef void setSimplex(self, REAL_t[:, ::1] simplex):
        self.simplex = simplex

    cdef REAL_t evalVolume(self):
        """Returns the simplex volume."""
        pass

    cdef REAL_t evalVolumeGradients(self,
                                    REAL_t[:, ::1] gradients):
        """Returns the simplex volume and the gradients of the barycentric coordinates
\nabla \lambda_i(x) for x \in K
        """
        pass

    cdef REAL_t evalVolumeGradientsInnerProducts(self,
                                                 REAL_t[:, ::1] gradients,
                                                 REAL_t[::1] innerProducts):
        """Returns the simplex volume, the gradients of the barycentric
coordinates
\nabla \lambda_i(x) for x \in K
 and the innerProducts of the gradients of the barycentric
coordinates
\int_K \nabla \lambda_i \cdot \nabla \lambda_j = vol * gradient_i * gradient_j
        """
        pass

    cdef REAL_t evalSimplexVolumeGradientsInnerProducts(self,
                                                        const REAL_t[:, ::1] simplex,
                                                        REAL_t[:, ::1] gradients,
                                                        REAL_t[::1] innerProducts):
        """Returns the simplex volume, the gradients of the barycentric
coordinates
\nabla \lambda_i(x) for x \in K
 and the innerProducts of the gradients of the barycentric
coordinates
\int_K \nabla \lambda_i \cdot \nabla \lambda_j = vol * gradient_i * gradient_j
        """
        pass


cdef class simplexComputations1D(simplexComputations):
    cdef:
        REAL_t[:, ::1] temp

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t evalVolume(self):
        cdef:
            REAL_t vol

        # Calculate volume
        vol = abs(self.simplex[0, 0]-self.simplex[1, 0])
        return vol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t evalVolumeGradients(self,
                                    REAL_t[:, ::1] gradients):
        cdef:
            REAL_t vol

        # Calculate volume
        vol = abs(self.simplex[0, 0]-self.simplex[1, 0])
        gradients[0, 0] = 1./vol
        gradients[1, 0] = -gradients[0, 0]
        return vol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t evalVolumeGradientsInnerProducts(self,
                                                 REAL_t[:, ::1] gradients,
                                                 REAL_t[::1] innerProducts):
        cdef:
            REAL_t vol

        # Calculate volume
        vol = abs(self.simplex[0, 0]-self.simplex[1, 0])
        gradients[0, 0] = 1./vol
        gradients[1, 0] = -gradients[0, 0]

        # inner product of barycentric gradients
        innerProducts[0] = vol*gradients[0, 0]*gradients[0, 0]
        innerProducts[1] = vol*gradients[0, 0]*gradients[1, 0]
        innerProducts[2] = vol*gradients[1, 0]*gradients[1, 0]
        return vol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t evalSimplexVolumeGradientsInnerProducts(self,
                                                        const REAL_t[:, ::1] simplex,
                                                        REAL_t[:, ::1] gradients,
                                                        REAL_t[::1] innerProducts):
        cdef:
            REAL_t vol

        # Calculate volume
        vol = abs(simplex[0, 0]-simplex[1, 0])
        gradients[0, 0] = 1./vol
        gradients[1, 0] = -gradients[0, 0]

        # inner product of barycentric gradients
        innerProducts[0] = vol*gradients[0, 0]*gradients[0, 0]
        innerProducts[1] = vol*gradients[0, 0]*gradients[1, 0]
        innerProducts[2] = vol*gradients[1, 0]*gradients[1, 0]
        return vol


cdef class simplexComputations2D(simplexComputations):
    cdef:
        REAL_t[:, ::1] temp

    def __init__(self):
        self.temp = uninitialized((2, 2), dtype=REAL)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t evalVolume(self):
        cdef:
            REAL_t vol
            INDEX_t j

        # Calculate volume
        for j in range(2):
            self.temp[0, j] = self.simplex[1, j]-self.simplex[0, j]
            self.temp[1, j] = self.simplex[2, j]-self.simplex[0, j]
        vol = volume2Dnew(self.temp)
        return vol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t evalVolumeGradients(self,
                                    REAL_t[:, ::1] gradients):
        cdef:
            REAL_t vol, f = 1
            INDEX_t j, k

        # Calculate volume
        for j in range(2):
            gradients[0, 1-j] = f*(self.simplex[2, j]-self.simplex[1, j])
            gradients[1, 1-j] = f*(self.simplex[0, j]-self.simplex[2, j])
            gradients[2, 1-j] = f*(self.simplex[1, j]-self.simplex[0, j])
            f = -1
        vol = volume2Dnew(gradients[1:, :])
        f = 0.5/vol
        for k in range(3):
            for j in range(2):
                gradients[k, j] *= f
        return vol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t evalVolumeGradientsInnerProducts(self,
                                                 REAL_t[:, ::1] gradients,
                                                 REAL_t[::1] innerProducts):
        cdef:
            REAL_t vol, f = 1
            INDEX_t j, k

        # Calculate volume
        for j in range(2):
            gradients[0, 1-j] = f*(self.simplex[2, j]-self.simplex[1, j])
            gradients[1, 1-j] = f*(self.simplex[0, j]-self.simplex[2, j])
            gradients[2, 1-j] = f*(self.simplex[1, j]-self.simplex[0, j])
            f = -1
        vol = volume2Dnew(gradients[1:, :])
        f = 0.5/vol
        for k in range(3):
            for j in range(2):
                gradients[k, j] *= f
        # inner product of barycentric gradients
        innerProducts[0] = vol*mydot(gradients[0, :], gradients[0, :])
        innerProducts[1] = vol*mydot(gradients[0, :], gradients[1, :])
        innerProducts[2] = vol*mydot(gradients[0, :], gradients[2, :])
        innerProducts[3] = vol*mydot(gradients[1, :], gradients[1, :])
        innerProducts[4] = vol*mydot(gradients[1, :], gradients[2, :])
        innerProducts[5] = vol*mydot(gradients[2, :], gradients[2, :])
        return vol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t evalSimplexVolumeGradientsInnerProducts(self,
                                                        const REAL_t[:, ::1] simplex,
                                                        REAL_t[:, ::1] gradients,
                                                        REAL_t[::1] innerProducts):
        cdef:
            REAL_t vol, f = 1
            INDEX_t j, k

        # Calculate volume
        for j in range(2):
            gradients[0, 1-j] = f*(simplex[2, j]-simplex[1, j])
            gradients[1, 1-j] = f*(simplex[0, j]-simplex[2, j])
            gradients[2, 1-j] = f*(simplex[1, j]-simplex[0, j])
            f = -1
        vol = volume2Dnew(gradients[1:, :])
        f = 0.5/vol
        for k in range(3):
            for j in range(2):
                gradients[k, j] *= f
        # inner product of barycentric gradients
        innerProducts[0] = vol*mydot(gradients[0, :], gradients[0, :])
        innerProducts[1] = vol*mydot(gradients[0, :], gradients[1, :])
        innerProducts[2] = vol*mydot(gradients[0, :], gradients[2, :])
        innerProducts[3] = vol*mydot(gradients[1, :], gradients[1, :])
        innerProducts[4] = vol*mydot(gradients[1, :], gradients[2, :])
        innerProducts[5] = vol*mydot(gradients[2, :], gradients[2, :])
        return vol



cdef class simplexComputations3D(simplexComputations):
    cdef:
        REAL_t[:, ::1] temp

    def __init__(self):
        self.temp = uninitialized((7, 3), dtype=REAL)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t evalVolume(self):
        cdef:
            REAL_t vol
            INDEX_t j

        # Calculate volume
        for j in range(3):
            self.temp[0, j] = self.simplex[1, j]-self.simplex[0, j]  # v01
            self.temp[1, j] = self.simplex[2, j]-self.simplex[0, j]  # v02
            self.temp[2, j] = self.simplex[3, j]-self.simplex[0, j]  # v03
            self.temp[3, j] = self.simplex[2, j]-self.simplex[1, j]  # v12
            self.temp[4, j] = self.simplex[3, j]-self.simplex[1, j]  # v13
            self.temp[5, j] = self.simplex[2, j]-self.simplex[3, j]  # v32
        vol = volume3Dnew(self.temp[:3, :], self.temp[6, :])
        return vol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t evalVolumeGradients(self,
                                    REAL_t[:, ::1] gradients):
        cdef:
            REAL_t vol, f
            INDEX_t j, k

        # Calculate volume
        for j in range(3):
            self.temp[0, j] = self.simplex[1, j]-self.simplex[0, j]  # v01
            self.temp[1, j] = self.simplex[2, j]-self.simplex[0, j]  # v02
            self.temp[2, j] = self.simplex[3, j]-self.simplex[0, j]  # v03
            self.temp[3, j] = self.simplex[2, j]-self.simplex[1, j]  # v12
            self.temp[4, j] = self.simplex[3, j]-self.simplex[1, j]  # v13
            self.temp[5, j] = self.simplex[2, j]-self.simplex[3, j]  # v32
        vol = volume3Dnew(self.temp[:3, :], self.temp[6, :])

        # v12 x v13
        vectorProduct(self.temp[3, :], self.temp[4, :], gradients[0, :])
        # v02 x v32
        vectorProduct(self.temp[1, :], self.temp[5, :], gradients[1, :])
        # v01 x v03
        vectorProduct(self.temp[0, :], self.temp[2, :], gradients[2, :])
        # v12 x v02
        vectorProduct(self.temp[3, :], self.temp[1, :], gradients[3, :])
        f = 0.16666666666666674/vol
        for k in range(4):
            for j in range(3):
                gradients[k, j] *= f
        return vol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t evalVolumeGradientsInnerProducts(self,
                                                 REAL_t[:, ::1] gradients,
                                                 REAL_t[::1] innerProducts):
        cdef:
            REAL_t vol, f
            INDEX_t j, k

        # Calculate volume
        for j in range(3):
            self.temp[0, j] = self.simplex[1, j]-self.simplex[0, j]  # v01
            self.temp[1, j] = self.simplex[2, j]-self.simplex[0, j]  # v02
            self.temp[2, j] = self.simplex[3, j]-self.simplex[0, j]  # v03
            self.temp[3, j] = self.simplex[2, j]-self.simplex[1, j]  # v12
            self.temp[4, j] = self.simplex[3, j]-self.simplex[1, j]  # v13
            self.temp[5, j] = self.simplex[2, j]-self.simplex[3, j]  # v32
        vol = volume3Dnew(self.temp[:3, :], self.temp[6, :])

        # v12 x v13
        vectorProduct(self.temp[3, :], self.temp[4, :], gradients[0, :])
        # v02 x v32
        vectorProduct(self.temp[1, :], self.temp[5, :], gradients[1, :])
        # v01 x v03
        vectorProduct(self.temp[0, :], self.temp[2, :], gradients[2, :])
        # v12 x v02
        vectorProduct(self.temp[3, :], self.temp[1, :], gradients[3, :])
        f = 0.16666666666666674/vol
        for k in range(4):
            for j in range(3):
                gradients[k, j] *= f
        # inner product of barycentric gradients
        innerProducts[0] = vol*mydot(gradients[0, :], gradients[0, :])
        innerProducts[1] = vol*mydot(gradients[0, :], gradients[1, :])
        innerProducts[2] = vol*mydot(gradients[0, :], gradients[2, :])
        innerProducts[3] = vol*mydot(gradients[0, :], gradients[3, :])
        innerProducts[4] = vol*mydot(gradients[1, :], gradients[1, :])
        innerProducts[5] = vol*mydot(gradients[1, :], gradients[2, :])
        innerProducts[6] = vol*mydot(gradients[1, :], gradients[3, :])
        innerProducts[7] = vol*mydot(gradients[2, :], gradients[2, :])
        innerProducts[8] = vol*mydot(gradients[2, :], gradients[3, :])
        innerProducts[9] = vol*mydot(gradients[3, :], gradients[3, :])
        return vol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef REAL_t evalSimplexVolumeGradientsInnerProducts(self,
                                                        const REAL_t[:, ::1] simplex,
                                                        REAL_t[:, ::1] gradients,
                                                        REAL_t[::1] innerProducts):
        cdef:
            REAL_t vol, f
            INDEX_t j, k

        # Calculate volume
        for j in range(3):
            self.temp[0, j] = simplex[1, j]-simplex[0, j]  # v01
            self.temp[1, j] = simplex[2, j]-simplex[0, j]  # v02
            self.temp[2, j] = simplex[3, j]-simplex[0, j]  # v03
            self.temp[3, j] = simplex[2, j]-simplex[1, j]  # v12
            self.temp[4, j] = simplex[3, j]-simplex[1, j]  # v13
            self.temp[5, j] = simplex[2, j]-simplex[3, j]  # v32
        vol = volume3Dnew(self.temp[:3, :], self.temp[6, :])

        # v12 x v13
        vectorProduct(self.temp[3, :], self.temp[4, :], gradients[0, :])
        # v02 x v32
        vectorProduct(self.temp[1, :], self.temp[5, :], gradients[1, :])
        # v01 x v03
        vectorProduct(self.temp[0, :], self.temp[2, :], gradients[2, :])
        # v12 x v02
        vectorProduct(self.temp[3, :], self.temp[1, :], gradients[3, :])
        f = 0.16666666666666674/vol
        for k in range(4):
            for j in range(3):
                gradients[k, j] *= f
        # inner product of barycentric gradients
        innerProducts[0] = vol*mydot(gradients[0, :], gradients[0, :])
        innerProducts[1] = vol*mydot(gradients[0, :], gradients[1, :])
        innerProducts[2] = vol*mydot(gradients[0, :], gradients[2, :])
        innerProducts[3] = vol*mydot(gradients[0, :], gradients[3, :])
        innerProducts[4] = vol*mydot(gradients[1, :], gradients[1, :])
        innerProducts[5] = vol*mydot(gradients[1, :], gradients[2, :])
        innerProducts[6] = vol*mydot(gradients[1, :], gradients[3, :])
        innerProducts[7] = vol*mydot(gradients[2, :], gradients[2, :])
        innerProducts[8] = vol*mydot(gradients[2, :], gradients[3, :])
        innerProducts[9] = vol*mydot(gradients[3, :], gradients[3, :])
        return vol


# TODO: double check
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline REAL_t simplexVolumeGradientsProducts1D(const REAL_t[:, ::1] simplex,
                                                    REAL_t[::1] innerProducts,
                                                    REAL_t[:, ::1] gradients):
    # innerProducts needs to be of size 2
    # temp needs to bed of size 2x1
    cdef:
        REAL_t vol, f = 1
        INDEX_t j

    # Calculate volume
    gradients[0, 0] = simplex[1, 0]-simplex[0, 0]
    gradients[1, 0] = -gradients[0, 0]
    vol = abs(gradients[0, 0])
    # inner product of barycentric gradients
    innerProducts[0] = vol**2
    innerProducts[1] = -vol**2
    innerProducts[2] = vol**2
    return vol


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline REAL_t simplexVolumeAndProducts2D(const REAL_t[:, ::1] simplex,
                                                REAL_t[::1] innerProducts,
                                                REAL_t[:, ::1] temp):
    # innerProducts needs to bed of size 6
    # temp needs to bed of size 3x2
    cdef:
        REAL_t vol
        INDEX_t j

    # Calculate volume
    for j in range(2):
        temp[0, j] = simplex[2, j]-simplex[1, j]
        temp[1, j] = simplex[0, j]-simplex[2, j]
        temp[2, j] = simplex[1, j]-simplex[0, j]
    vol = volume2Dnew(temp[1:, :])
    # inner product of barycentric gradients
    innerProducts[0] = mydot(temp[0, :], temp[0, :])
    innerProducts[1] = mydot(temp[0, :], temp[1, :])
    innerProducts[2] = mydot(temp[0, :], temp[2, :])
    innerProducts[3] = mydot(temp[1, :], temp[1, :])
    innerProducts[4] = mydot(temp[1, :], temp[2, :])
    innerProducts[5] = mydot(temp[2, :], temp[2, :])
    return vol

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline REAL_t simplexVolumeGradientsProducts2D(const REAL_t[:, ::1] simplex,
                                                      REAL_t[::1] innerProducts,
                                                      REAL_t[:, ::1] gradients):
    # innerProducts needs to bed of size 6
    # temp needs to bed of size 3x2
    cdef:
        REAL_t vol, f = 1
        INDEX_t j

    # Calculate volume
    for j in range(2):
        gradients[0, 1-j] = f*(simplex[2, j]-simplex[1, j])
        gradients[1, 1-j] = f*(simplex[0, j]-simplex[2, j])
        gradients[2, 1-j] = f*(simplex[1, j]-simplex[0, j])
        f = -1
    vol = volume2Dnew(gradients[1:, :])
    # inner product of barycentric gradients
    innerProducts[0] = mydot(gradients[0, :], gradients[0, :])
    innerProducts[1] = mydot(gradients[0, :], gradients[1, :])
    innerProducts[2] = mydot(gradients[0, :], gradients[2, :])
    innerProducts[3] = mydot(gradients[1, :], gradients[1, :])
    innerProducts[4] = mydot(gradients[1, :], gradients[2, :])
    innerProducts[5] = mydot(gradients[2, :], gradients[2, :])
    return vol


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline REAL_t mydot_rot2D(const REAL_t[::1] a, const REAL_t[::1] b):
    return -a[0]*b[1]+a[1]*b[0]


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline void coeffProducts2D(const REAL_t[:, ::1] simplex,
                                 REAL_t vol,
                                 vectorFunction coeff,
                                 REAL_t[::1] innerProducts,
                                 REAL_t[:, ::1] temp):
    # innerProducts needs to bed of size 3
    # temp needs to bed of size 4x2
    cdef:
        INDEX_t i, j
        REAL_t fac = 1./3.
    for j in range(2):
        temp[0, j] = 0.
    for i in range(3):
        for j in range(2):
            temp[0, j] += simplex[i, j]
    for j in range(2):
        temp[0, j] *= fac
    coeff.eval(temp[0, :], temp[3, :])

    # Calculate volume
    for j in range(2):
        temp[0, j] = simplex[2, j]-simplex[1, j]
        temp[1, j] = simplex[0, j]-simplex[2, j]
        temp[2, j] = simplex[1, j]-simplex[0, j]
    # inner product of coeffVec with barycentric gradients
    innerProducts[0] = 0.5/vol*mydot_rot2D(temp[3, :], temp[0, :])
    innerProducts[1] = 0.5/vol*mydot_rot2D(temp[3, :], temp[1, :])
    innerProducts[2] = 0.5/vol*mydot_rot2D(temp[3, :], temp[2, :])


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline REAL_t simplexVolumeAndProducts3D(const REAL_t[:, ::1] simplex,
                                                REAL_t[::1] innerProducts,
                                                REAL_t[:, ::1] temp):
    # innerProducts needs to bed of size 10
    # temp needs to bed of size 10x3
    cdef:
        REAL_t vol
        INDEX_t j

    # Calculate volume
    for j in range(3):
        temp[0, j] = simplex[1, j]-simplex[0, j]  # v01
        temp[1, j] = simplex[2, j]-simplex[0, j]  # v02
        temp[2, j] = simplex[3, j]-simplex[0, j]  # v03
        temp[3, j] = simplex[2, j]-simplex[1, j]  # v12
        temp[4, j] = simplex[3, j]-simplex[1, j]  # v13
        temp[5, j] = simplex[2, j]-simplex[3, j]  # v32
    vol = volume3Dnew(temp[:3, :], temp[6, :])

    # v12 x v13
    vectorProduct(temp[3, :], temp[4, :], temp[6, :])
    # v02 x v32
    vectorProduct(temp[1, :], temp[5, :], temp[7, :])
    # v01 x v03
    vectorProduct(temp[0, :], temp[2, :], temp[8, :])
    # v12 x v02
    vectorProduct(temp[3, :], temp[1, :], temp[9, :])
    # inner product of barycentric gradients
    innerProducts[0] = mydot(temp[6, :], temp[6, :])
    innerProducts[1] = mydot(temp[6, :], temp[7, :])
    innerProducts[2] = mydot(temp[6, :], temp[8, :])
    innerProducts[3] = mydot(temp[6, :], temp[9, :])
    innerProducts[4] = mydot(temp[7, :], temp[7, :])
    innerProducts[5] = mydot(temp[7, :], temp[8, :])
    innerProducts[6] = mydot(temp[7, :], temp[9, :])
    innerProducts[7] = mydot(temp[8, :], temp[8, :])
    innerProducts[8] = mydot(temp[8, :], temp[9, :])
    innerProducts[9] = mydot(temp[9, :], temp[9, :])
    return vol


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline REAL_t simplexVolumeGradientsProducts3D(const REAL_t[:, ::1] simplex,
                                                    REAL_t[::1] innerProducts,
                                                    REAL_t[:, ::1] gradients,
                                                    REAL_t[:, ::1] temp):
    # innerProducts needs to be of size 10
    # gradients needs to be of size 4x3
    # temp needs to be of size 7x3
    cdef:
        REAL_t vol
        INDEX_t j

    # Calculate volume
    for j in range(3):
        temp[0, j] = simplex[1, j]-simplex[0, j]  # v01
        temp[1, j] = simplex[2, j]-simplex[0, j]  # v02
        temp[2, j] = simplex[3, j]-simplex[0, j]  # v03
        temp[3, j] = simplex[2, j]-simplex[1, j]  # v12
        temp[4, j] = simplex[3, j]-simplex[1, j]  # v13
        temp[5, j] = simplex[2, j]-simplex[3, j]  # v32
    vol = volume3Dnew(temp[:3, :], temp[6, :])

    # v12 x v13
    vectorProduct(temp[3, :], temp[4, :], gradients[0, :])
    # v02 x v32
    vectorProduct(temp[1, :], temp[5, :], gradients[1, :])
    # v01 x v03
    vectorProduct(temp[0, :], temp[2, :], gradients[2, :])
    # v12 x v02
    vectorProduct(temp[3, :], temp[1, :], gradients[3, :])
    # inner product of barycentric gradients
    innerProducts[0] = mydot(gradients[0, :], gradients[0, :])
    innerProducts[1] = mydot(gradients[0, :], gradients[1, :])
    innerProducts[2] = mydot(gradients[0, :], gradients[2, :])
    innerProducts[3] = mydot(gradients[0, :], gradients[3, :])
    innerProducts[4] = mydot(gradients[1, :], gradients[1, :])
    innerProducts[5] = mydot(gradients[1, :], gradients[2, :])
    innerProducts[6] = mydot(gradients[1, :], gradients[3, :])
    innerProducts[7] = mydot(gradients[2, :], gradients[2, :])
    innerProducts[8] = mydot(gradients[2, :], gradients[3, :])
    innerProducts[9] = mydot(gradients[3, :], gradients[3, :])
    return vol


cdef class mass_1d(local_matrix_t):
    cdef:
        REAL_t[:, ::1] temp

    def __init__(self):
        self.temp = uninitialized((0, 1), dtype=REAL)
        local_matrix_t.__init__(self, 1)


cdef class drift_1d(local_matrix_t):
    cdef:
        REAL_t[:, ::1] temp
        REAL_t[::1] innerProducts

    def __init__(self):
        self.temp = uninitialized((2, 1), dtype=REAL)
        self.innerProducts = uninitialized((2), dtype=REAL)
        local_matrix_t.__init__(self, 1)


cdef class drift_1d_P1(drift_1d):
    cdef:
        vectorFunction coeff

    def __init__(self, vectorFunction coeff):
        drift_1d.__init__(self)
        self.coeff = coeff

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t fac = 0.5

        # coeff \cdot \grad\lambda_i should be scaled by 1/vol, but that gets killed by the factor vol of the integration
        coeffProducts1D(simplex, 1.0, self.coeff, self.innerProducts, self.temp)

        contrib[0] = self.innerProducts[0]*fac
        contrib[1] = self.innerProducts[1]*fac
        contrib[2] = self.innerProducts[0]*fac
        contrib[3] = self.innerProducts[1]*fac


cdef class stiffness_1d_sym(local_matrix_t):
    cdef:
        REAL_t[:, ::1] temp
        REAL_t[::1] innerProducts

    def __init__(self):
        self.innerProducts = uninitialized((0), dtype=REAL)
        self.temp = uninitialized((0, 1), dtype=REAL)
        local_matrix_t.__init__(self, 1)


cdef class mass_2d(local_matrix_t):
    cdef:
        REAL_t[:, ::1] temp
        REAL_t[::1] innerProducts

    def __init__(self):
        self.temp = uninitialized((3, 2), dtype=REAL)
        self.innerProducts = uninitialized((6), dtype=REAL)
        local_matrix_t.__init__(self, 2)


cdef class drift_2d(local_matrix_t):
    cdef:
        REAL_t[:, ::1] temp
        REAL_t[::1] innerProducts

    def __init__(self):
        self.temp = uninitialized((4, 2), dtype=REAL)
        self.innerProducts = uninitialized((3), dtype=REAL)
        local_matrix_t.__init__(self, 2)


cdef class drift_2d_P1(drift_2d):
    cdef:
        vectorFunction coeff

    def __init__(self, vectorFunction coeff):
        drift_2d.__init__(self)
        self.coeff = coeff

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t fac = 1./3.

        # coeff \cdot \grad\lambda_i should be scaled by 1/vol, but that gets killed by the factor vol of the integration
        coeffProducts2D(simplex, 1.0, self.coeff, self.innerProducts, self.temp)

        contrib[0] = self.innerProducts[0]*fac
        contrib[1] = self.innerProducts[1]*fac
        contrib[2] = self.innerProducts[2]*fac
        contrib[3] = self.innerProducts[0]*fac
        contrib[4] = self.innerProducts[1]*fac
        contrib[5] = self.innerProducts[2]*fac
        contrib[6] = self.innerProducts[0]*fac
        contrib[7] = self.innerProducts[1]*fac
        contrib[8] = self.innerProducts[2]*fac


cdef class stiffness_2d_sym(local_matrix_t):
    cdef:
        REAL_t[:, ::1] temp
        REAL_t[::1] innerProducts

    def __init__(self):
        self.innerProducts = uninitialized((6), dtype=REAL)
        self.temp = uninitialized((3, 2), dtype=REAL)
        local_matrix_t.__init__(self, 2)


cdef class curlcurl_2d_sym(local_matrix_t):
    cdef:
        REAL_t[:, ::1] temp
        REAL_t[::1] innerProducts

    def __init__(self):
        self.innerProducts = uninitialized((6), dtype=REAL)
        self.temp = uninitialized((3, 2), dtype=REAL)
        local_matrix_t.__init__(self, 2)


cdef class mass_3d(local_matrix_t):
    cdef:
        REAL_t[:, ::1] temp

    def __init__(self):
        self.temp = uninitialized((4, 3), dtype=REAL)
        local_matrix_t.__init__(self, 3)


cdef class stiffness_3d_sym(local_matrix_t):
    cdef:
        REAL_t[:, ::1] temp
        REAL_t[::1] innerProducts

    def __init__(self):
        self.innerProducts = uninitialized((10), dtype=REAL)
        self.temp = uninitialized((10, 3), dtype=REAL)
        local_matrix_t.__init__(self, 3)


# cdef class generic_matrix(local_matrix_t):
#     cdef:
#         REAL_t[::1] entries

#     def __init__(self, REAL_t[::1] entries):
#         self.entries = entries

#     @cython.initializedcheck(False)
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     cdef inline void eval(self,
#                           REAL_t[:, ::1] local_vertices,
#                           REAL_t[::1] contrib,
#                           REAL_t[:, ::1] span):
#         cdef:
#             REAL_t vol
#             INDEX_t k, j

#         # Calculate volume
#         for k in range(local_vertices.shape[0]-1):
#             for j in range(local_vertices.shape[1]):
#                 span[k, j] = local_vertices[k+1, j]-local_vertices[0, j]
#         # TODO: Fix this
#         vol = volume2Dnew(span)

#         for k in range(self.entries.shape[0]):
#             contrib[k] = vol*self.entries[k]


######################################################################
# Local mass matrices for subdmanifolds in 1d, 2d

cdef class mass_0d_in_1d_sym_P1(mass_1d):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        contrib[0] = 1.0


cdef class mass_1d_in_2d_sym_P1(mass_2d):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 1.0/6.0

        vol *= simplexVolume1Din2D(simplex, self.temp)

        contrib[0] = contrib[2] = 2.0*vol
        contrib[1] = vol


cdef class mass_2d_in_3d_sym_P1(mass_3d):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.0833333333333333

        vol *= simplexVolume2Din3D(simplex, self.temp)

        contrib[0] = 2*vol
        contrib[1] = vol
        contrib[2] = vol
        contrib[3] = 2*vol
        contrib[4] = vol
        contrib[5] = 2*vol



######################################################################
# Anisotropic local mass matrices in 1d, 2d, 3d

cdef class mass_quadrature_matrix(local_matrix_t):
    cdef:
        function diffusivity
        simplexQuadratureRule qr
        REAL_t[:, ::1] PHI
        REAL_t[::1] funVals
        REAL_t[:, ::1] temp

    def __init__(self, function diffusivity, DoFMap DoFMap, simplexQuadratureRule qr):
        cdef:
            INDEX_t I, k
        local_matrix_t.__init__(self, DoFMap.dim)
        self.diffusivity = diffusivity
        self.qr = qr

        # evaluate local shape functions on quadrature nodes
        self.PHI = uninitialized((DoFMap.dofs_per_element, qr.num_nodes), dtype=REAL)
        for I in range(DoFMap.dofs_per_element):
            for k in range(qr.num_nodes):
                self.PHI[I, k] = DoFMap.localShapeFunctions[I](np.ascontiguousarray(qr.nodes[:, k]))

        self.funVals = uninitialized((qr.num_nodes), dtype=REAL)
        self.temp = uninitialized((10, DoFMap.dim), dtype=REAL)


cdef class stiffness_quadrature_matrix(mass_quadrature_matrix):
    cdef:
        REAL_t[::1] innerProducts

    def __init__(self, function diffusivity, simplexQuadratureRule qr):
        from . DoFMaps import P1_DoFMap
        from . mesh import meshNd
        fakeMesh = meshNd(uninitialized((0, self.dim), dtype=REAL),
                          uninitialized((0, self.dim+1), dtype=INDEX))
        dm = P1_DoFMap(fakeMesh)
        super(stiffness_quadrature_matrix, self).__init__(diffusivity, dm, qr)
        self.innerProducts = uninitialized((((self.dim+1)*(self.dim+2))//2), dtype=REAL)


cdef class mass_1d_sym_scalar_anisotropic(mass_quadrature_matrix):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol
            INDEX_t p, I, J, k
        self.qr.evalFun(self.diffusivity, simplex, self.funVals)
        vol = simplexVolume1D(simplex, self.temp)

        p = 0
        for I in range(self.PHI.shape[0]):
            for J in range(I, self.PHI.shape[0]):
                contrib[p] = 0.
                for k in range(self.qr.num_nodes):
                    contrib[p] += vol * self.qr.weights[k] * self.funVals[k] * self.PHI[I, k] * self.PHI[J, k]
                p += 1


cdef class mass_2d_sym_scalar_anisotropic(mass_quadrature_matrix):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol
            INDEX_t p, I, J, k
        self.qr.evalFun(self.diffusivity, simplex, self.funVals)
        vol = simplexVolume2D(simplex, self.temp)

        p = 0
        for I in range(self.PHI.shape[0]):
            for J in range(I, self.PHI.shape[0]):
                contrib[p] = 0.
                for k in range(self.qr.num_nodes):
                    contrib[p] += vol * self.qr.weights[k] * self.funVals[k] * self.PHI[I, k] * self.PHI[J, k]
                p += 1


cdef class mass_3d_sym_scalar_anisotropic(mass_quadrature_matrix):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol
            INDEX_t p, I, J, k
        self.qr.evalFun(self.diffusivity, simplex, self.funVals)
        vol = simplexVolume3D(simplex, self.temp)

        p = 0
        for I in range(self.PHI.shape[0]):
            for J in range(I, self.PHI.shape[0]):
                contrib[p] = 0.
                for k in range(self.qr.num_nodes):
                    contrib[p] += vol * self.qr.weights[k] * self.funVals[k] * self.PHI[I, k] * self.PHI[J, k]
                p += 1


######################################################################
# Local stiffness matrices in 1d, 2d, 3d

cdef class stiffness_1d_in_2d_sym_P1(stiffness_2d_sym):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 1.0/6.0

        vol /= simplexVolume1Din2D(simplex, self.temp)

        contrib[0] = contrib[2] = vol
        contrib[1] = -vol


cdef class stiffness_2d_sym_anisotropic_P1(stiffness_2d_sym):
    cdef:
        function diffusivity, diff00, diff01, diff11
        REAL_t[::1] mean, temp2
        public REAL_t[:, ::1] K
        BOOL_t diffTensor

    def __init__(self, diffusivity):
        super(stiffness_2d_sym_anisotropic_P1, self).__init__()
        if isinstance(diffusivity, function):
            self.diffusivity = diffusivity
            self.diffTensor = False
        elif len(diffusivity) == 3:
            self.diff00, self.diff01, self.diff11 = diffusivity[0], diffusivity[1], diffusivity[2]
            self.temp2 = uninitialized((2), dtype=REAL)
            self.K = uninitialized((2, 2), dtype=REAL)
            self.diffTensor = True
        else:
            raise NotImplementedError()
        self.mean = uninitialized((2), dtype=REAL)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.25
            INDEX_t k, j, p

        # Calculate gradient matrix
        for j in range(2):
            self.mean[j] = (simplex[0, j] +
                            simplex[1, j] +
                            simplex[2, j])/3.0
        vol /= simplexVolumeAndProducts2D(simplex, self.innerProducts, self.temp)

        if self.diffTensor:
            # need to take into account rotation matrix, that's why
            # the entries are in a weird order
            self.K[0, 0] = self.diff11(self.mean)
            self.K[0, 1] = self.K[1, 0] = -self.diff01(self.mean)
            self.K[1, 1] = self.diff00(self.mean)

            p = 0
            for j in range(3):
                matvec(self.K, self.temp[j, :], self.temp2)
                for k in range(j, 3):
                    contrib[p] = mydot(self.temp2, self.temp[k, :])*vol
                    p += 1
        else:
            vol *= self.diffusivity(self.mean)

            p = 0
            for j in range(3):
                for k in range(j, 3):
                    contrib[p] = self.innerProducts[p]*vol
                    p += 1


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void matvec(const REAL_t[:, ::1] A, const REAL_t[::1] x, REAL_t[::1] y):
    cdef INDEX_t i, j
    for i in range(A.shape[0]):
        y[i] = 0.
        for j in range(A.shape[1]):
            y[i] += A[i, j]*x[j]


cdef class stiffness_2d_sym_anisotropic2_P1(stiffness_2d_sym):
    cdef:
        REAL_t alpha, beta, theta
        REAL_t[:, ::1] diffusivity
        REAL_t[::1] temp2

    def __init__(self, REAL_t alpha, REAL_t beta, REAL_t theta):
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        # TODO:
        # need to take into account rotation matrix, that's why
        # the entries should be in a weird order, see above
        Q = np.array([[cos(theta), -sin(theta)],
                      [sin(theta), cos(theta)]],
                     dtype=REAL)
        D = np.array([[alpha, 0.],
                      [0., beta]],
                     dtype=REAL)
        self.diffusivity = np.dot(Q, np.dot(D, Q.T))
        self.temp2 = uninitialized((2), dtype=REAL)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        # temp needs to bed of size 3x2
        cdef:
            REAL_t vol = 0.25
            INDEX_t k, j, p

        # Calculate gradient matrix
        vol /= simplexVolumeAndProducts2D(simplex, self.innerProducts, self.temp)

        p = 0
        for j in range(3):
            matvec(self.diffusivity, self.temp[j, :], self.temp2)
            for k in range(j, 3):
                contrib[p] = mydot(self.temp2, self.temp[k, :])*vol
                p += 1


cdef class stiffness_2d_sym_anisotropic3_P1(stiffness_2d_sym):
    cdef:
        matrixFunction K
        public REAL_t[:, ::1] diffusivity
        REAL_t[::1] temp2

    def __init__(self, matrixFunction K):
        self.K = K
        assert self.K.rows == 2
        assert self.K.columns == 2
        assert self.K.symmetric
        self.diffusivity = uninitialized((2, 2), dtype=REAL)
        self.temp = uninitialized((3, 2), dtype=REAL)
        self.temp2 = uninitialized((2), dtype=REAL)
        self.innerProducts = uninitialized((6), dtype=REAL)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        # temp needs to be of size 3x2
        cdef:
            REAL_t vol = 0.25
            INDEX_t k, j, p

        # Calculate gradient matrix
        vol /= simplexVolumeGradientsProducts2D(simplex, self.innerProducts, self.temp)

        # calculate center
        self.temp2[:] = 0.
        for j in range(3):
            for k in range(2):
                self.temp2[k] += simplex[j, k]
        for k in range(2):
            self.temp2[k] /= 3.
        self.K.eval(self.temp2, self.diffusivity)

        p = 0
        for j in range(3):
            matvec(self.diffusivity, self.temp[j, :], self.temp2)
            for k in range(j, 3):
                contrib[p] = mydot(self.temp2, self.temp[k, :])*vol
                p += 1



cdef class div_div_2d(local_matrix_t):
    cdef:
        REAL_t[:, ::1] gradients
        REAL_t[::1] innerProducts

    def __init__(self):
        cdef:
            INDEX_t dim = 2
        local_matrix_t.__init__(self, dim)
        self.innerProducts = uninitialized((((dim+1)*(dim+2))//2), dtype=REAL)
        self.gradients = uninitialized((dim+1, dim), dtype=REAL)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.25, fac1, fac2
            INDEX_t p, vertexNo1, vertexNo2, component1, component2
            INDEX_t dim = 2

        # Calculate gradient matrix
        vol /= simplexVolumeGradientsProducts2D(simplex, self.innerProducts, self.gradients)

        p = 0
        for vertexNo1 in range(dim+1):
            for component1 in range(dim):
                fac1 = self.gradients[vertexNo1, component1]
                # vertexNo2 = vertexNo1
                for component2 in range(component1, dim):
                    fac2 = self.gradients[vertexNo1, component2]
                    contrib[p] = vol * fac1 * fac2
                    p += 1
                for vertexNo2 in range(vertexNo1+1, dim+1):
                    for component2 in range(dim):
                        fac2 = self.gradients[vertexNo2, component2]
                        contrib[p] = vol * fac1 * fac2
                        p += 1


cdef class elasticity_1d_P1(local_matrix_t):
    cdef:
        REAL_t[:, ::1] gradients
        REAL_t[::1] innerProducts
        REAL_t lam, mu
        simplexComputations sC

    def __init__(self, REAL_t lam, REAL_t mu):
        cdef:
            INDEX_t dim = 1
        local_matrix_t.__init__(self, dim)
        self.lam = lam
        self.mu = mu
        self.innerProducts = uninitialized((((dim+1)*(dim+2))//2), dtype=REAL)
        self.gradients = uninitialized((dim+1, dim), dtype=REAL)
        self.sC = simplexComputations1D()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol, fac1, fac2
            INDEX_t p, vertexNo1, vertexNo2, component1, component2, j, q
            INDEX_t dim = 1

        # Calculate gradient matrix
        vol = self.sC.evalSimplexVolumeGradientsInnerProducts(simplex, self.gradients, self.innerProducts)

        p = 0
        for vertexNo1 in range(dim+1):
            for component1 in range(dim):
                fac1 = self.gradients[vertexNo1, component1]
                # vertexNo2 = vertexNo1
                for component2 in range(component1, dim):
                    fac2 = self.gradients[vertexNo1, component2]
                    contrib[p] = vol * self.gradients[vertexNo1, component2] * self.gradients[vertexNo1, component1]
                    if component1 == component2:
                        q = vertexNo1*(dim+1)
                        contrib[p] += self.innerProducts[q]
                    contrib[p] = self.lam*vol*fac1*fac2 + self.mu * contrib[p]
                    p += 1
                for vertexNo2 in range(vertexNo1+1, dim+1):
                    for component2 in range(dim):
                        fac2 = self.gradients[vertexNo2, component2]
                        contrib[p] = vol * self.gradients[vertexNo1, component2] * self.gradients[vertexNo2, component1]
                        if component1 == component2:
                            q = vertexNo1*dim + vertexNo2
                            contrib[p] += self.innerProducts[q]
                        contrib[p] = self.lam*vol*fac1*fac2 + self.mu * contrib[p]
                        p += 1


cdef class elasticity_2d_P1(local_matrix_t):
    cdef:
        REAL_t[:, ::1] gradients
        REAL_t[::1] innerProducts
        REAL_t lam, mu
        simplexComputations sC

    def __init__(self, REAL_t lam, REAL_t mu):
        cdef:
            INDEX_t dim = 2
        local_matrix_t.__init__(self, dim)
        self.lam = lam
        self.mu = mu
        self.innerProducts = uninitialized((((dim+1)*(dim+2))//2), dtype=REAL)
        self.gradients = uninitialized((dim+1, dim), dtype=REAL)
        self.sC = simplexComputations2D()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol, fac1, fac2
            INDEX_t p, vertexNo1, vertexNo2, component1, component2, j, q
            INDEX_t dim = 2

        # Calculate gradient matrix
        vol = self.sC.evalSimplexVolumeGradientsInnerProducts(simplex, self.gradients, self.innerProducts)

        p = 0
        for vertexNo1 in range(dim+1):
            for component1 in range(dim):
                fac1 = self.gradients[vertexNo1, component1]
                # vertexNo2 = vertexNo1
                for component2 in range(component1, dim):
                    fac2 = self.gradients[vertexNo1, component2]
                    contrib[p] = vol * self.gradients[vertexNo1, component2] * self.gradients[vertexNo1, component1]
                    if component1 == component2:
                        q = vertexNo1*(dim+1) - (vertexNo1*(vertexNo1-1))//2
                        contrib[p] += self.innerProducts[q]
                    contrib[p] = self.lam*vol*fac1*fac2 + self.mu * contrib[p]
                    p += 1
                for vertexNo2 in range(vertexNo1+1, dim+1):
                    for component2 in range(dim):
                        fac2 = self.gradients[vertexNo2, component2]
                        contrib[p] = vol * self.gradients[vertexNo1, component2] * self.gradients[vertexNo2, component1]
                        if component1 == component2:
                            q = vertexNo1*dim - (vertexNo1*(vertexNo1-1))//2 + vertexNo2
                            contrib[p] += self.innerProducts[q]
                        contrib[p] = self.lam*vol*fac1*fac2 + self.mu * contrib[p]
                        p += 1


cdef class elasticity_3d_P1(local_matrix_t):
    cdef:
        REAL_t[:, ::1] gradients
        REAL_t[::1] innerProducts
        REAL_t lam, mu
        simplexComputations sC

    def __init__(self, REAL_t lam, REAL_t mu):
        cdef:
            INDEX_t dim = 3
        local_matrix_t.__init__(self, dim)
        self.lam = lam
        self.mu = mu
        self.innerProducts = uninitialized((((dim+1)*(dim+2))//2), dtype=REAL)
        self.gradients = uninitialized((dim+1, dim), dtype=REAL)
        self.sC = simplexComputations3D()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol, fac1, fac2
            INDEX_t p, vertexNo1, vertexNo2, component1, component2, j, q
            INDEX_t dim = 3
            REAL_t temp

        # Calculate gradient matrix
        vol = self.sC.evalSimplexVolumeGradientsInnerProducts(simplex, self.gradients, self.innerProducts)

        p = 0
        for vertexNo1 in range(dim+1):
            for component1 in range(dim):
                fac1 = self.gradients[vertexNo1, component1]
                # vertexNo2 = vertexNo1
                for component2 in range(component1, dim):
                    fac2 = self.gradients[vertexNo1, component2]
                    contrib[p] = vol * self.gradients[vertexNo1, component2] * self.gradients[vertexNo1, component1]
                    if component1 == component2:
                        q = vertexNo1*(dim+1) - (vertexNo1*(vertexNo1-1))//2
                        contrib[p] += self.innerProducts[q]
                    contrib[p] = self.lam*vol*fac1*fac2 + self.mu * contrib[p]
                    p += 1
                for vertexNo2 in range(vertexNo1+1, dim+1):
                    for component2 in range(dim):
                        fac2 = self.gradients[vertexNo2, component2]
                        contrib[p] = vol * self.gradients[vertexNo1, component2] * self.gradients[vertexNo2, component1]
                        if component1 == component2:
                            q = vertexNo1*dim - (vertexNo1*(vertexNo1-1))//2 + vertexNo2
                            contrib[p] += self.innerProducts[q]
                        contrib[p] = self.lam*vol*fac1*fac2 + self.mu * contrib[p]
                        p += 1


cdef class mass_1d_in_2d_sym_P2(mass_2d):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        # needs no temp memory
        cdef:
            REAL_t vol = 1./30.
        vol *= simplexVolume1Din2D(simplex, self.temp)

        contrib[0] = contrib[3] = 4.0*vol
        contrib[1] = -vol
        contrib[2] = contrib[4] = 2.0*vol
        contrib[5] = 16.0*vol


include "mass_1D_P0.pxi"
include "mass_2D_P0.pxi"
include "mass_3D_P0.pxi"

include "mass_1D_P0_P1.pxi"
include "mass_2D_P0_P1.pxi"
include "mass_3D_P0_P1.pxi"

include "mass_1D_P1.pxi"
include "mass_2D_P1.pxi"
include "mass_3D_P1.pxi"
include "stiffness_1D_P1.pxi"
include "stiffness_2D_P1.pxi"
include "stiffness_3D_P1.pxi"
include "scalar_coefficient_stiffness_1D_P1.pxi"
include "scalar_coefficient_stiffness_2D_P1.pxi"
include "scalar_coefficient_stiffness_3D_P1.pxi"

include "mass_1D_P2.pxi"
include "mass_2D_P2.pxi"
include "mass_3D_P2.pxi"
include "stiffness_1D_P2.pxi"
include "stiffness_2D_P2.pxi"
include "stiffness_3D_P2.pxi"
include "scalar_coefficient_stiffness_1D_P2.pxi"
include "scalar_coefficient_stiffness_2D_P2.pxi"
include "scalar_coefficient_stiffness_3D_P2.pxi"


include "mass_1D_P3.pxi"
include "mass_2D_P3.pxi"
include "mass_3D_P3.pxi"
include "stiffness_1D_P3.pxi"
include "stiffness_2D_P3.pxi"
include "stiffness_3D_P3.pxi"



def assembleMass(DoFMap dm,
                 vector_t boundary_data=None,
                 vector_t rhs_contribution=None,
                 LinearOperator A=None,
                 INDEX_t start_idx=-1,
                 INDEX_t end_idx=-1,
                 BOOL_t sss_format=False,
                 BOOL_t reorder=False,
                 INDEX_t[::1] cellIndices=None,
                 coefficient=None,
                 simplexQuadratureRule qr=None):
    cdef:
        INDEX_t dim = dm.mesh.dim
        INDEX_t manifold_dim = dm.mesh.manifold_dim
        local_matrix_t local_matrix
    if coefficient is None:
        if dim == manifold_dim:
            if isinstance(dm, P0_DoFMap):
                if dim == 1:
                    local_matrix = mass_1d_sym_P0()
                elif dim == 2:
                    local_matrix = mass_2d_sym_P0()
                elif dim == 3:
                    local_matrix = mass_3d_sym_P0()
                else:
                    raise NotImplementedError()
            elif isinstance(dm, P1_DoFMap):
                if dim == 1:
                    local_matrix = mass_1d_sym_P1()
                elif dim == 2:
                    local_matrix = mass_2d_sym_P1()
                elif dim == 3:
                    local_matrix = mass_3d_sym_P1()
                else:
                    raise NotImplementedError()
            elif isinstance(dm, P2_DoFMap):
                if dim == 1:
                    local_matrix = mass_1d_sym_P2()
                elif dim == 2:
                    local_matrix = mass_2d_sym_P2()
                elif dim == 3:
                    local_matrix = mass_3d_sym_P2()
                else:
                    raise NotImplementedError()
            elif isinstance(dm, P3_DoFMap):
                if dim == 1:
                    local_matrix = mass_1d_sym_P3()
                elif dim == 2:
                    local_matrix = mass_2d_sym_P3()
                elif dim == 3:
                    local_matrix = mass_3d_sym_P3()
                else:
                    raise NotImplementedError()

            else:
                raise NotImplementedError(dm)
        else:
            assert manifold_dim == dim-1
            if isinstance(dm, P1_DoFMap):
                if dim == 1:
                    local_matrix = mass_0d_in_1d_sym_P1()
                elif dim == 2:
                    local_matrix = mass_1d_in_2d_sym_P1()
                elif dim == 3:
                    local_matrix = mass_2d_in_3d_sym_P1()
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    elif isinstance(coefficient, function):
        assert manifold_dim == dim
        if qr is None:
            qr = simplexXiaoGimbutas(2*dm.polynomialOrder+2, dim)
        if dim == 1:
            local_matrix = mass_1d_sym_scalar_anisotropic(coefficient, dm, qr)
        elif dim == 2:
            local_matrix = mass_2d_sym_scalar_anisotropic(coefficient, dm, qr)
        elif dim == 3:
            local_matrix = mass_3d_sym_scalar_anisotropic(coefficient, dm, qr)
        else:
            raise NotImplementedError(dim)
    return assembleMatrix(dm.mesh,
                          dm,
                          local_matrix,
                          boundary_data,
                          rhs_contribution,
                          A,
                          start_idx,
                          end_idx,
                          sss_format,
                          reorder,
                          cellIndices=cellIndices)


def getSurfaceDoFMap(meshBase mesh,
                     meshBase surface,
                     DoFMap volumeDoFMap,
                     INDEX_t[::1] boundaryCells=None):
    cdef:
        DoFMap dmS
        INDEX_t[:, ::1] v2d
        INDEX_t cellNo, localVertexNo, vertexNo, k

    if isinstance(volumeDoFMap, P0_DoFMap):
        assert boundaryCells is not None
        dmS = P0_DoFMap(surface, NO_BOUNDARY)
        for cellNo in range(surface.num_cells):
            dmS.dofs[cellNo, 0] = volumeDoFMap.dofs[boundaryCells[cellNo], 0]
        return dmS
    elif isinstance(volumeDoFMap, P1_DoFMap):
        dmS = P1_DoFMap(surface, NO_BOUNDARY)
    elif isinstance(volumeDoFMap, P2_DoFMap):
        dmS = P2_DoFMap(surface, NO_BOUNDARY)
    elif isinstance(volumeDoFMap, P3_DoFMap):
        dmS = P3_DoFMap(surface, NO_BOUNDARY)
    else:
        raise NotImplementedError()

    assert volumeDoFMap.dofs_per_edge == 0
    assert volumeDoFMap.dofs_per_face == 0

    dmS.num_dofs = volumeDoFMap.num_dofs

    v2d = uninitialized((mesh.num_vertices, volumeDoFMap.dofs_per_vertex), dtype=INDEX)
    volumeDoFMap.getVertexDoFs(v2d)

    for cellNo in range(surface.num_cells):
        for localVertexNo in range(surface.cells.shape[1]):
            vertexNo = surface.cells[cellNo, localVertexNo]
            for k in range(dmS.dofs_per_vertex):
                dmS.dofs[cellNo, localVertexNo*dmS.dofs_per_vertex+k] = v2d[vertexNo, k]
    return dmS


def assembleSurfaceMass(meshBase mesh,
                        meshBase surface,
                        DoFMap volumeDoFMap,
                        LinearOperator A=None,
                        BOOL_t sss_format=False,
                        BOOL_t reorder=False,
                        BOOL_t compress=False):
    cdef:
        INDEX_t dim = mesh.dim
        local_matrix_t local_matrix
        DoFMap dmS

    if isinstance(volumeDoFMap, P1_DoFMap):
        if dim == 1:
            local_matrix = mass_0d_in_1d_sym_P1()
        elif dim == 2:
            local_matrix = mass_1d_in_2d_sym_P1()
        elif dim == 3:
            local_matrix = mass_2d_in_3d_sym_P1()
        else:
            raise NotImplementedError()
    elif isinstance(volumeDoFMap, P2_DoFMap):
        if dim == 2:
            local_matrix = mass_1d_in_2d_sym_P2()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    dmS = getSurfaceDoFMap(mesh, surface, volumeDoFMap)

    if A is None:
        A = volumeDoFMap.buildSparsityPattern(mesh.cells,
                                              0,
                                              mesh.num_cells,
                                              symmetric=sss_format,
                                              reorder=reorder)

    A = assembleMatrix(surface,
                          dmS,
                          local_matrix,
                          A=A,
                          sss_format=sss_format,
                          reorder=reorder)

    if compress:
        A.eliminate_zeros()

    return A


def assembleMassNonSym(meshBase mesh,
                       DoFMap DoFMap1,
                       DoFMap DoFMap2,
                       LinearOperator A=None,
                       INDEX_t start_idx=-1,
                       INDEX_t end_idx=-1):
    cdef:
        INDEX_t dim = mesh.dim
        local_matrix_t local_matrix
        BOOL_t symLocalMatrix
    assert DoFMap1.mesh == DoFMap2.mesh
    if isinstance(DoFMap1, P0_DoFMap):
        if isinstance(DoFMap2, P0_DoFMap):
            symLocalMatrix = True
            if dim == 1:
                local_matrix = mass_1d_sym_P0()
            elif dim == 2:
                local_matrix = mass_2d_sym_P0()
            elif dim == 3:
                local_matrix = mass_3d_sym_P0()
            else:
                raise NotImplementedError()
        elif isinstance(DoFMap2, P1_DoFMap):
            symLocalMatrix = False
            if dim == 1:
                local_matrix = mass_1d_nonsym_P0_P1()
            elif dim == 2:
                local_matrix = mass_2d_nonsym_P0_P1()
            elif dim == 3:
                local_matrix = mass_3d_nonsym_P0_P1()
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    elif isinstance(DoFMap1, P1_DoFMap):
        if isinstance(DoFMap2, P1_DoFMap):
            symLocalMatrix = True
            if dim == 1:
                local_matrix = mass_1d_sym_P1()
            elif dim == 2:
                local_matrix = mass_2d_sym_P1()
            elif dim == 3:
                local_matrix = mass_3d_sym_P1()
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    # elif isinstance(DoFMap1, P1_DoFMap):
    #     if isinstance(DoFMap2, P0_DoFMap):
    #         if dim == 1:
    #             local_matrix = mass_1d_nonsym_P0_P1()
    #         elif dim == 2:
    #             local_matrix = mass_2d_nonsym_P0_P1()
    #         else:
    #             raise NotImplementedError()
    #     else:
    #         raise NotImplementedError()
    else:
        raise NotImplementedError()
    return assembleNonSymMatrix_CSR(mesh,
                                    local_matrix,
                                    DoFMap1,
                                    DoFMap2,
                                    A,
                                    start_idx,
                                    end_idx,
                                    cellIndices=None,
                                    symLocalMatrix=symLocalMatrix)


def assembleDrift(DoFMap dm,
                  vectorFunction coeff,
                  LinearOperator A=None,
                  INDEX_t start_idx=-1,
                  INDEX_t end_idx=-1,
                  INDEX_t[::1] cellIndices=None):
    cdef:
        INDEX_t dim = dm.mesh.dim
        local_matrix_t local_matrix
    if isinstance(dm, P1_DoFMap):
        if dim == 1:
            local_matrix = drift_1d_P1(coeff)
        elif dim == 2:
            local_matrix = drift_2d_P1(coeff)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return assembleNonSymMatrix_CSR(dm.mesh,
                                    local_matrix,
                                    dm,
                                    dm,
                                    A,
                                    start_idx,
                                    end_idx,
                                    cellIndices=cellIndices)


def assembleStiffness(DoFMap dm,
                      vector_t boundary_data=None,
                      vector_t rhs_contribution=None,
                      LinearOperator A=None,
                      INDEX_t start_idx=-1,
                      INDEX_t end_idx=-1,
                      BOOL_t sss_format=False,
                      BOOL_t reorder=False,
                      diffusivity=None,
                      INDEX_t[::1] cellIndices=None,
                      DoFMap dm2=None):
    cdef:
        INDEX_t dim = dm.mesh.dim
        local_matrix_t local_matrix
    if diffusivity is None:
        if isinstance(dm, P1_DoFMap):
            if dim == 1:
                local_matrix = stiffness_1d_sym_P1()
            elif dim == 2:
                local_matrix = stiffness_2d_sym_P1()
            elif dim == 3:
                local_matrix = stiffness_3d_sym_P1()
            else:
                raise NotImplementedError()
        elif isinstance(dm, P2_DoFMap):
            if dim == 1:
                local_matrix = stiffness_1d_sym_P2()
            elif dim == 2:
                local_matrix = stiffness_2d_sym_P2()
            elif dim == 3:
                local_matrix = stiffness_3d_sym_P2()
            else:
                raise NotImplementedError()
        elif isinstance(dm, P3_DoFMap):
            if dim == 1:
                local_matrix = stiffness_1d_sym_P3()
            elif dim == 2:
                local_matrix = stiffness_2d_sym_P3()
            elif dim == 3:
                local_matrix = stiffness_3d_sym_P3()
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    elif isinstance(diffusivity, matrixFunction):
        if isinstance(dm, P1_DoFMap):
            if dim == 1:
                local_matrix = scalar_coefficient_stiffness_1d_sym_P1(diffusivity[(0, 0)])
            elif dim == 2:
                local_matrix = stiffness_2d_sym_anisotropic3_P1(diffusivity)
            else:
                raise NotImplementedError()
        elif isinstance(dm, P2_DoFMap):
            if dim == 1:
                local_matrix = scalar_coefficient_stiffness_1d_sym_P2(diffusivity[(0, 0)])
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    else:
        if isinstance(dm, P1_DoFMap):
            if dim == 1:
                local_matrix = scalar_coefficient_stiffness_1d_sym_P1(diffusivity)
            elif dim == 2:
                local_matrix = scalar_coefficient_stiffness_2d_sym_P1(diffusivity)
            elif dim == 3:
                local_matrix = scalar_coefficient_stiffness_3d_sym_P1(diffusivity)
            else:
                raise NotImplementedError()
        elif isinstance(dm, P2_DoFMap):
            if dim == 1:
                local_matrix = scalar_coefficient_stiffness_1d_sym_P2(diffusivity)
            elif dim == 2:
                local_matrix = scalar_coefficient_stiffness_2d_sym_P2(diffusivity)
            elif dim == 3:
                local_matrix = scalar_coefficient_stiffness_3d_sym_P2(diffusivity)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    if dm2 is None:
        return assembleMatrix(dm.mesh,
                              dm,
                              local_matrix,
                              boundary_data,
                              rhs_contribution,
                              A,
                              start_idx,
                              end_idx,
                              sss_format,
                              reorder,
                              cellIndices=cellIndices)
    else:
        return assembleNonSymMatrix_CSR(dm.mesh,
                                        local_matrix,
                                        dm,
                                        dm2,
                                        A,
                                        start_idx,
                                        end_idx,
                                        cellIndices=cellIndices,
                                        symLocalMatrix=True)




def assembleMatrix(meshBase mesh,
                   DoFMap DoFMap,
                   local_matrix_t local_matrix,
                   vector_t boundary_data=None,
                   vector_t rhs_contribution=None,
                   LinearOperator A=None,
                   INDEX_t start_idx=-1,
                   INDEX_t end_idx=-1,
                   BOOL_t sss_format=False,
                   BOOL_t reorder=False,
                   INDEX_t[::1] cellIndices=None):
    if A is not None:
        sss_format = isinstance(A, SSS_LinearOperator)
        reorder = False
    if boundary_data is not None and rhs_contribution is None:
        rhs_contribution = np.zeros((DoFMap.num_dofs), dtype=REAL)
        if sss_format:
            return assembleSymMatrix_SSS(mesh,
                                         local_matrix, DoFMap,
                                         boundary_data,
                                         rhs_contribution,
                                         A,
                                         start_idx, end_idx,
                                         reorder=reorder), rhs_contribution
        else:
            return assembleSymMatrix_CSR(mesh,
                                         local_matrix, DoFMap,
                                         boundary_data,
                                         rhs_contribution,
                                         A,
                                         start_idx, end_idx,
                                         reorder=reorder,
                                         cellIndices=cellIndices), rhs_contribution
    else:
        if sss_format:
            return assembleSymMatrix_SSS(mesh,
                                         local_matrix, DoFMap,
                                         boundary_data, rhs_contribution,
                                         A,
                                         start_idx, end_idx,
                                         reorder=reorder)
        else:
            return assembleSymMatrix_CSR(mesh,
                                         local_matrix, DoFMap,
                                         boundary_data, rhs_contribution,
                                         A,
                                         start_idx, end_idx,
                                         reorder=reorder,
                                         cellIndices=cellIndices)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef assembleSymMatrix_CSR(meshBase mesh,
                           local_matrix_t local_matrix,
                           DoFMap DoFMap,
                           vector_t boundary_data=None,
                           vector_t rhs_contribution=None,
                           LinearOperator A=None,
                           INDEX_t start_idx=-1,
                           INDEX_t end_idx=-1,
                           BOOL_t reorder=False,
                           INDEX_t[::1] cellIndices=None):
    cdef:
        INDEX_t i, j, I, J, k, s
        REAL_t[:, ::1] local_vertices = uninitialized((mesh.manifold_dim+1,
                                                    mesh.dim), dtype=REAL)
        # local matrix entries
        REAL_t[::1] local_contrib = uninitialized((DoFMap.dofs_per_element *
                                                (DoFMap.dofs_per_element+1))//2,
                                               dtype=REAL)

    if start_idx == -1:
        start_idx = 0
    if end_idx == -1:
        end_idx = mesh.num_cells

    if A is None:
        A = DoFMap.buildSparsityPattern(mesh.cells,
                                        start_idx,
                                        end_idx,
                                        reorder=reorder)

    if boundary_data.shape[0] == 0:
        if cellIndices is None:
            for i in range(start_idx, end_idx):
                # Get local vertices
                mesh.getSimplex(i, local_vertices)

                if local_matrix.needsCellInfo:
                    local_matrix.setCell(mesh.cells[i, :])

                # Get symmetric local matrix
                local_matrix.eval(local_vertices, local_contrib)

                s = 0
                # enter the data into CSR matrix
                for j in range(DoFMap.dofs_per_element):
                    I = DoFMap.cell2dof(i, j)
                    if I < 0:
                        s += DoFMap.dofs_per_element-j
                        continue
                    for k in range(j, DoFMap.dofs_per_element):
                        J = DoFMap.cell2dof(i, k)
                        if J < 0:
                            s += 1
                            continue
                        if I == J:
                            A.addToEntry(I, I, local_contrib[s])
                        else:
                            A.addToEntry(I, J, local_contrib[s])
                            A.addToEntry(J, I, local_contrib[s])
                        s += 1
        else:
            for i in cellIndices:
                # Get local vertices
                mesh.getSimplex(i, local_vertices)

                # Get symmetric local matrix
                local_matrix.eval(local_vertices, local_contrib)

                s = 0
                # enter the data into CSR matrix
                for j in range(DoFMap.dofs_per_element):
                    I = DoFMap.cell2dof(i, j)
                    if I < 0:
                        s += DoFMap.dofs_per_element-j
                        continue
                    for k in range(j, DoFMap.dofs_per_element):
                        J = DoFMap.cell2dof(i, k)
                        if J < 0:
                            s += 1
                            continue
                        if I == J:
                            A.addToEntry(I, I, local_contrib[s])
                        else:
                            A.addToEntry(I, J, local_contrib[s])
                            A.addToEntry(J, I, local_contrib[s])
                        s += 1
    else:
        for i in range(start_idx, end_idx):
            # Get local vertices
            mesh.getSimplex(i, local_vertices)

            # Get symmetric local matrix
            local_matrix.eval(local_vertices, local_contrib)

            s = 0
            # enter the data into CSR matrix
            for j in range(DoFMap.dofs_per_element):
                I = DoFMap.cell2dof(i, j)
                for k in range(j, DoFMap.dofs_per_element):
                    J = DoFMap.cell2dof(i, k)
                    # write this in a better way
                    if I >= 0:
                        if J >= 0:
                            if I == J:
                                A.addToEntry(I, I, local_contrib[s])
                            else:
                                A.addToEntry(I, J, local_contrib[s])
                                A.addToEntry(J, I, local_contrib[s])
                        else:
                            rhs_contribution[I] -= local_contrib[s]*boundary_data[-J-1]
                    else:
                        if J >= 0:
                            rhs_contribution[J] -= local_contrib[s]*boundary_data[-I-1]
                    s += 1
    return A


@cython.boundscheck(False)
@cython.wraparound(False)
cdef assembleSymMatrix_SSS(meshBase mesh,
                           local_matrix_t local_matrix,
                           DoFMap DoFMap,
                           vector_t boundary_data=None,
                           vector_t rhs_contribution=None,
                           LinearOperator A=None,
                           INDEX_t start_idx=-1,
                           INDEX_t end_idx=-1,
                           BOOL_t reorder=False):
    cdef:
        INDEX_t i, j, I, J, k, s
        REAL_t[:, ::1] local_vertices = uninitialized((mesh.manifold_dim+1,
                                                    mesh.dim), dtype=REAL)
        # local matrix entries
        REAL_t[::1] local_contrib = uninitialized((DoFMap.dofs_per_element *
                                                (DoFMap.dofs_per_element+1))//2,
                                               dtype=REAL)

    if start_idx == -1:
        start_idx = 0
    if end_idx == -1:
        end_idx = mesh.num_cells

    if A is None:
        A = DoFMap.buildSparsityPattern(mesh.cells,
                                        start_idx,
                                        end_idx,
                                        symmetric=True,
                                        reorder=reorder)
    if boundary_data.shape[0] == 0:
        for i in range(start_idx, end_idx):
            # Get local vertices
            mesh.getSimplex(i, local_vertices)

            # Get symmetric local matrix
            local_matrix.eval(local_vertices, local_contrib)

            s = 0
            # enter the data into SSS matrix
            for j in range(DoFMap.dofs_per_element):
                I = DoFMap.cell2dof(i, j)
                if I < 0:
                    s += DoFMap.dofs_per_element-j
                    continue
                A.addToEntry(I, I, local_contrib[s])
                s += 1
                for k in range(j+1, DoFMap.dofs_per_element):
                    J = DoFMap.cell2dof(i, k)
                    if J < 0:
                        s += 1
                        continue
                    if I < J:
                        A.addToEntry(J, I, local_contrib[s])
                    else:
                        A.addToEntry(I, J, local_contrib[s])
                    s += 1
    else:
        for i in range(start_idx, end_idx):
            # Get local vertices
            mesh.getSimplex(i, local_vertices)

            # Get symmetric local matrix
            local_matrix.eval(local_vertices, local_contrib)

            s = 0
            # enter the data into SSS matrix
            for j in range(DoFMap.dofs_per_element):
                I = DoFMap.cell2dof(i, j)
                for k in range(j, DoFMap.dofs_per_element):
                    J = DoFMap.cell2dof(i, k)
                    # write this in a better way
                    if I >= 0:
                        if J >= 0:
                            if I == J:
                                A.addToEntry(I, I, local_contrib[s])
                            else:
                                if I < J:
                                    A.addToEntry(J, I, local_contrib[s])
                                else:
                                    A.addToEntry(I, J, local_contrib[s])
                        else:
                            rhs_contribution[I] -= local_contrib[s]*boundary_data[-J-1]
                    else:
                        if J >= 0:
                            rhs_contribution[J] -= local_contrib[s]*boundary_data[-I-1]
                    s += 1
    return A


@cython.boundscheck(False)
@cython.wraparound(False)
def assembleNonSymMatrix_CSR(meshBase mesh,
                             local_matrix_t local_matrix,
                             DoFMap DoFMap1,
                             DoFMap DoFMap2,
                             CSR_LinearOperator A=None,
                             INDEX_t start_idx=-1,
                             INDEX_t end_idx=-1,
                             INDEX_t[::1] cellIndices=None,
                             BOOL_t symLocalMatrix=False):
    cdef:
        INDEX_t i, j, I, J, k, s
        REAL_t[:, ::1] simplex = uninitialized((mesh.manifold_dim+1,
                                                  mesh.dim), dtype=REAL)
        # local matrix entries
        REAL_t[::1] contrib = uninitialized((DoFMap1.dofs_per_element *
                                               DoFMap2.dofs_per_element),
                                              dtype=REAL)
        REAL_t[::1] contribSym

    if start_idx == -1:
        start_idx = 0
    if end_idx == -1:
        end_idx = mesh.num_cells
    if symLocalMatrix:
        contribSym = uninitialized((DoFMap1.dofs_per_element *
                                    DoFMap2.dofs_per_element),
                                   dtype=REAL)

    if A is None:
        A = DoFMap1.buildNonSymmetricSparsityPattern(mesh.cells,
                                                     DoFMap2,
                                                     start_idx,
                                                     end_idx)

    if local_matrix.additiveAssembly:
        if cellIndices is None:
            for i in range(start_idx, end_idx):
                # Get local vertices
                mesh.getSimplex(i, simplex)

                if local_matrix.needsCellInfo:
                    local_matrix.setCell(mesh.cells[i, :])

                # Evaluate nonsymmetric local matrix
                if symLocalMatrix:
                    local_matrix.eval(simplex, contribSym)
                    s = 0
                    for j in range(DoFMap1.dofs_per_element):
                        for k in range(j, DoFMap2.dofs_per_element):
                            contrib[j*DoFMap1.dofs_per_element+k] = contribSym[s]
                            contrib[k*DoFMap1.dofs_per_element+j] = contribSym[s]
                            s += 1
                else:
                    local_matrix.eval(simplex, contrib)

                s = 0
                # enter the data into CSR matrix
                for j in range(DoFMap1.dofs_per_element):
                    I = DoFMap1.cell2dof(i, j)
                    if I < 0:
                        s += DoFMap2.dofs_per_element
                        continue
                    for k in range(DoFMap2.dofs_per_element):
                        J = DoFMap2.cell2dof(i, k)
                        if J < 0:
                            s += 1
                            continue
                        A.addToEntry(I, J, contrib[s])
                        s += 1
        else:
            for i in cellIndices:
                # Get local vertices
                mesh.getSimplex(i, simplex)

                if local_matrix.needsCellInfo:
                    local_matrix.setCell(mesh.cells[i, :])

                # Evaluate nonsymmetric local matrix
                if symLocalMatrix:
                    local_matrix.eval(simplex, contribSym)
                    s = 0
                    for j in range(DoFMap1.dofs_per_element):
                        for k in range(j, DoFMap2.dofs_per_element):
                            contrib[j*DoFMap1.dofs_per_element+k] = contribSym[s]
                            contrib[k*DoFMap1.dofs_per_element+j] = contribSym[s]
                            s += 1
                else:
                    local_matrix.eval(simplex, contrib)

                s = 0
                # enter the data into CSR matrix
                for j in range(DoFMap1.dofs_per_element):
                    I = DoFMap1.cell2dof(i, j)
                    if I < 0:
                        s += DoFMap2.dofs_per_element
                        continue
                    for k in range(DoFMap2.dofs_per_element):
                        J = DoFMap2.cell2dof(i, k)
                        if J < 0:
                            s += 1
                            continue
                        A.addToEntry(I, J, contrib[s])
                        s += 1
    else:
        if cellIndices is None:
            for i in range(start_idx, end_idx):
                # Get local vertices
                mesh.getSimplex(i, simplex)

                if local_matrix.needsCellInfo:
                    local_matrix.setCell(mesh.cells[i, :])

                # Evaluate nonsymmetric local matrix
                if symLocalMatrix:
                    local_matrix.eval(simplex, contribSym)
                    s = 0
                    for j in range(DoFMap1.dofs_per_element):
                        for k in range(j, DoFMap2.dofs_per_element):
                            contrib[j*DoFMap1.dofs_per_element+k] = contribSym[s]
                            contrib[k*DoFMap1.dofs_per_element+j] = contribSym[s]
                            s += 1
                else:
                    local_matrix.eval(simplex, contrib)

                s = 0
                # enter the data into CSR matrix
                for j in range(DoFMap1.dofs_per_element):
                    I = DoFMap1.cell2dof(i, j)
                    if I < 0:
                        s += DoFMap2.dofs_per_element
                        continue
                    for k in range(DoFMap2.dofs_per_element):
                        J = DoFMap2.cell2dof(i, k)
                        if J < 0:
                            s += 1
                            continue
                        A.setEntry(I, J, contrib[s])
                        s += 1
        else:
            for i in cellIndices:
                # Get local vertices
                mesh.getSimplex(i, simplex)

                if local_matrix.needsCellInfo:
                    local_matrix.setCell(mesh.cells[i, :])

                # Evaluate nonsymmetric local matrix
                if symLocalMatrix:
                    local_matrix.eval(simplex, contribSym)
                    s = 0
                    for j in range(DoFMap1.dofs_per_element):
                        for k in range(j, DoFMap2.dofs_per_element):
                            contrib[j*DoFMap1.dofs_per_element+k] = contribSym[s]
                            contrib[k*DoFMap1.dofs_per_element+j] = contribSym[s]
                            s += 1
                else:
                    local_matrix.eval(simplex, contrib)

                s = 0
                # enter the data into CSR matrix
                for j in range(DoFMap1.dofs_per_element):
                    I = DoFMap1.cell2dof(i, j)
                    if I < 0:
                        s += DoFMap2.dofs_per_element
                        continue
                    for k in range(DoFMap2.dofs_per_element):
                        J = DoFMap2.cell2dof(i, k)
                        if J < 0:
                            s += 1
                            continue
                        A.setEntry(I, J, contrib[s])
                        s += 1
    return A


ctypedef fused FUNCTION_t:
    function
    vectorFunction


@cython.boundscheck(False)
@cython.cdivision(True)
def assembleRHS(FUNCTION_t fun, DoFMap dm,
                simplexQuadratureRule qr=None):
    cdef:
        meshBase mesh = dm.mesh
        INDEX_t dim = mesh.dim
        INDEX_t dimManifold = mesh.manifold_dim
        INDEX_t num_vertices = dimManifold+1
        INDEX_t num_quad_nodes
        REAL_t[:, ::1] PHI
        REAL_t[:, :, ::1] PHIVector
        REAL_t[::1] weights
        INDEX_t i, k, j, l, I
        fe_vector dataVec
        vector_t data
        REAL_t vol
        REAL_t[:, ::1] span = uninitialized((mesh.manifold_dim, mesh.dim), dtype=REAL)
        REAL_t[:, ::1] simplex = uninitialized((mesh.manifold_dim+1, mesh.dim),
                                                 dtype=REAL)
        volume_t volume
        simplexComputations sC
        vectorShapeFunction phi
        REAL_t[::1] fvals
        REAL_t[:, ::1] fvalsVector
        INDEX_t fun_rows

    if qr is None:
        if dim == dimManifold:
            if dimManifold == 1:
                if isinstance(dm, P0_DoFMap):
                    qr = Gauss1D(order=3)
                elif isinstance(dm, P1_DoFMap):
                    qr = Gauss1D(order=3)
                elif isinstance(dm, P2_DoFMap):
                    qr = Gauss1D(order=5)
                volume = volume1Dnew
            elif dimManifold == 2:
                if isinstance(dm, P0_DoFMap):
                    qr = Gauss2D(order=2)
                elif isinstance(dm, P1_DoFMap):
                    qr = Gauss2D(order=2)
                elif isinstance(dm, P2_DoFMap):
                    qr = Gauss2D(order=5)
                volume = volume2Dnew
            elif dimManifold == 3:
                if isinstance(dm, P1_DoFMap):
                    qr = Gauss3D(order=3)
                elif isinstance(dm, P2_DoFMap):
                    qr = Gauss3D(order=3)
                volume = volume3D
            else:
                raise NotImplementedError()
        if qr is None:
            qr = simplexXiaoGimbutas(2*dm.polynomialOrder+2, dim, dimManifold)
            volume = qr.volume
    else:
        volume = qr.volume

    dataVec = dm.zeros()
    data = dataVec
    weights = qr.weights
    num_quad_nodes = qr.num_nodes

    if FUNCTION_t is function:
        # evaluate local shape functions on quadrature nodes
        PHI = uninitialized((dm.dofs_per_element, qr.num_nodes), dtype=REAL)
        for i in range(dm.dofs_per_element):
            for j in range(qr.num_nodes):
                PHI[i, j] = dm.localShapeFunctions[i](np.ascontiguousarray(qr.nodes[:, j]))

        fvals = uninitialized((num_quad_nodes), dtype=REAL)

        for i in range(mesh.num_cells):
            # Get local vertices
            mesh.getSimplex(i, simplex)

            # Calculate volume
            for k in range(num_vertices-1):
                for j in range(dim):
                    span[k, j] = simplex[k+1, j]-simplex[0, j]
            vol = volume(span)

            # Get function values at quadrature nodes
            qr.evalFun(fun, simplex, fvals)

            # Put everything together
            for k in range(dm.dofs_per_element):
                I = dm.cell2dof(i, k)
                if I < 0:
                    continue
                for j in range(num_quad_nodes):
                    data[I] += vol*weights[j]*fvals[j]*PHI[k, j]
    else:
        fun_rows = fun.rows
        PHIVector = uninitialized((dm.dofs_per_element, qr.num_nodes, fun_rows), dtype=REAL)
        gradients = uninitialized((mesh.dim+1, mesh.dim), dtype=REAL)

        fvalsVector = uninitialized((num_quad_nodes, fun_rows), dtype=REAL)

        if dim == 1:
            sC = simplexComputations1D()
        elif dim == 2:
            sC = simplexComputations2D()
        elif dim == 3:
            sC = simplexComputations3D()
        else:
            raise NotImplementedError()
        sC.setSimplex(simplex)

        phi = dm.localShapeFunctions[0]
        if phi.needsGradients:

            for i in range(mesh.num_cells):
                # Get local vertices
                mesh.getSimplex(i, simplex)

                # Calculate volume
                vol = sC.evalVolumeGradients(gradients)

                # evaluate local shape functions on quadrature nodes
                for k in range(dm.dofs_per_element):
                    phi = dm.localShapeFunctions[k]
                    phi.setCell(mesh.cells[i, :])
                    for j in range(num_quad_nodes):
                        phi.eval(np.ascontiguousarray(qr.nodes[:, j]), gradients, PHIVector[k, j, :])

                # Get function values at quadrature nodes
                qr.evalVectorFun(fun, simplex, fvalsVector)

                # Put everything together
                for k in range(dm.dofs_per_element):
                    I = dm.cell2dof(i, k)
                    if I < 0:
                        continue
                    for j in range(num_quad_nodes):
                        for l in range(fun_rows):
                            data[I] += vol*weights[j]*fvalsVector[j, l]*PHIVector[k, j, l]
        else:

            # evaluate local shape functions on quadrature nodes
            for k in range(dm.dofs_per_element):
                phi = dm.localShapeFunctions[k]
                for j in range(num_quad_nodes):
                    phi.eval(np.ascontiguousarray(qr.nodes[:, j]), gradients, PHIVector[k, j, :])

            for i in range(mesh.num_cells):
                # Get local vertices
                mesh.getSimplex(i, simplex)

                # Calculate volume
                vol = sC.evalVolume()

                # Get function values at quadrature nodes
                qr.evalVectorFun(fun, simplex, fvalsVector)

                # Put everything together
                for k in range(dm.dofs_per_element):
                    I = dm.cell2dof(i, k)
                    if I < 0:
                        continue
                    for j in range(num_quad_nodes):
                        for l in range(fun_rows):
                            data[I] += vol*weights[j]*fvalsVector[j, l]*PHIVector[k, j, l]
    return dataVec


@cython.boundscheck(False)
@cython.cdivision(True)
def assembleRHScomplex(complexFunction fun, DoFMap dm,
                       simplexQuadratureRule qr=None):
    cdef:
        meshBase mesh = dm.mesh
        INDEX_t dim = mesh.dim
        INDEX_t dimManifold = mesh.manifold_dim
        INDEX_t num_vertices = dimManifold+1
        INDEX_t num_quad_nodes
        REAL_t[:, ::1] PHI
        REAL_t[::1] weights
        INDEX_t i, k, j, I
        complex_fe_vector dataVec
        complex_vector_t data
        REAL_t vol
        REAL_t[:, ::1] span = uninitialized((mesh.manifold_dim, mesh.dim), dtype=REAL)
        REAL_t[:, ::1] simplex = uninitialized((mesh.manifold_dim+1, mesh.dim),
                                            dtype=REAL)
        volume_t volume
        COMPLEX_t[::1] fvals

    if qr is None:
        if dim == dimManifold:
            if dimManifold == 1:
                if isinstance(dm, P0_DoFMap):
                    qr = Gauss1D(order=3)
                elif isinstance(dm, P1_DoFMap):
                    qr = Gauss1D(order=3)
                elif isinstance(dm, P2_DoFMap):
                    qr = Gauss1D(order=5)
                volume = volume1Dnew
            elif dimManifold == 2:
                if isinstance(dm, P0_DoFMap):
                    qr = Gauss2D(order=2)
                elif isinstance(dm, P1_DoFMap):
                    qr = Gauss2D(order=2)
                elif isinstance(dm, P2_DoFMap):
                    qr = Gauss2D(order=5)
                volume = volume2Dnew
            elif dimManifold == 3:
                if isinstance(dm, P1_DoFMap):
                    qr = Gauss3D(order=3)
                elif isinstance(dm, P2_DoFMap):
                    qr = Gauss3D(order=3)
                volume = volume3D
            else:
                raise NotImplementedError()
        if qr is None:
            qr = simplexXiaoGimbutas(2*dm.polynomialOrder+2, dim, dimManifold)
            volume = qr.volume
    else:
        volume = qr.volume

    # evaluate local shape functions on quadrature nodes
    PHI = uninitialized((dm.dofs_per_element, qr.num_nodes), dtype=REAL)
    for i in range(dm.dofs_per_element):
        for j in range(qr.num_nodes):
            PHI[i, j] = dm.localShapeFunctions[i](np.ascontiguousarray(qr.nodes[:, j]))
    weights = qr.weights

    num_quad_nodes = qr.num_nodes

    dataVec = dm.zeros(dtype=COMPLEX)
    data = dataVec
    fvals = uninitialized((num_quad_nodes), dtype=COMPLEX)

    for i in range(mesh.num_cells):
        # Get local vertices
        mesh.getSimplex(i, simplex)

        # Calculate volume
        for k in range(num_vertices-1):
            for j in range(dim):
                span[k, j] = simplex[k+1, j]-simplex[0, j]
        vol = volume(span)

        # Get function values at quadrature nodes
        qr.evalComplexFun(fun, simplex, fvals)

        # Put everything together
        for k in range(dm.dofs_per_element):
            I = dm.cell2dof(i, k)
            if I < 0:
                continue
            for j in range(num_quad_nodes):
                data[I] = data[I] + vol*weights[j]*fvals[j]*PHI[k, j]
    return dataVec


@cython.boundscheck(False)
@cython.cdivision(True)
def assembleRHSgrad(FUNCTION_t fun, DoFMap dm,
                    vectorFunction coeff,
                    simplexQuadratureRule qr=None):
    # assembles (f, \grad v)
    cdef:
        meshBase mesh = dm.mesh
        INDEX_t dim = mesh.dim
        INDEX_t dimManifold = mesh.manifold_dim
        INDEX_t num_vertices = dimManifold+1
        INDEX_t num_quad_nodes
        REAL_t[:, ::1] PHI
        REAL_t[:, :, ::1] PHIVector
        REAL_t[::1] weights
        INDEX_t i, k, j, l, I
        fe_vector dataVec
        vector_t data
        REAL_t vol
        REAL_t[:, ::1] span = uninitialized((mesh.manifold_dim, mesh.dim), dtype=REAL)
        REAL_t[:, ::1] simplex = uninitialized((mesh.manifold_dim+1, mesh.dim),
                                                 dtype=REAL)
        volume_t volume
        vectorShapeFunction phi
        REAL_t[::1] fvals
        REAL_t[:, ::1] fvalsVector

    assert isinstance(dm, P1_DoFMap)
    assert dim == 1 or dim == 2

    if qr is None:
        if dim == dimManifold:
            if dimManifold == 1:
                if isinstance(dm, P0_DoFMap):
                    qr = Gauss1D(order=3)
                elif isinstance(dm, P1_DoFMap):
                    qr = Gauss1D(order=3)
                elif isinstance(dm, P2_DoFMap):
                    qr = Gauss1D(order=5)
                volume = volume1Dnew
            elif dimManifold == 2:
                if isinstance(dm, P0_DoFMap):
                    qr = Gauss2D(order=2)
                elif isinstance(dm, P1_DoFMap):
                    qr = Gauss2D(order=2)
                elif isinstance(dm, P2_DoFMap):
                    qr = Gauss2D(order=5)
                volume = volume2Dnew
            elif dimManifold == 3:
                if isinstance(dm, P1_DoFMap):
                    qr = Gauss3D(order=3)
                elif isinstance(dm, P2_DoFMap):
                    qr = Gauss3D(order=3)
                volume = volume3D
            else:
                raise NotImplementedError()
        if qr is None:
            qr = simplexXiaoGimbutas(2*dm.polynomialOrder+2, dim, dimManifold)
            volume = qr.volume
    else:
        volume = qr.volume

    dataVec = dm.zeros()
    data = dataVec
    weights = qr.weights
    num_quad_nodes = qr.num_nodes

    if FUNCTION_t is function:
        fvals = uninitialized((num_quad_nodes), dtype=REAL)
        temp = uninitialized((4, 2), dtype=REAL)
        innerProducts = uninitialized((3), dtype=REAL)

        for i in range(mesh.num_cells):
            # Get local vertices
            mesh.getSimplex(i, simplex)

            # Calculate volume
            for k in range(num_vertices-1):
                for j in range(dim):
                    span[k, j] = simplex[k+1, j]-simplex[0, j]
            vol = volume(span)

            if dim == 1:
                coeffProducts1D(simplex, vol, coeff, innerProducts, temp)
            elif dim == 2:
                coeffProducts2D(simplex, vol, coeff, innerProducts, temp)

            # Get function values at quadrature nodes
            qr.evalFun(fun, simplex, fvals)

            # Put everything together
            for k in range(dm.dofs_per_element):
                I = dm.cell2dof(i, k)
                if I < 0:
                    continue
                for j in range(num_quad_nodes):
                    data[I] += vol*weights[j]*fvals[j]*innerProducts[k]
    else:
        raise NotImplementedError()
    return dataVec



cdef class multi_function:
    def __init__(self, numInputs, numOutputs):
        self.numInputs = numInputs
        self.numOutputs = numOutputs

    def __call__(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.eval(x, y)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y):
        pass


cdef class power(multi_function):
    cdef:
        REAL_t k

    def __init__(self, k=2.):
        self.k = k
        multi_function.__init__(self, 1, 1)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t u
        u = x[0]
        y[0] = u**self.k


cdef class gray_scott(multi_function):
    cdef:
        REAL_t F, k

    def __init__(self, F=0.025, k=0.06):
        self.F = F
        self.k = k
        multi_function.__init__(self, 2, 2)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef inline void eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t u, v
        u = x[0]
        v = x[1]
        y[0] = -u*v**2 + self.F*(1.-u)
        y[1] = u*v**2 - (self.F+self.k)*v


cdef class gray_scott_gradient(multi_function):
    cdef:
        REAL_t F, k

    def __init__(self, F=0.025, k=0.06):
        self.F = F
        self.k = k
        multi_function.__init__(self, 4, 2)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef inline void eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef REAL_t u, v, unew, vnew
        u = x[0]
        v = x[1]
        unew = x[2]
        vnew = x[3]
        y[0] = -unew*v**2 - 2.*u*v*vnew - self.F*unew
        y[1] = unew*v**2 + 2.*u*v*vnew - (self.k+self.F)*vnew


cdef class brusselator(multi_function):
    cdef:
        REAL_t B, Q

    def __init__(self, B=0.025, Q=0.06):
        self.B = B
        self.Q = Q
        multi_function.__init__(self, 2, 2)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef inline void eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t u, v, z
        u = x[0]
        v = x[1]
        z = self.B*u + self.Q**2*v + self.B/self.Q*u**2 + 2.*self.Q*u*v + u**2*v
        y[0] = -u + z
        y[1] = -z


cdef class CahnHilliard_F_prime(multi_function):
    def __init__(self):
        multi_function.__init__(self, 1, 1)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t u
        u = x[0]
        y[0] = u**3-u


cdef class CahnHilliard_F(multi_function):
    def __init__(self):
        multi_function.__init__(self, 1, 1)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t u
        u = x[0]
        y[0] = 0.25*(1.-u**2)**2


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def assembleNonlinearity(meshBase mesh, multi_function fun, DoFMap DoFMap, multi_fe_vector U):
    cdef:
        INDEX_t dim = mesh.dim
        INDEX_t dimManifold = mesh.manifold_dim
        INDEX_t num_vertices = dimManifold+1
        INDEX_t num_quad_nodes
        REAL_t[:, ::1] PHI
        REAL_t[::1] weights
        INDEX_t num_cells = mesh.cells.shape[0]
        INDEX_t i, k, j, m, I
        # np.ndarray[REAL_t, ndim=1] data_mem
        REAL_t[::1] data
        REAL_t vol
        REAL_t[:, ::1] span = uninitialized((num_vertices-1, dim), dtype=REAL)
        REAL_t[:, ::1] local_vertices = uninitialized((num_vertices, dim),
                                                   dtype=REAL)
        volume_t volume
        REAL_t[:, ::1] fvals, fvals2
        REAL_t[:, ::1] u

    if dimManifold == 1:
        qr = Gauss1D(order=3)
        volume = volume1Dnew
    elif dimManifold == 2:
        if isinstance(DoFMap, P1_DoFMap):
            qr = Gauss2D(order=2)
        elif isinstance(DoFMap, P2_DoFMap):
            qr = Gauss2D(order=5)
        else:
            raise NotImplementedError()
        volume = volume2Dnew
    elif dimManifold == 3:
        if isinstance(DoFMap, P1_DoFMap):
            qr = Gauss3D(order=3)
        elif isinstance(DoFMap, P2_DoFMap):
            qr = Gauss3D(order=3)
        else:
            raise NotImplementedError()
        volume = volume3D
    else:
        raise NotImplementedError()

    assert U.numVectors == fun.numInputs, (U.numVectors, fun.numInputs)

    # evaluate local shape functions on quadrature nodes
    PHI = uninitialized((DoFMap.dofs_per_element, qr.num_nodes), dtype=REAL)
    for i in range(DoFMap.dofs_per_element):
        for j in range(qr.num_nodes):
            PHI[i, j] = DoFMap.localShapeFunctions[i](np.ascontiguousarray(qr.nodes[:, j]))
    weights = qr.weights
    # quad_points = qr.nodes

    num_quad_nodes = qr.num_nodes

    u = U.data

    dataList = multi_fe_vector(np.zeros((fun.numOutputs, DoFMap.num_dofs), dtype=REAL), DoFMap)
    # data = data_mem
    fvals = uninitialized((num_quad_nodes, fun.numInputs), dtype=REAL)
    fvals2 = uninitialized((num_quad_nodes, fun.numOutputs), dtype=REAL)

    for i in range(num_cells):
        # Get local vertices
        mesh.getSimplex(i, local_vertices)

        # Calculate volume
        for k in range(num_vertices-1):
            for j in range(dim):
                span[k, j] = local_vertices[k+1, j]-local_vertices[0, j]
        vol = volume(span)

        fvals[:] = 0.
        for m in range(DoFMap.dofs_per_element):
            I = DoFMap.cell2dof(i, m)
            if I >= 0:
                for k in range(num_quad_nodes):
                    for j in range(fun.numInputs):
                        # u = U[j]
                        fvals[k, j] += u[j, I]*PHI[m, k]
        for k in range(num_quad_nodes):
            fun.eval(fvals[k, :], fvals2[k, :])

        # Put everything together
        for m in range(fun.numOutputs):
            data = dataList[m]
            for k in range(DoFMap.dofs_per_element):
                I = DoFMap.cell2dof(i, k)
                if I >= 0:
                    for j in range(num_quad_nodes):
                        data[I] += vol*weights[j]*fvals2[j, m]*PHI[k, j]
                        # data[m*DoFMap.num_dofs+I] += vol*weights[j]*fvals2[j, m]*PHI[k, j]
    return dataList
    # return data_mem


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def assembleRHSfromFEfunction(meshBase mesh,
                              vector_t u,
                              DoFMap DoFMap,
                              DoFMap target,
                              simplexQuadratureRule qr=None):
    cdef:
        INDEX_t dim = mesh.dim
        INDEX_t dimManifold = mesh.manifold_dim
        INDEX_t num_vertices = dimManifold+1
        INDEX_t num_quad_nodes
        REAL_t[:, ::1] PHI, PHItarget
        REAL_t[::1] weights
        INDEX_t num_cells = mesh.cells.shape[0]
        INDEX_t i, k, j, m, I
        REAL_t vol
        REAL_t[:, ::1] span = uninitialized((num_vertices-1, dim), dtype=REAL)
        REAL_t[:, ::1] local_vertices = uninitialized((num_vertices, dim),
                                                   dtype=REAL)
        volume_t volume
        REAL_t[::1] fvals
        vector_t b

    assert DoFMap.mesh.num_vertices == target.mesh.num_vertices, "DoFmap and target have different meshes"
    assert DoFMap.mesh.num_cells == target.mesh.num_cells, "DoFmap and target have different meshes"
    assert u.shape[0] == DoFMap.num_dofs, "u and DoFMap have different number of DoFs: {} != {}".format(u.shape[0], DoFMap.num_dofs)

    if qr is None:
        if dimManifold == 1:
            qr = Gauss1D(order=3)
            volume = volume1Dnew
        elif dimManifold == 2:
            qr = Gauss2D(order=2)
            volume = volume2Dnew
        elif dimManifold == 3:
            qr = Gauss3D(order=3)
            volume = volume3D
        else:
            raise NotImplementedError()
    else:
        volume = qr.volume

    # evaluate local shape functions on quadrature nodes
    PHI = uninitialized((DoFMap.dofs_per_element, qr.num_nodes), dtype=REAL)
    for i in range(DoFMap.dofs_per_element):
        for j in range(qr.num_nodes):
            PHI[i, j] = DoFMap.localShapeFunctions[i](np.ascontiguousarray(qr.nodes[:, j]))
    weights = qr.weights
    num_quad_nodes = qr.num_nodes

    # evaluate local shape functions on quadrature nodes
    PHItarget = uninitialized((target.dofs_per_element, qr.num_nodes), dtype=REAL)
    for i in range(target.dofs_per_element):
        for j in range(qr.num_nodes):
            PHItarget[i, j] = target.localShapeFunctions[i](np.ascontiguousarray(qr.nodes[:, j]))

    fvals = uninitialized((num_quad_nodes), dtype=REAL)

    b = np.zeros((target.num_dofs), dtype=REAL)

    for i in range(num_cells):
        # Get local vertices
        mesh.getSimplex(i, local_vertices)

        # Calculate volume
        for k in range(num_vertices-1):
            for j in range(dim):
                span[k, j] = local_vertices[k+1, j]-local_vertices[0, j]
        vol = volume(span)

        # get u at quadrature nodes
        fvals[:] = 0.
        for m in range(DoFMap.dofs_per_element):
            I = DoFMap.cell2dof(i, m)
            if I >= 0:
                for k in range(num_quad_nodes):
                    fvals[k] += u[I]*PHI[m, k]

        # Integrate aggainst basis functions in target DoFMap
        for m in range(target.dofs_per_element):
            I = target.cell2dof(i, m)
            if I >= 0:
                for k in range(num_quad_nodes):
                    b[I] += vol*weights[k]*fvals[k]*PHItarget[m, k]
    return np.array(b, copy=False)


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def assembleJumpMatrix(meshBase mesh, P0_DoFMap dm):
    cdef:
        sparsityPattern sPat = sparsityPattern(dm.num_dofs)
        cells_t cells = mesh.cells
        INDEX_t dim = mesh.dim
        dict lookup
        ENCODE_t hv = 0
        INDEX_t hv1 = 0
        tuple hvFace
        INDEX_t cellNo, otherCellNo, vertexNo, I, J
        REAL_t[:, ::1] edgeSimplex = uninitialized((2, 2), dtype=REAL)
        REAL_t[:, ::1] faceSimplex = uninitialized((3, 3), dtype=REAL)
        REAL_t[:, ::1] temp = uninitialized((4, 3), dtype=REAL)
        REAL_t vol2
        CSR_LinearOperator A
        simplexMapper sM

    if dim == 1:
        lookup = {}
        for cellNo in range(mesh.num_cells):
            for vertexNo in range(dim+1):
                #vertex = sm.getVertexInCell(cellNo, vertexNo)
                vertex = cells[cellNo, vertexNo]
                try:
                    otherCellNo = lookup.pop(vertex)
                    I = dm.cell2dof(cellNo, 0)
                    J = dm.cell2dof(otherCellNo, 0)
                    sPat.add(I, I)
                    sPat.add(J, J)
                    sPat.add(I, J)
                    sPat.add(J, I)
                except KeyError:
                    lookup[vertex] = cellNo
        indptr, indices = sPat.freeze()
        del sPat
        nnz = indptr[dm.num_dofs]
        data = np.zeros((nnz), dtype=REAL)
        A = CSR_LinearOperator(indices, indptr, data)
        lookup = {}
        for cellNo in range(mesh.num_cells):
            for vertexNo in range(dim+1):
                #vertex = sm.getVertexInCell(cellNo, vertexNo)
                vertex = mesh.cells[cellNo, vertexNo]
                try:
                    otherCellNo, vol2 = lookup.pop(vertex)
                    I = dm.cell2dof(cellNo, 0)
                    J = dm.cell2dof(otherCellNo, 0)
                    A.addToEntry(I, I, vol2)
                    A.addToEntry(J, J, vol2)
                    A.addToEntry(I, J, -vol2)
                    A.addToEntry(J, I, -vol2)
                except KeyError:
                    vol2 = 1.
                    lookup[vertex] = cellNo, vol2
    elif dim == 2:
        sM = mesh.simplexMapper
        lookup = {}
        for cellNo in range(mesh.num_cells):
            sM.startLoopOverCellEdges(cells[cellNo, :])
            while sM.loopOverCellEdgesEncoded(&hv):
                try:
                    otherCellNo = lookup.pop(hv)
                    I = dm.cell2dof(cellNo, 0)
                    J = dm.cell2dof(otherCellNo, 0)
                    sPat.add(I, I)
                    sPat.add(J, J)
                    sPat.add(I, J)
                    sPat.add(J, I)
                except KeyError:
                    lookup[hv] = cellNo
        indptr, indices = sPat.freeze()
        del sPat
        nnz = indptr[dm.num_dofs]
        data = np.zeros((nnz), dtype=REAL)
        A = CSR_LinearOperator(indices, indptr, data)
        lookup = {}
        for cellNo in range(mesh.num_cells):
            sM.startLoopOverCellEdges(cells[cellNo, :])
            while sM.loopOverCellEdgesEncoded(&hv):
                try:
                    otherCellNo, vol2 = lookup.pop(hv)
                    I = dm.cell2dof(cellNo, 0)
                    J = dm.cell2dof(otherCellNo, 0)
                    A.addToEntry(I, I, vol2)
                    A.addToEntry(J, J, vol2)
                    A.addToEntry(I, J, -vol2)
                    A.addToEntry(J, I, -vol2)
                except KeyError:
                    sM.getEncodedEdgeSimplex(hv, edgeSimplex)
                    vol2 = simplexVolume1Din2D(edgeSimplex, temp)**2
                    lookup[hv] = cellNo, vol2
    elif dim == 3:
        sM = mesh.simplexMapper
        lookup = {}
        for cellNo in range(mesh.num_cells):
            sM.startLoopOverCellFaces(cells[cellNo, :])
            while sM.loopOverCellFacesEncoded(&hv1, &hv):
                hvFace = (hv1, hv)
                try:
                    otherCellNo = lookup.pop(hvFace)
                    I = dm.cell2dof(cellNo, 0)
                    J = dm.cell2dof(otherCellNo, 0)
                    sPat.add(I, I)
                    sPat.add(J, J)
                    sPat.add(I, J)
                    sPat.add(J, I)
                except KeyError:
                    lookup[hvFace] = cellNo
        indptr, indices = sPat.freeze()
        del sPat
        nnz = indptr[dm.num_dofs]
        data = np.zeros((nnz), dtype=REAL)
        A = CSR_LinearOperator(indices, indptr, data)
        lookup = {}
        for cellNo in range(mesh.num_cells):
            sM.startLoopOverCellFaces(cells[cellNo, :])
            while sM.loopOverCellFacesEncoded(&hv1, &hv):
                hvFace = (hv1, hv)
                try:
                    otherCellNo, vol2 = lookup.pop(hvFace)
                    I = dm.cell2dof(cellNo, 0)
                    J = dm.cell2dof(otherCellNo, 0)
                    A.addToEntry(I, I, vol2)
                    A.addToEntry(J, J, vol2)
                    A.addToEntry(I, J, -vol2)
                    A.addToEntry(J, I, -vol2)
                except KeyError:
                    sM.getEncodedFaceSimplex(hvFace, faceSimplex)
                    vol2 = simplexVolume2Din3D(faceSimplex, temp)**2
                    lookup[hvFace] = cellNo, vol2
    else:
        raise NotImplementedError()
    return A


cdef class matrixFreeOperator(LinearOperator):
    cdef:
        meshBase mesh
        DoFMap dm
        local_matrix_t local_matrix

    def __init__(self, meshBase mesh, DoFMap dm, local_matrix_t local_matrix):
        self.mesh = mesh
        self.dm = dm
        self.local_matrix = local_matrix
        LinearOperator.__init__(self,
                                dm.num_dofs,
                                dm.num_dofs)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1:
        cdef:
            INDEX_t i
        for i in range(self.dm.num_dofs):
            y[i] = 0.
        self.matvec_no_overwrite(x, y)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec_no_overwrite(self,
                                     REAL_t[::1] x,
                                     REAL_t[::1] y) except -1:
        cdef:
            INDEX_t i, s, j, k, I, J
            REAL_t[:, ::1] simplex = uninitialized((self.mesh.dim+1, self.mesh.manifold_dim), dtype=REAL)
            REAL_t[::1] local_contrib = uninitialized((self.dm.dofs_per_element *
                                                    (self.dm.dofs_per_element+1))//2,
                                                   dtype=REAL)

        for i in range(self.mesh.num_cells):
            # Get simplex
            self.mesh.getSimplex(i, simplex)
            self.local_matrix.eval(simplex, local_contrib)
            s = 0
            for j in range(self.dm.dofs_per_element):
                I = self.dm.cell2dof(i, j)
                if I < 0:
                    s += self.dm.dofs_per_element-j
                    continue
                for k in range(j, self.dm.dofs_per_element):
                    J = self.dm.cell2dof(i, k)
                    if J < 0:
                        s += 1
                        continue
                    if I == J:
                        y[I] += local_contrib[s] * x[I]
                    else:
                        y[I] += local_contrib[s] * x[J]
                        y[J] += local_contrib[s] * x[I]
                    s += 1

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_diagonal(self):
        cdef:
            INDEX_t i, s, j, k, I, J
            REAL_t[:, ::1] simplex = uninitialized((self.mesh.dim+1, self.mesh.manifold_dim), dtype=REAL)
            REAL_t[::1] local_contrib = uninitialized((self.dm.dofs_per_element *
                                                         (self.dm.dofs_per_element+1))//2,
                                                        dtype=REAL)
            REAL_t[::1] d = np.zeros((self.dm.num_dofs), dtype=REAL)

        for i in range(self.mesh.num_cells):
            # Get simplex
            self.mesh.getSimplex(i, simplex)
            self.local_matrix.eval(simplex, local_contrib)
            s = 0
            for j in range(self.dm.dofs_per_element):
                I = self.dm.cell2dof(i, j)
                if I < 0:
                    s += self.dm.dofs_per_element-j
                    continue
                for k in range(j, self.dm.dofs_per_element):
                    J = self.dm.cell2dof(i, k)
                    if J < 0:
                        s += 1
                        continue
                    if I == J:
                        d[I] += local_contrib[s]
                    s += 1
        return np.array(d, copy=False)

    diagonal = property(fget=get_diagonal)
