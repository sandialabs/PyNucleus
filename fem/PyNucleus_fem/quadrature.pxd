###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t
from . femCy cimport volume_t
from . functions cimport function, vectorFunction, complexFunction
from . meshCy cimport (vectorProduct,
                       volume0D,
                       volume1D, volume1Dnew,
                       volume1D_in_2D,
                       volume2Dnew,
                       volume3D, volume3Dnew,
                       volume2D_in_3Dnew as volume2D_in_3D,
                       meshBase)
cimport numpy as np
from libc.math cimport sqrt

include "config.pxi"


cdef class quadratureRule:
    cdef:
        public REAL_t[:, ::1] nodes  # num_bary x num_nodes
        public REAL_t[::1] weights
        readonly INDEX_t num_nodes
        readonly INDEX_t dim
        readonly INDEX_t manifold_dim

    cdef inline REAL_t eval(self,
                              const REAL_t[::1] fun_vals,
                              const REAL_t vol)


cdef class simplexQuadratureRule(quadratureRule):
    cdef:
        volume_t volume
        REAL_t[:, ::1] span
        REAL_t[::1] tempVec
        public list orders
    cdef inline void nodesInGlobalCoords(self,
                                         const REAL_t[:, ::1] simplexVertices,
                                         const REAL_t[:, ::1] coords)
    cdef inline void nodesInGlobalCoordsReorderer(self,
                                                  const REAL_t[:, ::1] simplexVertices,
                                                  REAL_t[:, ::1] coords,
                                                  const INDEX_t[::1] idx)
    cpdef void evalFun(self,
                       function fun,
                       const REAL_t[:, ::1] simplexVertices,
                       REAL_t[::1] fun_vals)
    cpdef void evalVectorFun(self,
                             vectorFunction fun,
                             const REAL_t[:, ::1] simplexVertices,
                             REAL_t[:, ::1] fun_vals)
    cpdef void evalComplexFun(self,
                              complexFunction fun,
                              const REAL_t[:, ::1] simplexVertices,
                              COMPLEX_t[::1] fun_vals)
    cdef REAL_t getSimplexVolume(self,
                                   const REAL_t[:, ::1] simplexVertices)


cdef class transformQuadratureRule(simplexQuadratureRule):
    cdef:
        simplexQuadratureRule qr
        REAL_t[:, ::1] A
        REAL_t[::1] b
    cpdef void setLinearBaryTransform(self, REAL_t[:, ::1] A)
    cpdef void setAffineBaryTransform(self, REAL_t[:, ::1] A, REAL_t[::1] b)
    cdef void compute(self)


cdef class Gauss1D(simplexQuadratureRule):
    cdef public INDEX_t order


cdef class Gauss2D(simplexQuadratureRule):
    cdef public INDEX_t order


cdef class Gauss3D(simplexQuadratureRule):
    cdef public INDEX_t order


cdef class doubleSimplexQuadratureRule(quadratureRule):
    cdef:
        public simplexQuadratureRule rule1, rule2

    # cdef inline REAL_t eval(self, const REAL_t[::1] fun_vals, const REAL_t vol)
    cpdef void evalFun(self,
                      function fun,
                      const REAL_t[:, ::1] simplexVertices1,
                      const REAL_t[:, ::1] simplexVertices2,
                      REAL_t[::1] fun_vals)


cdef:
    REAL_t[:, ::1] quad_point2D_order2
    REAL_t[::1] weights2D_order2

    REAL_t a1 = (6.0-sqrt(15.0))/21.0, a2 = (6.0+sqrt(15.0))/21.0
    REAL_t c1 = a1*(2.0*a1-1.0), c2 = a2*(2.0*a2-1.0)
    REAL_t d1 = (4.0*a1-1.0)*(2.0*a1-1.0), d2 = (4.0*a2-1.0)*(2.0*a2-1.0)
    REAL_t e1 = 4.0*a1**2, e2 = 4.0*a2**2
    REAL_t f1 = 4.0*a1*(1.0-2.0*a1), f2 = 4.0*a2*(1.0-2.0*a2)
    REAL_t w1 = (155.0-sqrt(15.0))/1200.0, w2 = (155.0+sqrt(15.0))/1200.0
    REAL_t[:, ::1] quad_point2D_order5
    REAL_t[::1] weights2D_order5

    REAL_t[:, ::1] quad_point3D_order3
    REAL_t[::1] weights3D_order3


cdef class quadQuadratureRule(quadratureRule):
    cdef:
        volume_t volume
        public list orders
    # cpdef REAL_t eval(self,
    #                    REAL_t[::1] fun_vals,
    #                    REAL_t vol)
    cpdef void nodesInGlobalCoords(self,
                                   const REAL_t[:, ::1] quadVertices,
                                   REAL_t[:, ::1] coords)
    cpdef void evalFun(self,
                      function fun,
                      const REAL_t[:, ::1] quadVertices,
                      REAL_t[::1] fun_vals)
    cpdef REAL_t getQuadVolume(self,
                                const REAL_t[:, ::1] quadVertices)


cdef class Gauss(quadQuadratureRule):
    cdef public INDEX_t order


cdef class GaussJacobi(quadQuadratureRule):
    cdef public INDEX_t order


cdef class simplexDuffyTransformation(simplexQuadratureRule):
    pass


cdef class simplexXiaoGimbutas(simplexQuadratureRule):
    cdef public INDEX_t order


cdef class sphericalQuadRule:
    cdef:
        public REAL_t[:, ::1] vertexOffsets
        public REAL_t[::1] weights
        readonly INDEX_t num_nodes


cdef class sphericalQuadRule1D(sphericalQuadRule):
    pass


cdef class sphericalQuadRule2D(sphericalQuadRule):
    pass


cdef class simplexJaskowiecSukumar(simplexQuadratureRule):
    cdef public INDEX_t order
