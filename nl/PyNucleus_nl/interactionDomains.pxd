###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t
from PyNucleus_fem.functions cimport function, matrixFunction
from . twoPointFunctions cimport parametrizedTwoPointFunction

cdef enum RELATIVE_POSITION_t:
    INTERACT
    REMOTE
    CUT


cdef class interactionDomain(parametrizedTwoPointFunction):
    cdef:
        public RELATIVE_POSITION_t relPos
        public BOOL_t complement
        public REAL_t dist2
        public function horizonFun
        BOOL_t identityMapping
        REAL_t[:, :, ::1] A_Simplex
        REAL_t[:, ::1] b_Simplex
        REAL_t[::1] vol_Simplex
        REAL_t[:, :, ::1] A_Node
        REAL_t[::1] vol_Node
        INDEX_t iter_Simplex, iterEnd_Simplex
        INDEX_t iter_Node, iterEnd_Node
        REAL_t[:, ::1] specialOffsets
        REAL_t[::1] intervals1, intervals2
    cdef BOOL_t isInside(self, REAL_t[::1] x, REAL_t[::1] y)
    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2)
    cdef void startLoopSubSimplices_Simplex(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2)
    cdef BOOL_t nextSubSimplex_Simplex(self, REAL_t[:, ::1] A, REAL_t[::1] b, REAL_t *vol)
    cdef void startLoopSubSimplices_Node(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2)
    cdef BOOL_t nextSubSimplex_Node(self, REAL_t[:, ::1] A, REAL_t *vol)


cdef class barycenterDomain(interactionDomain):
    pass


cdef class retriangulationDomain(interactionDomain):
    cdef computeSpecialPoints(self)
    cdef INDEX_t findIntersections(self, REAL_t[::1] x, REAL_t[:, ::1] simplex, INDEX_t start, INDEX_t end, REAL_t[::1] intersections)


cdef class linearTransformInteraction(interactionDomain):
    cdef:
        interactionDomain baseInteraction
        matrixFunction transform
        BOOL_t callTransform
        public REAL_t[:, ::1] A
        REAL_t[::1] vec, vec2
        REAL_t[:, ::1] simplex1, simplex2
    cdef void transformVectorForward(self, REAL_t* x, REAL_t* y, REAL_t* new_y)
    cdef void transformSimplexForwardPoint(self, REAL_t[::1] x, REAL_t[:, ::1] simplex, REAL_t[:, ::1] new_simplex)
    cdef void transformSimplexForward(self, REAL_t[:, ::1] simplex, REAL_t[:, ::1] simplex2)


cdef class fullSpace(interactionDomain):
    pass


cdef class ball1_retriangulation(linearTransformInteraction):
    pass


cdef class ball1_barycenter(linearTransformInteraction):
    pass


cdef class ball2_retriangulation(retriangulationDomain):
    pass


cdef class ball2_barycenter(barycenterDomain):
    pass


cdef class ballInf_retriangulation(retriangulationDomain):
    pass


cdef class ballInf_barycenter(barycenterDomain):
    pass


cdef class ellipse_retriangulation(linearTransformInteraction):
    cdef:
        public function a
        public function b
        public function theta


cdef class ellipse_barycenter(linearTransformInteraction):
    cdef:
        public function a
        public function b
        public function theta


cdef class ball2_dilation_barycenter(barycenterDomain):
    cdef:
        REAL_t[::1] w, wT
        REAL_t[:, ::1] mat
        REAL_t c, d
        REAL_t[:, ::1] tempSimplex1, tempSimplex2


cdef class ball2_dilation_retriangulation(retriangulationDomain):
    cdef:
        REAL_t[::1] w, wT
        REAL_t[:, ::1] mat
        REAL_t c, d
        REAL_t[:, ::1] tempSimplex1, tempSimplex2
