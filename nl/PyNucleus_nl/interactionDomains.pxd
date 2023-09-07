###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t
from . twoPointFunctions cimport parametrizedTwoPointFunction

cdef enum RELATIVE_POSITION_t:
    INTERACT
    REMOTE
    CUT


cdef class interactionDomain(parametrizedTwoPointFunction):
    cdef:
        public RELATIVE_POSITION_t relPos
        public BOOL_t complement
        REAL_t[::1] intervals1, intervals2
        REAL_t[:, :, ::1] A_Simplex
        REAL_t[:, ::1] b_Simplex
        REAL_t[::1] vol_Simplex
        REAL_t[:, :, ::1] A_Node
        REAL_t[:, ::1] b_Node
        REAL_t[::1] vol_Node
        INDEX_t iter_Simplex, iterEnd_Simplex
        INDEX_t iter_Node, iterEnd_Node
        BOOL_t identityMapping
        REAL_t[:, ::1] specialOffsets
    cdef BOOL_t isInside(self, REAL_t[::1] x, REAL_t[::1] y)
    cdef RELATIVE_POSITION_t getRelativePosition(self, REAL_t[:,::1] simplex1, REAL_t[:,::1] simplex2)
    cdef INDEX_t findIntersections(self, REAL_t[::1] x, REAL_t[:, ::1] simplex, INDEX_t start, INDEX_t end, REAL_t[::1] intersections)
    cdef void startLoopSubSimplices_Simplex(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2)
    cdef BOOL_t nextSubSimplex_Simplex(self, REAL_t[:, ::1] A, REAL_t[::1] b, REAL_t *vol)
    cdef void startLoopSubSimplices_Node(self, REAL_t[::1] node1, REAL_t[:, ::1] simplex2)
    cdef BOOL_t nextSubSimplex_Node(self, REAL_t[:, ::1] A, REAL_t *vol)


cdef class linearTransformInteraction(interactionDomain):
    cdef:
        interactionDomain baseInteraction
        public REAL_t[:, ::1] A, invA
        public REAL_t detA
        REAL_t[::1] vec, vec2
        REAL_t[:, ::1] simplex1, simplex2
    cdef void transformVectorForward(self, REAL_t[::1] x, REAL_t[::1] y)
    cdef void transformSimplexForward(self, REAL_t[:, ::1] simplex, REAL_t[:, ::1] simplex2)


cdef class fullSpace(interactionDomain):
    pass


cdef class ball1(linearTransformInteraction):
    pass


cdef class ball2(interactionDomain):
    pass


cdef class ballInf(interactionDomain):
    pass


cdef class ellipse(linearTransformInteraction):
    pass
