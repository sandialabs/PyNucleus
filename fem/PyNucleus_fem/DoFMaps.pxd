###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes cimport REAL_t, COMPLEX_t, INDEX_t, BOOL_t
from PyNucleus_base.ip_norm cimport ipBase, normBase, complexipBase, complexNormBase
from . meshCy cimport meshBase, vertices_t, cells_t
from . functions cimport function, complexFunction, vectorFunction

include "vector_decl_REAL.pxi"
include "vector_decl_COMPLEX.pxi"


cdef class DoFMap:
    cdef:
        public meshBase mesh
        readonly INDEX_t dim
        BOOL_t reordered
        public list localShapeFunctions
        public REAL_t[:, ::1] nodes
        public INDEX_t num_dofs
        public INDEX_t num_boundary_dofs
        public INDEX_t[:, ::1] dofs
        public INDEX_t polynomialOrder
        public list tag
        public INDEX_t dofs_per_vertex
        public INDEX_t dofs_per_edge
        public INDEX_t dofs_per_face
        public INDEX_t dofs_per_cell
        public INDEX_t dofs_per_element
        public ipBase inner
        public normBase norm
        public complexipBase complex_inner
        public complexNormBase complex_norm
    cdef INDEX_t cell2dof(self,
                          const INDEX_t cellNo,
                          const INDEX_t perCellNo)
    cpdef void reorder(self,
                       const INDEX_t[::1] perm)
    cdef void getNodalCoordinates(self, REAL_t[:, ::1] cell, REAL_t[:, ::1] coords)
    cpdef void getVertexDoFs(self, INDEX_t[:, ::1] v2d)
    cpdef void resetUsingIndicator(self, function indicator)
    cpdef void resetUsingFEVector(self, REAL_t[::1] ind)


cdef class P1_DoFMap(DoFMap):
    pass


cdef class P2_DoFMap(DoFMap):
    pass


cdef class P0_DoFMap(DoFMap):
    pass


cdef class P3_DoFMap(DoFMap):
    pass



cdef class shapeFunction:
    cdef:
        REAL_t[::1] bary
    cdef REAL_t eval(self, const REAL_t[::1] lam)
    cdef REAL_t evalStrided(self, const REAL_t* lam, INDEX_t stride)
    cdef REAL_t evalGlobal(self, REAL_t[:, ::1] simplex, REAL_t[::1] x)


cdef class vectorShapeFunction:
    cdef:
        INDEX_t dim
        INDEX_t[::1] cell
    cpdef void setCell(self, INDEX_t[::1] cell)
    cdef void eval(self, const REAL_t[::1] lam, const REAL_t[:, ::1] gradLam, REAL_t[::1] value)
    cdef void evalGlobal(self, const REAL_t[:, ::1] simplex, const REAL_t[::1] x, REAL_t[::1] value)
