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
        public meshBase mesh  #: The underlying mesh
        readonly INDEX_t dim  #: The spatial dimension of the underlying mesh
        BOOL_t reordered
        public list localShapeFunctions  #: List of local shape functions
        public REAL_t[:, ::1] nodes  #: The barycentric coordinates of the DoFs
        public REAL_t[:, ::1] dof_dual
        public INDEX_t num_dofs  #: The number of DoFs of the finite element space
        public INDEX_t num_boundary_dofs  #: The number of boundary DoFs of the finite element space
        public INDEX_t[:, ::1] dofs  #: Array with the mapping from cells to DoFs
        public INDEX_t polynomialOrder  #: The polynomial order of the finite element space
        public list tag
        public function tagFunction
        public INDEX_t dofs_per_vertex  #: The number of degrees of freedom per vertex
        public INDEX_t dofs_per_edge  #: The number of degrees of freedom per edge
        public INDEX_t dofs_per_face  #: The number of degrees of freedom per face
        public INDEX_t dofs_per_cell  #: The number of degrees of freedom per cell
        public INDEX_t dofs_per_element  #: The total number of degrees of freedom per element
        public ipBase inner  #: The inner product of the finite element space
        public normBase norm  #: The norm of the finite element space
        public complexipBase complex_inner  #: The complex inner product of the finite element space
        public complexNormBase complex_norm  #: The complex norm of the finite element space
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


cdef class N1e_DoFMap(DoFMap):
    pass


cdef class shapeFunction:
    cdef:
        REAL_t[::1] bary
    cdef REAL_t eval(self, const REAL_t[::1] lam)
    cdef REAL_t evalStrided(self, const REAL_t* lam, INDEX_t stride)
    cdef REAL_t evalGlobal(self, REAL_t[:, ::1] simplex, REAL_t[::1] x)
    cdef void evalGrad(self, const REAL_t[::1] lam, const REAL_t[:, ::1] gradLam, REAL_t[::1] value)


cdef class vectorShapeFunction:
    cdef:
        INDEX_t dim
        INDEX_t[::1] cell
        public BOOL_t needsGradients
    cpdef void setCell(self, INDEX_t[::1] cell)
    cdef void eval(self, const REAL_t[::1] lam, const REAL_t[:, ::1] gradLam, REAL_t[::1] value)
    cdef void evalGlobal(self, const REAL_t[:, ::1] simplex, const REAL_t[::1] x, REAL_t[::1] value)


cdef class Product_DoFMap(DoFMap):
    cdef:
        public INDEX_t numComponents
        public DoFMap scalarDM
