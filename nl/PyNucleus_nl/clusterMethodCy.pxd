###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


cimport numpy as np
from PyNucleus_fem.quadrature cimport (simplexDuffyTransformation,
                             simplexQuadratureRule,
                             simplexXiaoGimbutas)
from PyNucleus_fem.functions cimport function
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, BOOL_t
from PyNucleus_base.linear_operators cimport (LinearOperator,
                                               Dense_LinearOperator)
from PyNucleus_base.tupleDict cimport indexSet, indexSetIterator, arrayIndexSet, arrayIndexSetIterator, bitArray
from PyNucleus_fem.DoFMaps cimport DoFMap
from PyNucleus_fem.meshCy cimport meshBase
from . fractionalOrders cimport fractionalOrderBase
from . kernelsCy cimport FractionalKernel

cdef class tree_node:
    cdef:
        public tree_node parent
        public list children
        public INDEX_t dim
        public INDEX_t id
        public INDEX_t distFromRoot
        public indexSet _dofs
        INDEX_t _num_dofs
        public indexSet _cells
        public REAL_t[:, ::1] box
        public REAL_t[:, ::1] transferOperator
        public REAL_t[:, :, ::1] value
        public REAL_t[::1] coefficientsUp, coefficientsDown
        public BOOL_t mixed_node
        public BOOL_t canBeAssembled
    cdef indexSet get_dofs(self)
    cdef indexSet get_cells(self)
    cpdef INDEX_t findCell(self, meshBase mesh, REAL_t[::1] vertex, REAL_t[:, ::1] simplex, REAL_t[::1] bary)
    cpdef set findCells(self, meshBase mesh, REAL_t[::1] vertex, REAL_t r, REAL_t[:, ::1] simplex)
    cdef tree_node get_node(self, INDEX_t id)
    cdef BOOL_t trim(self, bitArray keep)
    cdef void upwardPassMatrix(self, dict coefficientsUp)


cdef class productIterator:
    cdef:
        INDEX_t m
        INDEX_t dim
        INDEX_t[::1] idx
    cdef void reset(self)
    cdef BOOL_t step(self)


cdef class farFieldClusterPair:
    cdef:
        public tree_node n1, n2
        public REAL_t[:, ::1] kernelInterpolant
    cpdef void apply(self, REAL_t[::1] x, REAL_t[::1] y)


cdef class H2Matrix(LinearOperator):
    cdef:
        public LinearOperator Anear
        public dict Pfar
        public tree_node tree
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1
