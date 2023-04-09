###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes cimport INDEX_t, {SCALAR}_t, BOOL_t
from PyNucleus_base.solvers cimport {SCALAR_label_lc_}solver, {SCALAR_label_lc_}iterative_solver


cdef class {SCALAR_label}coarseSolver({SCALAR_label_lc_}iterative_solver):
    cdef:
        MPI.Comm comm, subset_comm, subset_commFine
        MPI.Intercomm inter_comm
        INDEX_t myLeaderRank, otherLeaderRank
        public algebraicOverlapManager overlapsCoarse, overlapsFine, intraLevelCoarse, intraLevelFine
        public {SCALAR}_t[::1] rhs, x
        public {SCALAR_label_lc_}solver Ainv
        public BOOL_t inCG
        public BOOL_t inSubdomain
        list levels
        object hierarchy
        object hierarchyManager
        INDEX_t depth
        str solver_description
        str solverName
        dict kwargs
        public BOOL_t asynchronous
        str name
    cpdef BOOL_t canWriteRHS(self)
    cpdef void sendRHS(self, {SCALAR}_t[::1] b)
    cdef BOOL_t solve_cg(self)
    cpdef BOOL_t getSolution(self, {SCALAR}_t[::1] x)
