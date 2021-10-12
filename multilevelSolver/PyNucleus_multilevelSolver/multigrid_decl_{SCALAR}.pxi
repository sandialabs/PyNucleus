###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from mpi4py cimport MPI
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t
from PyNucleus_base.performanceLogger cimport PLogger, FakePLogger
from PyNucleus_base.solvers cimport {SCALAR_label_lc_}solver, {SCALAR_label_lc_}iterative_solver
from PyNucleus_fem.algebraicOverlaps cimport multilevelAlgebraicOverlapManager


cdef class {SCALAR_label}multigrid({SCALAR_label_lc_}iterative_solver):
    cdef:
        object hierarchyManager
        public multilevelAlgebraicOverlapManager overlap
        public CycleType cycle
        public {SCALAR_label_lc_}solver coarse_solver
        public MPI.Comm comm
        public list levels
        REAL_t _tol
    cdef void solveOnLevel(self, int lvlNo, {SCALAR}_t[::1] b, {SCALAR}_t[::1] x, BOOL_t simpleResidual=*)
    cdef int solve(self,
                   {SCALAR}_t[::1] b,
                   {SCALAR}_t[::1] x) except -1
    cpdef int solveFMG(self,
                       {SCALAR}_t[::1] b,
                       {SCALAR}_t[::1] x)
