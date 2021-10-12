###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.solvers cimport solver, complex_solver


cdef class {SCALAR_label}smoother:
    cdef:
        {SCALAR_label}LinearOperator _A
        public algebraicOverlapManager overlap
    cdef void eval(self,
                   {SCALAR}_t[::1] b,
                   {SCALAR}_t[::1] y,
                   BOOL_t postsmoother,
                   BOOL_t simpleResidual=*)


cdef class {SCALAR_label}preconditioner({SCALAR_label}LinearOperator):
    cdef void setPre(self)
    cdef void setPost(self)


cdef class {SCALAR_label}separableSmoother({SCALAR_label}smoother):
    cdef:
        public {SCALAR_label}preconditioner prec
        INDEX_t presmoothingSteps, postsmoothingSteps
        {SCALAR_label}LinearOperator _accA
        public {SCALAR}_t[::1] temporaryMemory
        public {SCALAR}_t[::1] temporaryMemory2


cdef class {SCALAR_label}jacobiPreconditioner({SCALAR_label}preconditioner):
    cdef:
        {SCALAR}_t[::1] invD
        public {SCALAR}_t omega
    cdef INDEX_t matvec(self, {SCALAR}_t[::1] x, {SCALAR}_t[::1] y) except -1


cdef class {SCALAR_label}jacobiSmoother({SCALAR_label}separableSmoother):
    pass


cdef class {SCALAR_label}blockJacobiPreconditioner({SCALAR_label}preconditioner):
    cdef:
        {SCALAR_label_lc_}solver invD
        public {SCALAR}_t omega
    cdef INDEX_t matvec(self, {SCALAR}_t[::1] x, {SCALAR}_t[::1] y) except -1


cdef class {SCALAR_label}blockJacobiSmoother({SCALAR_label}separableSmoother):
    pass


cdef class {SCALAR_label}gmresSmoother({SCALAR_label}smoother):
    cdef:
        public {SCALAR_label_lc_}solver solver
        INDEX_t presmoothingSteps, postsmoothingSteps
        {SCALAR_label}LinearOperator _accA
        public {SCALAR}_t[::1] temporaryMemory
        public {SCALAR}_t[::1] temporaryMemory2
