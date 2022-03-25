###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from . blas cimport updateScaled
from . solvers cimport solver
from . linear_operators cimport {SCALAR_label}LinearOperator
from . linear_operators cimport {SCALAR_label}CSR_LinearOperator


cdef class {SCALAR_label}SchurComplement({SCALAR_label}LinearOperator):
    cdef:
        public {SCALAR_label}LinearOperator A, A11, A12, A21, A22
        public {SCALAR_label}LinearOperator R1, P1, R2, P2
        public solver invA22
        public {SCALAR}_t[::1] temporaryMemory
        public {SCALAR}_t[::1] temporaryMemory2
        public {SCALAR}_t[::1] temporaryMemory3
    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1
    cdef INDEX_t matvec_no_overwrite(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] y) except -1
    cdef void _residual(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] rhs,
                        {SCALAR}_t[::1] result,
                        BOOL_t simpleResidual=*)

    cdef void _preconditionedResidual(self,
                                      {SCALAR}_t[::1] x,
                                      {SCALAR}_t[::1] rhs,
                                      {SCALAR}_t[::1] result,
                                      BOOL_t simpleResidual=*)
