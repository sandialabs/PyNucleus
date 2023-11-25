###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . blas cimport scaleScalar, assign, assign3
from . blas import uninitialized


cdef class {SCALAR_label}LinearOperator:
    cdef:
        public INDEX_t num_rows, num_columns
        {SCALAR}_t[::1] _diagonal
    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1
    cdef INDEX_t matvec_no_overwrite(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] y) except -1
    cdef INDEX_t matvec_multi(self,
                              {SCALAR}_t[:, ::1] x,
                              {SCALAR}_t[:, ::1] y) except -1
    cdef INDEX_t matvecTrans(self,
                             {SCALAR}_t[::1] x,
                             {SCALAR}_t[::1] y) except -1
    cdef INDEX_t matvecTrans_no_overwrite(self,
                                          {SCALAR}_t[::1] x,
                                          {SCALAR}_t[::1] y) except -1
    cdef void residual(self,
                       {SCALAR}_t[::1] x,
                       {SCALAR}_t[::1] rhs,
                       {SCALAR}_t[::1] result,
                       BOOL_t simpleResidual=*)
    cdef void preconditionedResidual(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[::1] rhs,
                                     {SCALAR}_t[::1] result,
                                     BOOL_t simpleResidual=*)
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
    cdef void addToEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val)
    cdef {SCALAR}_t getEntry(self, INDEX_t I, INDEX_t J)
    cdef void setEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val)


cdef class {SCALAR_label}TimeStepperLinearOperator({SCALAR_label}LinearOperator):
    cdef:
        public {SCALAR_label}LinearOperator M, S
        public {SCALAR}_t facM, facS
        {SCALAR}_t[::1] z
    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1


cdef class {SCALAR_label}Multiply_Linear_Operator({SCALAR_label}LinearOperator):
    cdef:
        public {SCALAR_label}LinearOperator A
        public {SCALAR}_t factor
    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1
    cdef void _residual(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] rhs,
                        {SCALAR}_t[::1] result,
                        BOOL_t simpleResidual=*)


cdef class {SCALAR_label}Product_Linear_Operator({SCALAR_label}LinearOperator):
    cdef:
        public {SCALAR_label}LinearOperator A, B
        public {SCALAR}_t[::1] temporaryMemory
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


cdef class {SCALAR_label}VectorLinearOperator:
    cdef:
        public INDEX_t num_rows, num_columns, vectorSize
        {SCALAR}_t[::1] _diagonal
    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[:, ::1] y) except -1
    cdef INDEX_t matvec_no_overwrite(self,
                                     {SCALAR}_t[::1] x,
                                     {SCALAR}_t[:, ::1] y) except -1
    cdef void addToEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t[::1] val)
    cdef void getEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t[::1] val)
    cdef void setEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t[::1] val)


cdef class {SCALAR_label}Transpose_Linear_Operator({SCALAR_label}LinearOperator):
    cdef:
        public {SCALAR_label}LinearOperator A
    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1
    cdef void _residual(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] rhs,
                        {SCALAR}_t[::1] result,
                        BOOL_t simpleResidual=*)
