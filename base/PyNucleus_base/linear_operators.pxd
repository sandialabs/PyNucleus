###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t, ENCODE_t

include "LinearOperator_decl_REAL.pxi"
include "LinearOperator_decl_COMPLEX.pxi"

include "LinearOperatorWrapper_decl_REAL.pxi"
include "LinearOperatorWrapper_decl_COMPLEX.pxi"

include "DenseLinearOperator_decl_REAL.pxi"
include "DenseLinearOperator_decl_COMPLEX.pxi"

include "CSR_LinearOperator_decl_REAL.pxi"
include "CSR_LinearOperator_decl_COMPLEX.pxi"

include "SSS_LinearOperator_decl_REAL.pxi"
include "SSS_LinearOperator_decl_COMPLEX.pxi"

include "DiagonalLinearOperator_decl_REAL.pxi"
include "DiagonalLinearOperator_decl_COMPLEX.pxi"

include "IJOperator_decl_REAL.pxi"
include "IJOperator_decl_COMPLEX.pxi"


cdef class Triple_Product_Linear_Operator(LinearOperator):
    cdef:
        public LinearOperator A, B, C
        public REAL_t[::1] temporaryMemory
        public REAL_t[::1] temporaryMemory2
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1
    cdef void _residual(self,
                        REAL_t[::1] x,
                        REAL_t[::1] rhs,
                        REAL_t[::1] result,
                        BOOL_t simpleResidual=*)


cdef class split_CSR_LinearOperator(LinearOperator):
    cdef:
        public CSR_LinearOperator A1, A2
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1


cdef class sparseGraph(LinearOperator):
    cdef:
        public INDEX_t[::1] indices, indptr
        public BOOL_t indices_sorted
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1
    cdef public INDEX_t matvec_no_overwrite(self,
                                            REAL_t[::1] x,
                                            REAL_t[::1] y) except -1


cdef class restrictionOp(sparseGraph):
    cdef:
        public int NoThreads
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1


cdef class prolongationOp(sparseGraph):
    cdef:
        public int NoThreads
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1


cdef BOOL_t sort_indices(INDEX_t[::1] indptr,
                       INDEX_t[::1] indices,
                       REAL_t[::1] data)


cdef class blockOperator(LinearOperator):
    cdef:
        INDEX_t[::1] blockInptrLeft, blockInptrRight
        REAL_t[::1] temp
        public list subblocks
        tuple blockShape
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1


cdef class nullOperator(LinearOperator):
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1
    cdef INDEX_t matvec_no_overwrite(self,
                                     REAL_t[::1] x,
                                     REAL_t[::1] y) except -1


cdef class identityOperator(LinearOperator):
    cdef:
        REAL_t alpha
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1
    cdef INDEX_t matvec_no_overwrite(self,
                                     REAL_t[::1] x,
                                     REAL_t[::1] y) except -1


cdef class blockLowerInverse(blockOperator):
    cdef:
        list diagonalInverses
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1


cdef class blockUpperInverse(blockOperator):
    cdef:
        list diagonalInverses
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1


cdef class wrapRealToComplex(ComplexLinearOperator):
    cdef:
        LinearOperator realA
        REAL_t[::1] temporaryMemory, temporaryMemory2
    cdef INDEX_t matvec(self,
                        COMPLEX_t[::1] x,
                        COMPLEX_t[::1] y) except -1


cdef class wrapRealToComplexCSR(ComplexLinearOperator):
    cdef:
        CSR_LinearOperator realA
    cdef INDEX_t matvec(self,
                        COMPLEX_t[::1] x,
                        COMPLEX_t[::1] y) except -1


cdef class HelmholtzShiftOperator(ComplexLinearOperator):
    cdef:
        CSR_LinearOperator M, S
        COMPLEX_t shift
    cdef INDEX_t matvec(self,
                        COMPLEX_t[::1] x,
                        COMPLEX_t[::1] y) except -1


cdef class delayedConstructionOperator(LinearOperator):
    cdef:
        public BOOL_t isConstructed
        public dict params
        public LinearOperator A
    cpdef int assure_constructed(self) except -1
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1
