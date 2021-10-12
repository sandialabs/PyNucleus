###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes cimport INDEX_t, {SCALAR}_t, BOOL_t
from PyNucleus_base.linear_operators cimport {SCALAR_label}LinearOperator, {SCALAR_label}CSR_LinearOperator
from . algebraicOverlaps cimport algebraicOverlapManager


cdef class {SCALAR_label}DistributedLinearOperator({SCALAR_label}LinearOperator):
    cdef:
        {SCALAR_label}LinearOperator A
        algebraicOverlapManager overlaps
        public BOOL_t asynchronous
        public BOOL_t doDistribute
        public BOOL_t keepDistributedResult
        {SCALAR}_t[::1] tempMemX
        public {SCALAR}_t[::1] tempMemY
    cdef void allocateTempMemory(self, INDEX_t sizeX, INDEX_t sizeY)
    cdef void setTempMemory(self, {SCALAR}_t[::1] tempMemX, {SCALAR}_t[::1] tempMemY)


cdef class {SCALAR_label}CSR_DistributedLinearOperator({SCALAR_label}DistributedLinearOperator):
    cdef:
        {SCALAR_label}CSR_LinearOperator csrA
        INDEX_t[::1] overlap_indices
