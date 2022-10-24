###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes cimport INDEX_t, {SCALAR}_t, BOOL_t
from PyNucleus_base.linear_operators cimport {SCALAR_label}LinearOperator, {SCALAR_label}CSR_LinearOperator
from . DoFMaps cimport DoFMap
from . algebraicOverlaps cimport algebraicOverlapManager
from mpi4py cimport MPI


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


cdef class {SCALAR_label}RowDistributedOperator({SCALAR_label}LinearOperator):
    cdef:
        public {SCALAR_label}CSR_LinearOperator localMat
        public MPI.Comm comm
        public DoFMap dm
        public DoFMap lcl_dm
        public LinearOperator lclR
        public LinearOperator lclP
        INDEX_t[::1] near_offsetReceives
        INDEX_t[::1] near_offsetSends
        INDEX_t[::1] near_remoteReceives
        INDEX_t[::1] near_remoteSends
        INDEX_t[::1] near_counterReceives
        INDEX_t[::1] near_counterSends
        {SCALAR}_t[::1] near_dataReceives
        {SCALAR}_t[::1] near_dataSends
        INDEX_t[::1] rowIdx
        INDEX_t[::1] colIdx
    cdef void setupNear(self)
    cdef void communicateNear(self, {SCALAR}_t[::1] src, {SCALAR}_t[::1] target)
    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1
