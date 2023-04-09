###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t
from mpi4py cimport MPI


cdef enum flush_type:
    no_flush,
    flush_local,
    flush_local_all,
    flush,
    flush_all


cdef class algebraicOverlap:
    # Tracks the algebraic overlap with the DoFMap of another subdomain.

    cdef:
        public INDEX_t num_subdomain_dofs, num_shared_dofs, mySubdomainNo, otherSubdomainNo, numSharedVecs
        public INDEX_t memOffset, totalMemSize
        public INDEX_t[::1] memOffsetOther, memOffsetTemp
        public INDEX_t[::1] shared_dofs
        REAL_t[:, ::1] exchangeIn, exchangeOut
        REAL_t[::1] myExchangeIn, myExchangeOut
        COMPLEX_t[:, ::1] exchangeInComplex, exchangeOutComplex
        COMPLEX_t[::1] myExchangeInComplex, myExchangeOutComplex
        MPI.Comm comm
        INDEX_t tagNoSend, tagNoRecv

    cdef MPI.Request send(self,
                          const REAL_t[::1] vec,
                          INDEX_t vecNo=*,
                          flush_type flushOp=*)
    cdef MPI.Request receive(self,
                             const REAL_t[::1] vec,
                             INDEX_t vecNo=*,
                             flush_type flushOp=*)
    cdef MPI.Request sendComplex(self,
                                 const COMPLEX_t[::1] vec,
                                 INDEX_t vecNo=*,
                                 flush_type flushOp=*)
    cdef MPI.Request receiveComplex(self,
                                    const COMPLEX_t[::1] vec,
                                    INDEX_t vecNo=*,
                                    flush_type flushOp=*)
    cdef void accumulateProcess(self,
                                REAL_t[::1] vec,
                                INDEX_t vecNo=*)
    cdef void accumulateProcessComplex(self,
                                       COMPLEX_t[::1] vec,
                                       INDEX_t vecNo=*)
    cdef void setOverlapLocal(self,
                              REAL_t[::1] vec,
                              INDEX_t vecNo=*)
    cdef void uniqueProcess(self,
                            REAL_t[::1] vec,
                            INDEX_t vecNo=*)


cdef class algebraicOverlapPersistent(algebraicOverlap):
    cdef:
        MPI.Prequest SendRequest, RecvRequest


cdef class algebraicOverlapOneSidedGet(algebraicOverlap):
    cdef:
        MPI.Win Window


cdef class algebraicOverlapOneSidedPut(algebraicOverlap):
    cdef:
        MPI.Win Window


cdef class algebraicOverlapOneSidedPutLockAll(algebraicOverlap):
    cdef:
        MPI.Win Window


cdef class algebraicOverlapManager:
    # Tracks the algebraic overlap with the DoFMaps of all other subdomains.
    cdef:
        public INDEX_t numSubdomains, num_subdomain_dofs, mySubdomainNo
        INDEX_t _max_cross
        public dict overlaps
        public MPI.Comm comm
        list requestsSend
        # public, because Gauss-Seidel needs it
        public INDEX_t[::1] Didx, DidxNonOverlapping
        public REAL_t[::1] Dval, DvalNonOverlapping
        public REAL_t[:, ::1] exchangeIn, exchangeOut
        public str type
        MPI.Win Window
        BOOL_t distribute_is_prepared, non_overlapping_distribute_is_prepared

    cpdef void distribute(self,
                          REAL_t[::1] vec,
                          REAL_t[::1] vec2=*,
                          BOOL_t nonOverlapping=*,
                          INDEX_t level=*)
    cdef void distributeComplex(self,
                                COMPLEX_t[::1] vec,
                                COMPLEX_t[::1] vec2=*,
                                BOOL_t nonOverlapping=*)
    cdef void redistribute(self,
                           REAL_t[::1] vec,
                           REAL_t[::1] vec2=*,
                           BOOL_t nonOverlapping=*,
                           BOOL_t asynchronous=*,
                           INDEX_t vecNo=*)
    cpdef void accumulate(self,
                          REAL_t[::1] vec,
                          REAL_t[::1] return_vec=*,
                          BOOL_t asynchronous=*,
                          INDEX_t vecNo=*,
                          INDEX_t level=*)
    cdef void accumulateComplex(self,
                                COMPLEX_t[::1] vec,
                                COMPLEX_t[::1] return_vec=*,
                                BOOL_t asynchronous=*,
                                INDEX_t vecNo=*)
    cdef void send(self,
                   REAL_t[::1] vec,
                   BOOL_t asynchronous=*,
                   INDEX_t vecNo=*,
                   flush_type flushOp=*)
    cdef void receive(self,
                      REAL_t[::1] return_vec,
                      BOOL_t asynchronous=*,
                      INDEX_t vecNo=*,
                      flush_type flushOp=*)
    cdef void sendComplex(self,
                          COMPLEX_t[::1] vec,
                          BOOL_t asynchronous=*,
                          INDEX_t vecNo=*,
                          flush_type flushOp=*)
    cdef void receiveComplex(self,
                             COMPLEX_t[::1] return_vec,
                             BOOL_t asynchronous=*,
                             INDEX_t vecNo=*,
                             flush_type flushOp=*)


cdef class multilevelAlgebraicOverlapManager:
    cdef:
        public list levels
        public MPI.Comm comm
        REAL_t[::1] reduceMem
        public MPI.Win ReduceWindow
        public BOOL_t useLockAll
        public BOOL_t useAsynchronousComm
        list ReduceWindows
        list ReduceMems
        INDEX_t rank
