###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from . myTypes cimport BOOL_t, REAL_t
ctypedef object OrderedDict_t
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py cimport MPI


cdef class FakeTimer:
    cdef void start(self)
    cdef void end(self)
    cpdef void enterData(self)


cdef class Timer(FakeTimer):
    cdef:
        double startTime
        double startTime_unsynced
        double elapsed
        double elapsed_unsynced
        double startMem
        double endMem
        str key
        FakePLogger parent
        BOOL_t manualDataEntry
        BOOL_t sync
        BOOL_t memoryProfiling
        BOOL_t memoryRegionsAreEnabled
        MPI.Comm comm
    cdef void start(self)
    cdef void end(self)
    cpdef void enterData(self)


cdef class FakePLogger:
    cdef:
        BOOL_t memoryProfiling
        object process
    cpdef void empty(self)
    cpdef void addValue(self, str key, value)
    cpdef FakeTimer Timer(self, str key, BOOL_t manualDataEntry=*)


cdef class PLogger(FakePLogger):
    cdef:
        public OrderedDict_t values
    cpdef void empty(self)
    cpdef void addValue(self, str key, value)


cdef class LoggingPLogger(PLogger):
    cdef:
        object logger
        object loggerLevel
    cpdef FakeTimer Timer(self, str key, BOOL_t manualDataEntry=*)
