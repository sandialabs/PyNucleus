###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from timeit import default_timer as time
from collections import OrderedDict
from . memProfile import memRegionsAreEnabled as memRegionsAreEnabledPy

cdef REAL_t MB = 1./2**20
cdef BOOL_t memRegionsAreEnabled = memRegionsAreEnabledPy
cdef dict memRegions = {}


cpdef void startMemRegion(str key):
    # use memory profiler if available
    global profile
    key = key.replace(' ', '_').replace('.', '')
    memRegions[key] = profile.timestamp(key)
    memRegions[key].__enter__()


cpdef void endMemRegion(str key):
    key = key.replace(' ', '_').replace('.', '')
    memRegions[key].__exit__()
    memRegions[key].timestamps[-1][0]
    del memRegions[key]


cdef class FakeTimer:
    def __init__(self):
        pass

    cdef void start(self):
        pass

    cdef void end(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass

    cpdef void enterData(self):
        pass


cdef class Timer(FakeTimer):
    def __init__(self, str key, FakePLogger parent, BOOL_t manualDataEntry=False, MPI.Comm comm=None, BOOL_t sync=False, BOOL_t forceMemRegionOff=False):
        self.key = key
        self.elapsed = 0.
        self.parent = parent
        self.manualDataEntry = manualDataEntry
        self.comm = comm
        self.sync = sync
        if forceMemRegionOff:
            self.memoryRegionsAreEnabled = False
        else:
            self.memoryRegionsAreEnabled = memRegionsAreEnabled
        self.memoryProfiling = self.parent.memoryProfiling
        if self.sync:
            assert self.comm is not None

    cdef void start(self):
        if self.sync:
            self.startTime_unsynced = time()
            self.comm.Barrier()
        if self.memoryProfiling:
            self.startMem = self.parent.process.memory_info()[0]*MB
        if self.memoryRegionsAreEnabled:
            startMemRegion(self.key)
        self.startTime = time()

    cdef void end(self):
        if self.sync:
            self.elapsed_unsynced += time()-self.startTime_unsynced
            self.comm.Barrier()
        self.elapsed += time()-self.startTime
        if self.memoryProfiling:
            self.endMem = self.parent.process.memory_info()[0]*MB
        if self.memoryRegionsAreEnabled:
            endMemRegion(self.key)
        if not self.manualDataEntry:
            self.parent.addValue(self.key, self.elapsed)

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.end()

    cpdef void enterData(self):
        self.parent.addValue(self.key, self.elapsed)

    def getInterval(self):
        return self.elapsed

    def getIntervalUnsynced(self):
        return self.elapsed_unsynced

    interval = property(fget=getInterval)
    interval_unsynced = property(fget=getIntervalUnsynced)


cdef class LoggingTimer(Timer):
    cdef:
        object logger
        object loggerLevel
        str StartMessage

    def __init__(self, logger, loggerLevel, str key, FakePLogger parent, BOOL_t manualDataEntry=False, MPI.Comm comm=None, BOOL_t sync=False, str StartMessage=''):
        super(LoggingTimer, self).__init__(key, parent, manualDataEntry, comm, sync)
        self.logger = logger
        self.loggerLevel = loggerLevel
        self.StartMessage = StartMessage

    def __enter__(self):
        if self.StartMessage != '':
            self.logger.log(self.loggerLevel, self.StartMessage)
        super(LoggingTimer, self).__enter__()

    def __exit__(self, type, value, traceback):
        super(LoggingTimer, self).__exit__(type, value, traceback)
        if not self.memoryProfiling:
            self.logger.log(self.loggerLevel, self.key + ' in {:.3} s'.format(self.elapsed))
        else:
            self.logger.log(self.loggerLevel, self.key + ' in {:.3} s, {} MB (Alloc: {} MB)'.format(self.elapsed, self.endMem, self.endMem-self.startMem))


cdef class FakePLogger:
    def __init__(self):
        self.memoryProfiling = False

    cpdef void empty(self):
        pass

    cpdef void addValue(self, str key, value):
        pass

    def __getitem__(self, str key):
        return None

    cpdef FakeTimer Timer(self, str key, BOOL_t manualDataEntry=False):
        return FakeTimer()


cdef class PLogger(FakePLogger):
    def __init__(self, process=None):
        self.values = OrderedDict()
        self.process = process
        self.memoryProfiling = self.process is not None

    cpdef void empty(self):
        self.values = OrderedDict()

    cpdef void addValue(self, str key, value):
        try:
            self.values[key].append(value)
        except KeyError:
            self.values[key] = [value]

    def __getitem__(self, str key):
        return self.values[key]

    cpdef FakeTimer Timer(self, str key, BOOL_t manualDataEntry=False):
        return Timer(key, self, manualDataEntry)

    def __repr__(self):
        return self.report()

    def report(self, totalsOnly=True):
        s = ''
        for key in sorted(self.values.keys()):
            if totalsOnly:
                s += '{}: {} ({} calls)\n'.format(str(key), sum(self.values[key]), len(self.values[key]))
            else:
                s += str(key) +': ' + self.values[key].__repr__() + '\n'
        return s


cdef class LoggingPLogger(PLogger):
    def __init__(self, logger, loggerLevel, process=None):
        PLogger.__init__(self, process)
        self.logger = logger
        self.loggerLevel = loggerLevel

    cpdef FakeTimer Timer(self, str key, BOOL_t manualDataEntry=False):
        return LoggingTimer(self.logger, self.loggerLevel, key, self, manualDataEntry)
