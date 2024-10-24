###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
import logging
import os
import atexit
import sys
import traceback
import re
import argparse
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from collections import OrderedDict
from copy import deepcopy
import inspect
from . myTypes import INDEX, REAL
from . performanceLogger import PLogger, LoggingTimer, Timer, LoggingPLogger
from . blas import uninitialized, uninitialized_like

_syncDefault = False


def setSyncDefault(sync):
    global _syncDefault
    _syncDefault = sync


class TimerManager:
    def __init__(self, logger, comm=None, print_rank=0, prefix='', myPLogger=None, memoryProfiling=False, loggingSubTimers=False):
        self.logger = logger
        self.comm = comm
        self.print_rank = print_rank
        self.prefix = prefix
        self.memoryProfiling = memoryProfiling
        if self.memoryProfiling:
            from psutil import Process
            self.process = Process(os.getpid())
        else:
            self.process = None
        if myPLogger is None:
            if loggingSubTimers:
                self.PLogger = LoggingPLogger(self.logger, logging.INFO, self.process)
            else:
                self.PLogger = PLogger(self.process)
            self.isSubManager = False
            self.totalTimer = Timer('total', self.PLogger, forceMemRegionOff=True)
            self.totalTimer.__enter__()
        else:
            self.isSubManager = True
            self.PLogger = myPLogger

    def getTimer(self, FinalMessage=None, StartMessage=None,
                 level=logging.INFO, overrideComm=None, sync=_syncDefault):
        if overrideComm is None:
            comm = self.comm
        else:
            comm = overrideComm
        if comm is None or comm.rank == self.print_rank:
            t = LoggingTimer(self.logger, level, FinalMessage, self.PLogger)
        else:
            t = Timer(FinalMessage, self.PLogger)
        return t

    def __call__(self, *args, **kwargs):
        return self.getTimer(*args, **kwargs)

    def setOutputGroup(self, rank, oG):

        def mergeOrdered(a_list, b_list):
            keys = []
            while (len(a_list) > 0) and (len(b_list) > 0):
                if a_list[0] in b_list:
                    if a_list[0] == b_list[0]:
                        keys.append(a_list[0])
                        a_list.pop(0)
                        b_list.pop(0)
                    else:
                        keys.append(b_list[0])
                        a_list.remove(b_list[0])
                        b_list.pop(0)
                elif b_list[0] in a_list:
                    keys.append(a_list[0])
                    a_list.pop(0)
                else:
                    keys.append(a_list[0])
                    a_list.pop(0)
            keys += a_list
            keys += b_list
            return keys

        if not self.isSubManager:
            self.totalTimer.__exit__(None, None, None)
        data = self.PLogger.values
        data2 = OrderedDict()
        for key in data.keys():
            val = data[key]
            # (number of calls, min over calls, mean over calls, med over calls, max over calls)
            # on rank
            try:
                data2[key] = (len(val), np.min(val), np.mean(val), np.median(val), np.max(val))
            except:
                pass
        data = data2
        # gather data for all ranks
        if self.comm is not None:
            data = self.comm.gather(data, root=rank)
        else:
            data = [data]
        if self.comm is None or self.comm.rank == rank:
            if self.comm is None:
                commSize = 1
            else:
                commSize = self.comm.size
            assert len(data) == commSize

            keys = list(data[0].keys())
            for i in range(1, len(data)):
                keys = mergeOrdered(keys, list(data[i].keys()))
            pData = {}
            for i in range(len(data)):
                for key in keys:
                    d = data[i].get(key, (0, np.inf, 0., 0., -np.inf))
                    try:
                        pData[key].append(d)
                    except KeyError:
                        pData[key] = [d]
            for key in pData:
                assert len(pData[key]) == commSize, key
                oG.add(key, pData[key])

    def getSubManager(self, logger):
        return TimerManager(logger, self.comm, self.print_rank, self.prefix, self.PLogger)


def getLoggingTimer(logger, comm=None, print_rank=0, prefix='', rootOutput=False):
    def getTimer(FinalMessage='', StartMessage='',
                 level=logging.INFO, overrideComm=None, sync=_syncDefault):
        from . performanceLogger import FakePLogger
        # return Timer(prefix=prefix, FinalMessage=FinalMessage, StartMessage=StartMessage,
        #              logger=logger, level=level, comm=comm, overrideComm=overrideComm, print_rank=print_rank, sync=sync)
        if StartMessage != '':
            StartMessage = prefix+StartMessage
        return LoggingTimer(logger, level, prefix+FinalMessage, FakePLogger(), StartMessage=StartMessage,
                            comm=comm, sync=sync)
    return getTimer


display_available = ("DISPLAY" in os.environ and
                     "SSH_CONNECTION" not in os.environ)


def computeErrors(levels, solutions, norm, timeVals=None):
    assert len(levels) == len(solutions)
    if timeVals is not None:
        for lvlNo in range(len(levels)):
            assert solutions[lvlNo].shape[0] == len(timeVals[lvlNo])
    for lvlNo in range(1, len(levels)):
        assert solutions[lvlNo].shape[1] == levels[lvlNo]['P'].shape[0]
        assert solutions[lvlNo-1].shape[1] == levels[lvlNo]['P'].shape[1]

    errors = []

    uFine = solutions[-1]
    for lvlNo in range(len(levels)-1):
        u = solutions[lvlNo]
        for j in range(lvlNo+1, len(levels)):
            u2 = uninitialized((u.shape[0], levels[j]['P'].shape[0]), dtype=REAL)
            for k in range(u.shape[0]):
                u2[k, :] = levels[j]['P']*u[k, :]
            u = u2
        if timeVals is not None:
            # linear interpolation in time
            uNew = uninitialized_like(uFine)
            uNew[0, :] = u[0, :]
            uNew[-1, :] = u[-1, :]
            for k in range(1, uFine.shape[0]-1):
                t = timeVals[-1][k]
                j = 0
                while timeVals[lvlNo][j+1] < t:
                    j += 1
                t0 = timeVals[lvlNo][j]
                t1 = timeVals[lvlNo][j+1]
                assert t0 <= t <= t1
                uNew[k, :] = (t1-t)/(t1-t0)*u[j, :] + (t-t0)/(t1-t0)*u[j+1, :]
            u = uNew
        errors.append(norm(u-uFine))
    return errors


def roc(idx, val, FillUp=False, exp=False):
    "Calculates the rate of convergence."
    idx, val = np.atleast_2d(idx), np.atleast_2d(val)
    if idx.shape[0] == 1:
        idx = idx.transpose()
    if idx.shape[0] != val.shape[0]:
        val = val.transpose()
    if idx.shape[0] != val.shape[0]:
        raise Exception('Arrays of sizes {} and {} not compatible.'.format(idx.shape[0], val.shape[0]))
    if exp:
        rate = np.log(val[0:-1, :]/val[1:, :])/(idx[0:-1, :]-idx[1:, :])
    else:
        rate = np.log(val[0:-1, :]/val[1:, :])/np.log(idx[0:-1, :]/idx[1:, :])
    if FillUp:
        return np.vstack([rate, [np.nan]])
    else:
        return rate


class exitHandler(object):
    def __init__(self, comm):
        self.comm = comm
        self.exit_code = None
        self.exception = None
        self.exc_type = None
        self._orig_exit = sys.exit
        sys.exit = self.exit
        sys.excepthook = self.exc_handler
        atexit.register(self.atExitHandler)

    def exit(self, code=0):
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, *args):
        self.exc_type = exc_type
        self.exception = exc

    def atExitHandler(self):
        if self.exit_code is not None and self.exit_code != 0:
            logging.error("death by sys.exit(%d)" % self.exit_code)
            self.comm.Abort(self.exit_code)
        elif self.exception is not None:
            lines = traceback.format_exception(self.exc_type, self.exception,
                                               tb=self.exception.__traceback__)
            msg = ''.join(['{}: {}'.format(self.comm.rank, line) for line in lines])
            logging.error('\n'+msg)
            self.comm.Abort(1234)


def saveDictToHDF5(params, f, ignore=set()):
    import h5py
    for key, val in params.items():
        if key in ignore:
            continue
        if isinstance(val, dict):
            g = f.create_group(key)
            saveDictToHDF5(val, g)
        elif isinstance(val, np.ndarray):
            f.create_dataset(key, data=val)
        elif isinstance(val, list):
            try:
                if isinstance(val[0], list) and isinstance(val[0][0], (int, float, INDEX, REAL)):
                    raise ValueError()
                f.create_dataset(key, data=np.array(val))
            except:
                if isinstance(val[0], list) and isinstance(val[0][0], (int, float, INDEX, REAL)):
                    g = f.create_group(key)
                    g.attrs['type'] = 'compressedList'
                    listItems = 0
                    for i in range(len(val)):
                        listItems += len(val[i])
                    indptr = uninitialized((len(val)+1), dtype=INDEX)
                    if isinstance(val[0][0], (int, INDEX)):
                        data = uninitialized((listItems), dtype=INDEX)
                    else:
                        data = uninitialized((listItems), dtype=REAL)
                    listItems = 0
                    for i in range(len(val)):
                        indptr[i] = listItems
                        data[listItems:listItems+len(val[i])] = val[i]
                        listItems += len(val[i])
                    indptr[-1] = listItems
                    g.create_dataset('indptr', data=indptr)
                    g.create_dataset('data', data=data)
                elif isinstance(val[0], str):
                    f.create_dataset(key, data=np.array(val, dtype=np.string_))
                else:
                    g = f.create_group(key)
                    g.attrs['type'] = 'list'
                    for k in range(len(val)):
                        g.attrs[str(k)] = val[k]
        elif val is None:
            try:
                f.attrs[key] = h5py.Empty(np.dtype("f"))
            except AttributeError:
                print('Failed to write \'{}\' because h5py is too old.'.format(key))
        elif hasattr(val, 'HDF5write') and callable(val.HDF5write):
            g = f.create_group(key)
            val.HDF5write(g)
        elif hasattr(val, 'toarray') and callable(val.toarray):
            f.create_dataset(key, data=val.toarray())
        else:
            try:
                f.attrs[key] = val
            except:
                try:
                    import pickle
                    f.attrs[key] = np.void(pickle.dumps(val))
                except:
                    print('Failed to write \'{}\''.format(key))
                    f.attrs[key] = str(val)


def loadDictFromHDF5(f):
    import h5py
    from . linear_operators import LinearOperator
    from PyNucleus_fem.DoFMaps import DoFMap
    params = {}
    for key in f.attrs:
        if isinstance(f.attrs[key], h5py.Empty):
            params[key] = None
        else:
            try:
                import pickle
                params[key] = pickle.loads(f.attrs[key])
            except:
                params[key] = f.attrs[key]
    for key in f:
        if isinstance(f[key], h5py.Group):
            if 'type' in f[key].attrs:
                if f[key].attrs['type'] == 'list':
                    myList = []
                    for k in range(len(f[key].attrs)-1):
                        myList.append(f[key].attrs[str(k)])
                    params[key] = myList
                elif f[key].attrs['type'] == 'compressedList':
                    myCompressedList = []
                    indptr = np.array(f[key]['indptr'], dtype=INDEX)
                    if isinstance(f[key]['data'], (int, INDEX)):
                        data = np.array(f[key]['data'], dtype=INDEX)
                    else:
                        data = np.array(f[key]['data'], dtype=REAL)
                    for i in range(len(indptr)-1):
                        myCompressedList.append(data[indptr[i]:indptr[i+1]].tolist())
                    params[key] = myCompressedList
                elif f[key].attrs['type'] == 'series':
                    d = loadDictFromHDF5(f[key])
                    grp = seriesOutputGroup(key)
                    grp.fromDict(d)
                    params[key] = grp
                elif f[key].attrs['type'] == 'DoFMap':
                    params[key] = DoFMap.HDF5read(f[key])
                elif f[key].attrs['type'] == 'h2':
                    from PyNucleus_nl.clusterMethodCy import H2Matrix
                    params[key] = H2Matrix.HDF5read(f[key])
                else:
                    params[key] = LinearOperator.HDF5read(f[key])
            elif 'vertices' in f[key] and 'cells' in f[key]:
                from PyNucleus_fem.mesh import meshNd
                params[key] = meshNd.HDF5read(f[key])
            else:
                params[key] = loadDictFromHDF5(f[key])
        else:
            params[key] = np.array(f[key])
            try:
                myList = []
                for i in range(len(params[key])):
                    myList.append(params[key][i].decode('utf-8'))
                params[key] = myList
            except:
                pass
    return params


def processDictForYaml(params):
    from PyNucleus_fem.functions import function
    paramsNew = {}
    for key in params:
        if isinstance(params[key], dict):
            paramsNew[key] = processDictForYaml(params[key])
        elif isinstance(params[key], REAL):
            paramsNew[key] = float(params[key])
        elif isinstance(params[key], np.ndarray):
            if params[key].dtype == REAL:
                if params[key].ndim == 1:
                    paramsNew[key] = params[key].tolist()
                    for i in range(len(paramsNew[key])):
                        paramsNew[key][i] = float(paramsNew[key][i])
                elif params[key].ndim == 2:
                    paramsNew[key] = params[key].tolist()
                    for i in range(len(paramsNew[key])):
                        for j in range(len(paramsNew[key][i])):
                            paramsNew[key][i][j] = float(paramsNew[key][i][j])
                else:
                    raise NotImplementedError()
            else:
                paramsNew[key] = params[key].tolist()
        elif isinstance(params[key], list):
            paramsNew[key] = params[key]
            for i in range(len(paramsNew[key])):
                if isinstance(paramsNew[key][i], REAL):
                    paramsNew[key][i] = float(paramsNew[key][i])
        elif isinstance(params[key], function):
            paramsNew[key] = str(params[key])
        else:
            paramsNew[key] = params[key]
    return paramsNew


def updateFromDefaults(params, defaults):
    for key in defaults:
        if key not in params:
            params[key] = defaults[key]
        elif isinstance(defaults[key], dict):
            updateFromDefaults(params[key], defaults[key])


def getMPIinfo(grp, verbose=False):
    from sys import modules
    if 'mpi4py.MPI' in modules:
        import mpi4py
        mpi4py.initialize = False
        from mpi4py import MPI
        if not MPI.Is_initialized():
            return
        t = {MPI.THREAD_SINGLE: 'single',
             MPI.THREAD_FUNNELED: 'funneled',
             MPI.THREAD_SERIALIZED: 'serialized',
             MPI.THREAD_MULTIPLE: 'multiple'}
        hosts = MPI.COMM_WORLD.gather(MPI.Get_processor_name())
        if MPI.COMM_WORLD.rank == 0:
            hosts = ','.join(set(hosts))
        grp.add('MPI library', '{}'.format(MPI.Get_library_version()[:-1]))
        if verbose:
            for label, value in [('MPI standard supported', MPI.Get_version()),
                                 ('Vendor', MPI.get_vendor()),
                                 ('Level of thread support', t[MPI.Query_thread()]),
                                 ('Is threaded', MPI.Is_thread_main()),
                                 ('Threads requested', mpi4py.rc.threads),
                                 ('Thread level requested', mpi4py.rc.thread_level)]:
                grp.add(label, value)
        for label, value in [('Hosts', hosts),
                             ('Communicator size', MPI.COMM_WORLD.size)]:
            grp.add(label, value)


def getEnvVariables(grp, envVars=[('OMP_NUM_THREADS', True)]):
    from os import environ
    s = []
    for var, printNotSet in envVars:
        if var in environ:
            varVal = environ[var]
        elif printNotSet:
            varVal = 'not set'
        else:
            continue
        grp.add(var, varVal)
    return '\n'.join(s)


def getSystemInfo(grp, argv=None, envVars=[('OMP_NUM_THREADS', True)]):
    from sys import executable
    if argv is not None:
        grp.add('Running', executable + ' ' + ' '.join(argv))
    else:
        grp.add('Running', executable)
    import mpi4py
    mpi4py.initialize = False
    from mpi4py import MPI
    if MPI.Is_initialized():
        getMPIinfo(grp)
    getEnvVariables(grp, envVars)
    import pkg_resources
    from PyNucleus import subpackages
    versions = {}
    for pkg in ['numpy', 'scipy', 'mpi4py', 'cython']:
        version = pkg_resources.get_distribution(pkg).version
        try:
            versions[version].append(pkg)
        except KeyError:
            versions[version] = [pkg]
    for version in versions:
        grp.add(','.join(versions[version]), version)

    import importlib

    versions = {}
    for pkg in sorted(subpackages.keys()):
        version = pkg_resources.get_distribution('PyNucleus_'+pkg).version
        module = importlib.import_module('PyNucleus_'+pkg+'.config')
        sha = module.gitSHA
        try:
            versions[(version, sha)].append(pkg)
        except KeyError:
            versions[(version, sha)] = [pkg]
    for version, sha in versions:
        grp.add('PyNucleus_'+(','.join(versions[(version, sha)])), '{}, {}'.format(version, sha))


class MPIFileHandler(logging.Handler):
    """
    A handler class which writes formatted logging records to disk files.
    """
    def __init__(self, filename, comm, mode=MPI.MODE_WRONLY | MPI.MODE_CREATE):
        from pathlib import Path
        # filename = os.fspath(filename)
        # keep the absolute path, otherwise derived classes which use this
        # may come a cropper when the current directory changes
        self.baseFilename = os.path.abspath(filename)
        assert len(self.baseFilename) <= 245, ('The length of the log file path \"{}\" is too long ' +
                                               'and will probably crash MPI. Try running with \"--disableFileLog\"').format(self.baseFilename)
        if Path(self.baseFilename).exists() and comm.rank == 0:
            from os import remove
            remove(self.baseFilename)
        self.mpiFile = MPI.File.Open(comm, self.baseFilename, mode)
        self.mpiFile.Set_atomicity(True)
        logging.Handler.__init__(self)

    def emit(self, record):
        try:
            msg = self.format(record)+'\n'
            mv = memoryview(bytes(msg, encoding='utf-8'))
            self.mpiFile.Write_shared((mv, len(mv), MPI.BYTE))
            self.flush()
        except Exception:
            self.handleError(record)

    def sync(self):
        self.mpiFile.Sync()

    def __repr__(self):
        level = logging.getLevelName(self.level)
        return '<%s %s (%s)>' % (self.__class__.__name__, self.baseFilename, level)

    def close(self):
        """
        Closes the stream.
        """
        self.acquire()
        try:
            self.mpiFile.Close()
        finally:
            self.release()


def columns(lines, returnColWidth=False, colWidth=0):
    if colWidth == 0:
        for line, _, _ in lines:
            colWidth = max(len(line), colWidth)
    s = []
    for line, f, v in lines:
        if isinstance(f, str):
            lf = '{:<'+str(colWidth+2)+'}'+f
            s.append(lf.format(line+':', v))
        else:
            lf = '{:<'+str(colWidth+2)+'}'+'{}'
            s.append(lf.format(line+':', f(v)))
    s = '\n'.join(s)
    if returnColWidth:
        return s, colWidth
    else:
        return s


class outputParam:
    def __init__(self, label, value, format=None, aTol=None, rTol=None, tested=False):
        self.label = label
        if format is None:
            if isinstance(value, bool):
                format = '{}'
            elif isinstance(value, (float, REAL)):
                format = '{:.3}'
            elif isinstance(value, (int, INDEX)):
                format = '{:,}'
            elif isinstance(value, np.ndarray):
                formatter = {'float_kind': lambda x: '{:.3}'.format(x)}
                format = lambda s: np.array2string(s, formatter=formatter, max_line_width=200)
            else:
                format = '{}'
        self.format = format
        self.value = value
        self.aTol = aTol
        self.rTol = rTol
        self.tested = tested


class outputGroup:
    def __init__(self, aTol=None, rTol=None, tested=False, driver=None):
        self.entries = []
        self.tested = tested
        self.aTol = aTol
        self.rTol = rTol
        self.driver = driver

    def add(self, label, value, format=None, aTol=None, rTol=None, tested=None):
        if aTol is None:
            aTol = self.aTol
        if rTol is None:
            rTol = self.rTol
        if tested is None:
            tested = self.tested
        p = outputParam(label, value, format, aTol, rTol, tested)
        self.entries.append(p)

    def __repr__(self):
        lines = [(p.label, p.format, p.value) for p in self.entries]
        return columns(lines)

    def log(self):
        if self.driver is not None:
            self.driver.logger.info('\n'+str(self))
        else:
            raise NotImplementedError()

    def __add__(self, other):
        c = outputGroup()
        from copy import deepcopy
        d = deepcopy(self.entries)
        d += other.entries
        c.entries = d
        return c

    def toDict(self, tested=False):
        if not tested:
            return {p.label: p.value for p in self.entries}
        else:
            return {p.label: p.value for p in self.entries if p.tested}

    def fromDict(self, d):
        for key, value in d.items():
            self.add(key, value)

    def __getattr__(self, key):
        for p in self.entries:
            if p.label == key:
                return p.value
        raise KeyError(key)

    def diff(self, d):
        result = {}
        d = deepcopy(d)
        for p in self.entries:
            if p.tested:
                if p.label in d:
                    aTol = p.aTol if p.aTol is not None else 1e-12
                    rTol = p.rTol if p.rTol is not None else 1e-12
                    if isinstance(p.value, (np.ndarray, list)):
                        if len(p.value) == len(d[p.label]):
                            if not np.allclose(p.value, d[p.label],
                                               rtol=rTol, atol=aTol):
                                result[p.label] = (p.value, d[p.label])
                        else:
                            result[p.label] = (p.value, d[p.label])
                    elif isinstance(p.value, (int, INDEX, REAL, float)):
                        if not np.allclose(p.value, d[p.label],
                                           rtol=rTol, atol=aTol) and not (np.isnan(p.value) and np.isnan(d[p.label])):
                            print(p.label, p.value, d[p.label], rTol, aTol, p.rTol, p.aTol)
                            result[p.label] = (p.value, d[p.label])
                    else:
                        if p.value != d[p.label]:
                            result[p.label] = (p.value, d[p.label])
                    d.pop(p.label)
                else:
                    result[p.label] = (p.value, 'Not available')
        for key in d:
            result[key] = ('Not available', d[key])
        return result


class statisticOutputGroup(outputGroup):
    def __init__(self, comm, driver=None):
        super(statisticOutputGroup, self).__init__(driver=driver)
        self.comm = comm
        self.doSum = {}

    def add(self, label, value, format=None, aTol=None, rTol=None, tested=None, sumOverRanks=True):
        value = self.comm.gather(value)
        if self.comm.rank == 0:
            self.doSum[label] = sumOverRanks
            super(statisticOutputGroup, self).add(label, value, format=format, aTol=aTol, rTol=rTol, tested=tested)

    def __repr__(self):
        lines = []
        header = ['quantity', 'min', 'mean', 'med', 'max', 'sum']
        for p in self.entries:
            key = p.label
            data = p.value
            if self.doSum[key]:
                lines.append((key, np.min(data), np.mean(data), np.median(data), np.max(data), np.sum(data)))
            else:
                lines.append((key, np.min(data), np.mean(data), np.median(data), np.max(data), None))
        from tabulate import tabulate
        return tabulate(lines, headers=header)


class timerOutputGroup(outputGroup):
    def __init__(self, driver=None):
        super(timerOutputGroup, self).__init__(driver=driver)

    def __repr__(self):
        lines = []
        if len(self.entries) > 0 and len(self.entries[0].value) > 1:
            header = ['timer', 'numCalls', 'minCall', 'meanCall', 'maxCall', 'minSum', 'meanSum', 'medSum', 'maxSum']
        else:
            header = ['timer', 'numCalls', 'minCall', 'meanCall', 'maxCall', 'sum']
        for p in self.entries:
            key = p.label
            data = p.value
            numCalls = np.array([p[0] for p in data])
            minNumCalls = np.min(numCalls)
            meanNumCalls = np.mean(numCalls)
            medNumCalls = np.median(numCalls)
            maxNumCalls = np.max(numCalls)

            # min over min call counts
            minCall = np.min([p[1] for p in data])
            # (\sum_{rank} numCalls*meanPerCall) / (\sum_{rank} numCalls)
            meanCall = np.sum([p[0]*p[2] for p in data])/numCalls.sum()
            # max over max call counts
            maxCall = np.max([p[4] for p in data])

            # total time per rank
            sums = [p[0]*p[2] for p in data]
            if len(sums) > 1:
                minSum = np.min(sums)
                meanSum = np.mean(sums)
                medSum = np.median(sums)
                maxSum = np.max(sums)
                if minNumCalls != maxNumCalls:
                    calls = (minNumCalls, meanNumCalls, medNumCalls, maxNumCalls)
                else:
                    calls = maxNumCalls
                lines.append((key, calls, minCall, meanCall, maxCall, minSum, meanSum, medSum, maxSum))
            else:
                lines.append((key, meanNumCalls, minCall, meanCall, maxCall, sums[0]))
        from tabulate import tabulate
        return tabulate(lines, headers=header)


class seriesOutputGroup:
    def __init__(self, name, aTol=None, rTol=None, tested=False, driver=None):
        self.name = name
        self.aTol = aTol
        self.rTol = rTol
        self.tested = tested
        self.groups = {}
        self.driver = driver

    def addGroup(self, label):
        label = str(label)
        if label in self.groups:
            group = self.groups[label]
        else:
            group = outputGroup(aTol=self.aTol, rTol=self.rTol, tested=self.tested, driver=self.driver)
            self.groups[label] = group
        return group

    def get(self, keyName, valueNames=[], sortBy=None, reverse=False):
        if sortBy is None:
            sortBy = keyName
        if not isinstance(valueNames, (list, tuple)):
            valueNames = [valueNames]
        keys = []
        values = {valueName: [] for valueName in valueNames}
        sortKeys = []
        for label in sorted(self.groups):
            try:
                key = getattr(self.groups[label], keyName)
                sortKey = getattr(self.groups[label], sortBy)
                v = {}
                for valueName in valueNames:
                    v[valueName] = getattr(self.groups[label], valueName)
                keys.append(key)
                for valueName in valueNames:
                    values[valueName].append(v[valueName])
                sortKeys.append(sortKey)
            except KeyError:
                pass
        idx = np.argsort(sortKeys)
        if reverse:
            idx = idx[::-1]
        keys = np.array(keys)[idx]
        for valueName in valueNames:
            values[valueName] = np.array(values[valueName])[idx]
        if len(valueNames) > 0:
            return keys, tuple([values[valueName] for valueName in valueNames])
        else:
            return keys

    def getPair(self, keyName, valueName, sortBy=None, reverse=False):
        if sortBy is None:
            sortBy = keyName
        keys = []
        values = []
        sortKeys = []
        for label in sorted(self.groups):
            try:
                key = getattr(self.groups[label], keyName)
                value = getattr(self.groups[label], valueName)
                sortKey = getattr(self.groups[label], sortBy)
                keys.append(key)
                values.append(value)
                sortKeys.append(sortKey)
            except KeyError:
                pass
        idx = np.argsort(sortKeys)
        if reverse:
            idx = idx[::-1]
        keys = np.array(keys)[idx]
        values = np.array(values)[idx]
        return keys, values

    def roc(self, keyName, valueName, reverse=False):
        keys, values = self.get(keyName, [valueName], reverse=reverse)
        return roc(keys, values).flatten()

    def toDict(self, tested=False):
        d = {'type': 'series'}
        for label in self.groups:
            d[label] = self.groups[label].toDict(tested)
        return d

    def fromDict(self, d):
        for label in d:
            if label != 'type':
                group = self.addGroup(label)
                group.fromDict(d[label])

    def getTable(self, keyName, valueNames=[], reverse=False):
        rocs = []
        if isinstance(valueNames, list):
            assert len(valueNames) > 0
            newValueNames = []
            for k in range(len(valueNames)):
                if valueNames[k][:3] == 'roc':
                    rocs.append(('RoC '+valueNames[k][3:],
                                 np.concatenate(([None], self.roc(keyName, valueNames[k][3:], reverse=reverse))),
                                 k))
                else:
                    newValueNames.append(valueNames[k])
            valueNames = newValueNames
        keys, values = self.get(keyName, valueNames, reverse=reverse)
        values = list(values)
        for label, value, pos in rocs:
            valueNames.insert(pos, label)
            values.insert(pos, value)
        header = [keyName]+valueNames
        lines = np.vstack((keys, *values)).T
        from tabulate import tabulate
        return tabulate(lines, headers=header)

    def plot(self, keyName, valueName, bnd=None, **kwargs):
        import matplotlib.pyplot as plt
        key, (value, ) = self.get(keyName, valueName)
        plt.plot(key, value, **kwargs)
        if bnd is not None:
            assert isinstance(bnd, dict)
            exponent = bnd.pop('exponent')
            pos = bnd.pop('pos', None)
            y = key**exponent
            if pos is None:
                pos = value.argmin()
            y *= value[pos]/y[pos]
            plt.plot(key, y, **bnd)

    def diff(self, d):
        result = {}
        for label in self.groups:
            p = self.groups[label].diff(d[label])
            if len(p) > 0:
                result[label] = p
        return result


class driverArgGroup:
    def __init__(self, parent, group):
        self.parent = parent
        self.group = group

    def add(self, *args, **kwargs):
        if self.parent is not None:
            kwargs['group'] = self.group
            self.parent.add(*args, **kwargs)


dependencyLogger = logging.getLogger('Dependencies')


class driver:
    def __init__(self, comm=None, setCommExitHandler=True, masterRank=0, description=None):
        self.comm = comm
        self._identifier = ''
        self.processHook = []
        self.masterRank = masterRank
        self.isMaster = (self.comm is None or self.comm.rank == self.masterRank)
        self.argGroups = {}
        self.outputGroups = {}
        self._logger = None
        self._timer = None
        self._figures = {}
        self._display_available = None
        if self.comm is not None and setCommExitHandler:
            exitHandler(self.comm)
        if self.isMaster:
            # self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            from os import environ
            width = int(environ.get('COLUMNS', 200))
            self.parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=50, width=width),
                                                  description=description)
            self.mainGroup = self.parser.add_argument_group('main')
            io = self.addGroup('input/output')
            io.add('disableHeader', False, help='Disable verbose header')
            io.add('displayConfig', False, help='Display configuration')
            io.add('displayRanks', False, help='Display MPI ranks in log')
            io.add('disableFileLog', False, help='Disable logging to file')
            io.add('logDependencies', False, help='log dependencies')
            io.add('disableTimeStamps', False, help='Disable time stamps in log')
            io.add('showTimers', True, help='Display timers')
            io.add('showMemory', False, help='Show memory info in timers')
            io.add('test', False, help='Run in test mode')
            io.add('yamlInput', '', help='YAML config file')
            io.add('hdf5Input', '', help='HDF5 config file')
            io.add('yamlOutput', '', help='YAML output file')
            io.add('hdf5Output', '', help='HDF5 output file')
            io.add('testCache', '', help='YAML cache file')
            io.add('overwriteCache', False, help='Overwrite the test cache file')

    def setIdentifier(self, identifier):
        self._identifier = identifier

    @property
    def identifier(self):
        return self._identifier

    def setLogFile(self, filename=None):
        if filename is None:
            from pathlib import Path as path
            logs = path('logs')
            assert self._identifier != ''
            filename = logs/(self._identifier+'.log')
            filename.parent.mkdir(exist_ok=True, parents=True)
        if self.comm is not None and self.comm.size > 1:
            fileHandler = MPIFileHandler(filename, self.comm)
        else:
            fileHandler = logging.FileHandler(filename, 'w')
        logging.getLogger().addHandler(fileHandler)
        fmt = '{asctime}  {name:40} {message}'
        fileHandler.setFormatter(logging.Formatter(fmt,
                                                   style='{',
                                                   datefmt="%Y-%m-%d %H:%M:%S"))

    def addGroup(self, name):
        if name in self.argGroups:
            return self.argGroups[name]
        if self.isMaster:
            argGroup = driverArgGroup(self, self.parser.add_argument_group(name))
            self.argGroups[name] = argGroup
            return argGroup
        else:
            return driverArgGroup(None, None)

    def add(self, name, defaultValue=None, acceptedValues=[], help='No help defined', argInterpreter=None, group=None):
        if self.isMaster:
            if group is None:
                group = self.mainGroup
            if len(acceptedValues) > 0:
                if defaultValue is None:
                    defaultValue = acceptedValues[0]
                else:
                    if defaultValue not in acceptedValues:
                        acceptedValues.insert(0, defaultValue)
            else:
                acceptedValues = None
            if isinstance(defaultValue, bool):
                if defaultValue is True:
                    action = 'store_false'
                    flagname = 'no-'+name
                elif defaultValue is False:
                    action = 'store_true'
                    flagname = name
                if len(flagname) == 1:
                    flagname = ('-'+flagname, '--'+flagname)
                else:
                    flagname = ('--'+flagname, )
                group.add_argument(*flagname,
                                   action=action,
                                   help=help,
                                   dest=name)
            else:
                if acceptedValues is not None:
                    types = [a for a in acceptedValues if type(a) == type]

                    if len(types) > 0 and argInterpreter is None:
                        acceptedValues2 = [a for a in acceptedValues if type(a) != type]

                        def argInterpreter(s):
                            from ast import literal_eval
                            if s in acceptedValues2:
                                return s
                            for t in types:
                                try:
                                    x = literal_eval(s)
                                    if isinstance(x, t):
                                        return x
                                except Exception as e:
                                    print(e)
                            raise argparse.ArgumentTypeError()

                        acceptedValues = None

                if argInterpreter is None:
                    argInterpreter = type(defaultValue)
                if len(name) == 1:
                    name = ('-'+name, '--'+name)
                else:
                    name = ('--'+name, )
                group.add_argument(*name,
                                   default=defaultValue,
                                   type=argInterpreter,
                                   choices=acceptedValues,
                                   help=help)

    def addPositional(self, name, nargs=1, group=None):
        if self.isMaster:
            if group is None:
                group = self.mainGroup
            group.add_argument(name, nargs=nargs)

    def addToProcessHook(self, fun):
        self.processHook.append(fun)

    def process(self, override={}):
        if self.isMaster:
            self.parser.set_defaults(**override)
        doTerminate = False
        if self.isMaster:
            if 'plots' in self.argGroups:
                io = self.addGroup('plots')
                io.add('plotFolder', '', help='folder for saving plots')
                io.add('plotFormat', acceptedValues=['pdf', 'png', 'jpeg', 'eps', 'ps', 'svg'], help='File format for saving plots')
            try:
                args, unknown = self.parser.parse_known_args()
            except SystemExit:
                doTerminate = True
        if self.comm:
            doTerminate = self.comm.bcast(doTerminate, root=0)
        if doTerminate:
            exit(0)
        if self.isMaster:
            if len(unknown) > 0:
                self.logger.warning('Unknown args: {}'.format(unknown))
            params = vars(args)
            if params['yamlInput'] != '':
                import yaml
                yaml_filename = params['yamlInput']
                conf = yaml.load(open(yaml_filename, 'r'), Loader=yaml.FullLoader)
                params.update(conf)
            if params['hdf5Input'] != '':
                import h5py
                hdf5_filename = params['hdf5Input']
                f = h5py.File(hdf5_filename, 'r')
                conf = loadDictFromHDF5(f)
                f.close()
                params.update(conf)
            if params['test']:
                params['displayConfig'] = True
            if self.comm:
                params['mpiGlobalCommSize'] = self.comm.size
            else:
                params['mpiGlobalCommSize'] = 1
        else:
            params = {}
        if self.comm:
            params = self.comm.bcast(params, root=0)
        self.params = params

        if params['test']:
            import psutil
            p = psutil.Process()
            try:
                p.cpu_affinity(list(range(psutil.cpu_count())))
            except AttributeError:
                pass
        self._timer = TimerManager(self.logger, comm=self.comm, memoryProfiling=params['showMemory'])

        for fun in self.processHook:
            fun(self.params)

        if self._identifier != '' and not self.params['disableFileLog']:
            self.setLogFile()

        prefix = ''
        if params['displayRanks'] and self.comm is not None and self.comm.size > 1:
            prefix = '{}: '.format(self.comm.rank)
        if not params['disableTimeStamps']:
            fmt = prefix+'{asctime}  {name:40} {message}'
        else:
            fmt = prefix+'{name:40} {message}'
        formatter = logging.Formatter(fmt=fmt,
                                      style='{',
                                      datefmt="%Y-%m-%d %H:%M:%S")
        if self.isMaster:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.WARN)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(formatter)
        if params['displayConfig']:
            from pprint import pformat
            self.logger.info('\n'+pformat(params))

        from sys import argv
        sysInfo = self.addOutputGroup('sysInfo')
        getSystemInfo(argv=argv, grp=sysInfo)
        if not params['disableHeader']:
            self.logger.info('\n'+str(sysInfo))
        if params['logDependencies']:
            dependencyLogger.setLevel(logging.DEBUG)
        else:
            dependencyLogger.setLevel(logging.INFO)
        return params

    def set(self, key, value):
        if hasattr(self, 'params'):
            self.params[key] = value
        else:
            raise KeyError

    def getLogger(self):
        if self._logger is None:
            import logging
            if self.isMaster:
                level = logging.INFO
            else:
                level = logging.WARNING
            fmt = '{asctime}  {name:40} {message}'
            logging.basicConfig(level=level,
                                format=fmt,
                                style='{',
                                datefmt="%Y-%m-%d %H:%M:%S")
            self._logger = logging.getLogger('__main__')
        return self._logger

    @property
    def logger(self):
        return self.getLogger()

    def getTimer(self):
        return self._timer

    @property
    def timer(self):
        return self.getTimer()

    def addOutputGroup(self, name, group=None, aTol=None, rTol=None, tested=False):
        if name in self.outputGroups:
            group = self.outputGroups[name]
        else:
            if group is None:
                group = outputGroup(tested=tested, aTol=aTol, rTol=rTol, driver=self)
            self.outputGroups[name] = group
        assert group.tested == tested
        assert group.aTol == aTol
        assert group.rTol == rTol
        return group

    def addStatsOutputGroup(self, name, group=None, aTol=None, rTol=None, tested=False):
        if name in self.outputGroups:
            group = self.outputGroups[name]
            assert isinstance(group, statisticOutputGroup)
            return group
        else:
            return self.addOutputGroup(name, statisticOutputGroup(comm=self.comm, driver=self))

    def addOutputSeries(self, name, aTol=None, rTol=None, tested=False):
        group = seriesOutputGroup(name, aTol, rTol, tested, driver=self)
        group = self.addOutputGroup(name, group, aTol, rTol, tested)
        return group

    def outputToDict(self, tested=False):
        d = {}
        for group in self.outputGroups:
            d[group] = self.outputGroups[group].toDict(tested=tested)
        return d

    def timerReport(self):
        t = self.addOutputGroup('Timers', timerOutputGroup())
        self.timer.setOutputGroup(self.masterRank, t)
        self.logger.info('\n'+str(t))

    def saveOutput(self):
        if self.isMaster:
            failAfterOutput = False
            if self.params['testCache'] != '':
                try:
                    import yaml
                    cache = yaml.load(open(self.params['testCache'], 'r'), Loader=yaml.FullLoader)
                    diff = {}
                    for name in self.outputGroups:
                        diff[name] = self.outputGroups[name].diff(cache.get(name, {}))
                        if len(diff[name]) == 0:
                            diff.pop(name)
                    from pprint import pformat
                    if len(diff) > 0:
                        if self.params['overwriteCache']:
                            failAfterOutput = True
                            self.params['yamlOutput'] = self.params['testCache']
                            self.logger.info('No match (observed, expected)\n' + str(pformat(diff)))
                        else:
                            assert False, 'No match (observed, expected)\n' + str(pformat(diff))
                    else:
                        self.logger.info('\nAll matched')
                except FileNotFoundError:
                    self.params['yamlOutput'] = self.params['testCache']
                    failAfterOutput = True

            if self.params['hdf5Output'] == 'auto' and self._identifier != '':
                self.params['hdf5Output'] = self._identifier + '.hdf5'
            if self.params['yamlOutput'] == 'auto' and self._identifier != '':
                self.params['yamlOutput'] = self._identifier + '.yaml'

            if self.params['hdf5Output'] != '' or self.params['yamlOutput'] != '':
                d = self.outputToDict(tested=self.params['test'])
                if not self.params['test']:
                    d.update(self.params)

            from pathlib import Path
            if self.params['hdf5Output'] != '':
                import h5py
                self.logger.info('Saving to {}'.format(self.params['hdf5Output']))
                Path(self.params['hdf5Output']).parent.mkdir(exist_ok=True, parents=True)
                f = h5py.File(self.params['hdf5Output'], 'w')
                saveDictToHDF5(d, f)
                f.close()
            if self.params['yamlOutput'] != '':
                import yaml
                self.logger.info('Saving to {}'.format(self.params['yamlOutput']))
                Path(self.params['yamlOutput']).parent.mkdir(exist_ok=True, parents=True)
                d = processDictForYaml(d)
                yaml.dump(d, open(self.params['yamlOutput'], 'w'))
            assert not failAfterOutput, 'No cache file'

    @property
    def display_available(self):
        if self._display_available is None:
            from os import environ
            available = ("DISPLAY" in environ and
                         "SSH_CONNECTION" not in environ and
                         not self.params['skipPlots'])
            if available:
                try:
                    import matplotlib.pyplot as plt
                except ImportError:
                    self.logger.warn('No Matplotlib')
                    available = False
            self._display_available = available
        return self._display_available

    def declareFigure(self, name, description='No help defined', default=True):
        if self.isMaster:
            addSkipOption = 'plots' not in self.argGroups
            plots = self.addGroup('plots')
            if addSkipOption:
                plots.add('skipPlots', False, help='Do not plot anything')
            plots.add('plot_'+name, default, help=description)
            self._figures[name] = None

    def willPlot(self, name):
        key = 'plot_'+name
        return key in self.params and self.params[key] and (self.display_available or self.params['plotFolder'] != '')

    def startPlot(self, name, **kwargs):
        if self.isMaster:
            if (('plot_'+name not in self.params or self.params['plot_'+name]) and (self.display_available or self.params['plotFolder'] != '')):
                import matplotlib.pyplot as plt
                from . plot_utils import latexOptions
                MPLconf = latexOptions(**kwargs)
                plt.rcParams.update(MPLconf)
                if (name not in self._figures) or (self._figures[name] is None):
                    fig = plt.figure()
                    self._figures[name] = fig
                    plt.get_current_fig_manager().set_window_title(name)
                else:
                    plt.figure(self._figures[name].number)
                return self._figures[name]
            elif 'plot_'+name in self.params:
                del self._figures[name]
            return None
        else:
            return None

    def savePlot(self, name, filenameSuffix='', **kwargs):
        if self._figures[name] is not None:
            if self._identifier != '':
                filename = self._identifier+'_'+name+filenameSuffix
                filename = filename.replace('_', '-')
                filename = filename.replace(' ', '-')
                filename = filename.replace('=', '')
            else:
                filename = name+filenameSuffix
            self._figures[name].tight_layout()
            from pathlib import Path
            Path(self.params['plotFolder']+'/'+filename+'.'+self.params['plotFormat']).parent.mkdir(exist_ok=True, parents=True)
            self._figures[name].savefig(self.params['plotFolder']+'/'+filename+'.'+self.params['plotFormat'], bbox_inches='tight', **kwargs)
        else:
            self.logger.warn('Figure \'{}\' not created'.format(name))

    def finishPlots(self, **kwargs):
        newFigures = {}
        for name in self._figures:
            if self._figures[name] is not None:
                newFigures[name] = self._figures[name]
        self._figures = newFigures
        if len(self._figures) > 0:
            if self.params['plotFolder'] != '':
                for name in self._figures:
                    self.savePlot(name, **kwargs)
            else:
                import matplotlib.pyplot as plt
                plt.show()

    def finish(self, **kwargs):
        t = self.addOutputGroup('Timers', timerOutputGroup())
        self.timer.setOutputGroup(self.masterRank, t)
        if self.params['showTimers'] and self.isMaster:
            self.logger.info('\n'+str(t))
        self.saveOutput()
        self.finishPlots(**kwargs)

    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return getattr(self, name)


def diffDict(d1, d2, aTol, relTol):
    diff = {}
    for key in d1:
        if isinstance(d1[key], dict):
            if key not in d2:
                p = diffDict(d1[key], {}, aTol, relTol)
                if len(p) > 0:
                    diff[key] = p
            else:
                p = diffDict(d1[key], d2[key], aTol, relTol)
                if len(p) > 0:
                    diff[key] = p
        else:
            if key not in d2:
                diff[key] = (d1[key], 'Not available')
            else:
                if isinstance(d1[key], (int, INDEX, REAL, float, np.ndarray, list)):
                    if not np.allclose(d1[key], d2[key],
                                       rtol=relTol, atol=aTol):
                        diff[key] = (d1[key], d2[key])
                elif d1[key] != d2[key]:
                    diff[key] = (d1[key], d2[key])
    for key in d2:
        if isinstance(d2[key], dict):
            if key not in d1:
                p = diffDict({}, d2[key], aTol, relTol)
                if len(p) > 0:
                    diff[key] = p
        else:
            if key not in d1:
                diff[key] = ('Not available', d2[key])
    return diff


def runDriver(path, py, python=None, timeout=900, ranks=None, cacheDir='',
              overwriteCache=False,
              aTol=1e-12, relTol=1e-2, extra=None):
    from subprocess import Popen, PIPE, TimeoutExpired
    import logging
    import os
    from pathlib import Path
    logger = logging.getLogger('__main__')
    if not isinstance(py, (list, tuple)):
        py = [py]
    autotesterOutput = Path('/home/caglusa/autotester/html')
    if autotesterOutput.exists():
        plotDir = autotesterOutput/('test-plots/'+''.join(py)+'/')
    else:
        extra = None
    if cacheDir != '':
        cache = cacheDir+'/cache_' + ''.join(py)
        runOutput = cacheDir+'/run_' + ''.join(py)
        if ranks is not None:
            cache += str(ranks)
            runOutput += str(ranks)
        py += ['--test', '--testCache={}'.format(cache)]
        if 'OVERWRITE_CACHE' in os.environ:
            overwriteCache = True
        if overwriteCache:
            py += ['--overwriteCache']
    else:
        py += ['--test']
    py += ['--disableFileLog']
    if extra is not None:
        plotDir.mkdir(exist_ok=True, parents=True)
        py += ['--plotFolder={}'.format(plotDir), '--plotFormat=png']
    else:
        py += ['--skipPlots']
    assert (Path(path)/py[0]).exists(), 'Driver \"{}\" does not exist'.format(Path(path)/py[0])
    if ranks is None:
        ranks = 1
    if python is None:
        import sys
        python = sys.executable
    cmd = [python] + py
    if 'MPIEXEC_FLAGS' in os.environ:
        mpi_flags = str(os.environ['MPIEXEC_FLAGS'])
    else:
        mpi_flags = '--bind-to none'
    cmd = ['mpiexec'] + mpi_flags.split(' ') + ['-n', str(ranks)]+cmd
    logger.info('Launching "{}" from "{}"'.format(' '.join(cmd), path))
    my_env = {}
    for key in os.environ:
        if key.find('OMPI') == -1:
            my_env[key] = os.environ[key]
    proc = Popen(cmd, cwd=path,
                 stdout=PIPE, stderr=PIPE,
                 universal_newlines=True,
                 env=my_env)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except TimeoutExpired:
        proc.kill()
        raise
    if len(stdout) > 0:
        logger.info(stdout)
    if len(stderr) > 0:
        logger.error(stderr)
    assert proc.returncode == 0, stderr+'\n\n'+stdout
    if extra is not None:
        from pytest_html import extras
        for img in plotDir.glob('*.png'):
            filename = img.relative_to(autotesterOutput)
            filename = 'http://geminga.sandia.gov:8080/'+str(filename)
            extra.append(extras.png(str(filename)))


class parametrizedArg:
    def __init__(self, name, params=[]):
        if not isinstance(params, (list, tuple)):
            params = [params]
        self.name = name
        self.params = params
        fields = []
        for p in self.params:
            if p == str:
                fields.append('[a-zA-Z]+')
            elif p == int:
                fields.append('[+-]?[0-9]+')
            elif p == float:
                fields.append('[+-]?[0-9]+\.[0-9]*')
            elif p == bool:
                fields.append('True|False')
            else:
                raise NotImplementedError()
        self.regexp = re.compile(name+'\(?'+','.join(['\s*(' + f + ')\s*' for f in fields])+'\)?')

    def match(self, s):
        return self.regexp.match(s) is not None

    def interpret(self, s):
        m = self.regexp.match(s)
        return [p(v) for v, p in zip(m.groups(), self.params)]

    def __repr__(self):
        params = []
        for p in self.params:
            params.append(p.__name__)
        return "{}({})".format(self.name, ','.join(params))


class propertyBuilder:
    def __init__(self, baseObj, fun):
        self.baseObj = baseObj
        self.fun = fun
        self.baseName = fun.__name__
        self.requiredProperties = inspect.getfullargspec(self.fun).args[1:]
        self.generatedProperties = set()
        self.cached_args = {}
        self.logLevel = logging.DEBUG

    def declareGeneratedProperty(self, prop):
        self.generatedProperties.add(prop)

    def declareGeneratedProperties(self, props):
        for prop in props:
            self.declareGeneratedProperty(prop)

    def __call__(self):
        from PyNucleus_fem.DoFMaps import fe_vector, complex_fe_vector
        cached_args = {}
        args = []
        needToBuild = False
        for prop in self.requiredProperties:
            try:
                newValue = getattr(self.baseObj, prop)
                oldValue = self.cached_args.get(prop, None)
                args.append(newValue)
                # TODO: keep hash?
                try:
                    if isinstance(newValue, np.ndarray):
                        cached_args[prop] = newValue.copy()
                        if (newValue != oldValue).any():
                            dependencyLogger.log(self.logLevel, 'Values for {} differ: \'{}\' != \'{}\', calling \'{}\''.format(prop, oldValue,
                                                                                                                                newValue, self.fun.__name__))
                            needToBuild = True
                        else:
                            dependencyLogger.log(self.logLevel, 'Values for {} are identical: \'{}\' == \'{}\''.format(prop, oldValue, newValue))
                    elif isinstance(newValue, (fe_vector, complex_fe_vector)):
                        cached_args[prop] = newValue.copy()
                        if oldValue is None or (newValue.toarray() != oldValue.toarray()).any():
                            dependencyLogger.log(self.logLevel, 'Values for {} differ: \'{}\' != \'{}\', calling \'{}\''.format(prop, oldValue,
                                                                                                                                newValue, self.fun.__name__))
                            needToBuild = True
                        else:
                            dependencyLogger.log(self.logLevel, 'Values for {} are identical: \'{}\' == \'{}\''.format(prop, oldValue, newValue))
                    elif newValue != oldValue:
                        cached_args[prop] = newValue
                        dependencyLogger.log(self.logLevel, 'Values for {} differ: \'{}\' != \'{}\', calling \'{}\''.format(prop, oldValue,
                                                                                                                            newValue, self.fun.__name__))
                        needToBuild = True
                    else:
                        dependencyLogger.log(self.logLevel, 'Values for {} are identical: \'{}\' == \'{}\''.format(prop, oldValue, newValue))
                except Exception as e:
                    dependencyLogger.log(logging.WARN,
                                         'Cannot compare values {}, {} for property \'{}\', exception {}, force call \'{}\''.format(oldValue, newValue,
                                                                                                                                    prop, e, self.fun.__name__))
                    needToBuild = True
            except AttributeError:
                raise AttributeError('Method \'{}\' has unsatisfied dependency on \'{}\''.format(self.fun.__name__, prop))
        if needToBuild:
            self.cached_args = cached_args
            self.fun(*args)
        else:
            dependencyLogger.log(self.logLevel, 'Skipping call to \'{}\''.format(self.fun.__name__))
            for prop in self.generatedProperties:
                self.baseObj.setState(prop, VALID)


VALID = 0
INVALID = 1


def generates(properties):
    def wrapper(fun):
        clsName = fun.__qualname__.split('.')[-2]
        if not isinstance(properties, (list, tuple, set)):
            props = [properties]
        else:
            props = properties
        try:
            generatorMethodsToProperties[clsName]
        except KeyError:
            generatorMethodsToProperties[clsName] = {}
        generatorMethodsToProperties[clsName][fun] = props
        return fun
    return wrapper


generatorMethodsToProperties = {}


class classWithComputedDependencies:

    generates = generates

    def __init__(self):
        # maps properties to the builder functions that depend on them
        self.requiredPropToBuilder = {}
        # maps builder functions to their produced properties
        self.builderToProps = {}
        # map properties to their builder
        self.generatedPropToBuilder = {}
        self.remoteRequiredProperties = {}
        self.remoteGeneratedProperties = {}
        self.methodToBuilder = {}

        clsNames = []
        classes = [self.__class__]
        while len(classes) > 0:
            newClasses = []
            for cls in classes:
                clsNames.append(cls.__name__)
                for parent in cls.__bases__:
                    if parent not in (classWithComputedDependencies, problem):
                        newClasses.append(parent)
            classes = newClasses

        registeredGenerators = set()
        for clsName in clsNames:
            if clsName in generatorMethodsToProperties:
                for clsMethod in generatorMethodsToProperties[clsName]:
                    methodName = clsMethod.__name__
                    if methodName in dir(self):
                        if methodName in registeredGenerators:
                            continue
                        registeredGenerators.add(methodName)
                        method = getattr(self, methodName)
                        props = generatorMethodsToProperties[clsName][clsMethod]
                        self.addProperties(props, method)

    def getState(self, prop):
        stateFlag = '__state_'+prop
        return getattr(self, stateFlag)

    def setState(self, prop, state):
        stateFlag = '__state_'+prop
        setattr(self, stateFlag, state)

    def getValue(self, prop):
        valueFlag = '__value_'+prop
        return getattr(self, valueFlag)

    def setValue(self, prop, value):
        valueFlag = '__value_'+prop
        setattr(self, valueFlag, value)

    def directlyGetWithoutChecks(self, prop):
        return self.getValue(prop)

    def directlySetWithoutChecks(self, prop, value):
        self.setState(prop, VALID)
        self.setValue(prop, value)

    @property
    def allProperties(self):
        return set(self.generatedPropToBuilder.keys()) | set(self.requiredPropToBuilder.keys())

    def addProperty(self, prop, method=None, postProcess=None):
        """Adds property that is generated by calling "method"
        The generated value is then cached."""
        if method is not None:
            assert inspect.ismethod(method), "Need the builder to be a method."
            if method in self.methodToBuilder:
                builder = self.methodToBuilder[method]
            else:
                builder = propertyBuilder(self, method)
                self.methodToBuilder[method] = builder

            try:
                self.builderToProps[builder].add(prop)
            except KeyError:
                self.builderToProps[builder] = set()
                self.builderToProps[builder].add(prop)

            builder.declareGeneratedProperty(prop)
            for reqProp in builder.requiredProperties:
                try:
                    self.requiredPropToBuilder[reqProp].add(builder)
                except KeyError:
                    self.requiredPropToBuilder[reqProp] = set()
                    self.requiredPropToBuilder[reqProp].add(builder)
        else:
            builder = None
        self.generatedPropToBuilder[prop] = builder

        self.setValue(prop, None)
        self.setState(prop, INVALID)

        def getter(self):
            builder = self.generatedPropToBuilder[prop]
            state = self.getState(prop)
            if builder is not None:
                if state == INVALID:
                    dependencyLogger.log(builder.logLevel, 'Calling \'{}\''.format(builder.baseName))
                    builder()
                    assert self.getState(prop) == VALID, 'Calling \'{}\' did not result in \'{}\' being set.'.format(builder.baseName, prop)
            else:
                assert state == VALID, 'Property \'{}\' needs to be set before it can be used.'.format(prop)
            return self.getValue(prop)

        def setter(self, value):
            # TODO: check whether we set from builder or not and don't allow setting generated properties directly

            dependencyLogger.log(logging.DEBUG, 'Setting \'{}\''.format(prop))
            self.invalidateDependencies(prop)
            if prop in self.remoteGeneratedProperties:
                for obj in self.remoteGeneratedProperties[prop]:
                    obj.invalidateDependencies(prop)
            self.setValue(prop, value)
            self.setState(prop, VALID)
            if postProcess is not None:
                postProcess()

        setattr(self.__class__, prop, property(getter, setter))

    def invalidateDependencies(self, prop):
        """Invalidate all properties that get generated by builders that depend on "prop"."""
        if prop in self.requiredPropToBuilder:
            for builder in self.requiredPropToBuilder[prop]:
                for genProp in builder.generatedProperties:
                    if self.getState(genProp) == VALID:
                        dependencyLogger.log(builder.logLevel, 'Changing {}.{} => invalidating {}.{}'.format(self.__class__.__name__,
                                                                                                             prop,
                                                                                                             self.__class__.__name__,
                                                                                                             genProp))
                    self.setState(genProp, INVALID)
                    self.invalidateDependencies(genProp)
        if prop in self.remoteGeneratedProperties:
            for obj in self.remoteGeneratedProperties[prop]:
                obj.invalidateDependencies(prop)

    def addProperties(self, properties, method=None, postProcess=None):
        """Adds multiple properties that are generated by calling "method"."""
        for prop in properties:
            self.addProperty(prop, method, postProcess=postProcess)

    def addRemoteRequiredProperty(self, obj, prop):
        """Adds a property of another object to self."""
        self.remoteRequiredProperties[prop] = obj
        obj.addRemoteGeneratedProperty(self, prop)

        def getter(self):
            obj = self.remoteRequiredProperties[prop]
            return getattr(obj, prop)

        # def setter(self, value):
        #     setattr(obj, prop, value)

        # setattr(cls, prop, property(getter, setter))
        setattr(self.__class__, prop, property(getter))

    def addRemoteRequiredProperties(self, obj, properties):
        """Adds multiple properties of another object."""
        for prop in properties:
            self.addRemoteRequiredProperty(obj, prop)

    def addRemoteGeneratedProperty(self, obj, prop):
        try:
            self.remoteGeneratedProperties[prop].add(obj)
        except KeyError:
            self.remoteGeneratedProperties[prop] = set([obj])

    def addRemote(self, obj):
        for prop in obj.allProperties:
            self.addRemoteRequiredProperty(obj, prop)

    def getGraph(self, includeRemote=False):
        import networkx as nx

        G = nx.DiGraph()
        for prop in self.allProperties:
            G.add_node(prop, color='blue')
        for builder in self.builderToProps:
            G.add_node(builder.baseName, color='green')
            for reqProp in builder.requiredProperties:
                G.add_edge(reqProp, builder.baseName)
            for prop in builder.generatedProperties:
                G.add_edge(builder.baseName, prop)

        if includeRemote:
            objs = set()
            for remoteProp in self.remoteRequiredProperties:
                G.add_node(remoteProp, color='blue')
                objs.add(self.remoteRequiredProperties[remoteProp])
            for remoteProp in self.remoteGeneratedProperties:
                G.add_node(remoteProp, color='blue')
                for obj in self.remoteGeneratedProperties[remoteProp]:
                    objs.add(obj)
            for obj in objs:
                G = nx.compose(G, obj.getGraph())

        return G

    def plot(self, includeRemote=False, layout='dot'):
        import networkx as nx
        import matplotlib.pyplot as plt
        G = self.getGraph(includeRemote)
        colors = [node[1]['color'] for node in G.nodes(data=True)]
        nx.draw(G,
                with_labels=True,
                # node_size=0,
                arrowsize=20,
                node_color=colors,
                pos=nx.nx_agraph.pygraphviz_layout(G, layout))
        plt.show()

    def changeLogLevel(self, properties, logLevel):
        for prop in properties:
            try:
                builder = self.generatedPropToBuilder[prop]
                builder.logLevel = logLevel
            except KeyError:
                pass

            try:
                for builder in self.requiredPropToBuilder[prop]:
                    builder.logLevel = logLevel
            except KeyError:
                pass


class driverAddon:
    def __init__(self, driver):
        self.driver = driver
        self._timer = None
        self.__parametrized_args__ = {}
        self.flags = []
        self.setDriverArgs()
        self.driver.addToProcessHook(self.process)

        try:
            self.driver.addGroup('input/output').add('identifier', 'auto', help='identifier used as prefix for output ("auto" sets the identifier based on invocation)')
        except argparse.ArgumentError:
            pass

    def addParametrizedArg(self, name, params=[]):
        self.__parametrized_args__[name] = parametrizedArg(name, params)

    def parametrizedArg(self, name):
        return self.__parametrized_args__[name]

    def argInterpreter(self, parametrizedArgs, acceptedValues=[]):
        from argparse import ArgumentTypeError

        def interpreter(v):
            if v in acceptedValues:
                return v
            for p in parametrizedArgs:
                if self.parametrizedArg(p).match(v):
                    return v
            raise ArgumentTypeError(("\"{}\" is not in list of accepted values {} " +
                                     "or cannot be interpreted as parametrized arg {}.").format(v, acceptedValues,
                                                                                                [repr(self.parametrizedArg(p))
                                                                                                 for p in parametrizedArgs]))

        return interpreter

    def conversionInterpreter(self):
        def interpreter(v):
            print(v)
            try:
                return int(v)
            except:
                pass
            try:
                return float(v)
            except:
                pass
            return v

        return interpreter

    def setDriverFlag(self, *args, **kwargs):
        flag = args[0]
        self.addProperty(flag)
        if 'group' in kwargs:
            group = kwargs.pop('group')
            group.add(*args, **kwargs)
        else:
            self.driver.add(*args, **kwargs)
        self.flags.append(flag)

    def setDriverArgs(self):
        pass

    @property
    def timer(self):
        if self._timer is None:
            self._timer = self.driver.getTimer()
        return self._timer

    def processCmdline(self, params):
        pass

    def getIdentifier(self, params):
        from sys import argv
        identifier = '-'.join(argv)
        pos = identifier.rfind('/')
        if pos >= 0:
            return identifier[pos+1:]
        else:
            return identifier

    def process(self, params):
        self.processCmdline(params)
        if params['identifier'] == 'auto':
            self.driver.setIdentifier(self.getIdentifier(params))
        else:
            self.driver.setIdentifier(params['identifier'])


class problem(classWithComputedDependencies,
              driverAddon):
    def __init__(self, driver):
        classWithComputedDependencies.__init__(self)
        driverAddon.__init__(self, driver)

        try:
            self.driver.addGroup('input/output').add('showDependencyGraph', False,
                                                     help="Show dependency graph of problem classes.")
            self.driver.addGroup('input/output').add('logProperties', '', help='log select properties')
        except argparse.ArgumentError:
            pass

    def processCmdline(self, params):
        driverAddon.processCmdline(self, params)
        for key in self.flags:
            self.directlySetWithoutChecks(key, params[key])
        for key in self.flags:
            params[key] = getattr(self, key)
        if params['logProperties'] != '':
            propertiesToLog = params['logProperties'].split(',')
            self.changeLogLevel(propertiesToLog, logging.INFO)

    def process(self, params):
        driverAddon.process(self, params)
        if self.driver.showDependencyGraph:
            if self.driver.isMaster:
                classWithComputedDependencies.plot(self, True)
            if self.driver.comm is not None:
                self.driver.comm.Barrier()
            exit(0)

    def __repr__(self):
        return str(type(self))
    #     lines = [(label, '{}', e) for label, e in self.__values__.items()]
    #     return columns(lines)
