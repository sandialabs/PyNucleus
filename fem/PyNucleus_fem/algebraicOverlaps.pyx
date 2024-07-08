###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes import INDEX, REAL, COMPLEX
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t
from PyNucleus_base.ip_norm cimport mydot
from PyNucleus_base import uninitialized
from . DoFMaps cimport DoFMap
from . mesh import INTERIOR_NONOVERLAPPING, INTERIOR
from . boundaryLayerCy import boundaryLayer
from PyNucleus_base.linear_operators import (LinearOperator_wrapper,
                                             diagonalOperator)
import numpy as np
cimport numpy as np
from numpy.linalg import norm

import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from sys import stdout


################################################################################
# overlap objects for DoFMaps

cdef class algebraicOverlap:
    # Tracks the algebraic overlap with the DoFMap of another subdomain.
    def __init__(self,
                 INDEX_t num_subdomain_dofs,
                 INDEX_t[::1] shared_dofs,
                 INDEX_t mySubdomainNo,
                 INDEX_t otherSubdomainNo,
                 comm,
                 INDEX_t numSharedVecs=1):
        self.num_subdomain_dofs = num_subdomain_dofs
        self.shared_dofs = shared_dofs
        self.num_shared_dofs = shared_dofs.shape[0]
        self.mySubdomainNo = mySubdomainNo
        self.otherSubdomainNo = otherSubdomainNo
        self.numSharedVecs = numSharedVecs
        self.comm = comm
        self.tagNoSend = 50
        self.tagNoRecv = 50
        self.exchangeInComplex = None
        self.exchangeOutComplex = None
        self.myExchangeInComplex = None
        self.myExchangeOutComplex = None

    def setMemory(self, REAL_t[:, ::1] exchangeIn, REAL_t[:, ::1] exchangeOut,
                  INDEX_t memOffset, INDEX_t totalMemSize):
        self.exchangeIn = exchangeIn
        self.exchangeOut = exchangeOut
        self.memOffset = memOffset
        self.totalMemSize = totalMemSize
        self.myExchangeIn = self.exchangeIn[0, self.memOffset:self.memOffset+self.num_shared_dofs]
        self.myExchangeOut = self.exchangeOut[0, self.memOffset:self.memOffset+self.num_shared_dofs]

    def setComplex(self):
        self.exchangeInComplex = uninitialized((self.exchangeIn.shape[0], self.exchangeIn.shape[1]), dtype=COMPLEX)
        self.exchangeOutComplex = uninitialized((self.exchangeOut.shape[0], self.exchangeOut.shape[1]), dtype=COMPLEX)
        self.myExchangeInComplex = self.exchangeInComplex[0, self.memOffset:self.memOffset+self.num_shared_dofs]
        self.myExchangeOutComplex = self.exchangeOutComplex[0, self.memOffset:self.memOffset+self.num_shared_dofs]

    def flushMemory(self, INDEX_t vecNo=0, REAL_t value=0.):
        self.exchangeIn[:, :] = value
        self.exchangeOut[:, :] = value

    def __repr__(self):
        return ('{} of subdomain {}' +
                ' with {} has {}/{} dofs').format(self.__class__.__name__,
                                                  self.mySubdomainNo,
                                                  self.otherSubdomainNo,
                                                  self.num_shared_dofs,
                                                  self.num_subdomain_dofs)

    cdef MPI.Request send(self,
                          const REAL_t[::1] vec,
                          INDEX_t vecNo=0,
                          flush_type flushOp=no_flush):
        cdef:
            INDEX_t j
            INDEX_t[::1] shared_dofs = self.shared_dofs
        for j in range(self.num_shared_dofs):
            self.myExchangeOut[j] = vec[shared_dofs[j]]
        self.tagNoSend += 1
        return self.comm.Isend(self.myExchangeOut,
                               dest=self.otherSubdomainNo,
                               tag=self.tagNoSend)

    cdef MPI.Request receive(self,
                             const REAL_t[::1] vec,
                             INDEX_t vecNo=0,
                             flush_type flushOp=no_flush):
        self.tagNoRecv += 1
        return self.comm.Irecv(self.myExchangeIn,
                               source=self.otherSubdomainNo,
                               tag=self.tagNoRecv)

    cdef void accumulateProcess(self,
                                REAL_t[::1] vec,
                                INDEX_t vecNo=0):
        cdef:
            INDEX_t j
            INDEX_t[::1] shared_dofs = self.shared_dofs
        for j in range(self.num_shared_dofs):
            vec[shared_dofs[j]] += self.myExchangeIn[j]

    cdef MPI.Request sendComplex(self,
                                 const COMPLEX_t[::1] vec,
                                 INDEX_t vecNo=0,
                                 flush_type flushOp=no_flush):
        cdef:
            INDEX_t j
            INDEX_t[::1] shared_dofs = self.shared_dofs
        if self.myExchangeOutComplex is None:
            self.setComplex()
        for j in range(self.num_shared_dofs):
            self.myExchangeOutComplex[j] = vec[shared_dofs[j]]
        self.tagNoSend += 1
        return self.comm.Isend(self.myExchangeOutComplex,
                               dest=self.otherSubdomainNo,
                               tag=self.tagNoSend)

    cdef MPI.Request receiveComplex(self,
                                    const COMPLEX_t[::1] vec,
                                    INDEX_t vecNo=0,
                                    flush_type flushOp=no_flush):
        if self.myExchangeInComplex is None:
            self.setComplex()
        self.tagNoRecv += 1
        return self.comm.Irecv(self.myExchangeInComplex,
                               source=self.otherSubdomainNo,
                               tag=self.tagNoRecv)

    cdef void accumulateProcessComplex(self,
                                       COMPLEX_t[::1] vec,
                                       INDEX_t vecNo=0):
        cdef:
            INDEX_t j
            INDEX_t[::1] shared_dofs = self.shared_dofs
        for j in range(self.num_shared_dofs):
            vec[shared_dofs[j]] = vec[shared_dofs[j]] + self.myExchangeInComplex[j]

    cdef void setOverlapLocal(self, REAL_t[::1] vec, INDEX_t vecNo=0):
        cdef:
            INDEX_t j
        for j in range(self.num_shared_dofs):
            vec[j] = self.myExchangeIn[j]

    cdef void uniqueProcess(self,
                            REAL_t[::1] vec,
                            INDEX_t vecNo=0):
        cdef:
            INDEX_t j
            INDEX_t[::1] shared_dofs = self.shared_dofs
            REAL_t[::1] exchangeIn = self.exchangeIn[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        for j in range(self.num_shared_dofs):
            vec[shared_dofs[j]] = exchangeIn[j]

    def HDF5write(self, node):
        compression = 'gzip'
        node.attrs['num_subdomain_dofs'] = self.num_subdomain_dofs
        node.attrs['mySubdomainNo'] = self.mySubdomainNo
        node.attrs['otherSubdomainNo'] = self.otherSubdomainNo
        node.create_dataset('shared_dofs', data=self.shared_dofs,
                            compression=compression)

    @staticmethod
    def HDF5read(node, comm):
        overlap = algebraicOverlap(node.attrs['num_subdomain_dofs'],
                                   np.array(node['shared_dofs'], dtype=INDEX),
                                   node.attrs['mySubdomainNo'],
                                   node.attrs['otherSubdomainNo'],
                                   comm)
        return overlap


cdef class algebraicOverlapPersistent(algebraicOverlap):
    def __init__(self,
                 INDEX_t num_subdomain_dofs,
                 INDEX_t[::1] shared_dofs,
                 INDEX_t mySubdomainNo,
                 INDEX_t otherSubdomainNo,
                 comm,
                 INDEX_t numSharedVecs=1):
        algebraicOverlap.__init__(self,
                                  num_subdomain_dofs,
                                  shared_dofs,
                                  mySubdomainNo,
                                  otherSubdomainNo,
                                  comm,
                                  numSharedVecs)

    def setMemory(self, REAL_t[:, ::1] exchangeIn, REAL_t[:, ::1] exchangeOut):
        super(algebraicOverlapPersistent, self).setMemory(exchangeIn, exchangeOut)
        cdef:
            INDEX_t vecNo = 0
        self.SendRequest = self.comm.Send_init(self.exchangeOut[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs],
                                               dest=self.otherSubdomainNo, tag=55)
        self.RecvRequest = self.comm.Recv_init(self.exchangeIn[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs],
                                               source=self.otherSubdomainNo, tag=55)

    cdef MPI.Request send(self,
                          const REAL_t[::1] vec,
                          INDEX_t vecNo=0,
                          flush_type flushOp=no_flush):
        cdef:
            INDEX_t j
            INDEX_t[::1] shared_dofs = self.shared_dofs
            REAL_t[::1] exchangeOut = self.exchangeOut[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        for j in range(self.num_shared_dofs):
            exchangeOut[j] = vec[shared_dofs[j]]
        self.SendRequest.Start()
        return self.SendRequest

    cdef MPI.Request receive(self,
                             const REAL_t[::1] vec,
                             INDEX_t vecNo=0,
                             flush_type flushOp=no_flush):
        self.RecvRequest.Start()
        return self.RecvRequest


cdef class algebraicOverlapBlocking(algebraicOverlap):
    def __init__(self,
                 INDEX_t num_subdomain_dofs,
                 INDEX_t[::1] shared_dofs,
                 INDEX_t mySubdomainNo,
                 INDEX_t otherSubdomainNo,
                 comm,
                 INDEX_t numSharedVecs=1):
        algebraicOverlap.__init__(self,
                                  num_subdomain_dofs,
                                  shared_dofs,
                                  mySubdomainNo,
                                  otherSubdomainNo,
                                  comm,
                                  numSharedVecs)

    def setMemory(self, REAL_t[:, ::1] exchangeIn, REAL_t[:, ::1] exchangeOut):
        super(algebraicOverlapPersistent, self).setMemory(exchangeIn, exchangeOut)
        cdef:
            INDEX_t vecNo = 0
        self.SendRequest = self.comm.Send_init(self.exchangeOut[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs],
                                               dest=self.otherSubdomainNo, tag=55)
        self.RecvRequest = self.comm.Recv_init(self.exchangeIn[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs],
                                               source=self.otherSubdomainNo, tag=55)

    cdef MPI.Request send(self,
                          const REAL_t[::1] vec,
                          INDEX_t vecNo=0,
                          flush_type flushOp=no_flush):
        cdef:
            INDEX_t j
            INDEX_t[::1] shared_dofs = self.shared_dofs
            REAL_t[::1] exchangeOut = self.exchangeOut[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        for j in range(self.num_shared_dofs):
            exchangeOut[j] = vec[shared_dofs[j]]
        self.SendRequest.Start()
        return self.SendRequest

    cdef MPI.Request receive(self,
                             const REAL_t[::1] vec,
                             INDEX_t vecNo=0,
                             flush_type flushOp=no_flush):
        self.RecvRequest.Start()
        return self.RecvRequest


cdef class algebraicOverlapOneSidedGet(algebraicOverlap):
    def __init__(self,
                 INDEX_t num_subdomain_dofs,
                 INDEX_t[::1] shared_dofs,
                 INDEX_t mySubdomainNo,
                 INDEX_t otherSubdomainNo,
                 comm,
                 INDEX_t numSharedVecs=1):
        algebraicOverlap.__init__(self,
                                  num_subdomain_dofs,
                                  shared_dofs,
                                  mySubdomainNo,
                                  otherSubdomainNo,
                                  comm,
                                  numSharedVecs)

    def setWindow(self, MPI.Win w):
        self.Window = w

    def exchangeMemOffsets(self, comm, INDEX_t tag=0):
        self.memOffsetOther = uninitialized((1), dtype=INDEX)
        self.memOffsetTemp = uninitialized((1), dtype=INDEX)
        self.memOffsetTemp[0] = self.memOffset
        return (comm.Isend(self.memOffsetTemp, dest=self.otherSubdomainNo, tag=tag),
                comm.Irecv(self.memOffsetOther, source=self.otherSubdomainNo, tag=tag))

    def flushMemory(self, INDEX_t vecNo=0, REAL_t value=0.):
        cdef:
            REAL_t[::1] exchangeOut = self.exchangeOut[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        exchangeOut[:] = value
        self.Window.Lock(self.mySubdomainNo, MPI.LOCK_EXCLUSIVE)
        self.Window.Put(exchangeOut, self.mySubdomainNo,
                        target=((vecNo*self.totalMemSize+self.memOffset)*MPI.REAL.size,
                                exchangeOut.shape[0],
                                MPI.REAL))
        self.Window.Unlock(self.mySubdomainNo)

    cdef MPI.Request send(self,
                          const REAL_t[::1] vec,
                          INDEX_t vecNo=0,
                          flush_type flushOp=no_flush):
        cdef:
            INDEX_t j
            INDEX_t[::1] shared_dofs = self.shared_dofs
            REAL_t[::1] exchangeOut = self.exchangeOut[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        for j in range(self.num_shared_dofs):
            exchangeOut[j] = vec[shared_dofs[j]]
        self.Window.Lock(self.mySubdomainNo, MPI.LOCK_EXCLUSIVE)
        self.Window.Put(exchangeOut, self.mySubdomainNo,
                        target=((vecNo*self.totalMemSize+self.memOffset)*MPI.REAL.size,
                                exchangeOut.shape[0],
                                MPI.REAL))
        self.Window.Unlock(self.mySubdomainNo)

    cdef MPI.Request receive(self,
                             const REAL_t[::1] vec,
                             INDEX_t vecNo=0,
                             flush_type flushOp=no_flush):
        cdef:
            REAL_t[::1] exchangeIn = self.exchangeIn[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        self.Window.Lock(self.otherSubdomainNo, MPI.LOCK_SHARED)
        self.Window.Get(exchangeIn, self.otherSubdomainNo,
                        target=((vecNo*self.totalMemSize+self.memOffsetOther[0])*MPI.REAL.size,
                                exchangeIn.shape[0],
                                MPI.REAL))
        self.Window.Unlock(self.otherSubdomainNo)


cdef class algebraicOverlapOneSidedPut(algebraicOverlap):
    def __init__(self,
                 INDEX_t num_subdomain_dofs,
                 INDEX_t[::1] shared_dofs,
                 INDEX_t mySubdomainNo,
                 INDEX_t otherSubdomainNo,
                 comm,
                 INDEX_t numSharedVecs=1):
        algebraicOverlap.__init__(self,
                                  num_subdomain_dofs,
                                  shared_dofs,
                                  mySubdomainNo,
                                  otherSubdomainNo,
                                  comm,
                                  numSharedVecs)

    def setWindow(self, MPI.Win w):
        self.Window = w

    def exchangeMemOffsets(self, comm, INDEX_t tag=0):
        self.memOffsetOther = uninitialized((1), dtype=INDEX)
        self.memOffsetTemp = uninitialized((1), dtype=INDEX)
        self.memOffsetTemp[0] = self.memOffset
        return (comm.Isend(self.memOffsetTemp, dest=self.otherSubdomainNo, tag=tag),
                comm.Irecv(self.memOffsetOther, source=self.otherSubdomainNo, tag=tag))

    def flushMemory(self, INDEX_t vecNo=0, REAL_t value=0.):
        cdef:
            REAL_t[::1] exchangeOut = self.exchangeOut[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        exchangeOut[:] = value
        self.Window.Lock(self.otherSubdomainNo, MPI.LOCK_SHARED)
        self.Window.Put(exchangeOut, self.mySubdomainNo,
                        target=((vecNo*self.totalMemSize+self.memOffset)*MPI.REAL.size,
                                exchangeOut.shape[0],
                                MPI.REAL))
        self.Window.Unlock(self.otherSubdomainNo)

    cdef MPI.Request send(self,
                          const REAL_t[::1] vec,
                          INDEX_t vecNo=0,
                          flush_type flushOp=no_flush):
        cdef:
            INDEX_t j
            INDEX_t[::1] shared_dofs = self.shared_dofs
            REAL_t[::1] exchangeOut = self.exchangeOut[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        for j in range(self.num_shared_dofs):
            exchangeOut[j] = vec[shared_dofs[j]]
        self.Window.Lock(self.otherSubdomainNo, MPI.LOCK_SHARED)
        self.Window.Put(exchangeOut, self.otherSubdomainNo,
                        target=((vecNo*self.totalMemSize+self.memOffsetOther[0])*MPI.REAL.size,
                                exchangeOut.shape[0],
                                MPI.REAL))
        self.Window.Unlock(self.otherSubdomainNo)

    cdef MPI.Request receive(self,
                             const REAL_t[::1] vec,
                             INDEX_t vecNo=0,
                             flush_type flushOp=no_flush):
        cdef:
            REAL_t[::1] exchangeIn = self.exchangeIn[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        self.Window.Lock(self.mySubdomainNo, MPI.LOCK_EXCLUSIVE)
        self.Window.Get(exchangeIn, self.mySubdomainNo,
                        target=((vecNo*self.totalMemSize+self.memOffset)*MPI.REAL.size,
                                exchangeIn.shape[0],
                                MPI.REAL))
        self.Window.Unlock(self.mySubdomainNo)

    cdef void accumulateProcess(self,
                                REAL_t[::1] vec,
                                INDEX_t vecNo=0):
        cdef:
            INDEX_t j
            INDEX_t[::1] shared_dofs = self.shared_dofs
            REAL_t[::1] exchangeIn = self.exchangeIn[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        for j in range(self.num_shared_dofs):
            vec[shared_dofs[j]] += exchangeIn[j]


cdef class algebraicOverlapOneSidedPutLockAll(algebraicOverlap):
    def __init__(self,
                 INDEX_t num_subdomain_dofs,
                 INDEX_t[::1] shared_dofs,
                 INDEX_t mySubdomainNo,
                 INDEX_t otherSubdomainNo,
                 comm,
                 INDEX_t numSharedVecs=1):
        algebraicOverlap.__init__(self,
                                  num_subdomain_dofs,
                                  shared_dofs,
                                  mySubdomainNo,
                                  otherSubdomainNo,
                                  comm,
                                  numSharedVecs)

    def setWindow(self, MPI.Win w):
        self.Window = w

    def exchangeMemOffsets(self, comm, INDEX_t tag=0):
        self.memOffsetOther = uninitialized((1), dtype=INDEX)
        self.memOffsetTemp = uninitialized((1), dtype=INDEX)
        self.memOffsetTemp[0] = self.memOffset
        return (comm.Isend(self.memOffsetTemp, dest=self.otherSubdomainNo, tag=tag),
                comm.Irecv(self.memOffsetOther, source=self.otherSubdomainNo, tag=tag))

    def flushMemory(self, INDEX_t vecNo=0, REAL_t value=0.):
        cdef:
            REAL_t[::1] exchangeOut = self.exchangeOut[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        exchangeOut[:] = value
        self.Window.Put(exchangeOut, self.otherSubdomainNo,
                        target=((vecNo*self.totalMemSize+self.memOffsetOther[0])*MPI.REAL.size,
                                exchangeOut.shape[0],
                                MPI.REAL))

    cdef MPI.Request send(self,
                          const REAL_t[::1] vec,
                          INDEX_t vecNo=0,
                          flush_type flushOp=flush):
        cdef:
            INDEX_t j
            INDEX_t[::1] shared_dofs = self.shared_dofs
            REAL_t[::1] exchangeOut = self.exchangeOut[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        for j in range(self.num_shared_dofs):
            exchangeOut[j] = vec[shared_dofs[j]]
        self.Window.Put(exchangeOut, self.otherSubdomainNo,
                        target=((vecNo*self.totalMemSize+self.memOffsetOther[0])*MPI.REAL.size,
                                exchangeOut.shape[0],
                                MPI.REAL))
        if flushOp == no_flush:
            pass
        elif flushOp == flush:
            self.Window.Flush(self.otherSubdomainNo)
        elif flushOp == flush_local:
            self.Window.Flush_local(self.otherSubdomainNo)
        elif flushOp == flush_local_all:
            self.Window.Flush_local_all()
        elif flushOp == flush_all:
            self.Window.Flush_all()
        else:
            raise NotImplementedError()

    cdef MPI.Request receive(self,
                             const REAL_t[::1] vec,
                             INDEX_t vecNo=0,
                             flush_type flushOp=no_flush):
        cdef:
            REAL_t[::1] exchangeIn = self.exchangeIn[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        if flushOp == no_flush:
            pass
        elif flushOp == flush:
            self.Window.Flush(self.mySubdomainNo)
        elif flushOp == flush_local:
            self.Window.Flush_local(self.mySubdomainNo)
        elif flushOp == flush_local_all:
            self.Window.Flush_local_all()
        elif flushOp == flush_all:
            self.Window.Flush_all()
        else:
            raise NotImplementedError()
        self.Window.Get(exchangeIn, self.mySubdomainNo,
                        target=((vecNo*self.totalMemSize+self.memOffset)*MPI.REAL.size,
                                exchangeIn.shape[0],
                                MPI.REAL))

    cdef void accumulateProcess(self,
                                REAL_t[::1] vec,
                                INDEX_t vecNo=0):
        cdef:
            INDEX_t j
            INDEX_t[::1] shared_dofs = self.shared_dofs
            REAL_t[::1] exchangeIn = self.exchangeIn[vecNo, self.memOffset:self.memOffset+self.num_shared_dofs]
        for j in range(self.num_shared_dofs):
            vec[shared_dofs[j]] += exchangeIn[j]


cdef class algebraicOverlapManager:
    # Tracks the algebraic overlap with the DoFMaps of all other subdomains.
    def __init__(self, numSubdomains, num_subdomain_dofs, comm):
        self.numSubdomains = numSubdomains
        self.num_subdomain_dofs = num_subdomain_dofs
        self.overlaps = {}
        self.mySubdomainNo = comm.rank
        self.comm = comm
        self.requestsSend = []
        self._max_cross = 0
        self.distribute_is_prepared = False
        self.non_overlapping_distribute_is_prepared = False

    def setComplex(self):
        for subdomain in self.overlaps:
            self.overlaps[subdomain].setComplex()

    def get_max_cross(self):
        if self._max_cross > 0:
            return self._max_cross
        else:
            return None

    max_cross = property(fget=get_max_cross)

    def get_num_shared_dofs(self, unique=False):
        if unique:
            return self.get_shared_dofs().shape[0]
        else:
            num_dofs = 0
            for otherSubdomain in self.overlaps:
                num_dofs += self.overlaps[otherSubdomain].num_shared_dofs
            return num_dofs

    num_shared_dofs = property(fget=get_num_shared_dofs)

    def get_shared_dofs(self):
        shared_dofs = set()
        for otherSubdomain in self.overlaps:
            shared_dofs |= set(list(np.array(self.overlaps[otherSubdomain].shared_dofs)))
        return np.array(list(shared_dofs), dtype=INDEX)

    def prepareDistribute(self):
        cdef:
            INDEX_t i, subdomainNo, k
            dict dofCount = {}
            INDEX_t[::1] Didx
            REAL_t[::1] Dval
        for subdomainNo in self.overlaps:
            for dof in self.overlaps[subdomainNo].shared_dofs:
                try:
                    dofCount[dof] += 1
                except KeyError:
                    dofCount[dof] = 2
        self.Didx = uninitialized((len(dofCount)), dtype=INDEX)
        self.Dval = uninitialized((len(dofCount)), dtype=REAL)
        Didx = self.Didx
        Dval = self.Dval
        k = 0
        for i in dofCount:
            Didx[k] = i
            Dval[k] = 1.0/dofCount[i]
            k += 1
        self.distribute_is_prepared = True

    def prepareDistributeRepartitionSend(self, DoFMap dm):
        cdef:
            REAL_t[::1] x
        x = np.ones((dm.num_dofs), dtype=REAL)
        self.send(x, asynchronous=False)

    def prepareDistributeRepartition(self, DoFMap dm, BOOL_t doSend=True):
        cdef:
            INDEX_t i
            INDEX_t dofCount = 0
            REAL_t[::1] y
        if doSend:
            self.prepareDistributeRepartitionSend(dm)
        y = np.zeros((dm.num_dofs), dtype=REAL)
        self.receive(y, asynchronous=False)
        for i in range(dm.num_dofs):
            if abs(y[i]-1.0) > 1e-8:
                dofCount += 1
        assert np.array(y, copy=False).min() >= 1.0, (self.mySubdomainNo, np.array(y, copy=False), np.array(y, copy=False).min())
        self.Didx = uninitialized((dofCount), dtype=INDEX)
        self.Dval = uninitialized((dofCount), dtype=REAL)
        dofCount = 0
        for i in range(dm.num_dofs):
            if abs(y[i]-1.0) > 1e-8:
                self.Didx[dofCount] = i
                self.Dval[dofCount] = 1.0/y[i]
                dofCount += 1
        self.distribute_is_prepared = True

    # returns 1-(distance from original subdomain)/meshsize
    # only sets the vertex dofs
    def getProtoPartition(self, REAL_t[::1] x, DoFMap dm, vertexLayers, INDEX_t depth):
        cdef:
            INDEX_t d, cellNo, vertexNo, dof
        for d in range(1, len(vertexLayers)+1):
            for cellNo, vertexNo in vertexLayers[d-1]:
                dof = dm.cell2dof(cellNo, vertexNo*dm.dofs_per_vertex)
                if dof >= 0:
                    if depth > 1:
                        x[dof] = max(1.0 - (<REAL_t>d)/(<REAL_t>(depth-1)), 0.)
                    else:
                        x[dof] = 0.
        # return x

    # returns 1 on initial domain, 0 otherwise
    # only sets the vertex dofs
    def getProtoPartitionNonOverlapping(self, REAL_t[::1] x, DoFMap dm, vertexLayers, INDEX_t depth):
        cdef:
            INDEX_t d, cellNo, vertexNo, dof
        for d in range(1, len(vertexLayers)+1):
            for cellNo, vertexNo in vertexLayers[d-1]:
                dof = dm.cell2dof(cellNo, vertexNo*dm.dofs_per_vertex)
                if dof >= 0:
                    x[dof] = 0.

    def prepareDistributeMeshOverlap(self, mesh, INDEX_t nc, DoFMap DoFMap, INDEX_t depth, meshOverlaps):

        # returns layers of vertices around the original subdomain
        def getVertexLayers(subdomain):
            cdef:
                dict v2c
                list boundaryVertices2
                set alreadyAdded, boundaryVertices
                set boundaryCellsSet
                list boundaryCellsList
                INDEX_t[::1] cellLayer
                INDEX_t v, cell, vertexNo, localVertexNo
                INDEX_t[:, ::1] cells = subdomain.cells

            boundaryCellsSet = set()
            boundaryVertices2 = []
            dim = subdomain.dim
            if dim == 1:
                alreadyAdded = set(list(subdomain.getBoundaryVerticesByTag([INTERIOR_NONOVERLAPPING]).ravel()))
                boundaryVertices = set(list(subdomain.getBoundaryVerticesByTag([INTERIOR_NONOVERLAPPING]).ravel()))
            elif dim == 2:
                alreadyAdded = set(list(subdomain.getBoundaryEdgesByTag([INTERIOR_NONOVERLAPPING]).ravel()))
                boundaryVertices = set(list(subdomain.getBoundaryEdgesByTag([INTERIOR_NONOVERLAPPING]).ravel()))
            elif dim == 3:
                alreadyAdded = set(list(subdomain.getBoundaryFacesByTag([INTERIOR_NONOVERLAPPING]).ravel()))
                boundaryVertices = set(list(subdomain.getBoundaryFacesByTag([INTERIOR_NONOVERLAPPING]).ravel()))
            else:
                raise NotImplementedError()

            exteriorBL = boundaryLayer(subdomain, depth,
                                       afterRefinements=0, startCell=nc)
            v2c = exteriorBL.vertex2cells(subdomain.cells[nc:, :])

            for v in boundaryVertices:
                boundaryCellsSet |= set(v2c[v])
            boundaryCellsList = exteriorBL.getLayer(depth, boundaryCellsSet, True, subdomain.cells[nc:, :])
            for cellLayer in boundaryCellsList:
                boundaryVertices2.append([])
                for cell in cellLayer:
                    for vertexNo in range(cells.shape[1]):
                        localVertexNo = cells[nc+cell, vertexNo]
                        if localVertexNo not in alreadyAdded:
                            boundaryVertices2[len(boundaryVertices2)-1].append((nc+cell, vertexNo))
                            alreadyAdded.add(localVertexNo)
            return boundaryVertices2

        # get linear interpolant in all dofs (the proto-partition is only given in the vertex dofs)
        def linearInterpolant(x):
            cdef:
                INDEX_t cellNo, vertexNo, i, dof
                REAL_t[::1] corner_vals
            # the values in the vertices
            corner_vals = uninitialized((mesh.dim+1), dtype=REAL)
            # linear interpolation for all other dofs
            for cellNo in sorted(overlapCells, reverse=True):
                # get values in the vertices
                for vertexNo in range(mesh.dim+1):
                    dof = DoFMap.cell2dof(cellNo, vertexNo*DoFMap.dofs_per_vertex)
                    if dof >= 0:
                        corner_vals[vertexNo] = x[dof]
                    elif cellNo < nc:
                        corner_vals[vertexNo] = 1.
                    else:
                        corner_vals[vertexNo] = 0.
                # set the correct linear interpolant
                for i in range(DoFMap.dofs_per_element):
                    dof = DoFMap.cell2dof(cellNo, i)
                    if dof >= 0:
                        x[dof] = mydot(DoFMap.nodes[i, :], corner_vals)

        def cutOff(x):
            cdef:
                INDEX_t cellNo, vertexNo, dof, i
                REAL_t[::1] corner_vals
            corner_vals = uninitialized((mesh.dim+1), dtype=REAL)
            for cellNo in sorted(overlapCells, reverse=True):
                for vertexNo in range(mesh.dim+1):
                    dof = DoFMap.cell2dof(cellNo, vertexNo*DoFMap.dofs_per_vertex)
                    if dof >= 0:
                        corner_vals[vertexNo] = x[dof]
                    elif cellNo < nc:
                        corner_vals[vertexNo] = 1.
                    else:
                        corner_vals[vertexNo] = 0.
                for i in range((mesh.dim+1)*DoFMap.dofs_per_vertex, DoFMap.dofs_per_element):
                    dof = DoFMap.cell2dof(cellNo, i)
                    if dof >= 0:
                        if mydot(DoFMap.nodes[i, :], corner_vals) < 1.0:
                            x[dof] = 0.
                        else:
                            x[dof] = 1.

        cdef:
            INDEX_t k, dof, subdomainNo, cellNo, m
            INDEX_t[::1] Didx
            REAL_t[::1] Dval
            dict overlapCells = {}
            REAL_t[::1] x, y
            set sharedDofs

        for subdomainNo in meshOverlaps.overlaps:
            for cellNo in meshOverlaps.overlaps[subdomainNo].cells:
                try:
                    overlapCells[cellNo] += 1
                except KeyError:
                    overlapCells[cellNo] = 2
        m = 0
        for cellNo in overlapCells:
            m = max(m, overlapCells[cellNo])
        self._max_cross = m

        vertexLayers = getVertexLayers(mesh)
        x = np.ones((DoFMap.num_dofs), dtype=REAL)
        self.getProtoPartition(x, DoFMap, vertexLayers, depth)
        linearInterpolant(x)

        y = uninitialized((DoFMap.num_dofs), dtype=REAL)
        self.accumulate(x, y, asynchronous=False)
        assert np.all(np.absolute(y) > 0), (np.array(x), np.array(y))
        for m in range(x.shape[0]):
            x[m] /= y[m]
        assert np.all(np.isfinite(x)), np.array(x)

        sharedDofs = set()
        for subdomainNo in self.overlaps:
            for dof in self.overlaps[subdomainNo].shared_dofs:
                sharedDofs.add(dof)
        self.Didx = uninitialized((len(sharedDofs)), dtype=INDEX)
        self.Dval = uninitialized((len(sharedDofs)), dtype=REAL)
        Didx = self.Didx
        Dval = self.Dval
        k = 0
        for dof in sharedDofs:
            Didx[k] = dof
            Dval[k] = x[dof]
            k += 1
        self.distribute_is_prepared = True

        x[:] = 1.0
        self.getProtoPartitionNonOverlapping(x, DoFMap, vertexLayers, depth)
        cutOff(x)
        self.accumulate(x, y, asynchronous=False)
        for m in range(x.shape[0]):
            x[m] /= y[m]
        assert np.all(np.isfinite(x)), np.array(x)

        self.DidxNonOverlapping = uninitialized((len(sharedDofs)), dtype=INDEX)
        self.DvalNonOverlapping = uninitialized((len(sharedDofs)), dtype=REAL)
        Didx = self.DidxNonOverlapping
        Dval = self.DvalNonOverlapping
        k = 0
        for dof in sharedDofs:
            Didx[k] = dof
            Dval[k] = x[dof]
            k += 1
        self.non_overlapping_distribute_is_prepared = True

    cpdef void distribute(self,
                          REAL_t[::1] vec,
                          REAL_t[::1] vec2=None,
                          BOOL_t nonOverlapping=False,
                          INDEX_t level=0):
        """
        Distribute an accumulated vector.
        """
        cdef:
            INDEX_t[::1] Didx
            REAL_t[::1] Dval
            INDEX_t i, dof, n = self.num_subdomain_dofs
        if nonOverlapping:
            assert self.non_overlapping_distribute_is_prepared, "Non-overlapping distribute has not been prepared for this algebraic overlap."
            Didx = self.DidxNonOverlapping
            Dval = self.DvalNonOverlapping
        else:
            assert self.distribute_is_prepared, "Distribute has not been prepared for this algebraic overlap."
            Didx = self.Didx
            Dval = self.Dval

        if vec2 is None:
            vec2 = vec
        else:
            for dof in range(n):
                vec2[dof] = vec[dof]
        for i in range(Didx.shape[0]):
            dof = Didx[i]
            vec2[dof] *= Dval[i]

    def distribute_py(self, vec, vec2=None, nonOverlapping=False):
        self.distribute(vec, vec2, nonOverlapping)

    cdef void distributeComplex(self,
                                COMPLEX_t[::1] vec,
                                COMPLEX_t[::1] vec2=None,
                                BOOL_t nonOverlapping=False):
        """
        Distribute an accumulated vector.
        """
        cdef:
            INDEX_t[::1] Didx
            REAL_t[::1] Dval
            INDEX_t i, dof, n = self.num_subdomain_dofs
        if nonOverlapping:
            assert self.non_overlapping_distribute_is_prepared, "Non-overlapping distribute has not been prepared for this algebraic overlap."
            Didx = self.DidxNonOverlapping
            Dval = self.DvalNonOverlapping
        else:
            assert self.distribute_is_prepared, "Distribute has not been prepared for this algebraic overlap."
            Didx = self.Didx
            Dval = self.Dval

        if vec2 is None:
            vec2 = vec
        else:
            for dof in range(n):
                vec2[dof] = vec[dof]
        for i in range(Didx.shape[0]):
            dof = Didx[i]
            vec2[dof] = vec2[dof]*Dval[i]

    cdef void redistribute(self,
                           REAL_t[::1] vec,
                           REAL_t[::1] vec2=None,
                           BOOL_t nonOverlapping=False,
                           BOOL_t asynchronous=False,
                           INDEX_t vecNo=0):
        if vec2 is None:
            self.accumulate(vec, None, asynchronous=asynchronous, vecNo=vecNo)
            self.distribute(vec, None, nonOverlapping=nonOverlapping)
        else:
            self.accumulate(vec, vec2, asynchronous=asynchronous, vecNo=vecNo)
            self.distribute(vec2, None, nonOverlapping=nonOverlapping)

    cpdef void accumulate(self,
                          REAL_t[::1] vec,
                          REAL_t[::1] return_vec=None,
                          BOOL_t asynchronous=False,
                          INDEX_t vecNo=0,
                          INDEX_t level=0):
        """
        Exchange information in the overlap.
        """
        cdef:
            INDEX_t j, subdomainNo
            list requestsReceive = [], requestsSend = self.requestsSend, requestsOneSidedGet, requestsOneSidedPut
            list requestsOneSidedPutLockAll
            BOOL_t setBarrier
            algebraicOverlap ov
            algebraicOverlapOneSidedGet ov1S
            algebraicOverlapOneSidedPut ov1SP
            algebraicOverlapOneSidedPutLockAll ov1SPLA
        MPI.Request.Waitall(requestsSend)
        del requestsSend[:]
        requestsOneSidedGet = []
        requestsOneSidedPut = []
        requestsOneSidedPutLockAll = []
        for subdomainNo in self.overlaps:
            # FIX here
            if isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedGet):
                ov1S = self.overlaps[subdomainNo]
                ov1S.send(vec, vecNo=vecNo)
                requestsOneSidedGet.append(subdomainNo)
            elif isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedPut):
                ov1SP = self.overlaps[subdomainNo]
                ov1SP.send(vec, vecNo=vecNo)
                requestsOneSidedPut.append(subdomainNo)
            elif isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedPutLockAll):
                ov1SPLA = self.overlaps[subdomainNo]
                ov1SPLA.send(vec, vecNo=vecNo)
                requestsOneSidedPutLockAll.append(subdomainNo)
            else:
                ov = self.overlaps[subdomainNo]
                requestsSend.append(ov.send(vec, vecNo=vecNo))
                requestsReceive.append(ov.receive(vec, vecNo=vecNo))
        # if len(requestsOneSidedPutLockAll) > 0:
        #     self.Window.Flush_all()

        if return_vec is None:
            return_vec = vec
        else:
            for j in range(vec.shape[0]):
                return_vec[j] = vec[j]

        if ((len(requestsOneSidedGet) > 0 or
             len(requestsOneSidedPut) > 0 or
             len(requestsOneSidedPutLockAll) > 0) and not asynchronous):
            setBarrier = True
            self.comm.Barrier()
        else:
            setBarrier = False

        while len(requestsReceive) > 0:
            status = MPI.Status()
            done = MPI.Request.Waitany(requestsReceive, status)
            assert status.error == 0
            requestsReceive.pop(done)
            subdomainNo = status.source
            ov = self.overlaps[subdomainNo]
            ov.accumulateProcess(return_vec, vecNo=vecNo)
        for subdomainNo in requestsOneSidedGet:
            ov1S = self.overlaps[subdomainNo]
            ov1S.receive(return_vec, vecNo=vecNo)
            ov1S.accumulateProcess(return_vec, vecNo=vecNo)
        for subdomainNo in requestsOneSidedPut:
            ov1SP = self.overlaps[subdomainNo]
            ov1SP.receive(return_vec, vecNo=vecNo)
            ov1SP.accumulateProcess(return_vec, vecNo=vecNo)
        for subdomainNo in requestsOneSidedPutLockAll:
            ov1SPLA = self.overlaps[subdomainNo]
            ov1SPLA.receive(return_vec, vecNo=vecNo)
            ov1SPLA.accumulateProcess(return_vec, vecNo=vecNo)
        if setBarrier:
            self.comm.Barrier()

    def accumulate_py(self,
                      vec,
                      return_vec=None,
                      asynchronous=False,
                      vecNo=0):
        self.accumulate(vec, return_vec, asynchronous, vecNo)

    cdef void accumulateComplex(self,
                                COMPLEX_t[::1] vec,
                                COMPLEX_t[::1] return_vec=None,
                                BOOL_t asynchronous=False,
                                INDEX_t vecNo=0):
        """
        Exchange information in the overlap.
        """
        cdef:
            INDEX_t j, subdomainNo
            list requestsReceive = [], requestsSend = self.requestsSend, requestsOneSidedGet, requestsOneSidedPut
            list requestsOneSidedPutLockAll
            BOOL_t setBarrier
            algebraicOverlap ov
            algebraicOverlapOneSidedGet ov1S
            algebraicOverlapOneSidedPut ov1SP
            algebraicOverlapOneSidedPutLockAll ov1SPLA
        MPI.Request.Waitall(requestsSend)
        del requestsSend[:]
        requestsOneSidedGet = []
        requestsOneSidedPut = []
        requestsOneSidedPutLockAll = []
        for subdomainNo in self.overlaps:
            # FIX here
            if isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedGet):
                ov1S = self.overlaps[subdomainNo]
                ov1S.sendComplex(vec, vecNo=vecNo)
                requestsOneSidedGet.append(subdomainNo)
            elif isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedPut):
                ov1SP = self.overlaps[subdomainNo]
                ov1SP.sendComplex(vec, vecNo=vecNo)
                requestsOneSidedPut.append(subdomainNo)
            elif isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedPutLockAll):
                ov1SPLA = self.overlaps[subdomainNo]
                ov1SPLA.sendComplex(vec, vecNo=vecNo)
                requestsOneSidedPutLockAll.append(subdomainNo)
            else:
                ov = self.overlaps[subdomainNo]
                requestsSend.append(ov.sendComplex(vec, vecNo=vecNo))
                requestsReceive.append(ov.receiveComplex(vec, vecNo=vecNo))
        # if len(requestsOneSidedPutLockAll) > 0:
        #     self.Window.Flush_all()

        if return_vec is None:
            return_vec = vec
        else:
            for j in range(vec.shape[0]):
                return_vec[j] = vec[j]

        if ((len(requestsOneSidedGet) > 0 or
             len(requestsOneSidedPut) > 0 or
             len(requestsOneSidedPutLockAll) > 0) and not asynchronous):
            setBarrier = True
            self.comm.Barrier()
        else:
            setBarrier = False

        while len(requestsReceive) > 0:
            status = MPI.Status()
            done = MPI.Request.Waitany(requestsReceive, status)
            assert status.error == 0
            requestsReceive.pop(done)
            subdomainNo = status.source
            ov = self.overlaps[subdomainNo]
            ov.accumulateProcessComplex(return_vec, vecNo=vecNo)
        for subdomainNo in requestsOneSidedGet:
            ov1S = self.overlaps[subdomainNo]
            ov1S.receiveComplex(return_vec, vecNo=vecNo)
            ov1S.accumulateProcessComplex(return_vec, vecNo=vecNo)
        for subdomainNo in requestsOneSidedPut:
            ov1SP = self.overlaps[subdomainNo]
            ov1SP.receiveComplex(return_vec, vecNo=vecNo)
            ov1SP.accumulateProcessComplex(return_vec, vecNo=vecNo)
        for subdomainNo in requestsOneSidedPutLockAll:
            ov1SPLA = self.overlaps[subdomainNo]
            ov1SPLA.receiveComplex(return_vec, vecNo=vecNo)
            ov1SPLA.accumulateProcessComplex(return_vec, vecNo=vecNo)
        if setBarrier:
            self.comm.Barrier()

    def unique(self, REAL_t[::1] vec, INDEX_t vecNo=0):
        """
        Return an accumulated vector by taking values from the highest
        rank.
        """
        cdef:
            INDEX_t i, subdomainNo
            dict requestsReceive = {}
            list requestsSend = []
            algebraicOverlap ov
        for subdomainNo in self.overlaps:
            # FIX: This is inefficient, we would only need to send
            # from the highest rank, not all higher ranks..
            ov = self.overlaps[subdomainNo]
            if subdomainNo < self.comm.rank:
                requestSend = ov.send(vec, vecNo=vecNo)
                requestsSend.append(requestSend)
            else:
                requestReceive = ov.receive(vec, vecNo=vecNo)
                requestsReceive[subdomainNo] = requestReceive

        for subdomainNo in self.overlaps:
            ov = self.overlaps[subdomainNo]
            if subdomainNo < self.comm.rank:
                continue
            status = MPI.Status()
            done = MPI.Request.Wait(requestsReceive[subdomainNo], status)
            i = status.source
            ov.uniqueProcess(vec, vecNo=vecNo)
        MPI.Request.Waitall(requestsSend)

    cdef void send(self,
                   REAL_t[::1] vec,
                   BOOL_t asynchronous=False,
                   INDEX_t vecNo=0,
                   flush_type flushOp=flush):
        """
        Send information in the overlap.
        """
        cdef:
            INDEX_t subdomainNo
            list requestsSend = self.requestsSend, requestsOneSidedGet, requestsOneSidedPut
            list requestsOneSidedPutLockAll
            algebraicOverlap ov
            algebraicOverlapOneSidedGet ov1S
            algebraicOverlapOneSidedPut ov1SP
            algebraicOverlapOneSidedPutLockAll ov1SPLA
        MPI.Request.Waitall(requestsSend)
        del requestsSend[:]
        requestsOneSidedGet = []
        requestsOneSidedPut = []
        requestsOneSidedPutLockAll = []
        for subdomainNo in self.overlaps:
            if isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedGet):
                ov1S = self.overlaps[subdomainNo]
                ov1S.send(vec, vecNo=vecNo)
                requestsOneSidedGet.append(subdomainNo)
            elif isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedPut):
                ov1SP = self.overlaps[subdomainNo]
                ov1SP.send(vec, vecNo=vecNo)
                requestsOneSidedPut.append(subdomainNo)
            elif isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedPutLockAll):
                ov1SPLA = self.overlaps[subdomainNo]
                ov1SPLA.send(vec, vecNo=vecNo, flushOp=flushOp)
                requestsOneSidedPutLockAll.append(subdomainNo)
            else:
                ov = self.overlaps[subdomainNo]
                requestsSend.append(ov.send(vec, vecNo=vecNo))

        if ((len(requestsOneSidedGet) > 0 or
             len(requestsOneSidedPut) > 0 or
             len(requestsOneSidedPutLockAll) > 0) and not asynchronous):
            self.comm.Barrier()

    def send_py(self, vec, asynchronous=False, vecNo=0):
        self.send(vec, asynchronous, vecNo)

    cdef void sendComplex(self,
                          COMPLEX_t[::1] vec,
                          BOOL_t asynchronous=False,
                          INDEX_t vecNo=0,
                          flush_type flushOp=flush):
        """
        Send information in the overlap.
        """
        cdef:
            INDEX_t subdomainNo
            list requestsSend = self.requestsSend, requestsOneSidedGet, requestsOneSidedPut
            list requestsOneSidedPutLockAll
            algebraicOverlap ov
            algebraicOverlapOneSidedGet ov1S
            algebraicOverlapOneSidedPut ov1SP
            algebraicOverlapOneSidedPutLockAll ov1SPLA
        MPI.Request.Waitall(requestsSend)
        del requestsSend[:]
        requestsOneSidedGet = []
        requestsOneSidedPut = []
        requestsOneSidedPutLockAll = []
        for subdomainNo in self.overlaps:
            if isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedGet):
                ov1S = self.overlaps[subdomainNo]
                ov1S.sendComplex(vec, vecNo=vecNo)
                requestsOneSidedGet.append(subdomainNo)
            elif isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedPut):
                ov1SP = self.overlaps[subdomainNo]
                ov1SP.sendComplex(vec, vecNo=vecNo)
                requestsOneSidedPut.append(subdomainNo)
            elif isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedPutLockAll):
                ov1SPLA = self.overlaps[subdomainNo]
                ov1SPLA.sendComplex(vec, vecNo=vecNo, flushOp=flushOp)
                requestsOneSidedPutLockAll.append(subdomainNo)
            else:
                ov = self.overlaps[subdomainNo]
                requestsSend.append(ov.sendComplex(vec, vecNo=vecNo))

        if ((len(requestsOneSidedGet) > 0 or
             len(requestsOneSidedPut) > 0 or
             len(requestsOneSidedPutLockAll) > 0) and not asynchronous):
            self.comm.Barrier()

    cdef void receive(self,
                      REAL_t[::1] return_vec,
                      BOOL_t asynchronous=False,
                      INDEX_t vecNo=0,
                      flush_type flushOp=no_flush):
        """
        Exchange information in the overlap.
        """
        cdef:
            INDEX_t subdomainNo
            list requestsReceive = [], requestsOneSidedGet, requestsOneSidedPut
            list requestsOneSidedPutLockAll
            BOOL_t setBarrier
            algebraicOverlap ov
            algebraicOverlapOneSidedGet ov1S
            algebraicOverlapOneSidedPut ov1SP
            algebraicOverlapOneSidedPutLockAll ov1SPLA
        requestsOneSidedGet = []
        requestsOneSidedPut = []
        requestsOneSidedPutLockAll = []
        for subdomainNo in self.overlaps:
            # FIX here
            if isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedGet):
                ov1S = self.overlaps[subdomainNo]
                requestsOneSidedGet.append(subdomainNo)
            elif isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedPut):
                ov1SP = self.overlaps[subdomainNo]
                requestsOneSidedPut.append(subdomainNo)
            elif isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedPutLockAll):
                ov1SPLA = self.overlaps[subdomainNo]
                requestsOneSidedPutLockAll.append(subdomainNo)
            else:
                ov = self.overlaps[subdomainNo]
                requestsReceive.append(ov.receive(return_vec, vecNo=vecNo))

        if ((len(requestsOneSidedGet) > 0 or
             len(requestsOneSidedPut) > 0 or
             len(requestsOneSidedPutLockAll) > 0) and not asynchronous):
            setBarrier = True
        else:
            setBarrier = False

        while len(requestsReceive) > 0:
            status = MPI.Status()
            done = MPI.Request.Waitany(requestsReceive, status)
            assert status.error == 0
            requestsReceive.pop(done)
            subdomainNo = status.source
            ov = self.overlaps[subdomainNo]
            ov.accumulateProcess(return_vec, vecNo=vecNo)
        for subdomainNo in requestsOneSidedGet:
            ov1S = self.overlaps[subdomainNo]
            ov1S.receive(return_vec, vecNo=vecNo)
            ov1S.accumulateProcess(return_vec, vecNo=vecNo)
        for subdomainNo in requestsOneSidedPut:
            ov1SP = self.overlaps[subdomainNo]
            ov1SP.receive(return_vec, vecNo=vecNo)
            ov1SP.accumulateProcess(return_vec, vecNo=vecNo)
        for subdomainNo in requestsOneSidedPutLockAll:
            ov1SPLA = self.overlaps[subdomainNo]
            ov1SPLA.receive(return_vec, vecNo=vecNo, flushOp=flushOp)
            ov1SPLA.accumulateProcess(return_vec, vecNo=vecNo)
        if setBarrier:
            self.comm.Barrier()

    def receive_py(self, return_vec, asynchronous=False, vecNo=0):
        self.receive(return_vec, asynchronous, vecNo)

    cdef void receiveComplex(self,
                             COMPLEX_t[::1] return_vec,
                             BOOL_t asynchronous=False,
                             INDEX_t vecNo=0,
                             flush_type flushOp=no_flush):
        """
        Exchange information in the overlap.
        """
        cdef:
            INDEX_t subdomainNo
            list requestsReceive = [], requestsOneSidedGet, requestsOneSidedPut
            list requestsOneSidedPutLockAll
            BOOL_t setBarrier
            algebraicOverlap ov
            algebraicOverlapOneSidedGet ov1S
            algebraicOverlapOneSidedPut ov1SP
            algebraicOverlapOneSidedPutLockAll ov1SPLA
        requestsOneSidedGet = []
        requestsOneSidedPut = []
        requestsOneSidedPutLockAll = []
        for subdomainNo in self.overlaps:
            # FIX here
            if isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedGet):
                ov1S = self.overlaps[subdomainNo]
                requestsOneSidedGet.append(subdomainNo)
            elif isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedPut):
                ov1SP = self.overlaps[subdomainNo]
                requestsOneSidedPut.append(subdomainNo)
            elif isinstance(self.overlaps[subdomainNo], algebraicOverlapOneSidedPutLockAll):
                ov1SPLA = self.overlaps[subdomainNo]
                requestsOneSidedPutLockAll.append(subdomainNo)
            else:
                ov = self.overlaps[subdomainNo]
                requestsReceive.append(ov.receiveComplex(return_vec, vecNo=vecNo))

        if ((len(requestsOneSidedGet) > 0 or
             len(requestsOneSidedPut) > 0 or
             len(requestsOneSidedPutLockAll) > 0) and not asynchronous):
            setBarrier = True
        else:
            setBarrier = False

        while len(requestsReceive) > 0:
            status = MPI.Status()
            done = MPI.Request.Waitany(requestsReceive, status)
            assert status.error == 0
            requestsReceive.pop(done)
            subdomainNo = status.source
            ov = self.overlaps[subdomainNo]
            ov.accumulateProcessComplex(return_vec, vecNo=vecNo)
        for subdomainNo in requestsOneSidedGet:
            ov1S = self.overlaps[subdomainNo]
            ov1S.receiveComplex(return_vec, vecNo=vecNo)
            ov1S.accumulateProcessComplex(return_vec, vecNo=vecNo)
        for subdomainNo in requestsOneSidedPut:
            ov1SP = self.overlaps[subdomainNo]
            ov1SP.receiveComplex(return_vec, vecNo=vecNo)
            ov1SP.accumulateProcessComplex(return_vec, vecNo=vecNo)
        for subdomainNo in requestsOneSidedPutLockAll:
            ov1SPLA = self.overlaps[subdomainNo]
            ov1SPLA.receiveComplex(return_vec, vecNo=vecNo, flushOp=flushOp)
            ov1SPLA.accumulateProcessComplex(return_vec, vecNo=vecNo)
        if setBarrier:
            self.comm.Barrier()

    def countDoFs(self):
        lv = set()
        for subdomainNo in self.overlaps:
            if subdomainNo < self.comm.rank:
                lv |= set(list(self.overlaps[subdomainNo].shared_dofs))
        return self.comm.allreduce(self.num_subdomain_dofs-len(lv))

    def getGlobalIndices(self):
        cdef:
            INDEX_t rank = self.comm.rank
            REAL_t[::1] v = rank*np.ones((self.num_subdomain_dofs),
                                         dtype=REAL)
            INDEX_t i, k, m

        self.unique(v)
        k = 0
        for i in range(self.num_subdomain_dofs):
            if v[i] == rank:
                k += 1
        m = self.comm.scan(k, MPI.SUM)
        m = m-k
        for i in range(self.num_subdomain_dofs):
            if v[i] == rank:
                v[i] = m
                m += 1
        self.unique(v)
        return np.asarray(v, dtype=INDEX)

    def __repr__(self):
        s = ''
        for subdomainNo in self.overlaps:
            s += self.overlaps[subdomainNo].__repr__() + '\n'
        return s

    def HDF5write(self, node):
        compression = 'gzip'
        node.attrs['numSubdomains'] = self.numSubdomains
        node.attrs['num_subdomain_dofs'] = self.num_subdomain_dofs
        node.attrs['max_cross'] = self._max_cross
        for subdomainNo in self.overlaps:
            grp = node.create_group(str(subdomainNo))
            self.overlaps[subdomainNo].HDF5write(grp)
        node.create_dataset('Dval', data=self.Dval,
                            compression=compression)
        node.create_dataset('Didx', data=self.Didx,
                            compression=compression)
        if hasattr(self, 'DidxNonOverlapping'):
            node.create_dataset('DvalNonOverlapping',
                                data=self.DvalNonOverlapping,
                                compression=compression)
            node.create_dataset('DidxNonOverlapping',
                                data=self.DidxNonOverlapping,
                                compression=compression)

    @staticmethod
    def HDF5read(node, comm):
        overlaps = algebraicOverlapManager(node.attrs['numSubdomains'],
                                           node.attrs['num_subdomain_dofs'],
                                           comm)
        overlaps._max_cross = node.attrs['max_cross']
        for grp in node:
            if grp == 'Didx':
                overlaps.Didx = np.array(node['Didx'], dtype=INDEX)
            elif grp == 'Dval':
                overlaps.Dval = np.array(node['Dval'], dtype=REAL)
            elif grp == 'DidxNonOverlapping':
                overlaps.DidxNonOverlapping = np.array(node['DidxNonOverlapping'], dtype=INDEX)
            elif grp == 'DvalNonOverlapping':
                overlaps.DvalNonOverlapping = np.array(node['DvalNonOverlapping'], dtype=REAL)
            else:
                overlaps.overlaps[int(grp)] = algebraicOverlap.HDF5read(node[grp],
                                                                        comm)
        return overlaps

    def check(self, mesh=None, DoFMap dm=None, interfaces=None, label="Algebraic overlap"):
        if mesh is not None:
            dof2Cell = uninitialized((dm.num_dofs, 2), dtype=INDEX)
            dof2Cells = [set() for dof in range(dm.num_dofs)]
            for cellNo in range(mesh.num_cells):
                for dofNo in range(dm.dofs_per_element):
                    dof = dm.cell2dof(cellNo, dofNo)
                    if dof >= 0:
                        dof2Cell[dof, 0] = cellNo
                        dof2Cell[dof, 1] = dofNo
                        dof2Cells[dof].add((cellNo, dofNo))

        requests = []
        requests2 = []
        success = True
        myDofNodes = {}
        for i in self.overlaps:
            requests.append(self.comm.isend(self.overlaps[i].num_shared_dofs, dest=i, tag=5))

            if mesh is not None:
                dofNodes = np.zeros((self.overlaps[i].num_shared_dofs, mesh.dim), dtype=REAL)
                nodalCoords = uninitialized((dm.dofs_per_element, mesh.dim), dtype=REAL)
                k = 0
                for dof in self.overlaps[i].shared_dofs:
                    cellNo, dofNo = dof2Cell[dof, 0], dof2Cell[dof, 1]
                    dm.getNodalCoordinates(mesh.vertices_as_array[mesh.cells_as_array[cellNo, :], :],
                                           nodalCoords)
                    dofNodes[k, :] = nodalCoords[dofNo, :]
                    k += 1
                myDofNodes[i] = dofNodes
                requests2.append(self.comm.Isend(myDofNodes[i], dest=i, tag=6))

        for i in self.overlaps:
            numDoFsOther = self.comm.recv(source=i, tag=5)
            if mesh is not None:
                otherDofNodes = uninitialized((numDoFsOther, mesh.dim), dtype=REAL)
                self.comm.Recv(otherDofNodes, source=i, tag=6)
            if numDoFsOther != self.overlaps[i].num_shared_dofs:
                print('{}: Subdomains {} and {} shared different number of DoFs: {} vs {}.'.format(label, self.comm.rank, i,
                                                                                                   self.overlaps[i].num_shared_dofs, numDoFsOther))
                success = False
            elif mesh is not None:
                diff = norm(myDofNodes[i]-otherDofNodes, axis=1)
                if diff.max() > 1e-9:
                    diffCount = (diff > 1e-9).sum()
                    s = '{}: Subdomains {} and {} shared {} different DoFs\n'.format(label, self.comm.rank, i, diffCount)
                    k = 0
                    for dof in self.overlaps[i].shared_dofs:
                        if diff[k] > 1e-9:
                            cellNo, dofNo = dof2Cell[dof, 0], dof2Cell[dof, 1]
                            s += 'cellNo {} dofNo {}: {} != {}\n'.format(cellNo, dofNo,
                                                                         myDofNodes[i][k, :],
                                                                         otherDofNodes[k, :])
                            if interfaces is not None:
                                interface = interfaces.interfaces[i]
                                for cellNo, dofNo in dof2Cells[dof]:
                                    for j in range(interface.num_vertices):
                                        if interface.vertices[j, 0] == cellNo:
                                            s += 'vertex {}\n'.format(interface.vertices[j, 1])
                                    for j in range(interface.num_edges):
                                        if interface.edges[j, 0] == cellNo:
                                            s += 'edge {} {}\n'.format(interface.edges[j, 1], interface.edges[j, 2])
                                    for j in range(interface.num_faces):
                                        if interface.faces[j, 0] == cellNo:
                                            s += 'face {} {}\n'.format(interface.faces[j, 1], interface.faces[j, 2])
                                    for j in range(interface.num_cells):
                                        if interface.cells[j] == cellNo:
                                            s += 'cell\n'

                        k += 1
                    print(s)
                    success = False

        MPI.Request.Waitall(requests)
        MPI.Request.Waitall(requests2)
        assert success
        self.comm.Barrier()
        if self.comm.rank == 0:
            print('{} check successful.'.format(label))
        stdout.flush()
        self.comm.Barrier()

    def getAccumulateOperator(self):
        a = lambda x, y: self.accumulate(x, y)
        acc = LinearOperator_wrapper(self.num_subdomain_dofs,
                                     self.num_subdomain_dofs,
                                     a)
        return acc

    def getDistributeOperator(self, BOOL_t nonOverlapping=False):
        d = lambda x, y: self.distribute(x, y, nonOverlapping=nonOverlapping)
        dist = LinearOperator_wrapper(self.num_subdomain_dofs,
                                      self.num_subdomain_dofs,
                                      d)
        return dist

    def getDistributeAsDiagonalOperator(self, BOOL_t nonOverlapping=False):
        d = np.ones((self.num_subdomain_dofs), dtype=REAL)
        self.distribute(d, vec2=None, nonOverlapping=nonOverlapping)
        return diagonalOperator(d)

    def flushMemory(self, INDEX_t vecNo=0, REAL_t value=0.):
        for i in self.overlaps:
            self.overlaps[i].flushMemory(vecNo=vecNo, value=value)

    def findMinPerOverlap(self, REAL_t[::1] indicator):
        # Takes local vector.
        # Returns the elementwise minimum in each overlap, i.e. a dict of vectors.
        cdef:
            algebraicOverlap ov
            INDEX_t j, subdomainNo, dof
            dict localIndicators
            REAL_t[::1] myLocalIndicator, otherLocalIndicator
        # FIXME: We don't really want to accumulate into a temporary vector here.
        indicatorTmp = uninitialized((indicator.shape[0]), dtype=REAL)
        self.accumulate(indicator, indicatorTmp, asynchronous=False)
        del indicatorTmp
        localIndicators = {}
        for subdomainNo in self.overlaps:
            ov = self.overlaps[subdomainNo]
            myLocalIndicator = uninitialized((ov.num_shared_dofs), dtype=REAL)
            for j in range(ov.num_shared_dofs):
                dof = ov.shared_dofs[j]
                myLocalIndicator[j] = indicator[dof]
            otherLocalIndicator = uninitialized((ov.num_shared_dofs), dtype=REAL)
            ov.setOverlapLocal(otherLocalIndicator)
            localIndicators[subdomainNo] = uninitialized((ov.num_shared_dofs), dtype=REAL)
            np.minimum(myLocalIndicator,
                       otherLocalIndicator,
                       out=localIndicators[subdomainNo])
        return localIndicators

    def reduce(self, REAL_t v, BOOL_t asynchronous=False):
        cdef:
            REAL_t v2_mem[1]
            REAL_t[::1] v2 = v2_mem
        assert not asynchronous
        v2[0] = v
        self.comm.Allreduce(MPI.IN_PLACE, v2)
        return v2[0]

    def cleanup(self):
        MPI.Request.Waitall(self.requestsSend)
        # make sure we don't delete any MPI windows that still hold data
        self.comm.Barrier()


cdef class multilevelAlgebraicOverlapManager:
    # Tracks the algebraic overlap with the DoFMaps of all other
    # subdomains on all levels.

    def __init__(self, comm, BOOL_t setupAsynchronousReduce=False):
        self.levels = []
        self.comm = comm
        self.useLockAll = False
        self.canUseAsynchronousReduce = setupAsynchronousReduce
        if setupAsynchronousReduce:
            numSubComms = 2
            maxCommSize = 300
            numSubComms = max(numSubComms, (comm.size-1)//maxCommSize+1)
            commSplits = np.around(np.linspace(1, comm.size, numSubComms+1)).astype(INDEX)
            comm.Barrier()
            # split comm into subcomms, rank 0 is in the intersection
            if self.comm.rank == 0:
                subcomms = []
                self.ReduceWindows = []
                self.ReduceMems = []
                for splitNo in range(numSubComms):
                    subcomms.append(comm.Split(0))
                for subcomm in subcomms:
                    self.ReduceWindows.append(MPI.Win.Allocate(MPI.REAL.size*subcomm.size, comm=subcomm))
                    self.ReduceMems.append(np.ones((subcomm.size), dtype=REAL))
            else:
                for splitNo in range(numSubComms):
                    if commSplits[splitNo] <= comm.rank  and comm.rank < commSplits[splitNo+1]:
                        color = 0
                    else:
                        color = MPI.UNDEFINED
                    subcommTemp = comm.Split(color)
                    if subcommTemp != MPI.COMM_NULL:
                        subcomm = subcommTemp
                self.rank = subcomm.rank
                self.ReduceWindow = MPI.Win.Allocate(MPI.REAL.size, comm=subcomm)
        self.useAsynchronousComm = False

    def LockAll(self):
        cdef:
            MPI.Win ReduceWindow
            INDEX_t i
        if self.useLockAll:
            if self.comm.rank == 0:
                for i in range(len(self.ReduceWindows)):
                    ReduceWindow = self.ReduceWindows[i]
                    ReduceWindow.Lock_all(MPI.MODE_NOCHECK)
            else:
                self.ReduceWindow.Lock_all(MPI.MODE_NOCHECK)

    def setComplex(self):
        for level in range(len(self.levels)):
            self.levels[level].setComplex()

    def getLevel(self, INDEX_t n):
        cdef:
            INDEX_t level
        for level in range(len(self.levels)-1, -1, -1):
            if self.levels[level].num_subdomain_dofs == n:
                return level
        else:
            raise NotImplementedError("Cannot find a level of size {}.\nLevel sizes: {}".format(n, [self.levels[level].num_subdomain_dofs
                                                                                                    for level in range(len(self.levels))]))

    def accumulate(self,
                   REAL_t[::1] vec,
                   REAL_t[::1] return_vec=None,
                   INDEX_t level=-1,
                   BOOL_t asynchronous=False,
                   INDEX_t vecNo=0):
        cdef:
            INDEX_t n
            algebraicOverlapManager ovM
        if level == -1:
            n = vec.shape[0]
            for level in range(len(self.levels)-1, -1, -1):
                if self.levels[level].num_subdomain_dofs == n:
                    break
            else:
                raise NotImplementedError("Cannot find a level of size {}.\nLevel sizes: {}".format(n, [self.levels[level].num_subdomain_dofs
                                                                                                        for level in range(len(self.levels))]))
        ovM = self.levels[level]
        ovM.accumulate(vec, return_vec, asynchronous=asynchronous, vecNo=vecNo)

    def accumulateComplex(self,
                          COMPLEX_t[::1] vec,
                          COMPLEX_t[::1] return_vec=None,
                          INDEX_t level=-1,
                          BOOL_t asynchronous=False,
                          INDEX_t vecNo=0):
        cdef:
            INDEX_t n
            algebraicOverlapManager ovM
        if level == -1:
            n = vec.shape[0]
            for level in range(len(self.levels)-1, -1, -1):
                if self.levels[level].num_subdomain_dofs == n:
                    break
            else:
                raise NotImplementedError("Cannot find a level of size {}.\nLevel sizes: {}".format(n, [self.levels[level].num_subdomain_dofs
                                                                                                        for level in range(len(self.levels))]))
        ovM = self.levels[level]
        ovM.accumulateComplex(vec, return_vec, asynchronous=asynchronous, vecNo=vecNo)

    def unique(self,
               REAL_t[::1] vec,
               INDEX_t vecNo=0):
        cdef INDEX_t level, n = vec.shape[0]
        for level in range(len(self.levels)):
            if self.levels[level].num_subdomain_dofs == n:
                break
        else:
            raise NotImplementedError()
        self.levels[level].unique(vec, vecNo=vecNo)

    def prepareDistribute(self):
        for i in range(len(self.levels)):
            self.levels[i].prepareDistribute()

    def prepareDistributeMeshOverlap(self, mesh, nc, DoFMap dm, depth, meshOverlaps):
        for i in range(len(self.levels)-1):
            self.levels[i].prepareDistribute()
        self.levels[len(self.levels)-1].prepareDistributeMeshOverlap(mesh, nc, dm, depth, meshOverlaps)

    def distribute(self,
                   REAL_t[::1] vec,
                   REAL_t[::1] vec2=None,
                   INDEX_t level=-1,
                   BOOL_t nonOverlapping=False):
        cdef:
            INDEX_t n
            algebraicOverlapManager ovM
        if level == -1:
            n = vec.shape[0]
            for level in range(len(self.levels)-1, -1, -1):
                if self.levels[level].num_subdomain_dofs == n:
                    break
            else:
                raise NotImplementedError()
        ovM = self.levels[level]
        ovM.distribute(vec, vec2, nonOverlapping=nonOverlapping)

    def distributeComplex(self,
                          COMPLEX_t[::1] vec,
                          COMPLEX_t[::1] vec2=None,
                          INDEX_t level=-1,
                          BOOL_t nonOverlapping=False):
        cdef:
            INDEX_t n
            algebraicOverlapManager ovM
        if level == -1:
            n = vec.shape[0]
            for level in range(len(self.levels)-1, -1, -1):
                if self.levels[level].num_subdomain_dofs == n:
                    break
            else:
                raise NotImplementedError()
        ovM = self.levels[level]
        ovM.distributeComplex(vec, vec2, nonOverlapping=nonOverlapping)

    def redistribute(self,
                     REAL_t[::1] vec,
                     REAL_t[::1] vec2=None,
                     level=None,
                     BOOL_t nonOverlapping=False,
                     BOOL_t asynchronous=False,
                     INDEX_t vecNo=0):
        cdef:
            algebraicOverlapManager ovM
        if level is None:
            level = len(self.levels)-1
        ovM = self.levels[level]
        ovM.redistribute(vec, vec2,
                         nonOverlapping=nonOverlapping,
                         asynchronous=asynchronous,
                         vecNo=vecNo)

    def countDoFs(self, localsize=None, level=None):
        # FIX: remove the localsize argument
        if level is None:
            level = len(self.levels)-1
        return self.levels[level].countDoFs()

    def get_num_shared_dofs(self, unique=False):
        return self.levels[len(self.levels)-1].get_num_shared_dofs(unique)

    num_shared_dofs = property(fget=get_num_shared_dofs)

    def __repr__(self):
        s = ''
        for i in range(len(self.levels)):
            s += 'Level {}\n'.format(i)+self.levels[i].__repr__()
        return s

    def HDF5write(self, node):
        for i, lvl in enumerate(self.levels):
            grp = node.create_group(str(i))
            self.levels[i].HDF5write(grp)

    @staticmethod
    def HDF5read(node, comm):
        overlaps = multilevelAlgebraicOverlapManager(comm)
        levels = node.keys()
        levels = sorted([int(lvl) for lvl in levels])
        for grp in levels:
            overlaps.levels.append(algebraicOverlapManager.HDF5read(node[str(grp)],
                                                                    comm))
        return overlaps

    def check(self, meshes, DoFMaps, label='Algebraic overlap'):
        for i in range(len(self.levels)):
            self.levels[i].check(meshes[i], DoFMaps[i], '{} Level {}'.format(label, i))
        print('Validation successful.')

    def reduce(self, REAL_t v, BOOL_t asynchronous=False):
        cdef:
            INDEX_t i
            REAL_t v2_mem[1]
            REAL_t[::1] v2 = v2_mem
            MPI.Win ReduceWindow
            REAL_t[::1] reduceMem
        if asynchronous or self.useAsynchronousComm:
            assert self.canUseAsynchronousReduce
            if self.comm.rank == 0:
                if not asynchronous:
                    # let all ranks write local contributions
                    self.comm.Barrier()
                # get all local residuals
                for j in range(len(self.ReduceWindows)):
                    ReduceWindow = self.ReduceWindows[j]
                    reduceMem = self.ReduceMems[j]
                    if not self.useLockAll:
                        ReduceWindow.Lock(0, MPI.LOCK_EXCLUSIVE)
                    else:
                        ReduceWindow.Flush_all()
                    ReduceWindow.Get(reduceMem, 0)
                    if not self.useLockAll:
                        ReduceWindow.Unlock(0)
                    else:
                        ReduceWindow.Flush_local(0)
                    # sum up
                    for i in range(1, reduceMem.shape[0]):
                        v += reduceMem[i]
                # put global residual in window
                v2[0] = v
                for j in range(len(self.ReduceWindows)):
                    ReduceWindow = self.ReduceWindows[j]
                    reduceMem = self.ReduceMems[j]
                    if not self.useLockAll:
                        ReduceWindow.Lock(0, MPI.LOCK_EXCLUSIVE)
                    # self.ReduceWindow.Put(v2, 0, target=(0, 1, MPI.REAL))
                    for rank in range(1, reduceMem.shape[0]):
                        ReduceWindow.Put(v2, rank, target=(0, 1, MPI.REAL))
                    if not self.useLockAll:
                        ReduceWindow.Unlock(0)
                    else:
                        ReduceWindow.Flush(0)
                if not asynchronous:
                    # let all ranks acces result
                    self.comm.Barrier()
                return v
            else:
                v2[0] = v
                # put local residual into window on master
                if not self.useLockAll:
                    self.ReduceWindow.Lock(0, MPI.LOCK_SHARED)
                self.ReduceWindow.Put(v2, 0, target=(self.rank*MPI.REAL.size, 1, MPI.REAL))
                if not self.useLockAll:
                    self.ReduceWindow.Unlock(0)
                # else:
                #     self.ReduceWindow.Flush_all()
                if not asynchronous:
                    # before rank 0 accesses local contributions
                    self.comm.Barrier()

                if not asynchronous:
                    # wait until rank 0 has published result
                    self.comm.Barrier()
                # get global residual from window on master
                if not self.useLockAll:
                    self.ReduceWindow.Lock(self.rank, MPI.LOCK_SHARED)
                # self.ReduceWindow.Get(v2, 0, target=(0, 1, MPI.REAL))
                self.ReduceWindow.Get(v2, self.rank, target=(0, 1, MPI.REAL))
                if not self.useLockAll:
                    self.ReduceWindow.Unlock(self.rank)
                else:
                    self.ReduceWindow.Flush_local(self.rank)
                return v2[0]
        else:
            v2[0] = v
            self.comm.Allreduce(MPI.IN_PLACE, v2)
            return v2[0]

    def getAccumulateOperator(self, level=None):
        if level is None:
            level = len(self.levels)-1
        return self.levels[level].getAccumulateOperator()

    def getDistributeOperator(self, level=None, BOOL_t nonOverlapping=False):
        if level is None:
            level = len(self.levels)-1
        return self.levels[level].getDistributeOperator(nonOverlapping)

    def getDistributeAsDiagonalOperator(self, level=None, BOOL_t nonOverlapping=False):
        if level is None:
            level = len(self.levels)-1
        return self.levels[level].getDistributeAsDiagonalOperator()

    def getGlobalIndices(self, level=None):
        if level is None:
            level = len(self.levels)-1
        return self.levels[level].getGlobalIndices()

    def flushMemory(self, level=None, INDEX_t vecNo=0):
        cdef:
            REAL_t v2_mem[1]
            REAL_t[::1] v2 = v2_mem
            MPI.Win ReduceWindow
            REAL_t[::1] reduceMem
            INDEX_t i, j
        if not self.canUseAsynchronousReduce:
            return
        if level is None:
            level = len(self.levels)-1
        self.levels[level].flushMemory(vecNo=vecNo)
        if self.comm.rank == 0:
            for i in range(len(self.ReduceWindows)):
                reduceMem = self.ReduceMems[i]
                ReduceWindow = self.ReduceWindows[i]
                for j in range(reduceMem.shape[0]):
                    reduceMem[j] = 1.
                if not self.useLockAll:
                    ReduceWindow.Lock(0, MPI.LOCK_EXCLUSIVE)
                ReduceWindow.Put(reduceMem, 0)
                if not self.useLockAll:
                    ReduceWindow.Unlock(0)
        else:
            v2[0] = 1.
            if not self.useLockAll:
                self.ReduceWindow.Lock(self.rank, MPI.LOCK_EXCLUSIVE)
            self.ReduceWindow.Put(v2, self.rank)
            if not self.useLockAll:
                self.ReduceWindow.Unlock(self.rank)
        self.comm.Barrier()

    def getOverlapLevel(self, num_subdomain_dofs):
        for lvl in range(len(self.levels)):
            if self.levels[lvl].num_subdomain_dofs == num_subdomain_dofs:
                return lvl
        raise NotImplementedError()
