###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from . myTypes import INDEX, REAL
from . ip_norm cimport ip_serial
from libc.math cimport sqrt
import numpy as np
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from . blas import uninitialized

cdef:
    INDEX_t TAG_CONVERGED = 32012


######################################################################
# convergence criteria

cdef class convergenceCriterion:
    def __init__(self, REAL_t tol, INDEX_t maxiter, overlaps, BOOL_t accumulated):
        self.tol = tol
        self.maxiter = maxiter
        self.iter = 0
        self.ov_level = len(overlaps.levels)-1
        self.norm = norm_distributed(overlaps, level=self.ov_level)
        self.comm = overlaps.comm
        self.accumulated = accumulated
        self.globalResidual = 1.
        self.hasClient = False

    cdef BOOL_t eval(self, REAL_t[::1] localResidual, BOOL_t asynchronous=False):
        pass

    cdef REAL_t getGlobalResidual(self):
        pass

    cdef void registerClient(self, MPI.Comm comm, INDEX_t rank):
        self.clientComm = comm
        self.clientRank = rank
        self.hasClient = True

    cdef void updateClients(self, BOOL_t converged):
        pass

    cdef void cleanup(self):
        pass


cdef class noOpConvergenceCriterion(convergenceCriterion):
    def __init__(self, REAL_t tol, INDEX_t maxiter, overlaps, BOOL_t accumulated):
        convergenceCriterion.__init__(self, tol, maxiter, overlaps, accumulated)

    cdef BOOL_t eval(self, REAL_t[::1] localResidual, BOOL_t asynchronous=False):
        return False


cdef class synchronousConvergenceCriterion(convergenceCriterion):
    def __init__(self, REAL_t tol, INDEX_t maxiter, overlaps, BOOL_t accumulated):
        convergenceCriterion.__init__(self, tol, maxiter, overlaps, accumulated)

    cdef BOOL_t eval(self, REAL_t[::1] residualVec, BOOL_t asynchronous=False):
        cdef:
            BOOL_t converged
        self.globalResidual = self.norm.eval(residualVec, self.accumulated, asynchronous=False)
        converged = (self.globalResidual < self.tol) or self.iter >= self.maxiter
        self.iter += 1
        return converged

    cdef REAL_t getGlobalResidual(self):
        return self.globalResidual

    cdef void updateClients(self, BOOL_t converged):
        if self.hasClient and self.comm.rank == 0:
            self.clientComm.isend(converged, dest=self.clientRank, tag=200)


######################################################################
# convergence master (needed if coarse grid is on separate communicator)

cdef class convergenceMaster:
    def __init__(self, MPI.Comm masterComm, INDEX_t masterRank, INDEX_t clientRank=0):
        self.masterComm = masterComm
        self.masterRank = masterRank
        self.clientRank = clientRank

    cdef void setStatus(self, BOOL_t converged):
        pass


cdef class noOpConvergenceMaster(convergenceMaster):
    def __init__(self, MPI.Comm masterComm, INDEX_t masterRank, INDEX_t clientRank=0):
        super(noOpConvergenceMaster, self).__init__(masterComm, masterRank, clientRank)


cdef class synchronousConvergenceMaster(convergenceMaster):
    def __init__(self, MPI.Comm masterComm, INDEX_t masterRank, INDEX_t clientRank=0):
        super(synchronousConvergenceMaster, self).__init__(masterComm, masterRank, clientRank)

    cdef void setStatus(self, BOOL_t converged):
        if self.masterComm.rank == self.masterRank:
            self.masterComm.send(converged, dest=self.clientRank, tag=201)


######################################################################
# convergence clients (needed if coarse grid is on separate communicator)

cdef class convergenceClient:
    def __init__(self, MPI.Comm masterComm, INDEX_t masterRank):
        self.masterComm = masterComm
        self.masterRank = masterRank

    cdef BOOL_t getStatus(self):
        pass

    cdef void cleanup(self):
        pass


cdef class noOpConvergenceClient(convergenceClient):
    cdef BOOL_t getStatus(self):
        return False


cdef class synchronousConvergenceClient(convergenceClient):
    def __init__(self, MPI.Comm masterComm, INDEX_t masterRank, tag=200):
        super(synchronousConvergenceClient, self).__init__(masterComm, masterRank)
        self.tag = tag

    cdef BOOL_t getStatus(self):
        cdef:
            BOOL_t converged
        converged = self.masterComm.recv(source=self.masterRank, tag=self.tag)
        return converged


cdef class synchronousConvergenceClientSubcomm(convergenceClient):
    def __init__(self, MPI.Comm masterComm, INDEX_t masterRank, MPI.Comm comm, tag=200):
        super(synchronousConvergenceClientSubcomm, self).__init__(masterComm, masterRank)
        self.comm = comm
        self.tag = tag

    cdef BOOL_t getStatus(self):
        cdef:
            BOOL_t converged = False
        if self.comm.rank == 0:
            converged = self.masterComm.recv(source=self.masterRank, tag=self.tag)
        converged = self.comm.bcast(converged, root=0)
        return converged


