###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py cimport MPI
from . myTypes cimport INDEX_t, REAL_t, BOOL_t
cimport numpy as np
from . ip_norm cimport normBase, norm_distributed

######################################################################
# convergence criteria

cdef class convergenceCriterion:
    cdef:
        REAL_t tol, localResidual, globalResidual
        INDEX_t maxiter, iter
        BOOL_t accumulated
        normBase norm
        MPI.Comm comm, clientComm
        INDEX_t clientRank
        BOOL_t hasClient
        INDEX_t ov_level

    cdef BOOL_t eval(self, REAL_t[::1] localResidual, BOOL_t asynchronous=*)
    cdef REAL_t getGlobalResidual(self)
    cdef void registerClient(self, MPI.Comm comm, INDEX_t rank)
    cdef void updateClients(self, BOOL_t converged)
    cdef void cleanup(self)


cdef class noOpConvergenceCriterion(convergenceCriterion):
    cdef BOOL_t eval(self, REAL_t[::1] localResidual, BOOL_t asynchronous=*)


cdef class synchronousConvergenceCriterion(convergenceCriterion):
    cdef BOOL_t eval(self, REAL_t[::1] residualVec, BOOL_t asynchronous=*)
    cdef REAL_t getGlobalResidual(self)
    cdef void updateClients(self, BOOL_t converged)



######################################################################
# convergence masters (needed if coarse grid is on separate communicator)

cdef class convergenceMaster:
    cdef:
        MPI.Comm masterComm
        INDEX_t masterRank, clientRank
    cdef void setStatus(self, BOOL_t converged)


cdef class noOpConvergenceMaster(convergenceMaster):
    pass


cdef class synchronousConvergenceMaster(convergenceMaster):
    cdef void setStatus(self, BOOL_t converged)

######################################################################
# convergence clients (needed if coarse grid is on separate communicator)

cdef class convergenceClient:
    cdef:
        MPI.Comm masterComm
        INDEX_t masterRank
    cdef BOOL_t getStatus(self)
    cdef void cleanup(self)


cdef class noOpConvergenceClient(convergenceClient):
    cdef BOOL_t getStatus(self)


cdef class synchronousConvergenceClient(convergenceClient):
    cdef:
        INDEX_t tag
    cdef BOOL_t getStatus(self)


cdef class synchronousConvergenceClientSubcomm(convergenceClient):
    cdef:
        INDEX_t tag
        MPI.Comm comm
    cdef BOOL_t getStatus(self)


