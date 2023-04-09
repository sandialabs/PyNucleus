###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from __future__ import division
import numpy as np
cimport numpy as np
from numpy.linalg import norm as normSeq
import logging
cimport cython
from libc.math cimport log

from . myTypes import INDEX, REAL
from . myTypes cimport INDEX_t, REAL_t
from . blas import uninitialized_like


LOGGER = logging.getLogger(__name__)


def UniformOnUnitSphere(dim, samples=1, norm=normSeq):
    "Uniform distribution on the unit sphere."
    if samples > 1:
        shape = (dim, samples)
        vec = np.random.normal(size=shape)
        for i in range(samples):
            vec[:, i] = vec[:, i]/norm(vec[:, i])
    else:
        shape = (dim)
        vec = np.random.normal(size=shape)
        vec = vec/norm(vec)
    return vec


import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py cimport MPI


cdef class FakeComm_class(MPI.Comm):
    cdef:
        int rank_, size_
        public dict values

    def setRankSize(self, rank, size):
        self.rank_ = rank
        self.size_ = size

    property size:
        """number of processes in communicator"""
        def __get__(self):
            return self.size_

    property rank:
        """rank of this process in communicator"""
        def __get__(self):
            return self.rank_

    def Barrier(self):
        pass

    def allreduce(self, v, *args, **kwargs):
        return v


def FakeComm(rank, size):
    c = FakeComm_class()
    c.setRankSize(rank, size)
    c.values = {}
    return c
