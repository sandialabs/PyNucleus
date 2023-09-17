###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

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
