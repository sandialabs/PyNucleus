###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
import numpy as np
cimport numpy as np
cimport cython
import logging
from PyNucleus_base.myTypes import INDEX, {SCALAR}, BOOL
from PyNucleus_base.myTypes cimport INDEX_t, {SCALAR}_t, BOOL_t
from PyNucleus_base import uninitialized
from PyNucleus_base.performanceLogger cimport PLogger, FakePLogger
from PyNucleus_base.linear_operators cimport {SCALAR_label}LinearOperator
from PyNucleus_base import solverFactory
from PyNucleus_fem.meshOverlaps import overlapManager

from time import sleep
from sys import stdout
include "config.pxi"
LOGGER = logging.getLogger(__name__)
MPI_BOOL = MPI.BOOL

from PyNucleus_fem.algebraicOverlaps cimport flush_type, no_flush, flush_local, flush_local_all, flush, flush_all


######################################################################

cdef class {SCALAR_label}coarseSolver({SCALAR_label_lc_}iterative_solver):
    """
    This coarse solver gathers the ride-hand side from all nodes, then
    solves the problem on a subcommunicator using a distributed solver
    and scatters the solution.
    """

    def __init__(self,
                 hierarchyManager,
                 FakePLogger PLogger,
                 solverName,
                 **kwargs):
        self.solverName = solverName
        self.kwargs = kwargs
        self.asynchronous = False

        self.hierarchyManager = hierarchyManager
        hierarchy = hierarchyManager.builtHierarchies[-1]
        self.hierarchy = hierarchy
        self.comm = hierarchy.connectorEnd.global_comm
        if not hierarchy.connectorEnd.is_overlapping:
            self.inter_comm = hierarchy.connectorEnd.interComm
            self.myLeaderRank = hierarchy.connectorEnd.myLeaderRank
            self.otherLeaderRank = hierarchy.connectorEnd.otherLeaderRank

        if hasattr(hierarchy, 'connectorEnd') and hasattr(hierarchy.connectorEnd, 'depth'):
            self.depth = hierarchy.connectorEnd.depth
        else:
            self.depth = 0

        if hasattr(hierarchy.connectorEnd, 'algOM'):
            # rank is part of coarse grid
            self.overlapsCoarse = hierarchy.connectorEnd.algOM
            self.levels = hierarchy.getLevelList(recurse=False)
            self.subset_comm = hierarchy.comm
            localSize = self.levels[len(self.levels)-1]['A'].shape[0]
            self.inCG = True
            self.x = uninitialized((localSize), dtype={SCALAR})
            self.rhs = uninitialized((localSize), dtype={SCALAR})
            self.intraLevelCoarse = hierarchy.algebraicLevels[-1].algebraicOverlaps
        
        else:
            self.inCG = False

        if hasattr(hierarchy.connectorEnd, 'algOMnew'):
            # rank is part of fine grid
            self.overlapsFine = hierarchy.connectorEnd.algOMnew
            self.inSubdomain = True
            self.intraLevelFine = hierarchy.connectorEnd.hierarchy2.algebraicLevels[0].algebraicOverlaps
            self.subset_commFine = hierarchy.connectorEnd.comm2
            {SCALAR_label_lc_}iterative_solver.__init__(self, num_rows=hierarchy.connectorEnd.hierarchy2.algebraicLevels[0].DoFMap.num_dofs)
        
        else:
            self.inSubdomain = False
            {SCALAR_label_lc_}iterative_solver.__init__(self, num_rows=0)
        self.PLogger = PLogger
        self.setAinv()
        if self.inCG:
            self.Ainv.PLogger = self.PLogger

    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef BOOL_t canWriteRHS(self):
        return True

    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void sendRHS(self, {SCALAR}_t[::1] b):
        self.overlapsFine.send{SCALAR_label}(b)

    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cdef BOOL_t solve_cg(self):
        self.rhs[:] = 0.
        self.overlapsCoarse.receive{SCALAR_label}(self.rhs)
        if self.intraLevelCoarse is not None:
            self.intraLevelCoarse.distribute{SCALAR_label}(self.rhs)
        with self.PLogger.Timer('solveTimeLocal'):
            if isinstance(self.Ainv, {SCALAR_label_lc_}iterative_solver):
                self.Ainv.tolerance = self.tolerance
                self.Ainv.maxIter = self.maxIter
            numIter = self.Ainv.solve(self.rhs, self.x)
            self.PLogger.addValue('coarsegrid.Iterations', numIter)
        self.overlapsCoarse.send{SCALAR_label}(self.x)
        return True

    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef BOOL_t getSolution(self, {SCALAR}_t[::1] x):
        x[:] = 0.
        self.overlapsFine.receive{SCALAR_label}(x)
        self.overlapsFine.distribute{SCALAR_label}(x)
        return True

    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cdef int solve(self,
                   {SCALAR}_t[::1] b,
                   {SCALAR}_t[::1] x) except -1:
        cdef:
            BOOL_t ret = True
        {SCALAR_label_lc_}iterative_solver.solve(self, b, x)
        if (self.overlapsFine is not None) or (b.shape[0] != self.Ainv.num_rows):
            with self.PLogger.Timer('solveTime'):
                if self.inSubdomain and self.canWriteRHS():
                    self.sendRHS(b)
                if self.inCG:
                    ret = self.solve_cg()
                if self.inSubdomain:
                    self.getSolution(x)
        else:
            with self.PLogger.Timer('solveTime'):
                if isinstance(self.Ainv, {SCALAR_label_lc_}iterative_solver):
                    self.Ainv.tolerance = self.tolerance
                    self.Ainv.maxIter = self.maxIter
                numIter = self.Ainv.solve(b, x)
                self.PLogger.addValue('coarsegrid.Iterations', numIter)
        return ret

    def __str__(self):
        return self.solver_description

    def setAinv(self):
        if self.inCG:
            if self.solverName in ('LU', 'Chol', 'IChol', 'ILU'):
                assert self.subset_comm.size == 1, 'Cannot run {} in distributed mode'.format(self.solverName)
            self.Ainv = solverFactory.build(self.solverName, A=self.levels[-1]['A'], hierarchy=self.hierarchyManager, **self.kwargs)

    cpdef void setup(self, {SCALAR_label}LinearOperator A=None):
        if self.Ainv is not None:
            self.Ainv.setup()
            self.solver_description = str(self.Ainv)
        root = 0
        if not self.hierarchy.connectorEnd.is_overlapping:
            if self.inCG:
                root = self.hierarchy.connectorEnd.myGlobalLeaderRank
            if self.inSubdomain:
                root = self.hierarchy.connectorEnd.otherLeaderRank
        else:
            root = self.hierarchy.connectorEnd.myGlobalLeaderRank
        self.solver_description = self.comm.bcast(self.solver_description, root=root)
        self.initialized = True
