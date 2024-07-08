###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from __future__ import print_function
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
import logging
import numpy as np
from PyNucleus_base.myTypes import REAL, INDEX
from . levels import meshLevel, algebraicLevel
from . hierarchies import EmptyHierarchy, hierarchy, pCoarsenHierarchy
from PyNucleus_base.utilsFem import TimerManager
from PyNucleus_fem.factories import meshFactory
from PyNucleus_fem.repartitioner import Repartitioner

LOGGER = logging.getLogger(__name__)


class hierarchyConnector(object):
    def __init__(self, global_comm, comm1, comm2, hierarchy1):
        self.global_comm = global_comm
        self.comm1 = comm1
        self.comm2 = comm2
        self.hierarchy1 = hierarchy1
        if self.hierarchy1 is not None:
            self.hierarchy1.connectorEnd = self
        self.Timer = TimerManager(LOGGER,
                                  comm=self.global_comm,
                                  # prefix=label
                                  )
        if self.comm1 is None and self.comm2 is not None:
            if self.global_comm.size == 1:
                self.is_overlapping = True
            else:
                self.is_overlapping = self.global_comm.allreduce(comm1 is not None and comm2 is not None, MPI.LOR)
        else:
            self.is_overlapping = self.global_comm.allreduce(comm1 is not None and comm2 is not None, MPI.LOR)
        if not self.is_overlapping:
            req1 = []
            req2 = []
            if self.comm1 is not None:
                self.myLeaderRank = 0
                self.myGlobalLeaderRank = self.comm1.bcast(self.global_comm.rank, root=self.myLeaderRank)
                if self.comm1.rank == self.myLeaderRank:
                    req1.append(self.global_comm.isend('me', dest=0, tag=770))
            if self.comm2 is not None:
                self.myLeaderRank = 0
                self.myGlobalLeaderRank = self.comm2.bcast(self.global_comm.rank, root=self.myLeaderRank)
                if self.comm2.rank == self.myLeaderRank:
                    req1.append(self.global_comm.isend('me', dest=0, tag=771))
            if self.global_comm.rank == 0:
                status = MPI.Status()
                self.global_comm.recv(source=MPI.ANY_SOURCE, status=status, tag=770)
                rank1 = status.source
                status = MPI.Status()
                self.global_comm.recv(source=MPI.ANY_SOURCE, status=status, tag=771)
                rank2 = status.source
                req2.append(self.global_comm.isend(rank2, dest=rank1, tag=772))
                req2.append(self.global_comm.isend(rank1, dest=rank2, tag=773))
            MPI.Request.Waitall(req1)
            if self.comm1 is not None:
                if self.comm1.rank == self.myLeaderRank:
                    self.otherLeaderRank = self.global_comm.recv(source=0, tag=772)
                    self.comm1.bcast(self.otherLeaderRank, root=self.myLeaderRank)
                else:
                    self.otherLeaderRank = self.comm1.bcast(-1, root=self.myLeaderRank)
            if self.comm2 is not None:
                if self.comm2.rank == self.myLeaderRank:
                    self.otherLeaderRank = self.global_comm.recv(source=0, tag=773)
                    self.comm2.bcast(self.otherLeaderRank, root=self.myLeaderRank)
                else:
                    self.otherLeaderRank = self.comm2.bcast(-1, root=self.myLeaderRank)
            MPI.Request.Waitall(req2)

            if self.comm1 is not None:
                self._oldSubdomainGlobalRank = np.array(self.comm1.allgather(self.global_comm.rank), dtype=INDEX)
                self._newSubdomainGlobalRank = None

                self.interComm = self.comm1.Create_intercomm(self.myLeaderRank, self.global_comm, self.otherLeaderRank)
                self.interComm.bcast(self._oldSubdomainGlobalRank, root=MPI.ROOT if self.comm1.rank == self.myLeaderRank else MPI.PROC_NULL)
                self._newSubdomainGlobalRank = self.interComm.bcast(self._newSubdomainGlobalRank, root=0)

            if self.comm2 is not None:
                self._oldSubdomainGlobalRank = None
                self._newSubdomainGlobalRank = np.array(self.comm2.allgather(self.global_comm.rank), dtype=INDEX)

                self.interComm = self.comm2.Create_intercomm(self.myLeaderRank, self.global_comm, self.otherLeaderRank)
                self._oldSubdomainGlobalRank = self.interComm.bcast(self._oldSubdomainGlobalRank, root=0)
                self.interComm.bcast(self._newSubdomainGlobalRank, root=MPI.ROOT if self.comm2.rank == self.myLeaderRank else MPI.PROC_NULL)
        else:
            inBothComms = self.comm1 is not None and self.comm2 is not None
            self.myGlobalLeaderRank = self.global_comm.allreduce(self.global_comm.rank if inBothComms else self.global_comm.size, op=MPI.MIN)
            self._oldSubdomainGlobalRank = np.arange(self.global_comm.size, dtype=INDEX)
            self._newSubdomainGlobalRank = np.arange(self.global_comm.size, dtype=INDEX)

        self._oldRankSubdomainNo = {self._oldSubdomainGlobalRank[subdomainNo]: subdomainNo for subdomainNo in range(self._oldSubdomainGlobalRank.shape[0])}
        self._newRankSubdomainNo = {self._newSubdomainGlobalRank[subdomainNo]: subdomainNo for subdomainNo in range(self._newSubdomainGlobalRank.shape[0])}

    def getNewHierarchy(self):
        raise NotImplementedError()

    def getLevelList(self):
        if self.hierarchy1 is not None:
            return self.hierarchy1.getLevelList()
        else:
            return []

    def build(self):
        pass

    def comm1SubdomainGlobalRank(self, subdomainNo):
        return self._oldSubdomainGlobalRank[subdomainNo]

    def comm2SubdomainGlobalRank(self, subdomainNo):
        return self._newSubdomainGlobalRank[subdomainNo]

    def comm1RankSubdomainNo(self, rank):
        return self._oldRankSubdomainNo[rank]

    def comm2RankSubdomainNo(self, rank):
        return self._newRankSubdomainNo[rank]


class inputConnector(hierarchyConnector):
    def __init__(self, global_comm, comm1, comm2, hierarchy1, domain,
                 algebraicLevelType=algebraicLevel, meshParams={}):
        super(inputConnector, self).__init__(global_comm, comm1, comm2, hierarchy1)
        self.domain = domain
        self.meshParams = meshParams
        self.algebraicLevelType = algebraicLevelType

    def getNewHierarchy(self, params):
        with self.Timer('Initializing mesh on \'{}\''.format(params['label'])):
            mesh = meshFactory.build(self.domain, **self.meshParams)
            if self.hierarchy1 is not None:
                startLevelNo = self.hierarchy1.meshLevels[-1].levelNo
            else:
                startLevelNo = 0
            level = meshLevel(mesh, params['params'], label=params['label'], comm=params['comm'], startLevelNo=startLevelNo)
            level.setAlgebraicLevelType(self.algebraicLevelType)
            h = hierarchy(level, params['params'], comm=params['comm'], label=params['label'])
            h.connectorStart = self
        return h


class repartitionConnector(hierarchyConnector):
    def __init__(self, global_comm, comm1, comm2, hierarchy1, partitionerType, partitionerParams,
                 debugOverlaps=False, commType='standard', algebraicLevelType=algebraicLevel):
        super(repartitionConnector, self).__init__(global_comm, comm1, comm2, hierarchy1)
        self.partitionerType = partitionerType
        self.partitionerParams = partitionerParams
        self.debugOverlaps = debugOverlaps
        self.commType = commType
        self.splitOM = None
        self.algebraicLevelType = algebraicLevelType

    def getNewHierarchy(self, params):
        if self.hierarchy1 is not None:
            label1 = self.hierarchy1.label
        else:
            label1 = ''
        self.label2 = params['label']
        with self.Timer('Repartitioning from \'{}\' to \'{}\' using \'{}\''.format(label1, params['label'], self.partitionerType)):
            if self.comm1 is not None:
                subdomain = self.hierarchy1.meshLevels[-1].mesh
                interfaces = self.hierarchy1.meshLevels[-1].interfaces
                rep = Repartitioner(subdomain, interfaces, self.global_comm, self.comm1, self.comm2)

                self.repartitioner = rep
                rep.getCellPartition(self.partitionerType, self.partitionerParams)
                subdomainNew, self.OM, self.OMnew, iM = rep.getRepartitionedSubdomains()

                if self.debugOverlaps and not self.is_overlapping:
                    self.OM.check(subdomain, self.global_comm, 'meshOverlaps from \'{}\' to \'{}\''.format(self.hierarchy1.label, params['label']))
                if self.hierarchy1 is not None:
                    startLevelNo = self.hierarchy1.meshLevels[-1].levelNo
                else:
                    startLevelNo = 0
            else:
                rep = Repartitioner(None, None, self.global_comm, self.comm1, self.comm2)
                self.repartitioner = rep
                subdomainNew, self.OM, self.OMnew, iM = rep.getRepartitionedSubdomains()
                if self.debugOverlaps and not self.is_overlapping:
                    self.OMnew.check(subdomainNew, self.global_comm, 'meshOverlaps from \'{}\' to \'{}\''.format(self.hierarchy1.label, params['label']))
                startLevelNo = 0
            startLevelNo = self.global_comm.bcast(startLevelNo)
            if self.comm2 is not None:
                hierarchy.updateParamsFromDefaults(params['params'])
                level = meshLevel(subdomainNew,
                                  params['params'],
                                  interfaces=iM,
                                  label=params['label'],
                                  comm=params['comm'],
                                  startLevelNo=startLevelNo)
                level.setAlgebraicLevelType(self.algebraicLevelType)
                h = hierarchy(level, params['params'], comm=params['comm'], label=params['label'])
                h.connectorStart = self
                self.hierarchy2 = h
            else:
                h = EmptyHierarchy(params['params'], label=params['label'])
                h.connectorStart = self
                self.hierarchy2 = h
        self.getLocalOverlap()
        return h

    def getLocalOverlap(self):
        if self.is_overlapping and self.comm1 is not None:
            subdomain = self.hierarchy1.meshLevels[-1].mesh
            if self.global_comm.rank in self.OM.overlaps:
                print(('cells kept local on rank {} in repartitioning: ' +
                       '{:,} / target: {:,}').format(self.global_comm.rank,
                                                     self.OM.overlaps[self.global_comm.rank].num_cells/subdomain.num_cells,
                                                     self.comm1.size/self.global_comm.size))
            else:
                print(('cells kept local on rank {} in repartitioning: ' +
                       '{:,} / target: {:,}').format(self.global_comm.rank,
                                                     0.,
                                                     self.comm1.size/self.global_comm.size))

    def build(self):
        if self.hierarchy1 is not None:
            label1 = self.hierarchy1.label
        else:
            label1 = ''
        self.global_comm.Barrier()
        if self.OM is not None and self.OMnew is None:
            self.global_comm.Barrier()
            with self.Timer('Building algebraic overlaps of type \'{}\' from \'{}\' to \'{}\' using Alltoallv'.format(self.commType, label1, self.label2)):
                subdomain = self.hierarchy1.meshLevels[-1].mesh
                dm = self.hierarchy1.algebraicLevels[-1].DoFMap
                self.algOM = self.OM.getDoFs(subdomain, dm, overlapType=self.commType,
                                             allowInteriorBoundary=True, useRequests=self.commType == 'standard', splitManager=self.splitOM)
                if self.debugOverlaps and not self.is_overlapping:
                    self.algOM.check(subdomain, dm, 'algebraicOverlaps from \'{}\' to \'{}\''.format(label1, self.label2))
            self.global_comm.Barrier()
            with self.Timer('Building distribute from \'{}\' to \'{}\''.format(label1, self.label2)):
                self.algOM.prepareDistributeRepartition(dm)
            if self.debugOverlaps:
                from PyNucleus_fem.factories import solSin1D, solSin2D, solSin3D
                if subdomain.dim == 1:
                    xOld = dm.interpolate(solSin1D)
                elif subdomain.dim == 2:
                    xOld = dm.interpolate(solSin2D)
                elif subdomain.dim == 3:
                    xOld = dm.interpolate(solSin3D)
                else:
                    raise NotImplementedError()
                self.algOM.send_py(xOld)

                yOld = np.zeros((dm.num_dofs), dtype=REAL)
                self.algOM.receive_py(yOld)
                self.algOM.distribute_py(yOld)
                assert np.linalg.norm(xOld-yOld) < 1e-9, (xOld, yOld)

        if self.OM is None and self.OMnew is not None:
            self.global_comm.Barrier()
            with self.Timer('Building algebraic overlaps of type \'{}\' from \'{}\' to \'{}\' using Alltoallv'.format(self.commType, label1, self.label2)):
                subdomainNew = self.hierarchy2.meshLevels[0].mesh
                dmNew = self.hierarchy2.algebraicLevels[0].DoFMap
                self.algOMnew = self.OMnew.getDoFs(subdomainNew, dmNew, overlapType=self.commType,
                                                   allowInteriorBoundary=True, useRequests=self.commType == 'standard', splitManager=self.splitOM)
                if self.debugOverlaps and not self.is_overlapping:
                    self.algOMnew.check(subdomainNew, dmNew, 'algebraicOverlaps from \'{}\' to \'{}\''.format(label1, self.label2))
            self.global_comm.Barrier()
            with self.Timer('Building distribute from \'{}\' to \'{}\''.format(label1, self.label2)):
                self.algOMnew.prepareDistributeRepartition(dmNew)
            if self.debugOverlaps:
                from PyNucleus_fem.factories import solSin1D, solSin2D, solSin3D
                if subdomainNew.dim == 1:
                    xNew = dmNew.interpolate(solSin1D)
                elif subdomainNew.dim == 2:
                    xNew = dmNew.interpolate(solSin2D)
                elif subdomainNew.dim == 3:
                    xNew = dmNew.interpolate(solSin3D)
                else:
                    raise NotImplementedError()

                yNew = np.zeros((dmNew.num_dofs), dtype=REAL)
                self.algOMnew.receive_py(yNew)
                self.algOMnew.distribute_py(yNew)
                assert np.linalg.norm(xNew-yNew) < 1e-9, (xNew, yNew)

                self.algOMnew.send_py(xNew)

        if self.OM is not None and self.OMnew is not None:
            self.global_comm.Barrier()
            with self.Timer('Building algebraic overlaps of type \'{}\' from \'{}\' to \'{}\' using Alltoallv'.format(self.commType, label1, self.label2)):
                subdomain = self.hierarchy1.meshLevels[-1].mesh
                dm = self.hierarchy1.algebraicLevels[-1].DoFMap
                assert dm.num_dofs > 0
                self.algOM = self.OM.getDoFs(subdomain, dm, overlapType=self.commType,
                                             allowInteriorBoundary=True, useRequests=self.commType == 'standard', waitRequests=False)

                subdomainNew = self.hierarchy2.meshLevels[0].mesh
                dmNew = self.hierarchy2.algebraicLevels[0].DoFMap
                self.algOMnew = self.OMnew.getDoFs(subdomainNew, dmNew, overlapType=self.commType,
                                                   allowInteriorBoundary=True, useRequests=self.commType == 'standard')
                MPI.Request.Waitall(self.OM.requests)
                self.OM.requests = []

            if self.debugOverlaps and not self.is_overlapping:
                self.algOM.check(subdomain, dm, 'algebraicOverlaps from \'{}\' to \'{}\''.format(label1, self.label2))
            self.global_comm.Barrier()
            with self.Timer('Building distribute from \'{}\' to \'{}\''.format(label1, self.label2)):
                self.algOMnew.prepareDistributeRepartitionSend(dmNew)
                self.algOM.prepareDistributeRepartition(dm, doSend=False)
                self.algOM.prepareDistributeRepartitionSend(dm)
                self.algOMnew.prepareDistributeRepartition(dmNew, doSend=False)
            if self.debugOverlaps:
                from PyNucleus_fem.factories import solSin1D, solSin2D, solSin3D
                if subdomain.dim == 1:
                    xOld = dm.interpolate(solSin1D)
                elif subdomain.dim == 2:
                    xOld = dm.interpolate(solSin2D)
                elif subdomain.dim == 3:
                    xOld = dm.interpolate(solSin3D)
                else:
                    raise NotImplementedError()

                if subdomain.dim == 1:
                    xNew = dmNew.interpolate(solSin1D)
                elif subdomain.dim == 2:
                    xNew = dmNew.interpolate(solSin2D)
                elif subdomain.dim == 3:
                    xNew = dmNew.interpolate(solSin3D)
                else:
                    raise NotImplementedError()

                self.algOM.send_py(xOld)
                yNew = np.zeros((dmNew.num_dofs), dtype=REAL)
                self.algOMnew.receive_py(yNew)
                self.algOMnew.distribute_py(yNew)
                assert np.linalg.norm(xNew-yNew) < 1e-9, (xNew, yNew)

                self.algOMnew.send_py(xNew)
                yOld = np.zeros((dm.num_dofs), dtype=REAL)
                self.algOM.receive_py(yOld)
                self.algOM.distribute_py(yOld)
                assert np.linalg.norm(xOld-yOld) < 1e-9, (xOld, yOld)


class pCoarsenConnector(hierarchyConnector):
    def __init__(self, global_comm, comm1, comm2, hierarchy1, algebraicLevelType=algebraicLevel):
        super(pCoarsenConnector, self).__init__(global_comm, comm1, comm2, hierarchy1)
        self.algebraicLevelType = algebraicLevelType

    def getNewHierarchy(self, params):
        startLevelNo = self.hierarchy1.meshLevels[-1].levelNo
        self.label2 = params['label']
        hierarchy.updateParamsFromDefaults(params['params'])
        level = meshLevel(self.hierarchy1.meshLevels[-1].mesh,
                          params['params'],
                          interfaces=self.hierarchy1.meshLevels[-1].interfaces,
                          label=params['label'],
                          comm=params['comm'],
                          startLevelNo=startLevelNo)
        level.setAlgebraicLevelType(self.algebraicLevelType)
        h = pCoarsenHierarchy(level, params['params'], comm=params['comm'], label=params['label'])
        h.connectorStart = self
        self.hierarchy2 = h
        return h
