###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
cimport numpy as np
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py cimport MPI

import PyNucleus_metisCy
from PyNucleus_metisCy.metisCy import idx as metis_idx, real as metis_real
from PyNucleus_metisCy.metisCy cimport idx_t
from PyNucleus_base.myTypes import INDEX, REAL, ENCODE, TAG
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, ENCODE_t, TAG_t, BOOL_t
from PyNucleus_base import uninitialized
from . DoFMaps import P1_DoFMap
from . mesh import (mesh1d, mesh2d, mesh3d,
                    PHYSICAL, INTERIOR_NONOVERLAPPING)
from . meshCy cimport meshBase
from . simplexMapper cimport simplexMapper
from . meshOverlaps import boundary1D, boundary2D, boundary3D
from . meshOverlaps cimport (meshOverlap, overlapManager,
                             meshInterface, interfaceManager)
from . meshCy cimport encode_edge, encode_face, sortEdge, sortFace, decode_edge
from functools import lru_cache


cdef class Repartitioner:
    cdef:
        meshBase subdomain
        interfaceManager interfaces
        INDEX_t dim
        MPI.Comm globalComm
        MPI.Comm oldComm
        MPI.Comm newComm
        MPI.Comm interComm
        BOOL_t is_overlapping
        INDEX_t[::1] _newSubdomainGlobalRank
        INDEX_t[::1] _oldSubdomainGlobalRank
        dict _newRankSubdomainNo
        dict _oldRankSubdomainNo
        INDEX_t[::1] cells, part
        INDEX_t cell_offset

    def __init__(self, meshBase subdomain, interfaceManager interfaces, MPI.Comm globalComm, MPI.Comm oldComm, MPI.Comm newComm):
        cdef:
            list req1, req2
            INDEX_t myLeaderRank = 0, rank1, rank2, otherLeader = 0
            MPI.Status status
        self.subdomain = subdomain
        self.interfaces = interfaces
        self.globalComm = globalComm
        self.oldComm = oldComm
        self.newComm = newComm
        self.is_overlapping = self.globalComm.allreduce(oldComm is not None and newComm is not None, MPI.LOR)
        if self.subdomain is not None:
            self.dim = self.subdomain.dim
        else:
            self.dim = 0
        self.dim = self.globalComm.allreduce(self.dim, MPI.MAX)
        if not self.is_overlapping:
            req1 = []
            req2 = []
            if self.oldComm is not None:
                myLeaderRank = 0
                if self.oldComm.rank == myLeaderRank:
                    req1.append(self.globalComm.isend('me', dest=0, tag=777))
            if self.newComm is not None:
                myLeaderRank = 0
                if self.newComm.rank == myLeaderRank:
                    req1.append(self.globalComm.isend('me', dest=0, tag=778))
            if self.globalComm.rank == 0:
                status = MPI.Status()
                self.globalComm.recv(source=MPI.ANY_SOURCE, status=status, tag=777)
                rank1 = status.source
                status = MPI.Status()
                self.globalComm.recv(source=MPI.ANY_SOURCE, status=status, tag=778)
                rank2 = status.source
                req2.append(self.globalComm.isend(rank2, dest=rank1, tag=779))
                req2.append(self.globalComm.isend(rank1, dest=rank2, tag=780))
            MPI.Request.Waitall(req1)
            if self.oldComm is not None:
                if self.oldComm.rank == myLeaderRank:
                    otherLeader = self.globalComm.recv(source=0, tag=779)
                else:
                    otherLeader = 0
                otherLeader = self.oldComm.bcast(otherLeader, root=myLeaderRank)
            if self.newComm is not None:
                if self.newComm.rank == myLeaderRank:
                    otherLeader = self.globalComm.recv(source=0, tag=780)
                else:
                    otherLeader = 0
                otherLeader = self.newComm.bcast(otherLeader, root=myLeaderRank)
            MPI.Request.Waitall(req2)

            if self.oldComm is not None:
                self._oldSubdomainGlobalRank = np.array(self.oldComm.allgather(self.globalComm.rank), dtype=INDEX)
                self._newSubdomainGlobalRank = None

                self.interComm = self.oldComm.Create_intercomm(myLeaderRank, self.globalComm, otherLeader)
                self.interComm.bcast(np.array(self._oldSubdomainGlobalRank), root=MPI.ROOT if self.oldComm.rank == myLeaderRank else MPI.PROC_NULL)
                self._newSubdomainGlobalRank = self.interComm.bcast(self._newSubdomainGlobalRank, root=0)

            if self.newComm is not None:
                self._oldSubdomainGlobalRank = None
                self._newSubdomainGlobalRank = np.array(self.newComm.allgather(self.globalComm.rank), dtype=INDEX)

                self.interComm = self.newComm.Create_intercomm(myLeaderRank, self.globalComm, otherLeader)
                self._oldSubdomainGlobalRank = self.interComm.bcast(self._oldSubdomainGlobalRank, root=0)
                self.interComm.bcast(np.array(self._newSubdomainGlobalRank), root=MPI.ROOT if self.newComm.rank == myLeaderRank else MPI.PROC_NULL)
        else:
            self._oldSubdomainGlobalRank = np.arange(self.globalComm.size, dtype=INDEX)
            self._newSubdomainGlobalRank = np.arange(self.globalComm.size, dtype=INDEX)

        self._oldRankSubdomainNo = {self._oldSubdomainGlobalRank[subdomainNo]: subdomainNo for subdomainNo in range(self._oldSubdomainGlobalRank.shape[0])}
        self._newRankSubdomainNo = {self._newSubdomainGlobalRank[subdomainNo]: subdomainNo for subdomainNo in range(self._newSubdomainGlobalRank.shape[0])}

    @lru_cache(maxsize=1)
    def getGlobalVertexIndices(self):
        if self.oldComm is not None:
            subdomain = self.subdomain
            if self.interfaces is None:
                return np.arange((subdomain.num_vertices), dtype=INDEX)
            else:
                dm = P1_DoFMap(subdomain, 10)
                ov = self.interfaces.getDoFs(self.subdomain, dm)
                return ov.getGlobalIndices()

    globalVertexIndices = property(fget=getGlobalVertexIndices)

    cdef INDEX_t oldSubdomainGlobalRank_c(self, INDEX_t subdomainNo):
        return self._oldSubdomainGlobalRank[subdomainNo]

    cdef INDEX_t newSubdomainGlobalRank_c(self, INDEX_t subdomainNo):
        return self._newSubdomainGlobalRank[subdomainNo]

    cdef INDEX_t oldRankSubdomainNo_c(self, INDEX_t rank):
        return self._oldRankSubdomainNo[rank]

    cdef INDEX_t newRankSubdomainNo_c(self, INDEX_t rank):
        return self._newRankSubdomainNo[rank]

    def oldSubdomainGlobalRank(self, INDEX_t subdomainNo):
        return self._oldSubdomainGlobalRank[subdomainNo]

    def newSubdomainGlobalRank(self, INDEX_t subdomainNo):
        return self._newSubdomainGlobalRank[subdomainNo]

    def oldRankSubdomainNo(self, INDEX_t rank):
        return self._oldRankSubdomainNo[rank]

    def newRankSubdomainNo(self, INDEX_t rank):
        return self._newRankSubdomainNo[rank]

    def getNumPartitions(self):
        if self.newComm is not None:
            return self.newComm.size
        else:
            return self.globalComm.size-self.oldComm.size

    numPartitions = property(fget=getNumPartitions)

    def getCellPartition(self, partitioner='parmetis', partitionerParams={}):
        cdef:
            INDEX_t i, c, dim, numVerticesPerCell, numCells
            meshBase subdomain
            INDEX_t[::1] globalVertexIndices
            INDEX_t[::1] cells
            idx_t[::1] cell_dist
            INDEX_t rank

        if self.oldComm is not None:
            dim = self.dim
            numVerticesPerCell = dim+1
            subdomain = self.subdomain
            numCells = subdomain.num_cells
            if 'partition_weights' in partitionerParams:
                partition_weights = partitionerParams['partition_weights']
            else:
                partition_weights = None

            if partitioner == 'parmetis':
                cell_ptr = np.arange(0, numVerticesPerCell*(numCells+1),
                                     numVerticesPerCell,
                                     dtype=metis_idx)

                cells2 = subdomain.cells_as_array.astype(metis_idx)
                cells2.resize((numVerticesPerCell*numCells, ))
                cells = cells2
                self.cells = cells

                globalVertexIndices = self.globalVertexIndices
                for i in range(cells.shape[0]):
                    c = cells[i]
                    cells[i] = globalVertexIndices[c]

                cell_dist = np.array(self.oldComm.allgather(numCells),
                                     dtype=metis_idx)

                cell_dist = np.concatenate((np.zeros((1), dtype=metis_idx),
                                            np.cumsum(cell_dist, dtype=metis_idx)))
                rank = self.oldComm.rank
                self.cell_offset = cell_dist[rank]

                if partition_weights is not None and partition_weights.dtype != metis_real:
                    partition_weights = partition_weights.astype(metis_real)
                if partition_weights is not None and partition_weights.ndim == 1:
                    partition_weights = partition_weights[:, np.newaxis]

                self.part = PyNucleus_metisCy.parmetisCy.PartMeshKway(cell_dist,
                                                                      cell_ptr,
                                                                      np.array(cells, copy=False).astype(metis_idx),
                                                                      subdomain.dim,
                                                                      self.numPartitions,
                                                                      self.oldComm,
                                                                      tpwgts=partition_weights)
                if self.oldComm.size > 1:
                    self.reorderPartitioning()
            elif partitioner in ('metis', 'regular'):
                cells2 = subdomain.cells_as_array.copy()
                cells2.resize((numVerticesPerCell*numCells, ))
                cells = cells2
                self.cells = cells

                globalVertexIndices = self.globalVertexIndices
                for i in range(cells.shape[0]):
                    c = cells[i]
                    cells[i] = globalVertexIndices[c]

                if self.oldComm.size == 1:
                    numPartitions = self.numPartitions
                    partitionOffset = 0
                else:
                    cellsPerSubdomain = self.oldComm.gather(numCells, root=0)
                    if self.oldComm.rank == 0:
                        cellsPerSubdomain = np.array(cellsPerSubdomain)
                        numPartitionsPerSubdomain = self.numPartitions*(cellsPerSubdomain/cellsPerSubdomain.sum())
                        numPartitionsPerSubdomainInt = np.around(numPartitionsPerSubdomain).astype(INDEX)

                        if numPartitionsPerSubdomainInt.sum() < self.numPartitions:
                            while numPartitionsPerSubdomainInt.sum() != self.numPartitions:
                                numPartitionsPerSubdomainInt[(numPartitionsPerSubdomain-numPartitionsPerSubdomainInt).argmax()] += 1
                        elif numPartitionsPerSubdomainInt.sum() > self.numPartitions:
                            while numPartitionsPerSubdomainInt.sum() != self.numPartitions:
                                numPartitionsPerSubdomainInt[(numPartitionsPerSubdomain-numPartitionsPerSubdomainInt).argmin()] -= 1

                        partitionOffsetPerSubdomain = np.concatenate(([0], np.cumsum(numPartitionsPerSubdomainInt)[:numPartitionsPerSubdomainInt.shape[0]-1]))
                    else:
                        numPartitionsPerSubdomainInt = None
                        partitionOffsetPerSubdomain = None
                    numPartitions = self.oldComm.scatter(numPartitionsPerSubdomainInt)
                    partitionOffset = self.oldComm.scatter(partitionOffsetPerSubdomain)

                cell_dist = np.array(self.oldComm.allgather(numCells),
                                     dtype=metis_idx)

                cell_dist = np.concatenate((np.zeros((1), dtype=metis_idx),
                                            np.cumsum(cell_dist, dtype=metis_idx)))
                rank = self.oldComm.rank
                self.cell_offset = cell_dist[rank]

                if partition_weights is not None:
                    myPartitionWeights = partition_weights[partitionOffset:partitionOffset+numPartitions]
                    myPartitionWeights /= myPartitionWeights.sum()
                    myPartitionWeights = myPartitionWeights.astype(metis_real)
                else:
                    myPartitionWeights = None

                if partitioner == 'metis':
                    from . meshPartitioning import metisMeshPartitioner
                    mP = metisMeshPartitioner(subdomain)
                elif partitioner == 'regular':
                    from . meshPartitioning import regularMeshPartitioner
                    mP = regularMeshPartitioner(subdomain)
                else:
                    raise NotImplementedError()
                partitionerParams['partition_weights'] = myPartitionWeights
                part, actualNumPartitions = mP.partitionCells(numPartitions, **partitionerParams)
                assert actualNumPartitions == numPartitions

                for i in range(part.shape[0]):
                    part[i] += partitionOffset
                self.part = part
            else:
                raise NotImplementedError(partitioner)
            return self.part

    def reorderPartitioning(self):
        """We collect the information of how many cells f_{p,q} of which
        partition q are on each subdomain p. Then we solve a linear
        program

        max_n \\sum_{p=1}^P \\sum_{q=1}^Q f_{p,q} * n_{p,q}
        subject to
        sum_q n_{p,q}  = 1  \\forall p = 1,..,P  (each subdomain gets one partition)
        sum_p n_{p,q} <= 1  \\forall q = 1,..,Q  (each partition gets at most one subdomain)

        """
        cdef:
            INDEX_t i, p
            INDEX_t[::1] count, counts, mapping
        if self.oldComm is not None:
            count = np.zeros((self.numPartitions), dtype=INDEX)
            for i in range(self.part.shape[0]):
                count[self.part[i]] += 1
            counts = self.oldComm.gather(count, root=0)
            if self.oldComm.rank == 0:
                P = self.oldComm.size
                Q = self.numPartitions
                F = np.concatenate(counts)
                A_eq = np.zeros((P, P*Q))
                for p in range(P):
                    A_eq[p, Q*p:Q*(p+1)] = 1
                b_eq = np.ones((P))

                A_ub = np.zeros((Q, P*Q))
                for q in range(Q):
                    for p in range(P):
                        A_ub[q, Q*p+q] = 1
                b_ub = np.ones((Q))

                from scipy.optimize import linprog
                res = linprog(-F.ravel(),
                              A_eq=A_eq, b_eq=b_eq,
                              A_ub=A_ub, b_ub=b_ub)
                mapping = uninitialized((Q), dtype=INDEX)
                leftOverParts = set(range(Q))
                for p in range(P):
                    mapping[p] = res.x[p*Q:(p+1)*Q].argmax()
                    leftOverParts.remove(mapping[p])
                for p in range(P, Q):
                    mapping[p] = leftOverParts.pop()
                assert np.unique(mapping).shape[0] == Q
            else:
                mapping = uninitialized((0), dtype=INDEX)
            mapping = self.oldComm.bcast(mapping)
            inv_mapping = uninitialized((self.numPartitions), dtype=INDEX)
            inv_mapping[mapping] = np.arange(self.numPartitions, dtype=INDEX)
            assert np.unique(inv_mapping).shape[0] == self.numPartitions
            for i in range(self.part.shape[0]):
                self.part[i] = inv_mapping[self.part[i]]
            # print(self.oldComm.rank, count[self.oldComm.rank], count[mapping[self.oldComm.rank]])

    def getRepartitionedSubdomains(self):
        cdef:
            INDEX_t dim, numVerticesPerCell = -1, i, j, k, m, l, p
            list req = [], sendRequests = []
            meshBase subdomain
            INDEX_t numCells = -1, subdomainNo
            INDEX_t[::1] part, cells, numCellsNew, numCellsNewLocal
            INDEX_t[::1] cellsToSend, cellsToRecv, globalCellIdxToSend, globalCellIdxToRecv, cellsToSendPtr, sendCount, sendDispl, recvCount, recvDispl
            dict globalToLocalCells
            INDEX_t rank, count
            INDEX_t[::1] gVITS
            REAL_t[:, ::1] vTS
            INDEX_t cellNo, vertexNo, vertex
            list partitionVertices
            INDEX_t[::1] globalVertexIndices
            MPI.Comm globalComm = self.globalComm
            INDEX_t globalCommSize = globalComm.size
            INDEX_t[::1] newSubdomainNos, counts
            INDEX_t[::1] c2
            REAL_t[:, ::1] newVertices
            INDEX_t[::1] overlapCount
            localInterfaceManager lIM
            interfaceProcessor iP
        dim = self.dim
        numVerticesPerCell = dim+1
        req = []
        if self.oldComm is not None:
            subdomain = self.subdomain
            numCells = subdomain.num_cells
            part = self.part
            numCellsNew = np.zeros((self.numPartitions), dtype=INDEX)
            for i in range(part.shape[0]):
                numCellsNew[part[i]] += 1
            numCellsNewLocal = np.zeros((self.numPartitions), dtype=INDEX)
            self.oldComm.Reduce(numCellsNew, numCellsNewLocal, root=0)
            if self.oldComm.rank == 0:
                req.append(globalComm.Isend(numCellsNewLocal, dest=self.newSubdomainGlobalRank_c(0), tag=15))

        if self.newComm is not None:
            numCellsNew = uninitialized((self.numPartitions), dtype=INDEX)
            numCellsNewLocal = uninitialized((1), dtype=INDEX)
            if self.newComm.rank == 0:
                globalComm.Recv(numCellsNew, source=self.oldSubdomainGlobalRank_c(0), tag=15)
            self.newComm.Scatter(numCellsNew, numCellsNewLocal, root=0)

        MPI.Request.Waitall(req)

        if self.oldComm is not None:
            # prepare cells for sending
            newSubdomainNos, countsLong = np.unique(part, return_counts=True)
            counts = countsLong.astype(INDEX)
            cells = self.cells
            cellsToSend = np.zeros((cells.shape[0]), dtype=INDEX)
            globalCellIdxToSend = np.zeros((cells.shape[0]//numVerticesPerCell), dtype=INDEX)
            cellsToSendPtr = np.zeros((globalCommSize+1), dtype=INDEX)
            sendCount = np.zeros((globalCommSize), dtype=INDEX)
            sendDispl = np.zeros((globalCommSize), dtype=INDEX)

            for k in range(newSubdomainNos.shape[0]):
                subdomainNo = newSubdomainNos[k]
                count = counts[k]
                rank = self.newSubdomainGlobalRank_c(subdomainNo)
                cellsToSendPtr[rank+1] = count
                sendCount[rank] = count
            for i in range(globalCommSize):
                cellsToSendPtr[i+1] += cellsToSendPtr[i]
                sendDispl[i] = cellsToSendPtr[i]

            for j in range(part.shape[0]):
                subdomainNo = part[j]
                rank = self.newSubdomainGlobalRank_c(subdomainNo)
                globalCellIdxToSend[cellsToSendPtr[rank]] = j+self.cell_offset
                cellsToSendPtr[rank] += 1
            for rank in range(globalCommSize, 0, -1):
                cellsToSendPtr[rank] = cellsToSendPtr[rank-1]
            cellsToSendPtr[0] = 0

            for i in range(numCells):
                rank = self.newSubdomainGlobalRank_c(part[i])
                j = cellsToSendPtr[rank]
                cellsToSendPtr[rank] += 1
                for k in range(numVerticesPerCell):
                    cellsToSend[numVerticesPerCell*j+k] = cells[numVerticesPerCell*i+k]
        else:
            globalCellIdxToSend = uninitialized((0), dtype=INDEX)
            sendCount = np.zeros((globalCommSize), dtype=INDEX)
            sendDispl = np.zeros((globalCommSize), dtype=INDEX)
            cellsToSend = np.zeros((0), dtype=INDEX)

        # prepare cells for receiving
        if self.oldComm is not None:
            sendRequests = []
            for i in range(newSubdomainNos.shape[0]):
                sendRequests.append(globalComm.Isend(counts[i:i+1],
                                                     dest=self.newSubdomainGlobalRank_c(newSubdomainNos[i]),
                                                     tag=121))
        if self.newComm is not None:
            c = 0
            c2 = uninitialized((1), dtype=INDEX)
            recvCount = np.zeros((globalCommSize), dtype=INDEX)
            recv_ranks = set()
            while c < numCellsNewLocal[0]:
                status = MPI.Status()
                globalComm.Recv(c2, source=MPI.ANY_SOURCE, status=status, tag=121)
                rank = status.source
                subdomainNo = self.oldRankSubdomainNo_c(rank)
                recvCount[rank] = c2[0]
                c += c2[0]
                recv_ranks.add(subdomainNo)
        if self.oldComm is not None:
            MPI.Request.Waitall(sendRequests)

        if self.newComm is not None:
            recvDispl = np.zeros((globalCommSize), dtype=INDEX)
            for rank in range(1, globalCommSize):
                recvDispl[rank] = recvDispl[rank-1]+recvCount[rank-1]

            globalCellIdxToRecv = uninitialized((numCellsNewLocal[0]), dtype=INDEX)
        else:
            globalCellIdxToRecv = uninitialized((0), dtype=INDEX)
            recvDispl = np.zeros((globalCommSize), dtype=INDEX)
            recvCount = np.zeros((globalCommSize), dtype=INDEX)

        globalComm.Alltoallv([globalCellIdxToSend, (sendCount, sendDispl)],
                             [globalCellIdxToRecv, (recvCount, recvDispl)])

        if self.newComm is not None:
            globalToLocalCells = {}
            for localCellNo in range(globalCellIdxToRecv.shape[0]):
                globalCellNo = globalCellIdxToRecv[localCellNo]
                globalToLocalCells[globalCellNo] = localCellNo

        for i in range(globalCommSize):
            sendCount[i] *= numVerticesPerCell
            sendDispl[i] *= numVerticesPerCell
            recvCount[i] *= numVerticesPerCell
            recvDispl[i] *= numVerticesPerCell

        if self.newComm is not None:
            cellsToRecv = uninitialized((numVerticesPerCell*numCellsNewLocal[0]), dtype=INDEX)
        else:
            cellsToRecv = uninitialized((0), dtype=INDEX)

        globalComm.Alltoallv([cellsToSend, (sendCount, sendDispl)],
                             [cellsToRecv, (recvCount, recvDispl)])

        ######################################################################
        # exchange vertices

        if self.oldComm is not None:
            sendRequests = []
            globalVertexIndicesToSend = {}
            verticesToSend = {}
            partitionVertices = [set() for p in range(globalCommSize)]
            for cellNo in range(subdomain.num_cells):
                p = part[cellNo]
                for vertexNo in range(subdomain.dim+1):
                    vertex = subdomain.cells[cellNo, vertexNo]
                    partitionVertices[p].add(vertex)
            globalVertexIndices = self.globalVertexIndices
            for i in newSubdomainNos:
                n = len(partitionVertices[i])
                gVITS = uninitialized(n, dtype=INDEX)
                vTS = uninitialized((n, dim), dtype=REAL)
                k = 0
                for vertex in partitionVertices[i]:
                    gVITS[k] = globalVertexIndices[vertex]
                    for j in range(dim):
                        vTS[k, j] = subdomain.vertices[vertex, j]
                    k += 1
                rank = self.newSubdomainGlobalRank_c(i)
                globalVertexIndicesToSend[i] = gVITS
                verticesToSend[i] = vTS
                sendRequests.append(globalComm.isend(n, dest=rank, tag=1))
                sendRequests.append(globalComm.Isend(globalVertexIndicesToSend[i],
                                                     dest=rank, tag=2))
                sendRequests.append(globalComm.Isend(verticesToSend[i],
                                                     dest=rank, tag=3))

        if self.newComm is not None:
            globalVertexIndicesToRecv = {}
            verticesToRecv = {}
            globalToLocal = {}
            for i in recv_ranks:
                rank = self.oldSubdomainGlobalRank_c(i)
                n = globalComm.recv(source=rank, tag=1)
                globalVertexIndicesToRecv[i] = np.zeros((n), dtype=INDEX)
                verticesToRecv[i] = np.zeros((n, dim), dtype=REAL)
                globalComm.Recv(globalVertexIndicesToRecv[i],
                                source=rank, tag=2)
                globalComm.Recv(verticesToRecv[i], source=rank, tag=3)
                for j in range(globalVertexIndicesToRecv[i].shape[0]):
                    if globalVertexIndicesToRecv[i][j] not in globalToLocal:
                        globalToLocal[globalVertexIndicesToRecv[i][j]] = (i, j)

            newVertices = uninitialized((len(globalToLocal), dim), dtype=REAL)
            k = 0
            for i in globalToLocal:
                j, m = globalToLocal[i]
                for l in range(dim):
                    newVertices[k, l] = verticesToRecv[j][m, l]
                globalToLocal[i] = k
                k += 1
            localToGlobal = uninitialized((len(globalToLocal)), dtype=INDEX)
            for globalVertexNo, localVertexNo in globalToLocal.items():
                localToGlobal[localVertexNo] = globalVertexNo

            cellsToRecv2 = uninitialized((numCellsNewLocal[0], numVerticesPerCell), dtype=INDEX)
            k = 0
            for i in range(numCellsNewLocal[0]):
                for j in range(numVerticesPerCell):
                    cellsToRecv2[i, j] = globalToLocal[cellsToRecv[k]]
                    k += 1

        if self.oldComm is not None:
            MPI.Request.Waitall(sendRequests)

        ######################################################################
        # build new subdomain

        if self.newComm is not None:
            if dim == 1:
                subdomainNew = mesh1d(newVertices, cellsToRecv2)
            elif dim == 2:
                subdomainNew = mesh2d(newVertices, cellsToRecv2)
            elif dim == 3:
                subdomainNew = mesh3d(newVertices, cellsToRecv2)
            else:
                raise NotImplementedError()
        else:
            subdomainNew = None

        ######################################################################
        # build mesh overlap between old and new partitioning

        if self.oldComm is not None:
            overlapCells = {}
            for k in range(newSubdomainNos.shape[0]):
                subdomainNo = newSubdomainNos[k]
                count = counts[k]
                overlapCells[subdomainNo] = uninitialized((count), dtype=INDEX)
            overlapCount = np.zeros((self.numPartitions), dtype=INDEX)
            for i in range(numCells):
                p = part[i]
                k = overlapCount[p]
                overlapCount[p] += 1
                overlapCells[p][k] = i

            OM = overlapManager(globalComm)
            for subdomainNo in newSubdomainNos:
                rank = self.newSubdomainGlobalRank_c(subdomainNo)
                OM.overlaps[rank] = meshOverlap(overlapCells[subdomainNo],
                                                globalComm.rank,
                                                rank, dim)
        else:
            OM = None

        if self.newComm is not None:
            OMnew = overlapManager(globalComm)
            for subdomainNo in recv_ranks:
                rank = self.oldSubdomainGlobalRank_c(subdomainNo)
                OMnew.overlaps[rank] = meshOverlap(np.arange(recvDispl[rank]//numVerticesPerCell,
                                                             (recvDispl[rank]+recvCount[rank])//numVerticesPerCell, dtype=INDEX),
                                                   globalComm.rank, rank, dim)
        else:
            OMnew = None

        ######################################################################
        # build mesh interfaces between new partitions

        ######################################################################
        # send out all interface information from old partition
        if self.oldComm is not None:

            lIM = localInterfaceManager(subdomain, self.interfaces, self.oldComm,
                                        self.part, self.cell_offset)
            for subdomainNo in newSubdomainNos:
                lIM.addSubdomain(subdomainNo)
            lIM.removeBoundary()
            packed_send_vertices, packed_send_edges, packed_send_faces = lIM.getPackedDataForSend()

            if subdomain.dim == 3:
                for subdomainNo in newSubdomainNos:
                    rank = self.newSubdomainGlobalRank_c(subdomainNo)
                    try:
                        sendRequests.append(globalComm.isend(packed_send_faces[subdomainNo].shape[0], dest=rank, tag=53))
                        sendRequests.append(globalComm.Isend(packed_send_faces[subdomainNo], dest=rank, tag=54))
                    except KeyError:
                        sendRequests.append(globalComm.isend(0, dest=rank, tag=53))

                    try:
                        n = packed_send_edges[subdomainNo].shape[0]
                        sendRequests.append(globalComm.isend(n, dest=rank, tag=55))
                        sendRequests.append(globalComm.Isend(packed_send_edges[subdomainNo], dest=rank, tag=56))
                    except KeyError:
                        sendRequests.append(globalComm.isend(0, dest=rank, tag=55))

                    try:
                        n = packed_send_vertices[subdomainNo].shape[0]
                        sendRequests.append(globalComm.isend(n, dest=rank, tag=57))
                        sendRequests.append(globalComm.Isend(packed_send_vertices[subdomainNo], dest=rank, tag=58))
                    except KeyError:
                        sendRequests.append(globalComm.isend(0, dest=rank, tag=57))
            elif subdomain.dim == 2:
                for subdomainNo in newSubdomainNos:
                    rank = self.newSubdomainGlobalRank_c(subdomainNo)
                    try:
                        sendRequests.append(globalComm.isend(packed_send_edges[subdomainNo].shape[0], dest=rank, tag=53))
                        sendRequests.append(globalComm.Isend(packed_send_edges[subdomainNo], dest=rank, tag=54))
                    except KeyError:
                        sendRequests.append(globalComm.isend(0, dest=rank, tag=53))
                    try:
                        n = packed_send_vertices[subdomainNo].shape[0]
                        sendRequests.append(globalComm.isend(n, dest=rank, tag=57))
                        sendRequests.append(globalComm.Isend(packed_send_vertices[subdomainNo], dest=rank, tag=58))
                    except KeyError:
                        sendRequests.append(globalComm.isend(0, dest=rank, tag=57))
            elif subdomain.dim == 1:
                for subdomainNo in newSubdomainNos:
                    rank = self.newSubdomainGlobalRank_c(subdomainNo)
                    try:
                        n = packed_send_vertices[subdomainNo].shape[0]
                        sendRequests.append(globalComm.isend(n, dest=rank, tag=57))
                        sendRequests.append(globalComm.Isend(packed_send_vertices[subdomainNo], dest=rank, tag=58))
                    except KeyError:
                        sendRequests.append(globalComm.isend(0, dest=rank, tag=57))
            else:
                raise NotImplementedError()

        ######################################################################
        # recv all interface information from old partition
        if self.newComm is not None:
            iP = interfaceProcessor(subdomainNew, self.newComm, localToGlobal, globalToLocalCells)

            if dim == 1:
                for subdomainNo in recv_ranks:
                    rank = self.oldSubdomainGlobalRank_c(subdomainNo)
                    n = globalComm.recv(source=rank, tag=57)
                    packed_recv_vertices = uninitialized((n, 3), dtype=INDEX)
                    if n > 0:
                        globalComm.Recv(packed_recv_vertices, source=rank, tag=58)
                    packed_recv_faces = uninitialized((0, 3), dtype=INDEX)
                    packed_recv_edges = uninitialized((0, 3), dtype=INDEX)

                    iP.processInterfaceInformation(packed_recv_faces, packed_recv_edges, packed_recv_vertices)
            elif dim == 2:
                for subdomainNo in recv_ranks:
                    rank = self.oldSubdomainGlobalRank_c(subdomainNo)
                    n = globalComm.recv(source=rank, tag=53)
                    packed_recv_edges = uninitialized((n, 3), dtype=INDEX)
                    if n > 0:
                        globalComm.Recv(packed_recv_edges, source=rank, tag=54)
                    n = globalComm.recv(source=rank, tag=57)
                    packed_recv_vertices = uninitialized((n, 3), dtype=INDEX)
                    if n > 0:
                        globalComm.Recv(packed_recv_vertices, source=rank, tag=58)
                    packed_recv_faces = uninitialized((0, 3), dtype=INDEX)

                    iP.processInterfaceInformation(packed_recv_faces, packed_recv_edges, packed_recv_vertices)
            elif dim == 3:
                for subdomainNo in recv_ranks:
                    rank = self.oldSubdomainGlobalRank_c(subdomainNo)
                    n = globalComm.recv(source=rank, tag=53)
                    packed_recv_faces = uninitialized((n, 3), dtype=INDEX)
                    if n > 0:
                        globalComm.Recv(packed_recv_faces, source=rank, tag=54)
                    n = globalComm.recv(source=rank, tag=55)
                    packed_recv_edges = uninitialized((n, 3), dtype=INDEX)
                    if n > 0:
                        globalComm.Recv(packed_recv_edges, source=rank, tag=56)
                    n = globalComm.recv(source=rank, tag=57)
                    packed_recv_vertices = uninitialized((n, 3), dtype=INDEX)
                    if n > 0:
                        globalComm.Recv(packed_recv_vertices, source=rank, tag=58)

                    iP.processInterfaceInformation(packed_recv_faces, packed_recv_edges, packed_recv_vertices)

            iP.setBoundaryInformation()
            iM = iP.getInterfaceManager()
        else:
            iM = None

        if self.oldComm is not None:
            MPI.Request.Waitall(sendRequests)

        return subdomainNew, OM, OMnew, iM


cdef class localInterfaceManager:
    cdef:
        meshBase mesh
        interfaceManager interfaces
        MPI.Comm oldComm
        INDEX_t[::1] part
        INDEX_t cell_offset
        simplexMapper sM
        dict faceLookup
        dict edgeLookup
        dict vertexLookup

    def __init__(self, meshBase mesh, interfaceManager interfaces=None, MPI.Comm comm=None, INDEX_t[::1] part=None, INDEX_t cell_offset=0):
        cdef:
            dict faceLookup, edgeLookup, vertexLookup
            INDEX_t interfaceVertexNo, interfaceEdgeNo, interfaceFaceNo, cellNo, vertexNo, edgeNo, faceNo, subdomainNo, otherSubdomainNo, vertex
            ENCODE_t hv = 0
            INDEX_t edge[2]
            INDEX_t face[3]
            tuple hvF
        self.mesh = mesh
        self.interfaces = interfaces
        self.oldComm = comm
        self.part = part
        self.cell_offset = cell_offset

        self.sM = self.mesh.simplexMapper

        self.faceLookup = {}
        self.edgeLookup = {}
        self.vertexLookup = {}
        if self.interfaces is not None:
            faceLookup = self.faceLookup
            edgeLookup = self.edgeLookup
            vertexLookup = self.vertexLookup

            # get new partitions for interface edges
            interfacePart = self.interfaces.exchangePartitioning(self.oldComm, self.part)
            # enter all interface faces in faceLookup
            for subdomainNo in self.interfaces.interfaces:
                # enter information for all previous interface vertices in
                # vertexLookup
                for interfaceVertexNo in range(self.interfaces.interfaces[subdomainNo].vertices.shape[0]):
                    cellNo = self.interfaces.interfaces[subdomainNo].vertices[interfaceVertexNo, 0]
                    vertexNo = self.interfaces.interfaces[subdomainNo].vertices[interfaceVertexNo, 1]
                    otherSubdomainNo = interfacePart[subdomainNo]['vertex'][interfaceVertexNo]
                    vertex = self.sM.getVertexInCell(cellNo, vertexNo)
                    try:
                        vertexLookup[vertex][otherSubdomainNo] = -1
                    except KeyError:
                        vertexLookup[vertex] = {otherSubdomainNo: -1}

                # enter information for all previous interface edges in
                # edgeLookup and vertexLookup
                for interfaceEdgeNo in range(self.interfaces.interfaces[subdomainNo].edges.shape[0]):
                    cellNo = self.interfaces.interfaces[subdomainNo].edges[interfaceEdgeNo, 0]
                    edgeNo = self.interfaces.interfaces[subdomainNo].edges[interfaceEdgeNo, 1]
                    otherSubdomainNo = interfacePart[subdomainNo]['edge'][interfaceEdgeNo]
                    self.sM.getEdgeInCell(cellNo, edgeNo, edge)
                    hv = self.sM.getEdgeInCellEncoded(cellNo, edgeNo)
                    try:
                        edgeLookup[hv][otherSubdomainNo] = -1
                    except KeyError:
                        edgeLookup[hv] = {otherSubdomainNo: -1}

                    try:
                        vertexLookup[edge[0]][otherSubdomainNo] = -1
                    except KeyError:
                        vertexLookup[edge[0]] = {otherSubdomainNo: -1}
                    try:
                        vertexLookup[edge[1]][otherSubdomainNo] = -1
                    except KeyError:
                        vertexLookup[edge[1]] = {otherSubdomainNo: -1}

                # enter information for all previous interface faces in
                # faceLookup, edgeLookup and vertexLookup
                for interfaceFaceNo in range(self.interfaces.interfaces[subdomainNo].faces.shape[0]):
                    cellNo = self.interfaces.interfaces[subdomainNo].faces[interfaceFaceNo, 0]
                    faceNo = self.interfaces.interfaces[subdomainNo].faces[interfaceFaceNo, 1]
                    otherSubdomainNo = interfacePart[subdomainNo]['face'][interfaceFaceNo]
                    self.sM.getFaceInCell(cellNo, faceNo, face)
                    hvF = self.sM.sortAndEncodeFace(face)

                    faceLookup[hvF] = {otherSubdomainNo: -1}

                    self.sM.startLoopOverFaceEdges(face)
                    while self.sM.loopOverFaceEdgesEncoded(&hv):
                        try:
                            edgeLookup[hv][otherSubdomainNo] = -1
                        except KeyError:
                            edgeLookup[hv] = {otherSubdomainNo: -1}

                    try:
                        vertexLookup[face[0]][otherSubdomainNo] = -1
                    except KeyError:
                        vertexLookup[face[0]] = {otherSubdomainNo: -1}
                    try:
                        vertexLookup[face[1]][otherSubdomainNo] = -1
                    except KeyError:
                        vertexLookup[face[1]] = {otherSubdomainNo: -1}
                    try:
                        vertexLookup[face[2]][otherSubdomainNo] = -1
                    except KeyError:
                        vertexLookup[face[2]] = {otherSubdomainNo: -1}

    cdef void addSubdomain(self, INDEX_t subdomainNo):
        cdef:
            dict faceLookup, edgeLookup, vertexLookup
            INDEX_t i, cellNo, faceNo, edgeNo, vertexNo
            INDEX_t face[3]
            INDEX_t edge[2]
            INDEX_t vertex
            ENCODE_t hv
            tuple hvF
            meshBase fakeSubdomain
            INDEX_t[:, ::1] bvertices, bedges, bfaces
            INDEX_t[::1] fakeToLocal
        faceLookup = self.faceLookup
        edgeLookup = self.edgeLookup
        vertexLookup = self.vertexLookup

        fakeCells = self.mesh.cells_as_array[np.array(self.part) == subdomainNo, :].copy()
        if self.mesh.dim == 1:
            fakeSubdomain = mesh1d(self.mesh.vertices, fakeCells)
            bvertices = boundary1D(fakeSubdomain)
        elif self.mesh.dim == 2:
            fakeSubdomain = mesh2d(self.mesh.vertices, fakeCells)
            bvertices, bedges = boundary2D(fakeSubdomain, assumeConnected=False)
        elif self.mesh.dim == 3:
            fakeSubdomain = mesh3d(self.mesh.vertices, fakeCells)
            bvertices, bedges, bfaces = boundary3D(fakeSubdomain, assumeConnected=False)
        else:
            raise NotImplementedError()

        fakeToLocal = np.arange((self.mesh.num_cells), dtype=INDEX)[np.array(self.part) == subdomainNo]
        if self.mesh.dim == 3:
            for i in range(bfaces.shape[0]):
                cellNo, faceNo = bfaces[i, 0], bfaces[i, 1]
                cellNo = fakeToLocal[cellNo]
                self.sM.getFaceInCell(cellNo, faceNo, face)
                hvF = self.sM.sortAndEncodeFace(face)

                try:
                    faceLookup[hvF][subdomainNo] = cellNo
                except KeyError:
                    faceLookup[hvF] = {subdomainNo: cellNo}

        if self.mesh.dim >= 2:
            for i in range(bedges.shape[0]):
                cellNo, edgeNo = bedges[i, 0], bedges[i, 1]
                cellNo = fakeToLocal[cellNo]
                self.sM.getEdgeInCell(cellNo, edgeNo, edge)
                hv = self.sM.sortAndEncodeEdge(edge)

                try:
                    edgeLookup[hv][subdomainNo] = cellNo
                except KeyError:
                    edgeLookup[hv] = {subdomainNo: cellNo}

        for i in range(bvertices.shape[0]):
            cellNo, vertexNo = bvertices[i, 0], bvertices[i, 1]
            cellNo = fakeToLocal[cellNo]
            vertex = self.sM.getVertexInCell(cellNo, vertexNo)

            try:
                vertexLookup[vertex][subdomainNo] = cellNo
            except KeyError:
                vertexLookup[vertex] = {subdomainNo: cellNo}

    cdef void removeBoundary(self, TAG_t tag=PHYSICAL):
        cdef:
            INDEX_t[:, ::1] bfaces, bedges
            INDEX_t[::1] bvertices
            INDEX_t i, v, localVertexNo
            ENCODE_t hv
            tuple hvF
            dict temp
        if self.mesh.dim == 3:
            bfaces = self.mesh.getBoundaryFacesByTag(tag)
            for i in range(bfaces.shape[0]):
                hvF = self.sM.sortAndEncodeFace(bfaces[i, :])
                self.faceLookup.pop(hvF)
        if self.mesh.dim >= 2:
            bedges = self.mesh.getBoundaryEdgesByTag(tag)
            for i in range(bedges.shape[0]):
                hv = self.sM.sortAndEncodeEdge(bedges[i, :])
                self.edgeLookup.pop(hv)
        bvertices = self.mesh.getBoundaryVerticesByTag(tag)
        for v in bvertices:
            self.vertexLookup.pop(v)

        if self.mesh.dim == 3:
            # kick out all faces that are shared by 1 subdomain
            temp = {}
            for hvF in self.faceLookup:
                if len(self.faceLookup[hvF]) > 1:
                    temp[hvF] = self.faceLookup[hvF]
            self.faceLookup = temp
            # kick out all edges that are shared by 3 subdomains or fewer
            temp = {}
            for hv in self.edgeLookup:
                if len(self.edgeLookup[hv]) > 3:
                    temp[hv] = self.edgeLookup[hv]
            self.edgeLookup = temp
            # kick out all vertices that are shared by 4 subdomains or fewer
            temp = {}
            for localVertexNo in self.vertexLookup:
                if len(self.vertexLookup[localVertexNo]) > 4:
                    temp[localVertexNo] = self.vertexLookup[localVertexNo]
            self.vertexLookup = temp
        elif self.mesh.dim == 2:
            # kick out all edges that are shared by 1 subdomains or fewer
            temp = {}
            for hv in self.edgeLookup:
                if len(self.edgeLookup[hv]) > 1:
                    temp[hv] = self.edgeLookup[hv]
            self.edgeLookup = temp
            # kick out all vertices that are shared by 3 subdomains or fewer
            temp = {}
            for localVertexNo in self.vertexLookup:
                if len(self.vertexLookup[localVertexNo]) > 3:
                    temp[localVertexNo] = self.vertexLookup[localVertexNo]
            self.vertexLookup = temp

    cdef tuple getDataForSend(self):
        cdef:
            dict faceLookup, interface_faces
            dict edgeLookup, interface_edges
            dict vertexLookup, interface_vertices
            tuple hvF
            INDEX_t subdomainNo, globalCellNo, cellNo, faceNo, edgeNo, vertexNo
            INDEX_t face[3]
            INDEX_t edge[2]
            INDEX_t vertex
            tuple key, val

        # write to interface_faces:
        # (receiving subdomainNo) -> (face in local indices) -> sharing subdomains -> (globalCellNo, faceNo)
        faceLookup = self.faceLookup
        interface_faces = {}
        for hvF in faceLookup:
            for subdomainNo in faceLookup[hvF]:
                cellNo = faceLookup[hvF][subdomainNo]
                if cellNo == -1:
                    continue
                faceNo = self.sM.findFaceInCellEncoded(cellNo, hvF)

                for otherSubdomainNo in faceLookup[hvF]:
                    globalCellNo = self.cell_offset+cellNo
                    self.sM.getFaceInCell(cellNo, faceNo, face, sorted=True)
                    key = (face[0], face[1])
                    val = (globalCellNo, faceNo)
                    try:
                        interface_faces[subdomainNo]
                        try:
                            interface_faces[subdomainNo][key]
                            try:
                                interface_faces[subdomainNo][key][otherSubdomainNo].add(val)
                            except KeyError:
                                interface_faces[subdomainNo][key][otherSubdomainNo] = set([val])
                        except KeyError:
                            interface_faces[subdomainNo][key] = {otherSubdomainNo: set([val])}
                    except KeyError:
                        interface_faces[subdomainNo] = {key: {otherSubdomainNo: set([val])}}

        # write to interface_edges:
        # (receiving subdomainNo) -> (edge in local indices) -> sharing subdomains -> (globalCellNo, edgeNo)
        edgeLookup = self.edgeLookup
        interface_edges = {}
        for hv in edgeLookup:
            for subdomainNo in edgeLookup[hv]:
                cellNo = edgeLookup[hv][subdomainNo]
                if cellNo == -1:
                    continue
                edgeNo = self.sM.findEdgeInCellEncoded(cellNo, hv)
                for otherSubdomainNo in edgeLookup[hv]:
                    globalCellNo = self.cell_offset+cellNo
                    self.sM.getEdgeInCell(cellNo, edgeNo, edge, sorted=True)
                    key = (edge[0], edge[1])
                    val = (globalCellNo, edgeNo)
                    try:
                        interface_edges[subdomainNo]
                        try:
                            interface_edges[subdomainNo][key]
                            try:
                                interface_edges[subdomainNo][key][otherSubdomainNo].add(val)
                            except KeyError:
                                interface_edges[subdomainNo][key][otherSubdomainNo] = set([val])
                        except KeyError:
                            interface_edges[subdomainNo][key] = {otherSubdomainNo: set([val])}
                    except KeyError:
                        interface_edges[subdomainNo] = {key: {otherSubdomainNo: set([val])}}

        # write to interface_vertices:
        # (receiving subdomainNo) -> (vertex in local indices) -> sharing subdomains -> (globalCellNo, vertexNo)
        vertexLookup = self.vertexLookup
        interface_vertices = {}
        for localVertexNo in vertexLookup:
            for subdomainNo in vertexLookup[localVertexNo]:
                cellNo = vertexLookup[localVertexNo][subdomainNo]
                if cellNo == -1:
                    continue
                vertexNo = self.sM.findVertexInCell(cellNo, localVertexNo)
                for otherSubdomainNo in vertexLookup[localVertexNo]:
                    globalCellNo = self.cell_offset+cellNo
                    vertex = self.sM.getVertexInCell(cellNo, vertexNo)
                    val = (globalCellNo, vertexNo)
                    try:
                        interface_vertices[subdomainNo]
                        try:
                            interface_vertices[subdomainNo][vertex]
                            try:
                                interface_vertices[subdomainNo][vertex][otherSubdomainNo].add(val)
                            except KeyError:
                                interface_vertices[subdomainNo][vertex][otherSubdomainNo] = set([val])
                        except KeyError:
                            interface_vertices[subdomainNo][vertex] = {otherSubdomainNo: set([val])}
                    except KeyError:
                        interface_vertices[subdomainNo] = {vertex: {otherSubdomainNo: set([val])}}

        return interface_vertices, interface_edges, interface_faces

    cdef tuple getPackedDataForSend(self):
        cdef:
            dict interface_vertices, interface_edges, interface_faces
            dict packed_send_vertices, packed_send_edges, packed_send_faces
            INDEX_t subdomainNo, otherSubdomainNo, numFaces, numEdges, numVertices
            tuple face, edge
            INDEX_t globalCellNo, faceNo, edgeNo, vertexNo, k
            INDEX_t[:, ::1] psf, pse, psv
        interface_vertices, interface_edges, interface_faces = self.getDataForSend()

        ##################################################
        # send interface faces
        packed_send_faces = {}
        for subdomainNo in interface_faces:
            numFaces = 0
            for face in interface_faces[subdomainNo]:
                for otherSubdomainNo in interface_faces[subdomainNo][face]:
                    for globalCellNo, faceNo in interface_faces[subdomainNo][face][otherSubdomainNo]:
                        if subdomainNo == otherSubdomainNo:
                            continue
                        numFaces += 1
            psf = uninitialized((numFaces, 3), dtype=INDEX)
            k = 0
            for face in interface_faces[subdomainNo]:
                for otherSubdomainNo in interface_faces[subdomainNo][face]:
                    for globalCellNo, faceNo in interface_faces[subdomainNo][face][otherSubdomainNo]:
                        if subdomainNo == otherSubdomainNo:
                            continue
                        psf[k, 0] = globalCellNo
                        psf[k, 1] = faceNo
                        psf[k, 2] = otherSubdomainNo
                        k += 1
            assert k == numFaces
            packed_send_faces[subdomainNo] = psf

        ##################################################
        # send interface edges
        packed_send_edges = {}
        for subdomainNo in interface_edges:
            numEdges = 0
            for edge in interface_edges[subdomainNo]:
                for otherSubdomainNo in interface_edges[subdomainNo][edge]:
                    for globalCellNo, edgeNo in interface_edges[subdomainNo][edge][otherSubdomainNo]:
                        if subdomainNo == otherSubdomainNo:
                            continue
                        numEdges += 1
            pse = uninitialized((numEdges, 3), dtype=INDEX)
            k = 0
            for edge in interface_edges[subdomainNo]:
                for otherSubdomainNo in interface_edges[subdomainNo][edge]:
                    for globalCellNo, edgeNo in interface_edges[subdomainNo][edge][otherSubdomainNo]:
                        if subdomainNo == otherSubdomainNo:
                            continue
                        pse[k, 0] = globalCellNo
                        pse[k, 1] = edgeNo
                        pse[k, 2] = otherSubdomainNo
                        k += 1
            assert k == numEdges, (k, numEdges)
            packed_send_edges[subdomainNo] = pse

        ##################################################
        # send interface vertices
        packed_send_vertices = {}
        for subdomainNo in interface_vertices:
            numVertices = 0
            for vertex in interface_vertices[subdomainNo]:
                for otherSubdomainNo in interface_vertices[subdomainNo][vertex]:
                    for globalCellNo, vertexNo in interface_vertices[subdomainNo][vertex][otherSubdomainNo]:
                        if subdomainNo == otherSubdomainNo:
                            continue
                        numVertices += 1
            psv = uninitialized((numVertices, 3), dtype=INDEX)
            k = 0
            for vertex in interface_vertices[subdomainNo]:
                for otherSubdomainNo in interface_vertices[subdomainNo][vertex]:
                    for globalCellNo, vertexNo in interface_vertices[subdomainNo][vertex][otherSubdomainNo]:
                        if subdomainNo == otherSubdomainNo:
                            continue
                        psv[k, 0] = globalCellNo
                        psv[k, 1] = vertexNo
                        psv[k, 2] = otherSubdomainNo
                        k += 1
            assert k == numVertices
            packed_send_vertices[subdomainNo] = psv

        return packed_send_vertices, packed_send_edges, packed_send_faces


cdef class interfaceProcessor:
    cdef:
        meshBase mesh
        MPI.Comm newComm
        INDEX_t[::1] localToGlobal
        dict globalToLocalCells
        dict interface_vertices
        dict interface_edges
        dict interface_faces
        simplexMapper sMnew
        set face_candidates, edge_candidates, vertex_candidates

    def __init__(self, meshBase mesh, MPI.Comm comm, INDEX_t[::1] localToGlobal, dict globalToLocalCells):
        cdef:
            INDEX_t[:, ::1] bvertices, bedges, bfaces
            set face_candidates, edge_candidates, vertex_candidates
            INDEX_t i
            INDEX_t localCellNo, faceNo, edgeNo, vertexNo
        self.mesh = mesh
        self.newComm = comm
        self.localToGlobal = localToGlobal
        self.globalToLocalCells = globalToLocalCells

        self.interface_vertices = {}
        self.interface_edges = {}
        self.interface_faces = {}

        self.sMnew = self.mesh.simplexMapper
        if self.mesh.dim == 3:
            bvertices, bedges, bfaces = boundary3D(self.mesh)

            # face_candidates contains all boundary faces of the new subdomain
            face_candidates = set()
            for i in range(bfaces.shape[0]):
                localCellNo = bfaces[i, 0]
                faceNo = bfaces[i, 1]
                face_candidates.add((localCellNo, faceNo))
            self.face_candidates = face_candidates

            # edge_candidates contains all boundary edges of the new subdomain
            edge_candidates = set()
            for i in range(bedges.shape[0]):
                localCellNo = bedges[i, 0]
                edgeNo = bedges[i, 1]
                edge_candidates.add((localCellNo, edgeNo))
            self.edge_candidates = edge_candidates

            vertex_candidates = set()
            for i in range(bvertices.shape[0]):
                localCellNo = bvertices[i, 0]
                vertexNo = bvertices[i, 1]
                vertex_candidates.add((localCellNo, vertexNo))
            self.vertex_candidates = vertex_candidates
        elif self.mesh.dim == 2:
            bvertices, bedges = boundary2D(self.mesh)

            # edge_candidates contains all boundary edges of the new subdomain
            edge_candidates = set()
            for i in range(bedges.shape[0]):
                localCellNo = bedges[i, 0]
                edgeNo = bedges[i, 1]
                edge_candidates.add((localCellNo, edgeNo))
            self.edge_candidates = edge_candidates

            vertex_candidates = set()
            for i in range(bvertices.shape[0]):
                localCellNo = bvertices[i, 0]
                vertexNo = bvertices[i, 1]
                vertex_candidates.add((localCellNo, vertexNo))
            self.vertex_candidates = vertex_candidates
        elif self.mesh.dim == 1:
            bvertices = boundary1D(self.mesh)

            vertex_candidates = set()
            for i in range(bvertices.shape[0]):
                localCellNo = bvertices[i, 0]
                vertexNo = bvertices[i, 1]
                vertex_candidates.add((localCellNo, vertexNo))
            self.vertex_candidates = vertex_candidates
        else:
            raise NotImplementedError()

    cdef void processInterfaceInformation(self, INDEX_t[:, ::1] packed_recv_faces, INDEX_t[:, ::1] packed_recv_edges, INDEX_t[:, ::1] packed_recv_vertices):
        cdef:
            INDEX_t i
            INDEX_t globalCellNo, localCellNo, faceNo, edgeNo, vertexNo, otherSubdomainNo
        for i in range(packed_recv_faces.shape[0]):
            globalCellNo = packed_recv_faces[i, 0]
            localCellNo = self.globalToLocalCells[globalCellNo]
            faceNo = packed_recv_faces[i, 1]
            otherSubdomainNo = packed_recv_faces[i, 2]
            try:
                self.interface_faces[otherSubdomainNo].append((localCellNo, faceNo))
            except KeyError:
                self.interface_faces[otherSubdomainNo] = [(localCellNo, faceNo)]

        for i in range(packed_recv_edges.shape[0]):
            globalCellNo = packed_recv_edges[i, 0]
            localCellNo = self.globalToLocalCells[globalCellNo]
            edgeNo = packed_recv_edges[i, 1]
            otherSubdomainNo = packed_recv_edges[i, 2]
            try:
                self.interface_edges[otherSubdomainNo].append((localCellNo, edgeNo))
            except KeyError:
                self.interface_edges[otherSubdomainNo] = [(localCellNo, edgeNo)]

        for i in range(packed_recv_vertices.shape[0]):
            globalCellNo = packed_recv_vertices[i, 0]
            localCellNo = self.globalToLocalCells[globalCellNo]
            vertexNo = packed_recv_vertices[i, 1]
            otherSubdomainNo = packed_recv_vertices[i, 2]
            try:
                self.interface_vertices[otherSubdomainNo].append((localCellNo, vertexNo))
            except KeyError:
                self.interface_vertices[otherSubdomainNo] = [(localCellNo, vertexNo)]

    cdef void setBoundaryInformation(self):
        cdef:
            INDEX_t[:, ::1] bvertices, bedges, bfaces
            set face_candidates, edge_candidates, vertex_candidates
            INDEX_t i, vertex
            INDEX_t[::1] edge = np.empty((2), dtype=INDEX)
            INDEX_t[::1] face = np.empty((3), dtype=INDEX)
            TAG_t tag
            INDEX_t localCellNo, faceNo, edgeNo, vertexNo
            INDEX_t kFace, kEdge, kVertex
            ENCODE_t hv
            INDEX_t[:, ::1] subdomainBoundaryFaces, subdomainBoundaryEdges
            INDEX_t[::1] subdomainBoundaryVertices
            TAG_t[::1] subdomainBoundaryFaceTags, subdomainBoundaryEdgeTags, subdomainBoundaryVertexTags
            dict boundaryEdgeTagsDict, boundaryVertexTagsDict
        if self.mesh.dim == 3:
            bvertices, bedges, bfaces = boundary3D(self.mesh)

            # face_candidates contains all boundary faces of the new subdomain
            face_candidates = set()
            for i in range(bfaces.shape[0]):
                localCellNo = bfaces[i, 0]
                faceNo = bfaces[i, 1]
                face_candidates.add((localCellNo, faceNo))

            # edge_candidates contains all boundary edges of the new subdomain
            edge_candidates = set()
            for i in range(bedges.shape[0]):
                localCellNo = bedges[i, 0]
                edgeNo = bedges[i, 1]
                hv = self.sMnew.getEdgeInCellEncoded(localCellNo, edgeNo)
                edge_candidates.add(hv)

            vertex_candidates = set()
            for i in range(bvertices.shape[0]):
                localCellNo = bvertices[i, 0]
                vertexNo = bvertices[i, 1]
                vertex = self.sMnew.getVertexInCell(localCellNo, vertexNo)
                vertex_candidates.add(vertex)
        elif self.mesh.dim == 2:
            bvertices, bedges = boundary2D(self.mesh)
            bfaces = uninitialized((0, 3), dtype=INDEX)

            face_candidates = set()

            # edge_candidates contains all boundary edges of the new subdomain
            edge_candidates = set()
            for i in range(bedges.shape[0]):
                localCellNo = bedges[i, 0]
                edgeNo = bedges[i, 1]
                hv = self.sMnew.getEdgeInCellEncoded(localCellNo, edgeNo)
                edge_candidates.add(hv)

            vertex_candidates = set()
            for i in range(bvertices.shape[0]):
                localCellNo = bvertices[i, 0]
                vertexNo = bvertices[i, 1]
                vertex = self.sMnew.getVertexInCell(localCellNo, vertexNo)
                vertex_candidates.add(vertex)
        elif self.mesh.dim == 1:
            bvertices = boundary1D(self.mesh)
            bedges = uninitialized((0, 2), dtype=INDEX)
            bfaces = uninitialized((0, 3), dtype=INDEX)

            face_candidates = set()
            edge_candidates = set()

            vertex_candidates = set()
            for i in range(bvertices.shape[0]):
                localCellNo = bvertices[i, 0]
                vertexNo = bvertices[i, 1]
                vertex = self.sMnew.getVertexInCell(localCellNo, vertexNo)
                vertex_candidates.add(vertex)
        else:
            raise NotImplementedError()

        subdomainBoundaryFaces = uninitialized((bfaces.shape[0], 3), dtype=INDEX)
        subdomainBoundaryFaceTags = np.zeros((bfaces.shape[0]), dtype=TAG)

        subdomainBoundaryEdges = uninitialized((bedges.shape[0], 2), dtype=INDEX)
        subdomainBoundaryEdgeTags = np.zeros((bedges.shape[0]), dtype=TAG)

        subdomainBoundaryVertices = uninitialized((bvertices.shape[0]), dtype=INDEX)
        subdomainBoundaryVertexTags = np.zeros((bvertices.shape[0]), dtype=TAG)

        kFace = 0
        kEdge = 0
        kVertex = 0

        ##################################################
        # boundary face tags

        for subdomainNo in self.interface_faces:
            for localCellNo, faceNo in self.interface_faces[subdomainNo]:
                try:
                    face_candidates.remove((localCellNo, faceNo))
                    # set boundary face tag
                    self.sMnew.getFaceInCell(localCellNo, faceNo, face, sorted=True)
                    subdomainBoundaryFaces[kFace, 0] = face[0]
                    subdomainBoundaryFaces[kFace, 1] = face[1]
                    subdomainBoundaryFaces[kFace, 2] = face[2]
                    subdomainBoundaryFaceTags[kFace] = INTERIOR_NONOVERLAPPING
                    kFace += 1
                except KeyError:
                    pass
        for localCellNo, faceNo in face_candidates:
            self.sMnew.getFaceInCell(localCellNo, faceNo, face, sorted=True)
            subdomainBoundaryFaces[kFace, 0] = face[0]
            subdomainBoundaryFaces[kFace, 1] = face[1]
            subdomainBoundaryFaces[kFace, 2] = face[2]
            subdomainBoundaryFaceTags[kFace] = PHYSICAL
            kFace += 1

        ##################################################
        # boundary edge tags

        for subdomainNo in self.interface_edges:
            for localCellNo, edgeNo in self.interface_edges[subdomainNo]:
                try:
                    hv = self.sMnew.getEdgeInCellEncoded(localCellNo, edgeNo)
                    edge_candidates.remove(hv)
                    # set boundary edge tag
                    self.sMnew.getEdgeInCell(localCellNo, edgeNo, edge, sorted=True)
                    subdomainBoundaryEdges[kEdge, 0] = edge[0]
                    subdomainBoundaryEdges[kEdge, 1] = edge[1]
                    subdomainBoundaryEdgeTags[kEdge] = INTERIOR_NONOVERLAPPING
                    kEdge += 1
                except KeyError:
                    pass

        # propagate from boundary faces to boundary edges
        # we exploit that PHYSICAL faces are ordered last in boundaryFaceTags
        boundaryEdgeTagsDict = {}
        for faceNo in range(subdomainBoundaryFaces.shape[0]):
            tag = subdomainBoundaryFaceTags[faceNo]
            face = subdomainBoundaryFaces[faceNo, :]
            self.sMnew.startLoopOverFaceEdges(face)
            while self.sMnew.loopOverFaceEdgesEncoded(&hv):
                boundaryEdgeTagsDict[hv] = tag
        for hv in boundaryEdgeTagsDict:
            try:
                # set boundary edge tag
                edge_candidates.remove(hv)
                tag = boundaryEdgeTagsDict[hv]
                decode_edge(hv, edge)
                subdomainBoundaryEdges[kEdge, 0] = edge[0]
                subdomainBoundaryEdges[kEdge, 1] = edge[1]
                subdomainBoundaryEdgeTags[kEdge] = tag
                kEdge += 1
            except KeyError:
                pass

        for hv in edge_candidates:
            decode_edge(hv, edge)
            subdomainBoundaryEdges[kEdge, 0] = edge[0]
            subdomainBoundaryEdges[kEdge, 1] = edge[1]
            subdomainBoundaryEdgeTags[kEdge] = PHYSICAL
            kEdge += 1

        ##################################################
        # boundary vertex tags
        for subdomainNo in self.interface_vertices:
            for localCellNo, vertexNo in self.interface_vertices[subdomainNo]:
                try:
                    vertex = self.sMnew.getVertexInCell(localCellNo, vertexNo)
                    vertex_candidates.remove(vertex)
                    # set boundary vertex tag
                    subdomainBoundaryVertices[kVertex] = vertex
                    subdomainBoundaryVertexTags[kVertex] = INTERIOR_NONOVERLAPPING
                    kVertex += 1
                except KeyError:
                    pass

        # propagate from boundary edges to boundary vertices
        boundaryVertexTagsDict = {}
        if self.mesh.dim == 1:
            pass
        elif self.mesh.dim == 2:
            # we exploit that PHYSICAL edges are ordered last in boundaryEdgeTags
            for edgeNo in range(subdomainBoundaryEdges.shape[0]):
                tag = subdomainBoundaryEdgeTags[edgeNo]
                boundaryVertexTagsDict[subdomainBoundaryEdges[edgeNo, 0]] = tag
                boundaryVertexTagsDict[subdomainBoundaryEdges[edgeNo, 1]] = tag
        elif self.mesh.dim == 3:
            for edgeNo in range(subdomainBoundaryEdges.shape[0]):
                tag = subdomainBoundaryEdgeTags[edgeNo]
                try:
                    tagOld = boundaryVertexTagsDict[subdomainBoundaryEdges[edgeNo, 0]]
                    boundaryVertexTagsDict[subdomainBoundaryEdges[edgeNo, 0]] = max(tag, tagOld)
                except KeyError:
                    boundaryVertexTagsDict[subdomainBoundaryEdges[edgeNo, 0]] = tag

                try:
                    tagOld = boundaryVertexTagsDict[subdomainBoundaryEdges[edgeNo, 1]]
                    boundaryVertexTagsDict[subdomainBoundaryEdges[edgeNo, 1]] = max(tag, tagOld)
                except KeyError:
                    boundaryVertexTagsDict[subdomainBoundaryEdges[edgeNo, 1]] = tag
        else:
            raise NotImplementedError()
        for vertex in boundaryVertexTagsDict:
            try:
                # set boundary vertex tag
                vertex_candidates.remove(vertex)
                tag = boundaryVertexTagsDict[vertex]
                subdomainBoundaryVertices[kVertex] = vertex
                subdomainBoundaryVertexTags[kVertex] = tag
                kVertex += 1
            except KeyError:
                pass

        for vertex in vertex_candidates:
            subdomainBoundaryVertices[kVertex] = vertex
            subdomainBoundaryVertexTags[kVertex] = PHYSICAL
            kVertex += 1

        assert kVertex == bvertices.shape[0]

        self.mesh._boundaryFaces = np.array(subdomainBoundaryFaces, copy=False)
        self.mesh._boundaryFaceTags = np.array(subdomainBoundaryFaceTags, copy=False)
        self.mesh._boundaryEdges = np.array(subdomainBoundaryEdges, copy=False)
        self.mesh._boundaryEdgeTags = np.array(subdomainBoundaryEdgeTags, copy=False)
        self.mesh._boundaryVertices = np.array(subdomainBoundaryVertices, copy=False)
        self.mesh._boundaryVertexTags = np.array(subdomainBoundaryVertexTags, copy=False)

    cdef interfaceManager getInterfaceManager(self):
        cdef:
            INDEX_t dim, subdomainNo, i, k
            list interface_faces, interface_edges, interface_vertices
            INDEX_t numFaces, numEdges, numVertices
            INDEX_t face[3]
            INDEX_t faceGlobal[3]
            INDEX_t edge[2]
            INDEX_t edgeGlobal[2]
            INDEX_t vertex, vertexGlobal
            tuple hvF
            ENCODE_t hv
            INDEX_t[:, ::1] faces, edges, vertices
            INDEX_t cellNo, faceNo, order, edgeNo, vertexNo
            INDEX_t c0, c1, c2
            list sortKeyF
            ENCODE_t[::1] sortKeyE
            INDEX_t[::1] sortKeyV
            interfaceManager iM
        dim = self.mesh.dim
        iM = interfaceManager(self.newComm)
        for subdomainNo in range(self.newComm.size):
            if subdomainNo in self.interface_faces:
                interface_faces = self.interface_faces[subdomainNo]
                numFaces = len(interface_faces)
                sortKeyF = []
                for i in range(numFaces):
                    cellNo, faceNo = interface_faces[i]
                    self.sMnew.getFaceInCell(cellNo, faceNo, face)
                    faceGlobal[0] = self.localToGlobal[face[0]]
                    faceGlobal[1] = self.localToGlobal[face[1]]
                    faceGlobal[2] = self.localToGlobal[face[2]]
                    sortFace(faceGlobal[0], faceGlobal[1], faceGlobal[2], faceGlobal)
                    hvF = encode_face(faceGlobal)
                    sortKeyF.append(hvF)
                sortIdx = [f[0] for f in sorted(enumerate(sortKeyF),
                                                key=lambda x: x[1])]
                faces = uninitialized((numFaces, 3), dtype=INDEX)
                k = 0
                for i in sortIdx:
                    cellNo, faceNo = interface_faces[i]
                    faces[k, 0] = cellNo
                    faces[k, 1] = faceNo
                    self.sMnew.getFaceInCell(cellNo, faceNo, face)
                    c0, c1, c2 = self.localToGlobal[face[0]], self.localToGlobal[face[1]], self.localToGlobal[face[2]]
                    if c0 < c1:
                        if c0 < c2:
                            if c1 < c2:
                                order = 0
                            else:
                                order = -2
                        else:
                            order = 2
                    else:
                        if c1 < c2:
                            if c0 < c2:
                                order = -1
                            else:
                                order = 1
                        else:
                            order = -3
                    faces[k, 2] = order
                    k += 1
            else:
                faces = uninitialized((0, 3), dtype=INDEX)

            if subdomainNo in self.interface_edges:
                interface_edges = self.interface_edges[subdomainNo]
                numEdges = len(interface_edges)
                sortKeyE = uninitialized((numEdges), dtype=ENCODE)
                for i in range(numEdges):
                    cellNo, edgeNo = interface_edges[i]
                    self.sMnew.getEdgeInCell(cellNo, edgeNo, edge)
                    edgeGlobal[0] = self.localToGlobal[edge[0]]
                    edgeGlobal[1] = self.localToGlobal[edge[1]]
                    sortEdge(edgeGlobal[0], edgeGlobal[1], edgeGlobal)
                    hv = encode_edge(edgeGlobal)
                    sortKeyE[i] = hv
                sortIdx = np.argsort(sortKeyE)
                edges = uninitialized((numEdges, 3), dtype=INDEX)
                k = 0
                for i in sortIdx:
                    cellNo, edgeNo = interface_edges[i]
                    edges[k, 0] = cellNo
                    edges[k, 1] = edgeNo
                    self.sMnew.getEdgeInCell(cellNo, edgeNo, edge)
                    if self.localToGlobal[edge[0]] < self.localToGlobal[edge[1]]:
                        edges[k, 2] = 0
                    else:
                        edges[k, 2] = 1
                    k += 1
            else:
                edges = np.zeros((0, 3), dtype=INDEX)

            # process interface vertices
            # Sort each vertex by global vertex id.
            if subdomainNo in self.interface_vertices:
                interface_vertices = self.interface_vertices[subdomainNo]
                numVertices = len(interface_vertices)
                sortKeyV = uninitialized((numVertices), dtype=INDEX)
                for i in range(numVertices):
                    cellNo, vertexNo = interface_vertices[i]
                    vertex = self.sMnew.getVertexInCell(cellNo, vertexNo)
                    vertexGlobal = self.localToGlobal[vertex]
                    sortKeyV[i] = vertexGlobal
                sortIdx = np.argsort(sortKeyV)
                vertices = uninitialized((numVertices, 2), dtype=INDEX)
                k = 0
                for i in sortIdx:
                    cellNo, vertexNo = interface_vertices[i]
                    vertices[k, 0] = cellNo
                    vertices[k, 1] = vertexNo
                    k += 1
            else:
                vertices = uninitialized((0, 2), dtype=INDEX)

            if vertices.shape[0]+edges.shape[0]+faces.shape[0] > 0:
                iM.interfaces[subdomainNo] = meshInterface(vertices,
                                                           edges,
                                                           faces,
                                                           self.newComm.rank,
                                                           subdomainNo,
                                                           dim)
        return iM
