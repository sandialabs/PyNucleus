###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from scipy.io import mmwrite
from . linear_operators cimport IJOperator, CSR_LinearOperator

from . myTypes import INDEX, REAL
from . linear_operators cimport LinearOperator

cimport cython


cdef class Map:
    def __init__(self, INDEX_t[:, ::1] GID_PID):
        assert GID_PID.shape[1] == 2
        assert np.array(GID_PID, copy=False)[:, 0].min() == 0
        self.GID_PID = GID_PID
        self.gbl_to_lcl = {}
        self.lcl_to_gbl = {}
        counter = {}
        for k in range(self.GID_PID.shape[0]):
            gid, pid = self.GID_PID[k, :]
            if pid not in self.lcl_to_gbl:
                self.lcl_to_gbl[pid] = {}
                self.gbl_to_lcl[pid] = {}
                counter[pid] = 0
            self.lcl_to_gbl[pid][counter[pid]] = gid
            self.gbl_to_lcl[pid][gid] = counter[pid]
            counter[pid] += 1

        self.localNumElements = counter
        self.lcl_to_gbl_offsets = np.zeros((self.numRanks()+1), dtype=INDEX)
        self.lcl_to_gbl_array = np.zeros((self.GID_PID.shape[0]), dtype=INDEX)
        for rank in range(self.numRanks()):
            size = len(self.lcl_to_gbl[rank])
            self.lcl_to_gbl_offsets[rank+1] = self.lcl_to_gbl_offsets[rank] + size
            lcl = np.array([self.lcl_to_gbl[rank][I] for I in range(size)], dtype=INDEX)
            for k in range(size):
                self.lcl_to_gbl_array[self.lcl_to_gbl_offsets[rank]+k] = lcl[k]


    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef INDEX_t getGlobalElement(self, INDEX_t pid, INDEX_t lid):
        cdef:
            INDEX_t offset1, offset2
        offset1 = self.lcl_to_gbl_offsets[pid]
        offset2 = self.lcl_to_gbl_offsets[pid+1]
        if lid < offset2-offset1:
            return self.lcl_to_gbl_array[offset1+lid]
        else:
            return -1

    cpdef INDEX_t getLocalElement(self, INDEX_t pid, INDEX_t gid):
        try:
            return self.gbl_to_lcl[pid][gid]
        except KeyError:
            return -1

    def getPIDs(self, INDEX_t gid):
        return self.GID_PID[np.where(np.array(self.GID_PID[:, 0], copy=False) == gid), 1]

    cpdef INDEX_t getLocalNumElements(self, INDEX_t pid):
        return self.localNumElements[pid]

    cpdef INDEX_t getGlobalNumElements(self):
        return np.array(self.GID_PID[:, 0], copy=False).max()+1

    def mmwrite(self, filename, binary=True):
        if binary:
            from struct import pack
            with open(filename, 'wb') as f:
                f.write(pack('NN', self.GID_PID.shape[0], self.GID_PID.shape[1]))
                f.write(np.array(self.GID_PID, copy=False).reshape((2*self.GID_PID.shape[0], 1)).astype(np.int64))
        else:
            with open(filename, 'wb') as f:
                mmwrite(f, np.array(self.GID_PID, copy=False).reshape((2*self.GID_PID.shape[0], 1)))

    cpdef INDEX_t numRanks(self):
        return np.array(self.GID_PID[:, 1], copy=False).max()+1


cdef class DistributedMap:
    def __init__(self, comm, INDEX_t[:,::1] GIDs):
        self.comm = comm
        assert GIDs.shape[1] == 2
        self.GIDs = GIDs
        self.GID2LID = {gid: lid for lid, gid in enumerate(GIDs[:, 0])}

    cpdef INDEX_t getGlobalElement(self, INDEX_t lid):
        return self.GIDs[lid, 0]

    cpdef INDEX_t getOwner(self, INDEX_t lid):
        return self.GIDs[lid, 1]

    cpdef BOOL_t isOwned(self, INDEX_t lid):
        return self.GIDs[lid, 1] == self.comm.rank

    cpdef INDEX_t getLocalElement(self, INDEX_t gid):
        try:
            return self.GID2LID[gid]
        except KeyError:
            return -1

    cpdef INDEX_t getLocalNumElements(self):
        return self.GIDs.shape[0]

    cpdef INDEX_t getGlobalNumElements(self):
        return self.comm.Allreduce(np.where(self.GIDs[:, 1] == self.comm.rank).sum())

    cpdef INDEX_t numRanks(self):
        return self.comm.size

    def gather(self):
        cdef:
            INDEX_t[::1] lclSize, gblSize
            INDEX_t p
        lclSize = np.zeros((1), dtype=INDEX)
        gblSize = np.zeros((self.comm.size), dtype=INDEX)
        GID_PID = np.array(self.GIDs, copy=True)
        GID_PID[:, 1] = self.comm.rank
        lclSize[0] = GID_PID.shape[0]
        self.comm.Gather(lclSize, gblSize)
        gblGID_PID = np.zeros((np.sum(gblSize), 2), dtype=INDEX)
        if self.comm.rank == 0:
            for p in range(self.comm.size):
                gblSize[p] *= 2
        self.comm.Gatherv(GID_PID, [gblGID_PID, (gblSize, None)])
        if self.comm.rank == 0:
            return Map(gblGID_PID)

    def mmwrite(self, filename, binary=True):
        m = self.gather()
        if self.comm.rank == 0:
            m.mmwrite(filename, binary)


cdef class Import:
    def __init__(self, DistributedMap ovMap, DistributedMap oneToOneMap):
        cdef:
            INDEX_t pid, gid, lid
            list receives_list
            INDEX_t[::1] receives, sends
        self.ovMap = ovMap
        self.oneToOneMap = oneToOneMap

        receives_list = [set() for pid in range(self.ovMap.numRanks())]
        for lid in range(self.ovMap.getLocalNumElements()):
            pid = self.ovMap.getOwner(lid)
            if pid != self.ovMap.comm.rank:
                gid = self.ovMap.getGlobalElement(lid)
                receives_list[pid].add(gid)
        self.countsReceive = np.array([len(receives_list[pid]) for pid in range(self.ovMap.numRanks())], dtype=INDEX)
        receives = np.concatenate([np.array(list(receives_list[pid]), dtype=INDEX) for pid in range(self.ovMap.numRanks())])
        self.countsSends = np.zeros_like(self.countsReceive)
        self.ovMap.comm.Alltoall(self.countsReceive, self.countsSends)
        sends = np.zeros((np.sum(self.countsSends)), dtype=INDEX)
        self.offsetsReceives = np.concatenate(([0], np.cumsum(self.countsReceive[:-1]))).astype(INDEX)
        self.offsetsSends = np.concatenate(([0], np.cumsum(self.countsSends[:-1]))).astype(INDEX)
        self.ovMap.comm.Alltoallv([receives, (self.countsReceive, self.offsetsReceives)],
                                  [sends, (self.countsSends, self.offsetsSends)])
        self.sendLIDs = np.array([self.oneToOneMap.getLocalElement(gid) for gid in sends], dtype=INDEX)
        self.receiveLIDs = np.array([self.ovMap.getLocalElement(gid) for gid in receives], dtype=INDEX)

    def __call__(self, x):
        cdef:
            INDEX_t i, lid
        assert x.shape[0] == self.oneToOneMap.getLocalNumElements()
        y = np.zeros((self.ovMap.getLocalNumElements()), dtype=np.array(x, copy=False).dtype)
        dataSend = np.zeros((self.sendLIDs.shape[0]), dtype=np.array(x, copy=False).dtype)
        for i in range(self.sendLIDs.shape[0]):
            lid = self.sendLIDs[i]
            dataSend[i] = x[lid]
        dataReceive = np.zeros((self.receiveLIDs.shape[0]), dtype=np.array(x, copy=False).dtype)
        self.ovMap.comm.Alltoallv([dataSend, (self.countsSends, self.offsetsSends)],
                                  [dataReceive, (self.countsReceive, self.offsetsReceives)])
        for lid in range(self.ovMap.getLocalNumElements()):
            if self.ovMap.isOwned(lid):
                y[lid] = x[lid]
        for i in range(self.receiveLIDs.shape[0]):
            lid = self.receiveLIDs[i]
            y[lid] = dataReceive[i]
        return y


@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.boundscheck(False)
def writeBinary(filename, CSR_LinearOperator mtx):
    cdef:
        INDEX_t i, start, end, rownnz
    from struct import pack
    with open(filename, 'wb') as f:
        f.write(pack('iii', mtx.shape[0], mtx.shape[1], mtx.nnz))
        for i in range(mtx.shape[0]):
            start = mtx.indptr[i]
            end = mtx.indptr[i+1]
            rownnz = end-start
            f.write(pack('ii', i, rownnz))
            f.write(mtx.indices[start:end])
            f.write(mtx.data[start:end])


def writeMatrix(fn, CSR_LinearOperator mat, BOOL_t binary=True):
    if binary:
        writeBinary(fn, mat)
    else:
        with open(fn, 'wb') as f:
            mmwrite(f, mat.to_csr(), symmetry='general')


@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.boundscheck(False)
def computeColMap(globalA, Map rowMap):
    cdef:
        INDEX_t pid, rlid, rgid, jj, cgid, k
        set toAdd
        INDEX_t[::1] globalA_indptr = globalA.indptr
        INDEX_t[::1] globalA_indices = globalA.indices
        list lcl_parts = []
        INDEX_t[:, ::1] lcl_part

    for pid in range(rowMap.numRanks()):
        toAdd = set()
        for rlid in range(rowMap.getLocalNumElements(pid)):
            rgid = rowMap.getGlobalElement(pid, rlid)
            for jj in range(globalA_indptr[rgid], globalA_indptr[rgid+1]):
                cgid = globalA_indices[jj]
                if rowMap.getLocalElement(pid, cgid) == -1:
                    toAdd.add(cgid)
        lcl_part = np.empty((len(toAdd), 2), dtype=INDEX)
        k = 0
        for cgid in toAdd:
            lcl_part[k, 0] = cgid
            lcl_part[k, 1] = pid
            k += 1
        lcl_parts.append(lcl_part)
    return Map(np.vstack((rowMap.GID_PID,
                          *lcl_parts)))


class DistMatrix:
    def __init__(self, lclMatrices, rowMap, colMap, domainMap=None):
        self.lclMatrices = lclMatrices
        self.rowMap = rowMap
        self.colMap = colMap
        if domainMap is None:
            self.domainMap = self.rowMap
        else:
            self.domainMap = domainMap

    def __str__(self):
        nnzs = np.array([self.lclMatrices[rank].nnz for rank in range(len(self.lclMatrices))])
        s = 'min/mean/med/max: {:,} / {:.4} / {:,} / {:,}'.format(np.min(nnzs),
                                                                  np.mean(nnzs),
                                                                  np.median(nnzs),
                                                                  np.max(nnzs))
        return s

    @staticmethod
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    def fromGlobalMatrix(CSR_LinearOperator globalA, Map rowMap):
        cdef:
            INDEX_t pid, rlid, rgid, jj, cgid, clid, k
            INDEX_t[::1] globalA_indptr = globalA.indptr
            INDEX_t[::1] globalA_indices = globalA.indices
            REAL_t[::1] globalA_data = globalA.data
            LinearOperator lclMatrix
            Map colMap
            list lclMatrices
            INDEX_t numLocalRows
            INDEX_t[::1] lcl_indptr, lcl_indices
            REAL_t[::1] lcl_data

        colMap = computeColMap(globalA, rowMap)
        lclMatrices = []
        for pid in range(rowMap.numRanks()):
            numLocalRows = rowMap.getLocalNumElements(pid)
            lcl_indptr = np.zeros((numLocalRows+1), dtype=INDEX)
            for rlid in range(rowMap.getLocalNumElements(pid)):
                rgid = rowMap.getGlobalElement(pid, rlid)
                lcl_indptr[rlid+1] = lcl_indptr[rlid] + (globalA_indptr[rgid+1]-globalA_indptr[rgid])
            lcl_indices = np.zeros((lcl_indptr[numLocalRows]), dtype=INDEX)
            lcl_data = np.zeros((lcl_indptr[numLocalRows]), dtype=REAL)
            for rlid in range(rowMap.getLocalNumElements(pid)):
                rgid = rowMap.getGlobalElement(pid, rlid)
                k = lcl_indptr[rlid]
                for jj in range(globalA_indptr[rgid], globalA_indptr[rgid+1]):
                    cgid = globalA_indices[jj]
                    clid = colMap.getLocalElement(pid, cgid)
                    lcl_indices[k] = clid
                    lcl_data[k] = globalA_data[jj]
                    k += 1
            lclMatrix = CSR_LinearOperator(lcl_indices, lcl_indptr, lcl_data)
            lclMatrix.num_columns = colMap.getLocalNumElements(pid)
            lclMatrices.append(lclMatrix)
        return DistMatrix(lclMatrices, rowMap, colMap)

    def mmwrite(self, filename, binary=False, local=False):
        if not local:
            gblMatrix = IJOperator(self.rowMap.getGlobalNumElements(),
                                   self.domainMap.getGlobalNumElements())
            for rank in range(len(self.lclMatrices)):
                for i in range(self.lclMatrices[rank].num_rows):
                    I = self.rowMap.getGlobalElement(rank, i)
                    assert I != -1
                    for jj in range(self.lclMatrices[rank].indptr[i],
                                    self.lclMatrices[rank].indptr[i+1]):
                        j = self.lclMatrices[rank].indices[jj]
                        J = self.colMap.getGlobalElement(rank, j)
                        assert J != -1
                        v = self.lclMatrices[rank].data[jj]
                        gblMatrix.setEntry_py(I, J, v)
            gblMatrix = gblMatrix.to_csr_linear_operator()
            if not binary:
                with open(filename, 'wb') as f:
                    mmwrite(f, gblMatrix.to_csr(), symmetry='general')
            else:
                writeBinary(filename, gblMatrix)
        else:
            for rank in range(len(self.lclMatrices)):
                fn = filename+'.'+str(len(self.lclMatrices))+'.'+str(rank)
                if binary:
                    writeBinary(fn, self.lclMatrices[rank])
                else:
                    with open(fn, 'wb') as f:
                        mmwrite(f, self.lclMatrices[rank].to_csr(), symmetry='general')


class DistVector:
    def __init__(self, gblVector, map):
        self.gblVector = gblVector
        if self.gblVector.ndim == 1:
            from PyNucleus_fem.DoFMaps import fe_vector
            if isinstance(self.gblVector, fe_vector):
                self.gblVector = self.gblVector.toarray()[:, np.newaxis]
            else:
                self.gblVector = self.gblVector[:, np.newaxis]
        self.map = map
        assert self.gblVector.shape[0] == self.map.getGlobalNumElements(), (self.gblVector.shape[0], self.map.getGlobalNumElements())

    def mmwrite(self, filename):
        reorderedVector = np.zeros_like(self.gblVector)
        j = 0
        for rank in range(self.map.numRanks()):
            for lid in range(self.map.getLocalNumElements(rank)):
                gid = self.map.getGlobalElement(rank, lid)
                for k in range(self.gblVector.shape[1]):
                    reorderedVector[j, k] = self.gblVector[gid, k]
                j += 1
        with open(filename, 'wb') as f:
            mmwrite(f, reorderedVector)
