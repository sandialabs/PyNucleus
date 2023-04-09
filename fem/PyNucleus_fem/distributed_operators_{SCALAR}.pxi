###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI


cdef class {SCALAR_label}DistributedLinearOperator({SCALAR_label}LinearOperator):
    def __init__(self,
                 {SCALAR_label}LinearOperator A,
                 algebraicOverlapManager overlaps,
                 BOOL_t doDistribute=False,
                 BOOL_t keepDistributedResult=False):
        super({SCALAR_label}DistributedLinearOperator, self).__init__(A.num_rows, A.num_columns)
        self.A = A
        self.overlaps = overlaps
        self.doDistribute = doDistribute
        self.keepDistributedResult = keepDistributedResult
        self.allocateTempMemory(A.shape[0], A.shape[1])
        self.asynchronous = False

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void allocateTempMemory(self, INDEX_t sizeX, INDEX_t sizeY):
        if self.doDistribute:
            self.tempMemX = uninitialized((sizeX), dtype=REAL)
        if self.keepDistributedResult:
            self.tempMemY = uninitialized((sizeY), dtype=REAL)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void setTempMemory(self, {SCALAR}_t[::1] tempMemX, {SCALAR}_t[::1] tempMemY):
        if self.doDistribute:
            self.tempMemX = tempMemX
        if self.keepDistributedResult:
            self.tempMemY = tempMemY

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        cdef:
            {SCALAR}_t[::1] z, w
        if self.doDistribute:
            z = self.tempMemX
            self.overlaps.distribute{SCALAR_label}(x, z)
        else:
            z = x
        if self.keepDistributedResult:
            w = self.tempMemY
        else:
            w = y
        self.A.matvec(z, w)
        if self.keepDistributedResult:
            assign(y, w)
        self.overlaps.accumulate{SCALAR_label}(y, return_vec=None, asynchronous=self.asynchronous)
        return 0

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void residual(self,
                       {SCALAR}_t[::1] x,
                       {SCALAR}_t[::1] rhs,
                       {SCALAR}_t[::1] resAcc,
                       BOOL_t simpleResidual=False):
        cdef:
            {SCALAR}_t[::1] z, w
        if self.doDistribute:
            z = self.tempMemX
            self.overlaps.distribute{SCALAR_label}(x, z)
        else:
            z = x
        if self.keepDistributedResult:
            w = self.tempMemY
        else:
            w = resAcc
        self.A.residual(z, rhs, w, simpleResidual=simpleResidual)
        if self.keepDistributedResult:
            assign(resAcc, w)
        self.overlaps.accumulate{SCALAR_label}(resAcc, return_vec=None, asynchronous=self.asynchronous)

    property diagonal:
        def __get__(self):
            d = self.A.diagonal
            self.overlaps.accumulate{SCALAR_label}(d, return_vec=None, asynchronous=False)
            return d


cdef class {SCALAR_label}CSR_DistributedLinearOperator({SCALAR_label}DistributedLinearOperator):
    def __init__(self,
                 {SCALAR_label}CSR_LinearOperator A,
                 algebraicOverlapManager overlaps,
                 BOOL_t doDistribute=False,
                 BOOL_t keepDistributedResult=False):
        super({SCALAR_label}CSR_DistributedLinearOperator, self).__init__(A, overlaps, doDistribute, keepDistributedResult)
        self.csrA = A
        self.overlap_indices = self.overlaps.get_shared_dofs()

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec(self,
                        {SCALAR}_t[::1] x,
                        {SCALAR}_t[::1] y) except -1:
        cdef:
            INDEX_t i, jj, j, k
            {SCALAR}_t temp
            {SCALAR}_t[::1] z, w
        if self.doDistribute:
            if self.tempMemory is None:
                self.allocateTempMemory(x.shape[0], y.shape[0])
            z = self.tempMem
            self.overlaps.distribute{SCALAR_label}(x, z)
        else:
            z = x

        if self.keepDistributedResult:
            if self.tempMemY is None:
                self.allocateTempMemory(x.shape[0], y.shape[0])
            w = self.tempMemY
        else:
            w = y

        for k in range(self.overlap_indices.shape[0]):
            i = self.overlap_indices[k]
            temp = 0.0
            for jj in range(self.csrA.indptr[i], self.csrA.indptr[i+1]):
                j = self.csrA.indices[jj]
                temp += self.csrA.data[jj]*z[j]
            w[i] = temp
        self.overlaps.send{SCALAR_label}(w, asynchronous=self.asynchronous)

        k = 0
        for i in range(self.num_rows):
            if self.overlap_indices[k] == i:
                k += 1
                continue
            temp = 0.0
            for jj in range(self.csrA.indptr[i], self.csrA.indptr[i+1]):
                j = self.csrA.indices[jj]
                temp += self.csrA.data[jj]*z[j]
            w[i] = temp

        if self.keepDistributedResult:
            assign(y, w)
        self.overlaps.receive{SCALAR_label}(y, asynchronous=self.asynchronous)
        return 0

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void residual(self,
                       {SCALAR}_t[::1] x,
                       {SCALAR}_t[::1] rhs,
                       {SCALAR}_t[::1] resAcc,
                       BOOL_t simpleResidual=False):
        cdef:
            INDEX_t i, jj, j, k
            {SCALAR}_t temp
            {SCALAR}_t[::1] z, w

        if self.doDistribute:
            z = self.tempMemX
            self.overlaps.distribute{SCALAR_label}(x, z)
        else:
            z = x

        if self.keepDistributedResult:
            w = self.tempMemY
        else:
            w = resAcc

        for k in range(self.overlap_indices.shape[0]):
            i = self.overlap_indices[k]
            temp = rhs[i]
            for jj in range(self.csrA.indptr[i], self.csrA.indptr[i+1]):
                j = self.csrA.indices[jj]
                temp -= self.csrA.data[jj]*z[j]
            w[i] = temp
        self.overlaps.send{SCALAR_label}(w, asynchronous=self.asynchronous)

        k = 0
        for i in range(self.num_rows):
            if self.overlap_indices[k] == i:
                k += 1
                continue
            temp = rhs[i]
            for jj in range(self.csrA.indptr[i], self.csrA.indptr[i+1]):
                j = self.csrA.indices[jj]
                temp -= self.csrA.data[jj]*z[j]
            w[i] = temp
        if self.keepDistributedResult:
            assign(resAcc, w)
        self.overlaps.receive{SCALAR_label}(resAcc, asynchronous=self.asynchronous)

    property diagonal:
        def __get__(self):
            d = self.csrA.diagonal
            self.overlaps.accumulate{SCALAR_label}(d, return_vec=None, asynchronous=False)
            return d


cdef class {SCALAR_label}RowDistributedOperator({SCALAR_label}LinearOperator):
    """
    Extracts the local parts of a matrix and creates a row-distributed operator.
    """
    def __init__(self, {SCALAR_label}CSR_LinearOperator localMat, MPI.Comm comm, DoFMap dm, DoFMap local_dm, CSR_LinearOperator lclR, CSR_LinearOperator lclP):
        super(RowDistributedOperator, self).__init__(local_dm.num_dofs, local_dm.num_dofs)
        self.localMat = localMat
        self.comm = comm
        self.dm = dm

        self.lcl_dm = local_dm
        self.lclR = lclR
        self.lclP = lclP

        self.setupNear()

    def __repr__(self):
        return '<Rank %d/%d, %s, %d local size>' % (self.comm.rank, self.comm.size, self.localMat, self.lcl_dm.num_dofs)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void setupNear(self):
        cdef:
            list remoteReceives_list
            INDEX_t commSize = self.comm.size, remoteRank
            INDEX_t local_dof, global_dof, k
            dict global_to_local
            INDEX_t[::1] indptr, indices
            {SCALAR}_t[::1] data
            INDEX_t[::1] new_indptr, new_indices
            {SCALAR}_t[::1] new_data
            INDEX_t jj, jjj, J
        dof_owner = (self.lclP*(self.comm.rank*self.lcl_dm.ones())).astype(INDEX)
        assert dof_owner.shape[0] == self.dm.num_dofs
        self.comm.Allreduce(MPI.IN_PLACE, dof_owner, op=MPI.MAX)
        indptr = self.localMat.indptr
        indices = self.localMat.indices
        data = self.localMat.data
        remoteReceives_list = [set() for p in range(commSize)]
        for dof in range(self.localMat.shape[0]):
            if dof_owner[dof] != self.comm.rank:
                continue
            for jj in range(indptr[dof], indptr[dof+1]):
                J = indices[jj]
                remoteRank = dof_owner[J]
                remoteReceives_list[remoteRank].add(J)

        counterReceives = np.array([len(remoteReceives_list[p]) for p in range(commSize)], dtype=INDEX)
        remoteReceives = np.concatenate([np.sort(list(remoteReceives_list[p])) for p in range(commSize)]).astype(INDEX)
        counterSends = np.zeros((commSize), dtype=INDEX)
        self.comm.Alltoall(counterReceives, counterSends)
        remoteSends = np.zeros(counterSends.sum(), dtype=INDEX)

        offsetsReceives = np.concatenate(([0], np.cumsum(counterReceives)[:commSize-1])).astype(INDEX)
        offsetsSends = np.concatenate(([0], np.cumsum(counterSends)[:commSize-1])).astype(INDEX)
        self.comm.Alltoallv([remoteReceives, (counterReceives, offsetsReceives)],
                            [remoteSends, (counterSends, offsetsSends)])

        self.near_offsetReceives = offsetsReceives
        self.near_offsetSends = offsetsSends
        self.near_remoteReceives = remoteReceives
        self.near_remoteSends = remoteSends
        self.near_counterReceives = counterReceives
        self.near_counterSends = counterSends
        self.near_dataReceives = uninitialized((np.sum(counterReceives)), dtype={SCALAR})
        self.near_dataSends = uninitialized((np.sum(counterSends)), dtype={SCALAR})

        global_to_local = {}
        for local_dof in range(self.lcl_dm.num_dofs):
            global_dof = self.lclR.indices[local_dof]
            global_to_local[global_dof] = local_dof
        for k in range(self.near_remoteSends.shape[0]):
            global_dof = self.near_remoteSends[k]
            local_dof = global_to_local[global_dof]
            self.near_remoteSends[k] = local_dof

        self.rowIdx = self.lclR.indices[:self.lcl_dm.num_dofs]
        self.colIdx = np.concatenate((self.rowIdx,
                                      self.near_remoteReceives))
        self.near_remoteReceives = np.arange(self.lcl_dm.num_dofs,
                                             self.lcl_dm.num_dofs+self.near_remoteReceives.shape[0],
                                             dtype=INDEX)

        for local_dof in range(self.lcl_dm.num_dofs,
                               self.lcl_dm.num_dofs+self.near_remoteReceives.shape[0]):
            global_dof = self.colIdx[local_dof]
            global_to_local[global_dof] = local_dof

        new_indptr = uninitialized((self.rowIdx.shape[0]+1), dtype=INDEX)
        new_indices = uninitialized((self.localMat.nnz), dtype=INDEX)
        new_data = uninitialized((self.localMat.nnz), dtype={SCALAR})
        new_indptr[0] = 0
        for local_dof in range(self.rowIdx.shape[0]):
            global_dof = self.rowIdx[local_dof]
            new_indptr[local_dof+1] = new_indptr[local_dof] + indptr[global_dof+1]-indptr[global_dof]
            jjj = new_indptr[local_dof]
            for jj in range(indptr[global_dof], indptr[global_dof+1]):
                J = indices[jj]
                new_indices[jjj] = global_to_local[J]
                new_data[jjj] = data[jj]
                jjj += 1
        self.localMat.indptr = new_indptr
        self.localMat.indices = new_indices
        self.localMat.data = new_data
        self.localMat.num_rows = self.rowIdx.shape[0]
        self.localMat.num_columns = self.colIdx.shape[0]

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void communicateNear(self, {SCALAR}_t[::1] src, {SCALAR}_t[::1] target):
        cdef:
            INDEX_t i
        # pack
        for i in range(self.near_remoteSends.shape[0]):
            self.near_dataSends[i] = src[self.near_remoteSends[i]]
        # comm
        self.comm.Alltoallv([self.near_dataSends, (self.near_counterSends, self.near_offsetSends)],
                            [self.near_dataReceives, (self.near_counterReceives, self.near_offsetReceives)])
        # unpack
        for i in range(self.near_remoteReceives.shape[0]):
            target[self.near_remoteReceives[i]] = self.near_dataReceives[i]

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec(self, {SCALAR}_t[::1] x, {SCALAR}_t[::1] y) except -1:
        cdef:
            {SCALAR_label}CSR_LinearOperator localMat = self.localMat
        # near field
        xTemp = uninitialized((localMat.shape[1]), dtype={SCALAR})
        xTemp[:localMat.shape[0]] = x
        self.communicateNear(x, xTemp)
        localMat(xTemp, y)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t matvec_no_overwrite(self, {SCALAR}_t[::1] x, {SCALAR}_t[::1] y) except -1:
        cdef:
            {SCALAR_label}CSR_LinearOperator localMat = self.localMat
        # near field
        xTemp = uninitialized((localMat.shape[1]), dtype={SCALAR})
        xTemp[:localMat.shape[0]] = x
        self.communicateNear(x, xTemp)
        localMat.matvec_no_overwrite(xTemp, y)


    property diagonal:
        def __get__(self):
            return self.localMat.diagonal
