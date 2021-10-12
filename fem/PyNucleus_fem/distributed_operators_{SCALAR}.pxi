###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################



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
