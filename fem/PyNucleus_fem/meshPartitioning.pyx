###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
cimport numpy as np
from PyNucleus_base.myTypes import INDEX, REAL
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t
from PyNucleus_base import uninitialized
from PyNucleus_base.linear_operators cimport LinearOperator, CSR_LinearOperator, sparseGraph
from itertools import chain
from libc.math cimport floor
cimport cython

include "config.pxi"


class PartitionerException(Exception):
    pass


@cython.initializedcheck(False)
@cython.wraparound(False)
def partition2sparseGraph(const INDEX_t[::1] partition,
                          const INDEX_t numPartitions):
    cdef:
        INDEX_t numVertices = partition.shape[0]
        np.ndarray[INDEX_t, ndim=1] indptr_mem = np.zeros((numPartitions+1),
                                                          dtype=INDEX)
        np.ndarray[INDEX_t, ndim=1] indices_mem = uninitialized((numVertices),
                                                           dtype=INDEX)
        INDEX_t[::1] indptr = indptr_mem, indices = indices_mem
        INDEX_t i, pos
    for i in range(numVertices):
        indptr[partition[i]+1] += 1
    for i in range(1, numPartitions+1):
        indptr[i] += indptr[i-1]
    for i in range(numVertices):
        pos = partition[i]
        indices[indptr[pos]] = i
        indptr[pos] += 1
    for i in range(numPartitions, 0, -1):
        indptr[i] = indptr[i-1]
    indptr[0] = 0
    return sparseGraph(indices, indptr, numPartitions, numVertices)


class vertexPartitioner(object):
    def __init__(self, REAL_t[:, ::1] vertices):
        self.vertices = vertices

    def partitionVertices(self, INDEX_t numPartitions):
        """
        Split the vertices into numPartitions partitions.
        Return a vector that contains the map vertexNo -> partitionNo.
        """
        raise PartitionerException("Don't call abstract class.")

    def inversePartitionVertices(self, numPartitions):
        """
        Split the vertices into numPartitions partitions.
        Return a sparse graph that contains the map partitionNo -> [vertexNo]
        """
        part, numPartitions = self.partitionVertices(numPartitions)
        return partition2sparseGraph(part, numPartitions)


class dofPartitioner(object):
    def __init__(self, LinearOperator A=None, dm=None):
        if A is not None:
            self.A = A
        elif dm is not None:
            self.dm = dm
            self.A = dm.buildSparsityPattern(dm.mesh.cells)
        else:
            raise NotImplementedError()

    def partitionDofs(self, numPartitions):
        raise PartitionerException("Don't call abstract class.")

    def inversePartitionDofs(self, numPartitions):
        """
        Split the DoFs into numPartitions partitions.
        Return a sparse graph that contains the map partitionNo -> [dofNo]
        """
        part, numPartitions = self.partitionDofs(numPartitions)
        return partition2sparseGraph(part, numPartitions)


class meshPartitioner(object):
    def __init__(self, mesh):
        self.mesh = mesh

    def partitionVertices(self, numPartitions):
        """
        Split the vertices into numPartitions partitions.
        Return a vector that contains the map vertexNo -> partitionNo.
        """
        raise PartitionerException("Don't call abstract class.")

    def inversePartitionVertices(self, numPartitions):
        """
        Split the vertices into numPartitions partitions.
        Return a sparse graph that contains the map partitionNo -> [vertexNo]
        """
        part, numPartitions = self.partitionVertices(numPartitions)
        return partition2sparseGraph(part, numPartitions)

    def partitionCells(self, numPartitions, partition_weights=None):
        """
        Split the cells into numPartitions partitions.
        If inverse is False, return a vector that contains the map cellNo -> partitionNo.
        If inverse is True, return a sparse graph that contains the map partitionNo -> [cellNo]
        """
        raise PartitionerException("Don't call abstract class.")

    def inversePartitionCells(self, numPartitions):
        """
        Split the cells into numPartitions partitions.
        Return a sparse graph that contains the map partitionNo -> [cellNo]
        """
        part, numPartitions = self.partitionCells(numPartitions)
        return partition2sparseGraph(part, numPartitions)


class regularVertexPartitioner(vertexPartitioner):
    def __init__(self, REAL_t[:, ::1] vertices, partitionedDimensions=None, numPartitionsPerDim=None):
        super(regularVertexPartitioner, self).__init__(vertices)
        self.partitionedDimensions = partitionedDimensions
        self.numPartitionsPerDim = numPartitionsPerDim
        self.mins = None
        self.maxs = None

    def getBoundingBox(self):
        cdef:
            REAL_t[::1] mins, maxs
            INDEX_t dim = self.vertices.shape[1]
            INDEX_t k, j
        mins = np.inf * np.ones((dim), dtype=REAL)
        maxs = -np.inf * np.ones((dim), dtype=REAL)
        for k in range(self.vertices.shape[0]):
            for j in range(dim):
                mins[j] = min(mins[j], self.vertices[k, j])
                maxs[j] = max(maxs[j], self.vertices[k, j])
        self.mins = mins
        self.maxs = maxs

    def balancePartitions(self, INDEX_t numPartitions):
        cdef:
            INDEX_t dim = self.vertices.shape[1]

        if self.partitionedDimensions is None:
            partitionedDimensions = dim
        else:
            partitionedDimensions = self.partitionedDimensions

        def primes(n):
            primfac = []
            d = 2
            while d*d <= n:
                while (n % d) == 0:
                    primfac.append(d)
                    n //= d
                d += 1
            if n > 1:
               primfac.append(n)
            return primfac

        numPartitionsPerDim = np.ones((dim), dtype=INDEX)
        self.getBoundingBox()
        extend = np.empty((dim), dtype=REAL)
        for j in range(dim):
            extend[j] = self.maxs[j]-self.mins[j]
        for p in sorted(primes(numPartitions), reverse=True):
            q = np.argmin((np.array(numPartitionsPerDim, copy=False)/extend)[:partitionedDimensions])
            numPartitionsPerDim[q] *= p
        return numPartitionsPerDim

    def partitionVertices(self, INDEX_t numPartitions, irregular=False):
        if irregular:
            return self.partitionVerticesIrregular(numPartitions)

        cdef:
            INDEX_t[::1] numPartitionsPerDim
            INDEX_t i, j, k
            INDEX_t dim = self.vertices.shape[1]
            REAL_t delta = 1e-5
            REAL_t w
            REAL_t[::1] z = uninitialized((dim), dtype=REAL)
            REAL_t[::1] mins, maxs
            INDEX_t[::1] part = uninitialized((self.vertices.shape[0]), dtype=INDEX)

        if self.numPartitionsPerDim is None:
            numPartitionsPerDim = self.balancePartitions(numPartitions)
        else:
            numPartitionsPerDim = self.numPartitionsPerDim

        numPartitionsTotal = np.prod(numPartitionsPerDim)
        if self.mins is None:
            self.getBoundingBox()
        mins = self.mins
        maxs = np.array(self.maxs, copy=True)
        for j in range(dim):
            maxs[j] += delta
        partitionCounter = np.zeros((numPartitionsTotal), dtype=INDEX)
        for i in range(self.vertices.shape[0]):
            w = 0
            for j in range(dim):
                z[j] = floor((self.vertices[i, j]-mins[j])/(maxs[j]-mins[j])*numPartitionsPerDim[j])
                for k in range(j):
                    z[j] *= numPartitionsPerDim[k]
                w += z[j]
            q = INDEX(w)
            part[i] = q
            partitionCounter[q] += 1
        if np.min(partitionCounter) == 0:
            raise PartitionerException('Regular partitioner returned empty partitions. PartitionCounter: {}'.format(np.array(partitionCounter)))
        numPartitions = np.unique(part).shape[0]
        return part, numPartitions

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def partitionDim(self, REAL_t[:, ::1] coord, INDEX_t[::1] part, INDEX_t[::1] numPartitionsPerDim, INDEX_t[::1] idx, INDEX_t d=0, INDEX_t offset=0):
        cdef:
            INDEX_t k, j, i
            REAL_t x
            INDEX_t dim = self.vertices.shape[1]
            REAL_t[::1] c = uninitialized((idx.shape[0]), dtype=REAL)
            REAL_t[::1] boundaries
            list idx_lists
            INDEX_t[::1] idx2
            INDEX_t numPart2
        for k in range(idx.shape[0]):
            c[k] = coord[idx[k], d]
        boundaries = np.quantile(c, np.linspace(0, 1, numPartitionsPerDim[d]+1)[1:numPartitionsPerDim[d]])

        if d == dim-1:
            for i in range(idx.shape[0]):
                k = idx[i]
                x = coord[k, d]
                for j in range(numPartitionsPerDim[d]-1):
                    if x < boundaries[j]:
                        part[k] = offset+j
                        break
                else:
                    part[k] = offset+numPartitionsPerDim[d]-1
            return numPartitionsPerDim[d]
        else:
            idx_lists = [[] for _ in range(numPartitionsPerDim[d])]
            for k in range(idx.shape[0]):
                x = c[k]
                for j in range(numPartitionsPerDim[d]-1):
                    if x < boundaries[j]:
                        idx_lists[j].append(idx[k])
                        break
                else:
                    idx_lists[numPartitionsPerDim[d]-1].append(idx[k])
            for j in range(numPartitionsPerDim[d]):
                idx2 = np.array(idx_lists[j], dtype=INDEX)
                numPart2 = self.partitionDim(coord, part, numPartitionsPerDim, idx2, d+1, offset)
                offset += numPart2
            return offset

    def partitionVerticesIrregular(self, INDEX_t numPartitions):
        cdef:
            INDEX_t[::1] numPartitionsPerDim
            INDEX_t num_vertices = self.vertices.shape[0]
        if self.numPartitionsPerDim is None:
            numPartitionsPerDim = self.balancePartitions(numPartitions)
        else:
            numPartitionsPerDim = self.numPartitionsPerDim

        part = uninitialized((num_vertices), dtype=INDEX)
        numPart = self.partitionDim(self.vertices, part, numPartitionsPerDim, np.arange(num_vertices, dtype=INDEX))
        return part, numPart


class regularMeshPartitioner(meshPartitioner):
    def partitionVertices(self, INDEX_t numPartitions, interiorOnly=True, partitionedDimensions=None, partition_weights=None, irregular=False):
        if numPartitions > self.mesh.num_vertices:
            raise PartitionerException("Cannot partition {} vertices in {} partitions.".format(self.mesh.num_vertices, numPartitions))
        if interiorOnly:
            vertices = self.mesh.vertices_as_array
            rVP = regularVertexPartitioner(vertices[self.mesh.interiorVertices],
                                           partitionedDimensions=partitionedDimensions,
                                           numPartitionsPerDim=partition_weights)
            part, numPartitions= rVP.partitionVertices(numPartitions, irregular=irregular)
        else:
            rVP = regularVertexPartitioner(self.mesh.vertices_as_array,
                                           partitionedDimensions=partitionedDimensions,
                                           numPartitionsPerDim=partition_weights)
            part, numPartitions= rVP.partitionVertices(numPartitions, irregular=irregular)
        return part, numPartitions

    def partitionCells(self, numPartitions, partitionedDimensions=None, partition_weights=None):
        if numPartitions > self.mesh.num_cells:
            raise PartitionerException("Cannot partition {} cells in {} partitions.".format(self.mesh.num_cells, numPartitions))
        rVP = regularVertexPartitioner(self.mesh.getProjectedCenters(),
                                       partitionedDimensions=partitionedDimensions,
                                       numPartitionsPerDim=partition_weights)
        part, numPartitions = rVP.partitionVertices(numPartitions)

        return part, numPartitions

    def __call__(self, numPartitions):
        return self.inversePartitionVertices(numPartitions)

    def __repr__(self):
        return 'Regular-Mesh'

class regularDofPartitioner(dofPartitioner):
    def partitionDofs(self, numPartitions, partitionedDimensions=None, partition_weights=None, irregular=False):
        assert self.dm is not None
        if numPartitions > self.dm.num_dofs:
            raise PartitionerException("Cannot partition {} DoFs in {} partitions.".format(self.dm.num_dofs, numPartitions))
        rVP = regularVertexPartitioner(self.dm.getDoFCoordinates(),
                                       partitionedDimensions=partitionedDimensions,
                                       numPartitionsPerDim=partition_weights)
        part, numPartitions= rVP.partitionVertices(numPartitions,
                                                   irregular=irregular)
        return part, numPartitions

    def __repr__(self):
        return 'Regular-DoF'

    def __call__(self, numPartitions):
        return self.inversePartitionDofs(numPartitions)



IF USE_METIS:
    import PyNucleus_metisCy

    class metisDofPartitioner(dofPartitioner):
        def partitionDofs(self, numPartitions, ufactor=30):
            if numPartitions == self.A.shape[0]:
                return np.arange(numPartitions, dtype=INDEX), numPartitions
            elif numPartitions > self.A.shape[0]:
                raise PartitionerException("Cannot partition {} DoFs in {} partitions.".format(self.A.shape[0], numPartitions))
            elif numPartitions == 1:
                return np.zeros((numPartitions), dtype=INDEX), numPartitions
            if isinstance(self.A, CSR_LinearOperator):
                A = self.A
            else:
                A = self.A.to_csr()
            options = PyNucleus_metisCy.SetDefaultOptions()
            options[PyNucleus_metisCy.OPTION_OBJTYPE] = PyNucleus_metisCy.OBJTYPE_VOL
            options[PyNucleus_metisCy.OPTION_CONTIG] = 1
            options[PyNucleus_metisCy.OPTION_UFACTOR] = ufactor
            partNos, numCuts = PyNucleus_metisCy.PartGraphKway(A.indptr,
                                                               A.indices,
                                                               numPartitions,
                                                               options=options)
            numPartitions = np.unique(partNos).shape[0]
            return np.array(partNos, dtype=INDEX), numPartitions

        def __repr__(self):
            return 'Metis-DoF'

        def __call__(self, numPartitions):
            return self.inversePartitionDofs(numPartitions)


    class metisMeshPartitioner(meshPartitioner):
        def partitionVertices(self, numPartitions, interiorOnly=True, ufactor=30):
            if numPartitions > self.mesh.num_vertices:
                raise PartitionerException("Cannot partition {} vertices in {} partitions.".format(self.mesh.num_vertices, numPartitions))
            if interiorOnly:
                raise NotImplementedError()
            options = PyNucleus_metisCy.SetDefaultOptions()
            options[PyNucleus_metisCy.OPTION_PTYPE] = PyNucleus_metisCy.PTYPE_KWAY
            options[PyNucleus_metisCy.OPTION_OBJTYPE] = PyNucleus_metisCy.OBJTYPE_VOL
            options[PyNucleus_metisCy.OPTION_CONTIG] = 1
            options[PyNucleus_metisCy.OPTION_UFACTOR] = ufactor
            # METIS requires cells as sparse graph
            cell_ptr = np.arange(0, (self.mesh.dim+1)*(self.mesh.num_cells+1),
                                 self.mesh.dim+1, dtype=INDEX)
            numCells = self.mesh.num_cells
            dim = self.mesh.dim
            numVertices = dim+1
            self.mesh.cells.resize((numVertices*numCells, ))
            cell_part, vertex_part, objval = PyNucleus_metisCy.PartMeshNodal(cell_ptr,
                                                                             self.mesh.cells,
                                                                             numPartitions,
                                                                             options=options)
            self.mesh.cells.resize((numCells, numVertices))
            numPartitions = np.unique(cell_part).shape[0]
            return vertex_part, numPartitions

        def partitionCells(self, numPartitions, inverse=False, ufactor=30, partition_weights=None):
            if numPartitions > self.mesh.num_cells:
                raise PartitionerException("Cannot partition {} cells in {} partitions.".format(self.mesh.num_cells, numPartitions))
            elif numPartitions == self.mesh.num_cells:
                cell_part = np.arange(numPartitions, dtype=INDEX)
            elif numPartitions == 1:
                cell_part = np.zeros(self.mesh.num_cells, dtype=INDEX)
            else:
                options = PyNucleus_metisCy.SetDefaultOptions()
                options[PyNucleus_metisCy.OPTION_PTYPE] = PyNucleus_metisCy.PTYPE_KWAY
                options[PyNucleus_metisCy.OPTION_OBJTYPE] = PyNucleus_metisCy.OBJTYPE_VOL
                options[PyNucleus_metisCy.OPTION_CONTIG] = 1
                options[PyNucleus_metisCy.OPTION_UFACTOR] = ufactor
                # METIS requires cells as sparse graph
                cell_ptr = np.arange(0, (self.mesh.dim+1)*(self.mesh.num_cells+1),
                                     self.mesh.dim+1, dtype=PyNucleus_metisCy.metisCy.idx)
                numCells = self.mesh.num_cells
                dim = self.mesh.dim
                numVertices = dim+1
                cells = self.mesh.cells_as_array
                cells.shape = (numVertices*numCells, )
                cell_part, vertex_part, objval = PyNucleus_metisCy.PartMeshDual(cell_ptr,
                                                                                cells.astype(PyNucleus_metisCy.metisCy.idx),
                                                                                2,
                                                                                numPartitions,
                                                                                tpwgts=partition_weights,
                                                                                options=options)
                numPartitions = np.unique(cell_part).shape[0]
            return cell_part, numPartitions

        def __repr__(self):
            return 'Metis-Mesh'
