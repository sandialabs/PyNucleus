###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from libc.math cimport sqrt
from PyNucleus_base import uninitialized
import numpy as np
cimport numpy as np
cimport cython
# from libcpp.unordered_map cimport unordered_map
# from libcpp.map cimport map
from libc.stdlib cimport malloc, realloc, free
from libc.stdlib cimport qsort

from PyNucleus_base.myTypes import INDEX, REAL, ENCODE, TAG
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, ENCODE_t, TAG_t
from PyNucleus_base.blas cimport mydot
from PyNucleus_base.intTuple cimport productIterator
import warnings

cdef INDEX_t MAX_INT = np.iinfo(INDEX).max


cdef class meshTransformer:
    def __init__(self):
        pass

    def __call__(self, meshBase mesh, dict lookup):
        raise NotImplementedError()


cdef class radialMeshTransformer(meshTransformer):
    def __init__(self):
        super(radialMeshTransformer, self).__init__()

    def __call__(self, meshBase mesh, dict lookup):
        cdef:
            INDEX_t[::1] e = uninitialized((2), dtype=INDEX)
            ENCODE_t encodeVal
            INDEX_t vertexNo
            REAL_t r1, r2, r, r3
            INDEX_t dim = mesh.dim
            REAL_t[:, ::1] vertices = mesh.vertices
        for encodeVal in lookup:
            decode_edge(encodeVal, e)
            vertexNo = lookup[encodeVal]
            r1 = 0.
            for i in range(dim):
                r1 += vertices[e[0], i]**2
            r1 = sqrt(r1)
            r2 = 0.
            for i in range(dim):
                r2 += vertices[e[1], i]**2
            r2 = sqrt(r2)
            r = 0.5*r1 + 0.5*r2
            r3 = 0.
            for i in range(dim):
                r3 += vertices[vertexNo, i]**2
            r3 = sqrt(r3)
            for i in range(dim):
                mesh.vertices[vertexNo, i] *= r/r3


cdef class gradedMeshTransformer(meshTransformer):
    def __init__(self, REAL_t mu=2., mu2=None, REAL_t radius=1.):
        super(gradedMeshTransformer, self).__init__()
        self.mu = mu
        if mu2 is None:
            self.mu2 = mu
        else:
            self.mu2 = mu2
        self.radius = radius

    def __call__(self, meshBase mesh, dict lookup):
        cdef:
            INDEX_t[::1] e = uninitialized((2), dtype=INDEX)
            ENCODE_t encodeVal
            INDEX_t vertexNo
            REAL_t r1, r2, r, r3, x1, x2, x3
            INDEX_t dim = mesh.dim
            REAL_t[:, ::1] vertices = mesh.vertices
        for encodeVal in lookup:
            decode_edge(encodeVal, e)
            vertexNo = lookup[encodeVal]
            r1 = 0.
            for i in range(dim):
                r1 += vertices[e[0], i]**2
            r1 = sqrt(r1)
            r2 = 0.
            for i in range(dim):
                r2 += vertices[e[1], i]**2
            r2 = sqrt(r2)
            r = 0.5*r1 + 0.5*r2
            r3 = 0.
            for i in range(dim):
                r3 += vertices[vertexNo, i]**2
            r3 = sqrt(r3)
            if vertices[vertexNo, 0] < 0:
                x1 = 1-(1-r1/self.radius)**(1/self.mu)
                x2 = 1-(1-r2/self.radius)**(1/self.mu)
                x3 = 0.5*x1+0.5*x2
                r = self.radius*(1-(1-x3)**self.mu)
            else:
                x1 = 1-(1-r1/self.radius)**(1/self.mu2)
                x2 = 1-(1-r2/self.radius)**(1/self.mu2)
                x3 = 0.5*x1+0.5*x2
                r = self.radius*(1-(1-x3)**self.mu2)
            for i in range(dim):
                mesh.vertices[vertexNo, i] *= r/r3


cdef class gradedHypercubeTransformer(meshTransformer):
    cdef:
        REAL_t[::1] factor, invFactor

    def __init__(self, factor=0.4):
        cdef:
            INDEX_t i

        if isinstance(factor, float):
            assert 0 < factor
            self.factor = factor*np.ones((3), dtype=REAL)
        else:
            for i in range(factor.shape[0]):
                assert 0 < factor[i]
            self.factor = factor
        self.invFactor = uninitialized((self.factor.shape[0]), dtype=REAL)
        for i in range(self.factor.shape[0]):
            self.invFactor[i] = 1./self.factor[i]

    def __call__(self, meshBase mesh, dict lookup):
        cdef:
            INDEX_t i, j
            INDEX_t[::1] e = uninitialized((2), dtype=INDEX)
            REAL_t v0, v1
            REAL_t[::1] boundMin = np.inf*np.ones((mesh.dim), dtype=REAL)
            REAL_t[::1] boundMax = -np.inf*np.ones((mesh.dim), dtype=REAL)
            REAL_t[::1] boundMaxInv = uninitialized((mesh.dim), dtype=REAL)
        for j in range(mesh.num_vertices):
            for i in range(mesh.dim):
                v0 = mesh.vertices[j, i]
                boundMin[i] = min(boundMin[i], v0)
                boundMax[i] = max(boundMax[i], v0)
        for i in range(mesh.dim):
            boundMax[i] = 1./(boundMax[i]-boundMin[i])
            boundMaxInv[i] = boundMax[i]-boundMin[i]
        for encodeVal in lookup:
            decode_edge(encodeVal, e)
            j = lookup[encodeVal]
            for i in range(mesh.dim):
                v0 = (boundMax[i]*(mesh.vertices[e[0], i]-boundMin[i]))**self.invFactor[i]
                v1 = (boundMax[i]*(mesh.vertices[e[1], i]-boundMin[i]))**self.invFactor[i]
                mesh.vertices[j, i] = boundMin[i] + boundMaxInv[i]*(0.5*v0 + 0.5*v1)**self.factor[i]


cdef class multiIntervalMeshTransformer(meshTransformer):
    def __init__(self, list intervals):
        super(multiIntervalMeshTransformer, self).__init__()
        self.intervals = intervals

    def __call__(self, meshBase mesh, dict lookup):
        cdef:
            INDEX_t[::1] e = uninitialized((2), dtype=INDEX)
            ENCODE_t encodeVal
            INDEX_t vertexNo
            REAL_t r1, r2, r, x1, x2, x3, radius
            REAL_t[:, ::1] vertices = mesh.vertices
        for encodeVal in lookup:
            decode_edge(encodeVal, e)
            vertexNo = lookup[encodeVal]
            r = mesh.vertices[vertexNo, 0]
            for interval in self.intervals:
                a, b, mu1, mu2 = interval
                r = vertices[vertexNo, 0]
                if (a < r) and (r <= b):
                    if mu1 is None:
                        if mu2 is None:
                            raise NotImplementedError()
                        else:
                            center = a
                            radius = b-a
                            r1 = abs(vertices[e[0], 0]-center)
                            r2 = abs(vertices[e[1], 0]-center)
                            x1 = 1-(1-r1/radius)**(1/mu2)
                            x2 = 1-(1-r2/radius)**(1/mu2)
                            x3 = 0.5*x1+0.5*x2
                            r = center + radius*(1-(1-x3)**mu2)
                    else:
                        if mu2 is None:
                            center = b
                            radius = b-a
                            r1 = abs(vertices[e[0], 0]-center)
                            r2 = abs(vertices[e[1], 0]-center)
                            x1 = 1-(1-r1/radius)**(1/mu1)
                            x2 = 1-(1-r2/radius)**(1/mu1)
                            x3 = 0.5*x1+0.5*x2
                            r = center - radius*(1-(1-x3)**mu1)
                        else:
                            center = 0.5*(a+b)
                            radius = 0.5*(b-a)
                            r1 = abs(vertices[e[0], 0]-center)
                            r2 = abs(vertices[e[1], 0]-center)
                            if r < center:
                                x1 = 1-(1-r1/radius)**(1/mu1)
                                x2 = 1-(1-r2/radius)**(1/mu1)
                                x3 = 0.5*x1+0.5*x2
                                r = center - radius*(1-(1-x3)**mu1)
                            else:
                                x1 = 1-(1-r1/radius)**(1/mu2)
                                x2 = 1-(1-r2/radius)**(1/mu2)
                                x3 = 0.5*x1+0.5*x2
                                r = center + radius*(1-(1-x3)**mu2)
                    break
            mesh.vertices[vertexNo, 0] = r


def radialMeshTransformation(mesh, dict lookup):
    cdef:
        INDEX_t[::1] e = uninitialized((2), dtype=INDEX)
        ENCODE_t encodeVal
        INDEX_t vertexNo
        REAL_t r1, r2, r, r3
        INDEX_t dim = mesh.dim
        REAL_t[:, ::1] vertices = mesh.vertices
    warnings.warn('"radialMeshTransformation" deprecated, use "radialMeshTransformer"', DeprecationWarning)
    for encodeVal in lookup:
        decode_edge(encodeVal, e)
        vertexNo = lookup[encodeVal]
        r1 = 0.
        for i in range(dim):
            r1 += vertices[e[0], i]**2
        r1 = sqrt(r1)
        r2 = 0.
        for i in range(dim):
            r2 += vertices[e[1], i]**2
        r2 = sqrt(r2)
        r = 0.5*r1 + 0.5*r2
        r3 = 0.
        for i in range(dim):
            r3 += vertices[vertexNo, i]**2
        r3 = sqrt(r3)
        for i in range(dim):
            mesh.vertices[vertexNo, i] *= r/r3


cdef class meshBase:
    def __init__(self, vertices_t vertices, cells_t cells):
        self.vertices = vertices
        self.cells = cells
        self.init()

    def __getstate__(self):
        return (self.vertices_as_array, self.cells_as_array, self.transformer)

    def __setstate__(self, state):
        self.vertices = state[0]
        self.cells = state[1]
        self.init()
        if state[2] is not None:
            self.setMeshTransformation(state[2])

    def init(self):
        self.resetMeshInfo()
        self.num_vertices = self.vertices.shape[0]
        self.num_cells = self.cells.shape[0]
        self.dim = self.vertices.shape[1]
        self.manifold_dim = self.cells.shape[1]-1
        if self.dim == 1:
            self.simplexMapper = simplexMapper1D(self)
        elif self.dim == 2:
            self.simplexMapper = simplexMapper2D(self)
        elif self.dim == 3:
            self.simplexMapper = simplexMapper3D(self)
        else:
            raise NotImplementedError()
        self.meshTransformer = None

    @property
    def vertices_as_array(self):
        return np.array(self.vertices, copy=False)

    @property
    def cells_as_array(self):
        return np.array(self.cells, copy=False)

    @property
    def sizeInBytes(self):
        s = 0
        s += self.num_vertices*self.dim*sizeof(REAL_t)
        s += self.num_cells*(self.manifold_dim+1)*sizeof(INDEX_t)
        if self._volVector is not None:
            s += self.num_cells*sizeof(REAL_t)
        if self._hVector is not None:
            s += self.num_cells*sizeof(REAL_t)
        if self.dim >= 1:
            s += self.boundaryVertices.shape[0]*sizeof(INDEX_t)
            s += self.boundaryVertices.shape[0]*sizeof(TAG_t)
        if self.dim >= 2:
            s += 2*self.boundaryEdges.shape[0]*sizeof(INDEX_t)
            s += self.boundaryEdgeTags.shape[0]*sizeof(TAG_t)
        if self.dim >= 3:
            s += 3*self.boundaryFaces.shape[0]*sizeof(INDEX_t)
            s += self.boundaryFaceTags.shape[0]*sizeof(TAG_t)
        return s

    cdef void computeMeshQuantities(self):
        (self._h, self._delta,
         self._volume, self._hmin,
         self._volVector, self._hVector) = hdeltaCy(self)

    def resetMeshInfo(self):
        self._h, self._delta, self._volume, self._hmin = 0., 0., 0., 0.
        self._volVector = None
        self._hVector = None

    @property
    def h(self):
        if self._h <= 0:
            self.computeMeshQuantities()
        return self._h

    @property
    def delta(self):
        if self._delta <= 0:
            self.computeMeshQuantities()
        return self._delta

    @property
    def volume(self):
        if self._volume <= 0:
            self.computeMeshQuantities()
        return self._volume

    @property
    def hmin(self):
        if self._hmin <= 0:
            self.computeMeshQuantities()
        return self._hmin

    @property
    def volVector(self):
        if self._volVector is None:
            self.computeMeshQuantities()
        return np.array(self._volVector, copy=False)

    @property
    def hVector(self):
        if self._hVector is None:
            self.computeMeshQuantities()
        return np.array(self._hVector, copy=False)

    def __eq__(self, meshBase other):
        if not self.dim == other.dim:
            return False
        if not self.manifold_dim == other.manifold_dim:
            return False
        if not self.num_vertices == other.num_vertices:
            return False
        if not self.num_cells == other.num_cells:
            return False
        if self.vertices[0, 0] != other.vertices[0, 0]:
            return False
        if self.h != other.h:
            return False
        if self.hmin != other.hmin:
            return False
        if self.delta != other.delta:
            return False
        if self.volume != other.volume:
            return False
        return True

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void getSimplex(meshBase self,
                         const INDEX_t cellIdx,
                         REAL_t[:, ::1] simplex):
        cdef:
            INDEX_t m, k, l
        for m in range(self.cells.shape[1]):
            k = self.cells[cellIdx, m]
            for l in range(self.vertices.shape[1]):
                simplex[m, l] = self.vertices[k, l]

    def getSimplex_py(self,
                      INDEX_t cellIdx,
                      REAL_t[:, ::1] simplex):
        self.getSimplex(cellIdx, simplex)

    # def refine(self, returnLookup=False, sortRefine=False):
    #     from . mesh import mesh1d, mesh2d, mesh3d
    #     cdef:
    #         INDEX_t[:, ::1] new_boundaryEdges
    #         TAG_t[::1] new_boundaryEdgeTags
    #         INDEX_t[::1] new_boundaryVertices
    #         TAG_t[::1] new_boundaryVertexTags
    #         INDEX_t i, nv

    #     if self.dim == 1:
    #         vertices, new_cells, lookup = refineCy1D(self.vertices, self.cells)
    #         newMesh = mesh1d(vertices, new_cells)
    #         newMesh.boundaryVertices = self.boundaryVertices.copy()
    #         newMesh.boundaryVertexTags = self.boundaryVertexTags.copy()
    #     elif self.dim == 2:
    #         if sortRefine:
    #             # Refine the mesh by sorting all edges. Seems faster, but
    #             # ordering of DoFs seems to cause the solution time to go
    #             # up.
    #             vertices, new_cells, lookup = refineCy2Dsort(self.vertices,
    #                                                          self.cells)
    #         else:
    #             # Refine the mesh by building a lookup table of all edges.
    #             vertices, new_cells, lookup = refineCy2DedgeVals(self.vertices,
    #                                                              self.cells)
    #         newMesh = mesh2d(vertices, new_cells)

    #         new_boundaryEdges = uninitialized((2*self.boundaryEdges.shape[0], 2), dtype=INDEX)
    #         new_boundaryEdgeTags = uninitialized((2*self.boundaryEdges.shape[0]), dtype=TAG)
    #         new_boundaryVertices = uninitialized((self.boundaryEdges.shape[0]), dtype=INDEX)
    #         new_boundaryVertexTags = uninitialized((self.boundaryEdges.shape[0]), dtype=TAG)
    #         for i in range(self.boundaryEdges.shape[0]):
    #             e = self.boundaryEdges[i, :]
    #             t = self.boundaryEdgeTags[i]
    #             if e[0] < e[1]:
    #                 nv = lookup[encode_edge(e)]
    #             else:
    #                 e2 = np.array([e[1], e[0]])
    #                 nv = lookup[encode_edge(e2)]
    #             new_boundaryEdges[2*i, 0] = e[0]
    #             new_boundaryEdges[2*i, 1] = nv
    #             new_boundaryEdges[2*i+1, 0] = nv
    #             new_boundaryEdges[2*i+1, 1] = e[1]
    #             new_boundaryEdgeTags[2*i] = t
    #             new_boundaryEdgeTags[2*i+1] = t
    #             new_boundaryVertices[i] = nv
    #             new_boundaryVertexTags[i] = t
    #         newMesh.boundaryVertices = np.concatenate((self.boundaryVertices,
    #                                                    new_boundaryVertices))
    #         newMesh.boundaryVertexTags = np.concatenate((self.boundaryVertexTags,
    #                                                      new_boundaryVertexTags))
    #         newMesh.boundaryEdges = np.array(new_boundaryEdges, copy=False)
    #         newMesh.boundaryEdgeTags = np.array(new_boundaryEdgeTags, copy=False)
    #     elif self.dim == 3:
    #         vertices, new_cells, lookup = refineCy3DedgeVals(self.vertices,
    #                                                      self.cells)
    #         newMesh = mesh3d(vertices, new_cells)

    #         (newBV, newBVtags,
    #          newMesh.boundaryEdges,
    #          newMesh.boundaryEdgeTags,
    #          newMesh.boundaryFaces,
    #          newMesh.boundaryFaceTags) = newBoundaryAndTags3D(lookup, self.boundaryVertices,
    #                                                           self.boundaryEdges, self.boundaryFaces,
    #                                                           self.boundaryEdgeTags, self.boundaryFaceTags)
    #         newMesh.boundaryVertices = np.concatenate((self.boundaryVertices,
    #                                                    newBV))
    #         newMesh.boundaryVertexTags = np.concatenate((self.boundaryVertexTags,
    #                                                      newBVtags))
    #     else:
    #         raise NotImplementedError()
    #     if returnLookup:
    #         return newMesh, lookup
    #     else:
    #         return newMesh

    def copy(self):
        newVertices = np.array(self.vertices, copy=True)
        newCells = np.array(self.cells, copy=True)
        newMesh = type(self)(newVertices, newCells)
        if self.transformer is not None:
            from copy import deepcopy
            newMesh.setMeshTransformation(deepcopy(self.transformer))
        return newMesh

    def setMeshTransformation(self, meshTransformer transformer):
        self.transformer = transformer

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def refine(meshBase self, BOOL_t returnLookup=False, BOOL_t sortRefine=False):
        from . mesh import mesh1d, mesh2d, mesh3d
        cdef:
            INDEX_t i, nv
            TAG_t t
            INDEX_t[::1] e = uninitialized((2), dtype=INDEX)
            INDEX_t[::1] e2 = uninitialized((2), dtype=INDEX)
            INDEX_t[::1] new_boundaryVertices
            INDEX_t[:, ::1] new_boundaryEdges
            TAG_t[::1] new_boundaryVertexTags
            TAG_t[::1] new_boundaryEdgeTags
        if self.manifold_dim == 1:
            vertices, new_cells, lookup = refineCy1D(self.vertices, self.cells)
            newMesh = mesh1d(vertices, new_cells)
            newMesh.boundaryVertices = self.boundaryVertices.copy()
            newMesh.boundaryVertexTags = self.boundaryVertexTags.copy()
        elif self.manifold_dim == 2:
            if sortRefine:
                # Refine the mesh by sorting all edges. Seems faster, but
                # ordering of DoFs seems to cause the solution time to go
                # up.
                vertices, new_cells, lookup = refineCy2Dsort(self.vertices,
                                                             self.cells)
            else:
                # Refine the mesh by building a lookup table of all edges.
                vertices, new_cells, lookup = refineCy2DedgeVals(self.vertices,
                                                                 self.cells)
            newMesh = mesh2d(vertices, new_cells)

            new_boundaryEdges = uninitialized((2*self.boundaryEdges.shape[0], 2), dtype=INDEX)
            new_boundaryEdgeTags = uninitialized((2*self.boundaryEdges.shape[0]), dtype=TAG)
            new_boundaryVertices = uninitialized((self.boundaryEdges.shape[0]), dtype=INDEX)
            new_boundaryVertexTags = uninitialized((self.boundaryEdges.shape[0]), dtype=TAG)
            for i in range(self.boundaryEdges.shape[0]):
                e[0] = self.boundaryEdges[i, 0]
                e[1] = self.boundaryEdges[i, 1]
                t = self.boundaryEdgeTags[i]
                sortEdge(e[0], e[1], e2)
                nv = lookup[encode_edge(e2)]
                new_boundaryEdges[2*i, 0] = e[0]
                new_boundaryEdges[2*i, 1] = nv
                new_boundaryEdges[2*i+1, 0] = nv
                new_boundaryEdges[2*i+1, 1] = e[1]
                new_boundaryEdgeTags[2*i] = t
                new_boundaryEdgeTags[2*i+1] = t
                new_boundaryVertices[i] = nv
                new_boundaryVertexTags[i] = t
            newMesh.boundaryVertices = np.concatenate((self.boundaryVertices,
                                                       new_boundaryVertices))
            newMesh.boundaryVertexTags = np.concatenate((self.boundaryVertexTags,
                                                         new_boundaryVertexTags))
            newMesh.boundaryEdges = np.array(new_boundaryEdges, copy=False)
            newMesh.boundaryEdgeTags = np.array(new_boundaryEdgeTags, copy=False)
        elif self.manifold_dim == 3:
            vertices, new_cells, lookup = refineCy3DedgeVals(self.vertices,
                                                             self.cells)
            newMesh = mesh3d(vertices, new_cells)

            (newBV, newBVtags,
             newMesh.boundaryEdges,
             newMesh.boundaryEdgeTags,
             newMesh.boundaryFaces,
             newMesh.boundaryFaceTags) = newBoundaryAndTags3D(lookup, self.boundaryVertices,
                                                              self.boundaryEdges, self.boundaryFaces,
                                                              self.boundaryEdgeTags, self.boundaryFaceTags)
            newMesh.boundaryVertices = np.concatenate((self.boundaryVertices,
                                                       newBV))
            newMesh.boundaryVertexTags = np.concatenate((self.boundaryVertexTags,
                                                         newBVtags))
        else:
            raise NotImplementedError()
        if self.transformer is not None:
            self.transformer(newMesh, lookup)
            newMesh.setMeshTransformation(self.transformer)
        if returnLookup:
            return newMesh, lookup
        else:
            return newMesh

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def removeUnusedVertices(self):
        cdef:
            INDEX_t[::1] mapping = -np.ones((self.num_vertices), dtype=INDEX)
            INDEX_t manifold_dim = self.manifold_dim
            INDEX_t k, i, j, v
            REAL_t[:, ::1] vertices = self.vertices
            INDEX_t[:, ::1] cells = self.cells
            REAL_t[:, ::1] new_vertices
            INDEX_t[:, ::1] new_cells
            INDEX_t[::1] boundaryVertices, newBoundaryVertices
            TAG_t[::1] boundaryVertexTags, newBoundaryVertexTags
            INDEX_t[:, ::1] boundaryEdges
            INDEX_t[:, ::1] boundaryFaces
        k = 0
        for i in range(self.num_cells):
            for j in range(self.manifold_dim+1):
                v = cells[i, j]
                if mapping[v] == -1:
                    mapping[v] = k
                    k += 1
        new_vertices = uninitialized((k, self.dim), dtype=REAL)
        for i in range(mapping.shape[0]):
            k = mapping[i]
            if k == -1:
                continue
            for j in range(self.dim):
                new_vertices[k, j] = vertices[i, j]
        self.vertices = new_vertices
        self.num_vertices = new_vertices.shape[0]
        new_cells = uninitialized((self.num_cells, self.manifold_dim+1), dtype=INDEX)
        for i in range(self.num_cells):
            for j in range(manifold_dim+1):
                v = cells[i, j]
                new_cells[i, j] = mapping[v]
        self.cells = new_cells
        self.num_cells = new_cells.shape[0]

        if hasattr(self, '_boundaryVertices'):
            numBoundaryVertices = 0
            boundaryVertices = self._boundaryVertices
            boundaryVertexTags = self._boundaryVertexTags
            for i in range(boundaryVertices.shape[0]):
                v = boundaryVertices[i]
                j = mapping[v]
                if j != -1:
                    numBoundaryVertices += 1
            newBoundaryVertices = uninitialized((numBoundaryVertices), dtype=INDEX)
            newBoundaryVertexTags = uninitialized((numBoundaryVertices), dtype=TAG)
            k = 0
            for i in range(boundaryVertices.shape[0]):
                v = boundaryVertices[i]
                j = mapping[v]
                if j != -1:
                    newBoundaryVertices[k] = j
                    newBoundaryVertexTags[k] = boundaryVertexTags[i]
                    k += 1
            self._boundaryVertices = np.array(newBoundaryVertices, copy=False)
            self._boundaryVertexTags = np.array(newBoundaryVertexTags, copy=False)

        if hasattr(self, '_boundaryEdges'):
            boundaryEdges = self._boundaryEdges
            for i in range(boundaryEdges.shape[0]):
                for j in range(2):
                    boundaryEdges[i, j] = mapping[boundaryEdges[i, j]]
        if hasattr(self, '_boundaryFaces'):
            boundaryFaces = self._boundaryFaces
            for i in range(boundaryFaces.shape[0]):
                for j in range(3):
                    boundaryFaces[i, j] = mapping[boundaryFaces[i, j]]
        if hasattr(self, '_interiorVertices'):
            del self._interiorVertices

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    def getCellCenters(self):
        cdef:
            REAL_t[:, ::1] simplex = uninitialized((self.dim+1, self.dim), dtype=REAL)
            INDEX_t j, k
            REAL_t[:, ::1] centers = np.zeros((self.num_cells, self.dim), dtype=REAL)
            REAL_t fac = 1./(self.dim+1)
            INDEX_t cellNo

        for cellNo in range(self.num_cells):
            self.getSimplex(cellNo, simplex)
            for j in range(self.dim+1):
                for k in range(self.dim):
                    centers[cellNo, k] += simplex[j, k]
            for k in range(self.dim):
                centers[cellNo, k] *= fac
        return centers

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    def getProjectedCenters(self):
        cdef:
            REAL_t[:, ::1] simplex = uninitialized((self.dim+1, self.dim), dtype=REAL)
            INDEX_t j, k
            REAL_t[::1] mins = uninitialized((self.dim), dtype=REAL)
            REAL_t[::1] maxs = uninitialized((self.dim), dtype=REAL)
            REAL_t[:, ::1] centers = np.zeros((self.num_cells, self.dim), dtype=REAL)
            INDEX_t cellNo

        for cellNo in range(self.num_cells):
            self.getSimplex(cellNo, simplex)
            for k in range(self.dim):
                mins[k] = simplex[0, k]
                maxs[k] = simplex[0, k]
            for j in range(1, self.dim+1):
                for k in range(self.dim):
                    mins[k] = min(mins[k], simplex[j, k])
                    maxs[k] = max(maxs[k], simplex[j, k])
            for k in range(self.dim):
                centers[cellNo, k] = 0.5*(mins[k]+maxs[k])
        return centers

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef BOOL_t vertexInCell(self, REAL_t[::1] vertex, INDEX_t cellNo, REAL_t[:, ::1] simplexMem, REAL_t[::1] baryMem, REAL_t tol=0.):
        cdef:
            INDEX_t i
        self.getSimplex(cellNo, simplexMem)
        if self.dim == 1:
            getBarycentricCoords1D(simplexMem, vertex, baryMem)
        elif self.dim == 2:
            getBarycentricCoords2D(simplexMem, vertex, baryMem)
        else:
            raise NotImplementedError()
        for i in range(self.dim+1):
            if baryMem[i] < -tol:
                return False
        return True

    def vertexInCell_py(self, REAL_t[::1] vertex, INDEX_t cellNo, REAL_t tol=0.):
        simplex = uninitialized((self.dim+1, self.dim), dtype=REAL)
        bary = uninitialized((self.dim+1), dtype=REAL)
        return self.vertexInCell(vertex, cellNo, simplex, bary, tol)

    def getCellConnectivity(self, INDEX_t common_nodes=-1):
        cdef:
            list v2c
            list c2c
            list cellConnectivity
            INDEX_t cellNo, cellNo2, vertexNo, vertex
        if common_nodes < 0:
            common_nodes = 1
        v2c = []
        for vertex in range(self.num_vertices):
            v2c.append(set())
        for cellNo in range(self.num_cells):
            for vertexNo in range(self.cells.shape[1]):
                vertex = self.cells[cellNo, vertexNo]
                v2c[vertex].add(cellNo)
        c2c = []
        for cellNo in range(self.num_cells):
            c2c.append({})
        for vertex in range(self.num_vertices):
            if len(v2c[vertex]) > 1:
                for cellNo in v2c[vertex]:
                    for cellNo2 in v2c[vertex]:
                        if cellNo != cellNo2:
                            try:
                                c2c[cellNo][cellNo2].add(vertex)
                            except KeyError:
                                c2c[cellNo][cellNo2] = set([vertex])
        cellConnectivity = []
        for cellNo in range(self.num_cells):
            cellConnectivity.append(set())
        for cellNo in range(self.num_cells):
            for cellNo2 in c2c[cellNo]:
                if len(c2c[cellNo][cellNo2]) >= common_nodes:
                    cellConnectivity[cellNo].add(cellNo2)
        return cellConnectivity


# Encoding, decoding and sorting of edges

cdef ENCODE_t MAX_VAL_EDGE = (<ENCODE_t>2)**(<ENCODE_t>31)


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef ENCODE_t encode_edge(const INDEX_t[::1] e):
    return MAX_VAL_EDGE*<ENCODE_t>e[0]+<ENCODE_t>e[1]


def encode_edge_python(INDEX_t[::1] e):
    return encode_edge(e)


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef void decode_edge(const ENCODE_t encodeVal, INDEX_t[::1] e):
    e[0] = encodeVal // MAX_VAL_EDGE
    e[1] = encodeVal % MAX_VAL_EDGE


def decode_edge_python(ENCODE_t encodeVal):
    e = uninitialized((2), dtype=INDEX)
    decode_edge(encodeVal, e)
    return e


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sortEdge(const INDEX_t c0, const INDEX_t c1, INDEX_t[::1] e):
    if c0 < c1:
        e[0], e[1] = c0, c1
    else:
        e[0], e[1] = c1, c0


def sortEdge_py(const INDEX_t c0, const INDEX_t c1, INDEX_t[::1] e):
    sortEdge(c0, c1, e)

# Encoding, decoding and sorting of faces

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple encode_face(const INDEX_t[::1] f):
    return (f[0], MAX_VAL_EDGE*<ENCODE_t>f[1]+<ENCODE_t>f[2])


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void decode_face(tuple encodeVal, INDEX_t[::1] f):
    f[0] = encodeVal[0]
    f[1] = encodeVal[1] // MAX_VAL_EDGE
    f[2] = encodeVal[1] % MAX_VAL_EDGE


def encode_face_python(INDEX_t[::1] f):
    return encode_face(f)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sortFace(const INDEX_t c0, const INDEX_t c1, const INDEX_t c2,
                   INDEX_t[::1] f):
    if c0 < c1:
        if c0 < c2:
            if c1 < c2:
                f[0], f[1], f[2] = c0, c1, c2
            else:
                f[0], f[1], f[2] = c0, c2, c1
        else:
            f[0], f[1], f[2] = c2, c0, c1
    else:
        if c1 < c2:
            if c0 < c2:
                f[0], f[1], f[2] = c1, c0, c2
            else:
                f[0], f[1], f[2] = c1, c2, c0
        else:
            f[0], f[1], f[2] = c2, c1, c0


def sortFace_py(const INDEX_t c0, const INDEX_t c1, const INDEX_t c2, INDEX_t[::1] f):
    sortFace(c0, c1, c2, f)


@cython.cdivision(True)
def refineCy1D(const REAL_t[:, ::1] vertices,
               const INDEX_t[:, ::1] cells):
    cdef:
        INDEX_t num_vertices = vertices.shape[0]
        INDEX_t num_cells = cells.shape[0]
        INDEX_t c0, c1, i, j, k, nv0, new_num_vertices
        np.ndarray[INDEX_t, ndim=2] new_cells_mem = uninitialized((2*num_cells, 2),
                                                                  dtype=INDEX)
        REAL_t[:, ::1] new_vertices
        INDEX_t[:, ::1] new_cells = new_cells_mem
        INDEX_t[::1] e0 = uninitialized((2), dtype=INDEX)
        ENCODE_t hv
        dict lookup = {}

    new_num_vertices = num_vertices
    for i in range(num_cells):
        c0, c1 = cells[i, 0], cells[i, 1]
        sortEdge(c0, c1, e0)
        lookup[encode_edge(e0)] = new_num_vertices
        new_num_vertices += 1
    # vertices = np.vstack((vertices,
    #                       uninitialized((new_num_vertices-num_vertices, 1))))
    new_vertices_mem = uninitialized((new_num_vertices, vertices.shape[1]))
    new_vertices = new_vertices_mem
    for i in range(num_vertices):
        for k in range(vertices.shape[1]):
            new_vertices[i, k] = vertices[i, k]
    for hv, j in lookup.iteritems():
        decode_edge(hv, e0)
        for k in range(vertices.shape[1]):
            new_vertices[j, k] = (vertices[e0[0], k] + vertices[e0[1], k])*0.5

    # Add new cells
    for i in range(num_cells):
        c0, c1 = cells[i, 0], cells[i, 1]
        sortEdge(c0, c1, e0)
        nv0 = lookup[encode_edge(e0)]
        new_cells[2*i,   0], new_cells[2*i,   1] = c0,  nv0
        new_cells[2*i+1, 0], new_cells[2*i+1, 1] = nv0, c1

    return new_vertices_mem, new_cells_mem, lookup


# @cython.initializedcheck(False)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def refineCy2D(const REAL_t[:, ::1] vertices,
#                const INDEX_t[:, ::1] cells):
#     cdef:
#         INDEX_t num_vertices = vertices.shape[0]
#         INDEX_t num_cells = cells.shape[0]
#         INDEX_t c0, c1, c2, i, j, new_num_vertices, k, nv0, nv1, nv2, vno
#         ENCODE_t hv
#         np.ndarray[INDEX_t, ndim=2] new_cells_mem = uninitialized((4*num_cells, 3),
#                                                              dtype=INDEX)
#         INDEX_t[:, ::1] new_cells = new_cells_mem
#         REAL_t[:, ::1] new_vertices
#         np.ndarray[REAL_t, ndim=2] new_vertices_mem
#         dict lookup = {}
#         # unordered_map[ENCODE_t, INDEX_t] lookup
#         # map[ENCODE_t, INDEX_t] lookup
#         INDEX_t[:, ::1] temp = uninitialized((3, 2), dtype=INDEX)
#         INDEX_t[::1] e0 = temp[0, :]
#         INDEX_t[::1] e1 = temp[1, :]
#         INDEX_t[::1] e2 = temp[2, :]

#     # Build lookup table
#     # edge -> midpoint vertex number
#     new_num_vertices = num_vertices
#     for i in range(num_cells):
#         c0, c1, c2 = cells[i, 0], cells[i, 1], cells[i, 2]
#         sortEdge(c0, c1, e0)
#         sortEdge(c0, c2, e1)
#         sortEdge(c1, c2, e2)
#         for k in range(3):
#             try:
#                 lookup[encode_edge(temp[k, :])]
#             except KeyError:
#                 lookup[encode_edge(temp[k, :])] = new_num_vertices
#                 new_num_vertices += 1
#     new_vertices_mem = uninitialized((new_num_vertices, 2), dtype=REAL)
#     new_vertices = new_vertices_mem
#     # copy over old vertices
#     for i in range(num_vertices):
#         for k in range(2):
#             new_vertices[i, k] = vertices[i, k]
#     # insert new vertices
#     for hv, j in lookup.iteritems():
#         decode_edge(hv, e0)
#         for k in range(2):
#             new_vertices[j, k] = (vertices[e0[0], k] + vertices[e0[1], k])*0.5

#     # Add new cells
#     for i in range(num_cells):
#         c0, c1, c2 = cells[i, 0], cells[i, 1], cells[i, 2]
#         sortEdge(c0, c1, e0)
#         sortEdge(c0, c2, e1)
#         sortEdge(c1, c2, e2)
#         nv0 = lookup[encode_edge(e0)]
#         nv1 = lookup[encode_edge(e2)]
#         nv2 = lookup[encode_edge(e1)]
#         new_cells[4*i,   0], new_cells[4*i,   1], new_cells[4*i,   2] = c0,  nv0, nv2
#         new_cells[4*i+1, 0], new_cells[4*i+1, 1], new_cells[4*i+1, 2] = c1,  nv1, nv0
#         new_cells[4*i+2, 0], new_cells[4*i+2, 1], new_cells[4*i+2, 2] = c2,  nv2, nv1
#         new_cells[4*i+3, 0], new_cells[4*i+3, 1], new_cells[4*i+3, 2] = nv0, nv1, nv2

#     return new_vertices_mem, new_cells_mem, lookup


cdef inline int compareEdges(const void *pa, const void *pb) nogil:
    cdef:
        INDEX_t *a
        INDEX_t *b
    a = <INDEX_t *> pa
    b = <INDEX_t *> pb
    return 2*((a[0] > b[0])-(a[0] < b[0])) + ((a[1] > b[1])-(a[1] < b[1]))
    # if a[0] < b[0]:
    #     return -1
    # elif a[0] > b[0]:
    #     return 1
    # else:
    #     if a[1] < b[1]:
    #         return -1
    #     else:
    #         return 1


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def refineCy2Dsort(const REAL_t[:, ::1] vertices,
                   const INDEX_t[:, ::1] cells):
    cdef:
        INDEX_t num_vertices = vertices.shape[0]
        INDEX_t num_cells = cells.shape[0]
        INDEX_t c0, c1, c2, i, j, new_num_vertices, k, nv0, nv1, nv2
        ENCODE_t hv, hvOld
        np.ndarray[INDEX_t, ndim=2] new_cells_mem = uninitialized((4*num_cells, 3),
                                                             dtype=INDEX)
        INDEX_t[:, ::1] new_cells = new_cells_mem
        # INDEX_t[:, ::1] cells_mv = cells
        REAL_t[:, ::1] new_vertices
        dict lookup = {}
        INDEX_t[:, ::1] temp = uninitialized((3, 2), dtype=INDEX)
        INDEX_t[::1] e0 = temp[0, :]
        INDEX_t[::1] e1 = temp[1, :]
        INDEX_t[::1] e2 = temp[2, :]
        np.ndarray[REAL_t, ndim=2] new_vertices_mem
        np.ndarray[INDEX_t, ndim=2] edges_mem = uninitialized((3*num_cells, 2),
                                                         dtype=INDEX)
        INDEX_t[:, ::1] edges = edges_mem
    for i in range(num_cells):
        c0, c1, c2 = cells[i, 0], cells[i, 1], cells[i, 2]
        sortEdge(c0, c1, edges[3*i, :])
        sortEdge(c0, c2, edges[3*i+1, :])
        sortEdge(c1, c2, edges[3*i+2, :])
    qsort(&edges[0, 0],
          edges.shape[0], edges.shape[1]*sizeof(INDEX_t),
          compareEdges)
    new_num_vertices = num_vertices
    hvOld = 0
    for i in range(3*num_cells):
        hv = encode_edge(edges[i, :])
        if hv != hvOld:
            lookup[hv] = new_num_vertices
            new_num_vertices += 1
            hvOld = hv
    del edges, edges_mem
    new_vertices_mem = uninitialized((new_num_vertices, 2), dtype=REAL)
    new_vertices = new_vertices_mem
    # copy over old vertices
    for i in range(num_vertices):
        for k in range(2):
            new_vertices[i, k] = vertices[i, k]
    for hv, j in lookup.iteritems():
        decode_edge(hv, e0)
        for k in range(2):
            new_vertices[j, k] = (vertices[e0[0], k] + vertices[e0[1], k])*0.5

    # Add new cells
    for i in range(num_cells):
        c0, c1, c2 = cells[i, 0], cells[i, 1], cells[i, 2]
        sortEdge(c0, c1, e0)
        sortEdge(c0, c2, e1)
        sortEdge(c1, c2, e2)
        nv0 = lookup[encode_edge(e0)]
        nv1 = lookup[encode_edge(e2)]
        nv2 = lookup[encode_edge(e1)]
        new_cells[4*i,   0], new_cells[4*i,   1], new_cells[4*i,   2] = c0,  nv0, nv2
        new_cells[4*i+1, 0], new_cells[4*i+1, 1], new_cells[4*i+1, 2] = c1,  nv1, nv0
        new_cells[4*i+2, 0], new_cells[4*i+2, 1], new_cells[4*i+2, 2] = c2,  nv2, nv1
        new_cells[4*i+3, 0], new_cells[4*i+3, 1], new_cells[4*i+3, 2] = nv0, nv1, nv2
    return new_vertices_mem, new_cells_mem, lookup


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def refineCy2DedgeVals(const REAL_t[:, ::1] vertices,
                       const INDEX_t[:, ::1] cells):
    cdef:
        INDEX_t num_vertices = vertices.shape[0]
        INDEX_t num_cells = cells.shape[0]
        INDEX_t c0, c1, c2, i, nv = 0, new_num_vertices, k, nv0, nv1, nv2
        np.ndarray[INDEX_t, ndim=2] new_cells_mem = uninitialized((4*num_cells, 3),
                                                             dtype=INDEX)
        INDEX_t[:, ::1] new_cells = new_cells_mem
        REAL_t[:, ::1] new_vertices
        INDEX_t dim = vertices.shape[1]
        np.ndarray[REAL_t, ndim=2] new_vertices_mem
        dict lookup = {}
        INDEX_t[:, ::1] temp = uninitialized((3, 2), dtype=INDEX)
        INDEX_t[::1] e0 = temp[0, :]
        INDEX_t[::1] e1 = temp[1, :]
        INDEX_t[::1] e2 = temp[2, :]
        tupleDictINDEX eV = tupleDictINDEX(num_vertices, deleteHits=False)

    # Build lookup table
    # edge -> midpoint vertex number
    new_num_vertices = num_vertices
    for i in range(num_cells):
        c0, c1, c2 = cells[i, 0], cells[i, 1], cells[i, 2]
        sortEdge(c0, c1, e0)
        sortEdge(c0, c2, e1)
        sortEdge(c1, c2, e2)
        for k in range(3):
            if eV.enterValue(temp[k, :], new_num_vertices) == new_num_vertices:
                new_num_vertices += 1
    new_vertices_mem = uninitialized((new_num_vertices, vertices.shape[1]), dtype=REAL)
    new_vertices = new_vertices_mem
    # copy over old vertices
    for i in range(num_vertices):
        for k in range(dim):
            new_vertices[i, k] = vertices[i, k]
    # insert new vertices
    eV.startIter()
    while eV.next(e0, &nv):
        # FIX: Can we get rid of this lookup table and just keep the edgeVals?
        lookup[encode_edge(e0)] = nv
        for k in range(dim):
            new_vertices[nv, k] = (vertices[e0[0], k] + vertices[e0[1], k])*0.5

    # Add new cells
    for i in range(num_cells):
        c0, c1, c2 = cells[i, 0], cells[i, 1], cells[i, 2]
        sortEdge(c0, c1, e0)
        sortEdge(c0, c2, e1)
        sortEdge(c1, c2, e2)
        nv0 = eV.getValue(e0)
        nv1 = eV.getValue(e2)
        nv2 = eV.getValue(e1)
        new_cells[4*i,   0], new_cells[4*i,   1], new_cells[4*i,   2] = c0,  nv0, nv2
        new_cells[4*i+1, 0], new_cells[4*i+1, 1], new_cells[4*i+1, 2] = c1,  nv1, nv0
        new_cells[4*i+2, 0], new_cells[4*i+2, 1], new_cells[4*i+2, 2] = c2,  nv2, nv1
        new_cells[4*i+3, 0], new_cells[4*i+3, 1], new_cells[4*i+3, 2] = nv0, nv1, nv2
    return new_vertices_mem, new_cells_mem, lookup


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def refineCy2Dhash(const REAL_t[:, ::1] vertices,
                   const INDEX_t[:, ::1] cells):
    cdef:
        INDEX_t num_vertices = vertices.shape[0]
        INDEX_t num_cells = cells.shape[0]
        INDEX_t c0, c1, c2, i, nv = 0, new_num_vertices, k, nv0, nv1, nv2
        np.ndarray[INDEX_t, ndim=2] new_cells_mem = uninitialized((4*num_cells, 3),
                                                             dtype=INDEX)
        INDEX_t[:, ::1] new_cells = new_cells_mem
        REAL_t[:, ::1] new_vertices
        np.ndarray[REAL_t, ndim=2] new_vertices_mem
        INDEX_t[:, ::1] temp = uninitialized((3, 2), dtype=INDEX)
        INDEX_t[::1] e0 = temp[0, :]
        INDEX_t[::1] e1 = temp[1, :]
        INDEX_t[::1] e2 = temp[2, :]
        intTuple t, t0, t1, t2
        dict eV = {}

    # Build lookup table
    # edge -> midpoint vertex number
    new_num_vertices = num_vertices
    for i in range(num_cells):
        c0, c1, c2 = cells[i, 0], cells[i, 1], cells[i, 2]
        sortEdge(c0, c1, e0)
        t = intTuple.create(e0)
        try:
            eV[t]
        except KeyError:
            eV[t] = new_num_vertices
            new_num_vertices += 1

        sortEdge(c0, c2, e1)
        t = intTuple.create(e1)
        try:
            eV[t]
        except KeyError:
            eV[t] = new_num_vertices
            new_num_vertices += 1

        sortEdge(c1, c2, e2)
        t = intTuple.create(e2)
        try:
            eV[t]
        except KeyError:
            eV[t] = new_num_vertices
            new_num_vertices += 1

    new_vertices_mem = uninitialized((new_num_vertices, 2), dtype=REAL)
    new_vertices = new_vertices_mem
    # copy over old vertices
    for i in range(num_vertices):
        for k in range(2):
            new_vertices[i, k] = vertices[i, k]
    # insert new vertices
    for t, nv in eV.items():
        # FIX: Can we get rid of this lookup table and just keep the edgeVals?
        t.get(&e0[0])
        for k in range(2):
            new_vertices[nv, k] = (vertices[e0[0], k] + vertices[e0[1], k])*0.5

    # Add new cells
    t0 = intTuple.createNonOwning(e0)
    t1 = intTuple.createNonOwning(e1)
    t2 = intTuple.createNonOwning(e2)
    for i in range(num_cells):
        c0, c1, c2 = cells[i, 0], cells[i, 1], cells[i, 2]
        sortEdge(c0, c1, e0)
        sortEdge(c0, c2, e1)
        sortEdge(c1, c2, e2)
        nv0 = eV[t0]
        nv1 = eV[t2]
        nv2 = eV[t1]
        new_cells[4*i,   0], new_cells[4*i,   1], new_cells[4*i,   2] = c0,  nv0, nv2
        new_cells[4*i+1, 0], new_cells[4*i+1, 1], new_cells[4*i+1, 2] = c1,  nv1, nv0
        new_cells[4*i+2, 0], new_cells[4*i+2, 1], new_cells[4*i+2, 2] = c2,  nv2, nv1
        new_cells[4*i+3, 0], new_cells[4*i+3, 1], new_cells[4*i+3, 2] = nv0, nv1, nv2
    return new_vertices_mem, new_cells_mem, eV


# @cython.initializedcheck(False)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def refineCy3D(const REAL_t[:, ::1] vertices,
#                const INDEX_t[:, ::1] cells):
#     cdef:
#         INDEX_t num_vertices = vertices.shape[0]
#         INDEX_t num_cells = cells.shape[0]
#         REAL_t[:, ::1] new_vertices
#         np.ndarray[REAL_t, ndim=2] new_vertices_mem
#         INDEX_t v0, v1, v2, v3, i, j, m, k, new_num_vertices
#         INDEX_t v01, v02, v03, v12, v13, v23
#         dict lookup
#         INDEX_t[:, ::1] edges = uninitialized((7, 2), dtype=INDEX)
#         INDEX_t[::1] e01 = edges[0, :]
#         INDEX_t[::1] e02 = edges[1, :]
#         INDEX_t[::1] e03 = edges[2, :]
#         INDEX_t[::1] e12 = edges[3, :]
#         INDEX_t[::1] e13 = edges[4, :]
#         INDEX_t[::1] e23 = edges[5, :]
#         INDEX_t[::1] e = edges[6, :]
#         INDEX_t[:, ::1] faceedges = uninitialized((3, 2), dtype=INDEX)
#         INDEX_t[::1] e0 = faceedges[0, :]
#         INDEX_t[::1] e1 = faceedges[1, :]
#         INDEX_t[::1] e2 = faceedges[2, :]
#         INDEX_t[:, ::1] faces = uninitialized((4, 3), dtype=INDEX)
#         INDEX_t[::1] f012 = faces[0, :]
#         INDEX_t[::1] f013 = faces[1, :]
#         INDEX_t[::1] f023 = faces[2, :]
#         INDEX_t[::1] f123 = faces[3, :]
#         REAL_t l0123, l0213, l0312
#         np.ndarray[INDEX_t, ndim=2] new_cells_mem = uninitialized((8*num_cells, 4),
#                                                              dtype=INDEX)
#         REAL_t[:, ::1] temp = uninitialized((3, 3), dtype=REAL)
#         INDEX_t[:, ::1] new_cells = new_cells_mem
#         ENCODE_t hv
#     # Build lookup table
#     # edge -> midpoint vertex number
#     lookup = {}
#     new_num_vertices = num_vertices
#     for i in range(num_cells):
#         v0, v1, v2, v3 = (cells[i, 0], cells[i, 1],
#                           cells[i, 2], cells[i, 3])
#         # f012, f013, f023, f123 point to faces
#         sortFace(v0, v1, v2, f012)
#         sortFace(v0, v1, v3, f013)
#         sortFace(v0, v2, v3, f023)
#         sortFace(v1, v2, v3, f123)
#         for m in range(4):
#             e0[0], e0[1] = faces[m, 0], faces[m, 1]
#             e1[0], e1[1] = faces[m, 1], faces[m, 2]
#             e2[0], e2[1] = faces[m, 0], faces[m, 2]
#             for k in range(3):
#                 try:
#                     lookup[encode_edge(faceedges[k, :])]
#                 except KeyError:
#                     lookup[encode_edge(faceedges[k, :])] = new_num_vertices
#                     new_num_vertices += 1
#     new_vertices_mem = uninitialized((new_num_vertices, 3), dtype=REAL)
#     new_vertices = new_vertices_mem
#     # copy over old vertices
#     for i in range(num_vertices):
#         for k in range(3):
#             new_vertices[i, k] = vertices[i, k]
#     for hv, j in lookup.iteritems():
#         decode_edge(hv, e)
#         for k in range(3):
#             new_vertices[j, k] = (vertices[e[0], k] + vertices[e[1], k])*0.5

#     # Add new cells
#     for i in range(num_cells):
#         v0, v1, v2, v3 = (cells[i, 0], cells[i, 1],
#                           cells[i, 2], cells[i, 3])
#         sortEdge(v0, v1, e01)
#         sortEdge(v0, v2, e02)
#         sortEdge(v0, v3, e03)
#         sortEdge(v1, v2, e12)
#         sortEdge(v1, v3, e13)
#         sortEdge(v2, v3, e23)
#         v01 = lookup[encode_edge(e01)]
#         v02 = lookup[encode_edge(e02)]
#         v03 = lookup[encode_edge(e03)]
#         v12 = lookup[encode_edge(e12)]
#         v13 = lookup[encode_edge(e13)]
#         v23 = lookup[encode_edge(e23)]

#         # calculate length^2 of diagonals of internal octahedron
#         for j in range(3):
#             temp[0, j] = new_vertices[v01, j]-new_vertices[v23, j]
#             temp[1, j] = new_vertices[v02, j]-new_vertices[v13, j]
#             temp[2, j] = new_vertices[v03, j]-new_vertices[v12, j]
#         l0123 = mydot(temp[0, :], temp[0, :])
#         l0213 = mydot(temp[1, :], temp[1, :])
#         l0312 = mydot(temp[2, :], temp[2, :])

#         # I want the cells always to be oriented in the same way.
#         # => don't use Bey's algorithm, but shortest interior edge refinement

#         # cut off corners
#         new_cells[8*i,   0], new_cells[8*i,   1], new_cells[8*i,   2], new_cells[8*i,   3] = v0, v01, v02, v03
#         new_cells[8*i+1, 0], new_cells[8*i+1, 1], new_cells[8*i+1, 2], new_cells[8*i+1, 3] = v01, v1, v12, v13
#         new_cells[8*i+2, 0], new_cells[8*i+2, 1], new_cells[8*i+2, 2], new_cells[8*i+2, 3] = v02, v12, v2, v23
#         new_cells[8*i+3, 0], new_cells[8*i+3, 1], new_cells[8*i+3, 2], new_cells[8*i+3, 3] = v03, v13, v23, v3

#         if (l0123 < l0213) and (l0123 < l0312):
#             # shortest diagonal v01 - v23
#             new_cells[8*i+4, 0], new_cells[8*i+4, 1], new_cells[8*i+4, 2], new_cells[8*i+4, 3] = v01, v12, v02, v23
#             new_cells[8*i+5, 0], new_cells[8*i+5, 1], new_cells[8*i+5, 2], new_cells[8*i+5, 3] = v01, v23, v03, v13
#             new_cells[8*i+6, 0], new_cells[8*i+6, 1], new_cells[8*i+6, 2], new_cells[8*i+6, 3] = v01, v02, v03, v23
#             new_cells[8*i+7, 0], new_cells[8*i+7, 1], new_cells[8*i+7, 2], new_cells[8*i+7, 3] = v01, v13, v12, v23
#         elif (l0213 < l0312):
#             # shortest diagonal v02 - v13
#             new_cells[8*i+4, 0], new_cells[8*i+4, 1], new_cells[8*i+4, 2], new_cells[8*i+4, 3] = v01, v02, v03, v13
#             new_cells[8*i+5, 0], new_cells[8*i+5, 1], new_cells[8*i+5, 2], new_cells[8*i+5, 3] = v01, v12, v02, v13
#             new_cells[8*i+6, 0], new_cells[8*i+6, 1], new_cells[8*i+6, 2], new_cells[8*i+6, 3] = v02, v03, v13, v23
#             new_cells[8*i+7, 0], new_cells[8*i+7, 1], new_cells[8*i+7, 2], new_cells[8*i+7, 3] = v02, v13, v12, v23
#         else:
#             # shortest diagonal v03 - v12
#             new_cells[8*i+4, 0], new_cells[8*i+4, 1], new_cells[8*i+4, 2], new_cells[8*i+4, 3] = v01, v13, v12, v03
#             new_cells[8*i+5, 0], new_cells[8*i+5, 1], new_cells[8*i+5, 2], new_cells[8*i+5, 3] = v03, v23, v13, v12
#             new_cells[8*i+6, 0], new_cells[8*i+6, 1], new_cells[8*i+6, 2], new_cells[8*i+6, 3] = v03, v23, v12, v02
#             new_cells[8*i+7, 0], new_cells[8*i+7, 1], new_cells[8*i+7, 2], new_cells[8*i+7, 3] = v01, v12, v02, v03

#     return new_vertices_mem, new_cells_mem, lookup


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def refineCy3DedgeVals(const REAL_t[:, ::1] vertices,
                       const INDEX_t[:, ::1] cells):
    cdef:
        INDEX_t num_vertices = vertices.shape[0]
        INDEX_t num_cells = cells.shape[0]
        REAL_t[:, ::1] new_vertices
        np.ndarray[REAL_t, ndim=2] new_vertices_mem
        INDEX_t v0, v1, v2, v3, i, j = 0, m, k, new_num_vertices
        INDEX_t v01, v02, v03, v12, v13, v23
        dict lookup
        INDEX_t[:, ::1] edges = uninitialized((7, 2), dtype=INDEX)
        INDEX_t[::1] e01 = edges[0, :]
        INDEX_t[::1] e02 = edges[1, :]
        INDEX_t[::1] e03 = edges[2, :]
        INDEX_t[::1] e12 = edges[3, :]
        INDEX_t[::1] e13 = edges[4, :]
        INDEX_t[::1] e23 = edges[5, :]
        INDEX_t[::1] e = edges[6, :]
        INDEX_t[:, ::1] faceedges = uninitialized((3, 2), dtype=INDEX)
        INDEX_t[::1] e0 = faceedges[0, :]
        INDEX_t[::1] e1 = faceedges[1, :]
        INDEX_t[::1] e2 = faceedges[2, :]
        INDEX_t[:, ::1] faces = uninitialized((4, 3), dtype=INDEX)
        INDEX_t[::1] f012 = faces[0, :]
        INDEX_t[::1] f013 = faces[1, :]
        INDEX_t[::1] f023 = faces[2, :]
        INDEX_t[::1] f123 = faces[3, :]
        REAL_t l0123, l0213, l0312
        np.ndarray[INDEX_t, ndim=2] new_cells_mem = uninitialized((8*num_cells, 4),
                                                             dtype=INDEX)
        REAL_t[:, ::1] temp = uninitialized((3, 3), dtype=REAL)
        INDEX_t[:, ::1] new_cells = new_cells_mem
        tupleDictINDEX eV = tupleDictINDEX(num_vertices, deleteHits=False)
    # Build lookup table
    # edge -> midpoint vertex number
    lookup = {}
    new_num_vertices = num_vertices
    for i in range(num_cells):
        v0, v1, v2, v3 = (cells[i, 0], cells[i, 1],
                          cells[i, 2], cells[i, 3])
        # f012, f013, f023, f123 point to faces
        sortFace(v0, v1, v2, f012)
        sortFace(v0, v1, v3, f013)
        sortFace(v0, v2, v3, f023)
        sortFace(v1, v2, v3, f123)
        for m in range(4):
            e0[0], e0[1] = faces[m, 0], faces[m, 1]
            e1[0], e1[1] = faces[m, 1], faces[m, 2]
            e2[0], e2[1] = faces[m, 0], faces[m, 2]
            for k in range(3):
                if eV.enterValue(faceedges[k, :], new_num_vertices) == new_num_vertices:
                    new_num_vertices += 1
    new_vertices_mem = uninitialized((new_num_vertices, 3), dtype=REAL)
    new_vertices = new_vertices_mem
    # copy over old vertices
    for i in range(num_vertices):
        for k in range(3):
            new_vertices[i, k] = vertices[i, k]
    eV.startIter()
    while eV.next(e, &j):
        lookup[encode_edge(e)] = j
        for k in range(3):
            new_vertices[j, k] = (vertices[e[0], k] + vertices[e[1], k])*0.5

    # Add new cells
    for i in range(num_cells):
        v0, v1, v2, v3 = (cells[i, 0], cells[i, 1],
                          cells[i, 2], cells[i, 3])
        sortEdge(v0, v1, e01)
        sortEdge(v0, v2, e02)
        sortEdge(v0, v3, e03)
        sortEdge(v1, v2, e12)
        sortEdge(v1, v3, e13)
        sortEdge(v2, v3, e23)
        v01 = eV.getValue(e01)
        v02 = eV.getValue(e02)
        v03 = eV.getValue(e03)
        v12 = eV.getValue(e12)
        v13 = eV.getValue(e13)
        v23 = eV.getValue(e23)

        # calculate length^2 of diagonals of internal octahedron
        for j in range(3):
            temp[0, j] = new_vertices[v01, j]-new_vertices[v23, j]
            temp[1, j] = new_vertices[v02, j]-new_vertices[v13, j]
            temp[2, j] = new_vertices[v03, j]-new_vertices[v12, j]
        l0123 = mydot(temp[0, :], temp[0, :])
        l0213 = mydot(temp[1, :], temp[1, :])
        l0312 = mydot(temp[2, :], temp[2, :])

        # I want the cells always to be oriented in the same way.
        # => don't use Bey's algorithm, but shortest interior edge refinement

        # cut off corners
        new_cells[8*i,   0], new_cells[8*i,   1], new_cells[8*i,   2], new_cells[8*i,   3] = v0, v01, v02, v03
        new_cells[8*i+1, 0], new_cells[8*i+1, 1], new_cells[8*i+1, 2], new_cells[8*i+1, 3] = v01, v1, v12, v13
        new_cells[8*i+2, 0], new_cells[8*i+2, 1], new_cells[8*i+2, 2], new_cells[8*i+2, 3] = v02, v12, v2, v23
        new_cells[8*i+3, 0], new_cells[8*i+3, 1], new_cells[8*i+3, 2], new_cells[8*i+3, 3] = v03, v13, v23, v3

        if (l0123 < l0213) and (l0123 < l0312):
            # shortest diagonal v01 - v23
            new_cells[8*i+4, 0], new_cells[8*i+4, 1], new_cells[8*i+4, 2], new_cells[8*i+4, 3] = v01, v12, v02, v23
            new_cells[8*i+5, 0], new_cells[8*i+5, 1], new_cells[8*i+5, 2], new_cells[8*i+5, 3] = v01, v23, v03, v13
            new_cells[8*i+6, 0], new_cells[8*i+6, 1], new_cells[8*i+6, 2], new_cells[8*i+6, 3] = v01, v02, v03, v23
            new_cells[8*i+7, 0], new_cells[8*i+7, 1], new_cells[8*i+7, 2], new_cells[8*i+7, 3] = v01, v13, v12, v23
        elif (l0213 < l0312):
            # shortest diagonal v02 - v13
            new_cells[8*i+4, 0], new_cells[8*i+4, 1], new_cells[8*i+4, 2], new_cells[8*i+4, 3] = v01, v02, v03, v13
            new_cells[8*i+5, 0], new_cells[8*i+5, 1], new_cells[8*i+5, 2], new_cells[8*i+5, 3] = v01, v12, v02, v13
            new_cells[8*i+6, 0], new_cells[8*i+6, 1], new_cells[8*i+6, 2], new_cells[8*i+6, 3] = v02, v03, v13, v23
            new_cells[8*i+7, 0], new_cells[8*i+7, 1], new_cells[8*i+7, 2], new_cells[8*i+7, 3] = v02, v13, v12, v23
        else:
            # shortest diagonal v03 - v12
            new_cells[8*i+4, 0], new_cells[8*i+4, 1], new_cells[8*i+4, 2], new_cells[8*i+4, 3] = v01, v13, v12, v03
            new_cells[8*i+5, 0], new_cells[8*i+5, 1], new_cells[8*i+5, 2], new_cells[8*i+5, 3] = v03, v23, v13, v12
            new_cells[8*i+6, 0], new_cells[8*i+6, 1], new_cells[8*i+6, 2], new_cells[8*i+6, 3] = v03, v23, v12, v02
            new_cells[8*i+7, 0], new_cells[8*i+7, 1], new_cells[8*i+7, 2], new_cells[8*i+7, 3] = v01, v12, v02, v03

    return new_vertices_mem, new_cells_mem, lookup


def newBoundaryAndTags3D(dict lookup,
                         INDEX_t[::1] boundaryVertices,
                         INDEX_t[:, ::1] boundaryEdges,
                         INDEX_t[:, ::1] boundaryFaces,
                         TAG_t[::1] boundaryEdgeTags,
                         TAG_t[::1] boundaryFaceTags):
    cdef:
        INDEX_t i, nv, nv01, nv02, nv12, I
        TAG_t t
        np.ndarray[INDEX_t, ndim=2] new_boundaryEdges_mem = uninitialized((2*boundaryEdges.shape[0] +
                                                                      3*boundaryFaces.shape[0], 2), dtype=INDEX)
        np.ndarray[INDEX_t, ndim=1] new_boundaryVertices_mem = uninitialized((boundaryEdges.shape[0]), dtype=INDEX)
        np.ndarray[INDEX_t, ndim=2] new_boundaryFaces_mem = uninitialized((4*boundaryFaces.shape[0], 3), dtype=INDEX)
        np.ndarray[TAG_t, ndim=1] new_boundaryFaceTags_mem = uninitialized((4*boundaryFaces.shape[0]), dtype=TAG)
        np.ndarray[TAG_t, ndim=1] new_boundaryEdgeTags_mem = uninitialized((2*boundaryEdges.shape[0] +
                                                                       3*boundaryFaces.shape[0]), dtype=TAG)
        np.ndarray[TAG_t, ndim=1] new_boundaryVertexTags_mem = uninitialized((boundaryEdges.shape[0]), dtype=TAG)
        INDEX_t[:, ::1] new_boundaryFaces = new_boundaryFaces_mem
        INDEX_t[:, ::1] new_boundaryEdges = new_boundaryEdges_mem
        INDEX_t[::1] new_boundaryVertices = new_boundaryVertices_mem
        TAG_t[::1] new_boundaryFaceTags = new_boundaryFaceTags_mem
        TAG_t[::1] new_boundaryEdgeTags = new_boundaryEdgeTags_mem
        TAG_t[::1] new_boundaryVertexTags = new_boundaryVertexTags_mem
        INDEX_t[::1] e = uninitialized((2), dtype=INDEX)
        INDEX_t[::1] f = uninitialized((3), dtype=INDEX)

    for i in range(boundaryEdges.shape[0]):
        e = boundaryEdges[i, :]
        t = boundaryEdgeTags[i]
        nv = lookup[encode_edge(e)]
        new_boundaryEdges[2*i, 0] = e[0]
        new_boundaryEdges[2*i, 1] = nv
        new_boundaryEdges[2*i+1, 0] = e[1]
        new_boundaryEdges[2*i+1, 1] = nv
        new_boundaryEdgeTags[2*i] = t
        new_boundaryEdgeTags[2*i+1] = t
        new_boundaryVertices[i] = nv
        new_boundaryVertexTags[i] = t
    I = 2*boundaryEdges.shape[0]
    for i in range(boundaryFaces.shape[0]):
        f = boundaryFaces[i, :]
        t = boundaryFaceTags[i]
        e[0] = f[0]
        e[1] = f[1]
        sortEdge(e[0], e[1], e)
        nv01 = lookup[encode_edge(e)]
        e[0] = f[1]
        e[1] = f[2]
        sortEdge(e[0], e[1], e)
        nv12 = lookup[encode_edge(e)]
        e[0] = f[0]
        e[1] = f[2]
        sortEdge(e[0], e[1], e)
        nv02 = lookup[encode_edge(e)]
        new_boundaryFaces[4*i, 0] = f[0]
        new_boundaryFaces[4*i, 1] = nv01
        new_boundaryFaces[4*i, 2] = nv02
        new_boundaryFaces[4*i+1, 0] = f[1]
        new_boundaryFaces[4*i+1, 1] = nv12
        new_boundaryFaces[4*i+1, 2] = nv01
        new_boundaryFaces[4*i+2, 0] = f[2]
        new_boundaryFaces[4*i+2, 1] = nv02
        new_boundaryFaces[4*i+2, 2] = nv12
        new_boundaryFaces[4*i+3, 0] = nv01
        new_boundaryFaces[4*i+3, 1] = nv12
        new_boundaryFaces[4*i+3, 2] = nv02
        new_boundaryFaceTags[4*i:4*i+4] = t
        nv0, nv1 = (nv01, nv12) if nv01 < nv12 else (nv12, nv01)
        new_boundaryEdges[I+3*i, 0] = nv0
        new_boundaryEdges[I+3*i, 1] = nv1
        nv0, nv1 = (nv12, nv02) if nv12 < nv02 else (nv02, nv12)
        new_boundaryEdges[I+3*i+1, 0] = nv0
        new_boundaryEdges[I+3*i+1, 1] = nv1
        nv0, nv1 = (nv02, nv01) if nv02 < nv01 else (nv01, nv02)
        new_boundaryEdges[I+3*i+2, 0] = nv0
        new_boundaryEdges[I+3*i+2, 1] = nv1
        new_boundaryEdgeTags[I+3*i:I+3*i+3] = t

    return (new_boundaryVertices_mem, new_boundaryVertexTags_mem,
            new_boundaryEdges_mem, new_boundaryEdgeTags_mem,
            new_boundaryFaces_mem, new_boundaryFaceTags_mem)


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void vectorProduct(const REAL_t[::1] v, const REAL_t[::1] w,
                        REAL_t[::1] z):
    z[0] = v[1]*w[2]-v[2]*w[1]
    z[1] = v[2]*w[0]-v[0]*w[2]
    z[2] = v[0]*w[1]-v[1]*w[0]


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef REAL_t volume0D(REAL_t[:, ::1] span):
    return 1.


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef REAL_t volume1D(REAL_t[::1] v0):
    cdef REAL_t s = 0.0
    for i in range(v0.shape[0]):
        s += v0[i]**2
    return sqrt(s)


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef REAL_t volume1Dnew(REAL_t[:, ::1] span):
    return abs(span[0, 0])


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef REAL_t volume1D_in_2D(REAL_t[:, ::1] span):
    return sqrt(span[0, 0]**2+span[0, 1]**2)


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef REAL_t volume2D(REAL_t[::1] v0, REAL_t[::1] v1):
    return abs(v0[0]*v1[1]-v1[0]*v0[1])*0.5


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef REAL_t volume2D_in_3D(REAL_t[::1] v0, REAL_t[::1] v1):
    cdef:
        REAL_t temp_mem[3]
        REAL_t[::1] temp = temp_mem
    vectorProduct(v0, v1, temp)
    return sqrt(mydot(temp, temp))*0.5


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef REAL_t volume2D_in_3Dnew(REAL_t[:, ::1] span):
    cdef:
        REAL_t temp_mem[3]
        REAL_t[::1] temp = temp_mem
    vectorProduct(span[0, :], span[1, :], temp)
    return sqrt(mydot(temp, temp))*0.5


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef REAL_t volume2Dnew(REAL_t[:, ::1] span):
    return abs(span[0, 0]*span[1, 1]-span[0, 1]*span[1, 0])*0.5


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef REAL_t volume3D(REAL_t[:, ::1] span):
    cdef:
        REAL_t temp_mem[3]
        REAL_t[::1] temp = temp_mem
    vectorProduct(span[0, :], span[1, :], temp)
    return abs(mydot(span[2, :], temp))/6.0


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef REAL_t volume3Dnew(REAL_t[:, ::1] span, REAL_t[::1] temp):
    vectorProduct(span[0, :], span[1, :], temp)
    return abs(mydot(span[2, :], temp))/6.0


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef REAL_t volume0Dsimplex(REAL_t[:, ::1] simplex):
    return 1.


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef REAL_t volume1Dsimplex(REAL_t[:, ::1] simplex):
    return abs(simplex[1, 0]-simplex[0, 0])


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef REAL_t volume2Dsimplex(REAL_t[:, ::1] simplex):
    cdef:
        REAL_t v00 = simplex[1, 0]-simplex[0, 0]
        REAL_t v01 = simplex[1, 1]-simplex[0, 1]
        REAL_t v10 = simplex[2, 0]-simplex[0, 0]
        REAL_t v11 = simplex[2, 1]-simplex[0, 1]
    return abs(v00*v11-v10*v01)*0.5


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef REAL_t volume1Din2Dsimplex(REAL_t[:, ::1] simplex):
    cdef:
        REAL_t v0 = simplex[1, 0]-simplex[0, 0]
        REAL_t v1 = simplex[1, 1]-simplex[0, 1]
    return sqrt(v0*v0 + v1*v1)

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def hdeltaCy(meshBase mesh):
    cdef:
        INDEX_t space_dim = mesh.dim
        INDEX_t dim = mesh.manifold_dim
        INDEX_t nc = mesh.num_cells
        INDEX_t num_vertices = dim+1
        INDEX_t i, j
        REAL_t delta = 0, h = 0, hl, he, hS, vol, volS, hmin = 100.
        REAL_t[:, ::1] local_vertices = uninitialized((num_vertices, space_dim),
                                                        dtype=REAL)
        INDEX_t num_edges = 6
        REAL_t[:, ::1] temp = uninitialized((num_edges, dim), dtype=REAL)
        REAL_t[::1] v01 = temp[0, :]
        REAL_t[::1] v02 = temp[1, :]
        REAL_t[::1] v03 = temp[2, :]
        REAL_t[::1] v12 = temp[3, :]
        REAL_t[::1] v13 = temp[4, :]
        REAL_t[::1] v32 = temp[5, :]
        REAL_t[:, ::1] gradient = uninitialized((num_vertices, space_dim), dtype=REAL)
        REAL_t totalVolume = 0.0
        REAL_t[::1] volVec = uninitialized((nc), dtype=REAL)
        REAL_t[::1] hVec = uninitialized((nc), dtype=REAL)
    if dim == 1 and space_dim == 1:
        for i in range(nc):
            # Get local vertices
            mesh.getSimplex(i, local_vertices)
            hl = abs(local_vertices[1, 0]-local_vertices[0, 0])
            h = max(h, hl)
            hmin = min(hmin, hl)
            totalVolume += hl
            volVec[i] = hl
            hVec[i] = hl
        return h, 1, totalVolume, hmin, volVec, hVec
    elif dim == 1 and space_dim == 2:
        for i in range(nc):
            # Get local vertices
            mesh.getSimplex(i, local_vertices)
            hl = sqrt((local_vertices[1, 0]-local_vertices[0, 0])**2+(local_vertices[1, 1]-local_vertices[0, 1])**2)
            h = max(h, hl)
            hmin = min(hmin, hl)
            totalVolume += hl
            volVec[i] = hl
            hVec[i] = hl
        return h, 1, totalVolume, hmin, volVec, hVec
    elif dim == 2 and space_dim == 2:
        for i in range(nc):
            # Get local vertices
            mesh.getSimplex(i, local_vertices)

            # Calculate gradient matrix
            for j in range(space_dim):
                gradient[0, j] = local_vertices[2, j]-local_vertices[1, j]
                gradient[1, j] = local_vertices[2, j]-local_vertices[0, j]
                gradient[2, j] = local_vertices[1, j]-local_vertices[0, j]
            vol = volume2Dnew(gradient[1:, :])
            hl = 0.0
            volS = 0.0
            for j in range(3):
                hS = sqrt(mydot(gradient[j, :], gradient[j, :]))
                hmin = min(hmin, hS)
                hl = max(hl, hS)
                volS += hS
            delta = max(delta, hl*volS/4.0/vol)
            h = max(h, hl)
            totalVolume += vol
            volVec[i] = vol
            hVec[i] = hl
        return h, delta, totalVolume, hmin, volVec, hVec
    elif dim == 2 and space_dim == 3:
        for i in range(nc):
            # Get local vertices
            mesh.getSimplex(i, local_vertices)

            # Calculate gradient matrix
            for j in range(space_dim):
                gradient[0, j] = local_vertices[2, j]-local_vertices[1, j]
                gradient[1, j] = local_vertices[2, j]-local_vertices[0, j]
                gradient[2, j] = local_vertices[1, j]-local_vertices[0, j]
            vol = volume2D_in_3D(gradient[1, :], gradient[2, :])
            # assert vol > 0., "Cell {} volume: 0.>={}".format(i, vol)
            hl = 0.0
            volS = 0.0
            for j in range(3):
                hS = sqrt(mydot(gradient[j, :], gradient[j, :]))
                hmin = min(hmin, hS)
                hl = max(hl, hS)
                volS += hS
            delta = max(delta, hl*volS/4.0/vol)
            h = max(h, hl)
            totalVolume += vol
            volVec[i] = vol
            hVec[i] = hl
        return h, delta, totalVolume, hmin, volVec, hVec
    elif dim == 3 and space_dim == 3:
        for i in range(nc):
            # Get local vertices
            mesh.getSimplex(i, local_vertices)

            # Calculate gradient matrix
            for j in range(space_dim):
                v01[j] = local_vertices[1, j]-local_vertices[0, j]
                v02[j] = local_vertices[2, j]-local_vertices[0, j]
                v03[j] = local_vertices[3, j]-local_vertices[0, j]
                v12[j] = local_vertices[2, j]-local_vertices[1, j]
                v13[j] = local_vertices[3, j]-local_vertices[1, j]
                v32[j] = local_vertices[2, j]-local_vertices[3, j]

            vol = volume3D(temp[:3, :])
            # assert vol > 0., "Cell {} volume: 0.>={}".format(i, vol)
            hl = 0.0
            for j in range(6):
                he = mydot(temp[j, :], temp[j, :])
                hmin = min(hmin, he)
                hl = max(hl, he)
            hl = sqrt(hl)
            volS = volume2D_in_3D(v01, v02)+volume2D_in_3D(v01, v03)+volume2D_in_3D(v02, v03)+volume2D_in_3D(v12, v13)
            delta = max(delta, hl*volS/6.0/vol)
            h = max(h, hl)
            totalVolume += vol
            volVec[i] = vol
            hVec[i] = hl
        return h, delta, totalVolume, hmin, volVec, hVec
    else:
        return None


def boundaryVertices(INDEX_t[:, ::1] cells):
    cdef:
        INDEX_t nc = cells.shape[0]
        INDEX_t c0, c1
        set bvertices = set()
    assert cells.shape[1] == 2

    for i in range(nc):
        c0, c1 = cells[i, 0], cells[i, 1]
        try:
            bvertices.remove(c0)
        except KeyError:
            bvertices.add(c0)
        try:
            bvertices.remove(c1)
        except KeyError:
            bvertices.add(c1)
    return np.array(list(bvertices), dtype=INDEX)


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
def boundaryEdges(INDEX_t[:, ::1] cells, BOOL_t returnBoundaryCells=False):
    cdef:
        INDEX_t nc = cells.shape[0]
        INDEX_t c0, c1, c2, cellNo, i, k
        ENCODE_t hv
        INDEX_t[:, ::1] temp = uninitialized((3, 2), dtype=INDEX)
        INDEX_t[::1] e0 = temp[0, :]
        INDEX_t[::1] e1 = temp[1, :]
        INDEX_t[::1] e2 = temp[2, :]
        INDEX_t[:, ::1] bedges_mv
        INDEX_t[::1] bcells_mv
        dict bedges = dict()
        BOOL_t orientation

    if not returnBoundaryCells:
        for cellNo in range(nc):
            c0, c1, c2 = cells[cellNo, 0], cells[cellNo, 1], cells[cellNo, 2]
            sortEdge(c0, c1, e0)
            sortEdge(c1, c2, e1)
            sortEdge(c2, c0, e2)
            for k in range(3):
                hv = encode_edge(temp[k, :])
                try:
                    del bedges[hv]
                except KeyError:
                    bedges[hv] = cells[cellNo, k] == temp[k, 0]
        bedges_mv = uninitialized((len(bedges), 2), dtype=INDEX)

        i = 0
        for hv in bedges:
            orientation = bedges[hv]
            decode_edge(hv, e0)
            if orientation:
                bedges_mv[i, 0], bedges_mv[i, 1] = e0[0], e0[1]
            else:
                bedges_mv[i, 0], bedges_mv[i, 1] = e0[1], e0[0]
            i += 1
        return np.array(bedges_mv, copy=False)
    else:
        for cellNo in range(nc):
            c0, c1, c2 = cells[cellNo, 0], cells[cellNo, 1], cells[cellNo, 2]
            sortEdge(c0, c1, e0)
            sortEdge(c1, c2, e1)
            sortEdge(c2, c0, e2)
            for k in range(3):
                hv = encode_edge(temp[k, :])
                try:
                    del bedges[hv]
                except KeyError:
                    bedges[hv] = (cells[cellNo, k] == temp[k, 0], cellNo)
        bedges_mv = uninitialized((len(bedges), 2), dtype=INDEX)
        bcells_mv = uninitialized((len(bedges)), dtype=INDEX)

        i = 0
        for hv in bedges:
            orientation, cellNo = bedges[hv]
            bcells_mv[i] = cellNo
            decode_edge(hv, e0)
            if orientation:
                bedges_mv[i, 0], bedges_mv[i, 1] = e0[0], e0[1]
            else:
                bedges_mv[i, 0], bedges_mv[i, 1] = e0[1], e0[0]
            i += 1
        return np.array(bedges_mv, copy=False), np.array(bcells_mv, copy=False)


def boundaryVerticesFromBoundaryEdges(INDEX_t[:, ::1] bedges):
    cdef:
        set bvertices = set()
        INDEX_t i, k
        INDEX_t[::1] boundaryVertices

    for i in range(bedges.shape[0]):
        bvertices.add(bedges[i, 0])
        bvertices.add(bedges[i, 1])
    boundaryVertices = uninitialized((len(bvertices)), dtype=INDEX)
    k = 0
    for i in bvertices:
        boundaryVertices[k] = i
        k += 1
    assert k == len(bvertices)
    return np.array(boundaryVertices, copy=False)


def boundaryFaces(INDEX_t[:, ::1] cells):
    cdef:
        INDEX_t num_cells = cells.shape[0], i, k, j
        INDEX_t v0, v1, v2, v3
        INDEX_t[:, ::1] faces = uninitialized((4, 3), dtype=INDEX)
        INDEX_t[::1] f012 = faces[0, :]
        INDEX_t[::1] f013 = faces[1, :]
        INDEX_t[::1] f023 = faces[2, :]
        INDEX_t[::1] f123 = faces[3, :]
        INDEX_t[::1] f = faces[3, :]
        set bfaces = set()
        np.ndarray[INDEX_t, ndim=2] bfaces_mem
        INDEX_t[:, ::1] bfaces_mv
        tuple hv
    for i in range(num_cells):
        v0, v1, v2, v3 = (cells[i, 0], cells[i, 1],
                          cells[i, 2], cells[i, 3])
        sortFace(v0, v1, v2, f012)
        sortFace(v0, v1, v3, f013)
        sortFace(v0, v2, v3, f023)
        sortFace(v1, v2, v3, f123)
        for k in range(4):
            hv = encode_face(faces[k, :])
            try:
                bfaces.remove(hv)
            except KeyError:
                bfaces.add(hv)
    bfaces_mem = uninitialized((len(bfaces), 3), dtype=INDEX)
    bfaces_mv = bfaces_mem
    for i, hv in enumerate(bfaces):
        decode_face(hv, f)
        for j in range(3):
            bfaces_mv[i, j] = f[j]
    return bfaces_mem


def boundaryEdgesFromBoundaryFaces(INDEX_t[:, ::1] bfaces):
    cdef:
        INDEX_t nc = bfaces.shape[0]
        INDEX_t c0, c1, c2, i, k
        ENCODE_t hv
        INDEX_t[:, ::1] temp = uninitialized((3, 2), dtype=INDEX)
        INDEX_t[::1] e0 = temp[0, :]
        INDEX_t[::1] e1 = temp[1, :]
        INDEX_t[::1] e2 = temp[2, :]
        np.ndarray[INDEX_t, ndim=2] bedges_mem
        INDEX_t[:, ::1] bedges_mv

    bedges = set()
    for i in range(nc):
        c0, c1, c2 = bfaces[i, 0], bfaces[i, 1], bfaces[i, 2]
        sortEdge(c0, c1, e0)
        sortEdge(c0, c2, e1)
        sortEdge(c1, c2, e2)
        for k in range(3):
            bedges.add(encode_edge(temp[k, :]))
    bedges_mem = uninitialized((len(bedges), 2), dtype=INDEX)
    bedges_mv = bedges_mem

    for i, hv in enumerate(bedges):
        decode_edge(hv, e0)
        bedges_mv[i, 0], bedges_mv[i, 1] = e0[0], e0[1]
    return bedges_mem


cdef class faceVals:
    def __init__(self,
                 INDEX_t num_dofs,
                 np.uint8_t initial_length=0,
                 np.uint8_t length_inc=3,
                 BOOL_t deleteHits=True):
        cdef:
            INDEX_t i
        self.num_dofs = num_dofs
        self.initial_length = initial_length
        self.length_inc = length_inc
        self.nnz = 0
        self.counts = np.zeros((num_dofs), dtype=np.uint8)
        self.lengths = initial_length*np.ones((num_dofs), dtype=np.uint8)
        self.indexL = <INDEX_t **>malloc(num_dofs*sizeof(INDEX_t *))
        self.indexR = <INDEX_t **>malloc(num_dofs*sizeof(INDEX_t *))
        self.vals = <INDEX_t **>malloc(num_dofs*sizeof(INDEX_t *))
        # reserve initial memory for array of variable column size
        for i in range(num_dofs):
            self.indexL[i] = <INDEX_t *>malloc(self.initial_length *
                                               sizeof(INDEX_t))
            self.indexR[i] = <INDEX_t *>malloc(self.initial_length *
                                               sizeof(INDEX_t))
            self.vals[i] = <INDEX_t *>malloc(self.initial_length *
                                             sizeof(INDEX_t))
        self.deleteHits = deleteHits

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cdef inline INDEX_t enterValue(self, const INDEX_t[::1] f, INDEX_t val):
        cdef:
            INDEX_t m, n, I = f[0], J = f[1], K = f[2]
        for m in range(self.counts[I]):
            if self.indexL[I][m] == J and self.indexR[I][m] == K:  # J, K is already present
                val = self.vals[I][m]
                if self.deleteHits:
                    for n in range(m+1, self.counts[I]):
                        self.indexL[I][n-1] = self.indexL[I][n]
                        self.indexR[I][n-1] = self.indexR[I][n]
                        self.vals[I][n-1] = self.vals[I][n]
                    self.counts[I] -= 1
                    self.nnz -= 1
                return val
        else:
            # J,K was not present
            # Do we need more space?
            if self.counts[I] == self.lengths[I]:
                self.indexL[I] = <INDEX_t *>realloc(self.indexL[I],
                                                    (self.lengths[I] +
                                                     self.length_inc) *
                                                    sizeof(INDEX_t))
                self.indexR[I] = <INDEX_t *>realloc(self.indexR[I],
                                                    (self.lengths[I] +
                                                     self.length_inc) *
                                                    sizeof(INDEX_t))
                self.vals[I] = <INDEX_t *>realloc(self.vals[I],
                                                  (self.lengths[I] +
                                                   self.length_inc) *
                                                  sizeof(INDEX_t))
                self.lengths[I] += self.length_inc
            # where should we insert?
            for m in range(self.counts[I]):
                if self.indexL[I][m] > J or (self.indexL[I][m] == J and self.indexR[I][m] > K):
                    # move previous indices out of the way
                    for n in range(self.counts[I], m, -1):
                        self.indexL[I][n] = self.indexL[I][n-1]
                        self.indexR[I][n] = self.indexR[I][n-1]
                        self.vals[I][n] = self.vals[I][n-1]
                    # insert in empty spot
                    self.indexL[I][m] = J
                    self.indexR[I][m] = K
                    self.vals[I][m] = val
                    break
            else:
                self.indexL[I][self.counts[I]] = J
                self.indexR[I][self.counts[I]] = K
                self.vals[I][self.counts[I]] = val
            self.counts[I] += 1
            self.nnz += 1
            return val

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cdef inline INDEX_t getValue(self, const INDEX_t[::1] f):
        cdef:
            INDEX_t m
        for m in range(self.counts[f[0]]):
            if self.indexL[f[0]][m] == f[1] and self.indexR[f[0]][m] == f[2]:  # J is already present
                return self.vals[f[0]][m]

    def __getitem__(self, INDEX_t[::1] face):
        return self.getValue(face)

    def __dealloc__(self):
        cdef:
            INDEX_t i
        for i in range(self.num_dofs):
            free(self.indexL[i])
            free(self.indexR[i])
            free(self.vals[i])
        free(self.indexL)
        free(self.indexR)
        free(self.vals)

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cdef void startIter(self):
        self.i = 0
        while self.i < self.num_dofs and self.counts[self.i] == 0:
            self.i += 1
        self.jj = 0

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cdef BOOL_t next(self, INDEX_t[::1] f, INDEX_t * val):
        cdef:
            INDEX_t i = self.i, jj = self.jj, j, k
        if i < self.num_dofs:
            j = self.indexL[i][jj]
            k = self.indexR[i][jj]
            val[0] = self.vals[i][jj]
        else:
            return False
        f[0] = i
        f[1] = j
        f[2] = k
        if jj < self.counts[i]-1:
            self.jj += 1
        else:
            self.jj = 0
            i += 1
            while i < self.num_dofs and self.counts[i] == 0:
                i += 1
            self.i = i
        return True


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline BOOL_t inCell1D(REAL_t[:, ::1] simplex, REAL_t[::1] x):
    cdef:
        REAL_t bary_mem[2]
        REAL_t[::1] bary = bary_mem
    getBarycentricCoords1D(simplex, x, bary)
    return bary[0] >= 0 and bary[1] >= 0


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline BOOL_t inCell2D(REAL_t[:, ::1] simplex, REAL_t[::1] x):
    cdef:
        REAL_t bary_mem[3]
        REAL_t[::1] bary = bary_mem
    getBarycentricCoords2D(simplex, x, bary)
    return bary[0] >= 0 and bary[1] >= 0 and bary[2] >= 0


cdef class cellFinder(object):
    def __init__(self, meshBase mesh, INDEX_t numCandidates=-1):
        cdef:
            REAL_t[:, ::1] cellCenters = mesh.getCellCenters()

        from scipy.spatial import cKDTree
        self.mesh = mesh
        self.kd = (cKDTree(cellCenters), )
        self.simplex = uninitialized((mesh.dim+1, mesh.dim), dtype=REAL)
        self.bary = uninitialized((mesh.dim+1), dtype=REAL)
        if numCandidates <= 0:
            if mesh.dim == 1:
                numCandidates = 2
            else:
                numCandidates = self.mesh.dim+2
        self.numCandidates = numCandidates

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t findCell(self, REAL_t[::1] vertex):
        cdef:
            INDEX_t[::1] cellIdx
            INDEX_t cellNo
        cellIdx = self.kd[0].query(vertex, self.numCandidates)[1].astype(INDEX)
        for cellNo in cellIdx:
            if self.mesh.vertexInCell(vertex, cellNo, self.simplex, self.bary):
                return cellNo
        return -1
        # raise NotImplementedError('Could not find {}'.format(np.array(vertex)))


cdef class cellFinder2:
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, meshBase mesh):
        cdef:
            INDEX_t L, j, k, cellNo, vertexNo, vertex
            REAL_t h
            intTuple t
            REAL_t[:, ::1] cellCenters = mesh.getCellCenters()
        self.key = uninitialized((mesh.dim), dtype=INDEX)
        L = 1
        h = mesh.h
        while L*h < 0.5:
            L *= 2
        self.x_min = mesh.vertices_as_array.min(axis=0)
        x_max = mesh.vertices_as_array.max(axis=0)
        self.diamInv = uninitialized((mesh.dim), dtype=REAL)
        for j in range(mesh.dim):
            self.diamInv[j] = L / (x_max[j]-self.x_min[j]) / 1.01
        self.simplex = uninitialized((mesh.dim+1, mesh.dim), dtype=REAL)
        self.bary = uninitialized((mesh.dim+1), dtype=REAL)
        self.lookup = {}
        for k in range(cellCenters.shape[0]):
            for j in range(mesh.dim):
                self.key[j] = <INDEX_t>((cellCenters[k, j]-self.x_min[j]) * self.diamInv[j])
            t = intTuple.create(self.key)
            try:
                self.lookup[t].add(k)
            except KeyError:
                self.lookup[t] = set([k])
        self.mesh = mesh

        self.v2c = {}
        for cellNo in range(mesh.num_cells):
            for vertexNo in range(mesh.dim+1):
                vertex = mesh.cells[cellNo, vertexNo]
                try:
                    self.v2c[vertex].add(cellNo)
                except KeyError:
                    self.v2c[vertex] = set([cellNo])
        self.myKey = intTuple.createNonOwning(self.key)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INDEX_t findCell(self, REAL_t[::1] vertex):
        cdef:
            INDEX_t j, cellNo, vertexNo, v
            set candidates, toCheck = set()
            productIterator pit
            INDEX_t[::1] keyCenter
        for j in range(self.mesh.dim):
            self.key[j] = <INDEX_t>((vertex[j]-self.x_min[j]) * self.diamInv[j])
        try:
            candidates = self.lookup[self.myKey]
        except KeyError:
            keyCenter = np.array(self.key, copy=True)
            pit = productIterator(3, self.mesh.dim)
            candidates = set()
            pit.reset()
            while pit.step():
                for j in range(self.mesh.dim):
                    self.key[j] = keyCenter[j] + pit.idx[j]-1
                try:
                    candidates |= self.lookup[self.myKey]
                except KeyError:
                    pass

        # check if the vertex is in any of the cells
        for cellNo in candidates:
            if self.mesh.vertexInCell(vertex, cellNo, self.simplex, self.bary):
                return cellNo
        # add neighboring cells of candidate cells
        for cellNo in candidates:
            for vertexNo in range(self.mesh.dim+1):
                v = self.mesh.cells[cellNo, vertexNo]
                toCheck |= self.v2c[v]
        toCheck -= candidates
        for cellNo in toCheck:
            if self.mesh.vertexInCell(vertex, cellNo, self.simplex, self.bary):
                return cellNo
        # allow for some extra room
        for cellNo in candidates:
            if self.mesh.vertexInCell(vertex, cellNo, self.simplex, self.bary, 1e-15):
                return cellNo
        for cellNo in toCheck:
            if self.mesh.vertexInCell(vertex, cellNo, self.simplex, self.bary, 1e-15):
                return cellNo
        return -1

    def findCell_py(self, REAL_t[::1] vertex):
        return self.findCell(vertex)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void getBarycentricCoords1D(REAL_t[:, ::1] simplex, REAL_t[::1] x, REAL_t[::1] bary):
    cdef:
        REAL_t vol
    vol = simplex[0, 0]-simplex[1, 0]
    bary[0] = (x[0]-simplex[1, 0])/vol
    bary[1] = 1.-bary[0]


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void getBarycentricCoords2D(REAL_t[:, ::1] simplex, REAL_t[::1] x, REAL_t[::1] bary):
    cdef:
        REAL_t vol
    vol = ((simplex[0, 0]-simplex[1,0])*(simplex[2, 1]-simplex[1,1]) -
           (simplex[0, 1]-simplex[1,1])*(simplex[2, 0]-simplex[1,0]))
    bary[0] = ((x[0]-simplex[1, 0])*(simplex[2, 1]-simplex[1, 1]) -
               (x[1]-simplex[1, 1])*(simplex[2, 0]-simplex[1, 0]))/vol
    bary[1] = ((x[0]-simplex[2, 0])*(simplex[0, 1]-simplex[2, 1]) -
               (x[1]-simplex[2, 1])*(simplex[0, 0]-simplex[2, 0]))/vol
    bary[2] = 1. - bary[0] - bary[1]


def getSubmesh2(mesh, INDEX_t[::1] newCellIndices, INDEX_t num_cells=-1):
    cdef:
        INDEX_t[:, ::1] cells, newCells
        INDEX_t j, k, i, m, dim
    cells = mesh.cells
    dim = mesh.dim
    new_mesh = mesh.copy()

    if newCellIndices.shape[0] == cells.shape[0]:
        if num_cells <= 0:
            num_cells = 0
            for i in newCellIndices:
                if i >= 0:
                    num_cells += 1
        newCells = uninitialized((num_cells, dim+1), dtype=INDEX)
        j = 0
        k = 0
        for i in newCellIndices:
            if i >= 0:
                for m in range(dim+1):
                    newCells[i, m] = cells[k, m]
                j += 1
            k += 1
        assert j == num_cells
    else:
        newCells = uninitialized((newCellIndices.shape[0], dim+1), dtype=INDEX)
        for k, i in enumerate(newCellIndices):
            for m in range(dim+1):
                newCells[k, m] = cells[i, m]
    new_mesh.cells = newCells
    new_mesh.init()
    new_mesh.removeUnusedVertices()
    return new_mesh


def getSubmesh(meshBase mesh, INDEX_t[::1] selectedCells):
    cdef:
        INDEX_t manifold_dim = mesh.manifold_dim
        INDEX_t[:, ::1] old_cells = mesh.cells
        INDEX_t[:, ::1] new_cells = uninitialized((selectedCells.shape[0], mesh.manifold_dim+1), dtype=INDEX)
        INDEX_t i, j, I
        dict lookup
        INDEX_t[::1] boundaryVertices = mesh.boundaryVertices
        TAG_t[::1] boundaryVertexTags = mesh.boundaryVertexTags
        INDEX_t[:, ::1] boundaryEdges
        TAG_t[::1] boundaryEdgeTags
        INDEX_t[:, ::1] boundaryFaces
        TAG_t[::1] boundaryFaceTags
        INDEX_t hv1
        ENCODE_t hv
        INDEX_t e[2]
        INDEX_t f[3]
        meshBase new_mesh

    from . mesh import mesh1d, mesh2d, mesh3d

    for i in range(selectedCells.shape[0]):
        I = selectedCells[i]
        for j in range(manifold_dim+1):
            new_cells[i, j] = old_cells[I, j]
    if mesh.dim == 1:
        new_mesh = mesh1d(mesh.vertices.copy(), new_cells)
    elif mesh.dim == 2:
        new_mesh = mesh2d(mesh.vertices.copy(), new_cells)
    elif mesh.dim == 3:
        new_mesh = mesh3d(mesh.vertices.copy(), new_cells)
    else:
        raise NotImplementedError()

    # copy boundary vertex tags
    lookup = {}
    for i in range(boundaryVertices.shape[0]):
        I = boundaryVertices[i]
        lookup[I] = boundaryVertexTags[i]
    boundaryVertices = new_mesh.boundaryVertices
    boundaryVertexTags = new_mesh.boundaryVertexTags
    for i in range(boundaryVertices.shape[0]):
        try:
            boundaryVertexTags[i] = lookup.pop(boundaryVertices[i])
        except KeyError:
            pass
    new_mesh.boundaryVertexTags = np.array(boundaryVertexTags, copy=False, dtype=TAG)

    if mesh.dim >= 2:
        boundaryEdges = mesh.boundaryEdges
        boundaryEdgeTags = mesh.boundaryEdgeTags
        # copy boundary edge tags
        lookup = {}
        for i in range(boundaryEdges.shape[0]):
            sortEdge(boundaryEdges[i, 0], boundaryEdges[i, 1], e)
            hv = encode_edge(e)
            lookup[hv] = boundaryEdgeTags[i]
        boundaryEdges = new_mesh.boundaryEdges
        boundaryEdgeTags = new_mesh.boundaryEdgeTags
        for i in range(boundaryEdges.shape[0]):
            sortEdge(boundaryEdges[i, 0], boundaryEdges[i, 1], e)
            hv = encode_edge(e)
            try:
                boundaryEdgeTags[i] = lookup.pop(hv)
            except KeyError:
                pass
        new_mesh.boundaryEdgeTags = np.array(boundaryEdgeTags, copy=False, dtype=TAG)

    if mesh.dim >= 3:
        boundaryFaces = mesh.boundaryFaces
        boundaryFaceTags = mesh.boundaryFaceTags
        # copy boundary face tags
        lookup = {}
        for i in range(boundaryFaces.shape[0]):
            sortFace(boundaryFaces[i, 0], boundaryFaces[i, 1], boundaryFaces[i, 2], f)
            hv1, hv = encode_face(e)
            lookup[hv1, hv] = boundaryFaceTags[i]
        boundaryFaces = new_mesh.boundaryFaces
        boundaryFaceTags = new_mesh.boundaryFaceTags
        for i in range(boundaryFaces.shape[0]):
            sortFace(boundaryFaces[i, 0], boundaryFaces[i, 1], boundaryFaces[i, 2], f)
            hv1, hv = encode_face(e)
            try:
                boundaryFaceTags[i] = lookup.pop((hv1, hv))
            except KeyError:
                pass
        new_mesh.boundaryFaceTags = np.array(boundaryFaceTags, copy=False, dtype=TAG)

    # TODO: same for faces

    new_mesh.removeUnusedVertices()
    return new_mesh
