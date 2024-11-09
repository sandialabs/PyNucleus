###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes import INDEX
from PyNucleus_base import uninitialized

from . meshCy cimport (sortEdge, sortFace,
                       encode_face,
                       decode_face,
                       decode_edge,
                       encode_edge)


cdef class simplexMapper:
    def __init__(self, mesh=None):
        if mesh is not None:
            self.vertices = mesh.vertices
            self.cells = mesh.cells
            assert self.dim == mesh.dim
            assert self.manifold_dim == mesh.manifold_dim
        self.temp_edge = uninitialized((2), dtype=INDEX)
        self.temp_edge2 = uninitialized((2), dtype=INDEX)
        self.temp_face = uninitialized((3), dtype=INDEX)
        self.temp_face2 = uninitialized((3), dtype=INDEX)
        self.temp_cell = uninitialized((4), dtype=INDEX)
        self.iteration_counter = 0

    cdef void startLoopOverCellNodes(self, INDEX_t[::1] cell):
        cdef:
            INDEX_t i
        self.iteration_counter = 0
        for i in range(self.dim+1):
            self.temp_cell[i] = cell[i]

    cdef BOOL_t loopOverCellNodes(self, INDEX_t *node):
        if self.iteration_counter < self.dim+1:
            node[0] = self.temp_cell[self.iteration_counter]
            self.iteration_counter += 1
            return True
        else:
            return False

    cdef void startLoopOverCellEdges(self, INDEX_t[::1] cell):
        raise NotImplementedError()

    cdef BOOL_t loopOverCellEdges(self, INDEX_t[::1] edge):
        if self.iteration_counter < self.temp_edges.shape[0]:
            edge[0], edge[1] = self.temp_edges[self.iteration_counter, 0], self.temp_edges[self.iteration_counter, 1]
            self.iteration_counter += 1
            return True
        else:
            return False

    cdef BOOL_t loopOverCellEdgesEncoded(self, ENCODE_t * hv):
        cdef:
            BOOL_t rval
        rval = self.loopOverCellEdges(self.temp_edge2)
        hv[0] = self.sortAndEncodeEdge(self.temp_edge2)
        return rval

    def startLoopOverCellEdges_py(self, INDEX_t[::1] cell):
        self.startLoopOverCellEdges(cell)

    def loopOverCellEdges_py(self, INDEX_t[::1] edge):
        return self.loopOverCellEdges(edge)

    def loopOverCellEdgesEncoded_py(self, ENCODE_t[::1] hv):
        cdef:
            BOOL_t rval
        rval = self.loopOverCellEdgesEncoded(&hv[0])
        return rval

    cdef void startLoopOverCellFaces(self, INDEX_t[::1] cell):
        raise NotImplementedError()

    cdef BOOL_t loopOverCellFaces(self, INDEX_t[::1] face):
        if self.iteration_counter < self.temp_faces.shape[0]:
            face[0], face[1], face[2] = (self.temp_faces[self.iteration_counter, 0],
                                         self.temp_faces[self.iteration_counter, 1],
                                         self.temp_faces[self.iteration_counter, 2])
            self.iteration_counter += 1
            return True
        else:
            return False

    cdef BOOL_t loopOverCellFacesEncoded(self, INDEX_t * t0, ENCODE_t * t1):
        cdef:
            BOOL_t rval
        rval = self.loopOverCellFaces(self.temp_face2)
        t0[0], t1[0] = self.sortAndEncodeFace(self.temp_face2)
        return rval

    cdef void startLoopOverFaceEdges(self, INDEX_t[::1] face):
        raise NotImplementedError()

    cdef BOOL_t loopOverFaceEdges(self, INDEX_t[::1] edge):
        if self.iteration_counter < self.temp_edges2.shape[0]:
            edge[0], edge[1] = self.temp_edges2[self.iteration_counter, 0], self.temp_edges2[self.iteration_counter, 1]
            self.iteration_counter += 1
            return True
        else:
            return False

    cdef BOOL_t loopOverFaceEdgesEncoded(self, ENCODE_t * hv):
        cdef:
            BOOL_t rval
        rval = self.loopOverFaceEdges(self.temp_edge2)
        hv[0] = self.sortAndEncodeEdge(self.temp_edge2)
        return rval

    def startLoopOverFaceEdges_py(self, INDEX_t[::1] face):
        self.startLoopOverFaceEdges(face)

    def loopOverFaceEdges_py(self, INDEX_t[::1] edge):
        return self.loopOverFaceEdges(edge)

    def loopOverFaceEdgesEncoded_py(self, list hv):
        cdef:
            ENCODE_t hv2
            BOOL_t rval
        rval = self.loopOverFaceEdgesEncoded(&hv2)
        hv[0] = hv2
        return rval

    cdef INDEX_t getVertexInCell(self, INDEX_t cellNo, INDEX_t vertexNo):
        return self.cells[cellNo, vertexNo]

    def getVertexInCell_py(self, INDEX_t cellNo, INDEX_t vertexNo):
        return self.getVertexInCell(cellNo, vertexNo)

    cdef INDEX_t findVertexInCell(self, INDEX_t cellNo, INDEX_t vertexNo):
        cdef:
            INDEX_t i
        for i in range(self.cells.shape[1]):
            if self.cells[cellNo, i] == vertexNo:
                return i
        return -1

    def findVertexInCell_py(self, INDEX_t cellNo, INDEX_t vertexNo):
        return self.findVertexInCell(cellNo, vertexNo)

    cdef void getEdgeVerticesLocal(self,
                                   INDEX_t edgeNo,
                                   INDEX_t order,
                                   INDEX_t[::1] indices):
        if edgeNo == 0:
            indices[0] = 0
            indices[1] = 1
        elif edgeNo == 1:
            indices[0] = 1
            indices[1] = 2
        elif edgeNo == 2:
            indices[0] = 2
            indices[1] = 0
        elif edgeNo == 3:
            indices[0] = 0
            indices[1] = 3
        elif edgeNo == 4:
            indices[0] = 1
            indices[1] = 3
        else:
            indices[0] = 2
            indices[1] = 3
        if order == 1:
            indices[0], indices[1] = indices[1], indices[0]

    cdef void getFaceVerticesLocal(self,
                                   INDEX_t faceNo,
                                   INDEX_t order,
                                   INDEX_t[::1] indices):
        if faceNo == 0:
            indices[0], indices[1], indices[2] = 0, 2, 1
        elif faceNo == 1:
            indices[0], indices[1], indices[2] = 0, 1, 3
        elif faceNo == 2:
            indices[0], indices[1], indices[2] = 1, 2, 3
        else:
            indices[0], indices[1], indices[2] = 2, 0, 3

        if order == 1:
            indices[0], indices[1], indices[2] = indices[1], indices[2], indices[0]
        elif order == 2:
            indices[0], indices[1], indices[2] = indices[2], indices[0], indices[1]
        elif order == -1:
            indices[0], indices[1], indices[2] = indices[1], indices[0], indices[2]
        elif order == -2:
            indices[0], indices[1], indices[2] = indices[0], indices[2], indices[1]
        elif order == -3:
            indices[0], indices[1], indices[2] = indices[2], indices[1], indices[0]

    cdef void getFaceEdgesLocal(self,
                                INDEX_t faceNo,
                                INDEX_t order,
                                INDEX_t[::1] indices,
                                INDEX_t[::1] orders):
        if faceNo == 0:
            indices[0], indices[1], indices[2] = 2, 1, 0
            orders[0], orders[1], orders[2] = 1, 1, 1
        elif faceNo == 1:
            indices[0], indices[1], indices[2] = 0, 4, 3
            orders[0], orders[1], orders[2] = 0, 0, 1
        elif faceNo == 2:
            indices[0], indices[1], indices[2] = 1, 5, 4
            orders[0], orders[1], orders[2] = 0, 0, 1
        else:
            indices[0], indices[1], indices[2] = 2, 3, 5
            orders[0], orders[1], orders[2] = 0, 0, 1

        if order == 1:
            indices[0], indices[1], indices[2] = indices[1], indices[2], indices[0]
            orders[0], orders[1], orders[2] = orders[1], orders[2], orders[0]
        elif order == 2:
            indices[0], indices[1], indices[2] = indices[2], indices[0], indices[1]
            orders[0], orders[1], orders[2] = orders[2], orders[0], orders[1]
        elif order == -1:
            indices[0], indices[1], indices[2] = indices[0], indices[2], indices[1]
            orders[0], orders[1], orders[2] = 1-orders[0], 1-orders[2], 1-orders[1]
        elif order == -2:
            indices[0], indices[1], indices[2] = indices[2], indices[1], indices[0]
            orders[0], orders[1], orders[2] = 1-orders[2], 1-orders[1], 1-orders[0]
        elif order == -3:
            indices[0], indices[1], indices[2] = indices[1], indices[0], indices[2]
            orders[0], orders[1], orders[2] = 1-orders[1], 1-orders[0], 1-orders[2]

    cdef void getEdgeVerticesGlobal(self,
                                    INDEX_t cellNo,
                                    INDEX_t edgeNo,
                                    INDEX_t order,
                                    INDEX_t[::1] indices):
        cdef:
            INDEX_t j
        self.getEdgeVerticesLocal(edgeNo, order, indices)
        for j in range(2):
            indices[j] = self.cells[cellNo, indices[j]]

    cdef void getFaceVerticesGlobal(self,
                                    INDEX_t cellNo,
                                    INDEX_t faceNo,
                                    INDEX_t order,
                                    INDEX_t[::1] indices):
        cdef:
            INDEX_t j
        self.getFaceVerticesLocal(faceNo, order, indices)
        for j in range(3):
            indices[j] = self.cells[cellNo, indices[j]]

    cdef ENCODE_t sortAndEncodeEdge(self, INDEX_t[::1] edge):
        sortEdge(edge[0], edge[1], self.temp_edge)
        return encode_edge(self.temp_edge)

    def sortAndEncodeEdge_py(self, INDEX_t[::1] edge):
        return self.sortAndEncodeEdge(edge)

    cdef INDEX_t findEdgeInCell(self,
                                INDEX_t cellNo,
                                INDEX_t[::1] edge):
        raise NotImplementedError()

    def findEdgeInCell_py(self, INDEX_t cellNo,
                          INDEX_t[::1] edge):
        return self.findEdgeInCell(cellNo, edge)

    cdef INDEX_t findEdgeInCellEncoded(self,
                                       INDEX_t cellNo,
                                       ENCODE_t hv):
        decode_edge(hv, self.temp_edge2)
        return self.findEdgeInCell(cellNo, self.temp_edge2)

    def findEdgeInCellEncoded_py(self,
                                 INDEX_t cellNo,
                                 ENCODE_t hv):
        return self.findEdgeInCellEncoded(cellNo, hv)

    cdef void getEdgeInCell(self,
                            INDEX_t cellNo,
                            INDEX_t edgeNo,
                            INDEX_t[::1] edge,
                            BOOL_t sorted=False):
        raise NotImplementedError()

    def getEdgeInCell_py(self,
                         INDEX_t cellNo,
                         INDEX_t edgeNo,
                         BOOL_t sorted=False):
        edge = uninitialized((2), dtype=INDEX)
        self.getEdgeInCell(cellNo, edgeNo, edge, sorted)
        return edge

    cdef ENCODE_t getEdgeInCellEncoded(self,
                                       INDEX_t cellNo,
                                       INDEX_t edgeNo):
        self.getEdgeInCell(cellNo, edgeNo, self.temp_edge)
        return self.sortAndEncodeEdge(self.temp_edge)

    def getEdgeInCellEncoded_py(self,
                                INDEX_t cellNo,
                                INDEX_t edgeNo):
        return self.getEdgeInCellEncoded(cellNo, edgeNo)

    cdef tuple sortAndEncodeFace(self, INDEX_t[::1] face):
        sortFace(face[0], face[1], face[2], self.temp_face)
        return encode_face(self.temp_face)

    def sortAndEncodeFace_py(self, INDEX_t[::1] face):
        return self.sortAndEncodeFace(face)

    cdef INDEX_t findFaceInCell(self,
                                INDEX_t cellNo,
                                INDEX_t[::1] face):
        raise NotImplementedError()

    def findFaceInCell_py(self,
                          INDEX_t cellNo,
                          INDEX_t[::1] face):
        return self.findFaceInCell(cellNo, face)

    cdef void getFaceInCell(self,
                            INDEX_t cellNo,
                            INDEX_t faceNo,
                            INDEX_t[::1] face,
                            BOOL_t sorted=False):
        raise NotImplementedError()

    def getFaceInCell_py(self,
                         INDEX_t cellNo,
                         INDEX_t faceNo,
                         BOOL_t sorted=False):
        face = uninitialized((3), dtype=INDEX)
        self.getFaceInCell(cellNo, faceNo, face, sorted)
        return face

    cdef tuple getFaceInCellEncoded(self,
                                    INDEX_t cellNo,
                                    INDEX_t faceNo):
        self.getFaceInCell(cellNo, faceNo, self.temp_face)
        return self.sortAndEncodeFace(self.temp_face)

    def getFaceInCellEncoded_py(self,
                                INDEX_t cellNo,
                                INDEX_t faceNo):
        return self.getFaceInCellEncoded(cellNo, faceNo)

    cdef INDEX_t findFaceInCellEncoded(self,
                                       INDEX_t cellNo,
                                       tuple hv):
        decode_face(hv, self.temp_face2)
        return self.findFaceInCell(cellNo, self.temp_face2)

    def findFaceInCellEncoded_py(self,
                                 INDEX_t cellNo,
                                 tuple hv):
        return self.findFaceInCellEncoded(cellNo, hv)

    cdef void getEdgeSimplex(self,
                             INDEX_t cellNo,
                             INDEX_t edgeNo,
                             REAL_t[:, ::1] edgeSimplex):
        cdef:
            INDEX_t vertexNo, j
        self.getEdgeInCell(cellNo, edgeNo, self.temp_edge)
        for vertexNo in range(2):
            for j in range(self.dim):
                edgeSimplex[vertexNo, j] = self.vertices[self.temp_edge[vertexNo], j]

    cdef void getEncodedEdgeSimplex(self,
                                    ENCODE_t hv,
                                    REAL_t[:, ::1] edgeSimplex):
        cdef:
            INDEX_t vertexNo, j
        decode_edge(hv, self.temp_edge)
        for vertexNo in range(2):
            for j in range(self.dim):
                edgeSimplex[vertexNo, j] = self.vertices[self.temp_edge[vertexNo], j]

    cdef void getFaceSimplex(self,
                             INDEX_t cellNo,
                             INDEX_t faceNo,
                             REAL_t[:, ::1] faceSimplex):
        cdef:
            INDEX_t vertexNo, j
        self.getFaceInCell(cellNo, faceNo, self.temp_face)
        for vertexNo in range(3):
            for j in range(self.dim):
                faceSimplex[vertexNo, j] = self.vertices[self.temp_face[vertexNo], j]

    cdef void getEncodedFaceSimplex(self,
                                    tuple hv,
                                    REAL_t[:, ::1] faceSimplex):
        cdef:
            INDEX_t vertexNo, j
        decode_face(hv, self.temp_face)
        for vertexNo in range(3):
            for j in range(self.dim):
                faceSimplex[vertexNo, j] = self.vertices[self.temp_face[vertexNo], j]


cdef class simplexMapper0D(simplexMapper):
    def __init__(self, mesh=None):
        self.manifold_dim = 0
        if mesh is not None:
            self.dim = mesh.dim
        super(simplexMapper0D, self).__init__(mesh)


cdef class simplexMapper1D(simplexMapper):
    def __init__(self, mesh=None):
        self.manifold_dim = 1
        if mesh is not None:
            self.dim = mesh.dim
        super(simplexMapper1D, self).__init__(mesh)


cdef class simplexMapper2D(simplexMapper):
    def __init__(self, mesh=None):
        self.manifold_dim = 2
        if mesh is not None:
            self.dim = mesh.dim
        super(simplexMapper2D, self).__init__(mesh)
        self.temp_edges = uninitialized((3, 2), dtype=INDEX)

    cdef void startLoopOverCellEdges(self, INDEX_t[::1] cell):
        cdef:
            INDEX_t c0, c1, c2
        c0, c1, c2 = cell[0], cell[1], cell[2]
        sortEdge(c0, c1, self.temp_edges[0, :])
        sortEdge(c0, c2, self.temp_edges[1, :])
        sortEdge(c1, c2, self.temp_edges[2, :])
        self.iteration_counter = 0

    cdef INDEX_t findEdgeInCell(self,
                                INDEX_t cellNo,
                                INDEX_t[::1] edge):
        sortEdge(self.cells[cellNo, 0], self.cells[cellNo, 1], self.temp_edge)
        if self.temp_edge[0] == edge[0] and self.temp_edge[1] == edge[1]:
            return 0
        sortEdge(self.cells[cellNo, 1], self.cells[cellNo, 2], self.temp_edge)
        if self.temp_edge[0] == edge[0] and self.temp_edge[1] == edge[1]:
            return 1
        return 2

    cdef void getEdgeInCell(self,
                            INDEX_t cellNo,
                            INDEX_t edgeNo,
                            INDEX_t[::1] edge,
                            BOOL_t sorted=False):
        if edgeNo == 0:
            edge[0], edge[1] = self.cells[cellNo, 0], self.cells[cellNo, 1]
        elif edgeNo == 1:
            edge[0], edge[1] = self.cells[cellNo, 1], self.cells[cellNo, 2]
        else:
            edge[0], edge[1] = self.cells[cellNo, 2], self.cells[cellNo, 0]
        if sorted:
            sortEdge(edge[0], edge[1], edge)


cdef class simplexMapper3D(simplexMapper):
    def __init__(self, mesh=None):
        self.manifold_dim = 3
        if mesh is not None:
            self.dim = mesh.dim
        super(simplexMapper3D, self).__init__(mesh)
        self.temp_edges = uninitialized((6, 2), dtype=INDEX)
        self.temp_faces = uninitialized((4, 3), dtype=INDEX)
        self.temp_edges2 = uninitialized((3, 2), dtype=INDEX)

    cdef void startLoopOverCellEdges(self, INDEX_t[::1] cell):
        cdef:
            INDEX_t c0, c1, c2, c3
        c0, c1, c2, c3 = cell[0], cell[1], cell[2], cell[3]
        sortEdge(c0, c1, self.temp_edges[0, :])
        sortEdge(c1, c2, self.temp_edges[1, :])
        sortEdge(c2, c0, self.temp_edges[2, :])
        sortEdge(c0, c3, self.temp_edges[3, :])
        sortEdge(c1, c3, self.temp_edges[4, :])
        sortEdge(c2, c3, self.temp_edges[5, :])
        self.iteration_counter = 0

    cdef void startLoopOverCellFaces(self, INDEX_t[::1] cell):
        cdef:
            INDEX_t c0, c1, c2, c3
        c0, c1, c2, c3 = cell[0], cell[1], cell[2], cell[3]
        sortFace(c0, c1, c2, self.temp_faces[0, :])
        sortFace(c0, c1, c3, self.temp_faces[1, :])
        sortFace(c1, c2, c3, self.temp_faces[2, :])
        sortFace(c0, c2, c3, self.temp_faces[3, :])
        self.iteration_counter = 0

    cdef void startLoopOverFaceEdges(self, INDEX_t[::1] face):
        cdef:
            INDEX_t c0, c1, c2
        c0, c1, c2 = face[0], face[1], face[2]
        sortEdge(c0, c1, self.temp_edges2[0, :])
        sortEdge(c1, c2, self.temp_edges2[1, :])
        sortEdge(c2, c0, self.temp_edges2[2, :])
        self.iteration_counter = 0

    cdef INDEX_t findEdgeInCell(self,
                                INDEX_t cellNo,
                                INDEX_t[::1] edge):
        sortEdge(self.cells[cellNo, 0], self.cells[cellNo, 1], self.temp_edge)
        if self.temp_edge[0] == edge[0] and self.temp_edge[1] == edge[1]:
            return 0
        sortEdge(self.cells[cellNo, 1], self.cells[cellNo, 2], self.temp_edge)
        if self.temp_edge[0] == edge[0] and self.temp_edge[1] == edge[1]:
            return 1
        sortEdge(self.cells[cellNo, 0], self.cells[cellNo, 2], self.temp_edge)
        if self.temp_edge[0] == edge[0] and self.temp_edge[1] == edge[1]:
            return 2
        sortEdge(self.cells[cellNo, 0], self.cells[cellNo, 3], self.temp_edge)
        if self.temp_edge[0] == edge[0] and self.temp_edge[1] == edge[1]:
            return 3
        sortEdge(self.cells[cellNo, 1], self.cells[cellNo, 3], self.temp_edge)
        if self.temp_edge[0] == edge[0] and self.temp_edge[1] == edge[1]:
            return 4
        return 5

    cdef void getEdgeInCell(self,
                            INDEX_t cellNo,
                            INDEX_t edgeNo,
                            INDEX_t[::1] edge,
                            BOOL_t sorted=False):
        if edgeNo == 0:
            edge[0], edge[1] = self.cells[cellNo, 0], self.cells[cellNo, 1]
        elif edgeNo == 1:
            edge[0], edge[1] = self.cells[cellNo, 1], self.cells[cellNo, 2]
        elif edgeNo == 2:
            edge[0], edge[1] = self.cells[cellNo, 2], self.cells[cellNo, 0]
        elif edgeNo == 3:
            edge[0], edge[1] = self.cells[cellNo, 0], self.cells[cellNo, 3]
        elif edgeNo == 4:
            edge[0], edge[1] = self.cells[cellNo, 1], self.cells[cellNo, 3]
        else:
            edge[0], edge[1] = self.cells[cellNo, 2], self.cells[cellNo, 3]
        if sorted:
            sortEdge(edge[0], edge[1], edge)

    cdef INDEX_t findFaceInCell(self,
                                INDEX_t cellNo,
                                INDEX_t[::1] face):
        sortFace(self.cells[cellNo, 0], self.cells[cellNo, 1], self.cells[cellNo, 2], self.temp_face)
        if self.temp_face[0] == face[0] and self.temp_face[1] == face[1] and self.temp_face[2] == face[2]:
            return 0
        sortFace(self.cells[cellNo, 0], self.cells[cellNo, 1], self.cells[cellNo, 3], self.temp_face)
        if self.temp_face[0] == face[0] and self.temp_face[1] == face[1] and self.temp_face[2] == face[2]:
            return 1
        sortFace(self.cells[cellNo, 1], self.cells[cellNo, 2], self.cells[cellNo, 3], self.temp_face)
        if self.temp_face[0] == face[0] and self.temp_face[1] == face[1] and self.temp_face[2] == face[2]:
            return 2
        sortFace(self.cells[cellNo, 0], self.cells[cellNo, 2], self.cells[cellNo, 3], self.temp_face)
        if self.temp_face[0] == face[0] and self.temp_face[1] == face[1] and self.temp_face[2] == face[2]:
            return 3
        return -1

    cdef void getFaceInCell(self,
                            INDEX_t cellNo,
                            INDEX_t faceNo,
                            INDEX_t[::1] face,
                            BOOL_t sorted=False):
        if faceNo == 0:
            face[0], face[1], face[2] = self.cells[cellNo, 0], self.cells[cellNo, 2], self.cells[cellNo, 1]
        elif faceNo == 1:
            face[0], face[1], face[2] = self.cells[cellNo, 0], self.cells[cellNo, 1], self.cells[cellNo, 3]
        elif faceNo == 2:
            face[0], face[1], face[2] = self.cells[cellNo, 1], self.cells[cellNo, 2], self.cells[cellNo, 3]
        else:
            face[0], face[1], face[2] = self.cells[cellNo, 2], self.cells[cellNo, 0], self.cells[cellNo, 3]
        if sorted:
            sortFace(face[0], face[1], face[2], face)
