###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, ENCODE_t, BOOL_t

cdef class simplexMapper:
    cdef:
        REAL_t[:, ::1] vertices
        INDEX_t[:, ::1] cells
        INDEX_t[::1] temp_cell
        INDEX_t[::1] temp_edge
        INDEX_t[::1] temp_edge2
        INDEX_t[::1] temp_face
        INDEX_t[::1] temp_face2
        INDEX_t[:, ::1] temp_edges
        INDEX_t[:, ::1] temp_faces
        INDEX_t[:, ::1] temp_edges2
        INDEX_t iteration_counter
        INDEX_t dim

    cdef void startLoopOverCellNodes(self, INDEX_t[::1] cell)
    cdef BOOL_t loopOverCellNodes(self, INDEX_t *node)
    cdef void startLoopOverCellEdges(self, INDEX_t[::1] cell)
    cdef BOOL_t loopOverCellEdges(self, INDEX_t[::1] edge)
    cdef BOOL_t loopOverCellEdgesEncoded(self, ENCODE_t * hv)
    cdef void startLoopOverCellFaces(self, INDEX_t[::1] cell)
    cdef BOOL_t loopOverCellFaces(self, INDEX_t[::1] face)
    cdef BOOL_t loopOverCellFacesEncoded(self, INDEX_t * t0, ENCODE_t * t1)
    cdef void startLoopOverFaceEdges(self, INDEX_t[::1] face)
    cdef BOOL_t loopOverFaceEdges(self, INDEX_t[::1] edge)
    cdef BOOL_t loopOverFaceEdgesEncoded(self, ENCODE_t * hv)
    cdef INDEX_t getVertexInCell(self, INDEX_t cellNo, INDEX_t vertexNo)
    cdef INDEX_t findVertexInCell(self, INDEX_t cellNo, INDEX_t vertexNo)
    cdef void getEdgeVerticesLocal(self,
                                   INDEX_t edgeNo,
                                   INDEX_t order,
                                   INDEX_t[::1] indices)
    cdef void getFaceVerticesLocal(self,
                                   INDEX_t faceNo,
                                   INDEX_t order,
                                   INDEX_t[::1] indices)
    cdef void getFaceEdgesLocal(self,
                                INDEX_t faceNo,
                                INDEX_t order,
                                INDEX_t[::1] indices,
                                INDEX_t[::1] orders)
    cdef void getEdgeVerticesGlobal(self,
                                    INDEX_t cellNo,
                                    INDEX_t edgeNo,
                                    INDEX_t order,
                                    INDEX_t[::1] indices)
    cdef void getFaceVerticesGlobal(self,
                                    INDEX_t cellNo,
                                    INDEX_t faceNo,
                                    INDEX_t order,
                                    INDEX_t[::1] indices)
    cdef ENCODE_t sortAndEncodeEdge(self, INDEX_t[::1] edge)
    cdef INDEX_t findEdgeInCell(self,
                                INDEX_t cellNo,
                                INDEX_t[::1] edge)
    cdef INDEX_t findEdgeInCellEncoded(self,
                                       INDEX_t cellNo,
                                       ENCODE_t hv)
    cdef void getEdgeInCell(self,
                            INDEX_t cellNo,
                            INDEX_t edgeNo,
                            INDEX_t[::1] edge,
                            BOOL_t sorted=*)
    cdef ENCODE_t getEdgeInCellEncoded(self,
                                       INDEX_t cellNo,
                                       INDEX_t edgeNo)
    cdef tuple sortAndEncodeFace(self, INDEX_t[::1] face)
    cdef INDEX_t findFaceInCell(self,
                                INDEX_t cellNo,
                                const INDEX_t[::1] face)
    cdef void getFaceInCell(self,
                            INDEX_t cellNo,
                            INDEX_t faceNo,
                            INDEX_t[::1] face,
                            BOOL_t sorted=*)
    cdef tuple getFaceInCellEncoded(self,
                                    INDEX_t cellNo,
                                    INDEX_t faceNo)
    cdef INDEX_t findFaceInCellEncoded(self,
                                       INDEX_t cellNo,
                                       tuple hv)
    cdef void getEdgeSimplex(self,
                             INDEX_t cellNo,
                             INDEX_t edgeNo,
                             REAL_t[:, ::1] edgeSimplex)
    cdef void getEncodedEdgeSimplex(self,
                                    ENCODE_t hv,
                                    REAL_t[:, ::1] edgeSimplex)
    cdef void getFaceSimplex(self,
                             INDEX_t cellNo,
                             INDEX_t faceNo,
                             REAL_t[:, ::1] faceSimplex)
    cdef void getEncodedFaceSimplex(self,
                                    tuple hv,
                                    REAL_t[:, ::1] faceSimplex)


cdef class simplexMapper1D(simplexMapper):
    pass


cdef class simplexMapper2D(simplexMapper):
    pass


cdef class simplexMapper3D(simplexMapper):
    pass
