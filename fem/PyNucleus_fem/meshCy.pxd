###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, ENCODE_t, BOOL_t
from PyNucleus_base.intTuple cimport intTuple
from PyNucleus_base.tupleDict cimport tupleDictINDEX
from PyNucleus_base.linear_operators cimport sparseGraph
from PyNucleus_base.intTuple cimport productIterator
cimport numpy as np
from . simplexMapper cimport simplexMapper, simplexMapper1D, simplexMapper2D, simplexMapper3D

ctypedef REAL_t[:, ::1] vertices_t
ctypedef INDEX_t[:, ::1] cells_t


cdef class meshTransformer:
    pass


cdef class gradedMeshTransformer(meshTransformer):
    cdef:
        REAL_t mu, mu2, radius


cdef class multiIntervalMeshTransformer(meshTransformer):
    cdef:
        list intervals


cdef class meshBase:
    cdef:
        public vertices_t vertices
        public cells_t cells
        readonly INDEX_t num_vertices, num_cells, dim, manifold_dim
        REAL_t _h, _delta, _volume, _hmin
        REAL_t[::1] _volVector
        REAL_t[::1] _hVector
        public simplexMapper simplexMapper
        public meshTransformer transformer
    cdef void computeMeshQuantities(self)
    cdef void getSimplex(self,
                         const INDEX_t cellIdx,
                         REAL_t[:, ::1] simplex)
    cdef BOOL_t vertexInCell(self, REAL_t[::1] vertex,
                             INDEX_t cellNo,
                             REAL_t[:, ::1] simplexMem,
                             REAL_t[::1] baryMem,
                             REAL_t tol=*)
    cdef BOOL_t vertexInCellPtr(self, REAL_t* vertex,
                                INDEX_t cellNo,
                                REAL_t[:, ::1] simplexMem,
                                REAL_t[::1] baryMem,
                                REAL_t tol=*)


cdef void decode_edge(ENCODE_t encodeVal, INDEX_t[::1] e)

cdef void vectorProduct(const REAL_t[::1] v, const REAL_t[::1] w, REAL_t[::1] z)

cdef REAL_t volume0D(REAL_t[:, ::1] span)
cdef REAL_t volume1D(REAL_t[::1] v0)
cdef REAL_t volume1Dnew(REAL_t[:, ::1] span)
cdef REAL_t volume1D_in_2D(REAL_t[:, ::1] span)
cdef REAL_t volume2D(REAL_t[::1] v0, REAL_t[::1] v1)
cdef REAL_t volume2Dnew(REAL_t[:, ::1] span)
cdef REAL_t volume2D_in_3D(REAL_t[::1] v0, REAL_t[::1] v1)
cdef REAL_t volume3D(REAL_t[:, ::1] span)
cdef REAL_t volume3Dnew(REAL_t[:, ::1] span, REAL_t[::1] temp)
cdef REAL_t volume2D_in_3Dnew(REAL_t[:, ::1] span)

cdef REAL_t volume0Dsimplex(REAL_t[:, ::1] simplex)
cdef REAL_t volume1Dsimplex(REAL_t[:, ::1] simplex)
cdef REAL_t volume2Dsimplex(REAL_t[:, ::1] simplex)
cdef REAL_t volume1Din2Dsimplex(REAL_t[:, ::1] simplex)
cdef REAL_t volume3Dsimplex(REAL_t[:, ::1] simplex)
cdef REAL_t volume2Din3Dsimplex(REAL_t[:, ::1] simplex)

cdef ENCODE_t encode_edge(INDEX_t[::1] e)
cdef void sortEdge(INDEX_t c0, INDEX_t c1, INDEX_t[::1] e)

cdef void sortFace(INDEX_t c0, INDEX_t c1, INDEX_t c2, INDEX_t[::1] f)
cdef tuple encode_face(INDEX_t[::1] f)
cdef void decode_face(tuple encodeVal, INDEX_t[::1] f)


cdef class faceVals:
    cdef:
        INDEX_t ** indexL
        INDEX_t ** indexR
        INDEX_t ** vals
        np.uint8_t[::1] counts
        np.uint8_t initial_length
        np.uint8_t length_inc
        np.uint8_t[::1] lengths
        INDEX_t num_dofs, nnz
        BOOL_t deleteHits
        INDEX_t i, jj
    cdef inline INDEX_t enterValue(self, const INDEX_t[::1] f, INDEX_t val)
    cdef inline INDEX_t getValue(self, const INDEX_t[::1] f)
    cdef void startIter(self)
    cdef BOOL_t next(self, INDEX_t[::1] f, INDEX_t * val)


cdef class cellFinder(object):
    cdef:
        meshBase mesh
        public REAL_t[:, ::1] simplex
        public REAL_t[::1] bary
        tuple kd
        INDEX_t numCandidates
    cdef INDEX_t findCell(self, REAL_t[::1] vertex)


cdef class cellFinder2:
    cdef:
        meshBase mesh
        REAL_t[::1] diamInv, x_min
        faceVals lookup
        REAL_t[:, ::1] simplex
        REAL_t[::1] bary
        INDEX_t[::1] key, key2
        sparseGraph graph
        sparseGraph v2c
        INDEX_t[::1] candidates
        productIterator pit
    cdef INDEX_t findCell(self, REAL_t[::1] vertex)
    cdef INDEX_t findCellPtr(self, REAL_t* vertex)

cdef void getBarycentricCoords1D(REAL_t[:, ::1] simplex, REAL_t[::1] x, REAL_t[::1] bary)
cdef void getBarycentricCoords2D(REAL_t[:, ::1] simplex, REAL_t[::1] x, REAL_t[::1] bary)
