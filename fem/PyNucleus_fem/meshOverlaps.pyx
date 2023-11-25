###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes import INDEX, REAL, TAG, BOOL
from PyNucleus_base import uninitialized, uninitialized_like
from PyNucleus_base.tupleDict cimport tupleDictINDEX
from . meshCy cimport (meshBase,
                       decode_edge, encode_edge,
                       encode_face, decode_face,
                       sortEdge, sortFace,
                       faceVals)
from . DoFMaps cimport DoFMap
from . mesh import mesh1d, mesh2d, mesh3d
from . mesh import INTERIOR_NONOVERLAPPING, INTERIOR, NO_BOUNDARY
from . meshPartitioning import PartitionerException
from . algebraicOverlaps cimport (algebraicOverlapManager,
                                  algebraicOverlap,
                                  algebraicOverlapPersistent,
                                  algebraicOverlapOneSidedPut,
                                  algebraicOverlapOneSidedGet,
                                  algebraicOverlapOneSidedPutLockAll)
from . simplexMapper cimport simplexMapper, simplexMapper2D, simplexMapper3D
from copy import deepcopy
import numpy as np
cimport numpy as np
from numpy.linalg import norm

import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py cimport MPI

from warnings import warn


def boundary1D(meshBase mesh):
    cdef:
        INDEX_t[:, ::1] cells = mesh.cells
        INDEX_t nc = mesh.num_cells
        INDEX_t i, j, k, m
        np.ndarray[INDEX_t, ndim=2] bvertices_mem
        INDEX_t[:, ::1] bvertices_mv
        dict added_vertices

    added_vertices = dict()
    for i in range(nc):
        for j in range(2):
            if cells[i, j] not in added_vertices:
                added_vertices[cells[i, j]] = (i, j)
            else:
                del added_vertices[cells[i, j]]
    bvertices_mem = uninitialized((len(added_vertices), 2), dtype=INDEX)
    bvertices_mv = bvertices_mem
    m = 0
    for k, (i, j) in added_vertices.items():
        bvertices_mv[m, 0] = i
        bvertices_mv[m, 1] = j
        m += 1
    return bvertices_mem


def boundary2D(meshBase mesh, BOOL_t assumeConnected=True):
    cdef:
        simplexMapper sM = mesh.simplexMapper
        INDEX_t[:, ::1] cells = mesh.cells
        INDEX_t nc = mesh.num_cells
        INDEX_t i = 0, k, l
        INDEX_t[::1] e = uninitialized((2), dtype=INDEX)
        np.ndarray[INDEX_t, ndim=2] bvertices_mem
        np.ndarray[INDEX_t, ndim=2] bedges_mem
        INDEX_t[:, ::1] bvertices_mv
        INDEX_t[:, ::1] bedges_mv
        tupleDictINDEX eV = tupleDictINDEX(mesh.num_vertices, deleteHits=True)
        set added_vertices

    for i in range(nc):
        sM.startLoopOverCellEdges(cells[i, :])
        while sM.loopOverCellEdges(e):
            eV.enterValue(e, i)
    bedges_mem = uninitialized((eV.nnz, 2), dtype=INDEX)
    bedges_mv = bedges_mem

    if assumeConnected:
        bvertices_mem = uninitialized((eV.nnz, 2), dtype=INDEX)
        bvertices_mv = bvertices_mem

        k = 0
        l = 0
        added_vertices = set()
        eV.startIter()
        while eV.next(e, &i):
            bedges_mv[k, 0] = i
            bedges_mv[k, 1] = sM.findEdgeInCell(i, e)
            if e[0] not in added_vertices:
                bvertices_mv[l, 0] = i
                bvertices_mv[l, 1] = sM.findVertexInCell(i, e[0])
                added_vertices.add(e[0])
                l += 1
            if e[1] not in added_vertices:
                bvertices_mv[l, 0] = i
                bvertices_mv[l, 1] = sM.findVertexInCell(i, e[1])
                added_vertices.add(e[1])
                l += 1
            k += 1
        if not l == eV.nnz:
            raise PartitionerException('Domain with a hole')
    else:
        k = 0
        l = 0
        added_vertices = set()
        eV.startIter()
        while eV.next(e, &i):
            bedges_mv[k, 0] = i
            bedges_mv[k, 1] = sM.findEdgeInCell(i, e)
            if e[0] not in added_vertices:
                added_vertices.add(e[0])
                l += 1
            if e[1] not in added_vertices:
                added_vertices.add(e[1])
                l += 1
            k += 1

        bvertices_mem = uninitialized((l, 2), dtype=INDEX)
        bvertices_mv = bvertices_mem

        l = 0
        added_vertices = set()
        eV.startIter()
        while eV.next(e, &i):
            if e[0] not in added_vertices:
                bvertices_mv[l, 0] = i
                bvertices_mv[l, 1] = sM.findVertexInCell(i, e[0])
                added_vertices.add(e[0])
                l += 1
            if e[1] not in added_vertices:
                bvertices_mv[l, 0] = i
                bvertices_mv[l, 1] = sM.findVertexInCell(i, e[1])
                added_vertices.add(e[1])
                l += 1
    return bvertices_mem, bedges_mem


def boundary3D(meshBase mesh, BOOL_t assumeConnected=True):
    cdef:
        simplexMapper3D sM = mesh.simplexMapper
        INDEX_t[:, ::1] cells = mesh.cells
        INDEX_t nc = mesh.num_cells
        INDEX_t cellNo = 0, vertexNo, k, l
        INDEX_t[::1] e = uninitialized((2), dtype=INDEX)
        INDEX_t[::1] f = uninitialized((3), dtype=INDEX)
        np.ndarray[INDEX_t, ndim=2] bvertices_mem
        np.ndarray[INDEX_t, ndim=2] bedges_mem
        np.ndarray[INDEX_t, ndim=2] bfaces_mem
        INDEX_t[:, ::1] bvertices_mv
        INDEX_t[:, ::1] bedges_mv
        INDEX_t[:, ::1] bfaces_mv
        tupleDictINDEX eV = tupleDictINDEX(mesh.num_vertices, deleteHits=False)
        faceVals fV = faceVals(mesh.num_vertices, deleteHits=True)
        set added_vertices

    # Get all boundary faces
    for cellNo in range(nc):
        sM.startLoopOverCellFaces(cells[cellNo, :])
        while sM.loopOverCellFaces(f):
            fV.enterValue(f, cellNo)

    # Get all boundary edges
    fV.startIter()
    while fV.next(f, &cellNo):
        sM.startLoopOverFaceEdges(f)
        while sM.loopOverFaceEdges(e):
            eV.enterValue(e, cellNo)

    bfaces_mem = uninitialized((fV.nnz, 2), dtype=INDEX)
    bfaces_mv = bfaces_mem
    bedges_mem = uninitialized((eV.nnz, 2), dtype=INDEX)
    bedges_mv = bedges_mem

    k = 0
    l = 0
    added_vertices = set()
    fV.startIter()
    while fV.next(f, &cellNo):
        bfaces_mv[k, 0] = cellNo
        bfaces_mv[k, 1] = sM.findFaceInCell(cellNo, f)
        for vertexNo in f:
            if vertexNo not in added_vertices:
                added_vertices.add(vertexNo)
                l += 1
        k += 1

    bvertices_mem = uninitialized((l, 2), dtype=INDEX)
    bvertices_mv = bvertices_mem

    l = 0
    added_vertices = set()
    fV.startIter()
    while fV.next(f, &cellNo):
        for vertexNo in f:
            if vertexNo not in added_vertices:
                bvertices_mv[l, 0] = cellNo
                bvertices_mv[l, 1] = sM.findVertexInCell(cellNo, vertexNo)
                added_vertices.add(vertexNo)
                l += 1

    if assumeConnected and not l == 2+eV.nnz-fV.nnz:
        warn('Domain with a hole')

    k = 0
    eV.startIter()
    while eV.next(e, &cellNo):
        bedges_mv[k, 0] = cellNo
        bedges_mv[k, 1] = sM.findEdgeInCell(cellNo, e)
        k += 1
    return bvertices_mem, bedges_mem, bfaces_mem


cdef class dofChecker:
    cdef void add(self, INDEX_t dof):
        pass

    cdef np.ndarray[INDEX_t, ndim=1] getDoFs(self):
        pass


cdef class dofCheckerSet(dofChecker):
    cdef:
        set dofs
        list orderedDoFs

    def __init__(self):
        self.dofs = set()
        self.orderedDoFs = list()

    cdef void add(self, INDEX_t dof):
        if dof >= 0 and dof not in self.dofs:
            self.dofs.add(dof)
            self.orderedDoFs.append(dof)

    cdef np.ndarray[INDEX_t, ndim=1] getDoFs(self):
        return np.array(self.orderedDoFs, dtype=INDEX)


cdef class dofCheckerArray(dofChecker):
    cdef:
        BOOL_t[::1] dofs
        list orderedDoFs

    def __init__(self, INDEX_t num_dofs):
        self.dofs = np.zeros((num_dofs), dtype=BOOL)
        self.orderedDoFs = list()

    cdef void add(self, INDEX_t dof):
        if dof >= 0 and not self.dofs[dof]:
            self.dofs[dof] = True
            self.orderedDoFs.append(dof)

    cdef np.ndarray[INDEX_t, ndim=1] getDoFs(self):
        return np.array(self.orderedDoFs, dtype=INDEX)


cdef class sharedMesh:
    def __init__(self,
                 INDEX_t[:, ::1] vertices,
                 INDEX_t[:, ::1] edges,
                 INDEX_t[:, ::1] faces,
                 INDEX_t[::1] cells,
                 INDEX_t mySubdomainNo,
                 INDEX_t otherSubdomainNo,
                 INDEX_t dim):
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.cells = cells
        self.mySubdomainNo = mySubdomainNo
        self.otherSubdomainNo = otherSubdomainNo
        self.dim = dim

    def get_num_vertices(self):
        return self.vertices.shape[0]

    def get_num_edges(self):
        return self.edges.shape[0]

    def get_num_faces(self):
        return self.faces.shape[0]

    def get_num_cells(self):
        return self.cells.shape[0]

    num_vertices = property(fget=get_num_vertices)
    num_edges = property(fget=get_num_edges)
    num_faces = property(fget=get_num_faces)
    num_cells = property(fget=get_num_cells)

    def __repr__(self):
        return ('Mesh interface of domain {} with {}: ' +
                '{} vertices, {} edges, {} faces, {} cells').format(self.mySubdomainNo,
                                                                    self.otherSubdomainNo,
                                                                    self.num_vertices,
                                                                    self.num_edges,
                                                                    self.num_faces,
                                                                    self.num_cells)

    def refine(self, mesh=None):
        cdef:
            INDEX_t[:, ::1] cells
            simplexMapper3D sM
            INDEX_t i, j, k, cellNo, edgeNo, faceNo, cellNo1, cellNo2, cellNo3, order
            INDEX_t C0, C1, C2
            INDEX_t subcellsPerRef
            INDEX_t middleCellNo, middleFaceNo, middleOrder
            INDEX_t[:, ::1] edges
            INDEX_t[:, ::1] faces
            INDEX_t[::1] face = uninitialized((3), dtype=INDEX)
            INDEX_t[::1] Face = uninitialized((3), dtype=INDEX)
            INDEX_t[::1] perm = uninitialized((3), dtype=INDEX)
            INDEX_t[::1] faceVertexIndices = uninitialized((3), dtype=INDEX)

        if self.dim == 1:
            subcellsPerRef = 2
        elif self.dim == 2:
            subcellsPerRef = 4
        elif self.dim == 3:
            subcellsPerRef = 8
        else:
            raise NotImplementedError()
        new_cells = uninitialized((subcellsPerRef*self.num_cells), dtype=INDEX)
        for i in range(self.num_cells):
            for k in range(subcellsPerRef):
                new_cells[subcellsPerRef*i+k] = subcellsPerRef*self.cells[i]+k
        self.cells = new_cells

        if self.dim == 1:
            for i in range(self.num_vertices):
                self.vertices[i, 0] = 2*self.vertices[i, 0] + self.vertices[i, 1]
        elif self.dim == 2:
            for i in range(self.num_vertices):
                self.vertices[i, 0] = 4*self.vertices[i, 0] + self.vertices[i, 1]
                self.vertices[i, 1] = 0
            edges = uninitialized((2*self.num_edges, 3), dtype=INDEX)
            for i in range(self.num_edges):
                cellNo = self.edges[i, 0]
                edgeNo = self.edges[i, 1]
                order = self.edges[i, 2]
                if order == 0:
                    edges[2*i, 0] = 4*cellNo + edgeNo
                    edges[2*i, 1] = 0
                    edges[2*i, 2] = order
                    edges[2*i+1, 0] = 4*cellNo + (edgeNo+1) % 3
                    edges[2*i+1, 1] = 2
                    edges[2*i+1, 2] = order
                else:
                    edges[2*i+1, 0] = 4*cellNo + edgeNo
                    edges[2*i+1, 1] = 0
                    edges[2*i+1, 2] = order
                    edges[2*i, 0] = 4*cellNo + (edgeNo+1) % 3
                    edges[2*i, 1] = 2
                    edges[2*i, 2] = order
            self.edges = edges
        elif self.dim == 3:
            sM = mesh.simplexMapper
            if self.num_vertices+self.num_edges+self.num_faces == 0:
                return
            cells = mesh.cells
            for i in range(self.num_vertices):
                self.vertices[i, 0] = 8*self.vertices[i, 0] + self.vertices[i, 1]
            edges = uninitialized((2*self.num_edges, 3),
                                  dtype=INDEX)
            for i in range(self.num_edges):
                cellNo = self.edges[i, 0]
                edgeNo = self.edges[i, 1]
                order = self.edges[i, 2]
                if order == 0:
                    if edgeNo < 3:
                        edges[2*i, 0] = 8*cellNo + edgeNo
                        edges[2*i, 1] = edgeNo
                        edges[2*i+1, 0] = 8*cellNo + (edgeNo+1) % 3
                        edges[2*i+1, 1] = edgeNo
                    else:
                        edges[2*i, 0] = 8*cellNo + edgeNo-3
                        edges[2*i, 1] = edgeNo
                        edges[2*i+1, 0] = 8*cellNo + 3
                        edges[2*i+1, 1] = edgeNo
                else:
                    if edgeNo < 3:
                        edges[2*i+1, 0] = 8*cellNo + edgeNo
                        edges[2*i+1, 1] = edgeNo
                        edges[2*i, 0] = 8*cellNo + (edgeNo+1) % 3
                        edges[2*i, 1] = edgeNo
                    else:
                        edges[2*i+1, 0] = 8*cellNo + edgeNo-3
                        edges[2*i+1, 1] = edgeNo
                        edges[2*i, 0] = 8*cellNo + 3
                        edges[2*i, 1] = edgeNo
                edges[2*i:2*i+2, 2] = order
            faces = uninitialized((4*self.num_faces, 3), dtype=INDEX)

            for i in range(self.num_faces):
                cellNo = self.faces[i, 0]
                faceNo = self.faces[i, 1]
                order = self.faces[i, 2]
                # find cellNo and faceNo for middle face
                sM.getFaceVerticesLocal(faceNo, order, faceVertexIndices)
                cellNo1, cellNo2, cellNo3 = faceVertexIndices[0], faceVertexIndices[1], faceVertexIndices[2]
                if faceNo == 0:
                    C0, C1, C2 = cells[8*cellNo, 2], cells[8*cellNo+1, 2], cells[8*cellNo, 1]
                elif faceNo == 1:
                    C0, C1, C2 = cells[8*cellNo, 1], cells[8*cellNo+1, 3], cells[8*cellNo, 3]
                elif faceNo == 2:
                    C0, C1, C2 = cells[8*cellNo+1, 2], cells[8*cellNo+2, 3], cells[8*cellNo+1, 3]
                else:
                    C0, C1, C2 = cells[8*cellNo, 2], cells[8*cellNo, 3], cells[8*cellNo+2, 3]
                faceSet = set([C0, C1, C2])
                sortFace(C0, C1, C2, face)
                for middleCellNo in range(8*cellNo+4, 8*cellNo+8):
                    if faceSet <= set(cells[middleCellNo, :]):
                        middleFaceNo = sM.findFaceInCell(middleCellNo, face)
                        break
                else:
                    raise Exception()

                # face without rotation
                sM.getFaceVerticesGlobal(middleCellNo, middleFaceNo, 0, face)

                # face with rotation
                Face[0], Face[1], Face[2] = ((set(cells[8*cellNo+cellNo1, :]) & set(cells[8*cellNo+cellNo2, :])).pop(),
                                             (set(cells[8*cellNo+cellNo2, :]) & set(cells[8*cellNo+cellNo3, :])).pop(),
                                             (set(cells[8*cellNo+cellNo3, :]) & set(cells[8*cellNo+cellNo1, :])).pop())

                for j in range(3):
                    for k in range(3):
                        if face[k] == Face[j]:
                            perm[j] = k
                            break

                if perm[0] == 0 and perm[1] == 1 and perm[2] == 2:
                    middleOrder = 0
                elif perm[0] == 1 and perm[1] == 2 and perm[2] == 0:
                    middleOrder = 1
                elif perm[0] == 2 and perm[1] == 0 and perm[2] == 1:
                    middleOrder = 2
                elif perm[0] == 1 and perm[1] == 0 and perm[2] == 2:
                    middleOrder = -1
                elif perm[0] == 0 and perm[1] == 2 and perm[2] == 1:
                    middleOrder = -2
                else:
                    middleOrder = -3

                faces[4*i, 0] = 8*cellNo+cellNo1
                faces[4*i, 1] = faceNo
                faces[4*i+1, 0] = 8*cellNo+cellNo2
                faces[4*i+1, 1] = faceNo
                faces[4*i+2, 0] = 8*cellNo+cellNo3
                faces[4*i+2, 1] = faceNo
                faces[4*i:4*i+3, 2] = order
                faces[4*i+3, 0] = middleCellNo
                faces[4*i+3, 1] = middleFaceNo
                faces[4*i+3, 2] = middleOrder
            self.edges = edges
            self.faces = faces
        else:
            raise NotImplementedError()

    def getDoFs(self, meshBase mesh, DoFMap dm, comm, overlapType='standard', INDEX_t numSharedVecs=1,
                BOOL_t allowInteriorBoundary=False):
        cdef:
            simplexMapper sM
            INDEX_t i, j, k, dof, dofOverlap, cellNo, cellNoOverlap, vertexNo, edgeNo, faceNo, edgeOrder, faceOrder
            INDEX_t vertices_per_element, edges_per_element, edgeOffset, faceOffset
            INDEX_t dofs_per_vertex, dofs_per_edge, dofs_per_face #, dofs_per_cell
            INDEX_t[::1] edgeVertexIndices = uninitialized((2), dtype=INDEX)
            INDEX_t[::1] faceVertexIndices = uninitialized((3), dtype=INDEX)
            INDEX_t[::1] faceEdgeIndices = uninitialized((3), dtype=INDEX)
            INDEX_t[::1] faceEdgeOrders = uninitialized((3), dtype=INDEX)
            DoFMap overlapDM
            dofChecker dofCheck
            INDEX_t[::1] overlapCells = self.cells
            INDEX_t[::1] dofs

        dofs_per_vertex = dm.dofs_per_vertex
        dofs_per_edge = dm.dofs_per_edge
        dofs_per_face = dm.dofs_per_face
        # dofs_per_cell = dm.dofs_per_cell
        dofs_per_element = dm.dofs_per_element

        vertices_per_element = mesh.dim+1
        if self.dim == 1:
            edges_per_element = 0
        elif self.dim == 2:
            edges_per_element = 3
        elif self.dim == 3:
            edges_per_element = 6
        else:
            raise NotImplementedError()
        sM = mesh.simplexMapper

        if self.num_cells > 0.01*mesh.num_cells:
            dofCheck = dofCheckerArray(dm.num_dofs)
        else:
            dofCheck = dofCheckerSet()
        if dofs_per_vertex > 0:
            # DoFs associated with cross vertices
            for i in range(self.num_vertices):
                cellNo = self.vertices[i, 0]
                vertexNo = self.vertices[i, 1]
                for k in range(dofs_per_vertex):
                    dof = dm.cell2dof(cellNo, dofs_per_vertex*vertexNo+k)
                    dofCheck.add(dof)
        if dofs_per_vertex+dofs_per_edge > 0:
            edgeOffset = dofs_per_vertex*vertices_per_element
            # DoFs associated with edges and vertices on these edges
            for i in range(self.num_edges):
                cellNo = self.edges[i, 0]
                edgeNo = self.edges[i, 1]
                edgeOrder = self.edges[i, 2]
                # dofs on vertices on shared edges
                sM.getEdgeVerticesLocal(edgeNo, edgeOrder, edgeVertexIndices)
                for j in range(2):
                    for k in range(dofs_per_vertex):
                        dof = dm.cell2dof(cellNo, dofs_per_vertex*edgeVertexIndices[j]+k)
                        dofCheck.add(dof)
                # dofs on shared edges
                if edgeOrder == 0:
                    for k in range(dofs_per_edge):
                        dof = dm.cell2dof(cellNo,
                                          edgeOffset +
                                          dofs_per_edge*edgeNo+k)
                        dofCheck.add(dof)
                else:
                    for k in range(dofs_per_edge-1, -1, -1):
                        dof = dm.cell2dof(cellNo,
                                          edgeOffset +
                                          dofs_per_edge*edgeNo+k)
                        dofCheck.add(dof)
        if dofs_per_vertex+dofs_per_edge+dofs_per_face > 0:
            # DoFs associated with faces and edges and vertices on these faces
            assert dofs_per_face <= 1
            edgeOffset = dofs_per_vertex*vertices_per_element
            faceOffset = dofs_per_vertex*vertices_per_element + edges_per_element*dofs_per_edge
            for i in range(self.num_faces):
                cellNo = self.faces[i, 0]
                faceNo = self.faces[i, 1]
                faceOrder = self.faces[i, 2]
                # dofs on shared vertices
                sM.getFaceVerticesLocal(faceNo, faceOrder, faceVertexIndices)
                for j in range(3):
                    vertexNo = faceVertexIndices[j]
                    for k in range(dofs_per_vertex):
                        dof = dm.cell2dof(cellNo, dofs_per_vertex*vertexNo+k)
                        dofCheck.add(dof)
                # dofs on shared edges
                sM.getFaceEdgesLocal(faceNo, faceOrder, faceEdgeIndices, faceEdgeOrders)
                for j in range(3):
                    edgeNo = faceEdgeIndices[j]
                    edgeOrder = faceEdgeOrders[j]
                    if edgeOrder == 0:
                        for k in range(dofs_per_edge):
                            dof = dm.cell2dof(cellNo,
                                              edgeOffset +
                                              dofs_per_edge*edgeNo+k)
                            dofCheck.add(dof)
                    else:
                        for k in range(dofs_per_edge-1, -1, -1):
                            dof = dm.cell2dof(cellNo,
                                              edgeOffset +
                                              dofs_per_edge*edgeNo+k)
                            dofCheck.add(dof)
                for k in range(dofs_per_face):
                    dof = dm.cell2dof(cellNo,
                                      faceOffset +
                                      dofs_per_face*faceNo+k)
                    dofCheck.add(dof)

        if self.num_cells > 0:
            if self.num_cells < mesh.num_cells:
                # TODO: This is not efficient!
                # build a fake mesh over the overlap
                # Why do we need to do this? Can't we just use the given DoFMap?
                cells = mesh.cells_as_array[self.cells, :]
                vertices = uninitialized((cells.max()+1, self.dim), dtype=REAL)
                if self.dim == 1:
                    overlap_mesh = mesh1d(vertices, cells)
                elif self.dim == 2:
                    overlap_mesh = mesh2d(vertices, cells)
                elif self.dim == 3:
                    overlap_mesh = mesh3d(vertices, cells)
                else:
                    raise NotImplementedError()
            else:
                overlap_mesh = mesh
            # build a DoFMap only over the overlapping region
            if allowInteriorBoundary:
                overlapDM = type(dm)(overlap_mesh, NO_BOUNDARY)
            else:
                overlapDM = type(dm)(overlap_mesh)

            for cellNoOverlap in range(self.num_cells):
                cellNo = overlapCells[cellNoOverlap]
                for k in range(dofs_per_element):
                    dofOverlap = overlapDM.cell2dof(cellNoOverlap, k)
                    if dofOverlap >= 0:
                        dof = dm.cell2dof(cellNo, k)
                        dofCheck.add(dof)

        dofs = dofCheck.getDoFs()
        if overlapType == 'standard':
            return algebraicOverlap(dm.num_dofs,
                                    dofs,
                                    self.mySubdomainNo, self.otherSubdomainNo,
                                    comm, numSharedVecs)
        elif overlapType == 'persistent':
            return algebraicOverlapPersistent(dm.num_dofs,
                                              dofs,
                                              self.mySubdomainNo,
                                              self.otherSubdomainNo,
                                              comm, numSharedVecs)
        elif overlapType == 'oneSidedGet':
            return algebraicOverlapOneSidedGet(dm.num_dofs,
                                               dofs,
                                               self.mySubdomainNo,
                                               self.otherSubdomainNo,
                                               comm, numSharedVecs)
        elif overlapType == 'oneSidedPut':
            return algebraicOverlapOneSidedPut(dm.num_dofs,
                                               dofs,
                                               self.mySubdomainNo,
                                               self.otherSubdomainNo,
                                               comm, numSharedVecs)
        elif overlapType == 'oneSidedPutLockAll':
            return algebraicOverlapOneSidedPutLockAll(dm.num_dofs,
                                                      dofs,
                                                      self.mySubdomainNo,
                                                      self.otherSubdomainNo,
                                                      comm, numSharedVecs)
        else:
            raise NotImplementedError()

    def validate(self, meshBase mesh):
        cdef:
            simplexMapper sM
            list vertices = []
            INDEX_t i, j, cellNo, vertexNo, edgeNo, faceNo, order
            INDEX_t[::1] edgeVertexIndices = uninitialized((2), dtype=INDEX)
            INDEX_t[::1] faceVertexIndices = uninitialized((3), dtype=INDEX)

        sM = mesh.simplexMapper

        offsets = []
        # DoFs associated with cross vertices
        for i in range(self.num_vertices):
            cellNo = self.vertices[i, 0]
            vertexNo = self.vertices[i, 1]
            vertices.append(mesh.vertices[mesh.cells[cellNo, vertexNo], :])
        offsets.append(len(vertices))
        # DoFs associated with edges and vertices on these edges
        for i in range(self.num_edges):
            cellNo = self.edges[i, 0]
            edgeNo = self.edges[i, 1]
            order = self.edges[i, 2]
            sM.getEdgeVerticesLocal(edgeNo, order, edgeVertexIndices)
            for j in range(2):
                vertices.append(mesh.vertices[mesh.cells[cellNo, edgeVertexIndices[j]], :])
        offsets.append(len(vertices))
        # DoFs associated with faces and edges and vertices on these faces
        for i in range(self.num_faces):
            cellNo = self.faces[i, 0]
            faceNo = self.faces[i, 1]
            order = self.faces[i, 2]
            sM.getFaceVerticesLocal(faceNo, order, faceVertexIndices)
            for j in range(3):
                vertices.append(mesh.vertices[mesh.cells[cellNo, faceVertexIndices[j]], :])
        offsets.append(len(vertices))
        for i in range(self.num_cells):
            cellNo = self.cells[i]
            for j in range(self.dim+1):
                vertices.append(mesh.vertices[mesh.cells[cellNo, j], :])
        offsets.append(len(vertices))
        myVertices = np.array(vertices, dtype=REAL)
        return myVertices, np.array(offsets, dtype=INDEX)

    def __getstate__(self):
        return (np.array(self.cells), np.array(self.faces), np.array(self.edges), np.array(self.vertices), self.mySubdomainNo, self.otherSubdomainNo, self.dim)

    def __setstate__(self, state):
        self.cells = state[0]
        self.faces = state[1]
        self.edges = state[2]
        self.vertices = state[3]
        self.mySubdomainNo = state[4]
        self.otherSubdomainNo = state[5]
        self.dim = state[6]

    def sendPartition(self, MPI.Comm comm, INDEX_t[::1] part):
        cdef:
            INDEX_t i, cellNo
            INDEX_t[::1] vertexPart = uninitialized((self.num_vertices), dtype=INDEX)
            INDEX_t[::1] edgePart = uninitialized((self.num_edges), dtype=INDEX)
            INDEX_t[::1] facePart = uninitialized((self.num_faces), dtype=INDEX)
            list requests = []
        for i in range(self.num_vertices):
            cellNo = self.vertices[i, 0]
            vertexPart[i] = part[cellNo]
        if self.num_vertices > 0:
            requests.append(comm.Isend(vertexPart, dest=self.otherSubdomainNo, tag=23))
        for i in range(self.num_edges):
            cellNo = self.edges[i, 0]
            edgePart[i] = part[cellNo]
        if self.num_edges > 0:
            requests.append(comm.Isend(edgePart, dest=self.otherSubdomainNo, tag=24))
        for i in range(self.num_faces):
            cellNo = self.faces[i, 0]
            facePart[i] = part[cellNo]
        if self.num_faces > 0:
            requests.append(comm.Isend(facePart, dest=self.otherSubdomainNo, tag=25))
        return requests

    def recvPartition(self, MPI.Comm comm):
        cdef:
            INDEX_t[::1] vertexPart = uninitialized((self.num_vertices), dtype=INDEX)
            INDEX_t[::1] edgePart = uninitialized((self.num_edges), dtype=INDEX)
            INDEX_t[::1] facePart = uninitialized((self.num_faces), dtype=INDEX)
            list requests = []
        if self.num_vertices > 0:
            requests.append(comm.Irecv(vertexPart, source=self.otherSubdomainNo, tag=23))
        if self.num_edges > 0:
            requests.append(comm.Irecv(edgePart, source=self.otherSubdomainNo, tag=24))
        if self.num_faces > 0:
            requests.append(comm.Irecv(facePart, source=self.otherSubdomainNo, tag=25))
        return requests, vertexPart, edgePart, facePart

    def shrink(self, REAL_t[::1] indicator, INDEX_t[::1] newCellIndices):
        # indicator vector wrt pre-shrink mesh
        # newCellIndices translate from pre-shrink to post-shrink cells
        cdef:
            INDEX_t i
            list cells
        # for i in range(self.vertices.shape[0]):
        #     self.vertices[i, 0] = newCellIndices[self.vertices[i, 0]]
        # for i in range(self.edges.shape[0]):
        #     self.edges[i, 0] = newCellIndices[self.edges[i, 0]]
        # for i in range(self.faces.shape[0]):
        #     self.faces[i, 0] = newCellIndices[self.faces[i, 0]]
        self.vertices = uninitialized((0, 2), dtype=INDEX)
        self.edges = uninitialized((0, 3), dtype=INDEX)
        self.faces = uninitialized((0, 3), dtype=INDEX)
        cells = []
        assert indicator.shape[0] == self.num_cells
        for i in range(self.num_cells):
            if indicator[i] > 1e-4:
                cells.append(newCellIndices[self.cells[i]])
        self.cells = uninitialized((len(cells)), dtype=INDEX)
        for i in range(len(cells)):
            self.cells[i] = cells[i]

    def getAllSharedVertices(self, meshBase mesh):
        """Returns a list of shared vertices, i.e. vertices that are on shared faces, edges as well."""
        cdef:
            simplexMapper sM
            set sharedVerticesSet
            list sharedVertices
            INDEX_t i, j, cellNo, vertexNo, edgeNo, faceNo, order, sVertexNo
            INDEX_t[::1] edgeVertexIndices = uninitialized((2), dtype=INDEX)
            INDEX_t[::1] faceVertexIndices = uninitialized((3), dtype=INDEX)

        sM = mesh.simplexMapper
        sharedVerticesSet = set()
        sharedVertices = []
        for i in range(self.num_vertices):
            cellNo = self.vertices[i, 0]
            vertexNo = self.vertices[i, 1]
            sVertexNo = mesh.cells[cellNo, vertexNo]
            if sVertexNo not in sharedVerticesSet:
                sharedVerticesSet.add(sVertexNo)
                sharedVertices.append((cellNo, vertexNo))
        for i in range(self.num_edges):
            cellNo = self.edges[i, 0]
            edgeNo = self.edges[i, 1]
            order = self.edges[i, 2]
            sM.getEdgeVerticesLocal(edgeNo, order, edgeVertexIndices)
            for j in range(2):
                sVertexNo = mesh.cells[cellNo, edgeVertexIndices[j]]
                if sVertexNo not in sharedVerticesSet:
                    sharedVerticesSet.add(sVertexNo)
                    sharedVertices.append((cellNo, edgeVertexIndices[j]))
        for i in range(self.num_faces):
            cellNo = self.faces[i, 0]
            faceNo = self.faces[i, 1]
            order = self.faces[i, 2]
            sM.getFaceVerticesLocal(faceNo, order, faceVertexIndices)
            for j in range(3):
                sVertexNo = mesh.cells[cellNo, faceVertexIndices[j]]
                if sVertexNo not in sharedVerticesSet:
                    sharedVerticesSet.add(sVertexNo)
                    sharedVertices.append((cellNo, faceVertexIndices[j]))
        return np.array(sharedVertices, dtype=INDEX)


cdef class meshInterface(sharedMesh):
    def __init__(self,
                 INDEX_t[:, ::1] vertices,
                 INDEX_t[:, ::1] edges,
                 INDEX_t[:, ::1] faces,
                 INDEX_t mySubdomainNo,
                 INDEX_t otherSubdomainNo,
                 INDEX_t dim):
        super(meshInterface, self).__init__(vertices, edges, faces, uninitialized((0), dtype=INDEX), mySubdomainNo, otherSubdomainNo, dim)


cdef class sharedMeshManager:
    def __init__(self, MPI.Comm comm):
        assert comm is not None
        self.comm = comm
        self.numSubdomains = comm.size
        self.sharedMeshes = {}
        self.requests = []
        self._rank2subdomain = {rank: rank for rank in range(self.numSubdomains)}
        self._subdomain2rank = {subdomainNo: subdomainNo for subdomainNo in range(self.numSubdomains)}

    cdef inline INDEX_t rank2subdomain(self, INDEX_t rank):
        return self._rank2subdomain[rank]

    cdef inline INDEX_t subdomain2rank(self, INDEX_t subdomainNo):
        return self._subdomain2rank[subdomainNo]

    def refine(self, mesh=None):
        cdef:
            INDEX_t subdomainNo
        for subdomainNo in self.sharedMeshes:
            self.sharedMeshes[subdomainNo].refine(mesh)

    def getDoFs(self, meshBase mesh, DoFMap DoFMap, overlapType='standard', INDEX_t numSharedVecs=1,
                BOOL_t allowInteriorBoundary=False,
                BOOL_t useRequests=True, BOOL_t waitRequests=True,
                splitSharedMeshManager splitManager=None):
        cdef:
            INDEX_t rank, subdomainNo, memSize, totalMemSize
            algebraicOverlap ov
            algebraicOverlapManager OM
            INDEX_t i
            INDEX_t[::1] counts, displ, myMemOffsets, otherMemOffsets
            BOOL_t useOneSided = overlapType in ('oneSidedGet', 'oneSidedPut', 'oneSidedPutLockAll')
            list managers
            sharedMeshManager manager
            INDEX_t k
            INDEX_t[::1] memSizePerManager
            MPI.Win window = None

        OM = algebraicOverlapManager(self.numSubdomains, DoFMap.num_dofs, self.comm)
        OM.type = overlapType
        if splitManager is None:
            managers = [self]
        else:
            managers = splitManager.managers
        memSizePerManager = uninitialized((len(managers)), dtype=INDEX)
        totalMemSize = 0
        for k, manager in enumerate(managers):
            memSize = 0
            for rank in sorted(manager.sharedMeshes):
                subdomainNo = manager.rank2subdomain(rank)
                ov = manager.sharedMeshes[rank].getDoFs(mesh, DoFMap, manager.comm, overlapType, numSharedVecs, allowInteriorBoundary)
                if ov.num_shared_dofs > 0:
                    OM.overlaps[subdomainNo] = ov
                    memSize += ov.num_shared_dofs
            memSizePerManager[k] = memSize
            totalMemSize += memSize

        OM.exchangeIn = np.zeros((numSharedVecs, totalMemSize), dtype=REAL)
        OM.exchangeOut = np.zeros((numSharedVecs, totalMemSize), dtype=REAL)

        for k, manager in enumerate(managers):
            memSize = 0
            if useOneSided:
                window = MPI.Win.Allocate(MPI.REAL.size*numSharedVecs*memSizePerManager[k], comm=manager.comm)
                if overlapType == 'oneSidedPutLockAll':
                    window.Lock_all(MPI.MODE_NOCHECK)
            for rank in sorted(manager.sharedMeshes):
                subdomainNo = manager.rank2subdomain(rank)
                if subdomainNo not in OM.overlaps:
                    continue
                ov = OM.overlaps[subdomainNo]
                ov.setMemory(OM.exchangeIn, OM.exchangeOut, memSize, memSizePerManager[k])
                memSize += ov.num_shared_dofs
                if useOneSided:
                    ov.setWindow(window)
                    if useRequests:
                        req1, req2 = ov.exchangeMemOffsets(self.comm, tag=2012)
                        manager.requests.append(req1)
                        manager.requests.append(req2)
            if useRequests and waitRequests and len(manager.requests) > 0:
                MPI.Request.Waitall(manager.requests)
                manager.requests = []

            if not useRequests and useOneSided:
                myMemOffsets = uninitialized((len(manager.sharedMeshes)), dtype=INDEX)
                otherMemOffsets = uninitialized((len(manager.sharedMeshes)), dtype=INDEX)
                displ = np.zeros((manager.comm.size), dtype=INDEX)
                counts = np.zeros((manager.comm.size), dtype=INDEX)
                for i, rank in enumerate(sorted(manager.sharedMeshes)):
                    counts[rank] = 1
                    displ[rank] = i
                    subdomainNo = manager.rank2subdomain(rank)
                    myMemOffsets[i] = OM.overlaps[subdomainNo].memOffset
                manager.comm.Alltoallv([myMemOffsets, (counts, displ)],
                                       [otherMemOffsets, (counts, displ)])
                for i, rank in enumerate(sorted(manager.sharedMeshes)):
                    subdomainNo = manager.rank2subdomain(rank)
                    OM.overlaps[subdomainNo].memOffsetOther = uninitialized((1), dtype=INDEX)
                    OM.overlaps[subdomainNo].memOffsetOther[0] = otherMemOffsets[i]
        return OM

    def __repr__(self):
        s = ''
        for subdomainNo in self.sharedMeshes:
            s += self.sharedMeshes[subdomainNo].__repr__() + '\n'
        return s

    def __getstate__(self):
        return (self.numSubdomains, self.sharedMeshes)

    def __setstate__(self, state):
        self.numSubdomains = state[0]
        self.sharedMeshes = state[1]

    def copy(self):
        newManager = sharedMeshManager(self.comm)
        for subdomainNo in self.sharedMeshes:
            newManager.sharedMeshes[subdomainNo] = deepcopy(self.sharedMeshes[subdomainNo])
        return newManager

    def exchangePartitioning(self, MPI.Comm comm, INDEX_t[::1] part):
        requests = []
        interfacePart = {subdomainNo: {'vertex': None, 'edge': None, 'face': None} for subdomainNo in self.sharedMeshes}

        for subdomainNo in self.sharedMeshes:
            requests += self.sharedMeshes[subdomainNo].sendPartition(comm, part)
        for subdomainNo in self.sharedMeshes:
            (req,
             interfacePart[subdomainNo]['vertex'],
             interfacePart[subdomainNo]['edge'],
             interfacePart[subdomainNo]['face']) = self.sharedMeshes[subdomainNo].recvPartition(comm)
            requests += req
        MPI.Request.Waitall(requests)
        return interfacePart

    def shrink(self, dict localIndicators, INDEX_t[::1] newCellIndices):
        toDelete = []
        for subdomainNo in self.sharedMeshes:
            try:
                self.sharedMeshes[subdomainNo].shrink(localIndicators[subdomainNo], newCellIndices)
            except KeyError:
                toDelete.append(subdomainNo)
        for subdomainNo in toDelete:
            del self.sharedMeshes[subdomainNo]

    def validate(self, mesh, comm, label='Mesh interface'):
        requests = []
        myVertices = {}
        offsets = {}
        validationPassed = True
        comm.Barrier()
        for subdomainNo in self.sharedMeshes:
            myVertices[subdomainNo], offsets[subdomainNo] = self.sharedMeshes[subdomainNo].validate(mesh)
            if comm.rank > subdomainNo:
                requests.append(comm.Isend(offsets[subdomainNo], dest=subdomainNo, tag=2))
                requests.append(comm.Isend(myVertices[subdomainNo], dest=subdomainNo, tag=1))
        for subdomainNo in self.sharedMeshes:
            if comm.rank < subdomainNo:
                otherOffsets = uninitialized((4), dtype=INDEX)
                comm.Recv(otherOffsets, source=subdomainNo, tag=2)
                otherVertices = uninitialized((otherOffsets[3], mesh.dim), dtype=REAL)
                comm.Recv(otherVertices, source=subdomainNo, tag=1)
                if not np.allclose(offsets[subdomainNo], otherOffsets):
                    validationPassed = False
                    print(('Subdomains {} and {} want' +
                           ' to share {}/{} vertices, ' +
                           '{}/{} edges, {}/{} faces ' +
                           'and {}/{} cells.').format(comm.rank,
                                                      subdomainNo,
                                                      offsets[subdomainNo][0],
                                                      otherOffsets[0],
                                                      (offsets[subdomainNo][1]-offsets[subdomainNo][0])//2,
                                                      (otherOffsets[1]-otherOffsets[0])//2,
                                                      (offsets[subdomainNo][2]-offsets[subdomainNo][1])//3,
                                                      (otherOffsets[2]-otherOffsets[1])//3,
                                                      (offsets[subdomainNo][3]-offsets[subdomainNo][2])//(mesh.dim+1),
                                                      (otherOffsets[3]-otherOffsets[2])//(mesh.dim+1)))
                else:
                    diff = norm(myVertices[subdomainNo]-otherVertices, axis=1)
                    if diff.max() > 1e-9:
                        incorrectVertices = np.sum(diff[:offsets[subdomainNo][0]] > 1e-9)
                        numVertices = offsets[subdomainNo][0]
                        incorrectEdges = int(np.ceil(np.sum(diff[offsets[subdomainNo][0]:offsets[subdomainNo][1]] > 1e-9)/2))
                        numEdges = (offsets[subdomainNo][1]-offsets[subdomainNo][0])//2
                        incorrectFaces = int(np.ceil(np.sum(diff[offsets[subdomainNo][1]:offsets[subdomainNo][2]] > 1e-9)/3))
                        numFaces = (offsets[subdomainNo][2]-offsets[subdomainNo][1])//3
                        incorrectCells = int(np.ceil(np.sum(diff[offsets[subdomainNo][2]:] > 1e-9)/(mesh.dim+1)))
                        numCells = (offsets[subdomainNo][3]-offsets[subdomainNo][2])//(mesh.dim+1)
                        diffSorted = (np.array(sorted(myVertices[subdomainNo], key=tuple))-np.array(sorted(otherVertices, key=tuple)))
                        if np.sum(diff > 1e-9) < 30:
                            s = "\n"
                            s += str(myVertices[subdomainNo][diff > 1e-9, :])
                            s += "\n"
                            s += str(otherVertices[diff > 1e-9, :])
                        else:
                            s = ""
                        print(('Rank {} has incorrect overlap with {} ' +
                               '(Diff sorted: {}, wrong vertices {}/{}, edges {}/{}, ' +
                               'faces {}/{}, cells {}/{}).{}').format(comm.rank,
                                                                      subdomainNo,
                                                                      np.sum(diffSorted),
                                                                      incorrectVertices,
                                                                      numVertices,
                                                                      incorrectEdges,
                                                                      numEdges,
                                                                      incorrectFaces,
                                                                      numFaces,
                                                                      incorrectCells,
                                                                      numCells,
                                                                      s))
                        validationPassed = False
        MPI.Request.Waitall(requests)
        comm.Barrier()
        assert validationPassed
        comm.Barrier()
        if comm.rank == 0:
            print('{} validation successful.'.format(label))


cdef class splitSharedMeshManager:
    cdef:
        public list managers

    cdef list splitCommInteriorBoundary(self, MPI.Comm comm, INDEX_t[::1] commRanks, BOOL_t inCG, INDEX_t numSubComms=2, maxCommSize=100):
        cdef:
            INDEX_t color
            INDEX_t[::1] myRank = uninitialized((1), dtype=INDEX)
            INDEX_t[::1] otherRanks
            INDEX_t[::1] old2new
            INDEX_t[::1] newRanks
            INDEX_t i
            MPI.Comm subcomm
            list commsAndRanks
            MPI.Comm subcommBoundary
        commsAndRanks = []
        myRank[0] = comm.rank

        # group all 'interior' subdomains with their parent
        if inCG:
            color = comm.rank
        elif commRanks.shape[0] == 1:
            color = commRanks[0]
        else:
            color = MPI.UNDEFINED
        subcomm = comm.Split(color)
        if subcomm is not None and subcomm != MPI.COMM_NULL and subcomm.size < comm.size:
            otherRanks = uninitialized((subcomm.size), dtype=INDEX)
            subcomm.Allgather(myRank, otherRanks)
            old2new = -np.ones((comm.size), dtype=INDEX)
            for i in range(subcomm.size):
                old2new[otherRanks[i]] = i
            newRanks = uninitialized((commRanks.shape[0]), dtype=INDEX)
            for i in range(commRanks.shape[0]):
                newRanks[i] = old2new[commRanks[i]]
            commsAndRanks.append((subcomm, newRanks))

        if inCG:
            color = 0
        elif commRanks.shape[0] == 1 and subcomm.size < comm.size:
            color = MPI.UNDEFINED
        else:
            color = 0
        subcommBoundary = comm.Split(color)
        if subcommBoundary == MPI.COMM_NULL:
            return commsAndRanks

        numSubComms = max(numSubComms, subcommBoundary.size//maxCommSize+1)

        # group all 'boundary' subdomains with all coarse grid ranks
        commSplits = np.around(np.linspace(0, subcommBoundary.size, numSubComms+1)).astype(INDEX)
        for splitNo in range(numSubComms):
            if inCG:
                color = 0
            elif commSplits[splitNo] <= subcommBoundary.rank  and subcommBoundary.rank < commSplits[splitNo+1]:
                color = 0
            else:
                color = MPI.UNDEFINED
            subcomm = subcommBoundary.Split(color)
            if subcomm is not None and subcomm != MPI.COMM_NULL:
                otherRanks = uninitialized((subcomm.size), dtype=INDEX)
                subcomm.Allgather(myRank, otherRanks)
                old2new = -np.ones((comm.size), dtype=INDEX)
                for i in range(subcomm.size):
                    old2new[otherRanks[i]] = i
                newRanks = uninitialized((commRanks.shape[0]), dtype=INDEX)
                for i in range(commRanks.shape[0]):
                    newRanks[i] = old2new[commRanks[i]]
                commsAndRanks.append((subcomm, newRanks))

        return commsAndRanks

    cdef list splitComm(self, MPI.Comm comm, INDEX_t[::1] commRanks, BOOL_t inCG, INDEX_t numSubComms=2):
        cdef:
            INDEX_t color
            INDEX_t[::1] myRank = uninitialized((1), dtype=INDEX)
            INDEX_t[::1] otherRanks
            INDEX_t[::1] old2new
            INDEX_t[::1] newRanks
            INDEX_t i, splitNo
            list commsAndRanks

        myRank[0] = comm.rank
        commsAndRanks = []
        commSplits = np.around(np.linspace(0, comm.size, numSubComms+1)).astype(INDEX)
        for splitNo in range(numSubComms):
            if inCG:
                color = 0
            elif commSplits[splitNo] <= comm.rank  and comm.rank < commSplits[splitNo+1]:
                color = 0
            else:
                color = MPI.UNDEFINED
            subcomm = comm.Split(color)

            if subcomm is not None and subcomm != MPI.COMM_NULL:
                otherRanks = uninitialized((subcomm.size), dtype=INDEX)
                subcomm.Allgather(myRank, otherRanks)
                old2new = -np.ones((comm.size), dtype=INDEX)
                for i in range(subcomm.size):
                    old2new[otherRanks[i]] = i
                newRanks = uninitialized((commRanks.shape[0]), dtype=INDEX)
                for i in range(commRanks.shape[0]):
                    newRanks[i] = old2new[commRanks[i]]
            else:
                newRanks = None
            commsAndRanks.append((subcomm, newRanks))
        return commsAndRanks

    def __init__(self, sharedMeshManager manager, MPI.Comm comm, BOOL_t inCG):
        cdef:
            MPI.Comm subcomm
            dict local2global, global2local
        commRanks = np.array([subdomainNo for subdomainNo in sorted(manager.sharedMeshes)], dtype=INDEX)
        commAndRanks = self.splitCommInteriorBoundary(comm, commRanks, inCG)
        # commAndRanks = self.splitComm(comm, commRanks, inCG)
        # print('On rank {}: Split comm of size {} into overlapping subcomms of sizes {}'.format(comm.rank, comm.size,
        #                                                                                        [c[0].size if c[0] != MPI.COMM_NULL else 0 for c in commAndRanks]))
        self.managers = []
        for subcomm, newRanks in commAndRanks:
            if subcomm is not None and subcomm != MPI.COMM_NULL:
                submanager = sharedMeshManager(subcomm)
                local2global = {}
                global2local = {}
                for i in range(commRanks.shape[0]):
                    oldRank = commRanks[i]
                    newRank = newRanks[i]
                    if newRank >= 0:
                        local2global[newRank] = oldRank
                        global2local[oldRank] = newRank
                        sharedMesh = manager.sharedMeshes[oldRank]
                        sharedMesh.mySubdomainNo = subcomm.rank
                        sharedMesh.otherSubdomainNo = newRank
                        submanager.sharedMeshes[newRank] = sharedMesh
                submanager._subdomain2rank = global2local
                submanager._rank2subdomain = local2global
                self.managers.append(submanager)


cdef class interfaceManager(sharedMeshManager):
    def __init__(self, MPI.Comm comm):
        super(interfaceManager, self).__init__(comm)

    def getInterfaces(self):
        return self.sharedMeshes

    def setInterfaces(self, interfaces):
        self.sharedMeshes = interfaces

    interfaces = property(fget=getInterfaces, fset=setInterfaces)

    def copy(self):
        newManager = interfaceManager(self.comm)
        for subdomainNo in self.sharedMeshes:
            newManager.sharedMeshes[subdomainNo] = deepcopy(self.sharedMeshes[subdomainNo])
        return newManager


cdef class meshOverlap(sharedMesh):
    def __init__(self,
                 INDEX_t[::1] cells,
                 INDEX_t mySubdomainNo,
                 INDEX_t otherSubdomainNo,
                 INDEX_t dim):
        super(meshOverlap, self).__init__(uninitialized((0, 2), dtype=INDEX),
                                          uninitialized((0, 3), dtype=INDEX),
                                          uninitialized((0, 3), dtype=INDEX),
                                          cells,
                                          mySubdomainNo, otherSubdomainNo,
                                          dim)


cdef class overlapManager(sharedMeshManager):
    def __init__(self, MPI.Comm comm):
        super(overlapManager, self).__init__(comm)

    def getOverlaps(self):
        return self.sharedMeshes

    def setOverlaps(self, overlaps):
        self.sharedMeshes = overlaps

    overlaps = property(fget=getOverlaps, fset=setOverlaps)

    def check(self, mesh, comm, label='Mesh overlap'):
        self.validate(mesh, comm, label)

    def copy(self):
        newManager = overlapManager(self.comm)
        for subdomainNo in self.sharedMeshes:
            newManager.sharedMeshes[subdomainNo] = deepcopy(self.sharedMeshes[subdomainNo])
        return newManager


cdef class vertexMap:
    cdef:
        public INDEX_t dim, num_vertices, num_interface_vertices
        public dict local2overlap
        public dict overlap2local

    # translates local to overlap vertex indices and the other way around
    def __init__(self, meshBase mesh, sharedMesh interface):
        cdef:
            simplexMapper sM
            INDEX_t k, i, j, cellNo, vertexNo, edgeNo, faceNo, order, vertex
            INDEX_t[::1] edgeVertexIndices = uninitialized((2), dtype=INDEX)
            INDEX_t[::1] faceVertexIndices = uninitialized((3), dtype=INDEX)
        self.local2overlap = dict()
        self.overlap2local = dict()
        self.dim = mesh.dim
        sM = mesh.simplexMapper
        k = 0
        for i in range(interface.num_vertices):
            cellNo = interface.vertices[i, 0]
            vertexNo = interface.vertices[i, 1]
            vertex = mesh.cells[cellNo, vertexNo]
            self.local2overlap[vertex] = k
            self.overlap2local[k] = vertex
            k += 1
        if self.dim >= 2:
            for i in range(interface.num_edges):
                cellNo = interface.edges[i, 0]
                edgeNo = interface.edges[i, 1]
                order = interface.edges[i, 2]
                sM.getEdgeVerticesGlobal(cellNo, edgeNo, order, edgeVertexIndices)
                for j in range(2):
                    self.local2overlap[edgeVertexIndices[j]] = k
                    self.overlap2local[k] = edgeVertexIndices[j]
                    k += 1
        if self.dim >= 3:
            for i in range(interface.num_faces):
                cellNo = interface.faces[i, 0]
                faceNo = interface.faces[i, 1]
                order = interface.faces[i, 2]
                sM.getFaceVerticesGlobal(cellNo, faceNo, order, faceVertexIndices)
                for j in range(3):
                    self.local2overlap[faceVertexIndices[j]] = k
                    self.overlap2local[k] = faceVertexIndices[j]
                    k += 1

        self.num_vertices = k
        self.num_interface_vertices = k

    def translateLocal2Overlap(self, data):
        # used when we send cells
        # Translates subdomain vertex indices into overlap vertex indices.
        # If a subdomain vertex is not yet in the map, we add it.
        cdef:
            INDEX_t i
        dataTranslated = uninitialized_like(data, dtype=INDEX)
        for x, i in zip(np.nditer(dataTranslated, op_flags=['writeonly']),
                        data.flat):
            try:
                x[...] = self.local2overlap[i]
            except KeyError:
                # the vertex with index i is not yet in the map
                # -> add it
                self.local2overlap[i] = self.num_vertices
                self.overlap2local[self.num_vertices] = i
                x[...] = self.num_vertices
                self.num_vertices += 1
        return dataTranslated

    def translateOverlap2Local(self, data, INDEX_t offset=0):
        # used when we receive cells
        # Translates overlap vertex indices into subdomain vertex indices.
        # If a vertex is not on the interface between subdomains, translate with an offset.
        cdef:
            INDEX_t i
        dataTranslated = uninitialized_like(data, dtype=INDEX)
        for x, i in zip(np.nditer(dataTranslated, op_flags=['writeonly']),
                        data.flat):
            if i < self.num_interface_vertices:
                # this vertex was already on the interface
                x[...] = self.overlap2local[i]
            else:
                # this vertex was added in the outer overlap
                x[...] = self.overlap2local[i+offset]
        return dataTranslated


class NotFound(Exception):
    pass


class vertexMapManager:
    def __init__(self, meshBase mesh, interfaceManager, mySubdomainNo):
        self.mySubdomainNo = mySubdomainNo
        self.mesh = mesh
        self.vertexMaps = {}
        # add all vertices that are on the interface
        for subdomainNo in interfaceManager.interfaces:
            self.vertexMaps[subdomainNo] = vertexMap(mesh, interfaceManager.interfaces[subdomainNo])
        # record which subdomains share the interface vertices
        self.sharedVertices = {}
        # loop over all interfaces
        for subdomainNo in self.vertexMaps:
            # loop over vertex indices in interface
            for vertexNo in self.vertexMaps[subdomainNo].local2overlap:
                try:
                    self.sharedVertices[vertexNo].add(subdomainNo)
                except KeyError:
                    self.sharedVertices[vertexNo] = set([subdomainNo])
        # we will keep track of which cells we send out to more than one subdomain
        # in form subdomain cell index -> set of subdomains
        self.sharedCells = {}
        # vertices we receive from other subdomains
        self.newVertices = []
        self.newVerticesShared = {}
        # cells we receive from other subdomains
        self.newCells = []
        self.newCellsLastLayer = []
        self.num_newCells = 0
        self.overlapCellNos = {}
        self.overlapCellsStart = {}

    def translateLocal2Overlap(self, sharedCells, subdomainNo, track=True):
        # Translate subdomain cell indices into an array of overlap vertex indices
        # if track is true, keep track of which subdomains we share each cell with
        cdef:
            INDEX_t[:, ::1] cells = self.mesh.cells
            np.ndarray[INDEX_t, ndim=2] data = uninitialized((len(sharedCells), cells.shape[1]), dtype=INDEX)
            INDEX_t[:, ::1] dataTranslated
            INDEX_t i, j, k
        k = 0
        for i in sharedCells:
            for j in range(cells.shape[1]):
                data[k, j] = cells[i, j]
            k += 1
        dataTranslated = self.vertexMaps[subdomainNo].translateLocal2Overlap(data)
        if track:
            for i, cellNo in enumerate(sharedCells):
                try:
                    self.sharedCells[cellNo].append((subdomainNo, i))
                except KeyError:
                    self.sharedCells[cellNo] = [(subdomainNo, i)]
        return dataTranslated

    def addVertex(self, vertex, subdomainNo):
        # add a vertex that was not part of the subdomain to the vertexMap with another subdomain
        vM = self.vertexMaps[subdomainNo]
        vM.overlap2local[vM.num_vertices] = self.mesh.num_vertices + len(self.newVertices)
        vM.local2overlap[self.mesh.num_vertices + len(self.newVertices)] = vM.num_vertices
        vM.num_vertices += 1
        self.newVertices.append(vertex)

    def addVertexShared(self, vertex, subdomainNo, sharedWith):
        # add a vertex that was not part of the subdomain and that is
        # also shared with other subdomains to the vertexMap with
        # another subdomain
        vM = self.vertexMaps[subdomainNo]
        otherSubdomainNo = list(sharedWith)[0]
        try:
            candidatesLocalVertexNos = self.newVerticesShared[otherSubdomainNo]
            for localVertexNo in candidatesLocalVertexNos:
                if np.linalg.norm(self.newVertices[localVertexNo-self.mesh.num_vertices] - vertex) < 1e-8:
                    vM.overlap2local[vM.num_vertices] = localVertexNo
                    vM.local2overlap[localVertexNo] = vM.num_vertices
                    vM.num_vertices += 1
                    break
            else:
                raise NotFound()
        except (NotFound, KeyError):
            localVertexNo = self.mesh.num_vertices + len(self.newVertices)
            self.addVertex(vertex, subdomainNo)
            try:
                self.newVerticesShared[subdomainNo].append(localVertexNo)
            except KeyError:
                self.newVerticesShared[subdomainNo] = [localVertexNo]

    def getVertexByOverlapIndex(self, subdomainNo, overlapIndex):
        vM = self.vertexMaps[subdomainNo]
        localIndex = vM.overlap2local[overlapIndex]
        return self.getVertexByLocalIndex(localIndex)

    def getVertexByLocalIndex(self, localIndex):
        if localIndex >= self.mesh.num_vertices:
            return self.newVertices[localIndex-self.mesh.num_vertices]
        else:
            return self.mesh.vertices[localIndex, :]

    def removeDuplicateVertices(self):
        temp = np.vstack(self.newVertices)
        _, idx, inverse = np.unique(temp, axis=0, return_index=True, return_inverse=True)
        inverse2 = uninitialized((self.mesh.num_vertices+len(self.newVertices)), dtype=INDEX)
        inverse2[:self.mesh.num_vertices] = np.arange(self.mesh.num_vertices, dtype=INDEX)
        for i in range(len(self.newVertices)):
            inverse2[self.mesh.num_vertices+i] = self.mesh.num_vertices+inverse[i]
        self.newVertices = [self.newVertices[i] for i in idx]
        for i in range(len(self.newCells)):
            for j in range(self.newCells[i].shape[0]):
                for k in range(self.newCells[i].shape[1]):
                    self.newCells[i][j, k] = inverse2[self.newCells[i][j, k]]
        return inverse2

    def addCells(self, cells, subdomainNo, numberCellswithoutLastLevel):
        old_num = self.num_newCells
        # add new cells in layers 1,..,delta
        self.newCells.append(cells[:numberCellswithoutLastLevel, :])
        self.newCells.append(cells[numberCellswithoutLastLevel:, :])
        # self.num_newCells = self.num_newCells+numberCellswithoutLastLevel
        self.num_newCells = self.num_newCells+cells.shape[0]

        # add new cells in layer delta+1
        self.newCellsLastLayer.append(cells[numberCellswithoutLastLevel:, :])

        # subdomain cell indices for cells in layers 1,.., delta
        localCellNos = np.arange(self.mesh.num_cells+old_num,
                                 self.mesh.num_cells+self.num_newCells, dtype=INDEX)

        # add subdomain cell indices
        # make sure that they match up for both subdomains
        if self.mySubdomainNo < subdomainNo:
            self.overlapCellsStart[subdomainNo] = 0
            self.overlapCellNos[subdomainNo] = np.concatenate((localCellNos,
                                                               self.overlapCellNos[subdomainNo]))
        else:
            self.overlapCellsStart[subdomainNo] = self.overlapCellNos[subdomainNo].shape[0]
            self.overlapCellNos[subdomainNo] = np.concatenate((self.overlapCellNos[subdomainNo],
                                                               localCellNos))

    def addSharedCells(self, sharedCells, subdomainNo):
        # Process cells that the subdomain subdomainNo shared with us
        # and other subdomains as well.
        for overlapCellNo, otherSubdomainNo in sharedCells:
            localCellNo = self.overlapCellNos[subdomainNo][self.overlapCellsStart[subdomainNo]+overlapCellNo]
            # FIX: this is not very efficient
            try:
                self.overlapCellNos[otherSubdomainNo] = np.concatenate((self.overlapCellNos[otherSubdomainNo],
                                                                        np.array([localCellNo], dtype=INDEX)))
            except KeyError:
                self.overlapCellNos[otherSubdomainNo] = np.array([localCellNo],
                                                                 dtype=INDEX)

    def writeOverlapToMesh(self):
        self.mesh.vertices = np.vstack((self.mesh.vertices,
                                        np.array(self.newVertices,
                                                 dtype=REAL)))
        self.mesh.cells = np.vstack((self.mesh.cells,
                                     *self.newCells,
                                     # *self.newCellsLastLayer
                                     ))
        self.mesh.init()
        numberCellsLastLayer = sum([len(c) for c in self.newCellsLastLayer])
        self.newVertices = []
        self.newCells = []
        self.newCellsLastLayer = []
        return numberCellsLastLayer


def updateBoundary1D(INDEX_t[:, ::1] cells,
                     INDEX_t[::1] oldVertices):
    cdef:
        INDEX_t nc = cells.shape[0]
        INDEX_t i, c0, c1
        set bvertices
    bvertices = set()
    for i in range(nc):
        c0, c1 = cells[i, 0], cells[i, 1]
        try:
            bvertices.remove(c0)
        except:
            bvertices.add(c0)
        try:
            bvertices.remove(c1)
        except:
            bvertices.add(c1)
    # remove all old (interior) boundary vertices
    for i in range(oldVertices.shape[0]):
        bvertices.discard(oldVertices[i])
    return np.array(list(bvertices), dtype=INDEX)


def updateBoundary2D(INDEX_t[:, ::1] cells,
                     INDEX_t[:, ::1] oldEdges,
                     INDEX_t[::1] oldVertices):
    cdef:
        INDEX_t nc = cells.shape[0]
        INDEX_t c0, c1, i
        ENCODE_t hv = 0
        INDEX_t[::1] e = uninitialized((2), dtype=INDEX)
        np.ndarray[INDEX_t, ndim=2] bedges_mem
        INDEX_t[:, ::1] bedges_mv
        simplexMapper sM = simplexMapper2D()
        set bedges, bvertices

    # get boundary edges for given cells
    bedges = set()
    for i in range(nc):
        sM.startLoopOverCellEdges(cells[i, :])
        while sM.loopOverCellEdgesEncoded(&hv):
            try:
                bedges.remove(hv)
            except KeyError:
                bedges.add(hv)

    # remove all old (interior) edges
    for i in range(oldEdges.shape[0]):
        c0, c1 = oldEdges[i, 0], oldEdges[i, 1]
        sortEdge(c0, c1, e)
        bedges.discard(encode_edge(e))
    bedges_mem = uninitialized((len(bedges), 2), dtype=INDEX)
    bedges_mv = bedges_mem

    # get all new boundary vertices
    bvertices = set()
    for i, hv in enumerate(bedges):
        decode_edge(hv, e)
        bedges_mv[i, 0], bedges_mv[i, 1] = e[0], e[1]
        bvertices.add(e[0])
        bvertices.add(e[1])
    # remove all old (interior) boundary vertices
    for i in range(oldVertices.shape[0]):
        bvertices.discard(oldVertices[i])
    return np.array(list(bvertices), dtype=INDEX), bedges_mem


def updateBoundary3D(INDEX_t[:, ::1] cells,
                     INDEX_t[:, ::1] oldFaces,
                     INDEX_t[:, ::1] oldEdges,
                     INDEX_t[::1] oldVertices):
    cdef:
        INDEX_t nc = cells.shape[0]
        INDEX_t c0, c1, c2, i
        ENCODE_t hv = 0
        INDEX_t[::1] f = uninitialized((3), dtype=INDEX)
        INDEX_t[::1] e = uninitialized((2), dtype=INDEX)
        np.ndarray[INDEX_t, ndim=2] bfaces_mem, bedges_mem
        INDEX_t[:, ::1] bfaces_mv, bedges_mv
        set bfaces, bedges, bvertices
        simplexMapper sM = simplexMapper3D()
        tuple t
        INDEX_t t0 = 0
        ENCODE_t t1 = 0

    # get boundary faces for given cells
    bfaces = set()
    for i in range(nc):
        sM.startLoopOverCellFaces(cells[i, :])
        while sM.loopOverCellFacesEncoded(&t0, &t1):
            try:
                bfaces.remove((t0, t1))
            except KeyError:
                bfaces.add((t0, t1))

    # remove all old (interior) faces
    for i in range(oldFaces.shape[0]):
        c0, c1, c2 = oldFaces[i, 0], oldFaces[i, 1], oldFaces[i, 2]
        sortFace(c0, c1, c2, f)
        bfaces.discard(encode_face(f))
    bfaces_mem = uninitialized((len(bfaces), 3), dtype=INDEX)
    bfaces_mv = bfaces_mem

    # get all new boundary edges
    bedges = set()
    for i, t in enumerate(bfaces):
        decode_face(t, bfaces_mv[i, :])
        sM.startLoopOverFaceEdges(bfaces_mv[i, :])
        while sM.loopOverFaceEdgesEncoded(&hv):
            bedges.add(hv)
    # remove all old (interior) boundary edges
    for i in range(oldEdges.shape[0]):
        bedges.discard(encode_edge(oldEdges[i, :]))
    bedges_mem = uninitialized((len(bedges), 2), dtype=INDEX)
    bedges_mv = bedges_mem

    # get all new boundary vertices
    bvertices = set()
    for i, hv in enumerate(bedges):
        decode_edge(hv, e)
        bedges_mv[i, 0], bedges_mv[i, 1] = e[0], e[1]
        bvertices.add(e[0])
        bvertices.add(e[1])
    # remove all old (interior) boundary vertices
    for i in range(oldVertices.shape[0]):
        bvertices.discard(oldVertices[i])

    return np.array(list(bvertices), dtype=INDEX), bedges_mem, bfaces_mem


def getBoundaryCells(meshBase subdomain, interface, simplexMapper sM, dict v2c):
    cdef:
        set vertices, boundaryCells
        INDEX_t i, cellNo, faceNo, edgeNo, vertexNo, order, v
        INDEX_t dim = subdomain.dim
        INDEX_t[::1] edgeVertexIndices = uninitialized((2), dtype=INDEX)
        INDEX_t[::1] faceVertexIndices = uninitialized((3), dtype=INDEX)

    vertices = set()
    for i in range(interface.num_vertices):
        cellNo = interface.vertices[i, 0]
        vertexNo = interface.vertices[i, 1]
        vertices.add(subdomain.cells[cellNo, vertexNo])
    if dim >= 2:
        for i in range(interface.num_edges):
            cellNo = interface.edges[i, 0]
            edgeNo = interface.edges[i, 1]
            order = interface.edges[i, 2]
            sM.getEdgeVerticesGlobal(cellNo, edgeNo, order, edgeVertexIndices)
            for j in range(2):
                vertices.add(edgeVertexIndices[j])
    if dim >= 3:
        for i in range(interface.num_faces):
            cellNo = interface.faces[i, 0]
            faceNo = interface.faces[i, 1]
            order = interface.faces[i, 2]
            sM.getFaceVerticesGlobal(cellNo, faceNo, order, faceVertexIndices)
            for j in range(3):
                vertices.add(faceVertexIndices[j])

    boundaryCells = set()
    for v in vertices:
        boundaryCells |= set(v2c[v])
    return boundaryCells


def extendOverlap(meshBase subdomain, interfaces, interiorBL, depth, comm, debug=False):
    # extend a (depth+1) layer overlap around each subdomain
    # track depth layers for data exchange
    # the (depth+1) layer should be dropped again after matrix assembly

    cdef:
        simplexMapper sM
        INDEX_t[::1] edgeVertexIndices = uninitialized((2), dtype=INDEX)
        INDEX_t[::1] faceVertexIndices = uninitialized((3), dtype=INDEX)
        INDEX_t dim, nc, k, j, n, I, i, l
        INDEX_t vertexNo, edgeNo, faceNo, cellNo, subdomainNo, localVertexNo, overlapVertexNo
        INDEX_t boundaryVertex
        algebraicOverlap ov
    sM = subdomain.simplexMapper

    dim = subdomain.dim
    nc = subdomain.num_cells
    vMM = vertexMapManager(subdomain, interfaces, comm.rank)
    v2c = interiorBL.vertex2cells(subdomain.cells)
    # dict: subdomainNo -> vertices that are shared with more than one subdomain
    sharedToSend = {}
    # dict: subdomainNo -> vertices that are send there
    verticesToSend = {}
    # dict: subdomainNo -> dict of overlapVertexNo -> tag
    boundaryVertexTagsToSend = {}
    boundaryEdgeTagsToSend = {}
    boundaryFaceTagsToSend = {}
    INVALID_VERTEX_TAG = np.iinfo(TAG).max
    boundaryVertexTagLookup = INVALID_VERTEX_TAG*np.ones((subdomain.num_vertices), dtype=TAG)
    boundaryVertexTagLookup[subdomain.boundaryVertices] = subdomain.boundaryVertexTags

    # mapping vertexNo -> boundaryEdges
    boundaryEdgeLookup = {}
    for edgeNo in range(subdomain.boundaryEdges.shape[0]):
        for k in range(2):
            boundaryVertex = subdomain.boundaryEdges[edgeNo, k]
            try:
                boundaryEdgeLookup[boundaryVertex].append(edgeNo)
            except KeyError:
                boundaryEdgeLookup[boundaryVertex] = [edgeNo]

    # mapping vertexNo -> boundaryFaces
    boundaryFaceLookup = {}
    for faceNo in range(subdomain.boundaryFaces.shape[0]):
        for k in range(3):
            boundaryVertex = subdomain.boundaryFaces[faceNo, k]
            try:
                boundaryFaceLookup[boundaryVertex].append(faceNo)
            except KeyError:
                boundaryFaceLookup[boundaryVertex] = [faceNo]

    # dict: subdomainNo -> cells that are send there
    cellsToSend = {}
    # list of all send requests
    sendRequests = []
    # dict: localVertexNo -> subdomains that that vertex is sent to
    sharedVertices = {}
    for subdomainNo in interfaces.interfaces:
        # indices of cells that touch the interface with the other subdomain
        interfaceBoundaryCellNos = getBoundaryCells(subdomain, interfaces.interfaces[subdomainNo], sM, v2c)
        # indices of cells that are in the interior overlap
        localCellNos = interiorBL.getLayer(depth, interfaceBoundaryCellNos, returnLayerNo=True, cells=subdomain.cells)
        # get layers 1, .., depth
        localCellNos = np.concatenate((localCellNos))
        # add cells in layers 1,..,depth to overlap
        vMM.overlapCellNos[subdomainNo] = localCellNos
        # cells in the interior overlap, in overlap indices
        cellsToSend[subdomainNo] = vMM.translateLocal2Overlap(localCellNos, subdomainNo)
        # number of cells in layers 1,..,depth
        noc = localCellNos.shape[0]
        # number of vertices in layers 1,..,depth
        nv = vMM.vertexMaps[subdomainNo].num_vertices
        # get the local overlap indices and local indices
        # of the vertices that need to be sent
        overlapVertexNosToSend = range(vMM.vertexMaps[subdomainNo].num_interface_vertices,
                                       vMM.vertexMaps[subdomainNo].num_vertices)
        localVertexNosToSend = [vMM.vertexMaps[subdomainNo].overlap2local[k]
                                for k in overlapVertexNosToSend]

        # update list of shared vertices
        for overlapVertexNo in range(vMM.vertexMaps[subdomainNo].num_vertices):
            localVertexNo = vMM.vertexMaps[subdomainNo].overlap2local[overlapVertexNo]
            try:
                sharedVertices[localVertexNo].add(subdomainNo)
            except KeyError:
                sharedVertices[localVertexNo] = set([subdomainNo])

        # get the vertices that need to be sent
        verticesToSend[subdomainNo] = uninitialized((len(localVertexNosToSend), dim), dtype=REAL)
        k = 0
        for vertexNo in localVertexNosToSend:
            for j in range(dim):
                verticesToSend[subdomainNo][k, j] = vMM.mesh.vertices[vertexNo, j]
            k += 1
        # get the boundary information for those vertices
        boundaryVertexTagsToSend[subdomainNo] = {}
        boundaryEdgeTagsToSend[subdomainNo] = {}
        boundaryFaceTagsToSend[subdomainNo] = {}
        for overlapVertexNo in overlapVertexNosToSend:
            localVertexNo = vMM.vertexMaps[subdomainNo].overlap2local[overlapVertexNo]
            vertexTag = boundaryVertexTagLookup[localVertexNo]
            # is this a vertex on the new interface?
            if vertexTag != INVALID_VERTEX_TAG and vertexTag != INTERIOR_NONOVERLAPPING:
                boundaryVertexTagsToSend[subdomainNo][overlapVertexNo] = vertexTag
                if dim >= 2:
                    # find all boundary edges that need to be sent
                    # for edgeNo in range(subdomain.boundaryEdges.shape[0]):
                    for edgeNo in boundaryEdgeLookup[localVertexNo]:
                        edgeTag = subdomain.boundaryEdgeTags[edgeNo]
                        if edgeTag != INTERIOR_NONOVERLAPPING:
                            edge = subdomain.boundaryEdges[edgeNo, :]
                            overlapEdge = vMM.vertexMaps[subdomainNo].translateLocal2Overlap(edge)
                            boundaryEdgeTagsToSend[subdomainNo][tuple(overlapEdge)] = edgeTag
                if dim >= 3:
                    # find all boundary faces that need to be sent
                    # for faceNo in range(subdomain.boundaryFaces.shape[0]):
                    for faceNo in boundaryFaceLookup[localVertexNo]:
                        faceTag = subdomain.boundaryFaceTags[faceNo]
                        if faceTag != INTERIOR_NONOVERLAPPING:
                            face = subdomain.boundaryFaces[faceNo, :]
                            overlapFace = vMM.vertexMaps[subdomainNo].translateLocal2Overlap(face)
                            boundaryFaceTagsToSend[subdomainNo][tuple(overlapFace)] = faceTag

        # send vertices
        num_send = verticesToSend[subdomainNo].shape[0]
        sendRequests.append(comm.isend(num_send,
                                       dest=subdomainNo, tag=0))
        sendRequests.append(comm.Isend(verticesToSend[subdomainNo],
                                       dest=subdomainNo, tag=1))
        sendRequests.append(comm.isend(boundaryVertexTagsToSend[subdomainNo],
                                       dest=subdomainNo, tag=7))
        sendRequests.append(comm.isend(boundaryEdgeTagsToSend[subdomainNo],
                                       dest=subdomainNo, tag=8))
        sendRequests.append(comm.isend(boundaryFaceTagsToSend[subdomainNo],
                                       dest=subdomainNo, tag=9))

        # build dict of vertices that are shared with more than one subdomain
        sharedToSend[subdomainNo] = {}
        for localVertexNo, overlapVertexNo in zip(localVertexNosToSend,
                                                  overlapVertexNosToSend):
            if localVertexNo in vMM.sharedVertices:
                sharedToSend[subdomainNo][overlapVertexNo] = vMM.sharedVertices[localVertexNo]
        # send that information
        sendRequests.append(comm.isend(sharedToSend[subdomainNo],
                                       dest=subdomainNo, tag=2))

        # send cells
        num_send = cellsToSend[subdomainNo].shape[0]
        sendRequests.append(comm.isend(num_send,
                                       dest=subdomainNo, tag=3))
        sendRequests.append(comm.Isend(cellsToSend[subdomainNo],
                                       dest=subdomainNo, tag=4))

        sendRequests.append(comm.isend((nv, noc),
                                       dest=subdomainNo, tag=5))
    del boundaryVertexTagLookup, boundaryEdgeLookup, boundaryFaceLookup

    # Collect cell numbers that are shared with more than one subdomain
    sharedCellsToSend = {subdomainNo: []
                         for subdomainNo in interfaces.interfaces}
    for cellNo in vMM.sharedCells:
        if len(vMM.sharedCells[cellNo]) > 1:
            for subdomainNo, subdomainOverlapCellNo in vMM.sharedCells[cellNo]:
                for otherSubdomainNo, otherSubdomainOverlapCellNo in vMM.sharedCells[cellNo]:
                    if subdomainNo != otherSubdomainNo:
                        sharedCellsToSend[subdomainNo].append((subdomainOverlapCellNo, otherSubdomainNo))
                        # for localVertexNo in subdomain.cells[cellNo, :]:
                        #     sharedVertices[localVertexNo].remove(subdomainNo)

    # Send cell numbers that are shared with another subdomain
    for subdomainNo in interfaces.interfaces:
        sendRequests.append(comm.isend(sharedCellsToSend[subdomainNo],
                                       dest=subdomainNo, tag=6))

    from copy import deepcopy
    sharedVerticesToSend = {subdomainNo: {} for subdomainNo in interfaces.interfaces}
    for vertexNo in sharedVertices:
        if len(sharedVertices[vertexNo]) > 1:
            for subdomainNo in sharedVertices[vertexNo]:
                sharedWith = deepcopy(sharedVertices[vertexNo])
                sharedWith.remove(subdomainNo)
                overlapVertexNo = vMM.vertexMaps[subdomainNo].translateLocal2Overlap(np.array([vertexNo], dtype=INDEX))[0]
                sharedVerticesToSend[subdomainNo][overlapVertexNo] = sharedWith
    del sharedVertices
    for subdomainNo in sharedVerticesToSend:
        sendRequests.append(comm.isend(sharedVerticesToSend[subdomainNo], dest=subdomainNo, tag=10))

    receivedVertexTags = {}
    receivedEdgeTags = {}
    receivedFaceTags = {}

    allReceivedSharedVertices = {}

    for subdomainNo in interfaces.interfaces:
        # receive vertices
        num_receive = comm.recv(source=subdomainNo, tag=0)
        receivedVertices = uninitialized((num_receive, dim), dtype=REAL)
        comm.Recv(receivedVertices, source=subdomainNo, tag=1)

        # receive information about shared vertices
        receivedSharedVertices = comm.recv(source=subdomainNo, tag=2)
        offset = vMM.vertexMaps[subdomainNo].num_vertices - vMM.vertexMaps[subdomainNo].num_interface_vertices

        # add vertices to vertexMapManager
        k = vMM.vertexMaps[subdomainNo].num_interface_vertices
        for vertex in receivedVertices:
            if k in receivedSharedVertices:
                vMM.addVertexShared(vertex, subdomainNo,
                                    receivedSharedVertices[k])
            else:
                vMM.addVertex(vertex, subdomainNo)
            k += 1

        # receive boundaryVertexTags
        vertexDict = comm.recv(source=subdomainNo, tag=7)
        keys = uninitialized((len(vertexDict)), dtype=INDEX)
        values = uninitialized((len(vertexDict)), dtype=TAG)
        for k, (key, value) in enumerate(vertexDict.items()):
            keys[k] = key
            values[k] = value
        if len(vertexDict) > 0:
            # translate receivedVertexTags to refer to local vertices
            keys = vMM.vertexMaps[subdomainNo].translateOverlap2Local(keys, offset)
            receivedVertexTags.update({key: value for key, value in zip(keys, values)})

        # receive boundaryEdgeTags
        edgeDict = comm.recv(source=subdomainNo, tag=8)
        for k, (key, value) in enumerate(edgeDict.items()):
            edge = np.array(key, dtype=INDEX)
            try:
                key = vMM.vertexMaps[subdomainNo].translateOverlap2Local(edge, offset)
                if key[0] > key[1]:
                    key[0], key[1] = key[1], key[0]
                receivedEdgeTags[tuple(key)] = value
            except KeyError:
                pass

        # receive boundaryFaceTags
        faceDict = comm.recv(source=subdomainNo, tag=9)
        for k, (key, value) in enumerate(faceDict.items()):
            face = np.array(key, dtype=INDEX)
            try:
                key = vMM.vertexMaps[subdomainNo].translateOverlap2Local(face, offset)
                sortFace(key[0], key[1], key[2], face)
                key = (face[0], face[1], face[2])
                receivedFaceTags[tuple(key)] = value
            except KeyError:
                pass

        # receive cells
        num_receive = comm.recv(source=subdomainNo, tag=3)
        receivedCells = uninitialized((num_receive, dim+1), dtype=INDEX)
        comm.Recv(receivedCells, source=subdomainNo, tag=4)

        # translate cells to local vertex numbers
        cells = vMM.vertexMaps[subdomainNo].translateOverlap2Local(receivedCells, offset)
        nv, noc = comm.recv(source=subdomainNo, tag=5)

        # add cells to vertexMapManager
        vMM.addCells(cells, subdomainNo, noc)

        # receive shared vertices
        receivedSharedVertices = comm.recv(source=subdomainNo, tag=10)
        receivedSharedVerticesTranslated = {}
        for overlapVertexNo in receivedSharedVertices:
            localVertexNo = vMM.vertexMaps[subdomainNo].translateOverlap2Local(np.array([overlapVertexNo], dtype=INDEX), offset=offset)[0]
            receivedSharedVerticesTranslated[localVertexNo] = receivedSharedVertices[overlapVertexNo]
        allReceivedSharedVertices[subdomainNo] = receivedSharedVerticesTranslated

    # We now have all the vertices and cells that we need to add.
    # Process cells that were also sent out to other subdomains. We
    # process information in order of subdomain number to make sure
    # that all enter the information in the correct order.
    for subdomainNo in sorted(interfaces.interfaces):
        receivedSharedCells = comm.recv(source=subdomainNo, tag=6)
        vMM.addSharedCells(receivedSharedCells, subdomainNo)

    # We change the local vertex indices by removing duplicates.
    # From now on, we need to use vertexMap[localVertexNo]
    vertexMap = vMM.removeDuplicateVertices()

    # adjust vertex tags for removed duplicate vertices
    receivedVertexTags = {vertexMap[vertexNo]: tag for vertexNo, tag in receivedVertexTags.items()}

    # adjust edges tags for removed duplicate vertices
    newReceivedEdgeTags = {}
    for key in receivedEdgeTags:
        tag = receivedEdgeTags[key]
        newKey = (vertexMap[key[0]], vertexMap[key[1]])
        if newKey[0] > newKey[1]:
            newKey = (newKey[1], newKey[0])
        newReceivedEdgeTags[newKey] = tag
    receivedEdgeTags = newReceivedEdgeTags

    # adjust faces tags for removed duplicate vertices
    newReceivedFaceTags = {}
    face = uninitialized((3), dtype=INDEX)
    for key in receivedFaceTags:
        tag = receivedFaceTags[key]
        sortFace(vertexMap[key[0]], vertexMap[key[1]], vertexMap[key[2]], face)
        newKey = (face[0], face[1], face[2])
        newReceivedFaceTags[newKey] = tag
    receivedFaceTags = newReceivedFaceTags

    allOK = True

    if debug:
        # Are we trying to add unique vectors?
        temp = np.vstack(vMM.newVertices)
        uniqVec = np.unique(temp, axis=0)
        if uniqVec.shape[0] != temp.shape[0]:
            print('Subdomain {} tries to add {} vertices, but only {} are unique.'.format(comm.rank, temp.shape[0], uniqVec.shape[0]))
            allOK = False

    # append vertices and cells to mesh
    numberCellsLastLayer = vMM.writeOverlapToMesh()

    if debug:
        # Have we added already present vectors?
        uniqVec = np.unique(subdomain.vertices, axis=0)
        if uniqVec.shape[0] != subdomain.num_vertices:
            print('Subdomain {} has {} vertices, but only {} are unique.'.format(comm.rank, subdomain.num_vertices, uniqVec.shape[0]))
            allOK = False

    # create mesh overlap objects for all overlaps
    overlap = overlapManager(comm)
    for subdomainNo in interfaces.interfaces:
        overlap.overlaps[subdomainNo] = meshOverlap(vMM.overlapCellNos[subdomainNo],
                                                    comm.rank, subdomainNo,
                                                    dim)

    if debug:
        if comm.rank in [8]:
            if np.unique(subdomain.vertices, axis=0).shape[0] < subdomain.num_vertices:
                v, inverse, counts = np.unique(subdomain.vertices, axis=0, return_inverse=True, return_counts=True)
                idx = []
                for i in range(inverse.shape[0]):
                    if counts[inverse[i]] > 1:
                        idx.append(i)
                print(v[counts > 1])
                for i in idx:
                    sharedWith = []
                    for n in overlap.overlaps:
                        ov = overlap.overlaps[n]
                        for k in range(ov.cells.shape[0]):
                            for j in range(subdomain.cells.shape[1]):
                                I = ov.cells[k]
                                if i == subdomain.cells[I, j]:
                                    sharedWith.append(n)
                    print(comm.rank, i, subdomain.vertices[i, :], sharedWith)
        comm.Barrier()

    assert allOK

    # set boundary vertices and boundary vertex tags
    oldBoundaryFaces = subdomain.boundaryFaces
    oldBoundaryEdges = subdomain.boundaryEdges
    oldBoundaryVertices = subdomain.getBoundaryVerticesByTag()
    if subdomain.dim == 1:
        newBverticesOut = updateBoundary1D(subdomain.cells[nc:, :],
                                           oldBoundaryVertices)
        newBedges = None
        newBfaces = None
    elif subdomain.dim == 2:
        newBverticesOut, newBedges = updateBoundary2D(subdomain.cells[nc:, :],
                                                      oldBoundaryEdges,
                                                      oldBoundaryVertices)
        newBfaces = None
    elif subdomain.dim == 3:
        newBverticesOut, newBedges, newBfaces = updateBoundary3D(subdomain.cells[nc:, :],
                                                                 oldBoundaryFaces,
                                                                 oldBoundaryEdges,
                                                                 oldBoundaryVertices)
    else:
        raise NotImplementedError()

    # set boundary vertices and boundary tags
    newBvertexTags = INTERIOR*np.ones((len(newBverticesOut)), dtype=TAG)
    # update the received boundary vertices
    for j in range(len(newBverticesOut)):
        try:
            newBvertexTags[j] = receivedVertexTags[newBverticesOut[j]]
        except KeyError:
            pass
    subdomain.boundaryVertices = np.hstack((oldBoundaryVertices,
                                            newBverticesOut))
    subdomain.boundaryVertexTags = np.hstack((subdomain.boundaryVertexTags,
                                              newBvertexTags))
    del newBverticesOut, newBvertexTags

    if subdomain.dim >= 2:
        # set boundary edges and boundary tags
        newBoundaryEdgeTags = INTERIOR*np.ones((len(newBedges)), dtype=TAG)
        # update the received boundary edges
        for j in range(len(newBedges)):
            try:
                newBoundaryEdgeTags[j] = receivedEdgeTags[(newBedges[j, 0],
                                                           newBedges[j, 1])]
            except KeyError:
                pass
        subdomain.boundaryEdges = np.vstack((oldBoundaryEdges, newBedges))
        subdomain.boundaryEdgeTags = np.hstack((subdomain.boundaryEdgeTags,
                                                newBoundaryEdgeTags))
        del newBedges, newBoundaryEdgeTags

    if subdomain.dim >= 3:
        # set boundary faces and boundary tags
        newBoundaryFaceTags = INTERIOR*np.ones((len(newBfaces)), dtype=TAG)
        # update the received boundary faces
        for j in range(len(newBfaces)):
            try:
                newBoundaryFaceTags[j] = receivedFaceTags[(newBfaces[j, 0],
                                                           newBfaces[j, 1],
                                                           newBfaces[j, 2])]
            except KeyError:
                pass
        subdomain.boundaryFaces = np.vstack((oldBoundaryFaces, newBfaces))
        subdomain.boundaryFaceTags = np.hstack((subdomain.boundaryFaceTags,
                                                newBoundaryFaceTags))
        del newBfaces, newBoundaryFaceTags

    MPI.Request.Waitall(sendRequests)

    sharedVertices = {}
    for subdomainNo in allReceivedSharedVertices:
        for localVertexNo in allReceivedSharedVertices[subdomainNo]:
            localVertexNoTranslated = vertexMap[localVertexNo]
            # find vertices in mesh
            found = False
            i = -1
            j = -1
            for i in range(nc, subdomain.num_cells):
                for j in range(dim+1):
                    if subdomain.cells[i, j] == localVertexNoTranslated:
                        found = True
                        break
                if found:
                    break
            assert found

            for otherSubdomainNo in allReceivedSharedVertices[subdomainNo][localVertexNo]:
                try:
                    sharedVertices[otherSubdomainNo].add((i, j))
                except KeyError:
                    sharedVertices[otherSubdomainNo] = set([(i, j)])
    for subdomainNo in sharedVertices:
        sharedVertices[subdomainNo] = list(sharedVertices[subdomainNo])
        # sort the vertices by coordinates
        key = np.zeros((len(sharedVertices[subdomainNo])), dtype=REAL)
        k = 0
        for i, j in sharedVertices[subdomainNo]:
            for l in range(dim):
                key[k] += subdomain.vertices[subdomain.cells[i, j], l] * 100**l
            k += 1
        idx = key.argsort()
        vertices = uninitialized((len(sharedVertices[subdomainNo]), 2), dtype=INDEX)
        for k in range(len(sharedVertices[subdomainNo])):
            i, j = sharedVertices[subdomainNo][idx[k]]
            vertices[k, 0] = i
            vertices[k, 1] = j
        if subdomainNo not in overlap.overlaps:
            overlap.overlaps[subdomainNo] = meshOverlap(uninitialized((0), dtype=INDEX),
                                                        comm.rank, subdomainNo,
                                                        dim)
        overlap.overlaps[subdomainNo].vertices = vertices

    return overlap, numberCellsLastLayer


def getMeshOverlapsWithOtherPartition(INDEX_t dim,
                                      INDEX_t myRank,
                                      MPI.Comm comm,
                                      INDEX_t[::1] localCellNos,
                                      INDEX_t[::1] partitions):
    cdef:
        INDEX_t numOtherPartitions, localCellNo, globalCellNo, partition
        INDEX_t otherPartition
        list sharedCells
        overlapManager overlaps
        list cells
        INDEX_t numMyPartitions = comm.size
    numOtherPartitions = len(np.unique(partitions))
    sharedCells = [[] for _ in range(numMyPartitions + numOtherPartitions)]
    for localCellNo, globalCellNo in enumerate(localCellNos):
        partition = partitions[globalCellNo]
        sharedCells[partition].append(localCellNo)
    overlaps = overlapManager(comm)
    for otherPartition, cells in enumerate(sharedCells):
        if len(cells)>0:
            overlaps.overlaps[otherPartition] = meshOverlap(np.array(cells, dtype=INDEX), myRank, otherPartition, dim)
    return overlaps
