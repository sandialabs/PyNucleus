###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from __future__ import division
import numpy as np
from PyNucleus_base.myTypes import INDEX
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, ENCODE_t, BOOL_t
from PyNucleus_base import uninitialized
from . meshCy cimport decode_edge, encode_edge, encode_face, decode_face
from . meshCy cimport sortEdge, sortFace, faceVals

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI


cdef class boundaryLayer(object):
    cdef:
        INDEX_t depth
        INDEX_t afterRefinements
        INDEX_t dim
        dict cell_connectivity
        list boundary_vertices
        set boundary_cells
        dict _v2c
        BOOL_t v2c_set_up

    """
    Tracks cell connectivity in a boundary layer.
    """
    def __init__(self, mesh, depth, afterRefinements, INDEX_t startCell=0):
        self.v2c_set_up = False
        self.getBoundaryAndConnectivity(mesh, startCell)
        self.depth = depth
        self.afterRefinements = afterRefinements

    def getBoundaryAndConnectivity(self, mesh, INDEX_t startCell=0):
        """
        Calculate the connectivity and the boundary cells and edges of the
        given cells.
        """
        cdef:
            INDEX_t[:, ::1] cells = mesh.cells[startCell:, :]
            INDEX_t i, c0, c1, c2, c3, j, k, m
            INDEX_t[:, ::1] tempEdge = uninitialized((3, 2), dtype=INDEX)
            INDEX_t[::1] e0 = tempEdge[0, :]
            INDEX_t[::1] e1 = tempEdge[1, :]
            INDEX_t[::1] e2 = tempEdge[2, :]
            INDEX_t[:, ::1] tempFace = uninitialized((4, 3), dtype=INDEX)
            INDEX_t[::1] f0 = tempFace[0, :]
            INDEX_t[::1] f1 = tempFace[1, :]
            INDEX_t[::1] f2 = tempFace[2, :]
            INDEX_t[::1] f3 = tempFace[3, :]
            ENCODE_t he
            faceVals faceLookup
            set boundary_cells
            dict lookup

        self.dim = cells.shape[1]-1
        if self.dim == 1:
            lookup = {}
            self.cell_connectivity = {i: [-1]*2 for i in range(cells.shape[0])}
            for i, c in enumerate(cells):
                c0, c1 = c
                for m, v in enumerate(c):
                    try:
                        j, k = lookup.pop(v)
                        self.cell_connectivity[i][m] = j
                        self.cell_connectivity[j][k] = i
                    except KeyError:
                        lookup[v] = i, m
            self.boundary_vertices = list(lookup.keys())
            self.boundary_cells = set([t[0] for t in lookup.values()])
        elif self.dim == 2:
            lookup = dict()
            self.cell_connectivity = {i: [-1]*3 for i in range(cells.shape[0])}
            for i in range(cells.shape[0]):
                c0, c1, c2 = cells[i, 0], cells[i, 1], cells[i, 2]
                sortEdge(c0, c1, e0)
                sortEdge(c1, c2, e1)
                sortEdge(c2, c0, e2)
                for m in range(3):
                    he = encode_edge(tempEdge[m, :])
                    try:
                        j, k = lookup.pop(he)
                        self.cell_connectivity[i][m] = j
                        self.cell_connectivity[j][k] = i
                    except KeyError:
                        lookup[he] = i, m
            boundary_cells = set()
            for he in lookup:
                boundary_cells.add(lookup[he][0])
            self.boundary_cells = boundary_cells
        elif self.dim == 3:
            faceLookup = faceVals(mesh.num_vertices)
            self.cell_connectivity = {i: [] for i in range(cells.shape[0])}
            for i in range(cells.shape[0]):
                c0, c1, c2, c3 = cells[i, 0], cells[i, 1], cells[i, 2], cells[i, 3]
                sortFace(c0, c1, c2, f0)
                sortFace(c0, c1, c3, f1)
                sortFace(c1, c2, c3, f2)
                sortFace(c2, c0, c3, f3)
                for m in range(4):
                    j = faceLookup.enterValue(tempFace[m, :], i)
                    if i != j:
                        self.cell_connectivity[i].append(j)
                        self.cell_connectivity[j].append(i)
            boundary_cells = set()
            faceLookup.startIter()
            while faceLookup.next(f0, &i):
                boundary_cells.add(i)
            self.boundary_cells = boundary_cells
        else:
            raise NotImplementedError()

    def getLayer(self, INDEX_t depth, set ofCells=None, BOOL_t returnLayerNo=False, INDEX_t[:, ::1] cells=None):
        """
        Returns depth layers of cells that are adjacent to ofCells.
        """
        cdef:
            list bcells
            INDEX_t k, v, i, j, numVerticesPerCell
            dict v2c
            set layerCells, candidates, bc, workset
        bcells = []
        if ofCells is None:
            bcells.append(self.boundary_cells)
        else:
            bcells.append(ofCells)
        if depth == 0:
            bcells = []

        assert cells is not None
        numVerticesPerCell = cells.shape[1]
        v2c = self.vertex2cells(cells)
        for k in range(depth-1):
            candidates = set()
            workset = bcells[k]
            for i in workset:
                for j in range(numVerticesPerCell):
                    v = cells[i, j]
                    candidates.add(v)
            workset = set()
            for v in candidates:
                workset |= set(v2c[v])
            workset -= bcells[k]
            if k > 0:
                workset -= bcells[k-1]
            bcells.append(workset)
        if not returnLayerNo:
            layerCells = set()
            for i in range(len(bcells)):
                layerCells |= bcells[i]
            return np.array(list(layerCells), dtype=INDEX)
        else:
            return [np.array(list(bc), dtype=INDEX) for bc in bcells]

    def prune(self, depth, pp=None, cells=None):
        """
        Remove cells that are to far from the boundary.
        """
        cdef:
            INDEX_t i, j
            INDEX_t[::1] layerCells
            dict new_cell_connectivity = {}
        layerCells = self.getLayer(depth, pp, cells=cells)
        for i in layerCells:
            new_cell_connectivity[i] = self.cell_connectivity.pop(i)
        new_cell_connectivity[-1] = [-1]*(self.dim+1)
        for i in new_cell_connectivity:
            for j in range(self.dim+1):
                try:
                    new_cell_connectivity[new_cell_connectivity[i][j]]
                except KeyError:
                    new_cell_connectivity[i][j] = -2
        new_cell_connectivity.pop(-1)
        self.cell_connectivity = new_cell_connectivity

    def refine(self, newMesh):
        """
        Refine the boundary layers.
        """
        cdef:
            INDEX_t i, j, posJ, k, posK, l, posL
            dict new_cell_connectivity
            set new_bcells
            INDEX_t[:, ::1] newCells = newMesh.cells
            INDEX_t c0, c1, c2, c3, cellNo, subcellNo, otherSubCellNo
            INDEX_t[:, ::1] tempEdge = uninitialized((3, 2), dtype=INDEX)
            INDEX_t[::1] e0 = tempEdge[0, :]
            INDEX_t[::1] e1 = tempEdge[1, :]
            INDEX_t[::1] e2 = tempEdge[2, :]
            INDEX_t[:, ::1] tempFace = uninitialized((4, 3), dtype=INDEX)
            INDEX_t[::1] f0 = tempFace[0, :]
            INDEX_t[::1] f1 = tempFace[1, :]
            INDEX_t[::1] f2 = tempFace[2, :]
            INDEX_t[::1] f3 = tempFace[3, :]
            faceVals faceLookup
            set pp

        new_cell_connectivity = {}
        new_bcells = set()
        if self.dim == 1:
            for i in self.cell_connectivity:
                new_cell_connectivity[2*i] = [-2]*2
                new_cell_connectivity[2*i+1] = [-2]*2
                j, k = self.cell_connectivity[i][:]
                new_cell_connectivity[2*i][1] = 2*i+1
                new_cell_connectivity[2*i+1][0] = 2*i
                if j > -1:
                    posJ = self.cell_connectivity[j].index(i)
                    new_cell_connectivity[2*i][0] = 2*j+posJ
                elif j == -1:
                    new_cell_connectivity[2*i][0] = -1
                    new_bcells.add(2*i)
                if k > -1:
                    posK = self.cell_connectivity[k].index(i)
                    new_cell_connectivity[2*i+1][1] = 2*k+posK
                elif k == -1:
                    new_cell_connectivity[2*i+1][1] = -1
                    new_bcells.add(2*i+1)
        elif self.dim == 2:
            for i in self.cell_connectivity:
                new_cell_connectivity[4*i] = [-2]*3
                new_cell_connectivity[4*i+1] = [-2]*3
                new_cell_connectivity[4*i+2] = [-2]*3
                new_cell_connectivity[4*i+3] = [4*i+1, 4*i+2, 4*i]
                new_cell_connectivity[4*i][1] = 4*i+3
                new_cell_connectivity[4*i+1][1] = 4*i+3
                new_cell_connectivity[4*i+2][1] = 4*i+3
                j, k, l = self.cell_connectivity[i][:]

                # is there an adjacent cell?
                if j > -1:
                    # on which edge is this cell adjacent in the other cell?
                    posJ = self.cell_connectivity[j].index(i)
                    new_cell_connectivity[4*i+1][2] = 4*j+posJ
                    posJ = (1+posJ) % 3
                    new_cell_connectivity[4*i][0] = 4*j+posJ
                elif j == -1:
                    new_cell_connectivity[4*i+1][2] = -1
                    new_cell_connectivity[4*i][0] = -1
                    new_bcells.add(4*i)
                    new_bcells.add(4*i+1)

                # is there an adjacent cell?
                if k > -1:
                    posK = self.cell_connectivity[k].index(i)
                    new_cell_connectivity[4*i+2][2] = 4*k+posK
                    posK = (1+posK) % 3
                    new_cell_connectivity[4*i+1][0] = 4*k+posK
                elif k == -1:
                    new_cell_connectivity[4*i+2][2] = -1
                    new_cell_connectivity[4*i+1][0] = -1
                    new_bcells.add(4*i+1)
                    new_bcells.add(4*i+2)

                # is there an adjacent cell?
                if l > -1:
                    posL = self.cell_connectivity[l].index(i)
                    new_cell_connectivity[4*i][2] = 4*l+posL
                    posL = (1+posL) % 3
                    new_cell_connectivity[4*i+2][0] = 4*l+posL
                elif l == -1:
                    new_cell_connectivity[4*i][2] = -1
                    new_cell_connectivity[4*i+2][0] = -1
                    new_bcells.add(4*i+2)
                    new_bcells.add(4*i)
        elif self.dim == 3:
            faceLookup = faceVals(newMesh.num_vertices)
            for cellNo in self.cell_connectivity:
                for subcellNo in range(8*cellNo, 8*cellNo+8):
                    c0, c1, c2, c3 = newCells[subcellNo, 0], newCells[subcellNo, 1], newCells[subcellNo, 2], newCells[subcellNo, 3]
                    sortFace(c0, c1, c2, f0)
                    sortFace(c0, c1, c3, f1)
                    sortFace(c1, c2, c3, f2)
                    sortFace(c2, c0, c3, f3)
                    new_cell_connectivity[subcellNo] = []
                    for m in range(4):
                        otherSubCellNo = faceLookup.enterValue(tempFace[m, :], subcellNo)
                        if otherSubCellNo != subcellNo:
                            new_cell_connectivity[subcellNo].append(otherSubCellNo)
                            new_cell_connectivity[otherSubCellNo].append(subcellNo)
            faceLookup.startIter()
            while faceLookup.next(f0, &subcellNo):
                # cellNo = subcellNo//8
                cellNo = subcellNo>>3
                if cellNo in self.boundary_cells:
                    new_bcells.add(subcellNo)
                    new_cell_connectivity[subcellNo].append(-1)
                else:
                    new_cell_connectivity[subcellNo].append(-1)
        self.cell_connectivity = new_cell_connectivity
        self.boundary_cells = new_bcells

        depth = int(np.ceil(<REAL_t>self.depth/<REAL_t>self.afterRefinements))
        if self.v2c_set_up:
            self.v2c_set_up = False
        v2c = self.vertex2cells(newMesh.cells)

        pp = set()
        for i in set(newMesh.boundaryVertices):
            pp |= set(v2c[i])

        self.prune(depth, pp, cells=newMesh.cells)
        self.afterRefinements -= 1

    def vertex2cells(self, const INDEX_t[:, ::1] cells):
        """
        Return a lookup dict
        vertex no -> cell no
        """
        cdef:
            dict v2c
            INDEX_t i, j, k
            INDEX_t numVerticesPerCell = cells.shape[1]
        if self.v2c_set_up:
            return self._v2c
        else:
            v2c = {}
            for i in self.cell_connectivity:
                for k in range(numVerticesPerCell):
                    j = cells[i, k]
                    try:
                        v2c[j].append(i)
                    except KeyError:
                        v2c[j] = [i]
            self._v2c = v2c
            self.v2c_set_up = True
            return v2c
