###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport isnan
from PyNucleus_base.myTypes import INDEX, REAL, COMPLEX, BOOL
from cpython cimport Py_buffer
from PyNucleus_base.blas cimport assign, assign3, assignScaled, matmat
from PyNucleus_base.ip_norm cimport vector_t, ip_serial, norm_serial, wrapRealInnerToComplex, wrapRealNormToComplex
from PyNucleus_base import uninitialized
from PyNucleus_base.sparsityPattern cimport sparsityPattern
from PyNucleus_base.linear_operators cimport LinearOperator
from PyNucleus_base.tupleDict cimport tupleDictINDEX
from . meshCy cimport (sortEdge, sortFace,
                       encode_edge,
                       faceVals,
                       getBarycentricCoords1D,
                       getBarycentricCoords2D,
                       cellFinder, cellFinder2,
                       intTuple)
from . meshCy import getSubmesh2
from . quadrature cimport simplexQuadratureRule, simplexXiaoGimbutas
from PyNucleus_base.linear_operators cimport (CSR_LinearOperator,
                                              SSS_LinearOperator,
                                              sparseGraph)
from PyNucleus_base.sparseGraph import cuthill_mckee

cdef INDEX_t MAX_INT = np.iinfo(INDEX).max


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline REAL_t evalP0(REAL_t[:, ::1] simplex, REAL_t[::1] uloc, REAL_t[::1] x):
    return uloc[0]


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline REAL_t evalP12D(REAL_t[:, ::1] simplex, REAL_t[::1] uloc, REAL_t[::1] x):
    cdef:
        REAL_t vol, l3, res, l
        INDEX_t i
    vol = ((simplex[0, 0]-simplex[1, 0])*(simplex[2, 1]-simplex[1, 1]) -
           (simplex[0, 1]-simplex[1, 1])*(simplex[2, 0]-simplex[1, 0]))
    l3 = vol
    res = 0.
    for i in range(2):
        l = ((x[0]-simplex[(i+1) % 3, 0])*(simplex[(i+2) % 3, 1]-simplex[(i+1) % 3, 1]) -
             (x[1]-simplex[(i+1) % 3, 1])*(simplex[(i+2) % 3, 0]-simplex[(i+1) % 3, 0]))
        l3 -= l
        res += uloc[i]*l
    res += uloc[2]*l3
    return res/vol


include "vector_REAL.pxi"
include "vector_COMPLEX.pxi"


cdef class DoFMap:
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    def __init__(self,
                 meshBase mesh,
                 INDEX_t dofs_per_vertex,
                 INDEX_t dofs_per_edge,
                 INDEX_t dofs_per_face,
                 INDEX_t dofs_per_cell,
                 tag=None,
                 INDEX_t skipCellsAfter=-1):

        cdef:
            INDEX_t vertices_per_element
            INDEX_t edges_per_element
            INDEX_t faces_per_element
            INDEX_t manifold_dim = mesh.manifold_dim

        self.mesh = mesh
        self.dim = self.mesh.dim

        if isinstance(tag, function):
            self.tag = [-10]
        elif isinstance(tag, list):
            self.tag = tag
        else:
            self.tag = [tag]
        if manifold_dim == 0:
            vertices_per_element = 1
            edges_per_element = 0
            faces_per_element = 0
        elif manifold_dim == 1:
            vertices_per_element = 2
            edges_per_element = 0
            faces_per_element = 0
        elif manifold_dim == 2:
            vertices_per_element = 3
            edges_per_element = 3
            faces_per_element = 0
        elif manifold_dim == 3:
            vertices_per_element = 4
            edges_per_element = 6
            faces_per_element = 4
        else:
            raise NotImplementedError()

        self.dofs_per_vertex = dofs_per_vertex
        if edges_per_element > 0:
            self.dofs_per_edge = dofs_per_edge
        else:
            self.dofs_per_edge = 0
        if faces_per_element > 0:
            self.dofs_per_face = dofs_per_face
        else:
            self.dofs_per_face = 0
        self.dofs_per_cell = dofs_per_cell
        self.dofs_per_element = (vertices_per_element*dofs_per_vertex +
                                 edges_per_element*dofs_per_edge +
                                 faces_per_element*dofs_per_face +
                                 dofs_per_cell)

        cdef:
            INDEX_t[:, ::1] cells = mesh.cells
            INDEX_t nc = cells.shape[0]
            INDEX_t c0, c1, c2, c3, i, j, k, dof, numDoFs, numBdofs, v
            INDEX_t[:, ::1] temp = uninitialized((6, 2), dtype=INDEX)
            INDEX_t[::1] e01 = temp[0, :]
            INDEX_t[::1] e12 = temp[1, :]
            INDEX_t[::1] e20 = temp[2, :]
            INDEX_t[::1] e03 = temp[3, :]
            INDEX_t[::1] e13 = temp[4, :]
            INDEX_t[::1] e23 = temp[5, :]
            INDEX_t[::1] edgeOrientations = uninitialized((edges_per_element), dtype=INDEX)
            INDEX_t[:, ::1] temp2 = uninitialized((4, 3), dtype=INDEX)
            INDEX_t[::1] f012 = temp2[0, :]
            INDEX_t[::1] f013 = temp2[1, :]
            INDEX_t[::1] f123 = temp2[2, :]
            INDEX_t[::1] f023 = temp2[3, :]
            np.ndarray[INDEX_t, ndim=2] dofs_mem = -MAX_INT*np.ones((nc,
                                                                     self.dofs_per_element),
                                                                    dtype=INDEX)
            INDEX_t[:, ::1] dofs = dofs_mem
            tupleDictINDEX eV
            faceVals fV
            INDEX_t[::1] boundaryVertices
            INDEX_t[:, ::1] boundaryEdges
            INDEX_t[:, ::1] boundaryFaces
            INDEX_t[::1] vertices = MAX_INT*np.ones((mesh.num_vertices),
                                                    dtype=INDEX)

        self.dofs = dofs

        numBdofs = -1
        if dofs_per_vertex > 0:
            if manifold_dim > 0:
                boundaryVertices = mesh.getBoundaryVerticesByTag(tag)
                for v in boundaryVertices:
                    vertices[v] = numBdofs
                    numBdofs -= dofs_per_vertex
        if dofs_per_edge > 0:
            eV = tupleDictINDEX(mesh.num_vertices, deleteHits=False)
            boundaryEdges = mesh.getBoundaryEdgesByTag(tag)
            for i in range(boundaryEdges.shape[0]):
                sortEdge(boundaryEdges[i, 0], boundaryEdges[i, 1], e01)
                eV.enterValue(e01, numBdofs)
                numBdofs -= dofs_per_edge
        if dofs_per_face > 0:
            fV = faceVals(mesh.num_vertices, deleteHits=True)
            boundaryFaces = mesh.getBoundaryFacesByTag(tag)
            for i in range(boundaryFaces.shape[0]):
                sortFace(boundaryFaces[i, 0], boundaryFaces[i, 1], boundaryFaces[i, 2], f012)
                fV.enterValue(f012, numBdofs)
                numBdofs -= dofs_per_face
        self.num_boundary_dofs = -numBdofs-1

        if skipCellsAfter == -1:
            skipCellsAfter = nc
        numDoFs = 0
        for i in range(nc):
            # dofs on the vertices
            if dofs_per_vertex > 0:
                for k in range(vertices_per_element):
                    v = cells[i, k]
                    dof = vertices[v]
                    if dof != MAX_INT:
                        # Vertex already has a DoF
                        if dof >= 0:
                            for j in range(k*dofs_per_vertex,
                                           (k+1)*dofs_per_vertex):
                                dofs[i, j] = dof
                                dof += 1
                        else:
                            for j in range(k*dofs_per_vertex,
                                           (k+1)*dofs_per_vertex):
                                dofs[i, j] = dof
                                dof -= 1
                    else:
                        # Vertex does not already have a DoF
                        # Do we want to assign one?
                        if i < skipCellsAfter:
                            vertices[v] = numDoFs
                            dof = numDoFs
                            for j in range(k*dofs_per_vertex,
                                           (k+1)*dofs_per_vertex):
                                dofs[i, j] = dof
                                dof += 1
                            numDoFs += dofs_per_vertex
            # dofs on the edges
            if dofs_per_edge > 0:
                c0, c1, c2 = cells[i, 0], cells[i, 1], cells[i, 2]
                sortEdge(c0, c1, e01)
                sortEdge(c1, c2, e12)
                sortEdge(c0, c2, e20)
                edgeOrientations[0] = c1-c0
                edgeOrientations[1] = c2-c1
                edgeOrientations[2] = c0-c2
                if manifold_dim == 3:
                    c3 = cells[i, 3]
                    sortEdge(c0, c3, e03)
                    sortEdge(c1, c3, e13)
                    sortEdge(c2, c3, e23)
                    edgeOrientations[3] = c3-c0
                    edgeOrientations[4] = c3-c1
                    edgeOrientations[5] = c3-c2

                for k in range(edges_per_element):
                    # Try to enter new DoF for edge
                    dof = eV.enterValue(temp[k, :], numDoFs)
                    # We got the DoF of that edge back.
                    if dof == numDoFs:
                        # It's the new one we assigned
                        if edgeOrientations[k] < 0:
                            dof += dofs_per_edge-1
                        if i < skipCellsAfter:
                            for j in range(vertices_per_element*dofs_per_vertex + k*dofs_per_edge,
                                           vertices_per_element*dofs_per_vertex + (k+1)*dofs_per_edge):
                                dofs[i, j] = dof
                                if edgeOrientations[k] < 0:
                                    dof -= 1
                                else:
                                    dof += 1
                            numDoFs += dofs_per_edge
                        else:
                            eV.removeValue(temp[k, :])
                    else:
                        # It was already present
                        if dof >= 0:
                            if edgeOrientations[k] < 0:
                                dof += dofs_per_edge-1
                            for j in range(vertices_per_element*dofs_per_vertex + k*dofs_per_edge,
                                           vertices_per_element*dofs_per_vertex + (k+1)*dofs_per_edge):
                                dofs[i, j] = dof
                                if edgeOrientations[k] < 0:
                                    dof -= 1
                                else:
                                    dof += 1
                        else:
                            if edgeOrientations[k] < 0:
                                dof += dofs_per_edge-1
                            for j in range(vertices_per_element*dofs_per_vertex + k*dofs_per_edge,
                                           vertices_per_element*dofs_per_vertex + (k+1)*dofs_per_edge):
                                dofs[i, j] = dof
                                if edgeOrientations[k] < 0:
                                    dof += 1
                                else:
                                    dof -= 1
            # dofs on the faces
            if dofs_per_face > 0:
                c0, c1, c2, c3 = cells[i, 0], cells[i, 1], cells[i, 2], cells[i, 3]
                sortFace(c0, c1, c2, f012)
                sortFace(c0, c1, c3, f013)
                sortFace(c1, c2, c3, f123)
                sortFace(c0, c2, c3, f023)
                for k in range(faces_per_element):
                    # Try to enter new DoF for face
                    dof = fV.enterValue(temp2[k, :], numDoFs)
                    # We got the DoF of that face back.
                    if dof == numDoFs:
                        # It's the new one we assigned
                        if i < skipCellsAfter:
                            for j in range(vertices_per_element*dofs_per_vertex + edges_per_element*dofs_per_edge + k*dofs_per_face,
                                           vertices_per_element*dofs_per_vertex + edges_per_element*dofs_per_edge + (k+1)*dofs_per_face):
                                dofs[i, j] = dof
                                dof += 1
                            numDoFs += dofs_per_face
                        else:
                            # FIX: This should not be commented out!
                            # fV.removeValue(temp2[k, :]) # Not implemented yet!!
                            pass
                    else:
                        # It was already present
                        if dof >= 0:
                            dof += dofs_per_face-1
                            for j in range(vertices_per_element*dofs_per_vertex + edges_per_element*dofs_per_edge + k*dofs_per_face,
                                           vertices_per_element*dofs_per_vertex + edges_per_element*dofs_per_edge + (k+1)*dofs_per_face):
                                dofs[i, j] = dof
                                dof -= 1
                        else:
                            for j in range(vertices_per_element*dofs_per_vertex + edges_per_element*dofs_per_edge + k*dofs_per_face,
                                           vertices_per_element*dofs_per_vertex + edges_per_element*dofs_per_edge + (k+1)*dofs_per_face):
                                dofs[i, j] = dof
                                dof -= 1
            if i < skipCellsAfter:
                # dofs in the interior of the cell
                for k in range(vertices_per_element*dofs_per_vertex + edges_per_element*dofs_per_edge + faces_per_element*dofs_per_face,
                               self.dofs_per_element):
                    dofs[i, k] = numDoFs
                    numDoFs += 1
        self.num_dofs = numDoFs

        if isinstance(tag, function):
            self.tagFunction = tag
            self.resetUsingIndicator(tag)
        else:
            self.tagFunction = None

        self.inner = ip_serial()
        self.norm = norm_serial()
        self.complex_inner = wrapRealInnerToComplex(ip_serial())
        self.complex_norm = wrapRealNormToComplex(norm_serial())

    cpdef void resetUsingIndicator(self, function indicator):
        cdef:
            fe_vector ind
        ind = self.interpolate(indicator)
        self.resetUsingFEVector(ind)

    cpdef void resetUsingFEVector(self, REAL_t[::1] ind):
        cdef:
            INDEX_t[:, ::1] new_dofs = uninitialized((self.mesh.num_cells,
                                                      self.dofs_per_element), dtype=INDEX)
            INDEX_t cellNo, dofNo, dofOld, dofNew = 0, dofNewBoundary = -1
            dict old2new = {}
        for cellNo in range(self.mesh.num_cells):
            for dofNo in range(self.dofs_per_element):
                dofOld = self.cell2dof(cellNo, dofNo)
                try:
                    new_dofs[cellNo, dofNo] = old2new[dofOld]
                except KeyError:
                    if dofOld >= 0 and ind[dofOld] > 0:
                        new_dofs[cellNo, dofNo] = dofNew
                        old2new[dofOld] = dofNew
                        dofNew += 1
                    else:
                        new_dofs[cellNo, dofNo] = dofNewBoundary
                        old2new[dofOld] = dofNewBoundary
                        dofNewBoundary -= 1
        self.dofs = new_dofs
        self.num_dofs = dofNew
        self.num_boundary_dofs = -dofNewBoundary-1

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cdef INDEX_t cell2dof(self,
                          const INDEX_t cellNo,
                          const INDEX_t perCellNo):
        return self.dofs[cellNo, perCellNo]

    def cell2dof_py(self, INDEX_t cellNo, INDEX_t perCellNo):
        return self.cell2dof(cellNo, perCellNo)

    cpdef void reorder(self, const INDEX_t[::1] perm):
        cdef INDEX_t i, j, dof
        for i in range(self.dofs.shape[0]):
            for j in range(self.dofs.shape[1]):
                dof = self.dofs[i, j]
                if dof >= 0:
                    self.dofs[i, j] = perm[dof]

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    def buildSparsityPattern(self, const cells_t cells,
                             INDEX_t start_idx=-1, INDEX_t end_idx=-1,
                             BOOL_t symmetric=False,
                             BOOL_t reorder=False):
        cdef:
            INDEX_t i, j, k, I, J, jj
            INDEX_t num_dofs = self.num_dofs
            INDEX_t num_cells = cells.shape[0]
            INDEX_t nnz = 0
            INDEX_t[::1] indptr, indices
            REAL_t[::1] data, diagonal
            sparsityPattern sPat = sparsityPattern(num_dofs)

        if self.reordered:
            reorder = False

        if start_idx == -1:
            start_idx = 0
        if end_idx == -1:
            end_idx = num_cells

        if symmetric and not reorder:
            for i in range(start_idx, end_idx):
                for j in range(self.dofs_per_element):
                    I = self.cell2dof(i, j)
                    if I < 0:
                        continue
                    for k in range(self.dofs_per_element):
                        J = self.cell2dof(i, k)
                        # This is the only line that differs from the
                        # non-symmetric code
                        if J < 0 or I <= J:
                            continue
                        # create entry (I, J)
                        sPat.add(I, J)
        else:
            for i in range(start_idx, end_idx):
                for j in range(self.dofs_per_element):
                    I = self.cell2dof(i, j)
                    if I < 0:
                        continue
                    for k in range(self.dofs_per_element):
                        J = self.cell2dof(i, k)
                        if J < 0:
                            continue
                        # create entry (I, J)
                        sPat.add(I, J)

        indptr, indices = sPat.freeze()
        del sPat
        if reorder:
            perm = uninitialized((indptr.shape[0]-1), dtype=INDEX)
            graph = sparseGraph(indices, indptr, indptr.shape[0]-1, indptr.shape[0]-1)
            cuthill_mckee(graph, perm)
            # get inverse of permutation
            iperm = uninitialized((perm.shape[0]), dtype=INDEX)
            for i in range(perm.shape[0]):
                iperm[perm[i]] = i

            sPat = sparsityPattern(num_dofs)
            if symmetric:
                for i in range(perm.shape[0]):
                    I = perm[i]
                    for jj in range(indptr[I], indptr[I+1]):
                        J = indices[jj]
                        j = iperm[J]
                        if i > j:
                            sPat.add(i, j)
                        elif i < j:
                            sPat.add(j, i)
            else:
                for i in range(perm.shape[0]):
                    I = perm[i]
                    for jj in range(indptr[I], indptr[I+1]):
                        J = indices[jj]
                        j = iperm[J]
                        sPat.add(i, j)
            indptr, indices = sPat.freeze()
            self.reorder(iperm)
            self.reordered = True
            del sPat
        nnz = indptr[num_dofs]
        data = np.zeros((nnz), dtype=REAL)
        if symmetric:
            diagonal = np.zeros((num_dofs), dtype=REAL)
            A = SSS_LinearOperator(indices, indptr, data, diagonal)
        else:
            A = CSR_LinearOperator(indices, indptr, data)
        A.sort_indices()
        return A

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    def buildNonSymmetricSparsityPattern(self,
                                         const cells_t cells,
                                         DoFMap dmOther,
                                         INDEX_t start_idx=-1,
                                         INDEX_t end_idx=-1):
        cdef:
            INDEX_t i, j, k, I, J
            INDEX_t num_dofs = self.num_dofs
            INDEX_t num_cells = cells.shape[0]
            INDEX_t nnz = 0
            INDEX_t[::1] indptr, indices
            REAL_t[::1] data
            sparsityPattern sPat = sparsityPattern(num_dofs)

        if start_idx == -1:
            start_idx = 0
        if end_idx == -1:
            end_idx = num_cells

        for i in range(start_idx, end_idx):
            for j in range(self.dofs_per_element):
                I = self.cell2dof(i, j)
                if I < 0:
                    continue
                for k in range(dmOther.dofs_per_element):
                    J = dmOther.cell2dof(i, k)
                    if J < 0:
                        continue
                    # create entry (I, J)
                    sPat.add(I, J)

        indptr, indices = sPat.freeze()
        del sPat
        nnz = indptr[num_dofs]
        data = np.zeros((nnz), dtype=REAL)
        A = CSR_LinearOperator(indices, indptr, data)
        A.num_columns = dmOther.num_dofs
        A.sort_indices()
        return A

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    def interpolate(self, fun):
        cdef:
            function real_fun
            vectorFunction real_vec_fun
            REAL_t[::1] real_vec_data
            fe_vector real_vec
            complexFunction complex_fun
            COMPLEX_t[::1] complex_vec_data
            complex_fe_vector complex_vec
            INDEX_t cellNo, i, dof
            REAL_t[:, ::1] simplex = uninitialized((self.mesh.manifold_dim+1,
                                                      self.mesh.dim), dtype=REAL)
            REAL_t[:, ::1] pos = uninitialized((self.dofs_per_element, self.mesh.dim), dtype=REAL)
        if isinstance(fun, function):
            real_fun = fun
            real_vec_data = self.full(fill_value=np.nan, dtype=REAL)
            real_vec = fe_vector(real_vec_data, self)
        elif isinstance(fun, complexFunction):
            complex_fun = fun
            complex_vec_data = self.full(fill_value=np.nan, dtype=COMPLEX)
            complex_vec = complex_fe_vector(complex_vec_data, self)
        elif isinstance(fun, vectorFunction):
            real_vec_fun = fun
            real_fvals = uninitialized((real_vec_fun.rows), dtype=REAL)
            real_vec_data = self.full(fill_value=np.nan, dtype=REAL)
            real_vec = fe_vector(real_vec_data, self)
        else:
            raise NotImplementedError()
        if isinstance(fun, function):
            for cellNo in range(self.mesh.num_cells):
                self.mesh.getSimplex(cellNo, simplex)
                matmat(self.nodes, simplex, pos)
                for i in range(self.dofs_per_element):
                    dof = self.cell2dof(cellNo, i)
                    if dof >= 0 and isnan(real_vec_data[dof]):
                        real_vec_data[dof] = real_fun.eval(pos[i, :])
            return real_vec
        elif isinstance(fun, complexFunction):
            for cellNo in range(self.mesh.num_cells):
                self.mesh.getSimplex(cellNo, simplex)
                matmat(self.nodes, simplex, pos)
                for i in range(self.dofs_per_element):
                    dof = self.cell2dof(cellNo, i)
                    if dof >= 0 and isnan(complex_vec_data[dof].real):
                        complex_vec_data[dof] = complex_fun.eval(pos[i, :])
            return complex_vec
        elif isinstance(fun, vectorFunction):
            for cellNo in range(self.mesh.num_cells):
                self.mesh.getSimplex(cellNo, simplex)
                matmat(self.nodes, simplex, pos)
                for i in range(self.dofs_per_element):
                    dof = self.cell2dof(cellNo, i)
                    if dof >= 0 and isnan(real_vec_data[dof]):
                        real_vec_fun.eval(pos[i, :], real_fvals)
                        real_vec_data[dof] = 0.
                        for k in range(real_vec_fun.rows):
                            real_vec_data[dof] += real_fvals[k]*self.dof_dual[i, k]
            return real_vec

    def getDoFCoordinates(self):
        from . functions import coordinate
        coords = uninitialized((self.num_dofs, self.mesh.dim), dtype=REAL)
        for i in range(self.mesh.dim):
            coords[:, i] = self.interpolate(coordinate(i))
        return coords

    def project(self, function, DoFMap=None, simplexQuadratureRule qr=None):
        from . femCy import assembleRHSfromFEfunction
        from scipy.sparse.linalg import spsolve
        if isinstance(function, np.ndarray):
            rhs = assembleRHSfromFEfunction(self.mesh, function, DoFMap, self, qr=qr)
        else:
            rhs = self.assembleRHS(function, qr=qr)
        mass = self.assembleMass()
        x = spsolve(mass.to_csr(), rhs)
        return fe_vector(x, self)

    def assembleMass(self,
                     vector_t boundary_data=None,
                     vector_t rhs_contribution=None,
                     LinearOperator A=None,
                     INDEX_t start_idx=-1,
                     INDEX_t end_idx=-1,
                     BOOL_t sss_format=False,
                     BOOL_t reorder=False,
                     INDEX_t[::1] cellIndices=None,
                     DoFMap dm2=None,
                     coefficient=None):
        """Assemble

        .. math::

           \int_D u(x) coefficient(x) v(x) dx

        """
        if not isinstance(self, Product_DoFMap):
            if dm2 is None:
                from . femCy import assembleMass
                return assembleMass(self,
                                    boundary_data,
                                    rhs_contribution,
                                    A,
                                    start_idx, end_idx,
                                    sss_format,
                                    reorder,
                                    cellIndices,
                                    coefficient=coefficient)
            else:
                assert self.mesh == dm2.mesh
                from . femCy import assembleMassNonSym
                return assembleMassNonSym(self.mesh, self, dm2, A, start_idx, end_idx)
        else:
            componentRs = []
            componentPs = []
            if dm2 is not None:
                assert isinstance(dm2, Product_DoFMap)
                assert dm2.numComponents == self.numComponents
            for row in range(self.numComponents):
                R, P = self.getRestrictionProlongation(row)
                if dm2 is None:
                    componentRs.append(R)
                componentPs.append(P)
            if dm2 is not None:
                for row in range(self.numComponents):
                    R, _ = dm2.getRestrictionProlongation(row)
                    componentRs.append(R)

            M_component = self.scalarDM.assembleMass(sss_format=sss_format, dm2=dm2.scalarDM if dm2 is not None else None, coefficient=coefficient)
            firstOp = True
            for row in range(self.numComponents):
                if firstOp:
                    M = componentPs[row]*M_component*componentRs[row]
                    firstOp = False
                else:
                    M = M+componentPs[row]*M_component*componentRs[row]
            return M

    def assembleDrift(self,
                      vectorFunction coeff,
                      LinearOperator A=None,
                      INDEX_t start_idx=-1,
                      INDEX_t end_idx=-1,
                      INDEX_t[::1] cellIndices=None):
        """Assemble

        .. math::

           \int_D (coeff(x) \cdot \nabla u(x)) v(x) dx

        """
        from . femCy import assembleDrift
        return assembleDrift(self,
                             coeff,
                             A,
                             start_idx, end_idx,
                             cellIndices)

    def assembleStiffness(self,
                          vector_t boundary_data=None,
                          vector_t rhs_contribution=None,
                          LinearOperator A=None,
                          INDEX_t start_idx=-1,
                          INDEX_t end_idx=-1,
                          BOOL_t sss_format=False,
                          BOOL_t reorder=False,
                          diffusivity=None,
                          INDEX_t[::1] cellIndices=None,
                          DoFMap dm2=None):
        """Assemble

        .. math::

           \int_D \nabla u(x) \cdot diffusivity(x) \nabla v(x) dx

        """
        from . femCy import assembleStiffness
        return assembleStiffness(self,
                                 boundary_data,
                                 rhs_contribution,
                                 A,
                                 start_idx, end_idx,
                                 sss_format,
                                 reorder,
                                 diffusivity,
                                 cellIndices,
                                 dm2=dm2)

    def assembleRHS(self,
                    fun,
                    simplexQuadratureRule qr=None):
        """Assemble

        .. math::

           \int_D fun(x) v(x) dx

        """
        from . femCy import assembleRHS, assembleRHScomplex
        if isinstance(fun, complexFunction):
            return assembleRHScomplex(fun, self, qr)
        else:
            return assembleRHS(fun, self, qr)

    def assembleRHSgrad(self,
                        fun,
                        vectorFunction coeff,
                        simplexQuadratureRule qr=None):
        """Assemble

        .. math::

           \int_D fun(x) (coeff(x) \cdot \nabla v(x)) dx

        """
        from . femCy import assembleRHSgrad
        return assembleRHSgrad(fun, self, coeff, qr)

    def assembleNonlocal(self, kernel, str matrixFormat='DENSE', DoFMap dm2=None, BOOL_t returnNearField=False, **kwargs):
        """Assemble a nonlocal operator of the form

        .. math::

           \int_D (u(x)-u(y)) \gamma(x, y) dy

        :param kernel: The kernel function :math:`\gamma`

        :param matrixFormat: The matrix format for the assembly. Valid
            values are `dense`, `diagonal`, `sparsified`, `sparse`,
            `H2` and `H2corrected`. `H2` assembles into a hierachical
            matrix format. `H2corrected` also assembles a hierachical
            matrix for an infinite horizon kernel and a correction
            term. `diagonal` returns the matrix diagonal. Both
            `sparsified` and `sparse` return a sparse matrix, but the
            assembly routines are different.

        """
        
        try:
            
            from PyNucleus_nl.kernelsCy import RangedFractionalKernel

            if isinstance(kernel, RangedFractionalKernel):
                from PyNucleus_base.linear_operators import multiIntervalInterpolationOperator
                from PyNucleus_nl import getChebyIntervalsAndNodes, delayedNonlocalOp
                s_left, s_right = kernel.admissibleOrders.ranges[0, 0], kernel.admissibleOrders.ranges[0, 1]
                horizonValue = min(self.mesh.diam, kernel.horizon.value)
                r = 1/2
                if kernel.errorBound <= 0.:
                    # errorBound = 0.25*self.mesh.h**0.5
                    errorBound = 0.1*self.mesh.h**0.5
                    # errorBound = 0.01*self.mesh.h**0.5
                    # errorBound = self.mesh.h**0.5
                else:
                    errorBound = kernel.errorBound
                intervals, nodes = getChebyIntervalsAndNodes(s_left, s_right, horizonValue, r,
                                                             errorBound,
                                                             M_min=kernel.M_min, M_max=kernel.M_max,
                                                             fixedXi=kernel.xi, variableOrder=True)
                ops = []
                for n in nodes:
                    intervalOps = []
                    for s in n:
                        gamma = kernel.getFrozenKernel(s)
                        intervalOps.append(delayedNonlocalOp(self, gamma, matrixFormat=matrixFormat, dm2=dm2, **kwargs))
                    ops.append(intervalOps)
                return multiIntervalInterpolationOperator(intervals, nodes, ops)
            
            else:
                from PyNucleus_nl import nonlocalBuilder

                builder = nonlocalBuilder(self.mesh, self, kernel, dm2=dm2, **kwargs)
                if matrixFormat.upper() == 'DENSE':
                    return builder.getDense()
                elif matrixFormat.upper() == 'DIAGONAL':
                    return builder.getDiagonal()
                elif matrixFormat.upper() == 'SPARSIFIED':
                    return builder.getDense(trySparsification=True)
                elif matrixFormat.upper() == 'SPARSE':
                    return builder.getSparse(returnNearField=returnNearField)
                elif matrixFormat.upper() == 'H2':
                    return builder.getH2(returnNearField=returnNearField)
                elif matrixFormat.upper() == 'H2CORRECTED':
                    A = builder.getH2FiniteHorizon()
                    A.setKernel(kernel)
                    return A
                else:
                    raise NotImplementedError('Unknown matrix format: {}'.format(matrixFormat))
        except ImportError as e:
            raise ImportError('\'PyNucleus_nl\' needs to be installed first.') from e

    def assembleElasticity(self,
                           lam=1.,
                           mu=1.,
                           DoFMap dm2=None):
        """Assemble

        .. math::

           \int_D sigma[u](x) : \epsilon[v](x) dx
           \epsilon[u] = (\nabla u + (\nabla u)^T) / 2
           \sigma[u]   = \lambda \nabla \cdot u I + 2\mu \epsilon[u]

        """
        from . femCy import (assembleMatrix,
                             assembleNonSymMatrix_CSR,
                             elasticity_1d_P1,
                             elasticity_2d_P1,
                             elasticity_3d_P1)

        dim = self.mesh.dim
        if isinstance(self, Product_DoFMap):
            if isinstance(self.scalarDM, P1_DoFMap):
                if dim == 1:
                    lm = elasticity_1d_P1(lam, mu)
                elif dim == 2:
                    lm = elasticity_2d_P1(lam, mu)
                elif dim == 3:
                    lm = elasticity_3d_P1(lam, mu)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        if dm2 is None:
            return assembleMatrix(self.mesh, self, lm)
        else:
            return assembleNonSymMatrix_CSR(self.mesh, lm, self, dm2, symLocalMatrix=True)

    def assembleNonlinearity(self, fun, multi_fe_vector U):
        from . femCy import assembleNonlinearity
        return assembleNonlinearity(self.mesh, fun, self, U)

    def __getstate__(self):
        return (self.mesh,
                np.array(self.dofs),
                self.dofs_per_vertex,
                self.dofs_per_edge,
                self.dofs_per_face,
                self.dofs_per_cell,
                self.dofs_per_element,
                self.num_dofs,
                self.num_boundary_dofs,
                self.tag,
                self.mesh,
                self.localShapeFunctions,
                np.array(self.nodes))

    def __setstate__(self, state):
        self.mesh = state[0]
        self.dofs = state[1]
        self.dofs_per_vertex = state[2]
        self.dofs_per_edge = state[3]
        self.dofs_per_face = state[4]
        self.dofs_per_cell = state[5]
        self.dofs_per_element = state[6]
        self.num_dofs = state[7]
        self.num_boundary_dofs = state[8]
        self.tag = state[9]
        self.mesh = state[10]
        self.localShapeFunctions = state[11]
        self.nodes = state[12]

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cdef void getNodalCoordinates(self, REAL_t[:, ::1] cell, REAL_t[:, ::1] coords):
        cdef:
            INDEX_t numCoords = self.nodes.shape[0]
            INDEX_t dim = cell.shape[1]
            INDEX_t i, j, k
        for i in range(numCoords):
            for k in range(dim):
                coords[i, k] = 0.
            for j in range(dim+1):
                for k in range(dim):
                    coords[i, k] += self.nodes[i, j]*cell[j, k]

    def getNodalCoordinates_py(self, REAL_t[:, ::1] cell):
        coords = np.zeros((self.nodes.shape[0], cell.shape[1]), dtype=REAL)
        self.getNodalCoordinates(cell, coords)
        return coords

    def zeros(self, INDEX_t numVecs=1, BOOL_t collection=False, dtype=REAL):
        if numVecs == 1:
            if dtype == REAL:
                return fe_vector(np.zeros((self.num_dofs), dtype=REAL), self)
            elif dtype == COMPLEX:
                return complex_fe_vector(np.zeros((self.num_dofs), dtype=COMPLEX), self)
            else:
                return np.zeros((self.num_dofs), dtype=dtype)
        else:
            if collection:
                return np.zeros((numVecs, self.num_dofs), dtype=dtype)
            else:
                if dtype == REAL:
                    return multi_fe_vector(np.zeros((numVecs, self.num_dofs), dtype=REAL), self)
                elif dtype == COMPLEX:
                    return complex_multi_fe_vector(np.zeros((numVecs, self.num_dofs), dtype=COMPLEX), self)
                else:
                    return np.zeros((numVecs, self.num_dofs), dtype=dtype)

    def ones(self, INDEX_t numVecs=1, BOOL_t collection=False, dtype=REAL):
        if numVecs == 1:
            if dtype == REAL:
                return fe_vector(np.ones((self.num_dofs), dtype=REAL), self)
            elif dtype == COMPLEX:
                return complex_fe_vector(np.ones((self.num_dofs), dtype=COMPLEX), self)
            else:
                np.ones((self.num_dofs), dtype=dtype)
        else:
            if collection:
                return np.ones((numVecs, self.num_dofs), dtype=dtype)
            else:
                if dtype == REAL:
                    return multi_fe_vector(np.ones((numVecs, self.num_dofs), dtype=REAL), self)
                elif dtype == COMPLEX:
                    return complex_multi_fe_vector(np.ones((numVecs, self.num_dofs), dtype=COMPLEX), self)
                else:
                    return np.ones((numVecs, self.num_dofs), dtype=dtype)

    def full(self, REAL_t fill_value, INDEX_t numVecs=1, BOOL_t collection=False, dtype=REAL):
        if numVecs == 1:
            if dtype == REAL:
                return fe_vector(np.full((self.num_dofs), fill_value=fill_value, dtype=REAL), self)
            elif dtype == COMPLEX:
                return complex_fe_vector(np.full((self.num_dofs), fill_value=fill_value, dtype=COMPLEX), self)
            else:
                return np.full((self.num_dofs), fill_value=fill_value, dtype=dtype)
        else:
            if collection:
                return np.full((numVecs, self.num_dofs), fill_value=fill_value, dtype=dtype)
            else:
                if dtype == REAL:
                    return multi_fe_vector(np.full((numVecs, self.num_dofs), fill_value=fill_value, dtype=REAL), self)
                elif dtype == COMPLEX:
                    return complex_multi_fe_vector(np.full((numVecs, self.num_dofs), fill_value=fill_value, dtype=COMPLEX), self)
                else:
                    return np.full((numVecs, self.num_dofs), fill_value=fill_value, dtype=dtype)

    def empty(self, INDEX_t numVecs=1, BOOL_t collection=False, dtype=REAL):
        if numVecs == 1:
            if dtype == REAL:
                return fe_vector(uninitialized((self.num_dofs), dtype=REAL), self)
            elif dtype == COMPLEX:
                return complex_fe_vector(uninitialized((self.num_dofs), dtype=COMPLEX), self)
            else:
                return uninitialized((self.num_dofs), dtype=dtype)
        else:
            if collection:
                return uninitialized((numVecs, self.num_dofs), dtype=dtype)
            else:
                if dtype == REAL:
                    return multi_fe_vector(uninitialized((numVecs, self.num_dofs), dtype=REAL), self)
                elif dtype == COMPLEX:
                    return complex_multi_fe_vector(uninitialized((numVecs, self.num_dofs), dtype=COMPLEX), self)
                else:
                    return uninitialized((numVecs, self.num_dofs), dtype=dtype)

    def fromArray(self, data):
        assert data.shape[0] == self.num_dofs, (data.shape[0], self.num_dofs)
        if data.dtype == COMPLEX:
            return complex_fe_vector(data, self)
        else:
            return fe_vector(data, self)

    def evalFun(self, const REAL_t[::1] u, INDEX_t cellNo, REAL_t[::1] x):
        cdef:
            REAL_t[:, ::1] simplex = uninitialized((self.dim+1, self.dim), dtype=REAL)
            REAL_t[::1] bary = uninitialized((self.dim+1), dtype=REAL)
            shapeFunction shapeFun
            REAL_t val
        self.mesh.getSimplex(cellNo, simplex)
        if self.dim == 1:
            getBarycentricCoords1D(simplex, x, bary)
        elif self.dim == 2:
            getBarycentricCoords2D(simplex, x, bary)
        else:
            raise NotImplementedError()
        val = 0.
        for k in range(self.dofs_per_element):
            dof = self.cell2dof(cellNo, k)
            if dof >= 0:
                shapeFun = self.localShapeFunctions[k]
                val += shapeFun.eval(bary)*u[dof]
        return val

    def getGlobalShapeFunction(self, INDEX_t dof):
        return globalShapeFunction(self, dof)

    def getCellLookup(self):
        cdef:
            INDEX_t cellNo, dofNo, dof
            list d2c
        d2c = [set() for dof in range(self.num_dofs)]
        for cellNo in range(self.mesh.num_cells):
            for dofNo in range(self.dofs_per_element):
                dof = self.cell2dof(cellNo, dofNo)
                if dof >= 0:
                    d2c[dof].add(cellNo)
        return d2c

    def getPatchLookup(self):
        return self.getCellLookup()

    def linearPart(self, fe_vector x):
        cdef:
            INDEX_t i, j, dof, dofP1
            DoFMap dm
            fe_vector y
            REAL_t[::1] yy
        if isinstance(self, P1_DoFMap):
            return x, self
        dm = P1_DoFMap(self.mesh, self.tag)
        y = dm.zeros(dtype=REAL)
        yy = y
        for i in range(self.mesh.num_cells):
            for j in range(0, (self.dim+1)*self.dofs_per_vertex, self.dofs_per_vertex):
                dofP1 = dm.cell2dof(i, j//self.dofs_per_vertex)
                if dofP1 >= 0:
                    dof = self.cell2dof(i, j)
                    if dof >= 0:
                        yy[dofP1] = x[dof]
                    else:
                        yy[dofP1] = 0.
        return y, dm

    def getComplementDoFMap(self):
        from copy import deepcopy
        cdef:
            bdm = deepcopy(self)
            INDEX_t i, j
        for i in range(self.mesh.num_cells):
            for j in range(self.dofs_per_element):
                bdm.dofs[i, j] = -bdm.dofs[i, j]-1
        bdm.num_dofs, bdm.num_boundary_dofs = self.num_boundary_dofs, self.num_dofs
        bdm.inner = ip_serial()
        bdm.norm = norm_serial()
        bdm.complex_inner = wrapRealInnerToComplex(ip_serial())
        bdm.complex_norm = wrapRealNormToComplex(norm_serial())
        return bdm

    def augmentWithZero(self, const REAL_t[::1] x):
        cdef:
            DoFMap dm = type(self)(self.mesh, tag=MAX_INT)
            fe_vector y = dm.empty(dtype=REAL)
            REAL_t[::1] yy = y
            INDEX_t i, k, dof, dof2, num_cells = self.mesh.num_cells
        for i in range(num_cells):
            for k in range(self.dofs_per_element):
                dof = self.cell2dof(i, k)
                dof2 = dm.cell2dof(i, k)
                if dof >= 0:
                    yy[dof2] = x[dof]
                else:
                    yy[dof2] = 0.
        return y, dm

    def augmentWithBoundaryData(self,
                                const REAL_t[::1] x,
                                const REAL_t[::1] boundaryData):
        cdef:
            DoFMap dm
            fe_vector y
            REAL_t[::1] yy
            INDEX_t i, k, dof, dof2, num_cells = self.mesh.num_cells

        if isinstance(self, Product_DoFMap):
            dm = Product_DoFMap(type(self.scalarDM)(self.mesh, tag=MAX_INT), self.numComponents)
        else:
            dm = type(self)(self.mesh, tag=MAX_INT)
        y = dm.empty(dtype=REAL)
        yy = y

        for i in range(num_cells):
            for k in range(self.dofs_per_element):
                dof = self.cell2dof(i, k)
                dof2 = dm.cell2dof(i, k)
                if dof >= 0:
                    yy[dof2] = x[dof]
                else:
                    yy[dof2] = boundaryData[-dof-1]
        return y

    def getFullDoFMap(self, DoFMap complement_dm):
        cdef:
            DoFMap dm
            INDEX_t i, k, dof, dof2, num_cells = self.mesh.num_cells
            INDEX_t[::1] indptr, indices, indptr_bc, indices_bc

        if isinstance(self, Product_DoFMap):
            dm = Product_DoFMap(type(self.scalarDM)(self.mesh, tag=MAX_INT), self.numComponents)
        else:
            dm = type(self)(self.mesh, tag=MAX_INT)

        indptr = np.arange(self.num_dofs+1, dtype=INDEX)
        indices = np.zeros((self.num_dofs), dtype=INDEX)
        data = np.ones((self.num_dofs), dtype=REAL)
        indptr_bc = np.arange(self.num_boundary_dofs+1, dtype=INDEX)
        indices_bc = np.zeros((self.num_boundary_dofs), dtype=INDEX)
        data_bc = np.ones((self.num_boundary_dofs), dtype=REAL)

        for i in range(num_cells):
            for k in range(self.dofs_per_element):
                dof = self.cell2dof(i, k)
                dof2 = dm.cell2dof(i, k)
                if dof >= 0:
                    indices[dof] = dof2
                else:
                    indices_bc[-dof-1] = dof2

        R = CSR_LinearOperator(indices, indptr, data)
        R.num_columns = dm.num_dofs
        R_bc = CSR_LinearOperator(indices_bc, indptr_bc, data_bc)
        R_bc.num_columns = dm.num_dofs
        return dm, R, R_bc

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    def getBoundaryData(self, function boundaryFunction):
        cdef:
            REAL_t[::1] data = uninitialized((self.num_boundary_dofs), dtype=REAL)
            cells_t cells = self.mesh.cells
            INDEX_t num_cells = self.mesh.num_cells
            vertices_t vertices = self.mesh.vertices
            INDEX_t dim = self.mesh.dim
            INDEX_t i, k, dof, vertexNo, vertexNo2
            REAL_t[::1] vertex = uninitialized((dim), dtype=REAL)

        # This is not how it should be done, cause it's not flexible. We
        # are only evaluating at vertices and edge midpoints.

        # FIX: Avoid visiting every dof several times
        if self.dofs_per_vertex > 0:
            for i in range(num_cells):
                for k in range(dim+1):
                    dof = self.cell2dof(i, k*self.dofs_per_vertex)
                    if dof < 0:
                        vertexNo = cells[i, k]
                        data[-dof-1] = boundaryFunction.eval(vertices[vertexNo, :])
        if self.dofs_per_edge > 0:
            if dim == 2:
                for i in range(num_cells):
                    for k in range(dim+1):
                        dof = self.cell2dof(i, (dim+1)*self.dofs_per_vertex+k*self.dofs_per_edge)
                        if dof < 0:
                            vertexNo = cells[i, k]
                            vertexNo2 = cells[i, k%(dim+1)]
                            for m in range(dim):
                                vertex[m] = 0.5*(vertices[vertexNo, m]+vertices[vertexNo2, m])
                            data[-dof-1] = boundaryFunction.eval(vertex)
            elif dim == 3:
                for i in range(num_cells):
                    for k in range(3):
                        dof = self.cell2dof(i, (dim+1)*self.dofs_per_vertex+k*self.dofs_per_edge)
                        if dof < 0:
                            vertexNo = cells[i, k]
                            vertexNo2 = cells[i, k%(dim)]
                            for m in range(dim):
                                vertex[m] = 0.5*(vertices[vertexNo, m]+vertices[vertexNo2, m])
                            data[-dof-1] = boundaryFunction.eval(vertex)
                    for k in range(3, 6):
                        dof = self.cell2dof(i, (dim+1)*self.dofs_per_vertex+k*self.dofs_per_edge)
                        if dof < 0:
                            vertexNo = cells[i, k-3]
                            vertexNo2 = cells[i, 3]
                            for m in range(dim):
                                vertex[m] = 0.5*(vertices[vertexNo, m]+vertices[vertexNo2, m])
                            data[-dof-1] = boundaryFunction.eval(vertex)
        return np.array(data, copy=False)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef void getVertexDoFs(self, INDEX_t[:, ::1] v2d):
        cdef:
            INDEX_t vertices_per_element
            meshBase mesh = self.mesh
            INDEX_t dim = mesh.manifold_dim
            INDEX_t cellNo, j, v, k, dof
        if dim == 1:
            vertices_per_element = 2
        elif dim == 2:
            vertices_per_element = 3
        elif dim == 3:
            vertices_per_element = 4
        else:
            raise NotImplementedError()

        for cellNo in range(mesh.num_cells):
            for j in range(vertices_per_element):
                v = mesh.cells[cellNo, j]
                for k in range(self.dofs_per_vertex):
                    dof = self.cell2dof(cellNo, self.dofs_per_vertex*j+k)
                    v2d[v, k] = dof

    def getCoordinateBlocks(self, INDEX_t[::1] idxDims, delta=1e-5):
        cdef:
            REAL_t[:, ::1] c
            dict blocks = {}
            INDEX_t[::1] key
            REAL_t[::1] fac, mins, maxs
            INDEX_t i, j
            intTuple hv
            INDEX_t numBlocks, block, temp, nnz
        c = self.getDoFCoordinates()
        key = np.empty((idxDims.shape[0]), dtype=INDEX)
        fac = np.empty((idxDims.shape[0]), dtype=REAL)
        delta = 1e-5
        mins = np.array(c, copy=False).min(axis=0)[idxDims]
        maxs = np.array(c, copy=False).max(axis=0)[idxDims]
        for j in range(idxDims.shape[0]):
            fac[j] = 1./(maxs[j] - mins[j]) / delta
        numBlocks = 0
        for i in range(self.num_dofs):
            for j in range(idxDims.shape[0]):
                key[j] = <INDEX_t>(fac[j] * (c[i, idxDims[j]] - mins[j]) + 0.5)
            hv = intTuple.create(key)
            try:
                blocks[hv] += 1
            except KeyError:
                blocks[hv] = 1
                numBlocks += 1
        indptr = np.empty((numBlocks+1), dtype=INDEX)
        numBlocks = 0
        for hv in blocks:
            indptr[numBlocks] = blocks[hv]
            blocks[hv] = numBlocks
            numBlocks += 1
        nnz = 0
        for i in range(numBlocks):
            temp = indptr[i]
            indptr[i] = nnz
            nnz += temp
        indptr[numBlocks] = nnz
        indices = np.empty((nnz), dtype=INDEX)
        for i in range(self.num_dofs):
            for j in range(idxDims.shape[0]):
                key[j] = <INDEX_t>(fac[j] * (c[i, idxDims[j]] - mins[j]) + 0.5)
            hv = intTuple.create(key)
            block = blocks[hv]
            indices[indptr[block]] = i
            indptr[block] += 1
        for i in range(numBlocks, 0, -1):
            indptr[i] = indptr[i-1]
        indptr[0] = 0
        return sparseGraph(indices, indptr, numBlocks, self.num_dofs)

    def getReducedMeshDoFMap(self, INDEX_t[::1] selectedCells=None):
        # Same DoFs, but on mesh that only contains elements with DoFs
        # Warning: This can also discard domain corners in some cases.
        cdef:
            INDEX_t cellNo, dofNo, cellNoNew, dof
            meshBase newMesh
            DoFMap newDM
            dict boundaryDoFMapping = {}
            BOOL_t[::1] dofCheck

        if selectedCells is None:
            selectedCells = uninitialized((self.mesh.num_cells), dtype=INDEX)
            cellNoNew = 0
            for cellNo in range(self.mesh.num_cells):
                for dofNo in range(self.dofs_per_element):
                    if self.cell2dof(cellNo, dofNo) >= 0:
                        selectedCells[cellNoNew] = cellNo
                        cellNoNew += 1
                        break
            selectedCells = selectedCells[:cellNoNew]
        else:
            dofCheck = np.full((self.num_dofs), dtype=BOOL, fill_value=False)
            for cellNoNew in range(selectedCells.shape[0]):
                cellNo = selectedCells[cellNoNew]
                for dofNo in range(self.dofs_per_element):
                    dof = self.cell2dof(cellNo, dofNo)
                    if dof >= 0:
                        dofCheck[dof] = True
            assert np.all(dofCheck), "New mesh does not contain all previous DoFs"

        newMesh = getSubmesh2(self.mesh, selectedCells)
        newDM = type(self)(newMesh)

        boundaryDoFNew = -1
        for cellNoNew in range(newMesh.num_cells):
            cellNo = selectedCells[cellNoNew]
            for dofNo in range(self.dofs_per_element):
                dof = self.dofs[cellNo, dofNo]
                if dof >= 0:
                    newDM.dofs[cellNoNew, dofNo] = self.dofs[cellNo, dofNo]
                else:
                    try:
                        newDM.dofs[cellNoNew, dofNo] = boundaryDoFMapping[dof]
                    except KeyError:
                        newDM.dofs[cellNoNew, dofNo] = boundaryDoFNew
                        boundaryDoFMapping[dof] = boundaryDoFNew
                        boundaryDoFNew -= 1
        newDM.num_dofs = self.num_dofs
        newDM.num_boundary_dofs = -boundaryDoFNew-1
        return newDM

    def plot(self, *args, **kwargs):
        self.mesh.plotDoFMap(self, *args, **kwargs)

    def sort(self):
        cdef:
            INDEX_t[::1] idx, invIdx
            INDEX_t k, dof, dof2
        coords = self.getDoFCoordinates()
        if self.mesh.dim == 1:
            idx = np.argsort(coords, axis=0).ravel().astype(INDEX)
        elif self.mesh.dim == 2:
            idx = np.argsort(coords.view('d,d'), order=['f1', 'f0'], axis=0).flat[:coords.shape[0]].astype(INDEX)
        elif self.mesh.dim == 3:
            idx = np.argsort(coords.view('d,d,d'), order=['f2', 'f1', 'f0'], axis=0).flat[:coords.shape[0]].astype(INDEX)
        else:
            raise NotImplementedError()
        invIdx = uninitialized((self.num_dofs), dtype=INDEX)
        k = 0
        for dof in range(self.num_dofs):
            dof2 = idx[dof]
            invIdx[dof2] = k
            k += 1
        self.reorder(invIdx)

    def HDF5write(self, node):
        COMPRESSION = 'gzip'
        node.attrs['type'] = 'DoFMap'
        node.create_group('mesh')
        self.mesh.HDF5write(node['mesh'])
        node.attrs['dim'] = self.dim
        node.attrs['reordered'] = self.reordered
        # localShapeFunctions
        if isinstance(self, P0_DoFMap):
            node.attrs['element'] = 'P0'
        elif isinstance(self, P1_DoFMap):
            node.attrs['element'] = 'P1'
        elif isinstance(self, P2_DoFMap):
            node.attrs['element'] = 'P2'
        elif isinstance(self, P3_DoFMap):
            node.attrs['element'] = 'P3'
        node.create_dataset('nodes', data=self.nodes,
                            compression=COMPRESSION)
        node.attrs['num_dofs'] = self.num_dofs
        node.attrs['num_boundary_dofs'] = self.num_boundary_dofs
        node.create_dataset('dofs', data=self.dofs,
                            compression=COMPRESSION)
        node.attrs['polynomialOrder'] = self.polynomialOrder
        # tag
        node.attrs['dofs_per_vertex'] = self.dofs_per_vertex
        node.attrs['dofs_per_edge'] = self.dofs_per_edge
        node.attrs['dofs_per_face'] = self.dofs_per_face
        node.attrs['dofs_per_cell'] = self.dofs_per_cell
        node.attrs['dofs_per_element'] = self.dofs_per_element

    @staticmethod
    def HDF5read(node):
        from . mesh import meshNd
        mesh = meshNd.HDF5read(node['mesh'])
        if node.attrs['element'] == 'P0':
            dm = P0_DoFMap(mesh)
        elif node.attrs['element'] == 'P1':
            dm = P1_DoFMap(mesh)
        elif node.attrs['element'] == 'P2':
            dm = P2_DoFMap(mesh)
        elif node.attrs['element'] == 'P3':
            dm = P3_DoFMap(mesh)
        else:
            dm = DoFMap(mesh,
                        node.attrs['dofs_per_vertex'],
                        node.attrs['dofs_per_edge'],
                        node.attrs['dofs_per_face'],
                        node.attrs['dofs_per_cell'])
        dm.reordered = node.attrs['reordered']
        dm.nodes = np.array(node['nodes'], dtype=REAL)
        dm.num_dofs = node.attrs['num_dofs']
        dm.num_boundary_dofs = node.attrs['num_boundary_dofs']
        dm.dofs = np.array(node['dofs'], dtype=INDEX)
        dm.polynomialOrder = node.attrs['polynomialOrder']
        return dm

    def __repr__(self):
        return 'DoFMap with {} DoFs and {} boundary DoFs.'.format(self.num_dofs,
                                                                  self.num_boundary_dofs)

    def set_ip_norm(self, ipBase inner, normBase norm):
        self.inner = inner
        self.norm = norm

    def set_complex_ip_norm(self, complexipBase inner, complexNormBase norm):
        self.complex_inner = inner
        self.complex_norm = norm

    def combine(self, DoFMap other):
        from copy import deepcopy

        cdef:
            INDEX_t cellNo, dofNo, dof1, dof2
            DoFMap dmCombined

        assert type(self) == type(other), "Cannot combine DoFMaps of different type"
        assert self.mesh == other.mesh, "Both DoFMaps need to have the same mesh"
        assert self.num_dofs == other.num_boundary_dofs, "DoFMaps need to be complementary"
        assert self.num_boundary_dofs == other.num_dofs, "DoFMaps need to be complementary"

        dmCombined = deepcopy(self)
        dmCombined.num_dofs = self.num_dofs+other.num_dofs
        dmCombined.num_boundary_dofs = 0
        for cellNo in range(self.mesh.num_cells):
            for dofNo in range(self.dofs_per_element):
                dof1 = self.cell2dof(cellNo, dofNo)
                dof2 = other.cell2dof(cellNo, dofNo)
                if dof1 >= 0 and dof2 < 0:
                   dmCombined.dofs[cellNo, dofNo] = dof1
                elif dof1 < 0 and dof2 >= 0:
                    dmCombined.dofs[cellNo, dofNo] = self.num_dofs+dof2
                else:
                    raise NotImplementedError()
        return dmCombined

    def applyPeriodicity(self, vectorFunction coordinateMapping, REAL_t eps=1e-8):
        """
        coordinateMapping(x) -> coordinate modulo periodicity
        """
        cdef:
            REAL_t[:, ::1] coords = self.getDoFCoordinates()
            INDEX_t[::1] remap
            REAL_t[::1] y
            dict remap2
            INDEX_t dof, k = 0, k2, cellNo, dofNo
        from scipy.spatial import KDTree

        assert coordinateMapping.rows == self.mesh.dim

        kd = KDTree(coords)
        remap = uninitialized((self.num_dofs), dtype=INDEX)
        y = uninitialized((self.mesh.dim), dtype=REAL)
        for dof in range(self.num_dofs):
            coordinateMapping.eval(coords[dof, :], y)
            remap[dof] = kd.query(y, eps)[1][0]
        remap2 = {}
        for k, k2 in enumerate(np.unique(remap)):
            remap2[k2] = k
        for i in range(self.num_dofs):
            remap[i] = remap2[remap[i]]
        for cellNo in range(self.mesh.num_cells):
            for dofNo in range(self.dofs_per_element):
                dof = self.cell2dof_py(cellNo, dofNo)
                if dof >= 0:
                    self.dofs[cellNo, dofNo] = remap[dof]
        self.num_dofs = k+1

    def __eq__(self, DoFMap other):
        if self.dofs_per_element != other.dofs_per_element:
            return False
        if self.dofs_per_vertex != other.dofs_per_vertex:
            return False
        if self.dofs_per_edge != other.dofs_per_edge:
            return False
        if self.dofs_per_face != other.dofs_per_face:
            return False
        if self.dofs_per_cell != other.dofs_per_cell:
            return False
        if self.num_dofs != other.num_dofs:
            return False
        if self.num_boundary_dofs != other.num_boundary_dofs:
            return False
        if np.absolute(np.array(self.dofs)-np.array(other.dofs)).max() > 0:
            return False
        if self.mesh != other.mesh:
            return False
        return True


cdef class globalShapeFunction(function):
    cdef:
        REAL_t[:, :, ::1] simplices
        INDEX_t[::1] dofNos
        REAL_t[::1] bary
        DoFMap dm

    def __init__(self, DoFMap dm, INDEX_t dof):
        cdef:
            list cellNos = []
            list dofNos = []
            INDEX_t cellNo, dofNo, k
        for cellNo in range(dm.mesh.num_cells):
            for dofNo in range(dm.dofs_per_element):
                if dm.cell2dof(cellNo, dofNo) == dof:
                    cellNos.append(cellNo)
                    dofNos.append(dofNo)
        self.simplices = uninitialized((len(cellNos), dm.mesh.dim+1, dm.mesh.dim), dtype=REAL)
        k = 0
        for cellNo in cellNos:
            dm.mesh.getSimplex(cellNo, self.simplices[k, :, :])
            k += 1
        self.dofNos = np.array(dofNos, dtype=INDEX)
        self.bary = uninitialized((4), dtype=REAL)
        self.dm = dm

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t dim = self.simplices.shape[2], dofNo, k, j
            BOOL_t doEval
            REAL_t result = 0.
            shapeFunction phi
        if dim == 1:
            for k in range(self.simplices.shape[0]):
                getBarycentricCoords1D(self.simplices[k, :, :], x, self.bary)
                doEval = True
                for j in range(dim+1):
                    if self.bary[j] < 0:
                        doEval = False
                        break
                if doEval:
                    dofNo = self.dofNos[k]
                    phi = self.dm.localShapeFunctions[dofNo]
                    result += phi.eval(self.bary)
        elif dim == 2:
            for k in range(self.simplices.shape[0]):
                getBarycentricCoords2D(self.simplices[k, :, :], x, self.bary)
                doEval = True
                for j in range(dim+1):
                    if self.bary[j] < 0:
                        doEval = False
                        break
                if doEval:
                    dofNo = self.dofNos[k]
                    phi = self.dm.localShapeFunctions[dofNo]
                    result += phi.eval(self.bary)
        return result


cdef class shapeFunction:
    def __init__(self):
        self.bary = uninitialized((4), dtype=REAL)

    def __call__(self, lam):
        return self.eval(np.array(lam))

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, const REAL_t[::1] lam):
        pass

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t evalStrided(self, const REAL_t* lam, INDEX_t stride):
        raise NotImplementedError()

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t evalGlobal(self, REAL_t[:, ::1] simplex, REAL_t[::1] x):
        if simplex.shape[1] == 1:
            getBarycentricCoords1D(simplex, x, self.bary)
        elif simplex.shape[1] == 2:
            getBarycentricCoords2D(simplex, x, self.bary)
        else:
            raise NotImplementedError()
        return self.eval(self.bary)

    def evalGlobalPy(self, REAL_t[:, ::1] simplex, REAL_t[::1] x):
        return self.evalGlobal(simplex, x)

    def __getstate__(self):
        return

    def __setstate__(self, state):
        self.bary = uninitialized((4), dtype=REAL)


cdef class vectorShapeFunction:
    def __init__(self, INDEX_t dim, BOOL_t needsGradients):
        self.dim = dim
        self.cell = uninitialized((dim+1), dtype=INDEX)
        self.needsGradients = needsGradients

    cpdef void setCell(self, INDEX_t[::1] cell):
        cdef:
            INDEX_t i
        for i in range(self.dim+1):
            self.cell[i] = cell[i]

    def __call__(self, lam, gradLam):
        value = uninitialized((self.dim), dtype=REAL)
        self.eval(np.array(lam), np.array(gradLam), value)
        return value

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void eval(self, const REAL_t[::1] lam, const REAL_t[:, ::1] gradLam, REAL_t[::1] value):
        pass

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void evalGlobal(self, const REAL_t[:, ::1] simplex, const REAL_t[::1] x, REAL_t[::1] value):
        raise NotImplementedError()

    def evalGlobalPy(self, REAL_t[:, ::1] simplex, REAL_t[::1] x):
        value = uninitialized((self.dim), dtype=REAL)
        self.evalGlobal(simplex, x, value)
        return value


cdef class shapeFunctionP0(shapeFunction):
    def __call__(self, lam):
        return 1.

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, const REAL_t[::1] lam):
        return 1.

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t evalStrided(self, const REAL_t* lam, INDEX_t stride):
        return 1.


cdef class P0_DoFMap(DoFMap):
    def __init__(self, meshBase mesh, tag=None,
                 INDEX_t skipCellsAfter=-1):
        self.polynomialOrder = 0
        if mesh.manifold_dim == 1:
            self.localShapeFunctions = [shapeFunctionP0()]
            self.nodes = np.array([[0.5, 0.5]], dtype=REAL)
            super(P0_DoFMap, self).__init__(mesh, 0, 0, 0, 1, tag, skipCellsAfter)
        elif mesh.manifold_dim == 2:
            self.localShapeFunctions = [shapeFunctionP0()]
            self.nodes = np.array([[1./3., 1./3., 1./3.]], dtype=REAL)
            super(P0_DoFMap, self).__init__(mesh, 0, 0, 0, 1, tag, skipCellsAfter)
        elif mesh.manifold_dim == 3:
            self.localShapeFunctions = [shapeFunctionP0()]
            self.nodes = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=REAL)
            super(P0_DoFMap, self).__init__(mesh, 0, 0, 0, 1, tag, skipCellsAfter)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return 'P0 DoFMap with {} DoFs and {} boundary DoFs.'.format(self.num_dofs,
                                                                     self.num_boundary_dofs)

    def interpolateFE(self, mesh, P0_DoFMap dm, REAL_t[::1] u):
        cdef:
            REAL_t[::1] uFine = np.zeros((self.num_dofs))
            REAL_t[::1] uloc = uninitialized((dm.dofs_per_element))
            REAL_t[:, ::1] other_vertices = mesh.vertices
            INDEX_t[:, ::1] other_cells = mesh.cells
            REAL_t[:, ::1] other_simplex = uninitialized((mesh.dim+1,
                                                       mesh.dim), dtype=REAL)
            REAL_t[:, ::1] my_vertices = self.mesh.vertices
            INDEX_t[:, ::1] my_cells = self.mesh.cells
            REAL_t[:, ::1] my_simplex = uninitialized((self.mesh.dim+1,
                                                    self.mesh.dim), dtype=REAL)
            INDEX_t k, other_dof, i, j, my_dof, my_cell, other_cell
            REAL_t[:, ::1] coords = uninitialized((dm.dofs_per_element, mesh.vertices.shape[1]), dtype=REAL)
            REAL_t[::1] vertex

        from . meshCy import cellFinder
        cF = cellFinder(mesh)
        for my_cell in range(self.mesh.num_cells):
            for i in range(my_cells.shape[1]):
                for j in range(my_vertices.shape[1]):
                    my_simplex[i, j] = my_vertices[my_cells[my_cell, i], j]
            self.getNodalCoordinates(my_simplex, coords)
            for k in range(self.dofs_per_element):
                my_dof = self.cell2dof(my_cell, k)
                if my_dof >= 0 and uFine[my_dof] == 0.:
                    vertex = coords[k, :]
                    other_cell = cF.findCell(vertex)
                    for i in range(other_cells.shape[1]):
                        for j in range(other_vertices.shape[1]):
                            other_simplex[i, j] = other_vertices[other_cells[other_cell, i], j]
                    for j in range(dm.dofs_per_element):
                        other_dof = dm.cell2dof(other_cell, j)
                        if other_dof >= 0:
                            uloc[j] = u[other_dof]
                        else:
                            uloc[j] = 0.
                    uFine[my_dof] = evalP0(other_simplex, uloc, vertex)
        return np.array(uFine, copy=False)


cdef class shapeFunctionP1(shapeFunction):
    cdef:
        INDEX_t vertexNo

    def __init__(self, INDEX_t vertexNo):
        self.vertexNo = vertexNo

    def __call__(self, lam):
        return lam[self.vertexNo]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, const REAL_t[::1] lam):
        return lam[self.vertexNo]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t evalStrided(self, const REAL_t* lam, INDEX_t stride):
        return lam[self.vertexNo*stride]

    def __getstate__(self):
        return self.vertexNo

    def __setstate__(self, state):
        self.vertexNo = state
        self.bary = uninitialized((4), dtype=REAL)


cdef class P1_DoFMap(DoFMap):
    def __init__(self, meshBase mesh, tag=None,
                 INDEX_t skipCellsAfter=-1):
        self.polynomialOrder = 1
        if mesh.manifold_dim == 0:
            self.localShapeFunctions = [shapeFunctionP1(0)]
            self.nodes = np.array([[1.]], dtype=REAL)
        elif mesh.manifold_dim == 1:
            self.localShapeFunctions = [shapeFunctionP1(0),
                                        shapeFunctionP1(1)]
            self.nodes = np.array([[1., 0.],
                                   [0., 1.]], dtype=REAL)
        elif mesh.manifold_dim == 2:
            self.localShapeFunctions = [shapeFunctionP1(0),
                                        shapeFunctionP1(1),
                                        shapeFunctionP1(2)]
            self.nodes = np.array([[1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]], dtype=REAL)
        elif mesh.manifold_dim == 3:
            self.localShapeFunctions = [shapeFunctionP1(0),
                                        shapeFunctionP1(1),
                                        shapeFunctionP1(2),
                                        shapeFunctionP1(3)]
            self.nodes = np.array([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]], dtype=REAL)
        super(P1_DoFMap, self).__init__(mesh, 1, 0, 0, 0, tag, skipCellsAfter)

    def __repr__(self):
        return 'P1 DoFMap with {} DoFs and {} boundary DoFs.'.format(self.num_dofs,
                                                                     self.num_boundary_dofs)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def getValuesAtVertices(self, REAL_t[::1] u):
        cdef:
            INDEX_t cellNo, dofNo, dof, vertex
            REAL_t[::1] z
        z = np.zeros((self.mesh.num_vertices), dtype=REAL)
        for cellNo in range(self.mesh.num_cells):
            for dofNo in range(self.dofs_per_element):
                dof = self.cell2dof(cellNo, dofNo)
                if dof >= 0:
                    vertex = self.mesh.cells[cellNo, dofNo]
                    z[vertex] = u[dof]
        return z


cdef class shapeFunctionP2_vertex(shapeFunction):
    cdef:
        INDEX_t vertexNo

    def __init__(self, INDEX_t vertexNo):
        self.vertexNo = vertexNo

    def __call__(self, lam):
        return lam[self.vertexNo]*(2.*lam[self.vertexNo]-1.)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, const REAL_t[::1] lam):
        return lam[self.vertexNo]*(2.*lam[self.vertexNo]-1.)

    def __getstate__(self):
        return self.vertexNo

    def __setstate__(self, state):
        self.vertexNo = state
        self.bary = uninitialized((4), dtype=REAL)


cdef class shapeFunctionP2_edge(shapeFunction):
    cdef:
        INDEX_t vertexNo1, vertexNo2

    def __init__(self, INDEX_t vertexNo1, INDEX_t vertexNo2):
        self.vertexNo1 = vertexNo1
        self.vertexNo2 = vertexNo2

    def __call__(self, lam):
        return 4.*lam[self.vertexNo1]*lam[self.vertexNo2]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, const REAL_t[::1] lam):
        return 4.*lam[self.vertexNo1]*lam[self.vertexNo2]

    def __getstate__(self):
        return (self.vertexNo1, self.vertexNo2)

    def __setstate__(self, state):
        self.vertexNo1 = state[0]
        self.vertexNo2 = state[1]
        self.bary = uninitialized((4), dtype=REAL)


cdef class P2_DoFMap(DoFMap):
    def __init__(self, meshBase mesh, tag=None,
                 INDEX_t skipCellsAfter=-1):
        self.polynomialOrder = 2
        if mesh.dim == 1:
            self.localShapeFunctions = [shapeFunctionP2_vertex(0),
                                        shapeFunctionP2_vertex(1),
                                        shapeFunctionP2_edge(0, 1)]
            self.nodes = np.array([[1., 0.],
                                   [0., 1.],
                                   [0.5, 0.5]], dtype=REAL)
            super(P2_DoFMap, self).__init__(mesh, 1, 0, 0, 1, tag, skipCellsAfter)
        elif mesh.dim == 2:
            self.localShapeFunctions = [shapeFunctionP2_vertex(0),
                                        shapeFunctionP2_vertex(1),
                                        shapeFunctionP2_vertex(2),
                                        shapeFunctionP2_edge(0, 1),
                                        shapeFunctionP2_edge(1, 2),
                                        shapeFunctionP2_edge(0, 2)]
            self.nodes = np.array([[1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 1.],
                                   [0.5, 0.5, 0.],
                                   [0., 0.5, 0.5],
                                   [0.5, 0., 0.5]], dtype=REAL)
            super(P2_DoFMap, self).__init__(mesh, 1, 1, 0, 0, tag, skipCellsAfter)
        elif mesh.dim == 3:
            self.localShapeFunctions = [shapeFunctionP2_vertex(0),
                                        shapeFunctionP2_vertex(1),
                                        shapeFunctionP2_vertex(2),
                                        shapeFunctionP2_vertex(3),
                                        shapeFunctionP2_edge(0, 1),
                                        shapeFunctionP2_edge(1, 2),
                                        shapeFunctionP2_edge(0, 2),
                                        shapeFunctionP2_edge(0, 3),
                                        shapeFunctionP2_edge(1, 3),
                                        shapeFunctionP2_edge(2, 3)]
            self.nodes = np.array([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.],
                                   [0.5, 0.5, 0., 0.],
                                   [0., 0.5, 0.5, 0.],
                                   [0.5, 0., 0.5, 0.],
                                   [0.5, 0., 0., 0.5],
                                   [0., 0.5, 0., 0.5],
                                   [0., 0., 0.5, 0.5]], dtype=REAL)
            super(P2_DoFMap, self).__init__(mesh, 1, 1, 0, 0, tag, skipCellsAfter)

    def __repr__(self):
        return 'P2 DoFMap with {} DoFs and {} boundary DoFs.'.format(self.num_dofs,
                                                                     self.num_boundary_dofs)


cdef class shapeFunctionP3_vertex(shapeFunction):
    cdef:
        INDEX_t vertexNo

    def __init__(self, INDEX_t vertexNo):
        self.vertexNo = vertexNo

    def __call__(self, lam):
        return 4.5*lam[self.vertexNo]*(lam[self.vertexNo]-1./3.)*(lam[self.vertexNo]-2./3.)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, const REAL_t[::1] lam):
        return 4.5*lam[self.vertexNo]*(lam[self.vertexNo]-1./3.)*(lam[self.vertexNo]-2./3.)

    def __getstate__(self):
        return self.vertexNo

    def __setstate__(self, state):
        self.vertexNo = state
        self.bary = uninitialized((4), dtype=REAL)


cdef class shapeFunctionP3_edge(shapeFunction):
    cdef:
        INDEX_t vertexNo1, vertexNo2

    def __init__(self, INDEX_t vertexNo1, INDEX_t vertexNo2):
        self.vertexNo1 = vertexNo1
        self.vertexNo2 = vertexNo2

    def __call__(self, lam):
        return 13.5*lam[self.vertexNo1]*lam[self.vertexNo2]*(lam[self.vertexNo1]-1./3.)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, const REAL_t[::1] lam):
        return 13.5*lam[self.vertexNo1]*lam[self.vertexNo2]*(lam[self.vertexNo1]-1./3.)

    def __getstate__(self):
        return (self.vertexNo1, self.vertexNo2)

    def __setstate__(self, state):
        self.vertexNo1 = state[0]
        self.vertexNo2 = state[1]
        self.bary = uninitialized((4), dtype=REAL)


cdef class shapeFunctionP3_face(shapeFunction):
    cdef:
        INDEX_t vertexNo1, vertexNo2, vertexNo3

    def __init__(self, INDEX_t vertexNo1, INDEX_t vertexNo2, INDEX_t vertexNo3):
        self.vertexNo1 = vertexNo1
        self.vertexNo2 = vertexNo2
        self.vertexNo3 = vertexNo3

    def __call__(self, lam):
        return 27.*lam[self.vertexNo1]*lam[self.vertexNo2]*lam[self.vertexNo3]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef REAL_t eval(self, const REAL_t[::1] lam):
        return 27.*lam[self.vertexNo1]*lam[self.vertexNo2]*lam[self.vertexNo3]

    def __getstate__(self):
        return (self.vertexNo1, self.vertexNo2, self.vertexNo3)

    def __setstate__(self, state):
        self.vertexNo1 = state[0]
        self.vertexNo2 = state[1]
        self.vertexNo3 = state[2]
        self.bary = uninitialized((4), dtype=REAL)


cdef class P3_DoFMap(DoFMap):
    def __init__(self, meshBase mesh, tag=None,
                 INDEX_t skipCellsAfter=-1):
        self.polynomialOrder = 3
        if mesh.dim == 1:
            self.localShapeFunctions = [shapeFunctionP3_vertex(0),
                                        shapeFunctionP3_vertex(1),
                                        shapeFunctionP3_edge(0, 1),
                                        shapeFunctionP3_edge(1, 0)]
            self.nodes = np.array([[1., 0.],
                                   [0., 1.],
                                   [2./3., 1./3.],
                                   [1./3., 2./3.]], dtype=REAL)
            super(P3_DoFMap, self).__init__(mesh, 1, 0, 0, 2, tag, skipCellsAfter)
        elif mesh.dim == 2:
            self.localShapeFunctions = [shapeFunctionP3_vertex(0),
                                        shapeFunctionP3_vertex(1),
                                        shapeFunctionP3_vertex(2),
                                        shapeFunctionP3_edge(0, 1),
                                        shapeFunctionP3_edge(1, 0),
                                        shapeFunctionP3_edge(1, 2),
                                        shapeFunctionP3_edge(2, 1),
                                        shapeFunctionP3_edge(2, 0),
                                        shapeFunctionP3_edge(0, 2),
                                        shapeFunctionP3_face(0, 1, 2)]
            self.nodes = np.array([[1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 1.],
                                   [2./3., 1./3., 0.],
                                   [1./3., 2./3., 0.],
                                   [0., 2./3., 1./3.],
                                   [0., 1./3., 2./3.],
                                   [1./3., 0., 2./3.],
                                   [2./3., 0., 1./3.],
                                   [1./3., 1./3., 1./3.]], dtype=REAL)
            super(P3_DoFMap, self).__init__(mesh, 1, 2, 0, 1, tag, skipCellsAfter)
        elif mesh.dim == 3:
            self.localShapeFunctions = [shapeFunctionP3_vertex(0),
                                        shapeFunctionP3_vertex(1),
                                        shapeFunctionP3_vertex(2),
                                        shapeFunctionP3_vertex(3),
                                        shapeFunctionP3_edge(0, 1),
                                        shapeFunctionP3_edge(1, 0),
                                        shapeFunctionP3_edge(1, 2),
                                        shapeFunctionP3_edge(2, 1),
                                        shapeFunctionP3_edge(2, 0),
                                        shapeFunctionP3_edge(0, 2),
                                        shapeFunctionP3_edge(0, 3),
                                        shapeFunctionP3_edge(3, 0),
                                        shapeFunctionP3_edge(1, 3),
                                        shapeFunctionP3_edge(3, 1),
                                        shapeFunctionP3_edge(2, 3),
                                        shapeFunctionP3_edge(3, 2),
                                        shapeFunctionP3_face(0, 1, 2),
                                        shapeFunctionP3_face(0, 1, 3),
                                        shapeFunctionP3_face(1, 2, 3),
                                        shapeFunctionP3_face(2, 0, 3)]
            self.nodes = np.array([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.],
                                   [2./3., 1./3., 0., 0.],
                                   [1./3., 2./3., 0., 0.],
                                   [0., 2./3., 1./3., 0.],
                                   [0., 1./3., 2./3., 0.],
                                   [1./3., 0., 2./3., 0.],
                                   [2./3., 0., 1./3., 0.],
                                   [2./3., 0., 0., 1./3.],
                                   [1./3., 0., 0., 2./3.],
                                   [0., 2./3., 0., 1./3.],
                                   [0., 1./3., 0., 2./3.],
                                   [0., 0., 2./3., 1./3.],
                                   [0., 0., 1./3., 2./3.],
                                   [1./3., 1./3., 1./3., 0.],
                                   [1./3., 1./3., 0., 1./3.],
                                   [0., 1./3., 1./3., 1./3.],
                                   [1./3., 0., 1./3., 1./3.]], dtype=REAL)
            super(P3_DoFMap, self).__init__(mesh, 1, 2, 1, 0, tag, skipCellsAfter)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return 'P3 DoFMap with {} DoFs and {} boundary DoFs.'.format(self.num_dofs,
                                                                     self.num_boundary_dofs)




def str2DoFMap(element):
    if element == 'P0':
        return P0_DoFMap
    elif element == 'P1':
        return P1_DoFMap
    elif element == 'P2':
        return P2_DoFMap
    elif element == 'P3':
        return P3_DoFMap
    
    else:
        raise NotImplementedError('Unknown DoFMap: {}'.format(element))


def getAvailableDoFMaps():
    return ['P0', 'P1', 'P2', 'P3',
            
            ]


def str2DoFMapOrder(element):
    if element in ('P0', 0, '0'):
        return 0
    elif element in ('P1', 1, '1'):
        return 1
    elif element in ('P2', 2, '2'):
        return 2
    elif element in ('P3', 3, '3'):
        return 3
    
    else:
        raise NotImplementedError('Unknown DoFMap: {}'.format(element))


def getSubMap(DoFMap dm, indicator):
    if isinstance(indicator, function):
        indicator = dm.interpolate(indicator)
    else:
        assert indicator.shape[0] == dm.num_dofs

    cdef:
        DoFMap dmSub_GD
        REAL_t[::1] ind = indicator
        INDEX_t i, dofOld, dofNew, dofNewBoundary
        dict old2new

    old2new = {}
    dmSub_GD = type(dm)(dm.mesh)
    dofNew = 0
    dofNewBoundary = -1
    for i in range(dmSub_GD.dofs.shape[0]):
        for k in range(dmSub_GD.dofs_per_element):
            dofOld = dm.cell2dof(i, k)
            if dofOld >= 0 and ind[dofOld] > 0:
                try:
                    dmSub_GD.dofs[i, k] = old2new[dofOld]
                except KeyError:
                    dmSub_GD.dofs[i, k] = dofNew
                    old2new[dofOld] = dofNew
                    dofNew += 1
            else:
                try:
                    dmSub_GD.dofs[i, k] = old2new[dofOld]
                except KeyError:
                    dmSub_GD.dofs[i, k] = dofNewBoundary
                    old2new[dofOld] = dofNewBoundary
                    dofNewBoundary -= 1
    dmSub_GD.num_dofs = dofNew
    dmSub_GD.num_boundary_dofs = -dofNewBoundary-1
    return dmSub_GD


def getSubMapRestrictionProlongation(DoFMap dm, DoFMap dmSub, indicator=None):
    cdef:
        INDEX_t numDoFsSub = dmSub.num_dofs
        INDEX_t i, k, cellNo, dofNo, dof, dofSub
        meshBase mesh = dm.mesh
        set subMapAdded

    if indicator is not None:
        data = np.ones((numDoFsSub), dtype=REAL)
        indptr = np.arange((numDoFsSub+1), dtype=INDEX)
        indices = np.zeros((numDoFsSub), dtype=INDEX)
        k = 0
        for i in range(dm.num_dofs):
            if indicator[i] > 0:
                indices[k] = i
                k += 1
    else:
        assert dm.mesh.num_vertices == dmSub.mesh.num_vertices
        assert dm.mesh.num_cells == dmSub.mesh.num_cells
        subMapAdded = set()
        k = 0
        for cellNo in range(mesh.num_cells):
            for dofNo in range(dm.dofs_per_element):
                dof = dm.cell2dof(cellNo, dofNo)
                dofSub = dmSub.cell2dof(cellNo, dofNo)
                if dof >= 0 and dofSub >= 0 and dofSub not in subMapAdded:
                    subMapAdded.add(dofSub)
                    k += 1
        data = np.ones((k), dtype=REAL)
        indptr = np.arange((numDoFsSub+1), dtype=INDEX)
        indices = np.zeros((k), dtype=INDEX)
        subMapAdded = set()
        k = 0
        for cellNo in range(mesh.num_cells):
            for dofNo in range(dm.dofs_per_element):
                dof = dm.cell2dof(cellNo, dofNo)
                dofSub = dmSub.cell2dof(cellNo, dofNo)
                if dof >= 0 and dofSub >= 0 and dofSub not in subMapAdded:
                    indices[dofSub] = dof
                    subMapAdded.add(dofSub)
                    k += 1
    R = CSR_LinearOperator(indices, indptr, data)
    R.num_columns = dm.num_dofs
    P = R.transpose()
    return R, P


def getSubMapRestrictionProlongation2(meshBase mesh, DoFMap dm, DoFMap dmSub, INDEX_t[::1] newCellIndices):
    cdef:
        INDEX_t cellNo, newCellNo, dofNo, dofNew, dof
        INDEX_t[::1] indices, indptr
        REAL_t[::1] data
        CSR_LinearOperator opUnreduced2Reduced
    indices = uninitialized((dmSub.num_dofs), dtype=INDEX)
    for cellNo in range(mesh.num_cells):
        newCellNo = newCellIndices[cellNo]
        if newCellNo >= 0:
            for dofNo in range(dm.dofs_per_element):
                dofNew = dmSub.cell2dof(newCellNo, dofNo)
                if dofNew >= 0:
                    dof = dm.cell2dof(cellNo, dofNo)
                    indices[dofNew] = dof

    indptr = np.arange((dmSub.num_dofs+1), dtype=INDEX)
    data = np.ones((dmSub.num_dofs), dtype=REAL)
    opUnreduced2Reduced = CSR_LinearOperator(indices, indptr, data)
    opUnreduced2Reduced.num_columns = dm.num_dofs
    return opUnreduced2Reduced


def generateLocalMassMatrix(DoFMap dm, DoFMap dm2=None):
    cdef:
        simplexQuadratureRule qr
        REAL_t[::1] entries
        REAL_t[::1] node
        INDEX_t m, i, j, k, l
        REAL_t s
    from . femCy import generic_matrix
    node = uninitialized((dm.dim+1), dtype=REAL)
    if dm2 is None:
        qr = simplexXiaoGimbutas(2*dm.polynomialOrder+1, dm.dim)
        entries = uninitialized(((dm.dofs_per_element*(dm.dofs_per_element+1))//2), dtype=REAL)
        m = 0
        for i in range(len(dm.localShapeFunctions)):
            for j in range(i, len(dm.localShapeFunctions)):
                s = 0.
                for k in range(qr.num_nodes):
                    for l in range(dm.dim+1):
                        node[l] = qr.nodes[l, k]
                    s += dm.localShapeFunctions[i](node) * dm.localShapeFunctions[j](node) * qr.weights[k]
                entries[m] = s
                m += 1
    else:
        qr = simplexXiaoGimbutas(dm.polynomialOrder+dm2.polynomialOrder+1, dm.dim)
        entries = uninitialized((dm.dofs_per_element*dm2.dofs_per_element), dtype=REAL)
        m = 0
        for i in range(len(dm.localShapeFunctions)):
            for j in range(len(dm2.localShapeFunctions)):
                s = 0.
                for k in range(qr.num_nodes):
                    for l in range(dm.dim+1):
                        node[l] = qr.nodes[l, k]
                    s += dm.localShapeFunctions[i](node) * dm2.localShapeFunctions[j](node) * qr.weights[k]
                entries[m] = s
                m += 1
    return generic_matrix(entries)


cdef class lookupFunction(function):
    cdef:
        meshBase mesh
        public DoFMap dm
        public REAL_t[::1] u
        public cellFinder2 cellFinder

    def __init__(self, meshBase mesh, DoFMap dm, REAL_t[::1] u, cellFinder2 cF=None):
        self.mesh = mesh
        self.dm = dm
        self.u = u
        if cF is None:
            self.cellFinder = cellFinder2(self.mesh)
        else:
            self.cellFinder = cF

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self, REAL_t[::1] x):
        cdef:
            shapeFunction shapeFun
            REAL_t val
            INDEX_t cellNo, dof, k
        cellNo = self.cellFinder.findCell(x)
        if cellNo == -1:
            return 0.
        val = 0.
        for k in range(self.dm.dofs_per_element):
            dof = self.dm.cell2dof(cellNo, k)
            if dof >= 0:
                shapeFun = self.dm.localShapeFunctions[k]
                val += shapeFun.eval(self.cellFinder.bary)*self.u[dof]
        return val


cdef class elementSizeFunction(function):
    cdef:
        meshBase mesh
        public cellFinder2 cellFinder
        REAL_t[::1] hVector

    def __init__(self, meshBase mesh, cellFinder2 cF=None):
        self.mesh = mesh
        self.hVector = mesh.hVector
        if cF is None:
            self.cellFinder = cellFinder2(self.mesh)
        else:
            self.cellFinder = cF

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t cellNo
        cellNo = self.cellFinder.findCell(x)
        if cellNo == -1:
            return -1.
        return self.hVector[cellNo]


cdef class productSpaceShapeFunction(vectorShapeFunction):
    cdef:
        shapeFunction phi
        INDEX_t k
        INDEX_t K

    def __init__(self, shapeFunction phi, INDEX_t K, INDEX_t k, INDEX_t dim):
        super(productSpaceShapeFunction, self).__init__(dim, False)
        self.phi = phi
        self.K = K
        self.k = k

    def __call__(self, lam, gradLam):
        cdef:
            INDEX_t i
        value = np.zeros((self.K), dtype=REAL)
        for i in range(self.K):
            if i == self.k:
                value[i] = self.phi.eval(lam)
            else:
                value[i] = 0.
        return value

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void eval(self, const REAL_t[::1] lam, const REAL_t[:, ::1] gradLam, REAL_t[::1] value):
        cdef:
            INDEX_t i
        for i in range(self.K):
            if i == self.k:
                value[i] = self.phi.eval(lam)
            else:
                value[i] = 0.

    def __getstate__(self):
        return (self.phi, self.K, self.k, self.dim)

    def __setstate__(self, state):
        self.phi = state[0]
        self.K = state[1]
        self.k = state[2]
        self.dim = state[3]


cdef class Product_DoFMap(DoFMap):
    def __init__(self, DoFMap dm, INDEX_t numComponents):
        cdef:
            INDEX_t component, dim, scalarDoF, dofNo, i, j
            shapeFunction phi
        dim = dm.mesh.dim
        super(Product_DoFMap, self).__init__(dm.mesh,
                                             dm.dofs_per_vertex*numComponents,
                                             dm.dofs_per_edge*numComponents,
                                             dm.dofs_per_face*numComponents,
                                             dm.dofs_per_cell*numComponents,
                                             dm.tag,
                                             -1)
        self.polynomialOrder = dm.polynomialOrder
        self.numComponents = numComponents
        self.scalarDM = dm

        dof = 0
        bdof = -1
        for cellNo in range(self.mesh.num_cells):
            for dofNo in range(self.scalarDM.dofs_per_element):
                scalarDoF = self.scalarDM.cell2dof(cellNo, dofNo)
                if scalarDoF >= 0:
                    for j in range(self.numComponents):
                        self.dofs[cellNo, dofNo*numComponents+j] = numComponents*scalarDoF+j
                else:
                    for j in range(self.numComponents):
                        self.dofs[cellNo, dofNo*numComponents+j] = numComponents*(scalarDoF+1)-j-1
        self.num_dofs = self.numComponents*self.scalarDM.num_dofs
        self.num_boundary_dofs = self.numComponents*self.scalarDM.num_boundary_dofs

        self.localShapeFunctions = []
        self.nodes = uninitialized((self.dofs_per_element, self.mesh.dim+1), dtype=REAL)
        self.dof_dual = np.zeros((self.dofs_per_element, numComponents), dtype=REAL)
        i = 0
        for dofNo in range(self.scalarDM.dofs_per_element):
            for component in range(numComponents):
                phi = self.scalarDM.localShapeFunctions[dofNo]
                self.localShapeFunctions.append(productSpaceShapeFunction(phi, numComponents, component, self.mesh.dim))
                for j in range(dim+1):
                    self.nodes[i, j] = self.scalarDM.nodes[dofNo, j]
                self.dof_dual[i, component] = 1.
                i += 1

    def __repr__(self):
        return '({})^{} with {} DoFs and {} boundary DoFs.'.format(type(self.scalarDM).__name__,
                                                                   self.numComponents,
                                                                   self.num_dofs,
                                                                   self.num_boundary_dofs)

    def getRestrictionProlongation(self, INDEX_t component):
        cdef:
            INDEX_t[::1] indices
            INDEX_t vertexOffset, k, cellNo, dofNo, dof, component_dof
        assert 0 <= component
        assert component < self.numComponents
        indices = uninitialized((self.scalarDM.num_dofs), dtype=INDEX)
        vertexOffset = 0
        for k in range(component):
            vertexOffset += self.scalarDM.dofs_per_vertex
        for cellNo in range(self.mesh.num_cells):
            for dofNo in range(self.scalarDM.dofs_per_element):
                dof = self.cell2dof(cellNo, dofNo*self.numComponents+component)
                if dof >= 0:
                    component_dof = self.scalarDM.cell2dof(cellNo, dofNo)
                    indices[component_dof] = dof
        indptr = np.arange(self.scalarDM.num_dofs+1, dtype=INDEX)
        vals = np.ones((self.scalarDM.num_dofs), dtype=REAL)
        R = CSR_LinearOperator(indices, indptr, vals)
        R.num_columns = self.num_dofs
        P = R.transpose()
        return R, P

    def __getstate__(self):
        return (self.scalarDM, self.numComponents)

    def __setstate__(self, state):
        self.__init__(state[0], state[1])

    def getComplementDoFMap(self):
        complement_dm = super(Product_DoFMap, self).getComplementDoFMap()
        complement_dm.scalarDM = self.scalarDM.getComplementDoFMap()
        complement_dm.numComponents = self.numComponents
        complement_dm.dof_dual = self.dof_dual
        return complement_dm

    def linearPart(self, fe_vector x):
        if isinstance(self.scalarDM, P1_DoFMap):
            return x, self

        components = []
        for component in range(self.numComponents):
            linear_component, linear_dm = self.scalarDM.linearPart(x.getComponent(component))
            components.append(linear_component)
        linear_dm = Product_DoFMap(linear_dm, self.numComponents)
        linear_x = linear_dm.zeros()
        for component in range(self.numComponents):
            _, P = linear_dm.getRestrictionProlongation(component)
            linear_x += P*components[component]
        return linear_x, linear_dm

    cpdef void getVertexDoFs(self, INDEX_t[:, ::1] v2d):
        self.scalarDM.getVertexDoFs(v2d)
