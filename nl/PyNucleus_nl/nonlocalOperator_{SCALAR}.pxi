###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}double_local_matrix_t:
    def __init__(self, INDEX_t dim, INDEX_t manifold_dim1, INDEX_t manifold_dim2, DoFMap dm):
        self.distantQuadRules = {}
        self.dim = dim
        self.manifold_dim1 = manifold_dim1
        self.manifold_dim2 = manifold_dim2
        self.symmetricLocalMatrix = True
        self.symmetricCells = True
        self.cellNo1 = -1
        self.cellNo2 = -1
        self.vol1 = np.nan
        self.vol2 = np.nan

        self.DoFMap = dm
        self.precomputePermutations()

        if dim == 1 and manifold_dim1 == 1:
            self.volume1 = volume1Dsimplex
        elif dim == 1 and manifold_dim1 == 0:
            self.volume1 = volume0Dsimplex
        elif dim == 2 and manifold_dim1 == 2:
            self.volume1 = volume2Dsimplex
        elif dim == 2 and manifold_dim1 == 1:
            self.volume1 = volume1Din2Dsimplex
        elif dim == 3 and manifold_dim1 == 3:
            self.volume1 = volume3Dsimplex
        elif dim == 3 and manifold_dim1 == 2:
            self.volume1 = volume2Din3Dsimplex
        else:
            raise NotImplementedError()

        if dim == 1 and manifold_dim2 == 1:
            self.volume2 = volume1Dsimplex
        elif dim == 1 and manifold_dim2 == 0:
            self.volume2 = volume0Dsimplex
        elif dim == 2 and manifold_dim2 == 2:
            self.volume2 = volume2Dsimplex
        elif dim == 2 and manifold_dim2 == 1:
            self.volume2 = volume1Din2Dsimplex
        elif dim == 3 and manifold_dim2 == 3:
            self.volume2 = volume3Dsimplex
        elif dim == 3 and manifold_dim2 == 2:
            self.volume2 = volume2Din3Dsimplex
        else:
            raise NotImplementedError()

        if manifold_dim1 == 1:
            self.IDENTICAL = COMMON_EDGE
        elif manifold_dim1 == 2:
            self.IDENTICAL = COMMON_FACE
        elif manifold_dim1 == 3:
            self.IDENTICAL = COMMON_VOLUME
        else:
            raise NotImplementedError()

        self.center1 = uninitialized((self.dim), dtype=REAL)
        self.center2 = uninitialized((self.dim), dtype=REAL)

    cdef void precomputePermutations(self):
        cdef:
            INDEX_t[:, ::1] perms, surface_perms
            INDEX_t r, j, dofPerm, dofOrig, index, k
            tuple permTuple
            INDEX_t[::1] perm
            REAL_t eps = 1e-10
            INDEX_t dim = self.DoFMap.mesh.manifold_dim

        perms = uninitialized((factorial(dim+1, True), dim+1), dtype=INDEX)
        surface_perms = uninitialized((factorial(dim, True), dim), dtype=INDEX)

        from itertools import permutations

        self.pI_volume = PermutationIndexer(dim+1)
        for permTuple in permutations(range(dim+1)):
            perm = np.array(permTuple, dtype=INDEX)
            index = self.pI_volume.rank(perm)
            for k in range(dim+1):
                perms[index, k] = perm[k]

        self.pI_surface = PermutationIndexer(dim)
        for permTuple in permutations(range(dim)):
            perm = np.array(permTuple, dtype=INDEX)
            index = self.pI_surface.rank(perm)
            for k in range(dim):
                surface_perms[index, k] = perm[k]

        self.precomputedVolumeSimplexPermutations = perms
        self.precomputedSurfaceSimplexPermutations = surface_perms
        self.precomputedDoFPermutations = uninitialized((perms.shape[0],
                                                         self.DoFMap.dofs_per_element), dtype=INDEX)
        for r in range(perms.shape[0]):
            for dofPerm in range(self.DoFMap.dofs_per_element):
                for dofOrig in range(self.DoFMap.dofs_per_element):
                    for j in range(dim+1):
                        if abs(self.DoFMap.nodes[dofPerm, j]-self.DoFMap.nodes[dofOrig, perms[r, j]]) > eps:
                            break
                    else:
                        self.precomputedDoFPermutations[r, dofPerm] = dofOrig
                        break
                else:
                    # We should never get here
                    raise NotImplementedError()

    cdef void precomputeSimplices(self):
        # mesh1 and mesh 2 will be the same
        cdef:
            INDEX_t cellNo1
            INDEX_t m, k, l
            REAL_t fac = 1./self.cells1.shape[1]
        self.precomputedSimplices = uninitialized((self.cells1.shape[0], self.cells1.shape[1], self.dim), dtype=REAL)
        self.precomputedCenters = np.zeros((self.cells1.shape[0], self.dim), dtype=REAL)
        for cellNo1 in range(self.cells1.shape[0]):
            for m in range(self.cells1.shape[1]):
                k = self.cells1[cellNo1, m]
                for l in range(self.vertices1.shape[1]):
                    self.precomputedSimplices[cellNo1, m, l] = self.vertices1[k, l]
                    self.precomputedCenters[cellNo1, l] += self.vertices1[k, l]
            for l in range(self.vertices1.shape[1]):
                self.precomputedCenters[cellNo1, l] *= fac

    cdef INDEX_t getCellPairIdentifierSize(self):
        return -1

    cdef void computeCellPairIdentifierBase(self, INDEX_t[::1] ID, INDEX_t *perm):
        raise NotImplementedError()

    cdef void computeCellPairIdentifier(self, INDEX_t[::1] ID, INDEX_t *perm):
        self.computeCellPairIdentifierBase(ID, perm)

    def computeCellPairIdentifier_py(self):
        cdef:
            INDEX_t perm = 0
        ID = uninitialized((self.getCellPairIdentifierSize()), dtype=INDEX)
        self.computeCellPairIdentifier(ID, &perm)
        return ID, perm

    cdef void setMesh1(self, meshBase mesh1):
        self.setVerticesCells1(mesh1.vertices, mesh1.cells)
        self.precomputedVolumes = mesh1.volVector
        self.precomputedH = mesh1.hVector
        h1 = 2.*mesh1.h
        d = 2.*mesh1.diam
        self.h1MaxInv = 1./h1
        self.dMaxInv = 1./d

    cdef void setVerticesCells1(self, REAL_t[:, ::1] vertices1, INDEX_t[:, ::1] cells1):
        self.vertices1 = vertices1
        self.cells1 = cells1
        self.simplex1 = uninitialized((self.cells1.shape[1], self.dim), dtype=REAL)
        self.perm1 = uninitialized((self.cells1.shape[1]), dtype=INDEX)
        self.cellNo1 = -1
        self.cellNo2 = -1
        if self.symmetricCells:
            # mesh1 and mesh 2 will be the same
            self.precomputeSimplices()

    cdef void setMesh2(self, meshBase mesh2):
        self.setVerticesCells2(mesh2.vertices, mesh2.cells)
        if mesh2.manifold_dim > 0:
            h2 = 2.*mesh2.h
            self.h2MaxInv = 1./h2
        else:
            self.h2MaxInv = 1.

    cdef void setVerticesCells2(self, REAL_t[:, ::1] vertices2, INDEX_t[:, ::1] cells2):
        self.vertices2 = vertices2
        self.cells2 = cells2
        self.simplex2 = uninitialized((self.cells2.shape[1], self.dim), dtype=REAL)
        self.perm2 = uninitialized((self.cells2.shape[1]), dtype=INDEX)
        self.perm = uninitialized((2*self.DoFMap.dofs_per_element), dtype=INDEX)
        self.cellNo1 = -1
        self.cellNo2 = -1

    cdef void setCell1(self, INDEX_t cellNo1):
        if self.cellNo1 == cellNo1:
            return
        self.cellNo1 = cellNo1
        if not self.symmetricCells:
            getSimplexAndCenter(self.cells1, self.vertices1, self.cellNo1, self.simplex1, self.center1)
            self.vol1 = self.volume1(self.simplex1)
        else:
            self.simplex1 = self.precomputedSimplices[cellNo1, :, :]
            self.center1 = self.precomputedCenters[cellNo1, :]
            self.vol1 = self.precomputedVolumes[cellNo1]

    cdef void setCell2(self, INDEX_t cellNo2):
        if self.cellNo2 == cellNo2:
            return
        self.cellNo2 = cellNo2
        if not self.symmetricCells:
            getSimplexAndCenter(self.cells2, self.vertices2, self.cellNo2, self.simplex2, self.center2)
            self.vol2 = self.volume2(self.simplex2)
        else:
            self.simplex2 = self.precomputedSimplices[cellNo2, :, :]
            self.center2 = self.precomputedCenters[cellNo2, :]
            self.vol2 = self.precomputedVolumes[cellNo2]

    def setMesh1_py(self, meshBase mesh1):
        self.setMesh1(mesh1)

    def setMesh2_py(self, meshBase mesh2):
        self.setMesh2(mesh2)

    def setCell1_py(self, INDEX_t cellNo1):
        self.setCell1(cellNo1)

    def setCell2_py(self, INDEX_t cellNo2):
        self.setCell2(cellNo2)

    cdef void swapCells(self):
        self.cellNo1, self.cellNo2 = self.cellNo2, self.cellNo1
        self.simplex1, self.simplex2 = self.simplex2, self.simplex1
        self.center1, self.center2 = self.center2, self.center1

    cdef void setSimplex1(self, REAL_t[:, ::1] simplex1):
        self.simplex1 = simplex1
        self.getSimplexCenter(self.simplex1, self.center1)
        self.vol1 = self.volume1(self.simplex1)

    cdef void setSimplex2(self, REAL_t[:, ::1] simplex2):
        self.simplex2 = simplex2
        self.getSimplexCenter(self.simplex2, self.center2)
        self.vol2 = self.volume2(self.simplex2)

    def __call__(self,
                 {SCALAR}_t[:, ::1] contrib,
                 panelType panel):
        return self.eval(contrib, panel)

    cdef void eval(self,
                   {SCALAR}_t[:, ::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        raise NotImplementedError()

    def eval_py(self,
                {SCALAR}_t[:, ::1] contrib,
                panel):
        self.eval(contrib, panel, ALL)

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        raise NotImplementedError()

    cdef REAL_t get_h_simplex(self, const REAL_t[:, ::1] simplex):
        raise NotImplementedError()

    cdef REAL_t get_h_surface_simplex(self, const REAL_t[:, ::1] simplex):
        raise NotImplementedError()

    cdef void getSimplexCenter(self,
                               const REAL_t[:, ::1] simplex,
                               REAL_t[::1] center):
        cdef:
            INDEX_t i, j
            REAL_t fac
        center[:] = 0.
        for i in range(simplex.shape[0]):
            for j in range(simplex.shape[1]):
                center[j] += simplex[i, j]
        fac = 1./simplex.shape[0]
        for j in range(simplex.shape[1]):
            center[j] *= fac

    cdef panelType getProtoPanelType(self):
        # Given two cells, determines their relationship:
        # - COMMON_FACE
        # - COMMON_EDGE
        # - COMMON_VERTEX
        # - DISTANT
        cdef:
            INDEX_t mask1 = 0, mask2 = 0
            INDEX_t numVertices1 = self.cells1.shape[1]
            INDEX_t numVertices2 = self.cells2.shape[1]
            INDEX_t vertexNo1, vertexNo2, vertex1, vertex2
            INDEX_t commonVertices = 0
            INDEX_t k, i
            INDEX_t dofs_per_vertex, dofs_per_edge, dofs_per_face, dofs_per_element = self.DoFMap.dofs_per_element
            panelType panel
            INDEX_t chosenPermutation
        if self.symmetricCells:
            if self.cellNo1 > self.cellNo2:
                return IGNORED

        if (self.cells1.shape[1] == self.cells2.shape[1]) and (self.cellNo1 == self.cellNo2):
            for k in range(numVertices1):
                self.perm1[k] = k
            for k in range(numVertices2):
                self.perm2[k] = k
            for k in range(dofs_per_element):
                self.perm[k] = k
            return self.IDENTICAL

        # now the two simplices can share at most numVertices1-1 vertices

        for vertexNo1 in range(numVertices1):
            vertex1 = self.cells1[self.cellNo1, vertexNo1]
            for vertexNo2 in range(numVertices2):
                if mask2 & (1 << vertexNo2):
                    continue
                vertex2 = self.cells2[self.cellNo2, vertexNo2]
                if vertex1 == vertex2:
                    self.perm1[commonVertices] = vertexNo1
                    self.perm2[commonVertices] = vertexNo2
                    mask1 += (1 << vertexNo1)
                    mask2 += (1 << vertexNo2)
                    commonVertices += 1
                    break

        if commonVertices == 0:
            for k in range(numVertices1):
                self.perm1[k] = k
            for k in range(numVertices2):
                self.perm2[k] = k
            for k in range(dofs_per_element):
                self.perm[k] = k
            return 0
        else:
            i = 0
            for k in range(commonVertices, numVertices1):
                while mask1 & (1 << i):
                    i += 1
                self.perm1[k] = i
                mask1 += (1 << i)

            i = 0
            for k in range(commonVertices, numVertices2):
                while mask2 & (1 << i):
                    i += 1
                self.perm2[k] = i
                mask2 += (1 << i)

        # we now have set permutations for the two simplices
        # we have at least one shared vertex

        chosenPermutation = self.pI_volume.rank(self.perm1)
        for k in range(dofs_per_element):
            self.perm[k] = self.precomputedDoFPermutations[chosenPermutation, k]

        if numVertices1 == numVertices2:
            dofs_per_vertex = self.DoFMap.dofs_per_vertex
            dofs_per_edge = self.DoFMap.dofs_per_edge
            dofs_per_face = self.DoFMap.dofs_per_face

            chosenPermutation = self.pI_volume.rank(self.perm2)
            if commonVertices == 1:
                for k in range(dofs_per_vertex, dofs_per_element):
                    self.perm[dofs_per_element+k-dofs_per_vertex] = dofs_per_element+self.precomputedDoFPermutations[chosenPermutation, k]
            elif commonVertices == 2:
                for k in range(2*dofs_per_vertex, numVertices2*dofs_per_vertex):
                    self.perm[dofs_per_element+k-2*dofs_per_vertex] = dofs_per_element+self.precomputedDoFPermutations[chosenPermutation, k]
                for k in range(numVertices2*dofs_per_vertex+dofs_per_edge, dofs_per_element):
                    self.perm[dofs_per_element+k-2*dofs_per_vertex-dofs_per_edge] = dofs_per_element+self.precomputedDoFPermutations[chosenPermutation, k]
            elif commonVertices == 3:
                # only in 3d
                for k in range(3*dofs_per_vertex, numVertices2*dofs_per_vertex):
                    self.perm[dofs_per_element+k-3*dofs_per_vertex] = dofs_per_element+self.precomputedDoFPermutations[chosenPermutation, k]
                for k in range(numVertices2*dofs_per_vertex+3*dofs_per_edge, numVertices2*dofs_per_vertex+6*dofs_per_edge):
                    self.perm[dofs_per_element+k-3*dofs_per_vertex-3*dofs_per_edge] = dofs_per_element+self.precomputedDoFPermutations[chosenPermutation, k]
                for k in range(numVertices2*dofs_per_vertex+6*dofs_per_edge+dofs_per_face, dofs_per_element):
                    self.perm[dofs_per_element+k-3*dofs_per_vertex-3*dofs_per_edge-dofs_per_face] = dofs_per_element+self.precomputedDoFPermutations[chosenPermutation, k]
        panel = -commonVertices
        return panel

    cdef void computeCenterDistance(self):
        cdef:
            INDEX_t j
            REAL_t d2 = 0.
        for j in range(self.dim):
            d2 += (self.center1[j]-self.center2[j])**2
        self.dcenter2 = d2

    cdef void computeExtremeDistances(self):
        cdef:
            INDEX_t i, k, j
            INDEX_t noSimplex1 = self.simplex1.shape[0]
            INDEX_t noSimplex2 = self.simplex2.shape[0]
            REAL_t d2
            REAL_t dmin2 = inf
            REAL_t dmax2 = 0.
        for i in range(noSimplex1):
            for k in range(noSimplex2):
                d2 = 0.
                for j in range(self.dim):
                    d2 += (self.simplex1[i, j] - self.simplex2[k, j])**2
                dmin2 = min(dmin2, d2)
                dmax2 = max(dmax2, d2)
        self.dmin2 = dmin2
        self.dmax2 = dmax2

    cpdef panelType getPanelType(self):
        raise NotImplementedError()

    cdef void addQuadRule(self, panelType panel):
        raise NotImplementedError()

    def addQuadRule_py(self, panelType panel):
        self.addQuadRule(panel)

    def __repr__(self):
        return '{}\n'.format(self.__class__.__name__)


cdef class {SCALAR_label}nonlocalOperator({SCALAR_label}double_local_matrix_t):
    def __init__(self,
                 {SCALAR_label}Kernel kernel,
                 meshBase mesh, DoFMap dm,
                 num_dofs=None, INDEX_t manifold_dim2=-1):
        cdef:
            shapeFunction sf
            INDEX_t i
        if manifold_dim2 < 0:
            manifold_dim2 = mesh.manifold_dim
        {SCALAR_label}double_local_matrix_t.__init__(self, mesh.dim, mesh.manifold_dim, manifold_dim2, dm)
        if num_dofs is None:
            self.num_dofs = dm.num_dofs
        else:
            self.num_dofs = num_dofs
        self.hmin = mesh.hmin
        self.H0 = mesh.diam/sqrt(8)
        self.localShapeFunctions = <void**>malloc(self.DoFMap.dofs_per_element*sizeof(void*))
        for i in range(self.DoFMap.dofs_per_element):
            sf = dm.localShapeFunctions[i]
            self.localShapeFunctions[i] = <void*>sf
        self.specialQuadRules = {}
        self.distantQuadRulesPtr = <void**>malloc(MAX_PANEL*sizeof(void*))
        for i in range(MAX_PANEL):
            self.distantQuadRulesPtr[i] = NULL

        self.x = uninitialized((0, self.dim), dtype=REAL)
        self.y = uninitialized((0, self.dim), dtype=REAL)

        self.vec = uninitialized((kernel.valueSize), dtype={SCALAR})
        self.vec2 = uninitialized((kernel.valueSize), dtype={SCALAR})
        self.temp = uninitialized((0, kernel.valueSize), dtype={SCALAR})

        self.n = uninitialized((self.dim), dtype=REAL)
        self.w = uninitialized((self.dim), dtype=REAL)

        self.kernel = kernel

        if self.kernel.variable:
            self.symmetricCells = self.kernel.symmetric
            self.symmetricLocalMatrix = self.kernel.symmetric
        else:
            self.symmetricCells = True
            self.symmetricLocalMatrix = True

        if self.kernel.variableHorizon:
            self.symmetricCells = False

    def __del__(self):
        free(self.localShapeFunctions)
        free(self.distantQuadRulesPtr)

    cdef void getNearQuadRule(self, panelType panel):
        raise NotImplementedError()

    cdef void computeCellPairIdentifier(self, INDEX_t[::1] ID, INDEX_t *perm):
        assert not self.kernel.variable
        if self.kernel.finiteHorizon:
            self.computeExtremeDistances()
            if self.dmax2 <= self.kernel.getHorizonValue2():
                # entirely within horizon
                self.computeCellPairIdentifierBase(ID, perm)
            elif self.dmin2 >= self.kernel.getHorizonValue2():
                # entirely outside of horizon
                ID[0] = IGNORED
            else:
                # on horizon
                ID[0] = ON_HORIZON
        else:
            self.computeCellPairIdentifierBase(ID, perm)

    cpdef panelType getPanelType(self):
        # Given two cells, determines their relationship:
        # - COMMON_FACE
        # - COMMON_EDGE
        # - COMMON_VERTEX
        # - DISTANT
        # - IGNORED
        cdef:
            panelType panel
            REAL_t d, h1, h2
            REAL_t alpha
        panel = self.getProtoPanelType()

        if panel == IGNORED:
            return IGNORED

        if self.kernel.variable:
            if self.kernel.piecewise:
                self.kernel.evalParams(self.center1, self.center2)
            else:
                self.kernel.evalParamsOnSimplices(self.center1, self.center2, self.simplex1, self.simplex2)

        if panel == DISTANT:
            if self.kernel.interaction.getRelativePosition(self.simplex1, self.simplex2) == REMOTE:
                return IGNORED

            self.computeCenterDistance()
            d = sqrt(self.dcenter2)

            if self.symmetricCells:
                h1 = self.precomputedH[self.cellNo1]
            else:
                h1 = self.get_h_simplex(self.simplex1)
            if self.cells1.shape[1] == self.cells2.shape[1]:
                if self.symmetricCells:
                    h2 = self.precomputedH[self.cellNo2]
                else:
                    h2 = self.get_h_simplex(self.simplex2)
            else:
                h2 = self.get_h_surface_simplex(self.simplex2)
            panel = self.getQuadOrder(h1, h2, d)
        elif self.kernel.variable:
            self.getNearQuadRule(panel)
        if panel < 0:
            alpha = self.kernel.getSingularityValue()
            if alpha == 0.:
                self.kernel.interaction.getRelativePosition(self.simplex1, self.simplex2)
        return panel

    def __repr__(self):
        return (super(nonlocalOperator, self).__repr__() +
                'kernel:                        {}\n'.format(self.kernel))

    cdef inline shapeFunction getLocalShapeFunction(self, INDEX_t local_dof):
        return <shapeFunction>self.localShapeFunctions[local_dof]

    cdef void addQuadRule(self, panelType panel):
        cdef:
            simplexQuadratureRule qr0, qr1
            doubleSimplexQuadratureRule qr2
            specialQuadRule sQR
            REAL_t[:, ::1] PSI
            INDEX_t I, k, i, j, m
            INDEX_t numQuadNodes0, numQuadNodes1, dofs_per_element
            shapeFunction sf
            REAL_t lcl_bary_x[4]
            REAL_t lcl_bary_y[4]
            REAL_t phi_x, phi_y
        qr0 = simplexXiaoGimbutas(panel, self.dim, self.manifold_dim1)
        qr1 = qr0
        qr2 = doubleSimplexQuadratureRule(qr0, qr1)
        numQuadNodes0 = qr0.num_nodes
        numQuadNodes1 = qr1.num_nodes
        dofs_per_element = self.DoFMap.dofs_per_element
        PSI = uninitialized((2*dofs_per_element,
                             qr2.num_nodes), dtype=REAL)
        # phi_i(x) - phi_i(y) = phi_i(x)
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    for m in range(self.dim+1):
                        lcl_bary_x[m] = qr0.nodes[m, i]
                    sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                    PSI[I, k] = phi_x
                    k += 1
        # phi_i(x) - phi_i(y) = -phi_i(y)
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    for m in range(self.dim+1):
                        lcl_bary_y[m] = qr1.nodes[m, j]
                    sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                    PSI[I+dofs_per_element, k] = -phi_y
                    k += 1
        sQR = specialQuadRule(qr2, PSI)
        self.distantQuadRules[panel] = sQR
        self.distantQuadRulesPtr[panel] = <void*>(self.distantQuadRules[panel])

        if numQuadNodes0 > self.x.shape[0]:
            self.x = uninitialized((numQuadNodes0, self.dim), dtype=REAL)
        if numQuadNodes1 > self.y.shape[0]:
            self.y = uninitialized((numQuadNodes1, self.dim), dtype=REAL)
        if numQuadNodes0*numQuadNodes1 > self.temp.shape[0]:
            self.temp = uninitialized((numQuadNodes0*numQuadNodes1, self.kernel.valueSize), dtype={SCALAR})

    cdef void addQuadRule_nonSym(self, panelType panel):
        cdef:
            simplexQuadratureRule qr0, qr1
            doubleSimplexQuadratureRule qr2
            specialQuadRule sQR
            REAL_t[:, ::1] PSI
            REAL_t[:, :, ::1] PHI
            INDEX_t I, k, i, j, m
            INDEX_t numQuadNodes0, numQuadNodes1, dofs_per_element
            shapeFunction sf
            REAL_t lcl_bary_x[4]
            REAL_t lcl_bary_y[4]
            REAL_t phi_x, phi_y
        qr0 = simplexXiaoGimbutas(panel, self.dim, self.manifold_dim1)
        qr1 = qr0
        qr2 = doubleSimplexQuadratureRule(qr0, qr1)
        numQuadNodes0 = qr0.num_nodes
        numQuadNodes1 = qr1.num_nodes
        dofs_per_element = self.DoFMap.dofs_per_element
        PSI = uninitialized((2*dofs_per_element,
                             qr2.num_nodes), dtype=REAL)
        PHI = uninitialized((2,
                             2*dofs_per_element,
                             qr2.num_nodes), dtype=REAL)
        # phi_i(x) - phi_i(y) = phi_i(x)
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    for m in range(self.dim+1):
                        lcl_bary_x[m] = qr0.nodes[m, i]
                    sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                    PHI[0, I, k] = phi_x
                    PHI[1, I, k] = 0.
                    PSI[I, k] = PHI[0, I, k]
                    k += 1
        # phi_i(x) - phi_i(y) = -phi_i(y)
        for I in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(I)
            k = 0
            for i in range(numQuadNodes0):
                for j in range(numQuadNodes1):
                    for m in range(self.dim+1):
                        lcl_bary_y[m] = qr1.nodes[m, j]
                    sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                    PHI[0, I+dofs_per_element, k] = 0.
                    PHI[1, I+dofs_per_element, k] = phi_y
                    PSI[I+dofs_per_element, k] = -PHI[1, I+dofs_per_element, k]
                    k += 1
        sQR = specialQuadRule(qr2, PSI, PHI3=PHI)
        self.distantQuadRules[panel] = sQR
        self.distantQuadRulesPtr[panel] = <void*>(self.distantQuadRules[panel])

        if numQuadNodes0 > self.x.shape[0]:
            self.x = uninitialized((numQuadNodes0, self.dim), dtype=REAL)
        if numQuadNodes1 > self.y.shape[0]:
            self.y = uninitialized((numQuadNodes1, self.dim), dtype=REAL)
        if numQuadNodes0*numQuadNodes1 > self.temp.shape[0]:
            self.temp = uninitialized((numQuadNodes0*numQuadNodes1, self.kernel.valueSize), dtype={SCALAR})
            self.temp2 = uninitialized((numQuadNodes0*numQuadNodes1, self.kernel.valueSize), dtype={SCALAR})

    cdef void getNonSingularNearQuadRule(self, panelType panel):
        cdef:
            simplexQuadratureRule qr0, qr1
            doubleSimplexQuadratureRule qr2
            specialQuadRule sQR
            REAL_t[:, ::1] PSI
            INDEX_t I, k, i, j, m
            INDEX_t numQuadNodes0, numQuadNodes1, dofs_per_element
            shapeFunction sf
            REAL_t lcl_bary_x[4]
            REAL_t lcl_bary_y[4]
            REAL_t phi_x, phi_y
        try:
            sQR = <specialQuadRule>(self.distantQuadRules[MAX_PANEL+panel])
        except KeyError:
            quadOrder = <panelType>max(ceil(self.target_order), 2)
            qr0 = simplexXiaoGimbutas(quadOrder, self.dim)
            qr1 = qr0
            qr2 = doubleSimplexQuadratureRule(qr0, qr1)
            numQuadNodes0 = qr0.num_nodes
            numQuadNodes1 = qr1.num_nodes
            dofs_per_element = self.DoFMap.dofs_per_element
            PSI = uninitialized((2*dofs_per_element,
                                 qr2.num_nodes),
                                dtype=REAL)

            # phi_i(x) - phi_i(y) = phi_i(x)
            for I in range(dofs_per_element):
                sf = self.getLocalShapeFunction(I)
                k = 0
                for i in range(numQuadNodes0):
                    for j in range(numQuadNodes1):
                        for m in range(self.dim+1):
                            lcl_bary_x[m] = qr0.nodes[m, i]
                        sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                        PSI[I, k] = phi_x
                        k += 1
            # phi_i(x) - phi_i(y) = -phi_i(y)
            for I in range(dofs_per_element):
                sf = self.getLocalShapeFunction(I)
                k = 0
                for i in range(numQuadNodes0):
                    for j in range(numQuadNodes1):
                        for m in range(self.dim+1):
                            lcl_bary_y[m] = qr1.nodes[m, j]
                        sf.evalPtr(&lcl_bary_y[0], NULL, &phi_y)
                        PSI[I+dofs_per_element, k] = -phi_y
                        k += 1
            sQR = specialQuadRule(qr2, PSI)
            self.distantQuadRules[MAX_PANEL+panel] = sQR
            self.distantQuadRulesPtr[MAX_PANEL+panel] = <void*>(self.distantQuadRules[MAX_PANEL+panel])
            if numQuadNodes0 > self.x.shape[0]:
                self.x = uninitialized((numQuadNodes0, self.dim), dtype=REAL)
            if numQuadNodes1 > self.y.shape[0]:
                self.y = uninitialized((numQuadNodes1, self.dim), dtype=REAL)
            if qr2.num_nodes > self.temp.shape[0]:
                self.temp = uninitialized((qr2.num_nodes, self.kernel.valueSize), dtype={SCALAR})

    cdef void eval_distant(self,
                           {SCALAR}_t[:, ::1] contrib,
                           panelType panel,
                           MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J, l
            REAL_t vol, vol1 = self.vol1, vol2 = self.vol2
            {SCALAR}_t val
            doubleSimplexQuadratureRule qr2
            REAL_t[:, ::1] PSI
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            INDEX_t dim = simplex1.shape[1]
            REAL_t c1, c2, PSI_I, PSI_J
            transformQuadratureRule qr0trans, qr1trans
            INDEX_t dofs_per_element, numQuadNodes0, numQuadNodes1
            REAL_t a_b1[3]
            REAL_t a_A1[3][3]
            REAL_t a_A2[3][3]
            REAL_t[::1] b1
            REAL_t[:, ::1] A1, A2
            BOOL_t cutElements = False
            INDEX_t valueSize = self.kernel.valueSize

        if self.kernel.finiteHorizon:
            # check if the horizon might cut the elements
            if self.kernel.interaction.relPos == CUT:
                cutElements = True
            if self.kernel.complement:
                cutElements = False
                # TODO: cutElements should be set to True, but
                #       need to figure out the element
                #       transformation.

        contrib[:] = 0.

        if not cutElements:
            vol = vol1*vol2
            if panel < 0:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[MAX_PANEL+panel])
            else:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[panel])
            qr2 = <doubleSimplexQuadratureRule>(sQR.qr)
            PSI = sQR.PSI
            qr2.rule1.nodesInGlobalCoords(simplex1, self.x)
            qr2.rule2.nodesInGlobalCoords(simplex2, self.y)

            k = 0
            for i in range(qr2.rule1.num_nodes):
                for j in range(qr2.rule2.num_nodes):
                    self.kernel.evalPtr(dim,
                                        &self.x[i, 0],
                                        &self.y[j, 0],
                                        &self.vec[0])
                    for l in range(valueSize):
                        self.temp[k, l] = qr2.weights[k]*self.vec[l]
                    k += 1

            k = 0
            for I in range(2*self.DoFMap.dofs_per_element):
                for J in range(I, 2*self.DoFMap.dofs_per_element):
                    if mask[k]:
                        for l in range(valueSize):
                            val = 0.
                            for i in range(qr2.num_nodes):
                                val += self.temp[i, l] * PSI[I, i] * PSI[J, i]
                            contrib[k, l] = val*vol
                    k += 1
        else:
            if panel < 0:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[MAX_PANEL+panel])
            else:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[panel])
            qr2 = <doubleSimplexQuadratureRule>(sQR.qr)
            if sQR.qrTransformed0 is not None:
                qr0trans = sQR.qrTransformed0
            else:
                qr0 = qr2.rule1
                qr0trans = transformQuadratureRule(qr0)
                sQR.qrTransformed0 = qr0trans
            if sQR.qrTransformed1 is not None:
                qr1trans = sQR.qrTransformed1
            else:
                qr1 = qr2.rule2
                qr1trans = transformQuadratureRule(qr1)
                sQR.qrTransformed1 = qr1trans

            numQuadNodes0 = qr0trans.num_nodes
            numQuadNodes1 = qr1trans.num_nodes

            vol = vol1*vol2
            dofs_per_element = self.DoFMap.dofs_per_element

            A1 = a_A1
            b1 = a_b1
            A2 = a_A2

            self.kernel.interaction.startLoopSubSimplices_Simplex(simplex1, simplex2)
            while self.kernel.interaction.nextSubSimplex_Simplex(A1, b1, &c1):
                qr0trans.setAffineBaryTransform(A1, b1)
                qr0trans.nodesInGlobalCoords(simplex1, self.x)
                for i in range(qr0trans.num_nodes):
                    self.kernel.interaction.startLoopSubSimplices_Node(self.x[i, :], simplex2)
                    while self.kernel.interaction.nextSubSimplex_Node(A2, &c2):
                        qr1trans.setLinearBaryTransform(A2)
                        qr1trans.nodesInGlobalCoords(simplex2, self.y)
                        for j in range(qr1trans.num_nodes):
                            self.kernel.evalPtr(dim, &self.x[i, 0], &self.y[j, 0], &self.vec[0])
                            val = qr0trans.weights[i]*qr1trans.weights[j]*self.vec[0]
                            val *= c1 * c2 * vol
                            k = 0
                            for I in range(2*dofs_per_element):
                                if I < dofs_per_element:
                                    self.getLocalShapeFunction(I).evalStrided(&qr0trans.nodes[0, i], NULL, numQuadNodes0, &PSI_I)
                                else:
                                    self.getLocalShapeFunction(I-dofs_per_element).evalStrided(&qr1trans.nodes[0, j], NULL, numQuadNodes1, &PSI_I)
                                    PSI_I *= -1
                                for J in range(I, 2*dofs_per_element):
                                    if mask[k]:
                                        if J < dofs_per_element:
                                            self.getLocalShapeFunction(J).evalStrided(&qr0trans.nodes[0, i], NULL, numQuadNodes0, &PSI_J)
                                        else:
                                            self.getLocalShapeFunction(J-dofs_per_element).evalStrided(&qr1trans.nodes[0, j], NULL, numQuadNodes1, &PSI_J)
                                            PSI_J *= -1
                                        contrib[k, 0] += val * PSI_I*PSI_J
                                    k += 1

    cdef void eval_distant_nonsym(self,
                                  {SCALAR}_t[:, ::1] contrib,
                                  panelType panel,
                                  MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J, l
            REAL_t vol, vol1 = self.vol1, vol2 = self.vol2
            {SCALAR}_t val, val2
            doubleSimplexQuadratureRule qr2
            REAL_t[:, ::1] PSI
            REAL_t[:, :, ::1] PHI
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            INDEX_t dim = simplex1.shape[1]
            BOOL_t cutElements = False
            REAL_t w
            REAL_t c1, c2, PHI_I_0, PHI_I_1, PSI_J
            transformQuadratureRule qr0trans, qr1trans
            INDEX_t dofs_per_element, numQuadNodes0, numQuadNodes1
            REAL_t a_b1[3]
            REAL_t a_A1[3][3]
            REAL_t a_A2[3][3]
            REAL_t[::1] b1
            REAL_t[:, ::1] A1, A2
            INDEX_t valueSize = self.kernel.valueSize

        if self.kernel.finiteHorizon:
            # check if the horizon might cut the elements
            if self.kernel.interaction.relPos == CUT:
                cutElements = True
            if self.kernel.complement:
                cutElements = False
                # TODO: cutElements should be set to True, but
                #       need to figure out the element
                #       transformation.

        contrib[:, :] = 0.

        if not cutElements:
            vol = vol1*vol2
            if panel < 0:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[MAX_PANEL+panel])
            else:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[panel])
            qr2 = <doubleSimplexQuadratureRule>(sQR.qr)
            PSI = sQR.PSI
            PHI = sQR.PHI3
            qr2.rule1.nodesInGlobalCoords(simplex1, self.x)
            qr2.rule2.nodesInGlobalCoords(simplex2, self.y)

            k = 0
            for i in range(qr2.rule1.num_nodes):
                for j in range(qr2.rule2.num_nodes):
                    w = qr2.weights[k]
                    self.kernel.evalPtr(dim,
                                        &self.x[i, 0],
                                        &self.y[j, 0],
                                        &self.vec[0])
                    self.kernel.evalPtr(dim,
                                        &self.y[j, 0],
                                        &self.x[i, 0],
                                        &self.vec2[0])
                    for l in range(valueSize):
                        self.temp[k, l] = w * self.vec[l]
                        self.temp2[k, l] = w * self.vec2[l]
                    k += 1

            k = 0
            for I in range(2*self.DoFMap.dofs_per_element):
                for J in range(2*self.DoFMap.dofs_per_element):
                    if mask[k]:
                        for l in range(valueSize):
                            val = 0.
                            for i in range(qr2.num_nodes):
                                val += (self.temp[i, l] * PHI[0, I, i] - self.temp2[i, l] * PHI[1, I, i]) * PSI[J, i]
                            contrib[k, l] = val*vol
                    k += 1
        else:
            if panel < 0:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[MAX_PANEL+panel])
            else:
                sQR = <specialQuadRule>(self.distantQuadRulesPtr[panel])
            qr2 = <doubleSimplexQuadratureRule>(sQR.qr)
            if sQR.qrTransformed0 is not None:
                qr0trans = sQR.qrTransformed0
            else:
                qr0 = qr2.rule1
                qr0trans = transformQuadratureRule(qr0)
                sQR.qrTransformed0 = qr0trans
            if sQR.qrTransformed1 is not None:
                qr1trans = sQR.qrTransformed1
            else:
                qr1 = qr2.rule2
                qr1trans = transformQuadratureRule(qr1)
                sQR.qrTransformed1 = qr1trans

            numQuadNodes0 = qr0trans.num_nodes
            numQuadNodes1 = qr1trans.num_nodes

            vol = vol1*vol2
            dofs_per_element = self.DoFMap.dofs_per_element

            A1 = a_A1
            b1 = a_b1
            A2 = a_A2

            self.kernel.interaction.startLoopSubSimplices_Simplex(simplex1, simplex2)
            while self.kernel.interaction.nextSubSimplex_Simplex(A1, b1, &c1):
                qr0trans.setAffineBaryTransform(A1, b1)
                qr0trans.nodesInGlobalCoords(simplex1, self.x)
                for i in range(qr0trans.num_nodes):
                    self.kernel.interaction.startLoopSubSimplices_Node(self.x[i, :], simplex2)
                    while self.kernel.interaction.nextSubSimplex_Node(A2, &c2):
                        qr1trans.setLinearBaryTransform(A2)
                        qr1trans.nodesInGlobalCoords(simplex2, self.y)
                        for j in range(qr1trans.num_nodes):
                            w = qr0trans.weights[i]*qr1trans.weights[j]*c1 * c2 * vol
                            self.kernel.evalPtr(dim, &self.x[i, 0], &self.y[j, 0], &self.vec[0])
                            self.kernel.evalPtr(dim, &self.y[j, 0], &self.x[i, 0], & self.vec2[0])
                            val = w*self.vec[0]
                            val2 = w*self.vec2[0]
                            k = 0
                            for I in range(2*dofs_per_element):
                                if I < dofs_per_element:
                                    self.getLocalShapeFunction(I).evalStrided(&qr0trans.nodes[0, i], NULL, numQuadNodes0, &PHI_I_0)
                                    PHI_I_1 = 0.
                                else:
                                    PHI_I_0 = 0.
                                    self.getLocalShapeFunction(I-dofs_per_element).evalStrided(&qr1trans.nodes[0, j], NULL, numQuadNodes1, &PHI_I_1)
                                for J in range(2*dofs_per_element):
                                    if mask[k]:
                                        if J < dofs_per_element:
                                            self.getLocalShapeFunction(J).evalStrided(&qr0trans.nodes[0, i], NULL, numQuadNodes0, &PSI_J)
                                        else:
                                            self.getLocalShapeFunction(J-dofs_per_element).evalStrided(&qr1trans.nodes[0, j], NULL, numQuadNodes1, &PSI_J)
                                            PSI_J *= -1
                                        contrib[k, 0] += (val * PHI_I_0 - val2 * PHI_I_1) * PSI_J
                                    k += 1

    cdef void addQuadRule_boundary(self, panelType panel):
        cdef:
            simplexQuadratureRule qr0, qr1
            doubleSimplexQuadratureRule qr2
            specialQuadRule sQR
            REAL_t[:, ::1] PHI
            INDEX_t i, j, k, l, m
            REAL_t lcl_bary_x[4]
            shapeFunction sf
            REAL_t phi_x
        qr0 = simplexXiaoGimbutas(panel, self.dim)
        qr1 = simplexDuffyTransformation(panel, self.dim, self.dim-1)
        qr2 = doubleSimplexQuadratureRule(qr0, qr1)
        PHI = uninitialized((self.DoFMap.dofs_per_element, qr2.num_nodes), dtype=REAL)
        for i in range(self.DoFMap.dofs_per_element):
            sf = self.getLocalShapeFunction(i)
            for j in range(qr2.rule1.num_nodes):
                for k in range(qr2.rule2.num_nodes):
                    l = j*qr2.rule2.num_nodes+k
                    for m in range(self.dim+1):
                        lcl_bary_x[m] = qr2.rule1.nodes[m, j]
                    sf.evalPtr(&lcl_bary_x[0], NULL, &phi_x)
                    PHI[i, l] = phi_x
        sQR = specialQuadRule(qr2, PHI=PHI)
        self.distantQuadRules[panel] = sQR
        self.distantQuadRulesPtr[panel] = <void*>(self.distantQuadRules[panel])

        if qr2.rule1.num_nodes > self.x.shape[0]:
            self.x = uninitialized((qr2.rule1.num_nodes, self.dim), dtype=REAL)
        if qr2.rule2.num_nodes > self.y.shape[0]:
            self.y = uninitialized((qr2.rule2.num_nodes, self.dim), dtype=REAL)
        if qr2.num_nodes > self.temp.shape[0]:
            self.temp = uninitialized((qr2.num_nodes, self.kernel.valueSize), dtype=REAL)

    cdef void eval_distant_boundary(self,
                                    {SCALAR}_t[:, ::1] contrib,
                                    panelType panel,
                                    MASK_t mask=ALL):
        cdef:
            INDEX_t k, m, i, j, I, J, l
            REAL_t vol, valReal, vol1 = self.vol1, vol2 = self.vol2
            {SCALAR}_t val
            doubleSimplexQuadratureRule qr2
            REAL_t[:, ::1] PHI
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            INDEX_t dim = simplex1.shape[1]
            REAL_t normW, nw
            INDEX_t valueSize = self.kernel.valueSize

        # Kernel:
        #  \Gamma(x,y) = n \dot (x-y) * C(d,s) / (2s) / |x-y|^{d+2s}
        # with inward normal n.
        #
        # Rewrite as
        #  \Gamma(x,y) = [ n \dot (x-y)/|x-y| ] * [ C(d,s) / (2s) / |x-y|^{d-1+2s} ]
        #                                         \--------------------------------/
        #                                                 |
        #                                           boundaryKernel
        #
        # n is independent of x and y
        if dim == 2:
            self.n[0] = simplex2[1, 1] - simplex2[0, 1]
            self.n[1] = simplex2[0, 0] - simplex2[1, 0]
            # F is same as vol2
            valReal = 1./sqrt(mydot(self.n, self.n))
            self.n[0] *= valReal
            self.n[1] *= valReal
        elif dim == 3:
            for j in range(dim):
                self.x[0, j] = simplex2[1, j]-simplex2[0, j]
            for j in range(dim):
                self.x[1, j] = simplex2[2, j]-simplex2[0, j]
            self.n[0] = self.x[0, 1]*self.x[1, 2]-self.x[0, 2]*self.x[1, 1]
            self.n[1] = self.x[0, 2]*self.x[1, 0]-self.x[0, 0]*self.x[1, 2]
            self.n[2] = self.x[0, 0]*self.x[1, 1]-self.x[0, 1]*self.x[1, 0]
            valReal = 1./sqrt(mydot(self.n, self.n))
            self.n[0] *= valReal
            self.n[1] *= valReal
            self.n[2] *= valReal

        contrib[:] = 0.

        vol = vol1*vol2
        if panel < 0:
            sQR = <specialQuadRule>(self.distantQuadRulesPtr[MAX_PANEL+panel])
        else:
            sQR = <specialQuadRule>(self.distantQuadRulesPtr[panel])
        qr2 = <doubleSimplexQuadratureRule>(sQR.qr)
        PHI = sQR.PHI
        qr2.rule1.nodesInGlobalCoords(simplex1, self.x)
        qr2.rule2.nodesInGlobalCoords(simplex2, self.y)

        for k in range(qr2.rule1.num_nodes):
            for m in range(qr2.rule2.num_nodes):
                if dim == 1:
                    nw = 1.
                else:
                    normW = 0.
                    for j in range(dim):
                        self.w[j] = self.y[m, j]-self.x[k, j]
                        normW += self.w[j]**2
                    normW = 1./sqrt(normW)
                    for j in range(dim):
                        self.w[j] *= normW
                    nw = mydot(self.n, self.w)
                i = k*qr2.rule2.num_nodes+m
                self.kernel.evalPtr(dim, &self.x[k, 0], &self.y[m, 0], &self.vec[0])
                for l in range(valueSize):
                    self.temp[i, l] = qr2.weights[i] * nw * self.vec[l]

        k = 0
        for I in range(self.DoFMap.dofs_per_element):
            for J in range(I, self.DoFMap.dofs_per_element):
                if mask[k]:
                    for m in range(valueSize):
                        val = 0.
                        for i in range(qr2.num_nodes):
                            val += self.temp[i, m] * PHI[I, i] * PHI[J, i]
                        contrib[k, m] = val*vol
                k += 1
