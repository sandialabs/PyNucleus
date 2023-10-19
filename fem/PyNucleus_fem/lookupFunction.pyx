###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes import REAL
from PyNucleus_base.blas import uninitialized
from . DoFMaps cimport shapeFunction, vectorShapeFunction
from . femCy cimport simplexComputations1D, simplexComputations2D, simplexComputations3D


cdef class lookupFunction(function):
    def __init__(self, meshBase mesh, DoFMap dm, REAL_t[::1] u, cellFinder2 cF=None):
        self.mesh = mesh
        self.dm = dm
        self.u = u
        if cF is None:
            self.cellFinder = cellFinder2(self.mesh)
        else:
            self.cellFinder = cF

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
                shapeFun = self.dm.getLocalShapeFunction(k)
                val += shapeFun.eval(self.cellFinder.bary)*self.u[dof]
        return val


cdef class vectorLookupFunction(vectorFunction):
    def __init__(self, meshBase mesh, DoFMap dm, REAL_t[::1] u, cellFinder2 cF=None):
        super(vectorLookupFunction, self).__init__(mesh.dim)
        self.mesh = mesh
        self.dm = dm
        self.u = u
        if cF is None:
            self.cellFinder = cellFinder2(self.mesh)
        else:
            self.cellFinder = cF
        if self.mesh.dim == 1:
            self.sC = simplexComputations1D()
        elif self.mesh.dim == 2:
            self.sC = simplexComputations2D()
        elif self.mesh.dim == 3:
            self.sC = simplexComputations3D()
        else:
            raise NotImplementedError()
        self.simplex = uninitialized((self.mesh.dim+1, self.mesh.dim), dtype=REAL)
        self.sC.setSimplex(self.simplex)
        self.temp = uninitialized((self.mesh.dim), dtype=REAL)
        self.gradients = uninitialized((self.mesh.dim+1, self.mesh.dim), dtype=REAL)

    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] vals):
        cdef:
            vectorShapeFunction shapeFun
            INDEX_t cellNo, dof, k, componentNo
        for componentNo in range(self.mesh.dim):
            vals[componentNo] = 0.
        cellNo = self.cellFinder.findCell(x)
        if cellNo == -1:
            return
        self.mesh.getSimplex(cellNo, self.simplex)
        self.sC.evalVolumeGradients(self.gradients)
        for k in range(self.dm.dofs_per_element):
            dof = self.dm.cell2dof(cellNo, k)
            if dof >= 0:
                shapeFun = self.dm.getLocalVectorShapeFunction(k)
                shapeFun.setCell(self.mesh.cells[cellNo, :])
                shapeFun.eval(self.cellFinder.bary, self.gradients, self.temp)
                for componentNo in range(self.mesh.dim):
                    vals[componentNo] += self.u[dof]*self.temp[componentNo]
