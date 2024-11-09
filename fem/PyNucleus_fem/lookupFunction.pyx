###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus_base.myTypes import REAL, INDEX
from PyNucleus_base.blas import uninitialized
from . DoFMaps cimport shapeFunction
from . femCy cimport simplexComputations1D, simplexComputations2D, simplexComputations3D
from libc.math cimport floor


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
            REAL_t val, val2
            INDEX_t cellNo, dof, k
        cellNo = self.cellFinder.findCell(x)
        if cellNo == -1:
            return 0.
        val = 0.
        for k in range(self.dm.dofs_per_element):
            dof = self.dm.cell2dof(cellNo, k)
            if dof >= 0:
                shapeFun = self.dm.getLocalShapeFunction(k)
                shapeFun.evalPtr(&self.cellFinder.bary[0], NULL, &val2)
                val += val2*self.u[dof]
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
            shapeFunction shapeFun
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
                shapeFun = self.dm.getLocalShapeFunction(k)
                shapeFun.setCell(self.mesh.cells[cellNo, :])
                shapeFun.eval(self.cellFinder.bary, self.gradients, self.temp)
                for componentNo in range(self.mesh.dim):
                    vals[componentNo] += self.u[dof]*self.temp[componentNo]


cdef class UniformLookup1D(function):
    def __init__(self, REAL_t a, REAL_t b, REAL_t[::1] vals):
        self.a = a
        self.b = b
        self.vals = vals
        self.dx = (b-a)/(self.vals.shape[0]-1)
        self.invDx = 1./self.dx

    cdef REAL_t eval(self, REAL_t[::1] x):
        cdef:
            INDEX_t k
            REAL_t theta
        k = max(min(INDEX((x[0]-self.a)*self.invDx), self.vals.shape[0]-2), 0)
        theta = (x[0]-self.a-k*self.dx)*self.invDx
        return (1-theta)*self.vals[k] + theta * self.vals[k+1]

    def __reduce__(self):
        return UniformLookup1D, (self.a, self.b, np.array(self.vals))
