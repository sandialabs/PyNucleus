###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . DoFMaps cimport shapeFunction


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
                shapeFun = self.dm.localShapeFunctions[k]
                val += shapeFun.eval(self.cellFinder.bary)*self.u[dof]
        return val
