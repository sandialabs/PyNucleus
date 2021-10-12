###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from PyNucleus_base import INDEX
from . import P0_DoFMap
from . import getSubmesh
from . DoFMaps import (getSubMapRestrictionProlongation,
                       getSubMapRestrictionProlongation2)


class meshSplitter:
    def __init__(self, mesh, indicators):
        self.mesh = mesh
        self.indicators = indicators
        self.submeshes = {}
        self.selectedCells = {}

    def getSubMesh(self, label):
        from . import function
        if label not in self.submeshes:
            ind = self.indicators[label]
            if isinstance(ind, function):
                dm = P0_DoFMap(self.mesh, ind)
                selectedCells = []
                for cellNo in range(self.mesh.num_cells):
                    for dofNo in range(dm.dofs_per_element):
                        if dm.dofs[cellNo, dofNo] >= 0:
                            selectedCells.append(cellNo)
                            break
                selectedCells = np.array(selectedCells, dtype=INDEX)
            else:
                selectedCells = ind
            self.selectedCells[label] = selectedCells
            new_mesh = getSubmesh(self.mesh, selectedCells)
            self.submeshes[label] = new_mesh
        return self.submeshes[label]

    def getSubMap(self, label, dm):
        subMesh = self.getSubMesh(label)
        sub_dm = type(dm)(subMesh, -1)
        sub_dof = 0
        sub_boundary_dof = -1
        assigned_dofs = {}
        for sub_cellNo, cellNo in enumerate(self.selectedCells[label]):
            for dofNo in range(dm.dofs_per_element):
                dof = dm.cell2dof_py(cellNo, dofNo)
                try:
                    sub_dm.dofs[sub_cellNo, dofNo] = assigned_dofs[dof]
                except KeyError:
                    if dof >= 0:
                        sub_dm.dofs[sub_cellNo, dofNo] = sub_dof
                        assigned_dofs[dof] = sub_dof
                        sub_dof += 1
                    else:
                        sub_dm.dofs[sub_cellNo, dofNo] = sub_boundary_dof
                        assigned_dofs[dof] = sub_boundary_dof
                        sub_boundary_dof -= 1
        sub_dm.num_dofs = sub_dof
        sub_dm.num_boundary_dofs = -sub_boundary_dof-1
        return sub_dm

    def getRestrictionProlongation(self, label, dm, sub_dm):
        cellIndices = -np.ones((self.mesh.num_cells), dtype=INDEX)
        cells = self.selectedCells[label]
        cellIndices[cells] = np.arange(cells.shape[0], dtype=INDEX)
        subR = getSubMapRestrictionProlongation2(self.mesh, dm, sub_dm, cellIndices)
        subP = subR.transpose()
        return subR, subP

    def plotSubMeshes(self):
        import matplotlib.pyplot as plt
        numSubMeshes = len(self.submeshes)
        for k, label in enumerate(self.submeshes):
            plt.subplot(numSubMeshes, 1, k+1)
            self.mesh.plot()
            submesh = self.getSubMesh(label)
            dm0 = P0_DoFMap(submesh, -1)
            dm0.ones().plot(flat=True)
            plt.title(label)


class dofmapSplitter:
    def __init__(self, dm, indicators):
        self.dm = dm
        self.indicators = indicators
        self.submeshes = {}
        self.submaps = {}
        self.selectedCells = {}

    def getSubMap(self, label):
        from copy import deepcopy
        from . import function
        if label not in self.submaps:
            self.submaps[label] = deepcopy(self.dm)
            if isinstance(self.indicators[label], function):
                self.submaps[label].resetUsingIndicator(self.indicators[label])
            else:
                self.submaps[label].resetUsingFEVector(self.indicators[label])
        return self.submaps[label]

    def getSubMesh(self, label):
        if label not in self.submeshes:
            subMap = self.getSubMap(label)
            selectedCells = []
            for cellNo in range(subMap.mesh.num_cells):
                for dofNo in range(subMap.dofs_per_element):
                    if subMap.cell2dof_py(cellNo, dofNo) >= 0:
                        selectedCells.append(cellNo)
                        break
            selectedCells = np.array(selectedCells, dtype=INDEX)
            self.selectedCells[label] = selectedCells
            new_mesh = getSubmesh(self.dm.mesh, selectedCells)
            self.submeshes[label] = new_mesh
        return self.submeshes[label]

    def getSubMapOnSubMesh(self, label):
        dm = self.getSubMap(label)
        subMesh = self.getSubMesh(label)
        sub_dm = type(dm)(subMesh, -1)
        num_boundary_dofs = -1
        boundary_dofs = {}
        for sub_cellNo, cellNo in enumerate(self.selectedCells[label]):
            for dofNo in range(dm.dofs_per_element):
                dof = dm.cell2dof_py(cellNo, dofNo)
                if dof < 0:
                    try:
                        dof = boundary_dofs[dof]
                    except KeyError:
                        boundary_dofs[dof] = num_boundary_dofs
                        dof = num_boundary_dofs
                        num_boundary_dofs -= 1
                sub_dm.dofs[sub_cellNo, dofNo] = dof
        sub_dm.num_dofs = dm.num_dofs
        sub_dm.num_boundary_dofs = -num_boundary_dofs-1
        return sub_dm

    def getRestrictionProlongation(self, label):
        return getSubMapRestrictionProlongation(self.dm, self.getSubMap(label))

    def plotSubMaps(self):
        import matplotlib.pyplot as plt
        numSubMaps = len(self.submaps)
        for k, label in enumerate(self.submaps):
            plt.subplot(numSubMaps, 1, k+1)
            submap = self.getSubMap(label)
            submap.plot()
            plt.title(label)
