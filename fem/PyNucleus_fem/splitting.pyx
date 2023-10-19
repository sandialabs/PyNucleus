###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus_base import INDEX
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t
from . meshCy cimport meshBase
from . DoFMaps cimport DoFMap, P0_DoFMap
from . meshCy import getSubmesh
from . functions cimport function
from . DoFMaps import (getSubMapRestrictionProlongation,
                       getSubMapRestrictionProlongation2)
from PyNucleus_base.linear_operators cimport sparseGraph


cdef class meshSplitter:
    cdef:
        meshBase mesh
        dict indicators
        dict submeshes
        dict selectedCells

    def __init__(self, meshBase mesh, indicators):
        self.mesh = mesh
        self.indicators = indicators
        self.submeshes = {}
        self.selectedCells = {}

    def getSubMesh(self, label):
        cdef:
            DoFMap dm
            list selectedCellsList
            INDEX_t cellNo, dofNo
        if label not in self.submeshes:
            ind = self.indicators[label]
            if isinstance(ind, function):
                dm = P0_DoFMap(self.mesh, ind)
                selectedCellsList = []
                for cellNo in range(self.mesh.num_cells):
                    for dofNo in range(dm.dofs_per_element):
                        if dm.dofs[cellNo, dofNo] >= 0:
                            selectedCellsList.append(cellNo)
                            break
                selectedCells = np.array(selectedCellsList, dtype=INDEX)
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

    def getSubMapOnFullMesh(self, label, dm):
        sub_dm = type(dm)(self.mesh, -1)
        sub_dof = 0
        sub_boundary_dof = -1
        assigned_dofs = {}
        for cellNo in self.selectedCells[label]:
            for dofNo in range(dm.dofs_per_element):
                dof = dm.cell2dof_py(cellNo, dofNo)
                try:
                    sub_dm.dofs[cellNo, dofNo] = assigned_dofs[dof]
                except KeyError:
                    if dof >= 0:
                        sub_dm.dofs[cellNo, dofNo] = sub_dof
                        assigned_dofs[dof] = sub_dof
                        sub_dof += 1
                    else:
                        sub_dm.dofs[cellNo, dofNo] = sub_boundary_dof
                        assigned_dofs[dof] = sub_boundary_dof
                        sub_boundary_dof -= 1
        for cellNo in range(self.mesh.num_cells):
            for dofNo in range(dm.dofs_per_element):
                dof = dm.cell2dof_py(cellNo, dofNo)
                try:
                    sub_dm.dofs[cellNo, dofNo] = assigned_dofs[dof]
                except KeyError:
                    sub_dm.dofs[cellNo, dofNo] = sub_boundary_dof
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



cdef class meshSplitter2(meshSplitter):
    cdef:
        INDEX_t[::1] cell2subdomain
        sparseGraph subdomain2cell

    def __init__(self, meshBase mesh, function indicator):
        cdef:
            INDEX_t subdomainNo
        super(meshSplitter2, self).__init__(mesh, {})
        self.cell2subdomain, self.subdomain2cell = self.createSubdomains(indicator)
        for subdomainNo in range(self.subdomain2cell.shape[0]):
            self.indicators[subdomainNo] = np.array(self.subdomain2cell.indices)[self.subdomain2cell.indptr[subdomainNo]:self.subdomain2cell.indptr[subdomainNo+1]]

    def createSubdomains(self, function indicator):
        cdef:
            REAL_t[:, ::1] centers
            INDEX_t[::1] cell2subdomain
            INDEX_t[::1] subdomains
            INDEX_t cellNo, num_subdomains, subdomainNo
            INDEX_t[::1] indices, indptr
            DoFMap dm
        dm = P0_DoFMap(self.mesh)
        cell2subdomain = np.around(dm.interpolate(indicator).toarray()).astype(INDEX)
        # centers = self.mesh.getCellCenters()
        # cell2subdomain = np.zeros((self.mesh.num_cells), dtype=INDEX)
        subdomains = np.unique(cell2subdomain)
        assert min(subdomains) == 0
        assert len(subdomains) == max(subdomains)+1
        num_subdomains = subdomains.shape[0]
        indptr = np.zeros((num_subdomains+1), dtype=INDEX)
        indices = np.zeros((self.mesh.num_cells), dtype=INDEX)
        for cellNo in range(self.mesh.num_cells):
            subdomainNo = cell2subdomain[cellNo]
            indptr[subdomainNo+1] += 1
        for subdomainNo in range(num_subdomains):
            indptr[subdomainNo+1] += indptr[subdomainNo]
        for cellNo in range(self.mesh.num_cells):
            subdomainNo = cell2subdomain[cellNo]
            indices[indptr[subdomainNo]] = cellNo
            indptr[subdomainNo] += 1
        for subdomainNo in range(num_subdomains-1, -1, -1):
            indptr[subdomainNo+1] = indptr[subdomainNo]
        indptr[0] = 0
        subdomain2cell = sparseGraph(indices, indptr, num_subdomains, self.mesh.num_cells)
        return cell2subdomain, subdomain2cell


class dofmapSplitter:
    def __init__(self, dm, indicators):
        self.dm = dm
        self.indicators = indicators
        self.submeshes = {}
        self.submaps = {}
        self.selectedCells = {}

    def getSubMap(self, label):
        from copy import deepcopy
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
