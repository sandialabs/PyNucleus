###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef buildRestriction_3D_P1(DoFMap coarse_DoFMap,
                            DoFMap fine_DoFMap):
    cdef:
        sparsityPattern sPat
        INDEX_t cellNo, dof, k, middleEdgeDof
        INDEX_t[::1] indptr, indices
        REAL_t[::1] data
        INDEX_t subCellNo0, subCellNo1, subCellNo2, subCellNo3, subCellNo4, subCellNo5, subCellNo6, subCellNo7
        CSR_LinearOperator R
    sPat = sparsityPattern(coarse_DoFMap.num_dofs)
    for cellNo in range(coarse_DoFMap.mesh.cells.shape[0]):
        subCellNo0 = 8*cellNo+0
        subCellNo1 = 8*cellNo+1
        subCellNo2 = 8*cellNo+2
        subCellNo3 = 8*cellNo+3
        subCellNo4 = 8*cellNo+4
        subCellNo5 = 8*cellNo+5
        subCellNo6 = 8*cellNo+6
        subCellNo7 = 8*cellNo+7

        dof = coarse_DoFMap.cell2dof(cellNo, 0)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 0))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 0))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 0))
        dof = coarse_DoFMap.cell2dof(cellNo, 1)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 1))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 0))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 2))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 1))
        dof = coarse_DoFMap.cell2dof(cellNo, 2)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 2))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 2))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 2))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 0))
        dof = coarse_DoFMap.cell2dof(cellNo, 3)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 1))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 3))
    indptr, indices = sPat.freeze()
    del sPat
    data = uninitialized((indices.shape[0]), dtype=REAL)
    R = CSR_LinearOperator(indices, indptr, data)
    R.num_columns = fine_DoFMap.num_dofs
    for cellNo in range(coarse_DoFMap.mesh.cells.shape[0]):
        subCellNo0 = 8*cellNo+0
        subCellNo1 = 8*cellNo+1
        subCellNo2 = 8*cellNo+2
        subCellNo3 = 8*cellNo+3
        subCellNo4 = 8*cellNo+4
        subCellNo5 = 8*cellNo+5
        subCellNo6 = 8*cellNo+6
        subCellNo7 = 8*cellNo+7

        dof = coarse_DoFMap.cell2dof(cellNo, 0)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 3), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 0), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 0), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 0), 1.0)
        dof = coarse_DoFMap.cell2dof(cellNo, 1)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 1), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 0), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 2), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 1), 1.0)
        dof = coarse_DoFMap.cell2dof(cellNo, 2)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 2), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 2), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 2), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 0), 0.5)
        dof = coarse_DoFMap.cell2dof(cellNo, 3)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 3), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 1), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 3), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 3), 1.0)
    return R
