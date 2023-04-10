###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef buildRestriction_2D_P2(DoFMap coarse_DoFMap,
                            DoFMap fine_DoFMap):
    cdef:
        sparsityPattern sPat
        INDEX_t cellNo, dof, k, middleEdgeDof
        INDEX_t[::1] indptr, indices
        REAL_t[::1] data
        INDEX_t subCellNo0, subCellNo1, subCellNo2, subCellNo3
        CSR_LinearOperator R
    sPat = sparsityPattern(coarse_DoFMap.num_dofs)
    for cellNo in range(coarse_DoFMap.mesh.cells.shape[0]):
        subCellNo0 = 4*cellNo+0
        subCellNo1 = 4*cellNo+1
        subCellNo2 = 4*cellNo+2
        subCellNo3 = 4*cellNo+3
        dof = coarse_DoFMap.cell2dof(cellNo, 0)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 0))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 4))
        dof = coarse_DoFMap.cell2dof(cellNo, 1)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 0))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 5))
        dof = coarse_DoFMap.cell2dof(cellNo, 2)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 0))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 5))
        dof = coarse_DoFMap.cell2dof(cellNo, 3)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 1))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 4))
        dof = coarse_DoFMap.cell2dof(cellNo, 4)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 1))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 5))
        dof = coarse_DoFMap.cell2dof(cellNo, 5)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 2))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 4))
    indptr, indices = sPat.freeze()
    del sPat
    data = uninitialized((indices.shape[0]), dtype=REAL)
    R = CSR_LinearOperator(indices, indptr, data)
    R.num_columns = fine_DoFMap.num_dofs
    for cellNo in range(coarse_DoFMap.mesh.cells.shape[0]):
        subCellNo0 = 4*cellNo+0
        subCellNo1 = 4*cellNo+1
        subCellNo2 = 4*cellNo+2
        subCellNo3 = 4*cellNo+3
        dof = coarse_DoFMap.cell2dof(cellNo, 0)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 0), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 3), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 4), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 5), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 3), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 4), -0.125)
        dof = coarse_DoFMap.cell2dof(cellNo, 1)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 3), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 0), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 3), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 5), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 4), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 5), -0.125)
        dof = coarse_DoFMap.cell2dof(cellNo, 2)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 3), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 4), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 0), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 3), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 5), 0.375)
        dof = coarse_DoFMap.cell2dof(cellNo, 3)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 1), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 3), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 4), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 5), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 4), 0.25)
        dof = coarse_DoFMap.cell2dof(cellNo, 4)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 1), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 3), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 4), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 4), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 5), 0.75)
        dof = coarse_DoFMap.cell2dof(cellNo, 5)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 2), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 4), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 3), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 4), 0.5)
    return R
