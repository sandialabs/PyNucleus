###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef buildRestriction_2D_P1_P3(DoFMap coarse_DoFMap,
                               DoFMap fine_DoFMap):
    cdef:
        sparsityPattern sPat
        INDEX_t cellNo, dof, k, middleEdgeDof
        INDEX_t[::1] indptr, indices
        REAL_t[::1] data
        INDEX_t subCellNo0
        CSR_LinearOperator R
    sPat = sparsityPattern(coarse_DoFMap.num_dofs)
    for cellNo in range(coarse_DoFMap.mesh.cells.shape[0]):
        subCellNo0 = 1*cellNo+0
        dof = coarse_DoFMap.cell2dof(cellNo, 0)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 0))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 9))
        dof = coarse_DoFMap.cell2dof(cellNo, 1)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 1))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 9))
        dof = coarse_DoFMap.cell2dof(cellNo, 2)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 2))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 9))
    indptr, indices = sPat.freeze()
    del sPat
    data = uninitialized((indices.shape[0]), dtype=REAL)
    R = CSR_LinearOperator(indices, indptr, data)
    R.num_columns = fine_DoFMap.num_dofs
    for cellNo in range(coarse_DoFMap.mesh.cells.shape[0]):
        subCellNo0 = 1*cellNo+0
        dof = coarse_DoFMap.cell2dof(cellNo, 0)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 0), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 3), 0.666666667)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), 0.333333333)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 7), 0.333333333)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 8), 0.666666667)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 9), 0.333333333)
        dof = coarse_DoFMap.cell2dof(cellNo, 1)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 1), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 3), 0.333333333)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), 0.666666667)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), 0.666666667)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 6), 0.333333333)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 9), 0.333333333)
        dof = coarse_DoFMap.cell2dof(cellNo, 2)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 2), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), 0.333333333)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 6), 0.666666667)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 7), 0.666666667)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 8), 0.333333333)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 9), 0.333333333)
    return R
