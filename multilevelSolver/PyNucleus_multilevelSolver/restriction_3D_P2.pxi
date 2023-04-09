###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef buildRestriction_3D_P2(DoFMap coarse_DoFMap,
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
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 0))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 7))
        dof = coarse_DoFMap.cell2dof(cellNo, 1)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 1))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 5))
        dof = coarse_DoFMap.cell2dof(cellNo, 2)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 2))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 9))
        dof = coarse_DoFMap.cell2dof(cellNo, 3)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 8))
        dof = coarse_DoFMap.cell2dof(cellNo, 4)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo4, 0))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
        dof = coarse_DoFMap.cell2dof(cellNo, 5)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 1))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 5))
        dof = coarse_DoFMap.cell2dof(cellNo, 6)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 2))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
        dof = coarse_DoFMap.cell2dof(cellNo, 7)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 7))
        dof = coarse_DoFMap.cell2dof(cellNo, 8)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 1))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 8))
        dof = coarse_DoFMap.cell2dof(cellNo, 9)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 2))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo2, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo3, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo1, 9))
        if fine_DoFMap.mesh.cells[subCellNo4, 0] == fine_DoFMap.mesh.cells[subCellNo5, 0]:
            if fine_DoFMap.mesh.cells[subCellNo4, 0] == fine_DoFMap.mesh.cells[subCellNo6, 0]:
                dof = coarse_DoFMap.cell2dof(cellNo, 0)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 4))
                dof = coarse_DoFMap.cell2dof(cellNo, 1)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 4))
                dof = coarse_DoFMap.cell2dof(cellNo, 2)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 4))
                dof = coarse_DoFMap.cell2dof(cellNo, 3)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 4))
                dof = coarse_DoFMap.cell2dof(cellNo, 4)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo7, 7))
                dof = coarse_DoFMap.cell2dof(cellNo, 5)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo7, 7))
                dof = coarse_DoFMap.cell2dof(cellNo, 6)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo7, 7))
                dof = coarse_DoFMap.cell2dof(cellNo, 7)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo7, 7))
                dof = coarse_DoFMap.cell2dof(cellNo, 8)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo7, 7))
                dof = coarse_DoFMap.cell2dof(cellNo, 9)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo7, 7))
            else:
                dof = coarse_DoFMap.cell2dof(cellNo, 0)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo6, 6))
                dof = coarse_DoFMap.cell2dof(cellNo, 1)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo6, 6))
                dof = coarse_DoFMap.cell2dof(cellNo, 2)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo6, 6))
                dof = coarse_DoFMap.cell2dof(cellNo, 3)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo6, 6))
                dof = coarse_DoFMap.cell2dof(cellNo, 4)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 9))
                dof = coarse_DoFMap.cell2dof(cellNo, 5)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 9))
                dof = coarse_DoFMap.cell2dof(cellNo, 6)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 9))
                dof = coarse_DoFMap.cell2dof(cellNo, 7)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 9))
                dof = coarse_DoFMap.cell2dof(cellNo, 8)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 9))
                dof = coarse_DoFMap.cell2dof(cellNo, 9)
                if dof >= 0:
                    add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 9))
        else:
            dof = coarse_DoFMap.cell2dof(cellNo, 0)
            if dof >= 0:
                add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 7))
            dof = coarse_DoFMap.cell2dof(cellNo, 1)
            if dof >= 0:
                add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 7))
            dof = coarse_DoFMap.cell2dof(cellNo, 2)
            if dof >= 0:
                add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 7))
            dof = coarse_DoFMap.cell2dof(cellNo, 3)
            if dof >= 0:
                add(sPat, dof, fine_DoFMap.cell2dof(subCellNo5, 7))
            dof = coarse_DoFMap.cell2dof(cellNo, 4)
            if dof >= 0:
                add(sPat, dof, fine_DoFMap.cell2dof(subCellNo7, 8))
            dof = coarse_DoFMap.cell2dof(cellNo, 5)
            if dof >= 0:
                add(sPat, dof, fine_DoFMap.cell2dof(subCellNo7, 8))
            dof = coarse_DoFMap.cell2dof(cellNo, 6)
            if dof >= 0:
                add(sPat, dof, fine_DoFMap.cell2dof(subCellNo7, 8))
            dof = coarse_DoFMap.cell2dof(cellNo, 7)
            if dof >= 0:
                add(sPat, dof, fine_DoFMap.cell2dof(subCellNo7, 8))
            dof = coarse_DoFMap.cell2dof(cellNo, 8)
            if dof >= 0:
                add(sPat, dof, fine_DoFMap.cell2dof(subCellNo7, 8))
            dof = coarse_DoFMap.cell2dof(cellNo, 9)
            if dof >= 0:
                add(sPat, dof, fine_DoFMap.cell2dof(subCellNo7, 8))
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
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 4), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 7), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 4), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 7), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 6), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 6), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 6), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 6), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 0), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 4), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 7), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 7), -0.125)
        dof = coarse_DoFMap.cell2dof(cellNo, 1)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 8), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 5), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 4), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 8), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 4), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 8), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 4), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 1), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 5), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 8), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 5), -0.125)
        dof = coarse_DoFMap.cell2dof(cellNo, 2)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 5), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 9), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 6), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 9), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 6), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 6), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 2), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 6), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 5), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 9), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 5), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 9), 0.375)
        dof = coarse_DoFMap.cell2dof(cellNo, 3)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 9), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 8), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 8), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 9), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 9), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 7), 0.375)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 8), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 7), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 3), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 9), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 7), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 7), -0.125)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 8), -0.125)
        dof = coarse_DoFMap.cell2dof(cellNo, 4)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo4, 0), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 4), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 7), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 4), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 4), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 8), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 6), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), 0.5)
        dof = coarse_DoFMap.cell2dof(cellNo, 5)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 5), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 9), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 4), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 8), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 1), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 5), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 6), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 5), 0.75)
        dof = coarse_DoFMap.cell2dof(cellNo, 6)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 2), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 7), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 4), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 9), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 6), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 6), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 6), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 6), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), 0.5)
        dof = coarse_DoFMap.cell2dof(cellNo, 7)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 7), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 3), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 7), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 9), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 7), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 8), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 4), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 6), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 7), 0.75)
        dof = coarse_DoFMap.cell2dof(cellNo, 8)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 9), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 1), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 5), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 7), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 8), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 8), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 8), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 4), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 8), 0.75)
        dof = coarse_DoFMap.cell2dof(cellNo, 9)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 7), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 5), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 2), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 9), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 9), 0.25)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 9), 0.75)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo2, 8), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo3, 6), 0.5)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo1, 9), 0.25)
        if fine_DoFMap.mesh.cells[subCellNo4, 0] == fine_DoFMap.mesh.cells[subCellNo5, 0]:
            if fine_DoFMap.mesh.cells[subCellNo4, 0] == fine_DoFMap.mesh.cells[subCellNo6, 0]:
                dof = coarse_DoFMap.cell2dof(cellNo, 0)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 4), -0.125)
                dof = coarse_DoFMap.cell2dof(cellNo, 1)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 4), -0.125)
                dof = coarse_DoFMap.cell2dof(cellNo, 2)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 4), -0.125)
                dof = coarse_DoFMap.cell2dof(cellNo, 3)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 4), -0.125)
                dof = coarse_DoFMap.cell2dof(cellNo, 4)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo7, 7), 0.25)
                dof = coarse_DoFMap.cell2dof(cellNo, 5)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo7, 7), 0.25)
                dof = coarse_DoFMap.cell2dof(cellNo, 6)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo7, 7), 0.25)
                dof = coarse_DoFMap.cell2dof(cellNo, 7)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo7, 7), 0.25)
                dof = coarse_DoFMap.cell2dof(cellNo, 8)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo7, 7), 0.25)
                dof = coarse_DoFMap.cell2dof(cellNo, 9)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo7, 7), 0.25)
            else:
                dof = coarse_DoFMap.cell2dof(cellNo, 0)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo6, 6), -0.125)
                dof = coarse_DoFMap.cell2dof(cellNo, 1)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo6, 6), -0.125)
                dof = coarse_DoFMap.cell2dof(cellNo, 2)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo6, 6), -0.125)
                dof = coarse_DoFMap.cell2dof(cellNo, 3)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo6, 6), -0.125)
                dof = coarse_DoFMap.cell2dof(cellNo, 4)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 9), 0.25)
                dof = coarse_DoFMap.cell2dof(cellNo, 5)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 9), 0.25)
                dof = coarse_DoFMap.cell2dof(cellNo, 6)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 9), 0.25)
                dof = coarse_DoFMap.cell2dof(cellNo, 7)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 9), 0.25)
                dof = coarse_DoFMap.cell2dof(cellNo, 8)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 9), 0.25)
                dof = coarse_DoFMap.cell2dof(cellNo, 9)
                if dof >= 0:
                    enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 9), 0.25)
        else:
            dof = coarse_DoFMap.cell2dof(cellNo, 0)
            if dof >= 0:
                enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 7), -0.125)
            dof = coarse_DoFMap.cell2dof(cellNo, 1)
            if dof >= 0:
                enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 7), -0.125)
            dof = coarse_DoFMap.cell2dof(cellNo, 2)
            if dof >= 0:
                enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 7), -0.125)
            dof = coarse_DoFMap.cell2dof(cellNo, 3)
            if dof >= 0:
                enterData(R, dof, fine_DoFMap.cell2dof(subCellNo5, 7), -0.125)
            dof = coarse_DoFMap.cell2dof(cellNo, 4)
            if dof >= 0:
                enterData(R, dof, fine_DoFMap.cell2dof(subCellNo7, 8), 0.25)
            dof = coarse_DoFMap.cell2dof(cellNo, 5)
            if dof >= 0:
                enterData(R, dof, fine_DoFMap.cell2dof(subCellNo7, 8), 0.25)
            dof = coarse_DoFMap.cell2dof(cellNo, 6)
            if dof >= 0:
                enterData(R, dof, fine_DoFMap.cell2dof(subCellNo7, 8), 0.25)
            dof = coarse_DoFMap.cell2dof(cellNo, 7)
            if dof >= 0:
                enterData(R, dof, fine_DoFMap.cell2dof(subCellNo7, 8), 0.25)
            dof = coarse_DoFMap.cell2dof(cellNo, 8)
            if dof >= 0:
                enterData(R, dof, fine_DoFMap.cell2dof(subCellNo7, 8), 0.25)
            dof = coarse_DoFMap.cell2dof(cellNo, 9)
            if dof >= 0:
                enterData(R, dof, fine_DoFMap.cell2dof(subCellNo7, 8), 0.25)
    return R
