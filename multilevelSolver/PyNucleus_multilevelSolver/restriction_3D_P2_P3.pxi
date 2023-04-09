###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

#@cython.initializedcheck(False)
#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef buildRestriction_3D_P2_P3(DoFMap coarse_DoFMap,
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
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 10))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 11))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 16))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 17))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 19))
        dof = coarse_DoFMap.cell2dof(cellNo, 1)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 1))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 12))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 13))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 16))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 17))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 18))
        dof = coarse_DoFMap.cell2dof(cellNo, 2)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 2))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 14))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 15))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 16))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 18))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 19))
        dof = coarse_DoFMap.cell2dof(cellNo, 3)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 3))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 10))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 11))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 12))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 13))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 14))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 15))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 17))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 18))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 19))
        dof = coarse_DoFMap.cell2dof(cellNo, 4)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 4))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 5))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 16))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 17))
        dof = coarse_DoFMap.cell2dof(cellNo, 5)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 6))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 7))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 16))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 18))
        dof = coarse_DoFMap.cell2dof(cellNo, 6)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 8))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 9))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 16))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 19))
        dof = coarse_DoFMap.cell2dof(cellNo, 7)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 10))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 11))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 17))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 19))
        dof = coarse_DoFMap.cell2dof(cellNo, 8)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 12))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 13))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 17))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 18))
        dof = coarse_DoFMap.cell2dof(cellNo, 9)
        if dof >= 0:
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 14))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 15))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 18))
            add(sPat, dof, fine_DoFMap.cell2dof(subCellNo0, 19))
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
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), 0.222222222)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 8), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 9), 0.222222222)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 10), 0.222222222)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 11), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 16), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 17), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 19), -0.111111111)
        dof = coarse_DoFMap.cell2dof(cellNo, 1)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 1), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), 0.222222222)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 6), 0.222222222)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 7), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 12), 0.222222222)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 13), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 16), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 17), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 18), -0.111111111)
        dof = coarse_DoFMap.cell2dof(cellNo, 2)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 2), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 6), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 7), 0.222222222)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 8), 0.222222222)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 9), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 14), 0.222222222)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 15), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 16), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 18), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 19), -0.111111111)
        dof = coarse_DoFMap.cell2dof(cellNo, 3)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 3), 1.0)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 10), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 11), 0.222222222)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 12), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 13), 0.222222222)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 14), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 15), 0.222222222)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 17), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 18), -0.111111111)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 19), -0.111111111)
        dof = coarse_DoFMap.cell2dof(cellNo, 4)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 4), 0.888888889)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 5), 0.888888889)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 16), 0.444444444)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 17), 0.444444444)
        dof = coarse_DoFMap.cell2dof(cellNo, 5)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 6), 0.888888889)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 7), 0.888888889)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 16), 0.444444444)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 18), 0.444444444)
        dof = coarse_DoFMap.cell2dof(cellNo, 6)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 8), 0.888888889)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 9), 0.888888889)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 16), 0.444444444)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 19), 0.444444444)
        dof = coarse_DoFMap.cell2dof(cellNo, 7)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 10), 0.888888889)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 11), 0.888888889)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 17), 0.444444444)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 19), 0.444444444)
        dof = coarse_DoFMap.cell2dof(cellNo, 8)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 12), 0.888888889)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 13), 0.888888889)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 17), 0.444444444)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 18), 0.444444444)
        dof = coarse_DoFMap.cell2dof(cellNo, 9)
        if dof >= 0:
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 14), 0.888888889)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 15), 0.888888889)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 18), 0.444444444)
            enterData(R, dof, fine_DoFMap.cell2dof(subCellNo0, 19), 0.444444444)
    return R
