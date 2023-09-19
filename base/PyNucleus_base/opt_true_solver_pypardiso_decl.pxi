###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . solvers cimport solver
from . myTypes cimport REAL_t, INDEX_t


cdef class pardiso_lu_solver(solver):
    cdef:
        INDEX_t[::1] perm
        object Ainv, lu
        object Asp
