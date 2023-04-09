###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cpdef enum CycleType:
    V = 1,
    W = 2,
    FMG_V = 666
    FMG_W = 667


include "multigrid_decl_REAL.pxi"
include "multigrid_decl_COMPLEX.pxi"
