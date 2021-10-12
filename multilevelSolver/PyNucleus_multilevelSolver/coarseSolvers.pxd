###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py cimport MPI
from PyNucleus_fem.algebraicOverlaps cimport algebraicOverlapManager

include "coarseSolvers_decl_REAL.pxi"
include "coarseSolvers_decl_COMPLEX.pxi"
