###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, BOOL_t
from PyNucleus_base.linear_operators cimport LinearOperator, CSR_LinearOperator
from . algebraicOverlaps cimport algebraicOverlapManager

include "distributed_operators_decl_REAL.pxi"
include "distributed_operators_decl_COMPLEX.pxi"
