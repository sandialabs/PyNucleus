###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


cimport numpy as np

include "config.pxi"

IF IDXTYPEWIDTH == 32:
    ctypedef np.int32_t idx_t
ELIF IDXTYPEWIDTH == 64:
    ctypedef np.int64_t idx_t

IF REALTYPEWIDTH == 32:
    ctypedef float real_t
ELIF REALTYPEWIDTH == 64:
    ctypedef np.float64_t real_t
