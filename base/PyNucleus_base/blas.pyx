###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
cimport numpy as np
from . myTypes import INDEX


cdef:
    INDEX_t MAX_INT = np.iinfo(INDEX).max
    REAL_t NAN = np.nan


include "allocation.pxi"
include "blas_routines.pxi"
include "mkl_routines.pxi"


cdef void updateScaledVector(REAL_t[::1] x, REAL_t[::1] y, REAL_t[::1] alpha):
    cdef:
        INDEX_t i
    for i in range(x.shape[0]):
        x[i] += alpha[i]*y[i]
