###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
cimport numpy as np
from . myTypes import INDEX, REAL, COMPLEX
from . import solverFactory
from . blas import uninitialized

include "SchurComplement_REAL.pxi"
include "SchurComplement_COMPLEX.pxi"
