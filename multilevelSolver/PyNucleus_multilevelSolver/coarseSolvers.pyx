###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
import numpy as np
cimport numpy as np
import logging
from PyNucleus_base.myTypes import INDEX, REAL, BOOL
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, BOOL_t
from PyNucleus_base import uninitialized
from PyNucleus_base.performanceLogger cimport PLogger, FakePLogger
from PyNucleus_base.linear_operators cimport LinearOperator
from PyNucleus_base import solverFactory
from PyNucleus_fem.meshOverlaps import overlapManager
from time import sleep
from sys import stdout
include "config.pxi"
LOGGER = logging.getLogger(__name__)
MPI_BOOL = MPI.BOOL

include "coarseSolvers_REAL.pxi"
include "coarseSolvers_COMPLEX.pxi"
