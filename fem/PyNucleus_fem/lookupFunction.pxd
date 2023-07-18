###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes cimport REAL_t, INDEX_t
from . functions cimport function
from . meshCy cimport meshBase, cellFinder2
from . DoFMaps cimport DoFMap


cdef class lookupFunction(function):
    cdef:
        meshBase mesh
        public DoFMap dm
        public REAL_t[::1] u
        public cellFinder2 cellFinder
