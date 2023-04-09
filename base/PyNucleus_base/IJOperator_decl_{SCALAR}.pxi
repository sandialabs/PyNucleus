###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . intTuple cimport intTuple


cdef class {SCALAR_label}IJOperator({SCALAR_label}LinearOperator):
    cdef:
        dict entries
    cdef {SCALAR}_t getEntry(self, INDEX_t I, INDEX_t J)
    cdef void setEntry(self, INDEX_t I, INDEX_t J, {SCALAR}_t val)
