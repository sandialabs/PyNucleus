###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . functions cimport function
from PyNucleus_base.myTypes cimport REAL_t


cdef class IP:
    cdef:
        function weight
        REAL_t a
        REAL_t b


cdef class LegendreShiftedIP(IP):
    pass


cdef class JacobiShiftedIP(IP):
    pass


cdef class LogJacobiShiftedIP(IP):
    pass


cdef class LogAffineJacobiShiftedIP(IP):
    pass
