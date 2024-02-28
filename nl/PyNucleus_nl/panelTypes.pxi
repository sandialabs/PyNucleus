###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef enum elementPositionType:
    DISTANT = 0
    COMMON_VERTEX = -1
    COMMON_EDGE = -2
    COMMON_FACE = -3
    COMMON_VOLUME = -4
    SEPARATED = -5
    IGNORED = -6
    ON_HORIZON = -7
