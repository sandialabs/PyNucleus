###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class discrete_curl_2d_P0_N1e(mass_2d):
    def __init__(self):
        mass_2d.__init__(self)
        self.needsCellInfo = True
        self.additiveAssembly = False

    cdef inline void eval(discrete_curl_2d_P0_N1e self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        if self.cell[0] > self.cell[1]:
            contrib[0] = -1
        else:
            contrib[0] = 1
        if self.cell[1] > self.cell[2]:
            contrib[1] = -1
        else:
            contrib[1] = 1
        if self.cell[2] > self.cell[0]:
            contrib[2] = -1
        else:
            contrib[2] = 1
