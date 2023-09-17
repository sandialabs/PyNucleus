###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class discrete_gradient_2d_N1e_P1(mass_2d):
    def __init__(self):
        mass_2d.__init__(self)
        self.needsCellInfo = True
        self.additiveAssembly = False

    cdef inline void eval(discrete_gradient_2d_N1e_P1 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        if self.cell[0] > self.cell[1]:
            contrib[0] = 1
            contrib[1] = -1
        else:
            contrib[0] = -1
            contrib[1] = 1
        if self.cell[1] > self.cell[2]:
            contrib[4] = 1
            contrib[5] = -1
        else:
            contrib[4] = -1
            contrib[5] = 1
        if self.cell[2] > self.cell[0]:
            contrib[6] = -1
            contrib[8] = 1
        else:
            contrib[6] = 1
            contrib[8] = -1
        contrib[2] = 0
        contrib[3] = 0
        contrib[7] = 0
