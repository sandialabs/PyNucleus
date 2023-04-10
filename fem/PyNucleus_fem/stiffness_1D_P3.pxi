###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class stiffness_1d_sym_P3(stiffness_1d_sym):
    cdef inline void eval(stiffness_1d_sym_P3 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.0250000000000000

        vol /= simplexVolume1D(simplex, self.temp)

        contrib[0] = 148*vol
        contrib[1] = -13*vol
        contrib[2] = -189*vol
        contrib[3] = 54*vol
        contrib[4] = 148*vol
        contrib[5] = 54*vol
        contrib[6] = -189*vol
        contrib[7] = 432*vol
        contrib[8] = -297*vol
        contrib[9] = 432*vol
