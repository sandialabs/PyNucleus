###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class stiffness_1d_sym_P2(stiffness_1d_sym):
    cdef inline void eval(stiffness_1d_sym_P2 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.333333333333333

        vol /= simplexVolume1D(simplex, self.temp)

        contrib[0] = 7*vol
        contrib[1] = vol
        contrib[2] = -8*vol
        contrib[3] = 7*vol
        contrib[4] = -8*vol
        contrib[5] = 16*vol
