###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class mass_1d_nonsym_P0_P1(mass_1d):
    cdef inline void eval(mass_1d_nonsym_P0_P1 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.500000000000000

        vol *= simplexVolume1D(simplex, self.temp)

        contrib[0] = vol
        contrib[1] = vol
