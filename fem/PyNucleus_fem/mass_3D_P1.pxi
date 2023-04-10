###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class mass_3d_sym_P1(mass_3d):
    cdef inline void eval(mass_3d_sym_P1 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.0500000000000000

        vol *= simplexVolume3D(simplex, self.temp)

        contrib[0] = 2*vol
        contrib[1] = vol
        contrib[2] = vol
        contrib[3] = vol
        contrib[4] = 2*vol
        contrib[5] = vol
        contrib[6] = vol
        contrib[7] = 2*vol
        contrib[8] = vol
        contrib[9] = 2*vol
