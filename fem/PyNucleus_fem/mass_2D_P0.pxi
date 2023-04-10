###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class mass_2d_sym_P0(mass_2d):
    cdef inline void eval(mass_2d_sym_P0 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 1.00000000000000

        vol *= simplexVolume2D(simplex, self.temp)

        contrib[0] = vol
