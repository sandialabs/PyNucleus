###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


cdef class mass_2d_sym_P2(mass_2d):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(mass_2d_sym_P2 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.00555555555555556

        vol *= simplexVolume2D(simplex, self.temp)

        contrib[0] = 6*vol
        contrib[1] = -vol
        contrib[2] = -vol
        contrib[3] = 0
        contrib[4] = -4*vol
        contrib[5] = 0
        contrib[6] = 6*vol
        contrib[7] = -vol
        contrib[8] = 0
        contrib[9] = 0
        contrib[10] = -4*vol
        contrib[11] = 6*vol
        contrib[12] = -4*vol
        contrib[13] = 0
        contrib[14] = 0
        contrib[15] = 32*vol
        contrib[16] = 16*vol
        contrib[17] = 16*vol
        contrib[18] = 32*vol
        contrib[19] = 16*vol
        contrib[20] = 32*vol
