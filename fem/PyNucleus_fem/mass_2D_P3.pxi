###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class mass_2d_sym_P3(mass_2d):
    cdef inline void eval(mass_2d_sym_P3 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.000148809523809524

        vol *= simplexVolume2D(simplex, self.temp)

        contrib[0] = 76*vol
        contrib[1] = 11*vol
        contrib[2] = 11*vol
        contrib[3] = 18*vol
        contrib[4] = 0
        contrib[5] = 27*vol
        contrib[6] = 27*vol
        contrib[7] = 0
        contrib[8] = 18*vol
        contrib[9] = 36*vol
        contrib[10] = 76*vol
        contrib[11] = 11*vol
        contrib[12] = 0
        contrib[13] = 18*vol
        contrib[14] = 18*vol
        contrib[15] = 0
        contrib[16] = 27*vol
        contrib[17] = 27*vol
        contrib[18] = 36*vol
        contrib[19] = 76*vol
        contrib[20] = 27*vol
        contrib[21] = 27*vol
        contrib[22] = 0
        contrib[23] = 18*vol
        contrib[24] = 18*vol
        contrib[25] = 0
        contrib[26] = 36*vol
        contrib[27] = 540*vol
        contrib[28] = -189*vol
        contrib[29] = -135*vol
        contrib[30] = -54*vol
        contrib[31] = -135*vol
        contrib[32] = 270*vol
        contrib[33] = 162*vol
        contrib[34] = 540*vol
        contrib[35] = 270*vol
        contrib[36] = -135*vol
        contrib[37] = -54*vol
        contrib[38] = -135*vol
        contrib[39] = 162*vol
        contrib[40] = 540*vol
        contrib[41] = -189*vol
        contrib[42] = -135*vol
        contrib[43] = -54*vol
        contrib[44] = 162*vol
        contrib[45] = 540*vol
        contrib[46] = 270*vol
        contrib[47] = -135*vol
        contrib[48] = 162*vol
        contrib[49] = 540*vol
        contrib[50] = -189*vol
        contrib[51] = 162*vol
        contrib[52] = 540*vol
        contrib[53] = 162*vol
        contrib[54] = 1944*vol
