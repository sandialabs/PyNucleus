###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class mass_3d_sym_P2(mass_3d):
    cdef inline void eval(mass_3d_sym_P2 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.00238095238095238

        vol *= simplexVolume3D(simplex, self.temp)

        contrib[0] = 6*vol
        contrib[1] = vol
        contrib[2] = vol
        contrib[3] = vol
        contrib[4] = -4*vol
        contrib[5] = -6*vol
        contrib[6] = -4*vol
        contrib[7] = -4*vol
        contrib[8] = -6*vol
        contrib[9] = -6*vol
        contrib[10] = 6*vol
        contrib[11] = vol
        contrib[12] = vol
        contrib[13] = -4*vol
        contrib[14] = -4*vol
        contrib[15] = -6*vol
        contrib[16] = -6*vol
        contrib[17] = -4*vol
        contrib[18] = -6*vol
        contrib[19] = 6*vol
        contrib[20] = vol
        contrib[21] = -6*vol
        contrib[22] = -4*vol
        contrib[23] = -4*vol
        contrib[24] = -6*vol
        contrib[25] = -6*vol
        contrib[26] = -4*vol
        contrib[27] = 6*vol
        contrib[28] = -6*vol
        contrib[29] = -6*vol
        contrib[30] = -6*vol
        contrib[31] = -4*vol
        contrib[32] = -4*vol
        contrib[33] = -4*vol
        contrib[34] = 32*vol
        contrib[35] = 16*vol
        contrib[36] = 16*vol
        contrib[37] = 16*vol
        contrib[38] = 16*vol
        contrib[39] = 8*vol
        contrib[40] = 32*vol
        contrib[41] = 16*vol
        contrib[42] = 8*vol
        contrib[43] = 16*vol
        contrib[44] = 16*vol
        contrib[45] = 32*vol
        contrib[46] = 16*vol
        contrib[47] = 8*vol
        contrib[48] = 16*vol
        contrib[49] = 32*vol
        contrib[50] = 16*vol
        contrib[51] = 16*vol
        contrib[52] = 32*vol
        contrib[53] = 16*vol
        contrib[54] = 32*vol
