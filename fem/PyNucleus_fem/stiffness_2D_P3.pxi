###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


cdef class stiffness_2d_sym_P3(stiffness_2d_sym):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(stiffness_2d_sym_P3 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.00312500000000000
            REAL_t l00, l01, l02, l11, l12, l22

        vol /= simplexVolumeAndProducts2D(simplex, self.innerProducts, self.temp)
        l00 = self.innerProducts[0]
        l01 = self.innerProducts[1]
        l02 = self.innerProducts[2]
        l11 = self.innerProducts[3]
        l12 = self.innerProducts[4]
        l22 = self.innerProducts[5]

        contrib[0] = 68*l00*vol
        contrib[1] = 14*l01*vol
        contrib[2] = 14*l02*vol
        contrib[3] = 6*vol*(l00 + 19*l01)
        contrib[4] = 6*vol*(l00 - 8*l01)
        contrib[5] = 6*vol*(l01 + l02)
        contrib[6] = 6*vol*(l01 + l02)
        contrib[7] = 6*vol*(l00 - 8*l02)
        contrib[8] = 6*vol*(l00 + 19*l02)
        contrib[9] = 18*vol*(l00 + l01 + l02)
        contrib[10] = 68*l11*vol
        contrib[11] = 14*l12*vol
        contrib[12] = 6*vol*(-8*l01 + l11)
        contrib[13] = 6*vol*(19*l01 + l11)
        contrib[14] = 6*vol*(l11 + 19*l12)
        contrib[15] = 6*vol*(l11 - 8*l12)
        contrib[16] = 6*vol*(l01 + l12)
        contrib[17] = 6*vol*(l01 + l12)
        contrib[18] = 18*vol*(l01 + l11 + l12)
        contrib[19] = 68*l22*vol
        contrib[20] = 6*vol*(l02 + l12)
        contrib[21] = 6*vol*(l02 + l12)
        contrib[22] = 6*vol*(-8*l12 + l22)
        contrib[23] = 6*vol*(19*l12 + l22)
        contrib[24] = 6*vol*(19*l02 + l22)
        contrib[25] = 6*vol*(-8*l02 + l22)
        contrib[26] = 18*vol*(l02 + l12 + l22)
        contrib[27] = 270*vol*(l00 + l01 + l11)
        contrib[28] = 54*vol*(-l00 + 2*l01 - l11)
        contrib[29] = -27*vol*(l01 + 2*l02 + l11 + l12)
        contrib[30] = -27*vol*(l01 + 2*l02 + l11 + l12)
        contrib[31] = -27*vol*(l00 + l01 + l02 + 2*l12)
        contrib[32] = 135*vol*(l00 + l01 + l02 + 2*l12)
        contrib[33] = 162*vol*(l01 + 2*l02 + l11 + l12)
        contrib[34] = 270*vol*(l00 + l01 + l11)
        contrib[35] = 135*vol*(l01 + 2*l02 + l11 + l12)
        contrib[36] = -27*vol*(l01 + 2*l02 + l11 + l12)
        contrib[37] = -27*vol*(l00 + l01 + l02 + 2*l12)
        contrib[38] = -27*vol*(l00 + l01 + l02 + 2*l12)
        contrib[39] = 162*vol*(l00 + l01 + l02 + 2*l12)
        contrib[40] = 270*vol*(l11 + l12 + l22)
        contrib[41] = 54*vol*(-l11 + 2*l12 - l22)
        contrib[42] = -27*vol*(2*l01 + l02 + l12 + l22)
        contrib[43] = -27*vol*(2*l01 + l02 + l12 + l22)
        contrib[44] = 162*vol*(2*l01 + l02 + l12 + l22)
        contrib[45] = 270*vol*(l11 + l12 + l22)
        contrib[46] = 135*vol*(2*l01 + l02 + l12 + l22)
        contrib[47] = -27*vol*(2*l01 + l02 + l12 + l22)
        contrib[48] = 162*vol*(l01 + 2*l02 + l11 + l12)
        contrib[49] = 270*vol*(l00 + l02 + l22)
        contrib[50] = 54*vol*(-l00 + 2*l02 - l22)
        contrib[51] = 162*vol*(l00 + l01 + l02 + 2*l12)
        contrib[52] = 270*vol*(l00 + l02 + l22)
        contrib[53] = 162*vol*(2*l01 + l02 + l12 + l22)
        contrib[54] = 648*vol*(l00 + l01 + l02 + l11 + l12 + l22)
