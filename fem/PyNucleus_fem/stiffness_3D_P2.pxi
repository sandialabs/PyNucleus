###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class stiffness_3d_sym_P2(stiffness_3d_sym):
    cdef inline void eval(stiffness_3d_sym_P2 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.00555555555555556
            REAL_t l00, l01, l02, l03, l11, l12, l13, l22, l23, l33

        vol /= simplexVolumeAndProducts3D(simplex, self.innerProducts, self.temp)
        l00 = self.innerProducts[0]
        l01 = self.innerProducts[1]
        l02 = self.innerProducts[2]
        l03 = self.innerProducts[3]
        l11 = self.innerProducts[4]
        l12 = self.innerProducts[5]
        l13 = self.innerProducts[6]
        l22 = self.innerProducts[7]
        l23 = self.innerProducts[8]
        l33 = self.innerProducts[9]

        contrib[0] = 3*l00*vol
        contrib[1] = -l01*vol
        contrib[2] = -l02*vol
        contrib[3] = -l03*vol
        contrib[4] = vol*(-l00 + 3*l01)
        contrib[5] = -vol*(l01 + l02)
        contrib[6] = vol*(-l00 + 3*l02)
        contrib[7] = vol*(-l00 + 3*l03)
        contrib[8] = -vol*(l01 + l03)
        contrib[9] = -vol*(l02 + l03)
        contrib[10] = 3*l11*vol
        contrib[11] = -l12*vol
        contrib[12] = -l13*vol
        contrib[13] = vol*(3*l01 - l11)
        contrib[14] = vol*(-l11 + 3*l12)
        contrib[15] = -vol*(l01 + l12)
        contrib[16] = -vol*(l01 + l13)
        contrib[17] = vol*(-l11 + 3*l13)
        contrib[18] = -vol*(l12 + l13)
        contrib[19] = 3*l22*vol
        contrib[20] = -l23*vol
        contrib[21] = -vol*(l02 + l12)
        contrib[22] = vol*(3*l12 - l22)
        contrib[23] = vol*(3*l02 - l22)
        contrib[24] = -vol*(l02 + l23)
        contrib[25] = -vol*(l12 + l23)
        contrib[26] = vol*(-l22 + 3*l23)
        contrib[27] = 3*l33*vol
        contrib[28] = -vol*(l03 + l13)
        contrib[29] = -vol*(l13 + l23)
        contrib[30] = -vol*(l03 + l23)
        contrib[31] = vol*(3*l03 - l33)
        contrib[32] = vol*(3*l13 - l33)
        contrib[33] = vol*(3*l23 - l33)
        contrib[34] = 8*vol*(l00 + l01 + l11)
        contrib[35] = 4*vol*(l01 + 2*l02 + l11 + l12)
        contrib[36] = 4*vol*(l00 + l01 + l02 + 2*l12)
        contrib[37] = 4*vol*(l00 + l01 + l03 + 2*l13)
        contrib[38] = 4*vol*(l01 + 2*l03 + l11 + l13)
        contrib[39] = 4*vol*(l02 + l03 + l12 + l13)
        contrib[40] = 8*vol*(l11 + l12 + l22)
        contrib[41] = 4*vol*(2*l01 + l02 + l12 + l22)
        contrib[42] = 4*vol*(l01 + l02 + l13 + l23)
        contrib[43] = 4*vol*(l11 + l12 + l13 + 2*l23)
        contrib[44] = 4*vol*(l12 + 2*l13 + l22 + l23)
        contrib[45] = 8*vol*(l00 + l02 + l22)
        contrib[46] = 4*vol*(l00 + l02 + l03 + 2*l23)
        contrib[47] = 4*vol*(l01 + l03 + l12 + l23)
        contrib[48] = 4*vol*(l02 + 2*l03 + l22 + l23)
        contrib[49] = 8*vol*(l00 + l03 + l33)
        contrib[50] = 4*vol*(2*l01 + l03 + l13 + l33)
        contrib[51] = 4*vol*(2*l02 + l03 + l23 + l33)
        contrib[52] = 8*vol*(l11 + l13 + l33)
        contrib[53] = 4*vol*(2*l12 + l13 + l23 + l33)
        contrib[54] = 8*vol*(l22 + l23 + l33)
