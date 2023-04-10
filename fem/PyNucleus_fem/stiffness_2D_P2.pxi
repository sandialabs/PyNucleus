###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class stiffness_2d_sym_P2(stiffness_2d_sym):
    cdef inline void eval(stiffness_2d_sym_P2 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.0833333333333333
            REAL_t l00, l01, l02, l11, l12, l22

        vol /= simplexVolumeAndProducts2D(simplex, self.innerProducts, self.temp)
        l00 = self.innerProducts[0]
        l01 = self.innerProducts[1]
        l02 = self.innerProducts[2]
        l11 = self.innerProducts[3]
        l12 = self.innerProducts[4]
        l22 = self.innerProducts[5]

        contrib[0] = 3*l00*vol
        contrib[1] = -l01*vol
        contrib[2] = -l02*vol
        contrib[3] = 4*l01*vol
        contrib[4] = 0
        contrib[5] = 4*l02*vol
        contrib[6] = 3*l11*vol
        contrib[7] = -l12*vol
        contrib[8] = 4*l01*vol
        contrib[9] = 4*l12*vol
        contrib[10] = 0
        contrib[11] = 3*l22*vol
        contrib[12] = 0
        contrib[13] = 4*l12*vol
        contrib[14] = 4*l02*vol
        contrib[15] = 8*vol*(l00 + l01 + l11)
        contrib[16] = 4*vol*(l01 + 2*l02 + l11 + l12)
        contrib[17] = 4*vol*(l00 + l01 + l02 + 2*l12)
        contrib[18] = 8*vol*(l11 + l12 + l22)
        contrib[19] = 4*vol*(2*l01 + l02 + l12 + l22)
        contrib[20] = 8*vol*(l00 + l02 + l22)
