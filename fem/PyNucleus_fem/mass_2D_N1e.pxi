###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class mass_2d_sym_N1e(mass_2d):
    def __init__(self):
        mass_2d.__init__(self)
        self.needsCellInfo = True

    cdef inline void eval(mass_2d_sym_N1e self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.020833333333333332/4.
            REAL_t l00, l01, l02, l11, l12, l22
            REAL_t o01 = 1., o12 = 1., o20 = 1.

        vol /= simplexVolumeAndProducts2D(simplex, self.innerProducts, self.temp)
        l00 = self.innerProducts[0]
        l01 = self.innerProducts[1]
        l02 = self.innerProducts[2]
        l11 = self.innerProducts[3]
        l12 = self.innerProducts[4]
        l22 = self.innerProducts[5]

        if self.cell[0] > self.cell[1]:
            o01 = -1.
        if self.cell[1] > self.cell[2]:
            o12 = -1.
        if self.cell[2] > self.cell[0]:
            o20 = -1.

        contrib[0] = 2*vol*(l00 - l01 + l11)
        contrib[1] = o01*o12 * vol*(l01 - 2*l02 - l11 + l12)
        contrib[2] = -o01*o20 * vol*(l00 - l01 - l02 + 2*l12)
        contrib[3] = 2*vol*(l11 - l12 + l22)
        contrib[4] = -o12*o20 * vol*(2*l01 - l02 - l12 + l22)
        contrib[5] = 2*vol*(l00 - l02 + l22)
