###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


cdef class stiffness_3d_sym_P1(stiffness_3d_sym):
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(stiffness_3d_sym_P1 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.0277777777777778
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

        contrib[0] = l00*vol
        contrib[1] = l01*vol
        contrib[2] = l02*vol
        contrib[3] = l03*vol
        contrib[4] = l11*vol
        contrib[5] = l12*vol
        contrib[6] = l13*vol
        contrib[7] = l22*vol
        contrib[8] = l23*vol
        contrib[9] = l33*vol
