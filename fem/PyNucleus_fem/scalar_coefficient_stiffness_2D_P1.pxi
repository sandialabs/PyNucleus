###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class scalar_coefficient_stiffness_2d_sym_P1(stiffness_quadrature_matrix):
    def __init__(self, function diffusivity, simplexQuadratureRule qr=None):
        self.dim = 2
        if qr is None:
            qr = simplexXiaoGimbutas(1, 2)
        super(scalar_coefficient_stiffness_2d_sym_P1, self).__init__(diffusivity, qr)

    cdef inline void eval(scalar_coefficient_stiffness_2d_sym_P1 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.250000000000000
            REAL_t l00, l01, l02, l11, l12, l22
            REAL_t I = 0.
            INDEX_t k

        vol /= simplexVolumeAndProducts2D(simplex, self.innerProducts, self.temp)
        l00 = self.innerProducts[0]
        l01 = self.innerProducts[1]
        l02 = self.innerProducts[2]
        l11 = self.innerProducts[3]
        l12 = self.innerProducts[4]
        l22 = self.innerProducts[5]

        self.qr.evalFun(self.diffusivity, simplex, self.funVals)
        for k in range(self.qr.num_nodes):
            I += self.qr.weights[k] * self.funVals[k]
        contrib[0] = (l00*vol) * I
        contrib[1] = (l01*vol) * I
        contrib[2] = (l02*vol) * I
        contrib[3] = (l11*vol) * I
        contrib[4] = (l12*vol) * I
        contrib[5] = (l22*vol) * I
