###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class scalar_coefficient_stiffness_2d_sym_P2(stiffness_quadrature_matrix):
    def __init__(self, function diffusivity, simplexQuadratureRule qr=None):
        self.dim = 2
        if qr is None:
            qr = simplexXiaoGimbutas(3, 2)
        super(scalar_coefficient_stiffness_2d_sym_P2, self).__init__(diffusivity, qr)

    cdef inline void eval(scalar_coefficient_stiffness_2d_sym_P2 self,
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
        contrib[0] = 0.
        for k in range(self.qr.num_nodes):
            contrib[0] += (l00*vol*(4*self.PHI[0, k] - 1)**2) * self.qr.weights[k] * self.funVals[k]
        contrib[1] = 0.
        for k in range(self.qr.num_nodes):
            contrib[1] += (l01*vol*(4*self.PHI[0, k] - 1)*(4*self.PHI[1, k] - 1)) * self.qr.weights[k] * self.funVals[k]
        contrib[2] = 0.
        for k in range(self.qr.num_nodes):
            contrib[2] += (l02*vol*(4*self.PHI[0, k] - 1)*(4*self.PHI[2, k] - 1)) * self.qr.weights[k] * self.funVals[k]
        contrib[3] = 0.
        for k in range(self.qr.num_nodes):
            contrib[3] += (4*vol*(4*self.PHI[0, k] - 1)*(l00*self.PHI[1, k] + l01*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[4] = 0.
        for k in range(self.qr.num_nodes):
            contrib[4] += (4*vol*(4*self.PHI[0, k] - 1)*(l01*self.PHI[2, k] + l02*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[5] = 0.
        for k in range(self.qr.num_nodes):
            contrib[5] += (4*vol*(4*self.PHI[0, k] - 1)*(l00*self.PHI[2, k] + l02*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[6] = 0.
        for k in range(self.qr.num_nodes):
            contrib[6] += (l11*vol*(4*self.PHI[1, k] - 1)**2) * self.qr.weights[k] * self.funVals[k]
        contrib[7] = 0.
        for k in range(self.qr.num_nodes):
            contrib[7] += (l12*vol*(4*self.PHI[1, k] - 1)*(4*self.PHI[2, k] - 1)) * self.qr.weights[k] * self.funVals[k]
        contrib[8] = 0.
        for k in range(self.qr.num_nodes):
            contrib[8] += (4*vol*(4*self.PHI[1, k] - 1)*(l01*self.PHI[1, k] + l11*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[9] = 0.
        for k in range(self.qr.num_nodes):
            contrib[9] += (4*vol*(4*self.PHI[1, k] - 1)*(l11*self.PHI[2, k] + l12*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[10] = 0.
        for k in range(self.qr.num_nodes):
            contrib[10] += (4*vol*(4*self.PHI[1, k] - 1)*(l01*self.PHI[2, k] + l12*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[11] = 0.
        for k in range(self.qr.num_nodes):
            contrib[11] += (l22*vol*(4*self.PHI[2, k] - 1)**2) * self.qr.weights[k] * self.funVals[k]
        contrib[12] = 0.
        for k in range(self.qr.num_nodes):
            contrib[12] += (4*vol*(4*self.PHI[2, k] - 1)*(l02*self.PHI[1, k] + l12*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[13] = 0.
        for k in range(self.qr.num_nodes):
            contrib[13] += (4*vol*(4*self.PHI[2, k] - 1)*(l12*self.PHI[2, k] + l22*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[14] = 0.
        for k in range(self.qr.num_nodes):
            contrib[14] += (4*vol*(4*self.PHI[2, k] - 1)*(l02*self.PHI[2, k] + l22*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[15] = 0.
        for k in range(self.qr.num_nodes):
            contrib[15] += (16*vol*(l00*self.PHI[1, k]**2 + 2*l01*self.PHI[0, k]*self.PHI[1, k] + l11*self.PHI[0, k]**2)) * self.qr.weights[k] * self.funVals[k]
        contrib[16] = 0.
        for k in range(self.qr.num_nodes):
            contrib[16] += (16*vol*(l01*self.PHI[1, k]*self.PHI[2, k] + l02*self.PHI[1, k]**2 + l11*self.PHI[0, k]*self.PHI[2, k] + l12*self.PHI[0, k]*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[17] = 0.
        for k in range(self.qr.num_nodes):
            contrib[17] += (16*vol*(l00*self.PHI[1, k]*self.PHI[2, k] + l01*self.PHI[0, k]*self.PHI[2, k] + l02*self.PHI[0, k]*self.PHI[1, k] + l12*self.PHI[0, k]**2)) * self.qr.weights[k] * self.funVals[k]
        contrib[18] = 0.
        for k in range(self.qr.num_nodes):
            contrib[18] += (16*vol*(l11*self.PHI[2, k]**2 + 2*l12*self.PHI[1, k]*self.PHI[2, k] + l22*self.PHI[1, k]**2)) * self.qr.weights[k] * self.funVals[k]
        contrib[19] = 0.
        for k in range(self.qr.num_nodes):
            contrib[19] += (16*vol*(l01*self.PHI[2, k]**2 + l02*self.PHI[1, k]*self.PHI[2, k] + l12*self.PHI[0, k]*self.PHI[2, k] + l22*self.PHI[0, k]*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[20] = 0.
        for k in range(self.qr.num_nodes):
            contrib[20] += (16*vol*(l00*self.PHI[2, k]**2 + 2*l02*self.PHI[0, k]*self.PHI[2, k] + l22*self.PHI[0, k]**2)) * self.qr.weights[k] * self.funVals[k]
