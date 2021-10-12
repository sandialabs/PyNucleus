###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


cdef class scalar_coefficient_stiffness_3d_sym_P2(stiffness_quadrature_matrix):
    def __init__(self, function diffusivity, simplexQuadratureRule qr=None):
        self.dim = 3
        if qr is None:
            qr = simplexXiaoGimbutas(3, 3)
        super(scalar_coefficient_stiffness_3d_sym_P2, self).__init__(diffusivity, qr)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(scalar_coefficient_stiffness_3d_sym_P2 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 0.0277777777777778
            REAL_t l00, l01, l02, l03, l11, l12, l13, l22, l23, l33
            REAL_t I = 0.
            INDEX_t k

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
            contrib[3] += (l03*vol*(4*self.PHI[0, k] - 1)*(4*self.PHI[3, k] - 1)) * self.qr.weights[k] * self.funVals[k]
        contrib[4] = 0.
        for k in range(self.qr.num_nodes):
            contrib[4] += (4*vol*(4*self.PHI[0, k] - 1)*(l00*self.PHI[1, k] + l01*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[5] = 0.
        for k in range(self.qr.num_nodes):
            contrib[5] += (4*vol*(4*self.PHI[0, k] - 1)*(l01*self.PHI[2, k] + l02*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[6] = 0.
        for k in range(self.qr.num_nodes):
            contrib[6] += (4*vol*(4*self.PHI[0, k] - 1)*(l00*self.PHI[2, k] + l02*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[7] = 0.
        for k in range(self.qr.num_nodes):
            contrib[7] += (4*vol*(4*self.PHI[0, k] - 1)*(l00*self.PHI[3, k] + l03*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[8] = 0.
        for k in range(self.qr.num_nodes):
            contrib[8] += (4*vol*(4*self.PHI[0, k] - 1)*(l01*self.PHI[3, k] + l03*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[9] = 0.
        for k in range(self.qr.num_nodes):
            contrib[9] += (4*vol*(4*self.PHI[0, k] - 1)*(l02*self.PHI[3, k] + l03*self.PHI[2, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[10] = 0.
        for k in range(self.qr.num_nodes):
            contrib[10] += (l11*vol*(4*self.PHI[1, k] - 1)**2) * self.qr.weights[k] * self.funVals[k]
        contrib[11] = 0.
        for k in range(self.qr.num_nodes):
            contrib[11] += (l12*vol*(4*self.PHI[1, k] - 1)*(4*self.PHI[2, k] - 1)) * self.qr.weights[k] * self.funVals[k]
        contrib[12] = 0.
        for k in range(self.qr.num_nodes):
            contrib[12] += (l13*vol*(4*self.PHI[1, k] - 1)*(4*self.PHI[3, k] - 1)) * self.qr.weights[k] * self.funVals[k]
        contrib[13] = 0.
        for k in range(self.qr.num_nodes):
            contrib[13] += (4*vol*(4*self.PHI[1, k] - 1)*(l01*self.PHI[1, k] + l11*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[14] = 0.
        for k in range(self.qr.num_nodes):
            contrib[14] += (4*vol*(4*self.PHI[1, k] - 1)*(l11*self.PHI[2, k] + l12*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[15] = 0.
        for k in range(self.qr.num_nodes):
            contrib[15] += (4*vol*(4*self.PHI[1, k] - 1)*(l01*self.PHI[2, k] + l12*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[16] = 0.
        for k in range(self.qr.num_nodes):
            contrib[16] += (4*vol*(4*self.PHI[1, k] - 1)*(l01*self.PHI[3, k] + l13*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[17] = 0.
        for k in range(self.qr.num_nodes):
            contrib[17] += (4*vol*(4*self.PHI[1, k] - 1)*(l11*self.PHI[3, k] + l13*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[18] = 0.
        for k in range(self.qr.num_nodes):
            contrib[18] += (4*vol*(4*self.PHI[1, k] - 1)*(l12*self.PHI[3, k] + l13*self.PHI[2, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[19] = 0.
        for k in range(self.qr.num_nodes):
            contrib[19] += (l22*vol*(4*self.PHI[2, k] - 1)**2) * self.qr.weights[k] * self.funVals[k]
        contrib[20] = 0.
        for k in range(self.qr.num_nodes):
            contrib[20] += (l23*vol*(4*self.PHI[2, k] - 1)*(4*self.PHI[3, k] - 1)) * self.qr.weights[k] * self.funVals[k]
        contrib[21] = 0.
        for k in range(self.qr.num_nodes):
            contrib[21] += (4*vol*(4*self.PHI[2, k] - 1)*(l02*self.PHI[1, k] + l12*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[22] = 0.
        for k in range(self.qr.num_nodes):
            contrib[22] += (4*vol*(4*self.PHI[2, k] - 1)*(l12*self.PHI[2, k] + l22*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[23] = 0.
        for k in range(self.qr.num_nodes):
            contrib[23] += (4*vol*(4*self.PHI[2, k] - 1)*(l02*self.PHI[2, k] + l22*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[24] = 0.
        for k in range(self.qr.num_nodes):
            contrib[24] += (4*vol*(4*self.PHI[2, k] - 1)*(l02*self.PHI[3, k] + l23*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[25] = 0.
        for k in range(self.qr.num_nodes):
            contrib[25] += (4*vol*(4*self.PHI[2, k] - 1)*(l12*self.PHI[3, k] + l23*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[26] = 0.
        for k in range(self.qr.num_nodes):
            contrib[26] += (4*vol*(4*self.PHI[2, k] - 1)*(l22*self.PHI[3, k] + l23*self.PHI[2, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[27] = 0.
        for k in range(self.qr.num_nodes):
            contrib[27] += (l33*vol*(4*self.PHI[3, k] - 1)**2) * self.qr.weights[k] * self.funVals[k]
        contrib[28] = 0.
        for k in range(self.qr.num_nodes):
            contrib[28] += (4*vol*(4*self.PHI[3, k] - 1)*(l03*self.PHI[1, k] + l13*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[29] = 0.
        for k in range(self.qr.num_nodes):
            contrib[29] += (4*vol*(4*self.PHI[3, k] - 1)*(l13*self.PHI[2, k] + l23*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[30] = 0.
        for k in range(self.qr.num_nodes):
            contrib[30] += (4*vol*(4*self.PHI[3, k] - 1)*(l03*self.PHI[2, k] + l23*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[31] = 0.
        for k in range(self.qr.num_nodes):
            contrib[31] += (4*vol*(4*self.PHI[3, k] - 1)*(l03*self.PHI[3, k] + l33*self.PHI[0, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[32] = 0.
        for k in range(self.qr.num_nodes):
            contrib[32] += (4*vol*(4*self.PHI[3, k] - 1)*(l13*self.PHI[3, k] + l33*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[33] = 0.
        for k in range(self.qr.num_nodes):
            contrib[33] += (4*vol*(4*self.PHI[3, k] - 1)*(l23*self.PHI[3, k] + l33*self.PHI[2, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[34] = 0.
        for k in range(self.qr.num_nodes):
            contrib[34] += (16*vol*(l00*self.PHI[1, k]**2 + 2*l01*self.PHI[0, k]*self.PHI[1, k] + l11*self.PHI[0, k]**2)) * self.qr.weights[k] * self.funVals[k]
        contrib[35] = 0.
        for k in range(self.qr.num_nodes):
            contrib[35] += (16*vol*(l01*self.PHI[1, k]*self.PHI[2, k] + l02*self.PHI[1, k]**2 + l11*self.PHI[0, k]*self.PHI[2, k] + l12*self.PHI[0, k]*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[36] = 0.
        for k in range(self.qr.num_nodes):
            contrib[36] += (16*vol*(l00*self.PHI[1, k]*self.PHI[2, k] + l01*self.PHI[0, k]*self.PHI[2, k] + l02*self.PHI[0, k]*self.PHI[1, k] + l12*self.PHI[0, k]**2)) * self.qr.weights[k] * self.funVals[k]
        contrib[37] = 0.
        for k in range(self.qr.num_nodes):
            contrib[37] += (16*vol*(l00*self.PHI[1, k]*self.PHI[3, k] + l01*self.PHI[0, k]*self.PHI[3, k] + l03*self.PHI[0, k]*self.PHI[1, k] + l13*self.PHI[0, k]**2)) * self.qr.weights[k] * self.funVals[k]
        contrib[38] = 0.
        for k in range(self.qr.num_nodes):
            contrib[38] += (16*vol*(l01*self.PHI[1, k]*self.PHI[3, k] + l03*self.PHI[1, k]**2 + l11*self.PHI[0, k]*self.PHI[3, k] + l13*self.PHI[0, k]*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[39] = 0.
        for k in range(self.qr.num_nodes):
            contrib[39] += (16*vol*(l02*self.PHI[1, k]*self.PHI[3, k] + l03*self.PHI[1, k]*self.PHI[2, k] + l12*self.PHI[0, k]*self.PHI[3, k] + l13*self.PHI[0, k]*self.PHI[2, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[40] = 0.
        for k in range(self.qr.num_nodes):
            contrib[40] += (16*vol*(l11*self.PHI[2, k]**2 + 2*l12*self.PHI[1, k]*self.PHI[2, k] + l22*self.PHI[1, k]**2)) * self.qr.weights[k] * self.funVals[k]
        contrib[41] = 0.
        for k in range(self.qr.num_nodes):
            contrib[41] += (16*vol*(l01*self.PHI[2, k]**2 + l02*self.PHI[1, k]*self.PHI[2, k] + l12*self.PHI[0, k]*self.PHI[2, k] + l22*self.PHI[0, k]*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[42] = 0.
        for k in range(self.qr.num_nodes):
            contrib[42] += (16*vol*(l01*self.PHI[2, k]*self.PHI[3, k] + l02*self.PHI[1, k]*self.PHI[3, k] + l13*self.PHI[0, k]*self.PHI[2, k] + l23*self.PHI[0, k]*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[43] = 0.
        for k in range(self.qr.num_nodes):
            contrib[43] += (16*vol*(l11*self.PHI[2, k]*self.PHI[3, k] + l12*self.PHI[1, k]*self.PHI[3, k] + l13*self.PHI[1, k]*self.PHI[2, k] + l23*self.PHI[1, k]**2)) * self.qr.weights[k] * self.funVals[k]
        contrib[44] = 0.
        for k in range(self.qr.num_nodes):
            contrib[44] += (16*vol*(l12*self.PHI[2, k]*self.PHI[3, k] + l13*self.PHI[2, k]**2 + l22*self.PHI[1, k]*self.PHI[3, k] + l23*self.PHI[1, k]*self.PHI[2, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[45] = 0.
        for k in range(self.qr.num_nodes):
            contrib[45] += (16*vol*(l00*self.PHI[2, k]**2 + 2*l02*self.PHI[0, k]*self.PHI[2, k] + l22*self.PHI[0, k]**2)) * self.qr.weights[k] * self.funVals[k]
        contrib[46] = 0.
        for k in range(self.qr.num_nodes):
            contrib[46] += (16*vol*(l00*self.PHI[2, k]*self.PHI[3, k] + l02*self.PHI[0, k]*self.PHI[3, k] + l03*self.PHI[0, k]*self.PHI[2, k] + l23*self.PHI[0, k]**2)) * self.qr.weights[k] * self.funVals[k]
        contrib[47] = 0.
        for k in range(self.qr.num_nodes):
            contrib[47] += (16*vol*(l01*self.PHI[2, k]*self.PHI[3, k] + l03*self.PHI[1, k]*self.PHI[2, k] + l12*self.PHI[0, k]*self.PHI[3, k] + l23*self.PHI[0, k]*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[48] = 0.
        for k in range(self.qr.num_nodes):
            contrib[48] += (16*vol*(l02*self.PHI[2, k]*self.PHI[3, k] + l03*self.PHI[2, k]**2 + l22*self.PHI[0, k]*self.PHI[3, k] + l23*self.PHI[0, k]*self.PHI[2, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[49] = 0.
        for k in range(self.qr.num_nodes):
            contrib[49] += (16*vol*(l00*self.PHI[3, k]**2 + 2*l03*self.PHI[0, k]*self.PHI[3, k] + l33*self.PHI[0, k]**2)) * self.qr.weights[k] * self.funVals[k]
        contrib[50] = 0.
        for k in range(self.qr.num_nodes):
            contrib[50] += (16*vol*(l01*self.PHI[3, k]**2 + l03*self.PHI[1, k]*self.PHI[3, k] + l13*self.PHI[0, k]*self.PHI[3, k] + l33*self.PHI[0, k]*self.PHI[1, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[51] = 0.
        for k in range(self.qr.num_nodes):
            contrib[51] += (16*vol*(l02*self.PHI[3, k]**2 + l03*self.PHI[2, k]*self.PHI[3, k] + l23*self.PHI[0, k]*self.PHI[3, k] + l33*self.PHI[0, k]*self.PHI[2, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[52] = 0.
        for k in range(self.qr.num_nodes):
            contrib[52] += (16*vol*(l11*self.PHI[3, k]**2 + 2*l13*self.PHI[1, k]*self.PHI[3, k] + l33*self.PHI[1, k]**2)) * self.qr.weights[k] * self.funVals[k]
        contrib[53] = 0.
        for k in range(self.qr.num_nodes):
            contrib[53] += (16*vol*(l12*self.PHI[3, k]**2 + l13*self.PHI[2, k]*self.PHI[3, k] + l23*self.PHI[1, k]*self.PHI[3, k] + l33*self.PHI[1, k]*self.PHI[2, k])) * self.qr.weights[k] * self.funVals[k]
        contrib[54] = 0.
        for k in range(self.qr.num_nodes):
            contrib[54] += (16*vol*(l22*self.PHI[3, k]**2 + 2*l23*self.PHI[2, k]*self.PHI[3, k] + l33*self.PHI[2, k]**2)) * self.qr.weights[k] * self.funVals[k]
