###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class scalar_coefficient_stiffness_1d_sym_P2(stiffness_quadrature_matrix):
    def __init__(self, function diffusivity, simplexQuadratureRule qr=None):
        self.dim = 1
        if qr is None:
            qr = simplexXiaoGimbutas(3, 1)
        super(scalar_coefficient_stiffness_1d_sym_P2, self).__init__(diffusivity, qr)

    cdef inline void eval(scalar_coefficient_stiffness_1d_sym_P2 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 1.00000000000000
            REAL_t I = 0.
            INDEX_t k

        vol /= simplexVolume1D(simplex, self.temp)

        self.qr.evalFun(self.diffusivity, simplex, self.funVals)
        contrib[0] = 0.
        for k in range(self.qr.num_nodes):
            contrib[0] += (vol*(4*self.PHI[0, k] - 1)**2) * self.qr.weights[k] * self.funVals[k]
        contrib[1] = 0.
        for k in range(self.qr.num_nodes):
            contrib[1] += (-vol*(4*self.PHI[0, k] - 1)*(4*self.PHI[1, k] - 1)) * self.qr.weights[k] * self.funVals[k]
        contrib[2] = 0.
        for k in range(self.qr.num_nodes):
            contrib[2] += (-4*vol*(self.PHI[0, k] - self.PHI[1, k])*(4*self.PHI[0, k] - 1)) * self.qr.weights[k] * self.funVals[k]
        contrib[3] = 0.
        for k in range(self.qr.num_nodes):
            contrib[3] += (vol*(4*self.PHI[1, k] - 1)**2) * self.qr.weights[k] * self.funVals[k]
        contrib[4] = 0.
        for k in range(self.qr.num_nodes):
            contrib[4] += (4*vol*(self.PHI[0, k] - self.PHI[1, k])*(4*self.PHI[1, k] - 1)) * self.qr.weights[k] * self.funVals[k]
        contrib[5] = 0.
        for k in range(self.qr.num_nodes):
            contrib[5] += (16*vol*(self.PHI[0, k] - self.PHI[1, k])**2) * self.qr.weights[k] * self.funVals[k]
