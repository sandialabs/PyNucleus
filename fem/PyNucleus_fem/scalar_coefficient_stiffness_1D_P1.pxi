###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class scalar_coefficient_stiffness_1d_sym_P1(stiffness_quadrature_matrix):
    def __init__(self, function diffusivity, simplexQuadratureRule qr=None):
        self.dim = 1
        if qr is None:
            qr = simplexXiaoGimbutas(1, 1)
        super(scalar_coefficient_stiffness_1d_sym_P1, self).__init__(diffusivity, qr)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline void eval(scalar_coefficient_stiffness_1d_sym_P1 self,
                          const REAL_t[:, ::1] simplex,
                          REAL_t[::1] contrib):
        cdef:
            REAL_t vol = 1.00000000000000
            REAL_t I = 0.
            INDEX_t k

        vol /= simplexVolume1D(simplex, self.temp)

        self.qr.evalFun(self.diffusivity, simplex, self.funVals)
        for k in range(self.qr.num_nodes):
            I += self.qr.weights[k] * self.funVals[k]
        contrib[0] = (vol) * I
        contrib[1] = (-vol) * I
        contrib[2] = (vol) * I
