###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class FractionalKernel(Kernel):
    cdef:
        public fractionalOrderBase s
        public INDEX_t derivative
        public INDEX_t termNo
        REAL_t[::1] tempVec
    cdef REAL_t getsValue(self)
    cdef void setsValue(self, REAL_t s)
    cdef REAL_t gettemperedValue(self)
    cdef void settemperedValue(self, REAL_t tempered)


cdef class FractionalBoundaryKernel(FractionalKernel):
    cdef:
        public BOOL_t isSphericalManifold


cdef class MultiSingularityFractionalKernel(MultiSingularityKernel):
    cdef:
        public BOOL_t variableOrder
        public INDEX_t derivative
