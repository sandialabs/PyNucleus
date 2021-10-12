###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, BOOL_t
from PyNucleus_fem.functions cimport function
from . twoPointFunctions cimport twoPointFunction, constantTwoPoint, parametrizedTwoPointFunction
from . interactionDomains cimport interactionDomain
from . fractionalOrders cimport fractionalOrderBase
from . kernelsCy cimport kernelCy

include "kernel_params_decl.pxi"


cdef class Kernel(twoPointFunction):
    cdef:
        public INDEX_t dim
        public kernelType kernelType
        public REAL_t min_singularity
        public REAL_t max_singularity
        public function horizon
        public interactionDomain interaction
        public twoPointFunction scaling
        public twoPointFunction phi
        public BOOL_t variableSingularity
        public BOOL_t variableHorizon
        public BOOL_t finiteHorizon
        public BOOL_t complement
        public BOOL_t variableScaling
        public BOOL_t variable
        public BOOL_t piecewise
        kernelCy c_kernel
        void *c_kernel_params
    cdef REAL_t getSingularityValue(self)
    cdef REAL_t getHorizonValue(self)
    cdef REAL_t getHorizonValue2(self)
    cdef REAL_t getScalingValue(self)
    cdef void evalParams(self, REAL_t[::1] x, REAL_t[::1] y)
    cdef void evalParamsPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y)


cdef class FractionalKernel(Kernel):
    cdef:
        public fractionalOrderBase s
        public BOOL_t variableOrder
    cdef REAL_t getsValue(self)
