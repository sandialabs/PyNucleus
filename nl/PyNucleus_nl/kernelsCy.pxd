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

include "kernel_params_decl.pxi"


ctypedef REAL_t (*kernel_fun_t)(REAL_t *x, REAL_t *y, void* user_data)


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
        kernel_fun_t kernelFun
        void *c_kernel_params
    cdef REAL_t getSingularityValue(self)
    cdef void setSingularityValue(self, REAL_t singularity)
    cdef REAL_t getHorizonValue(self)
    cdef void setHorizonValue(self, REAL_t horizon)
    cdef REAL_t getHorizonValue2(self)
    cdef REAL_t getScalingValue(self)
    cdef void setScalingValue(self, REAL_t scaling)
    cdef void evalParams(self, REAL_t[::1] x, REAL_t[::1] y)
    cdef void evalParamsPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y)
    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y)


cdef class FractionalKernel(Kernel):
    cdef:
        public fractionalOrderBase s
        public BOOL_t variableOrder
        public BOOL_t boundary
    cdef REAL_t getsValue(self)
    cdef void setsValue(self, REAL_t s)


cdef class RangedFractionalKernel(FractionalKernel):
    cdef:
        public admissibleOrders
        public BOOL_t normalized
        public REAL_t errorBound
        public INDEX_t M_min
        public INDEX_t M_max
        public REAL_t xi


cdef class RangedVariableFractionalKernel(FractionalKernel):
    cdef:
        public function blockIndicator
        public admissibleOrders
        public BOOL_t normalized
        public REAL_t errorBound
        public INDEX_t M_min
        public INDEX_t M_max
        public REAL_t xi


