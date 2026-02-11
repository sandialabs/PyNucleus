###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t
from PyNucleus_fem.functions cimport function
from . twoPointFunctions cimport twoPointFunction, ComplextwoPointFunction, constantTwoPoint, parametrizedTwoPointFunction
from . interactionDomains cimport interactionDomain
from . fractionalOrders cimport fractionalOrderBase, singleVariableUnsymmetricFractionalOrder

include "kernel_params_decl.pxi"
include "kernels_decl_REAL.pxi"
include "kernels_decl_COMPLEX.pxi"
include "fractionalKernel_decl.pxi"


cdef class RangedFractionalKernel(FractionalKernel):
    cdef:
        public admissibleOrders
        public BOOL_t normalized
        public REAL_t errorBound
        public INDEX_t M_min
        public INDEX_t M_max
        public REAL_t xi
        public REAL_t tempered


cdef class RangedVariableFractionalKernel(FractionalKernel):
    cdef:
        public function blockIndicator
        public admissibleOrders
        public BOOL_t normalized
        public REAL_t errorBound
        public INDEX_t M_min
        public INDEX_t M_max
        public REAL_t xi
