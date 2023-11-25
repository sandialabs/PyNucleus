###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t
from PyNucleus_base.linear_operators cimport LinearOperator, ComplexLinearOperator
from PyNucleus_fem.algebraicOverlaps cimport algebraicOverlapManager
from PyNucleus_fem.distributed_operators cimport (DistributedLinearOperator,
                                                  ComplexDistributedLinearOperator,
                                                  CSR_DistributedLinearOperator,
                                                  ComplexCSR_DistributedLinearOperator)

include "smoothers_decl_REAL.pxi"
include "smoothers_decl_COMPLEX.pxi"


cdef class chebyshevPreconditioner(preconditioner):
    cdef:
        public REAL_t[::1] coeffs
        REAL_t[::1] temporaryMemory
        LinearOperator A
        algebraicOverlapManager overlap
        BOOL_t has_overlap
        INDEX_t lvlNo
    cdef INDEX_t matvec(self, REAL_t[::1] x, REAL_t[::1] y) except -1


cdef class chebyshevSmoother(separableSmoother):
    pass


cdef class iluPreconditioner(preconditioner):
    cdef:
        REAL_t[::1] temporaryMemory
        LinearOperator A
        LinearOperator preconditioner
    cdef INDEX_t matvec(self, REAL_t[::1] x, REAL_t[::1] y) except -1


cdef class iluSmoother(separableSmoother):
    pass


cdef class flexibleSmoother(separableSmoother):
    pass
