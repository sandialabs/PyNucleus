###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}twoPointFunction:
    cdef:
        public BOOL_t symmetric
        public INDEX_t valueSize
    cdef void eval(self, REAL_t[::1] x, REAL_t[::1] y, {SCALAR}_t[::1] value)
    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, {SCALAR}_t* value)


cdef class {SCALAR_label}productTwoPoint({SCALAR_label}twoPointFunction):
    cdef:
        public {SCALAR_label}twoPointFunction f1, f2


cdef class {SCALAR_label}constantTwoPoint({SCALAR_label}twoPointFunction):
    cdef:
        public {SCALAR}_t value


cdef class {SCALAR_label}lookupTwoPoint({SCALAR_label}twoPointFunction):
    cdef:
        public DoFMap dm
        cellFinder2 cellFinder
        {SCALAR}_t[::1] vals1, vals2
        {SCALAR}_t[:, ::1] A


cdef class {SCALAR_label}parametrizedTwoPointFunction({SCALAR_label}twoPointFunction):
    cdef:
        void *params
    cdef void setParams(self, void *params)
    cdef void* getParams(self)


cdef class {SCALAR_label}productParametrizedTwoPoint({SCALAR_label}parametrizedTwoPointFunction):
    cdef:
        public {SCALAR_label}twoPointFunction f1, f2
