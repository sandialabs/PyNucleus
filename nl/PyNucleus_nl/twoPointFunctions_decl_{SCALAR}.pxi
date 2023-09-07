###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}twoPointFunction:
    cdef:
        public BOOL_t symmetric
    cdef {SCALAR}_t eval(self, REAL_t[::1] x, REAL_t[::1] y)
    cdef {SCALAR}_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y)


cdef class {SCALAR_label}productTwoPoint({SCALAR_label}twoPointFunction):
    cdef:
        public twoPointFunction f1, f2


cdef class {SCALAR_label}constantTwoPoint({SCALAR_label}twoPointFunction):
    cdef:
        public {SCALAR}_t value


cdef class {SCALAR_label}parametrizedTwoPointFunction({SCALAR_label}twoPointFunction):
    cdef:
        void *params
    cdef void setParams(self, void *params)
    cdef void* getParams(self)


cdef class {SCALAR_label}productParametrizedTwoPoint({SCALAR_label}parametrizedTwoPointFunction):
    cdef:
        public {SCALAR_label}twoPointFunction f1, f2
