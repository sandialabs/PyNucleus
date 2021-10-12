###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################



cdef class {SCALAR_label_lc_}fe_vector:
    cdef:
        {SCALAR}_t[::1] data
        bytes format
        public DoFMap dm

    cpdef REAL_t norm(self, BOOL_t acc=*, BOOL_t asynchronous=*)
    cpdef {SCALAR}_t inner(self, other, BOOL_t accSelf=*, BOOL_t accOther=*, BOOL_t asynchronous=*)
