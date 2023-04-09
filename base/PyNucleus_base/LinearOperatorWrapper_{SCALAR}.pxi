###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}LinearOperator_wrapper({SCALAR_label}LinearOperator):
    def __init__(self, INDEX_t num_rows, INDEX_t num_columns, matvec, {SCALAR}_t[::1] diagonal=None):
        super({SCALAR_label}LinearOperator_wrapper, self).__init__(num_rows, num_columns)
        self._matvec = matvec
        self._diagonal = diagonal

    cdef INDEX_t matvec(self, {SCALAR}_t[::1] x, {SCALAR}_t[::1] y) except -1:
        self._matvec(x, y)
        return 0
