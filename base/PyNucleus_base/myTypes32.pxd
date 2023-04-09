###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from numpy cimport int8_t, int32_t, int64_t, float32_t, complex64_t, npy_bool

ctypedef int32_t INDEX_t
ctypedef int8_t TAG_t
ctypedef int64_t ENCODE_t
ctypedef float32_t REAL_t
ctypedef complex64_t COMPLEX_t
ctypedef npy_bool BOOL_t
