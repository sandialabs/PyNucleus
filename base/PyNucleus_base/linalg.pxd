###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from . myTypes cimport INDEX_t, REAL_t, COMPLEX_t, BOOL_t

ctypedef fused SCALAR_t:
    REAL_t
    COMPLEX_t


cdef void forward_solve_csc(INDEX_t[::1] indptr,
                            INDEX_t[::1] indices,
                            SCALAR_t[::1] data,
                            SCALAR_t[::1] b,
                            SCALAR_t[::1] y,
                            BOOL_t unitDiagonal)

cdef void backward_solve_csc(INDEX_t[::1] indptr,
                             INDEX_t[::1] indices,
                             SCALAR_t[::1] data,
                             SCALAR_t[::1] b,
                             SCALAR_t[::1] y)

cdef void forward_solve_sss_noInverse(const INDEX_t[::1] indptr,
                                      const INDEX_t[::1] indices,
                                      const REAL_t[::1] data,
                                      const REAL_t[::1] invDiagonal,
                                      const REAL_t[::1] b,
                                      REAL_t[::1] y,
                                      BOOL_t unitDiagonal=*)
cdef void backward_solve_sss_noInverse(const INDEX_t[::1] indptr,
                                       const INDEX_t[::1] indices,
                                       const REAL_t[::1] data,
                                       const REAL_t[::1] invDiagonal,
                                       const REAL_t[::1] b,
                                       REAL_t[::1] y)
