###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from . myTypes cimport INDEX_t, BOOL_t
from . linear_operators cimport sparseGraph


cpdef void cuthill_mckee(sparseGraph graph,
                         INDEX_t[::1] order,
                         BOOL_t reverse=*)
