###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from pyamg import smoothed_aggregation_solver

cdef class pyamg_solver(iterative_solver):
    def __init__(self, LinearOperator A=None, num_rows=-1):
        iterative_solver.__init__(self, A, num_rows)

    cpdef void setup(self, LinearOperator A=None):
        iterative_solver.setup(self, A)
        # self.ml = ruge_stuben_solver(self.A.to_csr(),
        #                              coarse_solver='splu',
        #                              max_coarse=2500,
        #                              presmoother=('gauss_seidel', {'sweep': 'forward'}),
        #                              postsmoother=('gauss_seidel', {'sweep': 'backward'}))

        self.ml = smoothed_aggregation_solver(self.A.to_csr(),
                                              np.ones((self.num_rows)),
                                              smooth=None,
                                              coarse_solver='splu',
                                              max_coarse=2500,
                                              presmoother=('gauss_seidel', {'sweep': 'forward'}),
                                              postsmoother=('gauss_seidel', {'sweep': 'backward'}))
        self.initialized = True

    cdef int solve(self, vector_t b, vector_t x) except -1:
        residuals = []
        x_np = np.array(x, copy=False)
        if self.x0 is not None:
            x_np[:] = self.ml.solve(np.array(b, copy=False),
                                    x0=np.array(self.x0, copy=False),
                                    tol=self.tol, maxiter=self.maxIter, residuals=residuals, accel='cg')
        else:
            x_np[:] = self.ml.solve(np.array(b, copy=False),
                                    tol=self.tol, maxiter=self.maxIter, residuals=residuals, accel='cg')
        self.residuals = residuals
        return len(residuals)

    def __str__(self):
        return str(self.ml)
