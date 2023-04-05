###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from warnings import filterwarnings
filterwarnings("error", category=RuntimeWarning, module="importlib._bootstrap")

from . utilsFem import driver, problem
from . myTypes import REAL, INDEX, COMPLEX
from . blas import uninitialized, uninitialized_like


def get_include():
    import os
    return os.path.dirname(os.path.realpath(__file__))


from . solver_factory import solverFactory as solverFactoryClass

solverFactory = solverFactoryClass()
from . solvers import (noop_solver,
                       lu_solver, chol_solver,
                       cg_solver, gmres_solver, bicgstab_solver,
                       ichol_solver, ilu_solver,
                       jacobi_solver,
                       krylov_solver)
solverFactory.register(None, noop_solver)
solverFactory.register('lu', lu_solver)
solverFactory.register('chol', chol_solver, aliases=['cholesky', 'cholmod'])
solverFactory.register('cg', cg_solver)
solverFactory.register('gmres', gmres_solver)
solverFactory.register('bicgstab', bicgstab_solver)
solverFactory.register('ichol', ichol_solver)
solverFactory.register('ilu', ilu_solver)
solverFactory.register('jacobi', jacobi_solver, aliases=['diagonal'])

from . config import use_pyamg
if use_pyamg:
    from . solvers import pyamg_solver
    solverFactory.register('pyamg', pyamg_solver)
from . config import use_pypardiso
if use_pypardiso:
    from . solvers import pardiso_lu_solver
    solverFactory.register('pardiso_lu', pardiso_lu_solver)

from . solvers import complex_lu_solver, complex_gmres_solver
solverFactory.register('complex_lu', complex_lu_solver)
solverFactory.register('complex_gmres', complex_gmres_solver)


__all__ = [REAL, INDEX, COMPLEX,
           solverFactory,
           driver, problem]
