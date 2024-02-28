#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

"""
A simple PDE problem
================================
"""

# %%
# In this first example, we will construct a finite element discretization of a classical PDE problem and solve it.
# The full code of this example can be found in `examples/example_pde.py <https://github.com/sandialabs/PyNucleus/blob/master/examples/example_pde.py>`_ in the PyNucleus repository.
#
# Factories
# ---------
#
# The creation of different groups of objects, such as finite element spaces or meshes, use factories.
# The available classes that a factory provides can be displayed by calling the ``print()`` method of the factory.
# An object is built by passing the name of the desired class and additional parameters to the factory.
# If this sounds vague now, don't worry, the examples below will make it clear.
#
# Meshes
# ------
#
# The first object we need to create is a mesh to support the finite element discretization.
# The output of the code snippet is given below.

import matplotlib.pyplot as plt
from PyNucleus import meshFactory
meshFactory.print()

# %%
# We see what the different meshes that the ``meshFactory`` can construct and the default values for associated parameters.
# We choose to construct a mesh for a square domain :math:`\Omega=[0, 1] \times [0, 1]` and refining it uniformly three times:

mesh = meshFactory('square', ax=0., ay=0., bx=1., by=1.)
for _ in range(3):
    mesh = mesh.refine()
print('Mesh:', mesh)

# %%
# We have created a 2d mesh with 289 vertices and 512 cells.
# The mesh (as well as many other objects) can be plotted:

plt.figure().gca().set_title('Mesh')
mesh.plot()

# %%
# Many PyNucleus objects have a ``plot`` method, similar to the mesh that we just created.
#
# DoFMaps
# -------
#
# In the next step, we create a finite element space on the mesh.
# By default, we assume a Dirichlet condition on the entire boundary of the domain.
# We build a piecewise linear finite element space.

from PyNucleus import dofmapFactory
dofmapFactory.print()

# %%
# We use a piecewise linear continuous finite element space.
dm = dofmapFactory('P1c', mesh)
print('DoFMap:', dm)

# %%
# By default all degrees of freedom on the boundary of the domain are considered to be "boundary dofs".
# The ``dofmapFactory`` take an optional second argument that allows to pass an indicator function to control what is considered a boundary dof.
#
# Next, we plot the locations of the degrees of freedom on the mesh.
plt.figure().gca().set_title('DoFMap')
dm.plot()

# %%
# Functions and vectors
# ---------------------
#
# We will be solving the problem
#
# .. math::
#
#    -\Delta u &= f & \text{ in } \Omega, \\
#    u &= 0 & \text{ on } \partial \Omega,
#
# or, more precisely, its weak form
#
# .. math::
#
#    &\text{Find } u \in H^1(\Omega) \text{ such that}\\
#    &\int_\Omega \nabla u \cdot \nabla v = \int_\Omega fv  \qquad\forall v \in H^1_0(\Omega).
#
# Functions are created using the ``functionFactory``.
from PyNucleus import functionFactory
functionFactory.print()

# %%
# We will consider two different forcing functions :math:`f`.
# Pointwise evalutated functions such as :math:`f` can either be defined in Python, or in Cython.
# A generic Python function can be used via the ``Lambda`` function class.
rhs_1 = functionFactory('Lambda', lambda x: 2*x[0]*(1-x[0]) + 2*x[1]*(1-x[1]))
exact_solution_1 = functionFactory('Lambda', lambda x: x[0]*(1-x[0])*x[1]*(1-x[1]))

# %%
# The advantage of Cython functions is that their code is compiled, which speeds up evaluation significantly.
# A couple of compiled functions are already available via the ``functionFactory``.
rhs_2 = functionFactory('rhsFunSin2D')
exact_solution_2 = functionFactory('solSin2D')

# %%
# We assemble the right-hand side
#
# .. math::
#
#    \int_\Omega f v
#
# of the linear system by calling the ``assembleRHS`` method of the DoFMap object, and interpolate the exact solutions into the finite element space.

b1 = dm.assembleRHS(rhs_1)
u_interp_1 = dm.interpolate(exact_solution_1)

print('Linear system RHS:', b1)
print('Interpolated solution:', u_interp_1)

b2 = dm.assembleRHS(rhs_2)
u_interp_2 = dm.interpolate(exact_solution_2)

# %%
# We plot the interpolated exact solution.
plt.figure().gca().set_title('Interpolated solution')
u_interp_1.plot()

# %%
# Matrices
# --------
#
# We assemble the stiffness matrix associated with the Laplacian
#
# .. math::
#
#    \int_\Omega \nabla u \cdot \nabla v

laplacian = dm.assembleStiffness()

print('Linear system matrix:', laplacian)

# %%
# Solvers
# -------
#
# Now that we have assembled our linear system, we want to solve it.

from PyNucleus import solverFactory
solverFactory.print()

# %%
# We choose to set up an LU direct solver.

solver_direct = solverFactory('lu', A=laplacian)
solver_direct.setup()
print('Direct solver:', solver_direct)

# %%
# We also set up an iterative solver. Note that we need to specify some additional parameters.

solver_krylov = solverFactory('cg', A=laplacian)
solver_krylov.setup()
solver_krylov.maxIter = 100
solver_krylov.tolerance = 1e-8
print('Krylov solver:', solver_krylov)

# %%
# We allocate a zero vector for the solution of the linear system and solve the equation using the first right-hand side.

u1 = dm.zeros()
solver_direct(b1, u1)

# %%
# We use the interative solver for the second right-hand side.

u2 = dm.zeros()
numIter = solver_krylov(b2, u2)

print('Number of iterations:', numIter)

# %%
# We plot the difference between one of the numerical solutions we just computed and the interpolant of the known analytic solution.
plt.figure().gca().set_title('Error')
(u_interp_1-u1).plot(flat=True)


# %%
# Norms and inner products
# ------------------------
#
# Finally, we want to check that we actually solved the system by computing :math:`\ell^2` residual errors.

print('Residual error 1st solve: ', (b1-laplacian*u1).norm())
print('Residual error 2nd solve: ', (b2-laplacian*u2).norm())

# %%
# We observe that the first solution is accurate up to machine epsilon, and that the second solution satisfies the tolerance that we specified earlier.
#
# We also compute errors in :math:`H^1_0` and :math:`L^2` norms.
# In order to compute the :math:`L^2` error we need to assemble the mass matrix
#
# .. math::
#
#    \int_\Omega u v.
#
# For the :math:`H^1_0` error we can reuse the stiffness matrix that we assembled earlier.

mass = dm.assembleMass()

from numpy import sqrt

H10_error_1 = sqrt(b1.inner(u_interp_1-u1))
L2_error_1 = sqrt((u_interp_1-u1).inner(mass*(u_interp_1-u1)))
H10_error_2 = sqrt(b2.inner(u_interp_2-u2))
L2_error_2 = sqrt((u_interp_2-u2).inner(mass*(u_interp_2-u2)))

print('1st problem - H10:', H10_error_1, 'L2:', L2_error_1)
print('2nd problem - H10:', H10_error_2, 'L2:', L2_error_2)

# %%
# This concludes our first example.
# Next, we turn to nonlocal equations.

plt.show()
