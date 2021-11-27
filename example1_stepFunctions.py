import matplotlib.pyplot as plt
from numpy import sqrt

######################################################################
# Get a mesh and refine it

from PyNucleus import meshFactory

mesh = meshFactory('square', ax=0., ay=0., bx=1., by=1.)
for _ in range(3):
    mesh = mesh.refine()

######################################################################
# Construct a finite element space
from PyNucleus import dofmapFactory

# We use piecewise linears
dm = dofmapFactory('P1', mesh)

######################################################################
# Construct some simple functions
from PyNucleus import functionFactory

# show available options
functionFactory.print()

# functions defined via Python lambdas
rhs_1 = functionFactory('Lambda', lambda x: 2*x[0]*(1-x[0]) + 2*x[1]*(1-x[1]))
exact_solution_1 = functionFactory('Lambda', lambda x: x[0]*(1-x[0])*x[1]*(1-x[1]))

# Functions defined via Cython implementations -> faster evaluation
rhs_2 = functionFactory('rhsFunSin2D')
exact_solution_2 = functionFactory('solSin2D')

# assemble right-hand side vectors and interpolate the exact solutions
b1 = dm.assembleRHS(rhs_1)
u_interp_1 = dm.interpolate(exact_solution_1)

print('Linear system RHS:', b1)
print('Interpolated solution:', u_interp_1)

b2 = dm.assembleRHS(rhs_2)
u_interp_2 = dm.interpolate(exact_solution_2)

plt.figure().gca().set_title('Interpolated solution')
u_interp_1.plot()

