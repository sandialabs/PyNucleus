import matplotlib.pyplot as plt
from time import time

######################################################################
# Get a fractional kernel
from PyNucleus import kernelFactory

from numpy import inf
kernelFracInf = kernelFactory('fractional', dim=2, s=0.75, horizon=inf)

######################################################################
# Generate an appropriate mesh
from PyNucleus import nonlocalMeshFactory, HOMOGENEOUS_DIRICHLET

# Get a mesh that is appropriate for the problem, i.e. with the required interaction domain.
meshFracInf, _ = nonlocalMeshFactory('disc', kernel=kernelFracInf, boundaryCondition=HOMOGENEOUS_DIRICHLET, hTarget=0.15)

######################################################################
# Assemble the operator
from PyNucleus import dofmapFactory, functionFactory

dmFracInf = dofmapFactory('P1', meshFracInf)

rhs = functionFactory('constant', 1.)
exact_solution = functionFactory('solFractional', dim=2, s=0.75)

b = dmFracInf.assembleRHS(rhs)
u_exact = dmFracInf.interpolate(exact_solution)
u = dmFracInf.zeros()

# Assemble the operator in dense format.
start = time()
A_fracInf = dmFracInf.assembleNonlocal(kernelFracInf, matrixFormat='dense')

start = time()
A_fracInf_h2 = dmFracInf.assembleNonlocal(kernelFracInf, matrixFormat='h2')

######################################################################
# Solve the linear system
from PyNucleus import solverFactory
from numpy import sqrt

solver = solverFactory('lu', A=A_fracInf, setup=True)
solver(b, u)

Hs_err = sqrt(abs(b.inner(u-u_exact)))

print('Hs error: {}'.format(Hs_err))
plt.figure().gca().set_title('Numerical solution, fractional kernel')
u.plot()

