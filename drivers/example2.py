#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import matplotlib.pyplot as plt
from time import time

######################################################################
# Get a fractional kernel
from PyNucleus import kernelFactory

# show available options
kernelFactory.print()

from numpy import inf
kernelFracInf = kernelFactory('fractional', dim=2, s=0.75, horizon=inf)

print(kernelFracInf)
plt.figure().gca().set_title('Fractional kernel')
kernelFracInf.plot()

######################################################################
# Generate an appropriate mesh
from PyNucleus import nonlocalMeshFactory, HOMOGENEOUS_DIRICHLET

# Get a mesh that is appropriate for the problem, i.e. with the required interaction domain.
meshFracInf, _ = nonlocalMeshFactory('disc', kernel=kernelFracInf, boundaryCondition=HOMOGENEOUS_DIRICHLET, hTarget=0.15)

print(meshFracInf)
plt.figure().gca().set_title('Mesh for fractional kernel')
meshFracInf.plot()

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

print('Dense assembly took {}s'.format(time()-start))

start = time()
A_fracInf_h2 = dmFracInf.assembleNonlocal(kernelFracInf, matrixFormat='h2')

print('Hierarchical assembly took {}s'.format(time()-start))

print(A_fracInf)
print(A_fracInf_h2)

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

######################################################################
# Solve a problem with finite horizon
kernelConst = kernelFactory('constant', dim=2, horizon=0.2)

print(kernelConst)
plt.figure().gca().set_title('Constant kernel')
kernelConst.plot()

from PyNucleus import DIRICHLET

meshConst, nIConst = nonlocalMeshFactory('square', kernel=kernelConst, boundaryCondition=DIRICHLET, hTarget=0.18)

print(meshConst)
plt.figure().gca().set_title('Mesh for constant kernel')
meshConst.plot()

dmConst = dofmapFactory('P1', meshConst, nIConst['domain'])
dmConstInteraction = dmConst.getComplementDoFMap()

A_const = dmConst.assembleNonlocal(kernelConst, matrixFormat='sparsified')
B_const = dmConst.assembleNonlocal(kernelConst, dm2=dmConstInteraction, matrixFormat='sparsified')

g = functionFactory('Lambda', lambda x: -(x[0]**2 + x[1]**2)/4)
g_interp = dmConstInteraction.interpolate(g)

b = dmConst.assembleRHS(rhs)-(B_const*g_interp)
u = dmConst.zeros()

solver = solverFactory('cg', A=A_const, setup=True)
solver.maxIter = 1000
solver.tolerance = 1e-8

solver(b, u)

u_global = dmConst.augmentWithBoundaryData(u, g_interp)

plt.figure().gca().set_title('Numerical solution, constant kernel')
u_global.plot()

plt.figure().gca().set_title('Analytic solution, constant kernel')
u_global.dm.interpolate(g).plot()

print(A_const)

######################################################################
plt.show()
