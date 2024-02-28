#!/usr/bin/env python
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

"""
Dirichlet condition for infinite horizon kernel
==================================================================================
"""

# %%
# The following example can be found at `examples/example_InfHorizonDirichlet.py <https://github.com/sandialabs/PyNucleus/blob/master/examples/example_InfHorizonDirichlet.py>`_ in the PyNucleus repository.
#
# We want to solve a problem with infinite horizon fractional kernel
#
# .. math::
#
#    \gamma(x,y) = \frac{c}{|x-y|^{d+2s}},
#
# where :math:`c` is the usual normalization constant, and imposes an inhomogeneous Dirichlet volume condition.
# We will have to make some assumption on the volume condition in order to make it computationally tractable.
# Here we assume that it is zero at some distance away from the domain.
#
# .. math::
#
#    (-\Delta)^s u &= f &&~in~ \Omega:=B_{1/2}(0),\\
#    u &= g  &&~in~ \Omega_\mathcal{I}:=B_{1}(0)\setminus B_{1/2}(0), \\
#    u &= 0  &&~in~ \mathbb{R}^d \setminus B_{1}(0),
#
# where :math:`f\equiv 1` and :math:`g` is chosen to match the known exact solution :math:`u(x)=C(1-|x|^2)_+^s` with some constant :math:`C`.

import numpy as np
import matplotlib.pyplot as plt
from PyNucleus import meshFactory, dofmapFactory, kernelFactory, functionFactory, solverFactory

# %%
# Construct a mesh for :math:`\Omega\cup \Omega_\mathcal{I}` and set up degree of freedom maps on :math:`\Omega` and :math:`\Omega_\mathcal{I}`.

radius = 1.0
mesh = meshFactory('disc', n=8, radius=radius)
for _ in range(4):
    mesh = mesh.refine()

# %%
# Get an indicator function for :math:`\Omega`.
# Subtract 1e-6 to avoid roundoff errors for nodes that are exactly on :math:`|x| = 0.5`.
OmegaIndicator = functionFactory('radialIndicator', 0.5*radius-1e-6)
dm = dofmapFactory('P1', mesh, OmegaIndicator)
dmBC = dm.getComplementDoFMap()

plt.figure()
dm.plot(printDoFIndices=False)
plt.figure()
dmBC.plot(printDoFIndices=False)

# %%
# Set up the kernel, rhs and known solution.

s = 0.75
kernel = kernelFactory('fractional', dim=mesh.dim, s=s, horizon=np.inf)
rhs = functionFactory('constant', 1.)
uex = functionFactory('solFractional', dim=mesh.dim, s=s, radius=radius)

# %%
# Assemble the linear system
#
# .. math::
#
#   A_{\Omega,\Omega} u_{\Omega} + A_{\Omega,\Omega_\mathcal{I}} u_{\Omega_\mathcal{I}} &= f \\
#                    u_{\Omega_\mathcal{I}} &= g
#
# Set up a solver for :math:`A_{\Omega,\Omega}`.

A_OmegaOmega = dm.assembleNonlocal(kernel, matrixFormat='H2')
A_OmegaOmegaI = dm.assembleNonlocal(kernel, dm2=dmBC)
f = dm.assembleRHS(rhs)
g = g = dmBC.interpolate(uex)
solver = solverFactory('lu', A=A_OmegaOmega, setup=True)

# %%
# Compute FE solution and error between FE solution and interpolation of analytic expression

u_Omega = dm.zeros()
solver(f-A_OmegaOmegaI*g, u_Omega)
u = u_Omega.augmentWithBoundaryData(g)
err = u.dm.interpolate(uex) - u

# %%
# Plot FE solution and error

plt.figure()
plt.title('FE solution')
u.plot(flat=True)
plt.figure()
plt.title('error')
err.plot(flat=True)
