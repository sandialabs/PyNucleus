#!/usr/bin/env python
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

"""
Neumann condition for finite horizon kernel
==============================================================================
"""

# %%
# The following example can be found at `examples/example_Neumann.py <https://github.com/sandialabs/PyNucleus/blob/master/examples/example_Neumann.py>`_ in the PyNucleus repository.
#
# .. math::
#
#    \gamma(x,y) = c(\delta) \chi_{B_\delta(x)}(y),
#
# where :math:`c(\delta)` is the usual normalization constant.
#
# .. math::
#
#    \int_{\Omega\cup\Omega_\mathcal{I}} (u(x)-u(y))\gamma(x,y)  &= f(x) &&~in~ \Omega:=B_{1}(0),\\
#    \int_{\Omega\cup\Omega_\mathcal{I}} (u(x)-u(y))\gamma(x,y)  &= g(x)  &&~in~ \Omega_\mathcal{I}:=B_{1+\delta}(0)\setminus B_{1}(0),
#
# where :math:`f\equiv 2` and
#
# .. math::
#
#    g(x)=c(\delta) \left[ |x| \left((1+\delta-|x|)^2-\delta^2\right) + \frac{1}{3} \left((1+\delta-|x|)^3+\delta^3\right)\right].
#
# The exact solution is :math:`u(x)=C-x^2` where :math:`C` is an arbitrary constant.

import numpy as np
import matplotlib.pyplot as plt
from PyNucleus import (nonlocalMeshFactory, dofmapFactory, kernelFactory,
                       functionFactory, solverFactory, NEUMANN, NO_BOUNDARY)
from PyNucleus_base.linear_operators import Dense_LinearOperator

# %%
# Set up kernel, load $f$, the analytic solution and the flux data :math:`g`.

kernel = kernelFactory('constant', dim=1, horizon=0.4)
load = functionFactory('constant', 2.)
analyticSolution = functionFactory('Lambda', lambda x: -x[0]**2)

def fluxFun(x):
    horizon = kernel.horizonValue
    dist = 1+horizon-abs(x[0])
    assert dist >= 0
    return 2*kernel.scalingValue * (abs(x[0]) * (dist**2-horizon**2) + 1./3. * (dist**3+horizon**3))

flux = functionFactory('Lambda', fluxFun)

# %%
# Construct a degree of freedom map for the entire mesh

mesh, nI = nonlocalMeshFactory('interval', kernel=kernel, boundaryCondition=NEUMANN)
for _ in range(3):
    mesh = mesh.refine()

dm = dofmapFactory('P1', mesh, NO_BOUNDARY)
dm

# %%
# The second return value of the nonlocal mesh factory contains indicator functions:

dm.interpolate(nI['domain']).plot(label='domain')
dm.interpolate(nI['boundary']).plot(label='local boundary')
dm.interpolate(nI['interaction']).plot(label='interaction')
plt.legend()

# %%
# Assemble the RHS vector by using the load on the domain :math:`\Omega` and the flux function on the interaction domain :math:`\Omega_\mathcal{I}`

A = dm.assembleNonlocal(kernel)
b = dm.assembleRHS(load*nI['domain'] + flux*(nI['interaction']+nI['boundary']))

# %%
# Solve the linear system. Since it is singular (shifts by a constant form the nullspace) we augment the system and then project out the zero mode.

u = dm.zeros()

# %%
# Augment the system
correction = Dense_LinearOperator(np.ones(A.shape))
solver = solverFactory('lu', A=A+correction, setup=True)
solver(b, u)

# %%
# project out the nullspace component
const = dm.ones()
u = u - (u.inner(const)/const.inner(const))*const

# %%
# Interpolate the exact solution and project out zero mode as well.

uex = dm.interpolate(analyticSolution)
uex = uex - (uex.inner(const)/const.inner(const))*const

# %%
u.plot(label='numerical', marker='x')
uex.plot(label='analytic')
plt.legend()
