#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

"""
Operator interpolation
======================
"""

# %%
# This example demostrates the construction of a family of fractional
# Laplacians parametrized by the fractional order using operator
# interpolation. This can reduce the cost compared to assembling a new
# matrix for each value.
#
# The fractional Laplacian
#
# .. math::
#
#   (-\Delta)^{s} \text{ for } s \in [s_{\min}, s_{\max}] \subset (0, 1)
#
# is approximated by
#
# .. math::
#
#    (-\Delta)^{s} \approx \sum_{m=0}^{M} \Theta_{k,m}(s) (-\Delta)^{s_{k,m}} \text{ for } s \in \mathcal{S}_{k}
#
# for a sequence of intervals :math:`\mathcal{S}_{k}` that cover :math:`[s_{\min}, s_{\max}]` and scalar coefficients :math:`\Theta_{k,m}(s)`.
# The number of intervals and interpolation nodes is picked so that the interpolation error is dominated by the discretization error.
#
# The following example can be found at `examples/example_operator_interpolation.py <https://github.com/sandialabs/PyNucleus/blob/master/examples/example_operator_interpolation.py>`_ in the PyNucleus repository.

import logging
from PyNucleus_base.utilsFem import TimerManager
import numpy as np
import matplotlib.pyplot as plt

fmt = '{message}'
logging.basicConfig(level=logging.INFO,
                    format=fmt,
                    style='{',
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger('__main__')
timer = TimerManager(logger)

# %%
# We set up a mesh, a dofmap and a fractional kernel.
# Instead of specifying a single value for the fractional order, we allow a range of values :math:`[s_{\min}, s_{\max}]=[0.05, 0.95]`.

from PyNucleus import meshFactory, dofmapFactory, kernelFactory, functionFactory, solverFactory
from PyNucleus_nl.operatorInterpolation import admissibleSet

# Set up a mesh and a dofmap on it.
mesh = meshFactory('interval', hTarget=2e-3, a=-1, b=1)
dm = dofmapFactory('P1', mesh)

# Construct a RHS vector and a vector for the solution.
f = functionFactory('constant', 1.)
b = dm.assembleRHS(f)
u = dm.zeros()

# construct a fractional kernel with fractional order in S = [0.05, 0.95]
s = admissibleSet([0.05, 0.95])
kernel = kernelFactory('fractional', s=s, dim=mesh.dim)

# %%
# Next, we call the assembly of a nonlocal operator as before.
# The operator is set up to be constructed on-demand.
# We partition the interval S into several sub-interval and construct a Chebyshev interpolant on each sub-interval.
# Therefore this operation is fast.
with timer('operator creation'):
    A = dm.assembleNonlocal(kernel, matrixFormat='H2')

# %%
# Next, we choose the value of the fractional order. This needs to be within the range that we specified earlier.
# We set s = 0.75.
A.set(0.75)

# %%
# Let's solve a system.
# This triggers the assembly of the operators for the matrices at the interpolation nodes of the interval that contains s.
# The required matrices are constructed on-demand and then stay in memory.
with timer('solve 1 (slow)'):
    solver = solverFactory('cg-jacobi', A=A, setup=True)
    solver.maxIter = 1000
    numIter = solver(b, u)
logger.info('Solved problem for s={} in {} iterations (residual norm {})'.format(A.get(), numIter, solver.residuals[-1]))

# %%
# This solve is relatively slow, as it involves the assembly of the nonlocal operators that are needed for the interpolation.
# We select a different value for the fractional order that is close to the first.
# Solving a linear system with this value is faster as we have already assembled the operator needed for the interpolation.

with timer('solve 2 (fast)'):
    A.set(0.76)
    solver = solverFactory('cg-jacobi', A=A, setup=True)
    solver.maxIter = 1000
    numIter = solver(b, u)
logger.info('Solved problem for s={} in {} iterations (residual norm {})'.format(A.get(), numIter, solver.residuals[-1]))

# %%
# Next, we save the operator to file.
# This first triggers the assembly of all operators nescessary to represent every value in :math:`s\in[0.05,0.95]`.

import h5py

with timer('save operator'):
    h5_file = h5py.File('test.hdf5', 'w')
    A.HDF5write(h5_file.create_group('A'))
    dm.HDF5write(h5_file.create_group('dm'))
    h5_file.close()

# %%
# Next, we read the operator back in.

from PyNucleus_base.linear_operators import LinearOperator
from PyNucleus_fem.DoFMaps import DoFMap

with timer('load operator'):
    h5_file = h5py.File('test.hdf5', 'r')
    A_2 = LinearOperator.HDF5read(h5_file['A'])
    dm_2 = DoFMap.HDF5read(h5_file['dm'])
    h5_file.close()

# %%
# Finally, we set up and solve a series of linear systems with the operator we loaded.
f_2 = functionFactory('constant', 2.)
b_2 = dm_2.assembleRHS(f_2)
for sVal in np.linspace(0.1, 0.9, 9):
    with timer('solve 3 (fast)'):
        u_2 = dm_2.zeros()
        A_2.set(sVal)
        solver = solverFactory('cg-jacobi', A=A_2, setup=True)
        solver.maxIter = 1000
        numIter = solver(b_2, u_2)
    logger.info('Solved problem for s={:.1} in {} iterations (residual norm {})'.format(A_2.get(), numIter, solver.residuals[-1]))
    u_2.plot(label='s={:.1}'.format(sVal))
plt.legend()

# %%
plt.show()
