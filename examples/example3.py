#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

######################################################################
# preamble
import logging
from PyNucleus_base.utilsFem import TimerManager
from PyNucleus import meshFactory, dofmapFactory, kernelFactory, functionFactory, solverFactory
from PyNucleus_nl.operatorInterpolation import admissibleSet
import h5py
from PyNucleus_base.linear_operators import LinearOperator
from PyNucleus_fem.DoFMaps import DoFMap

fmt = '{message}'
logging.basicConfig(level=logging.INFO,
                    format=fmt,
                    style='{',
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger('__main__')
timer = TimerManager(logger)

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

######################################################################
# The operator is set up to be constructed on-demand.
# We partition the interval S into several sub-interval and construct a Chebyshev interpolant on each sub-interval.
# Therefore this operation is fast.
with timer('operator creation'):
    A = dm.assembleNonlocal(kernel, matrixFormat='H2')

######################################################################
# Set s = 0.75.
A.set(0.75)

# Let's solve a system.
# This triggers the assembly of the operators for the matrices at the interpolation nodes of the interval that contains s.
# The required matrices are constructed on-demand and then stay in memory.
with timer('solve 1 (slow)'):
    solver = solverFactory('cg-jacobi', A=A, setup=True)
    solver.maxIter = 1000
    numIter = solver(b, u)
logger.info('Solved problem for s={} in {} iterations (residual norm {})'.format(A.get(), numIter, solver.residuals[-1]))

######################################################################
# Let's solve a second sytem for a closeby value of s.
# This should be faster since we no longer need to assemble any matrices.
with timer('solve 2 (fast)'):
    A.set(0.76)
    solver = solverFactory('cg-jacobi', A=A, setup=True)
    solver.maxIter = 1000
    numIter = solver(b, u)
logger.info('Solved problem for s={} in {} iterations (residual norm {})'.format(A.get(), numIter, solver.residuals[-1]))

######################################################################
# We can save the operator and the DoFMap to a file.
# This will trigger the assembly of all matrices.
with timer('save operator'):
    h5_file = h5py.File('test.hdf5', 'w')
    A.HDF5write(h5_file.create_group('A'))
    dm.HDF5write(h5_file.create_group('dm'))
    h5_file.close()

######################################################################
# Now we can read them back in.
with timer('load operator'):
    h5_file = h5py.File('test.hdf5', 'r')
    A_2 = LinearOperator.HDF5read(h5_file['A'])
    dm_2 = DoFMap.HDF5read(h5_file['dm'])
    h5_file.close()

# Set up and solve a system with the operator we loaded.
f_2 = functionFactory('constant', 2.)
b_2 = dm_2.assembleRHS(f_2)
u_2 = dm_2.zeros()
with timer('solve 3 (fast)'):
    A_2.set(0.8)
    solver = solverFactory('cg-jacobi', A=A_2, setup=True)
    solver.maxIter = 1000
    numIter = solver(b_2, u_2)
logger.info('Solved problem for s={} in {} iterations (residual norm {})'.format(A_2.get(), numIter, solver.residuals[-1]))

