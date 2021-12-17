#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from PyNucleus.packageTools.sphinxTools import codeRegionManager

mgr = codeRegionManager()

with mgr.add('imports'):
    import matplotlib.pyplot as plt
    from time import time

with mgr.add('kernelFracInf'):
    ######################################################################
    # Get a fractional kernel
    from PyNucleus import kernelFactory

with mgr.add('kernelFracInf', onlyIfFinal=True):
    # show available options
    kernelFactory.print()

with mgr.add('kernelFracInf'):
    from numpy import inf
    kernelFracInf = kernelFactory('fractional', dim=2, s=0.75, horizon=inf)

with mgr.add('kernelFracInf', onlyIfFinal=True):
    print(kernelFracInf)
    plt.figure().gca().set_title('Fractional kernel')
    kernelFracInf.plot()


with mgr.add('meshFracInf'):
    ######################################################################
    # Generate an appropriate mesh
    from PyNucleus import nonlocalMeshFactory, HOMOGENEOUS_DIRICHLET

    # Get a mesh that is appropriate for the problem, i.e. with the required interaction domain.
    meshFracInf, _ = nonlocalMeshFactory('disc', kernel=kernelFracInf, boundaryCondition=HOMOGENEOUS_DIRICHLET, hTarget=0.15)

with mgr.add('meshFracInf', onlyIfFinal=True):
    print(meshFracInf)
    plt.figure().gca().set_title('Mesh for fractional kernel')
    meshFracInf.plot()

with mgr.add('assemblyFracInf'):
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
with mgr.add('assemblyFracInf', onlyIfFinal=True):
    print('Dense assembly took {}s'.format(time()-start))

with mgr.add('assemblyFracInf'):
    start = time()
    A_fracInf_h2 = dmFracInf.assembleNonlocal(kernelFracInf, matrixFormat='h2')
with mgr.add('assemblyFracInf', onlyIfFinal=True):
    print('Hierarchical assembly took {}s'.format(time()-start))

    print(A_fracInf)
    print(A_fracInf_h2)

with mgr.add('solveFracInf'):
    ######################################################################
    # Solve the linear system
    from PyNucleus import solverFactory
    from numpy import sqrt

    solver = solverFactory('lu', A=A_fracInf, setup=True)
    solver(b, u)

    Hs_err = sqrt(abs(b.inner(u-u_exact)))

with mgr.add('solveFracInf', onlyIfFinal=True):
    print('Hs error: {}'.format(Hs_err))
    plt.figure().gca().set_title('Numerical solution, fractional kernel')
    u.plot()

with mgr.add('finiteHorizon'):
    ######################################################################
    # Solve a problem with finite horizon
    kernelConst = kernelFactory('constant', dim=2, horizon=0.2)

with mgr.add('finiteHorizon', onlyIfFinal=True):
    print(kernelConst)
    plt.figure().gca().set_title('Constant kernel')
    kernelConst.plot()

with mgr.add('finiteHorizon'):
    from PyNucleus import DIRICHLET

    meshConst, nIConst = nonlocalMeshFactory('square', kernel=kernelConst, boundaryCondition=DIRICHLET, hTarget=0.18)

with mgr.add('finiteHorizon', onlyIfFinal=True):
    print(meshConst)
    plt.figure().gca().set_title('Mesh for constant kernel')
    meshConst.plot()

with mgr.add('finiteHorizon'):
    dmConst = dofmapFactory('P1', meshConst, nIConst['domain'])
    dmConstInteraction = dmConst.getComplementDoFMap()

    A_const = dmConst.assembleNonlocal(kernelConst, matrixFormat='sparse')
    B_const = dmConst.assembleNonlocal(kernelConst, dm2=dmConstInteraction, matrixFormat='sparse')

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

with mgr.add('finiteHorizon', onlyIfFinal=True):
    print(A_const)

with mgr.add('final'):
    ######################################################################
    plt.show()
