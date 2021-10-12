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
    from numpy import sqrt

with mgr.add('mesh'):
    ######################################################################
    # Get a mesh and refine it

    from PyNucleus import meshFactory

with mgr.add('mesh', onlyIfFinal=True):
    # show available options
    meshFactory.print()

with mgr.add('mesh'):
    mesh = meshFactory('square', ax=0., ay=0., bx=1., by=1.)
    for _ in range(3):
        mesh = mesh.refine()
with mgr.add('mesh', onlyIfFinal=True):
    print('Mesh:', mesh)
    plt.figure().gca().set_title('Mesh')
    mesh.plot()

with mgr.add('dofmap'):
    ######################################################################
    # Construct a finite element space
    from PyNucleus import dofmapFactory

with mgr.add('dofmap', onlyIfFinal=True):
    # show available options
    dofmapFactory.print()

with mgr.add('dofmap'):
    # We use piecewise linears
    dm = dofmapFactory('P1', mesh)
with mgr.add('dofmap', onlyIfFinal=True):
    print('DoFMap:', dm)

    plt.figure().gca().set_title('DoFMap')
    dm.plot()

with mgr.add('functions'):
    ######################################################################
    # Construct some simple functions
    from PyNucleus import functionFactory

with mgr.add('functions', onlyIfFinal=True):
    # show available options
    functionFactory.print()

with mgr.add('functions'):
    # functions defined via Python lambdas
    rhs_1 = functionFactory('Lambda', lambda x: 2*x[0]*(1-x[0]) + 2*x[1]*(1-x[1]))
    exact_solution_1 = functionFactory('Lambda', lambda x: x[0]*(1-x[0])*x[1]*(1-x[1]))

    # Functions defined via Cython implementations -> faster evaluation
    rhs_2 = functionFactory('rhsFunSin2D')
    exact_solution_2 = functionFactory('solSin2D')

    # assemble right-hand side vectors and interpolate the exact solutions
    b1 = dm.assembleRHS(rhs_1)
    u_interp_1 = dm.interpolate(exact_solution_1)

with mgr.add('functions', onlyIfFinal=True):
    print('Linear system RHS:', b1)
    print('Interpolated solution:', u_interp_1)

with mgr.add('functions'):
    b2 = dm.assembleRHS(rhs_2)
    u_interp_2 = dm.interpolate(exact_solution_2)

with mgr.add('functions', onlyIfFinal=True):
    plt.figure().gca().set_title('Interpolated solution')
    u_interp_1.plot()

with mgr.add('matrices'):
    ######################################################################
    # Assemble mass and Laplacian stiffness matrices
    mass = dm.assembleMass()
    laplacian = dm.assembleStiffness()
with mgr.add('matrices', onlyIfFinal=True):
    print('Linear system matrix:', laplacian)

with mgr.add('solvers'):
    ######################################################################
    # Construct solvers
    from PyNucleus import solverFactory

with mgr.add('solvers', onlyIfFinal=True):
    # show available options
    solverFactory.print()

with mgr.add('solvers'):
    solver_direct = solverFactory('lu', A=laplacian)
    solver_direct.setup()
with mgr.add('solvers', onlyIfFinal=True):
    print('Direct solver:', solver_direct)

with mgr.add('solvers'):
    solver_krylov = solverFactory('cg', A=laplacian)
    solver_krylov.setup()
    solver_krylov.maxIter = 100
    solver_krylov.tolerance = 1e-8
with mgr.add('solvers', onlyIfFinal=True):
    print('Krylov solver:', solver_krylov)

with mgr.add('solvers'):
    u1 = dm.zeros()
    solver_direct(b1, u1)

    u2 = dm.zeros()
    numIter = solver_krylov(b2, u2)
with mgr.add('solvers', onlyIfFinal=True):
    print('Number of iterations:', numIter)

    plt.figure().gca().set_title('Error')
    (u_interp_1-u1).plot(flat=True)

with mgr.add('innerNorm', onlyIfFinal=True):
    ######################################################################
    # Inner products and norms
    print('Residual norm 1st solve: ', (b1-laplacian*u1).norm())
    print('Residual norm 2nd solve: ', (b2-laplacian*u2).norm())

with mgr.add('innerNorm'):
    # Compute errors
    H10_error_1 = sqrt(b1.inner(u_interp_1-u1))
    L2_error_1 = sqrt((u_interp_1-u1).inner(mass*(u_interp_1-u1)))
    H10_error_2 = sqrt(b2.inner(u_interp_2-u2))
    L2_error_2 = sqrt((u_interp_2-u2).inner(mass*(u_interp_2-u2)))

with mgr.add('innerNorm', onlyIfFinal=True):
    print('1st problem - H10:', H10_error_1, 'L2:', L2_error_1)
    print('2nd problem - H10:', H10_error_2, 'L2:', L2_error_2)


with mgr.add('final'):
    plt.show()
