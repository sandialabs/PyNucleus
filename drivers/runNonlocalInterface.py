#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus_base import driver, solverFactory
from PyNucleus_base.solvers import iterative_solver
from PyNucleus_base.ip_norm import ip_serial, norm_serial
from PyNucleus_fem import NO_BOUNDARY, HOMOGENEOUS_NEUMANN
from PyNucleus_fem.functions import indicatorFunctor
from PyNucleus_fem.quadrature import simplexXiaoGimbutas
from PyNucleus_fem.DoFMaps import fe_vector, str2DoFMap
from PyNucleus_fem.splitting import meshSplitter, dofmapSplitter
from PyNucleus_nl.helpers import getFracLapl
from PyNucleus_nl import FRACTIONAL
from PyNucleus_nl.nonlocalProblems import nonlocalInterfaceProblem

d = driver()
nIP = nonlocalInterfaceProblem(d)
d.add('solver', acceptedValues=['lu', 'chol', 'cg', 'alternatingSchwarz-lu', 'RAS-lu', 'alternatingSchwarz-chol', 'RAS-chol'])
d.add('tol', 1e-5)
d.add('maxiter', 100)
d.add('forceRebuild', True)
d.add('genKernel', False)

d.declareFigure('solutions-flat')
d.declareFigure('solutions-3d')
d.declareFigure('errors-flat')
d.declareFigure('errors-3d')
d.declareFigure('rhs-flat', default=False)
d.declareFigure('rhs-3d', default=False)
d.declareFigure('sparsity', default=False)

d.process()

######################################################################

# Global DoFMap used for getting consistent indexing across the two domains
dm = str2DoFMap(d.element)(nIP.mesh, NO_BOUNDARY)

split = meshSplitter(nIP.mesh, {'mesh1': nIP.subdomainIndicator1,
                                'mesh2': nIP.subdomainIndicator2,
                                'interface': nIP.interfaceIndicator})

# submesh for domain 1
domain1Mesh = split.getSubMesh('mesh1')
dm1 = split.getSubMap('mesh1', dm)
R1, P1 = split.getRestrictionProlongation('mesh1', dm, dm1)

# submesh for domain 2
domain2Mesh = split.getSubMesh('mesh2')
dm2 = split.getSubMap('mesh2', dm)
R2, P2 = split.getRestrictionProlongation('mesh2', dm, dm2)

meshInfo = d.addOutputGroup('meshInfo')
meshInfo.add('h_domain1', domain1Mesh.h)
meshInfo.add('h_domain2', domain2Mesh.h)
meshInfo.add('horizon1', nIP.horizon1)
meshInfo.add('horizon2', nIP.horizon2)
meshInfo.add('num_dofs_domain1', dm1.num_dofs)
meshInfo.add('num_dofs_domain2', dm2.num_dofs)
d.logger.info('\n'+str(meshInfo))

# The interface DoFs are discretized by domain 1. Hence, we split dm1
# into interior+interface and boundary. We will also need an interface
# restriction.
dmSplit1 = dofmapSplitter(dm1, {'interface': nIP.interfaceIndicator,
                                'domain': nIP.domainIndicator1+nIP.interfaceIndicator,
                                'bc': nIP.dirichletIndicator1})
R1I, P1I = dmSplit1.getRestrictionProlongation('interface')
R1D, P1D = dmSplit1.getRestrictionProlongation('domain')
R1B, P1B = dmSplit1.getRestrictionProlongation('bc')

dmSplit2 = dofmapSplitter(dm2, {'interface': nIP.interfaceIndicator,
                                'domain': nIP.domainIndicator2+nIP.interfaceIndicator,
                                'bc': nIP.dirichletIndicator2})
R2I, P2I = dmSplit2.getRestrictionProlongation('interface')
R2D, P2D = dmSplit2.getRestrictionProlongation('domain')
R2B, P2B = dmSplit2.getRestrictionProlongation('bc')

np.testing.assert_equal(P1D.num_columns+P1B.num_columns, P1D.num_rows)
np.testing.assert_equal(P2D.num_columns+P2B.num_columns, P2D.num_rows)
np.testing.assert_allclose((P1*P1D*np.ones((P1D.num_columns))+P2*P2D*np.ones((P2D.num_columns))).min(), 0.)
np.testing.assert_allclose((P1*P1D*np.ones((P1D.num_columns))+P2*P2D*np.ones((P2D.num_columns))).max(), 2.)
np.testing.assert_equal(P1I.num_columns, P2I.num_columns)


with d.timer('assemble matrices'):
    buildDense = True

    A1 = getFracLapl(dm1, nIP.kernel1, boundaryCondition=HOMOGENEOUS_NEUMANN, dense=buildDense,
                     forceRebuild=d.forceRebuild, genKernel=d.genKernel, trySparsification=True, doSave=True)
    A2 = getFracLapl(dm2, nIP.kernel2, boundaryCondition=HOMOGENEOUS_NEUMANN, dense=buildDense,
                     forceRebuild=d.forceRebuild, genKernel=d.genKernel, trySparsification=True, doSave=True)

# domain-domain interaction
A = (P1*P1D*(R1D*A1*P1D)*R1D*R1) + \
    (P2*P2D*(R2D*A2*P2D)*R2D*R2)
# Fake Dirichlet condition. We really only want to solve on interior
# and interface unknowns. We make the boundary unknowns of the global
# problem an identity block and set the rhs to zero for these.
A += (P1*P1B*R1B*R1) + \
    (P2*P2B*R2B*R2)

f = indicatorFunctor(nIP.forcing_left, nIP.localSubdomainIndicator1) + \
    indicatorFunctor(nIP.forcing_right, nIP.localSubdomainIndicator2) + \
    indicatorFunctor(nIP.mult*nIP.flux_jump, nIP.interfaceIndicator)

dmSplitRHS = dofmapSplitter(dm, {'domain':
                                 nIP.localSubdomainIndicator1 +
                                 nIP.localSubdomainIndicator2 +
                                 nIP.localInterfaceIndicator})
dmRHS = dmSplitRHS.getSubMap('domain')
R_RHS, P_RHS = dmSplitRHS.getRestrictionProlongation('domain')
with d.timer('assemble rhs'):
    if (nIP.kernel1.kernelType == FRACTIONAL) or (nIP.kernel2.kernelType == FRACTIONAL):
        if nIP.mesh.dim == 1:
            b = P_RHS*dmRHS.assembleRHS(f, qr=simplexXiaoGimbutas(80, nIP.mesh.dim))
        else:
            b = P_RHS*dmRHS.assembleRHS(f, qr=simplexXiaoGimbutas(30, nIP.mesh.dim))
    else:
        b = P_RHS*dmRHS.assembleRHS(f, qr=simplexXiaoGimbutas(3, nIP.mesh.dim))

# solution jump
h = dmSplit2.getSubMap('interface').interpolate(nIP.sol_jump)
b -= (P2*P2D*(R2D*A2*P2I))*h
# Dirichlet BCs
g1 = dmSplit1.getSubMap('bc').interpolate(nIP.diri_left)
g2 = dmSplit2.getSubMap('bc').interpolate(nIP.diri_right)
b -= P1*P1D*(R1D*A1*P1B)*g1
b -= P2*P2D*(R2D*A2*P2B)*g2
if d.startPlot('sparsity', ratio=1.):
    assert d.mesh.dim == 1
    import matplotlib.pyplot as plt
    Ad = ((P1*P1D*(R1D*A1*P1D)*R1D*R1)+(P1*P1B*R1B*R1)).toarray()
    Ad[np.absolute(Ad) < 1e-10] = 0.
    Ad[np.absolute(Ad) > 1e-10] = 1.

    Ad2 = ((P2*P2D*(R2D*A2*P2D)*R2D*R2)+(P2*P2B*R2B*R2)).toarray()
    Ad2[np.absolute(Ad2) < 1e-10] = 0.
    Ad2[np.absolute(Ad2) > 1e-10] = 2.
    Ad += Ad2
    Ad[np.absolute(Ad) < 1e-10] = np.nan

    x = dm.getDoFCoordinates()[:, 0]
    X, Y = np.meshgrid(x, x)
    plt.pcolormesh(X, Y, Ad, shading='nearest')
    plt.axis('equal')
    plt.xlim([-d.horizon1-d.mesh.h/2, 2+d.horizon2+d.mesh.h/2])
    plt.ylim([-d.horizon1-d.mesh.h/2, 2+d.horizon2+d.mesh.h/2])
    plt.xlabel(r'$y$')
    plt.ylabel(r'$x$')
    if d.horizon1 > 0:
        plt.axhline(1.+d.horizon1, color='g')
        plt.axvline(1.+d.horizon1, color='g')
    plt.axhline(1., color='r')
    plt.axvline(1., color='r')
    plt.axhline(1-d.horizon2, color='b')
    plt.axvline(1-d.horizon2, color='b')


u = dm.zeros()
with d.timer('solve'):
    if d.solver in ('lu', 'chol', 'cg'):
        solver = solverFactory.build(d.solver, A=A, setup=True)
        if isinstance(solver, iterative_solver):
            solver.maxIter = d.maxiter
            solver.tolerance = d.tol
            r = dm.zeros()
            A.residual_py(u, b, r)
            norm = norm_serial()
            residualNorm0 = norm(r)
        its = solver(b, u)
        if isinstance(solver, iterative_solver):
            A.residual_py(u, b, r)
            residualNorm = norm(r)
            d.logger.info('{} solver obtained residual norm {}/{} = {} after {} iterations'.format(d.solver, residualNorm, residualNorm0,
                                                                                                   residualNorm/residualNorm0, its))
    elif (d.solver.find('alternatingSchwarz') >= 0) or (d.solver.find('RAS') >= 0):
        if d.solver.find('chol'):
            subsolver = 'chol'
        else:
            subsolver = 'lu'
        a1inv = solverFactory.build(subsolver, A=R1*A*P1, setup=True)
        a2inv = solverFactory.build(subsolver, A=R2*A*P2, setup=True)
        u1 = dm1.zeros()
        u2 = dm2.zeros()
        r = dm.zeros()
        A.residual_py(u, b, r)
        norm = norm_serial()
        k = 0
        residualNorm0 = residualNorm = norm(r)
        if d.solver.find('alternatingSchwarz') >= 0:
            while k < d.maxiter and residualNorm/residualNorm0 > d.tol:
                b1 = R1*r
                a1inv(b1, u1)
                u += P1*u1
                A.residual_py(u, b, r)

                b2 = R2*r
                a2inv(b2, u2)
                u += P2*u2
                A.residual_py(u, b, r)

                residualNorm = norm(r)
                k += 1
            d.logger.info('Alternating Schwarz solver obtained residual norm {}/{} = {} after {} iterations'.format(residualNorm, residualNorm0,
                                                                                                                    residualNorm/residualNorm0, k))
        else:
            u1.assign(1.)
            u2.assign(1.)
            dg = P1*u1+P2*u2
            d1inv = fe_vector(1./(R1*dg), dm1)
            d2inv = fe_vector(1./(R2*dg), dm2)
            while k < d.maxiter and residualNorm/residualNorm0 > d.tol:
                b1 = R1*r
                a1inv(b1, u1)
                u += P1*(u1*d1inv)

                b2 = R2*r
                a2inv(b2, u2)
                u += P2*(u2*d2inv)

                A.residual_py(u, b, r)
                residualNorm = norm(r)
                k += 1
            d.logger.info('RAS solver obtained residual norm {}/{} = {} after {} iterations'.format(residualNorm, residualNorm0, residualNorm/residualNorm0, k))

    else:
        raise NotImplementedError(d.solver)

u1 = dm1.zeros()
u1.assign(R1*u + P1B*g1)

u2 = dm2.zeros()
u2.assign(R2*u + P2I*h + P2B*g2)

results = d.addOutputGroup('results', tested=True)
if nIP.sol_1 is not None and nIP.sol_2 is not None:
    M1 = dm1.assembleMass()
    M2 = dm2.assembleMass()
    u1ex = dm1.interpolate(nIP.sol_1)
    u2ex = dm2.interpolate(nIP.sol_2)

    inner = ip_serial()

    results.add('domain1L2err', np.sqrt(inner(M1*(u1-u1ex), u1-u1ex)), rTol=1e-2)
    results.add('domain2L2err', np.sqrt(inner(M2*(u2-u2ex), u2-u2ex)), rTol=1e-2)
d.logger.info('\n'+str(results))

data = d.addOutputGroup('data', tested=False)
data.add('fullDomain1Mesh', u1.dm.mesh)
data.add('fullDomain1DoFMap', u1.dm)
data.add('full_u1', u1)
data.add('fullDomain2Mesh', u2.dm.mesh)
data.add('fullDomain2DoFMap', u2.dm)
data.add('full_u2', u2)


if d.startPlot('solutions-flat'):
    import matplotlib.pyplot as plt
    plotKwargs = {}
    if nIP.mesh.dim == 2:
        vmin = min(u1.min(), u2.min())
        vmax = max(u1.max(), u2.max())

        plotKwargs['vmin'] = vmin
        plotKwargs['vmax'] = vmax
        plotKwargs['flat'] = True
    if nIP.mesh.dim == 2:
        plt.subplot(1, 2, 1)
    u1.plot(**plotKwargs)
    plt.xlabel(r'$\mathbf{x}$')
    if nIP.mesh.dim == 2:
        plt.subplot(1, 2, 2)
    u2.plot(**plotKwargs)
    plt.xlabel(r'$\mathbf{x}$')

if d.startPlot('rhs-3d'):
    dm.project(f, qr=simplexXiaoGimbutas(3, nIP.mesh.dim)).plot()

if nIP.mesh.dim == 2 and d.startPlot('rhs-flat'):
    dm.project(f, qr=simplexXiaoGimbutas(3, nIP.mesh.dim)).plot(flat=True)

if nIP.mesh.dim == 2 and d.startPlot('solutions-3d'):
    vmin = min(u1.min(), u2.min())
    vmax = max(u1.max(), u2.max())
    plotKwargs = {}
    plotKwargs['vmin'] = vmin
    plotKwargs['vmax'] = vmax
    ax = u1.plot(**plotKwargs)
    plotKwargs['ax'] = ax
    ax = u2.plot(**plotKwargs)

if nIP.sol_1 is not None and nIP.sol_2 is not None:
    if d.startPlot('errors-flat'):
        import matplotlib.pyplot as plt
        plotKwargs = {}
        if nIP.mesh.dim == 2:
            plotKwargs['flat'] = True
        if nIP.mesh.dim == 2:
            plt.subplot(1, 2, 1)
        (u1-u1ex).plot(**plotKwargs)
        plt.xlabel(r'$\mathbf{x}$')
        if nIP.mesh.dim == 2:
            plt.subplot(1, 2, 2)
        (u2-u2ex).plot(**plotKwargs)
        plt.xlabel(r'$\mathbf{x}$')
    if nIP.mesh.dim == 2 and d.startPlot('errors-3d'):
        plotKwargs = {}
        ax = (u1-u1ex).plot(**plotKwargs)
        plotKwargs['ax'] = ax
        ax = (u2-u2ex).plot(**plotKwargs)
d.finish()
