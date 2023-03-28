#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from PyNucleus.base import REAL, driver, solverFactory
from PyNucleus.base.ip_norm import norm_serial
from PyNucleus.fem import (simpleInterval, uniformSquare,
                           squareIndicator, P1_DoFMap,
                           constant, Lambda, NO_BOUNDARY, INTERIOR, PHYSICAL,
                           functionFactory,
                           getSurfaceDoFMap)
from PyNucleus.fem.DoFMaps import fe_vector
from PyNucleus.fem.splitting import meshSplitter, dofmapSplitter


d = driver()
d.add('domain', acceptedValues=['doubleInterval', 'doubleSquare'])
d.add('problem', acceptedValues=['polynomial', 'sin', 'sin-solJump-fluxJump', 'sin-nojump', 'sin1d-solJump-fluxJump'])
d.add('coeff1', 1.0)
d.add('coeff2', 1.0)
d.add('hTarget', 0.05)
d.add('solver', acceptedValues=['lu', 'alternatingSchwarz', 'RAS'])

d.declareFigure('solutions-flat')
d.declareFigure('solutions-3d')
d.declareFigure('errors')

params = d.process()

L2ex_left = None
L2ex_right = None
H10ex_left = None
H10ex_right = None
if d.domain == 'doubleInterval':
    a, b, c = 0, 2, 1
    mesh = simpleInterval(a, b)
    mesh = mesh.refine()

    eps = 1e-9
    domainIndicator1 = squareIndicator(np.array([a+eps], dtype=REAL),
                                       np.array([c-eps], dtype=REAL))
    domainIndicator2 = squareIndicator(np.array([c+eps], dtype=REAL),
                                       np.array([b-eps], dtype=REAL))
    interfaceIndicator = squareIndicator(np.array([c-eps], dtype=REAL),
                                         np.array([c+eps], dtype=REAL))
    dirichletIndicator1 = constant(1.)-domainIndicator1-interfaceIndicator
    dirichletIndicator2 = constant(1.)-domainIndicator2-interfaceIndicator

    if d.problem == 'polynomial':
        sol_1 = Lambda(lambda x: x[0]**2)
        sol_2 = Lambda(lambda x: (x[0]-1)**2)
        diri_left = sol_1
        diri_right = sol_2
        forcing_left = constant(-2*d.coeff1)
        forcing_right = constant(-2*d.coeff2)
        sol_jump = sol_2-sol_1
        flux_jump = constant(2*d.coeff1)
    elif d.problem == 'sin-solJump-fluxJump':
        sin = functionFactory('sin1d')
        one = functionFactory('constant', 1)
        sol_1 = sin
        sol_2 = one-2*sin
        diri_left = sol_1
        diri_right = sol_2
        forcing_left = np.pi**2 * d.coeff1 * sin
        forcing_right = -2*np.pi**2 * d.coeff2 * sin
        sol_jump = sol_2-sol_1
        flux_jump = functionFactory('constant', -np.pi*d.coeff1 - 2*np.pi*d.coeff2)
        L2ex_left = 0.5
        L2ex_right = 3.+8/np.pi
        H10ex_left = np.pi**2 * d.coeff1 * 0.5
        H10ex_right = np.pi**2 * d.coeff2 * (2.0 + 4/np.pi)
    elif d.problem == 'sin-nojump':
        sol_1 = Lambda(lambda x: np.sin(np.pi*x[0])/d.coeff1)
        sol_2 = Lambda(lambda x: np.sin(np.pi*x[0])/d.coeff2)
        diri_left = sol_1
        diri_right = sol_2
        forcing_left = Lambda(lambda x: np.pi**2*np.sin(np.pi*x[0]))
        forcing_right = Lambda(lambda x: np.pi**2*np.sin(np.pi*(x[0])))
        sol_jump = sol_2-sol_1
        flux_jump = constant(0)
    elif d.problem == 'sin-soljump':
        sol_1 = Lambda(lambda x: np.sin(np.pi*x[0]))
        sol_2 = Lambda(lambda x: 1.+np.sin(np.pi*(x[0]-1)))
        diri_left = sol_1
        diri_right = sol_2
        forcing_left = Lambda(lambda x: np.pi**2*np.sin(np.pi*x[0])*d.coeff1)
        forcing_right = Lambda(lambda x: np.pi**2*np.sin(np.pi*(x[0]-1))*d.coeff2)
        sol_jump = sol_2-sol_1
        flux_jump = constant(-np.pi*d.coeff1 - np.pi*d.coeff2)
    else:
        raise NotImplementedError(d.problem)

elif d.domain == 'doubleSquare':
    ax = 0
    ay = 0
    bx = 2
    by = 1
    cx = 1
    mesh = uniformSquare(2, 2, ax, ay, bx, by)
    mesh = mesh.refine()

    eps = 1e-9
    domainIndicator1 = squareIndicator(np.array([ax+eps, ay+eps], dtype=REAL),
                                       np.array([cx-eps, by-eps], dtype=REAL))
    domainIndicator2 = squareIndicator(np.array([cx+eps, ay+eps], dtype=REAL),
                                       np.array([bx-eps, by-eps], dtype=REAL))
    interfaceIndicator = squareIndicator(np.array([cx-eps, ay+eps], dtype=REAL),
                                         np.array([cx+eps, by-eps], dtype=REAL))
    dirichletIndicator1 = constant(1.)-domainIndicator1-interfaceIndicator
    dirichletIndicator2 = constant(1.)-domainIndicator2-interfaceIndicator

    if d.problem == 'polynomial':
        sol_1 = Lambda(lambda x: x[0]**2)
        sol_2 = Lambda(lambda x: (x[0]-1)**2)
        diri_left = sol_1
        diri_right = sol_2
        forcing_left = constant(-2*d.coeff1)
        forcing_right = constant(-2*d.coeff2)
        sol_jump = sol_2-sol_1
        flux_jump = constant(2*d.coeff1)
    elif d.problem == 'sin':
        sol_1 = Lambda(lambda x: np.sin(np.pi*x[0]))
        sol_2 = Lambda(lambda x: np.sin(np.pi*(x[0]-1)))
        diri_left = sol_1
        diri_right = sol_2
        forcing_left = Lambda(lambda x: np.pi**2*np.sin(np.pi*x[0])*d.coeff1)
        forcing_right = Lambda(lambda x: np.pi**2*np.sin(np.pi*(x[0]-1))*d.coeff2)
        sol_jump = sol_2-sol_1
        flux_jump = constant(-np.pi*d.coeff1 - np.pi*d.coeff2)
    elif d.problem == 'sin1d-solJump-fluxJump':
        # the local problem has a know exact solution
        sin = functionFactory('sin1d')
        one = functionFactory('constant', 1)
        sol_1 = sin
        sol_2 = one-2*sin
        diri_left = sol_1
        diri_right = sol_2
        forcing_left = d.coeff1 * np.pi**2 * sin
        forcing_right = -2*d.coeff2 * np.pi**2 * sin
        sol_jump = one
        flux_jump = constant(-np.pi*d.coeff1 - 2*np.pi*d.coeff2)
        L2ex_left = 0.5
        L2ex_right = 3.+8/np.pi
        H10ex_left = np.pi**2 * d.coeff1 * 0.5
        H10ex_right = np.pi**2 * d.coeff2 * (2.0 + 4/np.pi)
    elif params['problem'] == 'sin-solJump-fluxJump':
        # the local problem has a know exact solution
        sin2d = functionFactory('Lambda', lambda x: np.sin(np.pi*x[0])*np.sin(2*np.pi*x[1]))
        sin = functionFactory('sin2d')
        one = functionFactory('constant', 1)
        sol_1 = 2*one+2*sin2d
        sol_2 = one-sin
        diri_left = sol_1
        diri_right = sol_2
        forcing_left = d.coeff1 * 2*5*np.pi**2 * sin2d
        forcing_right = -d.coeff2 * 2*np.pi**2 * sin
        sol_jump = -one
        flux_jump = (-2*np.pi*d.coeff1 * functionFactory('Lambda', lambda x: np.sin(2*np.pi*x[1]))
                     - np.pi*d.coeff2 * functionFactory('Lambda', lambda x: np.sin(np.pi*x[1])))
        L2ex_left = 5.
        L2ex_right = 1.25 + 8./np.pi**2
        H10ex_left = np.pi**2 * d.coeff1 * 5
        H10ex_right = np.pi**2 * d.coeff2 * 0.5
    else:
        raise NotImplementedError(d.problem)
else:
    raise NotImplementedError(d.domain)

######################################################################

while mesh.h > params['hTarget']:
    mesh = mesh.refine()

# Global DoFMap used for getting consistent indexing across the two domains
dm = P1_DoFMap(mesh, NO_BOUNDARY)

split = meshSplitter(mesh, {'mesh1': domainIndicator1,
                            'mesh2': domainIndicator2})

# submesh for domain 1
domain1Mesh = split.getSubMesh('mesh1')
domain1Mesh.tagBoundaryVertices(lambda x: INTERIOR if interfaceIndicator(x) > 0.5 else PHYSICAL)
domain1Mesh.tagBoundaryEdges(lambda x, y: INTERIOR if (interfaceIndicator(x) > 0.5 and interfaceIndicator(y) > 0.5) else PHYSICAL)
dm1 = split.getSubMap('mesh1', dm)
R1, P1 = split.getRestrictionProlongation('mesh1', dm, dm1)

# Surface mesh used for assembling the flux jump condition
interface = domain1Mesh.get_surface_mesh(INTERIOR)
dmInterface = getSurfaceDoFMap(mesh, interface, dm1)

# submesh for domain 2
domain2Mesh = split.getSubMesh('mesh2')
dm2 = split.getSubMap('mesh2', dm)
R2, P2 = split.getRestrictionProlongation('mesh2', dm, dm2)

meshInfo = d.addOutputGroup('meshInfo')
meshInfo.add('h_domain1', domain1Mesh.h)
meshInfo.add('h_domain2', domain2Mesh.h)
meshInfo.add('num_dofs_domain1', dm1.num_dofs)
meshInfo.add('num_dofs_domain2', dm2.num_dofs)
d.logger.info('\n'+str(meshInfo))

# The interface DoFs are discretized by domain 1. Hence, we split dm1
# into interior+interface and boundary. We will also need an interface
# restriction.
dmSplit1 = dofmapSplitter(dm1, {'interface': interfaceIndicator,
                                'domain': domainIndicator1+interfaceIndicator,
                                'bc': dirichletIndicator1})
R1I, P1I = dmSplit1.getRestrictionProlongation('interface')
R1D, P1D = dmSplit1.getRestrictionProlongation('domain')
R1B, P1B = dmSplit1.getRestrictionProlongation('bc')

dmSplit2 = dofmapSplitter(dm2, {'interface': interfaceIndicator,
                                'domain': domainIndicator2+interfaceIndicator,
                                'bc': dirichletIndicator2})
R2I, P2I = dmSplit2.getRestrictionProlongation('interface')
R2D, P2D = dmSplit2.getRestrictionProlongation('domain')
R2B, P2B = dmSplit2.getRestrictionProlongation('bc')

# np.testing.assert_equal(P1D.num_columns+P1B.num_columns, P1D.num_rows)
# np.testing.assert_equal(P2D.num_columns+P2B.num_columns+P2I.num_columns, P2D.num_rows)
# np.testing.assert_allclose((P1*P1D*np.ones((P1D.num_columns))+P2*P2D*np.ones((P2D.num_columns))).max(), 1.)
# np.testing.assert_equal(P1I.num_columns, P2I.num_columns)


A1 = dm1.assembleStiffness()
A1.scale(d.coeff1)
A2 = dm2.assembleStiffness()
A2.scale(d.coeff2)

# domain-domain interaction
A = (P1*P1D*(R1D*A1*P1D)*R1D*R1) + \
    (P2*P2D*(R2D*A2*P2D)*R2D*R2)
# Fake Dirichlet condition. We really only want to solve on interior
# and interface unknowns. We make the boundary unknowns of the global
# problem an identity block and set the rhs to zero for these.
A += (P1*P1B*R1B*R1) + \
    (P2*P2B*R2B*R2)

# forcing
b = P1*P1D*dmSplit1.getSubMap('domain').assembleRHS(forcing_left) + \
    P2*P2D*dmSplit2.getSubMap('domain').assembleRHS(forcing_right)
# flux jump forcing term
b += P1*dmInterface.assembleRHS(flux_jump)
# solution jump
h = dmSplit2.getSubMap('interface').interpolate(sol_jump)
b -= (P2*P2D*(R2D*A2*P2I))*h
# Dirichlet BCs
g1 = dmSplit1.getSubMap('bc').interpolate(diri_left)
g2 = dmSplit2.getSubMap('bc').interpolate(diri_right)
b -= P1*P1D*(R1D*A1*P1B)*g1
b -= P2*P2D*(R2D*A2*P2B)*g2

u = dm.zeros()
with d.timer('solve'):
    if d.solver == 'lu':
        lu = solverFactory.build('lu', A=A, setup=True)
        lu(b, u)
    elif (d.solver == 'alternatingSchwarz') or (d.solver == 'RAS'):
        a1inv = solverFactory.build('lu', A=R1*A*P1, setup=True)
        a2inv = solverFactory.build('lu', A=R2*A*P2, setup=True)
        u1 = dm1.zeros()
        u2 = dm2.zeros()
        r = dm.zeros()
        A.residual_py(u, b, r)
        norm = norm_serial()
        k = 0
        residualNorm0 = residualNorm = norm(r)
        if d.solver == 'alternatingSchwarz':
            while k < 100 and residualNorm/residualNorm0 > 1e-5:
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
            while k < 100 and residualNorm/residualNorm0 > 1e-5:
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

M1 = dm1.assembleMass()
M2 = dm2.assembleMass()
u1ex = dm1.interpolate(sol_1)
u2ex = dm2.interpolate(sol_2)

results = d.addOutputGroup('results')
if L2ex_left is not None:
    z1 = dm1.assembleRHS(sol_1)
    results.add('domain1L2err', np.sqrt(abs(u1.inner(M1*u1)-2*z1.inner(u1)+L2ex_left)), rTol=1e-2)
if L2ex_right is not None:
    z2 = dm2.assembleRHS(sol_2)
    results.add('domain2L2err', np.sqrt(abs(u2.inner(M2*u2)-2*z2.inner(u2)+L2ex_right)), rTol=1e-2)
if H10ex_left is not None:
    b1 = dm1.assembleRHS(forcing_left)
    results.add('domain1H1err', np.sqrt(abs(H10ex_left-b1.inner(u1))), rTol=1e-2)
if H10ex_right is not None:
    b2 = dm2.assembleRHS(forcing_right)
    results.add('domain2H1err', np.sqrt(abs(H10ex_right-b2.inner(u2))), rTol=1e-2)
d.logger.info('\n'+str(results))

data = d.addOutputGroup('data', tested=False)
data.add('fullDomain1Mesh', u1.dm.mesh)
data.add('fullDomain1DoFMap', u1.dm)
data.add('full_u1', u1)
data.add('fullDomain2Mesh', u2.dm.mesh)
data.add('fullDomain2DoFMap', u2.dm)
data.add('full_u2', u2)

if d.startPlot('solutions-flat'):
    if mesh.dim == 1:
        u1.plot()
        u2.plot()
    else:
        vmin = min(u1.min(), u2.min())
        vmax = max(u1.max(), u2.max())
        plotKwargs = {}
        plotKwargs['vmin'] = vmin
        plotKwargs['vmax'] = vmax
        plotKwargs['flat'] = True
        u1.plot(**plotKwargs)
        u2.plot(**plotKwargs)

if mesh.dim == 2 and dm.num_dofs < 60000 and d.startPlot('solutions-3d'):
    vmin = min(u1.min(), u2.min())
    vmax = max(u1.max(), u2.max())
    plotKwargs = {}
    if mesh.dim == 2:
        plotKwargs['vmin'] = vmin
        plotKwargs['vmax'] = vmax
    ax = u1.plot(**plotKwargs)
    if mesh.dim == 2:
        plotKwargs['ax'] = ax
    ax = u2.plot(**plotKwargs)

if d.startPlot('errors'):
    plotKwargs = {}
    if dm.num_dofs >= 60000:
        plotKwargs['flat'] = True
    ax = (u1-u1ex).plot(**plotKwargs)
    if mesh.dim == 2:
        plotKwargs['ax'] = ax
    ax = (u2-u2ex).plot(**plotKwargs)
d.finish()
