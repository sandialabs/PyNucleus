#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from PyNucleus.base import driver, solverFactory, INDEX
from PyNucleus.base.linear_operators import Dense_LinearOperator
from PyNucleus.fem import (simplexXiaoGimbutas, plotManager,
                           P0_DoFMap, NO_BOUNDARY)
from PyNucleus.multilevelSolver import hierarchyManager
from PyNucleus.nl import (multilevelDirichletCondition,
                          paramsForFractionalHierarchy,
                          nonlocalProblem)
from PyNucleus.nl.nonlocalProblems import (DIRICHLET, HOMOGENEOUS_DIRICHLET,
                                           NEUMANN, HOMOGENEOUS_NEUMANN)

d = driver()
p = nonlocalProblem(d)

d.add('solver', acceptedValues=['lu', 'mg', 'cg-mg'])
d.add('dense', False)
d.add('forceRebuild', True)
d.add('genKernel', False)
d.add('maxiter', 100)
d.add('tol', 1e-6)

d.add('plotRHS', False)
d.add('plotOnFullDomain', True)

d.declareFigure('solution')
d.declareFigure('analyticSolution')

params = d.process()

if d.kernel != 'fractional':
    # Hierarchical matrices are only implemented for fractional kernels
    d.dense = True

with d.timer('hierarchy'):
    params['domain'] = p.mesh
    params['keepMeshes'] = 'all'
    params['keepAllDoFMaps'] = True
    params['assemble'] = 'ALL' if params['solver'].find('mg') >= 0 else 'last'
    params['dense'] = d.dense
    params['logging'] = True
    params['genKernel'] = d.genKernel
    hierarchies, connectors = paramsForFractionalHierarchy(p.noRef, params)
    hM = hierarchyManager(hierarchies, connectors, params)
    hM.setup()
mesh = hM['fine'].meshLevels[-1].mesh
assert 2*mesh.h < p.horizon.value, "h = {}, horizon = {}".format(mesh.h, p.horizon.value)

if not p.boundaryCondition == HOMOGENEOUS_DIRICHLET:
    bc = multilevelDirichletCondition(hM.getLevelList(), p.domainIndicator, p.fluxIndicator)
    fullDoFMap = bc.fullDoFMap
    naturalDoFMap = bc.naturalDoFMap
    b = naturalDoFMap.assembleRHS(p.rhs, qr=simplexXiaoGimbutas(3, mesh.dim))
    bc.setDirichletData(p.dirichletData)
    bc.applyRHScorrection(b)
    hierarchy = bc.naturalLevels
else:
    hierarchy = hM.getLevelList()
    naturalDoFMap = hierarchy[-1]['DoFMap']
    b = naturalDoFMap.assembleRHS(p.rhs, qr=simplexXiaoGimbutas(3, mesh.dim))


# pure Neumann condition -> project out nullspace
if p.boundaryCondition in (NEUMANN, HOMOGENEOUS_NEUMANN):
    assert bc.dirichletDoFMap.num_dofs == 0, bc.dirichletDoFMap
    if params['solver'].find('mg') >= 0:
        bc.naturalLevels[0]['A'] = bc.naturalLevels[0]['A'] + Dense_LinearOperator.ones(*bc.naturalLevels[0]['A'].shape)
    const = bc.naturalDoFMap.ones()
    b -= b.inner(const)/const.inner(const)*const

u = naturalDoFMap.zeros()

with d.timer('solve'):
    if params['solver'].find('mg') >= 0:
        ml = solverFactory.build('mg', hierarchy=hierarchy, setup=True, tolerance=params['tol'], maxIter=params['maxiter'])
        d.logger.info('\n'+str(ml))
    if d.solver == 'mg':
        its = ml(b, u)
        res = ml.residuals
    elif d.solver == 'cg-mg':
        cg = solverFactory.build('cg', A=hierarchy[-1]['A'], setup=True, tolerance=params['tol'], maxIter=params['maxiter'])
        cg.setPreconditioner(ml.asPreconditioner())
        its = cg(b, u)
        res = cg.residuals
    elif d.solver == 'lu':
        lu = solverFactory.build(d.solver, A=hierarchy[-1]['A'], setup=True)
        its = lu(b, u)
    else:
        raise NotImplementedError(d.solver)

# pure Neumann condition -> add nullspace components to match analytic solution
if p.boundaryCondition in (NEUMANN, HOMOGENEOUS_NEUMANN) and p.analyticSolution is not None:
    uEx = bc.naturalDoFMap.interpolate(p.analyticSolution)
    u += (const.inner(uEx)-const.inner(u))/const.inner(const) * const

vectors = d.addOutputGroup('vectors')
vectors.add('u', u)

meshes = d.addOutputGroup('meshes')
meshes.add('fullMesh', mesh)

results = d.addOutputGroup('results')
results.add('full h', mesh.h)
results.add('natural DoFs', naturalDoFMap.num_dofs)
results.add('iterations', its)

if p.boundaryCondition in (DIRICHLET, ):
    results.add('full DoFs', bc.fullDoFMap.num_dofs)
    u_full = bc.augmentDirichlet(u)
    vectors.add('u_full', u_full)
else:
    u_full = bc.naturalP*u

errors = d.addOutputGroup('errors', tested=True)
resNorm = (b-hierarchy[-1]['A']*u).norm(False)
errors.add('residual norm', resNorm)

if p.analyticSolution is not None:
    uEx = bc.naturalDoFMap.interpolate(p.analyticSolution)
    M_natural = naturalDoFMap.assembleMass()
    L2err_natural = np.sqrt(abs((u-uEx).inner(M_natural*(u-uEx))))
    relL2err_natural = L2err_natural/np.sqrt(abs(uEx.inner(M_natural*uEx)))

    uEx_domain = bc.domainDoFMap.interpolate(p.analyticSolution)
    M_domain = bc.domainDoFMap.assembleMass()
    u_domain = bc.domainDoFMap.fromArray(bc.domainR*u_full)
    L2err_domain = np.sqrt(abs((u_domain-uEx_domain).inner(M_domain*(u_domain-uEx_domain))))
    relL2err_domain = L2err_domain/np.sqrt(abs(uEx_domain.inner(M_domain*uEx_domain)))

    Linferr_natural = np.abs((u-uEx)).max()
    relLinferr_natural = Linferr_natural/np.abs(uEx).max()
    vectors.add('uEx', uEx)
    errors.add('L2 error natural', L2err_natural, rTol=3e-2)
    errors.add('rel L2 error natural', relL2err_natural, rTol=3e-2)
    errors.add('L2 error domain', L2err_domain, rTol=3e-2)
    errors.add('rel L2 error domain', relL2err_domain, rTol=3e-2)
    errors.add('Linf error natural', Linferr_natural, rTol=3e-2)
    errors.add('rel Linf error natural', relLinferr_natural, rTol=3e-2)

    if p.boundaryCondition in (DIRICHLET, NEUMANN):
        uEx_full = bc.fullDoFMap.interpolate(p.analyticSolution)
        M_full = bc.fullDoFMap.assembleMass()
        L2err_full = np.sqrt(abs((uEx_full-u_full).inner(M_full*(uEx_full-u_full))))
        vectors.add('uEx_full', uEx_full)
        errors.add('L2 error including Dirichlet domain', L2err_full, rTol=3e-2)
d.logger.info('\n'+str(results+errors))

if d.startPlot('solution'):
    import matplotlib.pyplot as plt

    plotDefaults = {}
    if p.dim == 2:
        plotDefaults['flat'] = True
    if p.element != 'P0':
        plotDefaults['shading'] = 'gouraud'
    if p.boundaryCondition in (DIRICHLET, NEUMANN):
        pM = plotManager(bc.fullDoFMap.mesh, bc.fullDoFMap, defaults=plotDefaults)
        if p.dim == 1:
            pMerr = plotManager(bc.fullDoFMap.mesh, bc.fullDoFMap, defaults=plotDefaults)
        else:
            pMerr = pM
        pM.add(u_full, label='solution')
        if d.plotRHS:
            pM.add(bc.augmentDirichlet(b), label='rhs')
        if p.analyticSolution is not None:
            pM.add(uEx_full, label='analytic solution')
            pMerr.add(u_full-uEx_full, label='error')
    else:
        if d.plotOnFullDomain:
            pM = plotManager(naturalDoFMap.mesh, naturalDoFMap, defaults=plotDefaults)
            if p.dim == 1:
                pMerr = plotManager(naturalDoFMap.mesh, naturalDoFMap, defaults=plotDefaults)
            else:
                pMerr = pM
        else:
            indicator = P0_DoFMap(naturalDoFMap.mesh, NO_BOUNDARY).interpolate(p.domainIndicator)
            selectedCells = np.flatnonzero(indicator.toarray() >= 1e-9).astype(INDEX)
            reducedDM = naturalDoFMap.getReducedMeshDoFMap(selectedCells)
            pM = plotManager(reducedDM.mesh, reducedDM, defaults=plotDefaults)
            if p.dim == 1:
                pMerr = plotManager(reducedDM.mesh, reducedDM, defaults=plotDefaults)
            else:
                pMerr = pM
        pM.add(u, label='solution')
        if d.plotRHS:
            pM.add(b, label='rhs')
        if p.analyticSolution is not None:
            pM.add(uEx, label='analytic solution')
            if p.dim == 1:
                pMerr.add(u-uEx, label='error')
    if p.dim == 1 and p.analyticSolution is not None:
        plt.subplot(1, 2, 1)
        pM.plot()
        plt.subplot(1, 2, 2)
        pMerr.plot()
    else:
        pM.plot()
d.finish()
