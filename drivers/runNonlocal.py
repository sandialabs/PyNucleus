#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from PyNucleus import driver, DIRICHLET, NEUMANN
from PyNucleus.nl import (nonlocalPoissonProblem,
                          discretizedNonlocalProblem)

##################################################


d = driver()
p = nonlocalPoissonProblem(d)
discrProblem = discretizedNonlocalProblem(d, p)

d.declareFigure('solution')
d.declareFigure('error')
d.declareFigure('analyticSolution')

d.process()

##################################################

naturalDoFMap = discrProblem.dmInterior
bc = discrProblem.bc

mS = discrProblem.modelSolution

u = mS.uInterior
u_full = mS.u

##################################################

vectors = d.addOutputGroup('vectors')
vectors.add('u_full', u_full)
vectors.add('uNatural', mS.uInterior)

meshes = d.addOutputGroup('meshes')
meshes.add('fullMesh', discrProblem.finalMesh)

results = d.addOutputGroup('results')
discrProblem.report(results)
mS.reportSolve(results)

errors = d.addOutputGroup('errors', tested=True)
mS.reportErrors(errors)

if p.analyticSolution is not None:
    uEx = bc.naturalDoFMap.interpolate(p.analyticSolution)
    M_natural = bc.naturalDoFMap.assembleMass()
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

##################################################

plotDefaults = {}
if p.dim == 2:
    plotDefaults['flat'] = True
    if p.element != 'P0':
        plotDefaults['shading'] = 'gouraud'

if d.startPlot('solution'):
    mS.plotSolution()
if mS.error is not None and d.startPlot('error'):
    mS.error.plot(**plotDefaults)

d.finish()
