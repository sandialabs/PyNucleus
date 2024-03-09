#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from mpi4py import MPI
from PyNucleus import driver
from PyNucleus_nl import (nonlocalPoissonProblem,
                          discretizedNonlocalProblem)

##################################################

description = """Solves a nonlocal Poisson problem with finite horizon."""

d = driver(MPI.COMM_WORLD, description=description)
p = nonlocalPoissonProblem(d)
discrProblem = discretizedNonlocalProblem(d, p)

d.declareFigure('solution')
d.declareFigure('error')
d.declareFigure('analyticSolution')

d.process()

##################################################

mS = discrProblem.modelSolution

##################################################

vectors = d.addOutputGroup('vectors')
vectors.add('dm', mS.u.dm)
vectors.add('u', mS.u)
if mS.u_interp is not None:
    vectors.add('uEx', mS.u_interp)

meshes = d.addOutputGroup('meshes')
meshes.add('fullMesh', discrProblem.finalMesh)

results = d.addOutputGroup('results')
discrProblem.report(results)
mS.reportSolve(results)

errors = d.addOutputGroup('errors', tested=True)
mS.reportErrors(errors)

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
