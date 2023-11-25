#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from mpi4py import MPI
from PyNucleus import driver
from PyNucleus.nl import (fractionalLaplacianProblem,
                          discretizedNonlocalProblem)
from PyNucleus_nl.fractionalOrders import singleVariableUnsymmetricFractionalOrder

##################################################


d = driver(MPI.COMM_WORLD)
d.add('saveOperators', False)
d.add('vtkOutput', "")
p = fractionalLaplacianProblem(d, False)
discrProblem = discretizedNonlocalProblem(d, p)

d.declareFigure('solution')
d.declareFigure('error')
d.declareFigure('analyticSolution')
d.declareFigure('fractionalOrder')

d.process(override={'adaptive': None})

##################################################

mS = discrProblem.modelSolution

##################################################

vectors = d.addOutputGroup('vectors')
vectors.add('u', mS.u)
vectors.add('uInterior', mS.uInterior)

if d.saveOperators:
    matrices = d.addOutputGroup('matrices')
    matrices.add('A', discrProblem.A)
    matrices.add('A_BC', discrProblem.A_BC)

meshes = d.addOutputGroup('meshes')
meshes.add('fullMesh', discrProblem.finalMesh)

results = d.addOutputGroup('results')
discrProblem.report(results)
mS.reportSolve(results)
results.log()

errors = d.addOutputGroup('errors', tested=True)
mS.reportErrors(errors)
errors.log()

##################################################

plotDefaults = {}
if p.dim == 2:
    plotDefaults['flat'] = True
    if p.element != 'P0':
        plotDefaults['shading'] = 'gouraud'

if p.dim < 3 and d.startPlot('solution'):
    mS.plotSolution()
if p.dim < 3 and mS.error is not None and d.startPlot('error'):
    mS.error.plot(**plotDefaults)

if d.vtkOutput != "":
    mS.exportVTK(d.vtkOutput)

if isinstance(p.kernel.s, singleVariableUnsymmetricFractionalOrder) and d.startPlot('fractionalOrder'):
    mS.u.dm.interpolate(p.kernel.s.sFun).plot(**plotDefaults)

d.finish()
