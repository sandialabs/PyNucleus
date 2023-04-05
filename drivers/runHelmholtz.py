#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from mpi4py import MPI
import numpy as np
from PyNucleus.base import COMPLEX, driver, solverFactory
from PyNucleus.base.linear_operators import wrapRealToComplexCSR
from PyNucleus.fem.femCy import (assembleSurfaceMass,
                                 getSurfaceDoFMap)
from PyNucleus.fem import (PHYSICAL,
                           NO_BOUNDARY,
                           helmholtzProblem)
from PyNucleus.fem.mesh import plotManager
from PyNucleus.fem.functions import real, imag
from PyNucleus.multilevelSolver import (EmptyHierarchy,
                                        hierarchyManager,
                                        inputConnector,
                                        paramsForMG)


d = driver(MPI.COMM_WORLD)
p = helmholtzProblem(d)

d.add('partitioner', 'regular')
d.add('partitionerParams', {})
d.add('debugOverlaps', False)
d.add('maxiter', 300)

d.declareFigure('solution')
d.declareFigure('error')

params = d.process()

params['reaction'] = None
params['buildMass'] = True
params['tag'] = NO_BOUNDARY


with d.timer('setup levels'):
    hierarchies, connectors = paramsForMG(p.noRef,
                                          range(d.comm.size),
                                          params, p.dim, p.element)
    connectors['input'] = {'type': inputConnector,
                           'params': {'domain': p.domain}}
    FINE = 'fine'
    hierarchies[-1]['label'] = FINE

    hM = hierarchyManager(hierarchies, connectors, params, d.comm)
    hM.setup()
    hM.display()

    h = hM[FINE].meshLevels[-1].mesh.h
    tol = {'P1': 0.1*h**2,
           'P2': 0.001*h**3,
           'P3': 0.001*h**4}['P1']
    tol = 1e-5
    tol = max(tol, 2e-9)


def getOp(S, M, MB, frequency, shift=0):
    A = S - (M * frequency**2) + (MB * (1j*frequency))
    if shift == 0:
        return A.to_csr_linear_operator()
    else:
        B = M * (1j*shift*frequency**2)
        return (A + B).to_csr_linear_operator()


for h in hM.builtHierarchies:
    if isinstance(h, EmptyHierarchy):
        continue
    mesh = h.meshLevels[-1].mesh
    dm = h.algebraicLevels[-1].DoFMap

    surface = mesh.get_surface_mesh(PHYSICAL)
    MB = h.algebraicLevels[-1].S.copy()
    MB.setZero()
    assembleSurfaceMass(mesh, surface, dm, MB, sss_format=p.symmetric)
    h.algebraicLevels[-1].MB = MB

    for lvl in range(len(h.algebraicLevels)-2, -1, -1):
        if h.algebraicLevels[lvl].A is None:
            continue
        h.algebraicLevels[lvl].MB = h.algebraicLevels[lvl].A.copy()
        h.algebraicLevels[lvl].MB.setZero()
        h.algebraicLevels[lvl+1].R.restrictMatrix(h.algebraicLevels[lvl+1].MB,
                                                  h.algebraicLevels[lvl].MB)
    # for lvl in range(len(h.algebraicLevels)-2, -1, -1):
    #     if h.algebraicLevels[lvl].A is None:
    #         continue
    #     h.algebraicLevels[lvl].MB = h.algebraicLevels[lvl].A.copy()
    #     h.algebraicLevels[lvl].MB.data[:] = 0.
    #     mesh = h.meshLevels[lvl].mesh
    #     dm = h.algebraicLevels[lvl].DoFMap
    #     surface = mesh.get_surface_mesh(PHYSICAL)
    #     assembleSurfaceMass(mesh, surface, dm, h.algebraicLevels[lvl].MB)

    for lvl in range(len(h.algebraicLevels)):
        if h.algebraicLevels[lvl].S is None:
            continue
        if h.algebraicLevels[lvl].P is not None:
            h.algebraicLevels[lvl].P = wrapRealToComplexCSR(h.algebraicLevels[lvl].P)
        if h.algebraicLevels[lvl].R is not None:
            h.algebraicLevels[lvl].R = wrapRealToComplexCSR(h.algebraicLevels[lvl].R)
        h.algebraicLevels[lvl].A = getOp(h.algebraicLevels[lvl].S,
                                         h.algebraicLevels[lvl].M,
                                         h.algebraicLevels[lvl].MB,
                                         p.frequency,
                                         shift=0.5)

overlaps = hM[FINE].multilevelAlgebraicOverlapManager

ml = solverFactory.build('complex_mg',
                         hierarchy=hM,
                         smoother=('jacobi',
                                   {'omega': 0.8,
                                    # 'omega': min(2./3., 8./(4+3*self.dim)),
                                    'presmoothingSteps': 2,
                                    'postsmoothingSteps': 2}),
                         setup=True)
msg = '\n'+str(ml)
d.logger.info(msg)

mesh = hM[FINE].meshLevels[-1].mesh
dm = hM[FINE].algebraicLevels[-1].DoFMap
A = getOp(hM[FINE].algebraicLevels[-1].S,
          hM[FINE].algebraicLevels[-1].M,
          hM[FINE].algebraicLevels[-1].MB,
          p.frequency)
M = wrapRealToComplexCSR(hM[FINE].algebraicLevels[-1].M)
interfaces = hM[FINE].meshLevels[-1].interfaces

with d.timer('assemble RHS'):
    b = dm.assembleRHS(p.rhs)

    if p.boundaryCond is not None:
        surface = mesh.get_surface_mesh(PHYSICAL)
        dmS = getSurfaceDoFMap(mesh, surface, dm)
        b += dmS.assembleRHS(p.boundaryCond)

x = dm.zeros(dtype=COMPLEX)
gmres = solverFactory.build('complex_gmres', A=A, maxIter=d.maxiter, tolerance=tol, setup=True)
gmres.setPreconditioner(ml.asPreconditioner(), left=False)
gmres.setNormInner(ml.norm, ml.inner)
res = []
with d.timer('solve'):
    gmres(b, x)
    res = gmres.residuals

results = d.addOutputGroup('results', tested=True)
results.add('Tolerance', tol)
results.add('numIter', len(res))
results.add('res', res[-1], rTol=3e-1)
L2 = np.sqrt(abs(ml.inner(M*x, x)))
results.add('solution L2 norm', L2, rTol=1e-6)
if p.solEx is not None:
    solExReal = real(p.solEx)
    solExImag = imag(p.solEx)
    xEx = dm.interpolate(solExReal)+1j*dm.interpolate(solExImag)
    L2err = np.sqrt(abs(ml.inner(M*(x-xEx), x-xEx)))
    results.add('L2 error', L2err, rTol=2.)
d.logger.info('\n'+str(results))

if mesh.dim < 3:
    plotDefaults = {}
    if mesh.dim == 2:
        plotDefaults['flat'] = True
        plotDefaults['shading'] = 'gouraud'
    if d.willPlot('solution'):
        pM = plotManager(mesh, dm, defaults=plotDefaults, interfaces=interfaces)
        pM.add(x.real, label='solution (real)')
        pM.add(x.imag, label='solution (imag)')
        if p.solEx is not None:
            pM.add(xEx.real, label='exact solution (real)')
            pM.add(xEx.imag, label='exact solution (imag)')
        pM.preparePlots(tag=NO_BOUNDARY)
    if d.startPlot('solution'):
        pM.plot()
    if p.solEx is not None:
        if d.willPlot('error'):
            pMerr = plotManager(mesh, dm, defaults=plotDefaults, interfaces=interfaces)
            pMerr.add((x-xEx).real, label='error (real)')
            pMerr.add((x-xEx).imag, label='error (imag)')
            pMerr.preparePlots(tag=NO_BOUNDARY)
        if d.startPlot('error'):
            pMerr.plot()
d.finish()
