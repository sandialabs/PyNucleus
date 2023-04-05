#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from numpy.linalg import norm as npnorm
from PyNucleus.base.ip_norm import ip_serial, norm_serial
from PyNucleus.base import driver, solverFactory, krylov_solver
from PyNucleus.base.linear_operators import invDiagonal
from PyNucleus.fem import (str2DoFMap,
                           meshFactory,
                           functionFactory)
from PyNucleus.fem.mesh import plotManager
from PyNucleus.nl.fractionalOrders import (constFractionalOrder,
                                           variableConstFractionalOrder,
                                           leftRightFractionalOrder,
                                           smoothedLeftRightFractionalOrder,
                                           innerOuterFractionalOrder)
from PyNucleus.nl.kernels import getFractionalKernel


d = driver()
d.add('domain', acceptedValues=['interval', 'square', 'circle'])
d.add('do_dense', True)
d.add('do_h2', False)
d.add('do_transpose', False)
d.add('solver', acceptedValues=['lu', 'cg', 'gmres'])
d.add('maxIter', 1000)
d.add('tol', 1e-7)
d.add('element', acceptedValues=['P1', 'P0'])
d.add('s1', 0.25)
d.add('s2', 0.75)
d.add('normalizePlot', False)

d.declareFigure('variableOrder')
d.declareFigure('error')

params = d.process()

s1 = d.s1
s2 = d.s2
sol1 = sol2 = None
smean = 0.5*(s1+s2)
if d.domain == 'interval':
    mesh = meshFactory.build(d.domain, noRef=8, a=-1, b=1)
    if d.element == 'P0':
        assert s1 < 0.5 and s2 < 0.5
        sVals = [constFractionalOrder(s1),
                 constFractionalOrder(s2),
                 leftRightFractionalOrder(s1, s2),
                 leftRightFractionalOrder(s1, s2, s1, smean),
                 leftRightFractionalOrder(s1, s2, s2, smean),
                 ]
    elif d.element == 'P1':
        sNonSym = leftRightFractionalOrder(s1, s2)
        sNonSym.symmetric = False
        sVals = [
            constFractionalOrder(s1),
            constFractionalOrder(s2),
            variableConstFractionalOrder(s1),
            variableConstFractionalOrder(s2),
            leftRightFractionalOrder(s1, s2, s1, s1),
            leftRightFractionalOrder(s1, s2, smean, smean),
            leftRightFractionalOrder(s1, s2, s2, s2),
            # sNonSym,
            
            # smoothedLeftRightFractionalOrder(s1, s2, slope=1000.),
            # leftRightFractionalOrder(s1, s2, s1, (s1+s2)/2),
            # leftRightFractionalOrder(s1, s2, s2, (s1+s2)/2)
        ]
    rhs = functionFactory.build('constant', value=1.)
    sol1 = functionFactory.build('solFractional', s=s1, dim=mesh.dim)
    sol2 = functionFactory.build('solFractional', s=s2, dim=mesh.dim)
elif d.domain == 'square':
    mesh = meshFactory.build(d.domain, noRef=5, N=2, M=2, ax=-1, ay=-1, bx=1, by=1)
    sVals = [
        # constFractionalOrder(s1),
        # constFractionalOrder(s2),
        leftRightFractionalOrder(s1, s2)
        # innerOuterFractionalOrder(mesh.dim, s1, s2, 0.3)
    ]
    # rhs = functionFactory.build('Lambda', fun=lambda x: 1. if x[0] > 0 else 0.)
    rhs = functionFactory.build('constant', value=1.)
elif d.domain == 'circle':
    mesh = meshFactory.build(d.domain, noRef=5, n=8)
    sVals = [
        innerOuterFractionalOrder(mesh.dim, s2, s1, 0.5),
    ]
    rhs = functionFactory.build('constant', value=1.)
    sol1 = functionFactory.build('solFractional', s=s1, dim=mesh.dim)
    sol2 = functionFactory.build('solFractional', s=s2, dim=mesh.dim)
else:
    raise NotImplementedError()

DoFMap = str2DoFMap(d.element)
dm = DoFMap(mesh)
d.logger.info(mesh)
d.logger.info(dm)

centerDoF = np.argmin(npnorm(dm.getDoFCoordinates(), axis=1))
horizon = functionFactory.build('constant', value=np.inf)
norm = norm_serial()
inner = ip_serial()


if d.willPlot('variableOrder'):
    plotDefaults = {}
    if mesh.dim == 2:
        plotDefaults['flat'] = True
    pM = plotManager(mesh, dm, defaults=plotDefaults)
    if d.do_dense and d.do_h2:
        pMerr = plotManager(mesh, dm, defaults=plotDefaults)

for s in sVals:
    b = dm.assembleRHS(rhs)
    err = None
    kernel = getFractionalKernel(mesh.dim, s, horizon)

    for label, do in [('dense', d.do_dense),
                      ('dense_general', d.do_dense),
                      ('H2', d.do_h2)]:
        if not do:
            continue
        if label == 'dense_general' and kernel.symmetric:
            continue
        with d.timer(label+' assemble '+str(s)):
            if label == 'dense':
                A = dm.assembleNonlocal(kernel, matrixFormat='dense')
            elif label == 'dense_general':
                A = dm.assembleNonlocal(kernel, matrixFormat='dense', params={'genKernel': True})
            elif label == 'H2':
                A = dm.assembleNonlocal(kernel, matrixFormat='H2')
        import matplotlib.pyplot as plt
        # from fractionalLaplacian.clusterMethodCy import getFractionalOrders
        # if label == 'H2':
        #     plt.figure()
        #     A2 = builder.getDense()
        #     plt.pcolormesh(np.log10(np.absolute(A.toarray()-A2.toarray())))
        #     plt.colorbar()
        #     if mesh.dim == 1:
        #         plt.figure()
        #         _, Pnear = builder.getH2(True)

        #         for c in Pnear:
        #             c.plot()
        #         # cell_orders = []
        #         # for c1 in c.n1.cells:
        #         #     for c2 in c.n2.cells:
        #         #         cell_orders.append(orders[c1, c2])
        #         # box1 = c.n1.box
        #         # box2 = c.n2.box
        #         # plt.text(0.5*(box1[0, 0]+box1[0, 1]), 0.5*(box2[0, 0]+box2[0, 1]), '({},{})'.format(min(cell_orders), max(cell_orders)))
        #         for lvl in A.Pfar:
        #             for c in A.Pfar[lvl]:
        #                 c.plot(color='blue' if c.constantKernel else 'green')
        #             # cell_orders = []
        #             # for c1 in c.n1.cells:
        #             #     for c2 in c.n2.cells:
        #             #         cell_orders.append(orders[c1, c2])
        #             # box1 = c.n1.box
        #             # box2 = c.n2.box
        #             # plt.text(0.5*(box1[0, 0]+box1[0, 1]), 0.5*(box2[0, 0]+box2[0, 1]), '({},{})'.format(min(cell_orders), max(cell_orders)))
        #     else:
        #         plt.figure()
        #         plt.spy(A.Anear.toarray())
        # d.logger.info(str(A))

        with d.timer(label+' solve '+str(s)):
            solver = solverFactory.build(d.solver, A=A,
                                         maxIter=d.maxIter, tolerance=d.tol,
                                         setup=True)
            if isinstance(solver, krylov_solver):
                Dinv = invDiagonal(A)
                solver.setPreconditioner(Dinv, False)
            x = dm.zeros()
            numIter = solver(b, x)
        # if err is None:
        #     err = x.copy()
        # else:
        #     err -= x
        #     pMerr.add(err, label=str(s))
        #     M = dm.assembleMass()
        #     L2err = np.sqrt(abs(inner(err, M*err)))
        #     d.logger.info('L2 error: {}'.format(L2err))
        d.logger.info('{}: resNorm {} in {} iters, norm {}'.format(s, norm(A*x-b), numIter, norm(x)))
        if d.normalizePlot:
            x /= x[centerDoF]
        if d.willPlot('variableOrder'):
            pM.add(x, label=label+' '+str(s))
        if not s.symmetric and d.do_transpose and d.do_dense:
            At = A.transpose()
            with d.timer(label+' solve transpose '+str(s)):
                solver = solverFactory.build(d.solver, A=At,
                                             maxIter=d.maxIter, tolerance=d.tol,
                                             setup=True)
                if isinstance(solver, krylov_solver):
                    Dinv = invDiagonal(At)
                    solver.setPreconditioner(Dinv, False)
                xt = dm.zeros()
                numIter = solver(b, xt)
            d.logger.info('{} transpose: resNorm {} in {} iters'.format(s, norm(At*xt-b), numIter))
            if d.normalizePlot:
                xt /= xt[centerDoF]
            if d.willPlot('variableOrder'):
                pM.add(xt, label=label+' transpose '+str(s))

if d.startPlot('variableOrder'):
    for s, sol in [(s1, sol1), (s2, sol2)]:
        if sol is not None:
            x = dm.interpolate(sol)
            if d.normalizePlot:
                x /= x[centerDoF]
            pM.add(x, label='exact '+str(s), ls='--')
    pM.plot(legendOutside=d.plotFolder != '')
if d.do_dense and d.do_h2:
    if d.startPlot('error'):
        pMerr.plot(legendOutside=d.plotFolder != '')
d.finish()
