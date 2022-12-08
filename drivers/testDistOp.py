#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from mpi4py import MPI
from PyNucleus.base import driver
from PyNucleus import (dofmapFactory,
                       INDEX,
                       REAL,
                       solverFactory,
                       nonlocalMeshFactory, HOMOGENEOUS_DIRICHLET)
from PyNucleus.fem import plotManager
from PyNucleus.base.ip_norm import (ip_distributed_nonoverlapping,
                                    norm_distributed_nonoverlapping)
from PyNucleus.base.utilsFem import TimerManager, timerOutputGroup
from PyNucleus.nl import (nonlocalBuilder,
                          nonlocalProblem)
import numpy as np



##################################################
##################################################

d = driver(MPI.COMM_WORLD)
nonlocalProblem(d)

d.add('buildDense', False)
d.add('buildSingle', False)
d.add('buildReduced', False)
d.add('buildDistributedBcast', False)
d.add('buildDistributedHalo', True)
d.add('doSolve', False)


d.declareFigure('solution')

params = d.process(override={'adaptive': None, 'horizon': np.inf})

d.comm.Barrier()

if d.domain == 'disc':
    h = 0.04/2**(d.noRef-3)
    mesh, _ = nonlocalMeshFactory(d.domain,
                                  kernel=d.kernel,
                                  boundaryCondition=HOMOGENEOUS_DIRICHLET,
                                  h=h,
                                  max_volume=h**2/2,
                                  projectNodeToOrigin=False)
elif d.domain == 'gradedDisc':
    h = 0.04/2**(d.noRef-6)
    mesh, _ = nonlocalMeshFactory(d.domain,
                                  kernel=d.kernel,
                                  boundaryCondition=HOMOGENEOUS_DIRICHLET,
                                  h=h,
                                  max_volume=h**2/2)
else:
    mesh = d.mesh
    for _ in range(d.noRef):
        mesh = mesh.refine()
dm = dofmapFactory(d.element, mesh)

if d.isMaster:
    print("Global mesh: ", dm.mesh)
    print("Mesh aspect ratio: ", dm.mesh.h/dm.mesh.hmin)
    print("Global DM: ", dm)

d.comm.Barrier()

if d.analyticSolution is not None:
    x = dm.interpolate(d.analyticSolution)
else:
    x = dm.ones()
    x.assign(np.random.rand(dm.num_dofs))

if d.buildDense:
    if d.isMaster:
        with d.timer('dense build'):
            Ad = dm.assembleNonlocal(d.kernel, matrixFormat='dense')
        with d.timer('dense matvec'):
            print('Dense: ', Ad)
            yd = Ad*x
    else:
        yd = None
    yd = d.comm.bcast(yd)
    d.comm.Barrier()

if d.buildSingle:
    if d.isMaster:
        with d.timer('single rank build'):
            A0 = dm.assembleNonlocal(d.kernel, matrixFormat='H2')
        with d.timer('single rank matvec'):
            print('Single rank: ', A0)
            y0 = A0*x
    else:
        y0 = None
    y0 = d.comm.bcast(y0)
    d.comm.Barrier()

if d.buildReduced:
    with d.timer('distributed, summed build'):
        A1 = dm.assembleNonlocal(d.kernel, matrixFormat='H2', comm=d.comm)
    if d.isMaster:
        with d.timer('distributed, summed matvec'):
            print('Reduced:     ', A1)
            y1 = A1*x
    d.comm.Barrier()

if d.buildDistributedBcast:
    with d.timer('distributed, bcast build'):
        A2 = dm.assembleNonlocal(d.kernel, matrixFormat='H2', comm=d.comm,
                                 params={'assembleOnRoot': False,
                                         'forceUnsymmetric': True})
    with d.timer('distributed, bcast matvec'):
        print('Distributed:     ', A2)
        y2 = A2*x

if d.buildDistributedHalo:
    tm = TimerManager(d.logger, comm=d.comm, memoryProfiling=True, loggingSubTimers=True)
    with d.timer('distributed, halo build'):
        A3 = dm.assembleNonlocal(d.kernel, matrixFormat='H2', comm=d.comm,
                                 params={'assembleOnRoot': False,
                                         'forceUnsymmetric': True,
                                         'localFarFieldIndexing': True},
                                 PLogger=tm.PLogger)
    t = d.addOutputGroup('TimersH2', timerOutputGroup())
    tm.setOutputGroup(d.masterRank, t)
    d.logger.info('\n'+str(t))

    stats = d.addStatsOutputGroup('stats')
    stats.add('number of tree levels', A3.localMat.tree.numLevels, sumOverRanks=False)
    stats.add('number of tree nodes', A3.localMat.tree.nodes)
    stats.add('number of near field cluster pairs', len(A3.Pnear))
    stats.add('number of near field entries', A3.localMat.nearField_size)
    stats.add('number of far field cluster pairs', A3.localMat.num_far_field_cluster_pairs)
    stats.add('memory size (MB)', A3.localMat.getMemorySize()/1024**2)
    d.logger.info('\n'+str(stats))


    x_local = A3.lcl_dm.zeros()
    x_local.assign(A3.lclR*x)
    y3 = A3.lcl_dm.zeros()
    for k in range(100):
        with d.timer('distributed, halo matvec'):
            A3(x_local, y3)

##################################################
##################################################

d.comm.Barrier()

matvecErrors = d.addOutputGroup('matvec errors', tested=True, rTol=1.)
if d.buildDense and d.buildDistributedHalo:
    diff_lcl = (A3.lclR*yd)-y3
    err_dense_dist = np.sqrt(d.comm.allreduce(np.vdot(diff_lcl, diff_lcl)))

if d.buildSingle and d.buildDistributedHalo:
    diff_lcl = (A3.lclR*y0)-y3
    err_single_dist = np.sqrt(d.comm.allreduce(np.vdot(diff_lcl, diff_lcl)))

if d.isMaster:
    if d.buildDense:
        if d.buildReduced:
            matvecErrors.add('|(A_dense - A_reduced) * x|', np.linalg.norm(yd-y1))
        if d.buildDistributedBcast:
            matvecErrors.add('|(A_dense - A_distributed_bcast) * x|', np.linalg.norm(yd-y2))
        if d.buildDistributedHalo:
            matvecErrors.add('|(A_dense - A_distributed_halo) * x|', err_dense_dist)
    if d.buildSingle:
        if d.buildDense:
            matvecErrors.add('|(A_dense - A_single) * x |', np.linalg.norm(yd-y0))
        if d.buildReduced:
            matvecErrors.add('|(A_single - A_reduced) * x |', np.linalg.norm(y0-y1))
        if d.buildDistributedBcast:
            matvecErrors.add('|(A_single - A_distributed_bcast) * x|', np.linalg.norm(y0-y2))
        if d.buildDistributedHalo:
            matvecErrors.add('|(A_single - A_distributed_halo) * x|', err_single_dist)
d.logger.info('\n'+str(matvecErrors))

##################################################
##################################################


if d.doSolve and d.buildDistributedHalo:
    b = A3.lcl_dm.assembleRHS(d.rhs)

    cg = solverFactory('cg', A=A3, setup=True)
    cg.setNormInner(norm_distributed_nonoverlapping(A3.comm),
                    ip_distributed_nonoverlapping(A3.comm))
    cg.maxIter = 1000
    u = A3.lcl_dm.zeros()
    cg(b, u)

    residuals = cg.residuals
    solveGroup = d.addOutputGroup('solve', tested=True, rTol=1e-1)
    solveGroup.add('residual norm', residuals[-1])

    u_global = d.comm.reduce(A3.lclP*u)
    if d.isMaster and d.analyticSolution is not None:
        u = A3.dm.zeros()
        u.assign(u_global)
        M = A3.dm.assembleMass()
        u_ex = A3.dm.interpolate(d.analyticSolution)
        errL2 = np.sqrt(np.vdot(u-u_ex, M*(u-u_ex)))
        solveGroup.add('L2 error', errL2, rTol=1e-1)
    d.logger.info('\n'+str(solveGroup))

    if d.startPlot('solution'):
        plotDefaults = {}
        if d.dim == 2:
            plotDefaults['flat'] = True
        if d.element != 'P0':
            plotDefaults['shading'] = 'gouraud'
        pM = plotManager(dm.mesh, dm, defaults=plotDefaults)
        pM.add(u_global, label='numerical solution')
        if d.analyticSolution is not None:
            pM.add(d.analyticSolution, label='analytic solution')
        pM.plot()

d.finish()
