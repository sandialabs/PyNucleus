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
                       solverFactory,
                       nonlocalMeshFactory,
                       HOMOGENEOUS_DIRICHLET,
                       DIRICHLET,
                       NEUMANN,
                       HOMOGENEOUS_NEUMANN)
from PyNucleus.fem import plotManager
from PyNucleus.base.utilsFem import TimerManager, timerOutputGroup
from PyNucleus.nl import nonlocalPoissonProblem
import numpy as np



##################################################
##################################################

d = driver(MPI.COMM_WORLD)
nPP = nonlocalPoissonProblem(d)

d.add('buildDense', False)
d.add('buildSparse', False)
d.add('buildSingle', False)
d.add('buildReduced', False)
d.add('buildDistributedBcast', False)
d.add('buildDistributedHalo', True)
d.add('buildDistributedSparse', False)
d.add('doSolve', False)
d.add('horizonToMeshSize', -1.)


d.declareFigure('solution')

d.process(override={'horizon': np.inf})

if d.buildSparse:
    assert nPP.kernel.horizon.value < np.inf
if d.buildDistributedSparse:
    assert not d.buildDistributedHalo
    assert nPP.kernel.horizon.value < np.inf

d.comm.Barrier()

if nPP.domain == 'disc':
    if d.horizonToMeshSize <= 0. or d.kernel.horizon.value == np.inf:
        h = 0.04/2**(nPP.noRef-3)
    else:
        h = nPP.kernel.horizon.value/d.horizonToMeshSize
    mesh, _ = nonlocalMeshFactory(nPP.domain,
                                  kernel=nPP.kernel,
                                  boundaryCondition=HOMOGENEOUS_DIRICHLET,
                                  h=h,
                                  max_volume=h**2/2,
                                  projectNodeToOrigin=False)
elif d.domain == 'gradedDisc':
    if d.horizonToMeshSize <= 0. or nPP.kernel.horizon.value == np.inf:
        h = 0.04/2**(d.noRef-6)
    else:
        h = nPP.kernel.horizon.value/d.horizonToMeshSize
    mesh, _ = nonlocalMeshFactory(nPP.domain,
                                  kernel=nPP.kernel,
                                  boundaryCondition=HOMOGENEOUS_DIRICHLET,
                                  h=h,
                                  max_volume=h**2/2)
else:
    if d.horizonToMeshSize <= 0. or nPP.kernel.horizon.value == np.inf:
        mesh = nPP.mesh
        for _ in range(nPP.noRef):
            mesh = mesh.refine()
    else:
        mesh = nPP.mesh
        while d.horizonToMeshSize > np.around(nPP.kernel.horizon.value/mesh.h, 5):
            mesh = mesh.refine()
if nPP.boundaryCondition == HOMOGENEOUS_DIRICHLET:
    dm = dofmapFactory(nPP.element, mesh, nPP.domainIndicator)
else:
    dm = dofmapFactory(nPP.element, mesh, nPP.domainIndicator+nPP.fluxIndicator)

assert d.comm.allreduce(dm.num_dofs, op=MPI.MAX) == dm.num_dofs

info = d.addOutputGroup('info')
info.add('Global mesh', dm.mesh)
info.add('mesh size', dm.mesh.h)
info.add('Mesh aspect ratio', dm.mesh.h/dm.mesh.hmin)
info.add('Global DM', dm)
info.add('Kernel', nPP.kernel)
info.add('horizon/h', nPP.kernel.horizon.value/dm.mesh.h)
d.logger.info('\n'+str(info))

d.comm.Barrier()

if nPP.analyticSolution is not None:
    x = dm.interpolate(nPP.analyticSolution)
else:
    x = dm.ones()
    if d.isMaster:
        xd = np.random.rand(dm.num_dofs)
    else:
        xd = None
    xd = d.comm.bcast(xd)
    x.assign(xd)

if d.buildDense:
    if d.isMaster:
        with d.timer('dense build'):
            Ad = dm.assembleNonlocal(nPP.kernel, matrixFormat='dense')
        with d.timer('dense matvec'):
            print('Dense: ', Ad)
            yd = Ad*x
    else:
        yd = None
    yd = d.comm.bcast(yd)
    yd = dm.fromArray(yd)
    d.comm.Barrier()

if d.buildSparse:
    if d.isMaster:
        with d.timer('sparse build'):
            As = dm.assembleNonlocal(nPP.kernel, matrixFormat='sparse')
        with d.timer('sparse matvec'):
            print('Sparse: ', As)
            ys = As*x
    else:
        ys = None
    ys = d.comm.bcast(ys)
    ys = dm.fromArray(ys)
    d.comm.Barrier()

if d.buildSingle:
    if d.isMaster:
        with d.timer('single rank build'):
            A0 = dm.assembleNonlocal(nPP.kernel, matrixFormat='H2')
        with d.timer('single rank matvec'):
            print('Single rank: ', A0)
            y0 = A0*x
    else:
        y0 = None
    y0 = d.comm.bcast(y0)
    y0 = dm.fromArray(y0)
    d.comm.Barrier()

if d.buildReduced:
    with d.timer('distributed, summed build'):
        A1 = dm.assembleNonlocal(nPP.kernel, matrixFormat='H2', comm=d.comm)
    if d.isMaster:
        with d.timer('distributed, summed matvec'):
            print('Reduced:     ', A1)
            y1 = A1*x
    d.comm.Barrier()

if d.buildDistributedBcast:
    with d.timer('distributed, bcast build'):
        A2 = dm.assembleNonlocal(nPP.kernel, matrixFormat='H2', comm=d.comm,
                                 params={'assembleOnRoot': False,
                                         'forceUnsymmetric': True})
    with d.timer('distributed, bcast matvec'):
        print('Distributed:     ', A2)
        y2 = A2*x

if d.buildDistributedHalo:
    tm = TimerManager(d.logger, comm=d.comm, memoryProfiling=True, loggingSubTimers=True)
    with d.timer('distributed, halo build'):
        A3 = dm.assembleNonlocal(nPP.kernel, matrixFormat='H2', comm=d.comm,
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

if d.buildDistributedSparse:
    with d.timer('distributed, sparse build'):
        A3 = dm.assembleNonlocal(nPP.kernel, matrixFormat='sparse', comm=d.comm)

    stats = d.addStatsOutputGroup('stats')
    stats.add('number of near field entries', A3.localMat.nnz)
    stats.add('memory size (MB)', A3.localMat.getMemorySize()/1024**2)
    d.logger.info('\n'+str(stats))

if d.buildDistributedHalo or d.buildDistributedSparse:
    lcl_dm = A3.lcl_dm
    x_local = lcl_dm.zeros()
    x_local.assign(A3.lclR*x)
    y3 = lcl_dm.zeros()
    for k in range(100):
        with d.timer('distributed, halo matvec'):
            A3(x_local, y3)

##################################################
##################################################

d.comm.Barrier()

matvecErrors = d.addOutputGroup('matvec errors', tested=True, rTol=1.)
if d.buildDense and (d.buildDistributedHalo or d.buildDistributedSparse):
    diff_lcl = y3-(A3.lclR*yd)
    err_dense_dist = diff_lcl.norm()

if d.buildSparse and (d.buildDistributedHalo or d.buildDistributedSparse):
    diff_lcl = y3-(A3.lclR*ys)
    err_sparse_dist = diff_lcl.norm()

if d.buildSingle and (d.buildDistributedHalo or d.buildDistributedSparse):
    diff_lcl = y3-(A3.lclR*y0)
    err_single_dist = diff_lcl.norm()

if d.isMaster:
    if d.buildDense:
        if d.buildReduced:
            matvecErrors.add('|(A_dense - A_reduced) * x|', np.linalg.norm(yd-y1))
        if d.buildDistributedBcast:
            matvecErrors.add('|(A_dense - A_distributed_bcast) * x|', np.linalg.norm(yd-y2))
        if d.buildDistributedHalo or d.buildDistributedSparse:
            matvecErrors.add('|(A_dense - A_distributed_halo) * x|', err_dense_dist)
    if d.buildSparse:
        if d.buildReduced:
            matvecErrors.add('|(A_sparse - A_reduced) * x|', np.linalg.norm(ys-y1))
        if d.buildDistributedBcast:
            matvecErrors.add('|(A_sparse - A_distributed_bcast) * x|', np.linalg.norm(ys-y2))
        if d.buildDistributedHalo or d.buildDistributedSparse:
            matvecErrors.add('|(A_sparse - A_distributed_halo) * x|', err_sparse_dist)
    if d.buildSingle:
        if d.buildDense:
            matvecErrors.add('|(A_dense - A_single) * x |', np.linalg.norm(yd-y0))
        if d.buildReduced:
            matvecErrors.add('|(A_single - A_reduced) * x |', np.linalg.norm(y0-y1))
        if d.buildDistributedBcast:
            matvecErrors.add('|(A_single - A_distributed_bcast) * x|', np.linalg.norm(y0-y2))
        if d.buildDistributedHalo or d.buildDistributedSparse:
            matvecErrors.add('|(A_single - A_distributed_halo) * x|', err_single_dist)
d.logger.info('\n'+str(matvecErrors))

##################################################
##################################################


if d.doSolve and (d.buildDistributedHalo or d.buildDistributedSparse):
    b = lcl_dm.assembleRHS(nPP.rhs)

    if nPP.boundaryCondition == DIRICHLET:
        raise NotImplementedError()
    # pure Neumann condition -> project out nullspace
    if nPP.boundaryCondition in (NEUMANN, HOMOGENEOUS_NEUMANN):
        assert dm.num_boundary_dofs == 0, dm.num_boundary_dofs
        const = lcl_dm.ones()
        b -= b.inner(const)/const.inner(const)*const

    cg = solverFactory('cg', A=A3, setup=True)
    cg.setNormInner(lcl_dm.norm,
                    lcl_dm.inner)
    cg.maxIter = 1000
    u = lcl_dm.zeros()
    cg(b, u)

    residuals = cg.residuals
    solveGroup = d.addOutputGroup('solve', tested=True, rTol=1e-1)
    solveGroup.add('residual norm', residuals[-1])

    # pure Neumann condition -> add nullspace components to match analytic solution
    if nPP.boundaryCondition in (NEUMANN, HOMOGENEOUS_NEUMANN) and nPP.analyticSolution is not None:
        uEx = lcl_dm.interpolate(nPP.analyticSolution)
        u += (const.inner(uEx)-const.inner(u))/const.inner(const) * const

    u_global = d.comm.reduce(A3.lclP*u)
    if d.isMaster and nPP.analyticSolution is not None:
        u = A3.dm.zeros()
        u.assign(u_global)
        M = A3.dm.assembleMass()
        u_ex = A3.dm.interpolate(nPP.analyticSolution)
        errL2 = np.sqrt(np.vdot(u-u_ex, M*(u-u_ex)))
        solveGroup.add('L2 error', errL2, rTol=1e-1)
    d.logger.info('\n'+str(solveGroup))

    if d.startPlot('solution'):
        plotDefaults = {}
        if nPP.dim == 2:
            plotDefaults['flat'] = True
        if nPP.element != 'P0':
            plotDefaults['shading'] = 'gouraud'
        pM = plotManager(dm.mesh, dm, defaults=plotDefaults)
        pM.add(u_global, label='numerical solution')
        if nPP.analyticSolution is not None:
            pM.add(nPP.analyticSolution, label='analytic solution')
        pM.plot()

d.finish()
