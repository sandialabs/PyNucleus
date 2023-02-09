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
from PyNucleus.nl import nonlocalPoissonProblem, FRACTIONAL
import numpy as np



##################################################
##################################################

d = driver(MPI.COMM_WORLD)
nPP = nonlocalPoissonProblem(d)

d.add('buildDense', False, help='Assemble dense matrix on rank 0')
d.add('buildSparse', False, help='Assemble sparse matrix on rank 0')
d.add('buildSparsified', False, help='Assemble sparsified matrix on rank 0')
d.add('buildH2', False, help='Assemble H2 matrix on rank 0')
d.add('buildSparseReduced', False, help='Assemble sparse matrix on global communicator, reduce to rank 0')
d.add('buildH2Reduced', False, help='Assemble H2 matrix on global communicator, reduce to rank 0')
d.add('buildDistributedH2Bcast', False, help='Assemble H2 matrix on global communicator, apply to global vectors using bcast')
d.add('buildDistributedH2', True, help='Assemble H2 matrix on global communicator, apply to local vectors')
d.add('buildDistributedSparse', False, help='Assemble sparse matrix on global communicator, apply to local vectors')
d.add('doSolve', False)
d.add('numApplies', 1)
d.add('horizonToMeshSize', -1.)


d.declareFigure('solution')

d.process(override={'horizon': np.inf})

if nPP.kernel.kernelType != FRACTIONAL:
    # H2 matrix assembly is only implemented for fractional kernels
    assert not d.buildH2
    assert not d.buildH2Reduced
    assert not d.buildDistributedH2
if d.buildSparse or d.buildSparsified or d.buildSparseReduced or d.buildDistributedSparse:
    assert nPP.kernel.horizon.value < np.inf
if d.buildDistributedSparse:
    assert not d.buildDistributedH2
    assert nPP.kernel.horizon.value < np.inf

if nPP.domain == 'disc':
    if d.horizonToMeshSize <= 0. or nPP.kernel.horizon.value == np.inf:
        h = 0.04/2**(nPP.noRef-3)
    else:
        h = nPP.kernel.horizon.value/d.horizonToMeshSize/np.sqrt(2)
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
info.log()

if nPP.analyticSolution is not None:
    x = dm.interpolate(nPP.analyticSolution)
else:
    x = dm.ones()
    if d.isMaster:
        xd = np.random.rand(dm.num_dofs)
    else:
        xd = None
    from PyNucleus import functionFactory
    xd = dm.interpolate(functionFactory('sin1d'))
    xd = d.comm.bcast(xd)
    x.assign(xd)

d.comm.Barrier()

##################################################

if d.buildDense:
    if d.isMaster:
        with d.timer('dense build'):
            A_dense = dm.assembleNonlocal(nPP.kernel, matrixFormat='dense')
        d.logger.info('Dense: {}'.format(A_dense))
        with d.timer('dense matvec'):
            y_dense = A_dense*x
    else:
        y_dense = None
    y_dense = d.comm.bcast(y_dense)
    y_dense = dm.fromArray(y_dense)
    d.comm.Barrier()

##################################################

if d.buildSparse:
    if d.isMaster:
        tm = TimerManager(d.logger, comm=d.comm, memoryProfiling=True, loggingSubTimers=True)
        with d.timer('sparse build'):
            A_sparse = dm.assembleNonlocal(nPP.kernel, matrixFormat='sparse',
                                           PLogger=tm.PLogger)
        d.logger.info('Sparse: {}'.format(A_sparse))
        with d.timer('sparse matvec'):
            y_sparse = A_sparse*x
    else:
        y_sparse = None
    y_sparse = d.comm.bcast(y_sparse)
    y_sparse = dm.fromArray(y_sparse)
    d.comm.Barrier()

##################################################

if d.buildSparsified:
    if d.isMaster:
        tm = TimerManager(d.logger, comm=d.comm, memoryProfiling=True, loggingSubTimers=True)
        with d.timer('sparsified build'):
            A_sparsified = dm.assembleNonlocal(nPP.kernel, matrixFormat='sparsified',
                                               PLogger=tm.PLogger)
        d.logger.info('Sparsified: {}'.format(A_sparse))
        with d.timer('sparsified matvec'):
            y_sparsified = A_sparsified*x
    else:
        y_sparsified = None
    y_sparsified = d.comm.bcast(y_sparsified)
    y_sparsified = dm.fromArray(y_sparsified)
    d.comm.Barrier()

##################################################

if d.buildH2:
    if d.isMaster:
        with d.timer('H2 rank build'):
            A_h2 = dm.assembleNonlocal(nPP.kernel, matrixFormat='H2')
        d.logger.info('H2: {}'.format(A_h2))
        with d.timer('H2 matvec'):
            y_h2 = A_h2*x
    else:
        y_h2 = None
    y_h2 = d.comm.bcast(y_h2)
    y_h2 = dm.fromArray(y_h2)
    d.comm.Barrier()

##################################################

if d.buildH2Reduced:
    with d.timer('distributed, summed build'):
        A_h2_reduced = dm.assembleNonlocal(nPP.kernel, matrixFormat='H2', comm=d.comm)
    if d.isMaster:
        d.logger.info('Reduced:     {}'.format(A_h2_reduced))
        with d.timer('distributed, summed matvec'):
            y_h2_reduced = A_h2_reduced*x
    d.comm.Barrier()

##################################################

if d.buildSparseReduced:
    with d.timer('distributed, summed build'):
        A_sparse_reduced = dm.assembleNonlocal(nPP.kernel, matrixFormat='sparse', comm=d.comm, params={'assembleOnRoot': True})
    if d.isMaster:
        d.logger.info('Reduced:     {}'.format(A_sparse_reduced))
        with d.timer('distributed, summed matvec'):
            y_sparse_reduced = A_sparse_reduced*x
    d.comm.Barrier()

##################################################

if d.buildDistributedH2Bcast:
    with d.timer('distributed, bcast build'):
        A_distributedH2Bcast = dm.assembleNonlocal(nPP.kernel, matrixFormat='H2', comm=d.comm,
                                                 params={'assembleOnRoot': False,
                                                         'forceUnsymmetric': True})
    with d.timer('distributed, bcast matvec'):
        print('Distributed:     ', A_distributedH2Bcast)
        y_distributedH2Bcast = A_distributedH2Bcast*x

##################################################

if d.buildDistributedH2:
    tm = TimerManager(d.logger, comm=d.comm, memoryProfiling=True, loggingSubTimers=True)
    with d.timer('distributed, halo build'):
        A_distributedH2 = dm.assembleNonlocal(nPP.kernel, matrixFormat='H2', comm=d.comm,
                                                params={'assembleOnRoot': False,
                                                        'forceUnsymmetric': True,
                                                        'localFarFieldIndexing': True},
                                                PLogger=tm.PLogger)
    t = d.addOutputGroup('TimersH2', timerOutputGroup(driver=d))
    tm.setOutputGroup(d.masterRank, t)
    t.log()

    stats = d.addStatsOutputGroup('stats')
    stats.add('number of tree levels', A_distributedH2.localMat.tree.numLevels, sumOverRanks=False)
    stats.add('number of tree nodes', A_distributedH2.localMat.tree.nodes)
    stats.add('number of near field cluster pairs', len(A_distributedH2.Pnear))
    stats.add('number of near field entries', A_distributedH2.localMat.nearField_size)
    stats.add('number of far field cluster pairs', A_distributedH2.localMat.num_far_field_cluster_pairs)
    stats.add('memory size (MB)', A_distributedH2.localMat.getMemorySize()/1024**2)
    stats.log()

##################################################

if d.buildDistributedSparse:
    tm = TimerManager(d.logger, comm=d.comm, memoryProfiling=True, loggingSubTimers=True)
    with d.timer('distributed, sparse build'):
        A_distributedSparse = dm.assembleNonlocal(nPP.kernel, matrixFormat='sparse', comm=d.comm,
                                                  params={'assembleOnRoot': False,
                                                          'forceUnsymmetric': True,
                                                          'localFarFieldIndexing': True},
                                                  PLogger=tm.PLogger)
    stats = d.addStatsOutputGroup('stats')
    stats.add('number of near field entries', A_distributedSparse.localMat.nnz)
    stats.add('memory size (MB)', A_distributedSparse.localMat.getMemorySize()/1024**2)
    stats.log()

##################################################

if d.buildDistributedH2:
    A_distributed = A_distributedH2
elif d.buildDistributedSparse:
    A_distributed = A_distributedSparse
else:
    A_distributed = None
if (d.buildDistributedH2 or d.buildDistributedSparse) and (d.numApplies > 0):
    lcl_dm = A_distributed.lcl_dm
    x_local = lcl_dm.zeros()
    x_local.assign(A_distributed.lclR*x)
    y_distributed = lcl_dm.zeros()
    for k in range(d.numApplies):
        with d.timer('distributed, halo matvec'):
            A_distributed(x_local, y_distributed)

##################################################
##################################################

d.comm.Barrier()

matvecErrors = d.addOutputGroup('matvec errors', tested=True, rTol=1.)
if d.buildDense and (d.buildDistributedH2 or d.buildDistributedSparse):
    diff_lcl = y_distributed-(A_distributed.lclR*y_dense)
    err_dense_dist = diff_lcl.norm()

if d.buildSparse and (d.buildDistributedH2 or d.buildDistributedSparse):
    diff_lcl = y_distributed-(A_distributed.lclR*y_sparse)
    err_sparse_dist = diff_lcl.norm()

if d.buildH2 and (d.buildDistributedH2 or d.buildDistributedSparse):
    diff_lcl = y_distributed-(A_distributed.lclR*y_h2)
    err_h2_dist = diff_lcl.norm()

if d.isMaster:
    if d.buildDense:
        if d.buildSparse:
            matvecErrors.add('|(A_dense - A_sparse) * x|', np.linalg.norm(y_dense-y_sparse))
        if d.buildSparsified:
            matvecErrors.add('|(A_dense - A_sparsified) * x|', np.linalg.norm(y_dense-y_sparsified))
        if d.buildH2:
            matvecErrors.add('|(A_dense - A_h2) * x |', np.linalg.norm(y_dense-y_h2))
        if d.buildSparseReduced:
            matvecErrors.add('|(A_dense - A_sparse_reduced) * x|', np.linalg.norm(y_dense-y_sparse_reduced))
        if d.buildH2Reduced:
            matvecErrors.add('|(A_dense - A_h2_reduced) * x|', np.linalg.norm(y_dense-y_h2_reduced))
        if d.buildDistributedH2Bcast:
            matvecErrors.add('|(A_dense - A_distributed_bcast) * x|', np.linalg.norm(y_dense-y_distributedH2Bcast))
        if d.buildDistributedH2:
            matvecErrors.add('|(A_dense - A_distributed_halo) * x|', err_dense_dist)
        if d.buildDistributedSparse:
            matvecErrors.add('|(A_dense - A_distributed_sparse) * x|', err_dense_dist)
    if d.buildSparse:
        if d.buildSparsified:
            matvecErrors.add('|(A_sparse - A_sparsified) * x|', np.linalg.norm(y_sparse-y_sparsified))
        if d.buildSparseReduced:
            matvecErrors.add('|(A_sparse - A_sparse_reduced) * x|', np.linalg.norm(y_sparse-y_sparse_reduced))
        if d.buildDistributedH2Bcast:
            matvecErrors.add('|(A_sparse - A_distributed_bcast) * x|', np.linalg.norm(y_sparse-y_distributedH2Bcast))
        if d.buildDistributedH2:
            matvecErrors.add('|(A_sparse - A_distributed_halo) * x|', err_sparse_dist)
        if d.buildDistributedSparse:
            matvecErrors.add('|(A_sparse - A_distributed_sparse) * x|', err_sparse_dist)
    if d.buildH2:
        if d.buildH2Reduced:
            matvecErrors.add('|(A_h2 - A_h2_reduced) * x |', np.linalg.norm(y_h2-y_h2_reduced))
        if d.buildDistributedH2Bcast:
            matvecErrors.add('|(A_h2 - A_distributed_bcast) * x|', np.linalg.norm(y_h2-y_distributedH2Bcast))
        if d.buildDistributedH2 or d.buildDistributedSparse:
            matvecErrors.add('|(A_h2 - A_distributed_halo) * x|', err_h2_dist)
matvecErrors.log()

##################################################
##################################################


if d.doSolve and (d.buildDistributedH2 or d.buildDistributedSparse):
    b = lcl_dm.assembleRHS(nPP.rhs)

    if nPP.boundaryCondition == DIRICHLET:
        assert not d.buildDistributedH2 and d.buildDistributedSparse

        raise NotImplementedError()
    elif nPP.boundaryCondition in (NEUMANN, HOMOGENEOUS_NEUMANN):
        # pure Neumann condition -> project out nullspace
        assert dm.num_boundary_dofs == 0, dm.num_boundary_dofs
        const = lcl_dm.ones()
        b -= b.inner(const)/const.inner(const)*const

    cg = solverFactory('cg', A=A_distributed, setup=True)
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

    u_global = d.comm.reduce(A_distributed.lclP*u)
    if d.isMaster and nPP.analyticSolution is not None:
        u = A_distributed.dm.zeros()
        u.assign(u_global)
        M = A_distributed.dm.assembleMass()
        u_ex = A_distributed.dm.interpolate(nPP.analyticSolution)
        errL2 = np.sqrt(np.vdot(u-u_ex, M*(u-u_ex)))
        solveGroup.add('L2 error', errL2, rTol=1e-1)
    solveGroup.log()

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
