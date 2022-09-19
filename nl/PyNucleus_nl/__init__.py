###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from PyNucleus_base import INDEX
from PyNucleus_base.linear_operators import (LinearOperator,
                                             diagonalOperator,
                                             multiIntervalInterpolationOperator,
                                             delayedConstructionOperator)
from . twoPointFunctions import constantTwoPoint
from . fractionalOrders import (constantFractionalLaplacianScaling,
                                variableFractionalLaplacianScaling,
                                constantIntegrableScaling,
                                constFractionalOrder,
                                variableFractionalOrder,
                                variableConstFractionalOrder,
                                leftRightFractionalOrder,
                                smoothedLeftRightFractionalOrder,
                                innerOuterFractionalOrder,
                                islandsFractionalOrder,
                                layersFractionalOrder)
from . kernelsCy import (Kernel,
                         FractionalKernel,
                         getKernelEnum,
                         FRACTIONAL, INDICATOR, PERIDYNAMIC)
from . kernels import getKernel, getIntegrableKernel, getFractionalKernel
from . nonlocalLaplacian import (assembleNonlocalOperator,
                                 
                                 nonlocalBuilder)
from . clusterMethodCy import H2Matrix
from . fractionalLaplacian1D import (fractionalLaplacian1D_P1,
                                     fractionalLaplacian1D_P1_boundary)

from . fractionalLaplacian2D import (fractionalLaplacian2D_P1,
                                     fractionalLaplacian2D_P1_boundary)
from . nonlocalLaplacianND import (integrable1D,
                                   integrable2D)

from PyNucleus_fem import (DIRICHLET, HOMOGENEOUS_DIRICHLET,
                           NEUMANN, HOMOGENEOUS_NEUMANN,
                           NORM, boundaryConditions)
from . nonlocalProblems import (fractionalLaplacianProblem,
                                nonlocalProblem,
                                fractionalOrderFactory,
                                interactionFactory,
                                kernelFactory,
                                nonlocalMeshFactory)
from PyNucleus_fem import (P0_DoFMap, getSubmesh,
                           constant, Lambda)
from PyNucleus_fem.DoFMaps import (getSubMapRestrictionProlongation,
                                   getSubMapRestrictionProlongation2)
from PyNucleus_multilevelSolver.levels import (algebraicLevelBase,
                                               SPARSITY_PATTERN,
                                               ASSEMBLY,
                                               NO_BUILD)
from PyNucleus_multilevelSolver import hierarchyManager
from PyNucleus_multilevelSolver.connectors import (inputConnector,
                                                   repartitionConnector)
from pathlib import Path
import h5py
import logging
from PyNucleus_base import getLoggingTimer

LOGGER = logging.getLogger(__name__)


class fractionalLevel(algebraicLevelBase):
    def __init__(self, meshLevel, buildType):
        self.A = None
        self.S = None
        self.M = None
        super(fractionalLevel, self).__init__(meshLevel, buildType)

    def build(self, buildType):
        super(fractionalLevel, self).build(buildType)

        # diffusivity = self.params['diffusivity']
        symmetric = self.params.get('symmetric', False)
        reorder = self.params.get('reorder', False)
        buildMass = self.params.get('buildMass', False)

        if buildType & SPARSITY_PATTERN and buildMass:
            # set up sparsity patterns only
            DoFMap = self.DoFMap
            mesh = self.meshLevel.mesh
            self.fullyAssembled = False
            with self.Timer('Prepared sparsity patterns'):
                self.M = DoFMap.buildSparsityPattern(mesh.cells,
                                                     symmetric=symmetric,
                                                     reorder=reorder)

        if buildType & ASSEMBLY:
            # fully build matrices
            DoFMap = self.DoFMap
            mesh = self.meshLevel.mesh
            self.fullyAssembled = True
            with self.Timer('Assembled matrices'):
                self.params.pop('mesh', None)
                self.S = getFracLapl(mesh, DoFMap, **self.params)
                self.A = self.S
                # if not s.symmetric:
                #     from PyNucleus_base.linear_operators import Dense_LinearOperator
                #     self.A = Dense_LinearOperator(np.ascontiguousarray(self.A.toarray().T))
                if buildMass:
                    self.M = DoFMap.assembleMass(sss_format=symmetric,
                                                 reorder=reorder)

    def buildCoarserMatrices(self):
        """
        Recursively build matrices on coarser levels
        """
        if self.S is not None and self.P is not None and self.previousLevel.S is not None and not self.previousLevel.fullyAssembled:
            assert self.P.shape[0] == self.S.shape[0], (self.P.shape[0], self.S.shape[0])
            assert self.P.shape[1] == self.previousLevel.S.shape[0]
            with self.Timer('Restrict stiffness matrix'):
                self.P.restrictMatrix(self.S, self.previousLevel.S)
            if self.previousLevel.A is None:
                self.previousLevel.A = self.previousLevel.S
        if self.M is not None and self.P is not None and self.previousLevel.M is not None and not self.previousLevel.fullyAssembled:
            assert self.P.shape[0] == self.M.shape[0]
            assert self.P.shape[1] == self.previousLevel.M.shape[0]
            with self.Timer('Restrict mass matrix'):
                self.P.restrictMatrix(self.M, self.previousLevel.M)
        if self.previousLevel is not None:
            self.previousLevel.fullyAssembled = True
            self.previousLevel.buildCoarserMatrices()

    @classmethod
    def getKeys(cls):
        return algebraicLevelBase.getKeys() + ['A', 'S', 'M']


def paramsForFractionalHierarchy(noRef, global_params):

    noRefCoarse = global_params.get('noRefCoarse', 0)

    if noRefCoarse > 0:
        hierarchies = [
            {'label': 'seed',
             'ranks': set([0]),
             'connectorStart': 'input',
             'connectorEnd': 'breakUp',
             'params': {'noRef': noRefCoarse,
                        'assemble': 'dofmaps only'}
            },
            {'label': 'fine',
             'ranks': set([0]),
             'connectorStart': 'breakUp',
             'connectorEnd': None,
             'params': {'noRef': noRef-noRefCoarse,
                        'keepMeshes': global_params.get('keepMeshes', 'last'),
                        'keepAllDoFMaps': global_params.get('keepAllDoFMaps', False),
                        'assemble': global_params.get('assemble', 'ALL'),
                        'solver': 'LU',
                        'kernel': global_params.get('kernel', None),
                        'genKernel': global_params.get('genKernel', False),
                        'target_order': global_params.get('target_order', None),
                        'rangedOpParams': global_params.get('rangedOpParams', {}),
                        'cached': global_params.get('cached', False),
                        'boundaryCondition': global_params.get('boundaryCondition', HOMOGENEOUS_DIRICHLET),
                        'logging': global_params.get('logging', False)
             }
            }]
        connectors = {}
        connectors['input'] = {'type': inputConnector,
                               'params': {'domain': global_params['domain'],
                                          'meshParams': global_params.get('meshParams', {}),
                                          'algebraicLevelType': fractionalLevel}}
        connectors['breakUp'] = {'type': repartitionConnector,
                                 'params': {'partitionerType': global_params.get('coarsePartitioner', global_params.get('partitioner', 'regular')),
                                            'partitionerParams': global_params.get('coarsePartitionerParams', global_params.get('partitionerParams', {})),
                                            'debugOverlaps': global_params.get('debugOverlaps', False),
                                            'algebraicLevelType': fractionalLevel
                                 }}
    else:
        hierarchies = [
            {'label': 'fine',
             'ranks': set([0]),
             'connectorStart': 'input',
             'connectorEnd': None,
             'params': {'noRef': noRef,
                        'keepMeshes': global_params.get('keepMeshes', 'last'),
                        'keepAllDoFMaps': global_params.get('keepAllDoFMaps', False),
                        'assemble': global_params.get('assemble', 'ALL'),
                        'solver': 'LU',
                        'kernel': global_params.get('kernel', None),
                        'genKernel': global_params.get('genKernel', False),
                        'target_order': global_params.get('target_order', None),
                        'rangedOpParams': global_params.get('rangedOpParams', {}),
                        'cached': global_params.get('cached', False),
                        'boundaryCondition': global_params.get('boundaryCondition', HOMOGENEOUS_DIRICHLET),
                        'logging': global_params.get('logging', False)
             }
            }]
        connectors = {}
        connectors['input'] = {'type': inputConnector,
                               'params': {'domain': global_params['domain'],
                                          'meshParams': global_params.get('meshParams', {}),
                                          'algebraicLevelType': fractionalLevel}}

    return hierarchies, connectors


def fractionalHierarchy(mesh, s, NoRef, tag=None, eta=3.,
                        buildMass=False, dense=False,
                        driftCoeff=None,
                        keepMeshes='finest',
                        keepAllDoFMaps=False,
                        target_order=None, dataDir='DATA',
                        boundaryCondition=HOMOGENEOUS_DIRICHLET,
                        comm=None,
                        forceRebuild=False,
                        horizon=np.inf,
                        errorBound=None):

    global_params = {'domain': mesh,
                     'kernel': getFractionalKernel(mesh.dim, s=s, horizon=np.inf),
                     'horizon': horizon,
                     'tag': tag,
                     'boundaryCondition': boundaryCondition,
                     'eta': eta,
                     'buildMass': buildMass,
                     'dense': dense,
                     'driftCoeff': driftCoeff,
                     'keepMeshes': keepMeshes,
                     'keepAllDoFMaps': keepAllDoFMaps,
                     'interpolationErrorBound': errorBound,
                     'forceRebuild': forceRebuild}
    hierarchies, connectors = paramsForFractionalHierarchy(NoRef, global_params)
    hM = hierarchyManager(hierarchies, connectors, global_params, comm=comm)
    hM.setup()
    return hM


def processBC(tag, boundaryCondition, kernel):
    if tag is None:
        if boundaryCondition == HOMOGENEOUS_DIRICHLET:
            if kernel is not None:
                if kernel.s.max < 0.5:
                    tag = -1
                else:
                    tag = 0
                zeroExterior = True
            else:
                tag = 0
                zeroExterior = -1
        elif boundaryCondition == HOMOGENEOUS_NEUMANN:
            tag = -1
            zeroExterior = False
        elif boundaryCondition == NORM:
            tag = 0
            zeroExterior = kernel.s.max >= 0.5
        else:
            raise NotImplementedError('{}, {}, {}'.format(tag, boundaryCondition, kernel))
    else:
        if boundaryCondition == HOMOGENEOUS_DIRICHLET:
            if kernel is not None:
                zeroExterior = True
            else:
                raise NotImplementedError()
        elif boundaryCondition == HOMOGENEOUS_NEUMANN:
            zeroExterior = False
        elif boundaryCondition == NORM:
            zeroExterior = kernel.s.max >= 0.5
        else:
            raise NotImplementedError('{}, {}, {}'.format(tag, boundaryCondition, kernel))

    # variableOrder = isinstance(s, variableFractionalOrder)
    # if tag is None:
    #     if boundaryCondition == 'Dirichlet':
    #         if isinstance(s, admissibleSet):
    #             tag = 0
    #             zeroExterior = True
    #         elif (variableOrder and (s.max < 0.5)) or (not variableOrder and (s.value < 0.5)):
    #             tag = -1
    #             zeroExterior = True
    #         else:
    #             tag = 0
    #             zeroExterior = True
    #     elif boundaryCondition == 'Neumann':
    #         tag = -1
    #         zeroExterior = False
    #     elif boundaryCondition == 'norm':
    #         tag = 0
    #         zeroExterior = s >= 0.5
    #     else:
    #         raise NotImplementedError()
    # else:
    #     if boundaryCondition == 'Dirichlet':
    #         zeroExterior = True
    #     elif boundaryCondition == 'Neumann':
    #         zeroExterior = False
    #     elif boundaryCondition == 'norm':
    #         zeroExterior = s >= 0.5
    #     else:
    #         raise NotImplementedError()
    # if not ((horizon == np.inf) or
    #         (isinstance(horizon, constant) and horizon.value == np.inf) or
    #         (isinstance(horizon, admissibleSet) and horizon.getLowerBounds()[0] == np.inf)):
    #     if isinstance(horizon, admissibleSet):
    #         tag = 0
    #         zeroExterior = True
    #     else:
    #         tag = -1
    #         zeroExterior = False
    return tag, zeroExterior


def getFracLapl(mesh, DoFMap, kernel=None, rangedOpParams={}, **kwargs):

    assert kernel is not None or 's' in rangedOpParams, (kernel, rangedOpParams)

    boundaryCondition = kwargs.get('boundaryCondition', 'Dirichlet')
    tag = kwargs.get('tag', None)
    zeroExterior = kwargs.get('zeroExterior', None)
    dense = kwargs.get('dense', False)
    diagonal = kwargs.get('diagonal', False)
    cached = kwargs.get('cached', False)
    trySparsification = kwargs.get('trySparsification', False)
    logging = kwargs.get('logging', False)
    timer = kwargs.get('timer', None)

    target_order = kwargs.get('target_order', None)
    eta = kwargs.get('eta', 3.)
    returnNearField = kwargs.get('returnNearField', False)

    comm = kwargs.get('assemblyComm', None)

    dataDir = kwargs.get('dataDir', 'DATA')
    doSave = kwargs.get('doSave', False)
    overrideFileName = kwargs.get('overrideFileName', None)
    forceRebuild = kwargs.get('forceRebuild', False)

    if timer is None:
        timer = getLoggingTimer(LOGGER, comm=comm, rootOutput=True)
        kwargs['timer'] = timer

    if kernel is None:
         raise NotImplementedError()
    else:
        horizon = kernel.horizon
        scaling = kernel.scaling
        normalized = not isinstance(scaling, constantTwoPoint)

    dataDir = Path(dataDir)
    dataDir.mkdir(exist_ok=True, parents=True)

    if tag is None or zeroExterior is None:
        tag, zeroExterior = processBC(tag, boundaryCondition, kernel)

    if overrideFileName is not None:
        filename = overrideFileName
    else:
        base = mesh.vertices_as_array.min(axis=0)
        if diagonal:
            sparseDense = 'diagonal'
        elif dense:
            sparseDense = 'dense'
        else:
            sparseDense = 'sparse'
        filename = dataDir/'{}-{}-{}-{:.5}-{}-{}-{}-{}-{}-{}-{:.5}-{:.5}-{}.hdf5'.format(sparseDense, base, mesh.dim, mesh.diam, mesh.num_vertices, mesh.num_cells, kernel, tag, target_order, eta, mesh.h, mesh.hmin, boundaryCondition)
    A = None
    Pnear = None
    if ((isinstance(kernel, FractionalKernel) and (kernel.s.min == kernel.s.max == 1.)) or
        (isinstance(horizon, constant) and (horizon.value == 0.))):
        with timer('Sparse matrix'):
            if kernel.phi is not None:
                kappa = Lambda(lambda x: kernel.phi(x, x))
            else:
                kappa = None
            A = DoFMap.assembleStiffness(diffusivity=kappa)
    elif isinstance(kernel, FractionalKernel) and (kernel.s.min == kernel.s.max == 0.):
        with timer('Sparse matrix'):
            A = DoFMap.assembleMass()
    elif not forceRebuild and filename.exists():
        if comm is None or comm.rank == 0:
            f = h5py.File(str(filename), 'r')
            if f.attrs['type'] == 'h2':
                A = H2Matrix.HDF5read(f)
            else:
                A = LinearOperator.HDF5read(f)
            f.close()
        else:
            A = None
    else:
        params = {'target_order': target_order,
                  'eta': eta,
                  'forceUnsymmetric': kwargs.get('forceUnsymmetric', False)}
        if 'genKernel' in kwargs:
            params['genKernel'] = kwargs['genKernel']
        if kernel is None:
            kernel = getFractionalKernel(mesh.dim, s, constant(horizon.ranges[0, 0]), scaling=scaling, normalized=normalized)
        builder = nonlocalBuilder(mesh, DoFMap, kernel, params, zeroExterior=zeroExterior, comm=comm, logging=logging)
        if diagonal:
            with timer('Assemble diagonal matrix {}, zeroExterior={}'.format(kernel, zeroExterior)):
                A = builder.getDiagonal()
        elif dense:
            with timer('Assemble dense matrix {}, zeroExterior={}'.format(kernel, zeroExterior)):
                if cached:
                    A = builder.getDenseCached()
                else:
                    A = builder.getDense(trySparsification=trySparsification)
        else:
            with timer('Assemble sparse matrix {}, zeroExterior={}'.format(kernel, zeroExterior)):
                if isinstance(horizon, constant):
                    A, Pnear = builder.getH2(returnNearField=True)
                else:
                    A = builder.getH2FiniteHorizon()
        if doSave and (comm is None or (comm and comm.rank == 0)):
            if hasattr(A, 'HDF5write'):
                with timer('Saving'):
                    try:
                        f = h5py.File(str(filename), 'w')
                        A.HDF5write(f)
                        f.flush()
                        f.close()
                    except OSError as e:
                        LOGGER.warn('Unable to save to {}, reason: {}'.format(str(filename), e))
            # else:
            #     LOGGER.warn('Cannot save {}'.format(str(A)))

    if returnNearField:
        return A, Pnear
    else:
        return A


class delayedNonlocalOp(delayedConstructionOperator):
    def __init__(self, dm, kernel, **kwargs):
        super().__init__(dm.num_dofs,
                         dm.num_dofs)
        self.dm = dm
        self.kernel = kernel
        self.kwargs = kwargs

    def construct(self):
        from copy import copy
        d = copy(self.kwargs)
        d.update(self.params)
        A = self.dm.assembleNonlocal(self.kernel, **d)
        return A


class delayedFractionalLaplacianOp(delayedConstructionOperator):
    def __init__(self, mesh, dm, kernel, **kwargs):
        super().__init__(dm.num_dofs,
                         dm.num_dofs)
        self.mesh = mesh
        self.dm = dm
        self.kernel = kernel
        self.kwargs = kwargs

    def construct(self):
        from copy import copy
        d = copy(self.kwargs)
        d.update(self.params)
        A = getFracLapl(self.mesh, self.dm, self.kernel, **d)
        return A




NONE = -10
DIRICHLET_EXTERIOR = 0
DIRICHLET_INTERIOR = 1


class DirichletCondition:
    def __init__(self, fullMesh, fullDoFMap, fullOp, domainIndicator, fluxIndicator):
        # The mesh is partitioned into
        #  * 'domain'    (domainIndicator > 0)
        #  * 'Neumann'   (fluxIndicator > 0)
        #  * 'Dirichlet' (domainIndicator == 0 and fluxIndicator == 0)
        # For computations, we keep domain and Neumann together as 'natural'.

        self.fullMesh = fullMesh
        self.fullDoFMap = fullDoFMap
        self.domainIndicator = domainIndicator
        self.fluxIndicator = fluxIndicator
        self.fullOp = fullOp
        self.setup()

    def setup(self):
        # from PyNucleus_fem import constant

        # dmIndicator = P0_DoFMap(self.fullMesh)
        dirichletIndicator = constant(1.)-self.domainIndicator-self.fluxIndicator
        # dirichletIndicatorVec = dmIndicator.interpolate(dirichletIndicator).toarray()
        # naturalCells = np.flatnonzero(dirichletIndicatorVec < 1e-9).astype(INDEX)

        from PyNucleus_fem.splitting import meshSplitter, dofmapSplitter
        from PyNucleus_fem.DoFMaps import getSubMapRestrictionProlongation

        split = dofmapSplitter(self.fullDoFMap, {'Dirichlet': dirichletIndicator})
        self.dirichletDoFMap = split.getSubMap('Dirichlet')
        self.dirichletR, self.dirichletP = split.getRestrictionProlongation('Dirichlet')
        self.naturalDoFMap = self.dirichletDoFMap.getComplementDoFMap()
        self.naturalR, self.naturalP = getSubMapRestrictionProlongation(self.fullDoFMap, self.naturalDoFMap)

        # self.naturalMesh = getSubmesh(self.fullMesh, naturalCells)
        # self.naturalMesh.replaceBoundaryVertexTags(lambda x: DIRICHLET_EXTERIOR if dirichletIndicator(x) >= 1e-9 else DIRICHLET_INTERIOR,
        #                                            set([DIRICHLET_EXTERIOR]))
        # self.naturalMesh.replaceBoundaryEdgeTags(lambda x, y: DIRICHLET_EXTERIOR if dirichletIndicator(0.5*(np.array(x)+np.array(y))) >= 1e-9 else DIRICHLET_INTERIOR,
        #                                          set([DIRICHLET_EXTERIOR]))

        # self.naturalDoFMap = type(self.fullDoFMap)(self.fullMesh, self.domainIndicator+self.fluxIndicator)
        # self.naturalR, self.naturalP = getSubMapRestrictionProlongation(self.fullDoFMap, self.naturalDoFMap)

        # self.dirichletDoFMap = type(self.fullDoFMap)(self.fullMesh, dirichletIndicator)
        # self.dirichletR, self.dirichletP = getSubMapRestrictionProlongation(self.fullDoFMap, self.dirichletDoFMap)


        # import matplotlib.pyplot as plt
        # plt.figure()
        # self.dirichletDoFMap.plot()
        # # self.dirichletMesh.plot(info=True)
        # plt.figure()
        # self.naturalDoFMap.plot()
        # # self.naturalMesh.plot(info=True)
        # plt.show()

        assert self.fullDoFMap.num_dofs == self.naturalDoFMap.num_dofs+self.dirichletDoFMap.num_dofs, (self.fullDoFMap.num_dofs, self.naturalDoFMap.num_dofs, self.dirichletDoFMap.num_dofs)

        self.naturalA = self.naturalR*(self.fullOp*self.naturalP)

        self.domainDoFMap = type(self.fullDoFMap)(self.fullMesh, self.domainIndicator)
        self.domainR, self.domainP = getSubMapRestrictionProlongation(self.fullDoFMap, self.domainDoFMap)

    def setDirichletData(self, dirichletData):
        if self.dirichletDoFMap.num_dofs > 0:
            self.dirichletVector = self.dirichletDoFMap.interpolate(dirichletData)

    def applyRHScorrection(self, b):
        assert b.shape[0] == self.naturalDoFMap.num_dofs
        if self.dirichletDoFMap.num_dofs > 0:
            b -= self.naturalR*(self.fullOp*(self.dirichletP*self.dirichletVector))
            # b -= self.naturalR*(self.domainP*(self.domainR*(self.fullOp*(self.dirichletP*self.dirichletVector))))

    def augmentDirichlet(self, u):
        return self.naturalP*u + self.dirichletP*self.dirichletVector

    def plot(self):
        if self.fullMesh.dim == 1:
            x = self.dirichletP*self.dirichletDoFMap.ones() + 2*(self.naturalP*self.naturalDoFMap.ones())
            self.fullMesh.plotFunction(x)
        else:
            raise NotImplementedError()


class multilevelDirichletCondition(DirichletCondition):
    def __init__(self, levels, domainIndicator, fluxIndicator):
        super(multilevelDirichletCondition, self).__init__(levels[-1]['mesh'],
                                                           levels[-1]['DoFMap'],
                                                           levels[-1]['A'],
                                                           domainIndicator,
                                                           fluxIndicator)
        self.levels = levels
        self.setupHierarchy()

    def setupCoarseOps(self, mesh, dm):
        from PyNucleus_fem import constant

        dmIndicator = P0_DoFMap(mesh)
        dirichletIndicator = constant(1.)-self.domainIndicator-self.fluxIndicator
        dirichletIndicatorVec = dmIndicator.interpolate(dirichletIndicator).toarray()
        naturalCells = np.flatnonzero(dirichletIndicatorVec < 1e-9).astype(INDEX)

        naturalMesh = getSubmesh(mesh, naturalCells)
        naturalMesh.replaceBoundaryVertexTags(lambda x: DIRICHLET_EXTERIOR if dirichletIndicator(x) >= 1e-9 else DIRICHLET_INTERIOR,
                                              set([DIRICHLET_EXTERIOR]))
        naturalMesh.replaceBoundaryEdgeTags(lambda x, y: DIRICHLET_EXTERIOR if dirichletIndicator(0.5*(np.array(x)+np.array(y))) >= 1e-9 else DIRICHLET_INTERIOR,
                                            set([DIRICHLET_EXTERIOR]))

        naturalDoFMap = type(dm)(mesh, self.domainIndicator+self.fluxIndicator)
        naturalR, naturalP = getSubMapRestrictionProlongation(dm, naturalDoFMap)

        return naturalMesh, naturalDoFMap, naturalR, naturalP

    def setupHierarchy(self):
        levelsNew = []
        prevNaturalR, prevNaturalP = None, None
        for lvl in range(len(self.levels)):
            levelsNew.append({})
            naturalMesh, naturalDoFMap, naturalR, naturalP = self.setupCoarseOps(self.levels[lvl]['mesh'],
                                                                                 self.levels[lvl]['DoFMap'])
            for key in self.levels[lvl]:
                if key == 'A':
                    levelsNew[lvl][key] = naturalR*(self.levels[lvl][key]*naturalP)
                    levelsNew[lvl][key].diagonal = naturalR*self.levels[lvl][key].diagonal
                elif key == 'S':
                    levelsNew[lvl][key] = naturalR*(self.levels[lvl][key]*naturalP)
                    levelsNew[lvl][key].diagonal = naturalR*self.levels[lvl][key].diagonal
                elif key == 'M':
                    levelsNew[lvl][key] = naturalR*(self.levels[lvl][key]*naturalP)
                    levelsNew[lvl][key].diagonal = naturalR*self.levels[lvl][key].diagonal
                elif key == 'R':
                    levelsNew[lvl][key] = (prevNaturalR*(self.levels[lvl][key]*naturalP)).to_csr_linear_operator()
                elif key == 'P':
                    levelsNew[lvl][key] = (naturalR*(self.levels[lvl][key]*prevNaturalP)).to_csr_linear_operator()
                elif key == 'DoFMap':
                    levelsNew[lvl][key] = naturalDoFMap
                elif key == 'mesh':
                    levelsNew[lvl][key] = naturalMesh
                else:
                    levelsNew[lvl][key] = self.levels[lvl][key]
            levelsNew[lvl]['naturalR'] = naturalR
            levelsNew[lvl]['naturalP'] = naturalP
            prevNaturalR, prevNaturalP = naturalR, naturalP
        self.naturalLevels = levelsNew


