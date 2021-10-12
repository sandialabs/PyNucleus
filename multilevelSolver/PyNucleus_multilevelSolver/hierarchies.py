###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
import logging
import numpy as np
from copy import deepcopy
from PyNucleus_base.myTypes import REAL
from PyNucleus_base import TimerManager, updateFromDefaults
from PyNucleus_fem import boundaryLayer
from PyNucleus_fem.algebraicOverlaps import multilevelAlgebraicOverlapManager
from . levels import meshLevel, algebraicLevel
from . levels import (DELETE_MESH, KEEP_MESH,
                      SPARSITY_PATTERN, DOFMAPS,
                      NO_BUILD, RESTRICTION_PROLONGATION_ONLY,
                      SPARSITY_ONLY, SINGLE_LEVEL, FULL_BUILD)

LOGGER = logging.getLogger(__name__)


class EmptyHierarchy(object):
    def __init__(self, params, label=''):
        self.params = params
        self.updateParamsFromDefaults()
        self.label = label
        self.connectorEnd = None

    def isSetUp(self):
        return False

    def updateParamsFromDefaults(self):
        defaults = {}
        updateFromDefaults(self.params, defaults)


class hierarchy:
    def __init__(self, meshLevel, params, comm=None,
                 label=''):
        self._isSetUp = False
        self.connectorStart = None
        self.connectorEnd = None
        self.params = params
        self.updateParamsFromDefaults(self.params)
        self.comm = comm
        self.label = label

        self.Timer = TimerManager(LOGGER, comm=self.comm, prefix=self.label+': ')

        if self.params['keepMeshes'] == 'all':
            self.meshInformationPolicy = [KEEP_MESH]*self.params['noRef'] + [KEEP_MESH]
        elif self.params['keepMeshes'] == 'last':
            self.meshInformationPolicy = [DELETE_MESH]*self.params['noRef'] + [KEEP_MESH]
        elif self.params['keepMeshes'] == 'none':
            self.meshInformationPolicy = [DELETE_MESH]*self.params['noRef'] + [DELETE_MESH]
        else:
            raise NotImplementedError()

        if self.params['assemble'] == 'all':
            if self.params['meshTransformation'] is None:
                self.buildType = [SPARSITY_ONLY]*self.params['noRef'] + [FULL_BUILD]
            else:
                self.buildType = [FULL_BUILD]*self.params['noRef'] + [FULL_BUILD]
        elif self.params['assemble'] == 'ALL':
            self.buildType = [FULL_BUILD]*self.params['noRef'] + [FULL_BUILD]
        elif self.params['assemble'] == 'last':
            self.buildType = [RESTRICTION_PROLONGATION_ONLY]*self.params['noRef'] + [FULL_BUILD]
        elif self.params['assemble'] == 'first+last':
            self.buildType = [FULL_BUILD]+[RESTRICTION_PROLONGATION_ONLY]*(self.params['noRef']-1) + [FULL_BUILD]
        elif self.params['assemble'] == 'dofmaps only':
            self.buildType = [DOFMAPS]*(self.params['noRef']+1)
        elif self.params['assemble'] == 'none':
            self.buildType = [NO_BUILD]*(self.params['noRef']+1)
        elif self.params['assemble'] == 'restrictionProlongation':
            self.buildType = [RESTRICTION_PROLONGATION_ONLY]*(self.params['noRef']+1)
        else:
            raise NotImplementedError()

        if 'buildInteriorBL' in params and params['buildInteriorBL']:
            meshLevel.interiorBL = boundaryLayer(meshLevel.mesh, params['depth']+1,
                                                 afterRefinements=params['noRef'])

        if meshLevel is not None:
            self.meshLevels = [meshLevel]
            self.algebraicLevels = [self.meshLevels[-1].getAlgebraicLevel(self.buildType[0])]
        else:
            self.meshLevels = []
            self.algebraicLevels = []
        self.multilevelAlgebraicOverlapManager = None

    @staticmethod
    def updateParamsFromDefaults(params):
        defaults = {
            'keepMeshes': 'last',
            'assemble': 'all',
            'depth': 0,
            'noRef': 0,
            'buildInteriorBL': False,
            'debugOverlaps': False,
            'meshTransformation': None,
            'diffusivity': None,
            'reaction': None,
            'symmetric': False,
            'reorder': False,
            'buildMass': False,
            'element': 'P1',
            'commType': 'standard',
            'keepAllDoFMaps': False,
            'interiorBC': 'homogeneousNeumann',
            'buildSurfaceMass': False,
            'buildSurfaceStiffness': False,
            'overlapMatvec': False,
            'meshTransformation': None,
            'debugOverlaps': False
        }
        updateFromDefaults(params, defaults)

    def refine(self, isLastLevel=False):
        # refine mesh level
        meshInformationPolicy = self.meshInformationPolicy[self.meshLevels[-1].levelNo+1-self.meshLevels[0].levelNo]
        self.meshLevels.append(self.meshLevels[-1].refine(meshInformationPolicy))
        self.meshLevels[-1].isLastLevel = isLastLevel

        # build algebraic level
        buildType = self.buildType[self.meshLevels[-1].levelNo-self.meshLevels[0].levelNo]
        self.algebraicLevels.append(self.meshLevels[-1].getAlgebraicLevel(buildType))

        # clean up unneeded data
        if len(self.meshLevels) > 1:
            self.meshLevels[-2].clean()
            self.algebraicLevels[-2].clean()

    def build(self):
        for k in range(self.params['noRef']):
            self.refine(k == self.params['noRef']-1)
        self.algebraicLevels[-1].buildCoarserMatrices()

        if self.algebraicLevels[-1].algebraicOverlaps is not None:
            with self.Timer("Build multilevel overlaps"):
                if False:
                    raise NotImplementedError()
                
                else:
                    multLvlAlgOvManager = multilevelAlgebraicOverlapManager(self.comm)
                    for lvl in range(len(self.algebraicLevels)):
                        multLvlAlgOvManager.levels.append(self.algebraicLevels[lvl].algebraicOverlaps)
                        if self.buildType[lvl] & DOFMAPS:
                            multLvlAlgOvManager.levels[lvl].prepareDistribute()

                    if self.params['debugOverlaps']:
                        from PyNucleus_fem import solSin1D, solSin2D, solSin3D
                        for lvl in range(len(self.algebraicLevels)):
                            if self.algebraicLevels[lvl].DoFMap is not None:
                                dm = self.algebraicLevels[lvl].DoFMap
                                if self.meshLevels[-1].mesh.dim == 1:
                                    x = dm.interpolate(solSin1D)
                                elif self.meshLevels[-1].mesh.dim == 2:
                                    x = dm.interpolate(solSin2D)
                                elif self.meshLevels[-1].mesh.dim == 3:
                                    x = dm.interpolate(solSin3D)
                                else:
                                    raise NotImplementedError()
                                y = np.zeros((dm.num_dofs), dtype=REAL)
                                y[:] = x
                                multLvlAlgOvManager.levels[lvl].distribute_py(y)
                                multLvlAlgOvManager.levels[lvl].accumulate_py(y)
                                assert np.linalg.norm(x-y) < 1e-9, (x, y)
                self.multilevelAlgebraicOverlapManager = multLvlAlgOvManager
        self._isSetUp = True

    def isSetUp(self):
        return self._isSetUp

    def getLevelList(self, recurse=True):
        if self.connectorStart is not None and recurse:
            levels = self.connectorStart.getLevelList()
        else:
            levels = []
        levelsMesh = [mL.getLevelDict() for mL in self.meshLevels]
        levelsAlg = [aL.getLevelDict() for aL in self.algebraicLevels]
        for i in range(len(levelsAlg)):
            levelsAlg[i].update(levelsMesh[i])
        levelsAlg = levels[:-1]+levelsAlg

        if self.multilevelAlgebraicOverlapManager is not None:
            levelsAlg[-1]['multilevelAlgebraicOverlapManager'] = self.multilevelAlgebraicOverlapManager
        return levelsAlg

    @staticmethod
    def fromLevelList(levels, params={}, comm=None, label=''):
        hierarchy.updateParamsFromDefaults(params)
        params['assemble'] = 'none'
        meshLevels = []
        prevMeshLevel = None
        algebraicLevels = []
        for lvl in levels:
            meshLevels.append(meshLevel.fromLevelDict(lvl, params=params, previousLevel=prevMeshLevel, comm=comm, label=label))
            prevMeshLevel = meshLevels[-1]
            algebraicLevels.append(algebraicLevel.fromLevelDict(prevMeshLevel, lvl))
        h = hierarchy(meshLevels[0], params, comm=comm)
        h.meshLevels = meshLevels
        h.algebraicLevels = algebraicLevels
        h._isSetUp = True
        return h

    def buildCollapsedRestrictionProlongation(self):
        self.P = self.algebraicLevels[1].P
        for lvlNo in range(2, len(self.algebraicLevels)):
            self.P = self.algebraicLevels[lvlNo].P*self.P
        self.P = self.P.to_csr_linear_operator()
        self.R = self.P.transpose()

    def gatherInformation(self, root=0):
        import platform

        subdomain = self.meshLevels[-1].mesh
        A = self.algebraicLevels[-1].A
        overlaps = self.multilevelAlgebraicOverlapManager
        info = {}
        info['numberVertices'] = self.comm.gather(subdomain.num_vertices, root=root)
        info['numberCells'] = self.comm.gather(subdomain.num_cells, root=root)
        info['numberDoFs'] = self.comm.gather(A.shape[0], root=root)
        if self.comm.size > 1:
            info['globalNumDoFs'] = overlaps.countDoFs()
            info['numberSharedDoFs'] = self.comm.gather(overlaps.get_num_shared_dofs(unique=False), root=root)
            info['maxCross'] = overlaps.levels[-1].max_cross
            neighbors = [subdomainNo for subdomainNo in overlaps.levels[-1].overlaps]
        else:
            info['globalNumDoFs'] = A.shape[0]
            info['numberSharedDoFs'] = [0]
            info['maxCross'] = [0]
            neighbors = []
        info['nnz'] = self.comm.gather(A.nnz, root=root)
        hostname = platform.node()
        info['hostnames'] = self.comm.gather(hostname, root=root)
        info['neighbors'] = self.comm.gather(neighbors, root=root)
        info['rank'] = self.comm.gather(MPI.COMM_WORLD.rank, root=root)
        return info

    def __len__(self):
        return len(self.meshLevels)

    def getSubHierarchy(self, numLevels):
        assert 0 <= numLevels < len(self)

        h = hierarchy(None, self.params, self.comm, self.label)
        h.connectorStart = self.connectorStart
        h.connectorEnd = self.connectorEnd
        h.meshLevels = self.meshLevels[:numLevels+1]
        h.algebraicLevels = self.algebraicLevels[:numLevels+1]
        h.multilevelAlgebraicOverlapManager = self.multilevelAlgebraicOverlapManager
        return h



class hierarchyManager(object):
    def __init__(self, hierarchyDefs, connectorDefs, params, comm=None, doDeepCopy=True):
        if doDeepCopy:
            self.hierarchies = deepcopy(hierarchyDefs)
            self.connectors = deepcopy(connectorDefs)
        else:
            self.hierarchies = hierarchyDefs
            self.connectors = connectorDefs
        self.params = params
        if comm is None:
            from PyNucleus_base.utilsCy import FakeComm
            comm = FakeComm(0, 1)
        self.comm = comm
        for h in self.hierarchies:
            updateFromDefaults(h['params'], self.params)
        self._printRank = -1

    def getPrintRank(self):
        if self._printRank == -1:
            self._printRank = self.comm.allreduce(self.comm.rank if not isinstance(self.builtHierarchies[-1], EmptyHierarchy) else self.comm.size, op=MPI.MIN)
        return self._printRank

    def setCommunicators(self):
        for k in range(len(self.hierarchies)):
            h = self.hierarchies[k]
            if k == 0 or h['ranks'] != self.hierarchies[k-1]:
                if (self.comm is not None) and (len(h['ranks']) < self.comm.size):
                    if self.comm.rank in h['ranks']:
                        h['comm'] = self.comm.Split(0)
                    else:
                        self.comm.Split(MPI.UNDEFINED)
                        h['comm'] = None
                else:
                    h['comm'] = self.comm
            else:
                h['comm'] = self.hierarchies[k-1]['comm']
            if h['connectorEnd'] is not None:
                self.connectors[h['connectorEnd']]['comm1'] = h['comm']
            if h['connectorStart'] is not None:
                self.connectors[h['connectorStart']]['comm2'] = h['comm']

        for conn in sorted(self.connectors):
            c = self.connectors[conn]
            if 'comm1' in c:
                if c['comm1'] is not None or c['comm2'] is not None:
                    c['global_comm'] = self.comm.Split(0)
                else:
                    self.comm.Split(MPI.UNDEFINED)
                    c['global_comm'] = None
            else:
                c['comm1'] = None
                c['global_comm'] = c['comm2']

    def buildHierarchies(self):
        builtHierarchies = []
        builtConnectors = {}
        currentHierarchy = None
        for k in range(len(self.hierarchies)):
            h = self.hierarchies[k]
            c_params = self.connectors[h['connectorStart']]
            self.comm.Barrier()
            if c_params['global_comm'] is not None:
                connector = c_params['type'](c_params['global_comm'], c_params['comm1'], c_params['comm2'], currentHierarchy, **c_params['params'])
                currentHierarchy = connector.getNewHierarchy(h)
                builtConnectors[h['connectorStart']] = connector
                builtHierarchies.append(currentHierarchy)
            else:
                currentHierarchy = EmptyHierarchy(h['params'], label=h['label'])
                builtHierarchies.append(currentHierarchy)
            if c_params['global_comm'] is not None:
                connector.build()
            if h['comm'] is not None:
                currentHierarchy.build()
        self.builtHierarchies = builtHierarchies
        self.builtConnectors = builtConnectors

    def setup(self):
        self.setCommunicators()
        self.buildHierarchies()

    def display(self, info=False):
        msg = []
        if self.comm.rank == 0:
            msg.append('{:30} {}'.format('', ' '.join([str(i) for i in range(self.comm.size)])))
        h = self.hierarchies[0]
        if h['connectorStart'] is not None:
            conn = h['connectorStart']
            t = self.comm.gather(self.connectors[conn]['global_comm'] is not None)
            if self.comm.rank == 0:
                msg.append('{:30} {}'.format(conn, ' '.join(["-" if tt else " " for tt in t])))
        for k, h in enumerate(self.hierarchies):
            t = self.comm.gather(h['comm'] is not None, root=min(h['ranks']))
            if self.comm.rank == min(h['ranks']):
                msg2 = []
                for j in range(len(self.builtHierarchies[k].meshLevels)):
                    l = self.builtHierarchies[k].meshLevels[j]
                    msg2.append('{:30} {}'.format(l.levelID, ' '.join(["o" if tt else " " for tt in t])))
                    if info:
                        algLevel = self.builtHierarchies[k].algebraicLevels[j]
                        msg2[-1] += '  '
                        keys = algLevel.getKeys()
                        msg2[-1] += ' '.join(key for key in keys if getattr(algLevel, key) is not None)
                msg2 = '\n'.join(msg2)
                self.comm.send(msg2, dest=0, tag=7767)
            if self.comm.rank == 0:
                s2 = self.comm.recv(source=min(h['ranks']), tag=7767)
                msg.append(s2)
            if h['connectorEnd'] is not None:
                conn = h['connectorEnd']
                if self.connectors[conn]['comm1'] is not None and self.connectors[conn]['comm2'] is None:
                    # symbol = '┴'
                    symbol = '-'
                elif self.connectors[conn]['comm1'] is not None and self.connectors[conn]['comm2'] is not None:
                    # symbol = '┼'
                    symbol = '-'
                elif self.connectors[conn]['comm1'] is None and self.connectors[conn]['comm2'] is not None:
                    # symbol = '┬'
                    symbol = '-'
                else:
                    symbol = ' '
                t = self.comm.gather(symbol)
                if self.comm.rank == 0:
                    s = t[0]
                    for i in range(1, len(t)):
                        if t[i-1] != ' ' and t[i] != ' ':
                            # s += '─' + t[i]
                            s += '-' + t[i]
                        else:
                            s += ' ' + t[i]
                    msg.append('{:30} {}'.format(conn, s))
        if self.comm.rank == 0:
            LOGGER.info('\n' + '\n'.join(msg))

    def getLevelList(self):
        k = len(self.builtHierarchies)-1
        while self.builtHierarchies[k] is None:
            k -= 1
        return self.builtHierarchies[k].getLevelList()

    @staticmethod
    def fromLevelList(levels, params={}, comm=None):
        # TODO: Assumes single rank so far
        if comm is None:
            comm = MPI.COMM_SELF
            comm = None
        hierarchyDefs = [{'label': 'fine',
                          'ranks': set([0]),
                          'connectorStart': None,
                          'connectorEnd': None,
                          'params': {'solver': 'LU'}}]
        connectorDefs = {}
        hM = hierarchyManager(hierarchyDefs, connectorDefs, params, comm)
        hM.setCommunicators()
        hM.builtHierarchies = [hierarchy.fromLevelList(levels, params=hierarchyDefs[0]['params'], comm=comm, label=hierarchyDefs[0]['label'])]
        hM.builtConnectors = {}
        return hM

    def getComm(self):
        k = len(self.builtHierarchies)-1
        while self.builtHierarchies[k] is None:
            k -= 1
        return self.builtHierarchies[k].comm

    def hierarchyIsSetUp(self, label):
        for h in self.builtHierarchies:
            if h is not None and h.label == label:
                return h.isSetUp()
        return False

    def getHierarchy(self, label):
        for h in self.builtHierarchies:
            if h is not None and h.label == label:
                return h
        return None

    def __getitem__(self, label):
        return self.getHierarchy(label)

    def getSubManager(self, label=None):
        if label is not None:
            for k, h in enumerate(self.builtHierarchies):
                if h is not None and h.label == label:
                    subManager = hierarchyManager(self.hierarchies[:k+1], self.connectors, self.params, self.comm, doDeepCopy=False)
                    subManager.builtHierarchies = self.builtHierarchies[:k+1]
                    subManager.builtConnectors = self.builtConnectors
                    return subManager
            raise Exception()
        else:
            k = len(self.hierarchies)-2
            subManager = hierarchyManager(self.hierarchies[:k+1], self.connectors, self.params, self.comm, doDeepCopy=False)
            subManager.builtHierarchies = self.builtHierarchies[:k+1]
            subManager.builtConnectors = self.builtConnectors
            return subManager

    def collectInformation(self, hierarchies, root=-1):
        if root == -1:
            root = self.getPrintRank()
        info = {}
        tag = 263
        req = []
        for label in hierarchies:
            if not isinstance(self[label], EmptyHierarchy):
                i = self[label].gatherInformation(root=0)
                if self[label].comm.rank == 0:
                    req.append(self.comm.isend(i, dest=root, tag=tag))
            if self.comm.rank == root:
                info[label] = self.comm.recv(source=MPI.ANY_SOURCE, tag=tag)
            tag += 1
        MPI.Request.Waitall(req)
        return info

    def getSubHierarchy(self, numFineLevels):
        hM = hierarchyManager(self.hierarchies, self.connectors, self.params, self.comm, doDeepCopy=False)
        hM.builtHierarchies = self.builtHierarchies[:-1]
        hM.builtConnectors = self.builtConnectors
        h = self.builtHierarchies[-1].getSubHierarchy(numFineLevels)
        hM.builtHierarchies.append(h)

        return hM
