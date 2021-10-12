###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from __future__ import division
import logging
import numpy as np
from PyNucleus_base.ip_norm import norm_serial as norm
from PyNucleus_base import getLoggingTimer
from PyNucleus_base import REAL, INDEX, uninitialized
from PyNucleus_fem import P1_DoFMap
from PyNucleus_base.linear_operators import LinearOperator

LOGGER = logging.getLogger(__name__)


def paramsForSerialMG(noRef, global_params):
    symmetric = global_params.get('symmetric', False)
    hierarchies = [
        {'label': 'fine',
         'ranks': set([0]),
         'connectorStart': 'input',
         'connectorEnd': None,
         'params': {'noRef': noRef,
                    'keepMeshes': 'all' if global_params.get('keepMeshes', False) else 'none',
                    'keepAllDoFMaps': global_params.get('keepAllDoFMaps', False),
                    'assemble': 'all',
                    'symmetric': symmetric,
                    'solver': 'Chol' if symmetric else 'LU'
         }
        }]
    connectors = {}

    return hierarchies, connectors


def paramsForMG(noRef, onRanks, global_params, dim, element, repartitionFactor=0.05,
                max_coarse_grid_size=5000):
    from . connectors import repartitionConnector

    numProcsAvail = len(onRanks)
    onRanks = np.array(list(onRanks), dtype=INDEX)
    if dim == 1:
        numInitialCells = 2
        if element in ('P1', 1):
            cells2dofsFactor = 1
        elif element in ('P2', 2):
            cells2dofsFactor = 2
        elif element in ('P3', 3):
            cells2dofsFactor = 3
        else:
            raise NotImplementedError()
    elif dim == 2:
        numInitialCells = 8
        if element in ('P1', 1):
            cells2dofsFactor = 0.5
        elif element in ('P2', 2):
            cells2dofsFactor = 2
        elif element in ('P3', 3):
            cells2dofsFactor = 4.5
        else:
            raise NotImplementedError()
    elif dim == 3:
        numInitialCells = 6
        if element in ('P1', 1):
            cells2dofsFactor = 1./6.
        elif element in ('P2', 2):
            cells2dofsFactor = 1.35
        elif element in ('P3', 3):
            cells2dofsFactor = 4.5
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    uniformRefinementMutiplier = 2**dim
    numCells = numInitialCells * uniformRefinementMutiplier**np.arange(noRef+1)
    cg = 0
    while numCells[cg+1]*cells2dofsFactor < max_coarse_grid_size and cg < noRef-1:
        cg += 1
    cellsPerProc = numCells[-1]/numProcsAvail
    numProcs = uninitialized((noRef+1),dtype=int)
    numProcs[-1] = numProcsAvail
    numProcs[:cg+1] = 1
    for i in range(noRef-1, cg, -1):
        if numCells[i]/numProcs[i+1] < repartitionFactor * cellsPerProc:
            numProcs[i] = int(np.ceil(numCells[i]/cellsPerProc))
        else:
            numProcs[i] = numProcs[i+1]

    buildMass = global_params.get('buildMass', False)
    symmetric = global_params.get('symmetric', False)
    reaction = global_params.get('reaction', None)

    hierarchies = [
        {'label': 'seed',
         'ranks': set([onRanks[0]]),
         'connectorStart': 'input',
         'connectorEnd': None,
         'params': {'noRef': cg,
                    'keepMeshes': 'all' if global_params.get('keepMeshes', False) else 'none',
                    'assemble': 'all',
                    'symmetric': symmetric,
                    'reaction': reaction,
                    'buildMass': buildMass,
                    'element': element,
                    'solver': 'Chol' if symmetric else 'LU',
                    'solver_params': {}
         }
        }]

    lvl = cg+1
    hierarchies.append({'label': str(len(hierarchies)),
                        'ranks': set(onRanks[range(0, numProcs[lvl])]),
                        'connectorStart': None,
                        'connectorEnd': None,
                        'params': {'noRef': 1,
                                   'keepMeshes': 'all' if global_params.get('keepMeshes', False) else 'last',
                                   'assemble': 'all',
                                   'symmetric': symmetric,
                                   'reaction': reaction,
                                   'buildMass': buildMass,
                                   'element': element,
                                   'solver': 'MG',
                                   'solver_params': {
                                       'maxIter': 1,
                                       'tolerance': 0.,
                                   },
                        }
    })

    lvl += 1
    while lvl < noRef:
        if numProcs[lvl] == numProcs[lvl-1]:
            hierarchies[-1]['params']['noRef'] += 1
        else:
            hierarchies.append({'label': str(len(hierarchies)),
                                'ranks': set(onRanks[range(0, numProcs[lvl])]),
                                'connectorStart': None,
                                'connectorEnd': None,
                                'params': {'noRef': 1,
                                           'keepMeshes': 'all' if global_params.get('keepMeshes', False) else 'last',
                                           'assemble': 'all',
                                           'symmetric': symmetric,
                                           'reaction': reaction,
                                           'buildMass': buildMass,
                                           'element': element,
                                           'solver': 'MG',
                                           'solver_params': {
                                               'maxIter': 1,
                                               'tolerance': 0.,
                                           },
                                }
            })
        lvl +=1

    if 'tag' in global_params:
        for i in range(len(hierarchies)):
            h = hierarchies[i]
            h['params']['tag'] = global_params['tag']

    connectors = {}
    for i in range(1, len(hierarchies)):
        label = 'breakUp_' + hierarchies[i-1]['label'] + ':' + hierarchies[i]['label']
        connectors[label] = {'type': repartitionConnector,
                             'params': {'partitionerType': global_params.get('coarsePartitioner', global_params.get('partitioner', 'regular')),
                                        'partitionerParams': global_params.get('coarsePartitionerParams', global_params.get('partitionerParams', {})),
                                        'debugOverlaps': global_params.get('debugOverlaps', False)}}
        hierarchies[i-1]['connectorEnd'] = label
        hierarchies[i]['connectorStart'] = label

    return hierarchies, connectors



def writeToHDF(filename, levels, mesh):
    import h5py
    f = h5py.File(filename, 'w')
    for i, lvl in enumerate(levels):
        for key in lvl:
            if key in ('P', 'R', 'A', 'mesh'):
                val = lvl[key]
                grp = f.create_group(str(i) + '/' + key)
                val.HDF5write(grp)
    if 'mesh' not in f[str(i)]:
        grp = f.create_group(str(i) + '/' + 'mesh')
        mesh.HDF5write(grp)
    f.flush()
    f.close()


def readFromHDF(filename):
    import h5py
    f = h5py.File(filename, 'r')
    LOGGER.info('Reading hierarchy from {}'.format(filename))
    maxLvl = 0
    for lvl in f:
        maxLvl = max(maxLvl, int(lvl))
    levels = [{} for i in range(maxLvl+1)]
    for lvl in f:
        for key in f[lvl]:
            if key in ('P', 'R', 'A'):
                levels[int(lvl)][key] = LinearOperator.HDF5read(f[lvl + '/' + key])
    return levels
