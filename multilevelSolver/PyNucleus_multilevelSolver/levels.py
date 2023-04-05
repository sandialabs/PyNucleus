###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import logging
import numpy as np
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from PyNucleus_fem.DoFMaps import P1_DoFMap
from PyNucleus_fem import str2DoFMap
from PyNucleus_base.myTypes import REAL
from PyNucleus_base.linear_operators import CSR_LinearOperator
from PyNucleus_base.linear_operators import SSS_LinearOperator
from . restrictionProlongation import buildRestrictionProlongation
from PyNucleus_base.utilsFem import TimerManager
from PyNucleus_base.ip_norm import (ip_serial, norm_serial,
                                    ip_distributed, norm_distributed,
                                    wrapRealInnerToComplex, wrapRealNormToComplex)
from PyNucleus_fem.femCy import assembleMatrix
from PyNucleus_fem.functions import function
from PyNucleus_fem.distributed_operators import (DistributedLinearOperator,
                                                 CSR_DistributedLinearOperator)
from PyNucleus_fem import (DIRICHLET, NEUMANN,
                           HOMOGENEOUS_DIRICHLET, HOMOGENEOUS_NEUMANN,
                           boundaryConditions)
from PyNucleus_fem.femCy import stiffness_1d_in_2d_sym_P1
from PyNucleus_fem.mesh import (PHYSICAL, NO_BOUNDARY, INTERIOR_NONOVERLAPPING, INTERIOR)
LOGGER = logging.getLogger(__name__)

# what should be built
DOFMAPS = 1
RESTRICTION_PROLONGATION = 2
SPARSITY_PATTERN = 4
OVERLAPS = 8
ASSEMBLY = 16

NO_BUILD = 0
DOFMAPS_ONLY = DOFMAPS
RESTRICTION_PROLONGATION_ONLY = DOFMAPS + RESTRICTION_PROLONGATION
SPARSITY_ONLY = DOFMAPS + OVERLAPS + RESTRICTION_PROLONGATION + SPARSITY_PATTERN
SINGLE_LEVEL = DOFMAPS + OVERLAPS + ASSEMBLY
FULL_BUILD = DOFMAPS + OVERLAPS + RESTRICTION_PROLONGATION + ASSEMBLY

# What information is retained in meshLevels
DELETE_MESH = 0
KEEP_MESH = 1


class level:
    def __init__(self, params, previousLevel=None,
                 comm=None, label='', startLevelNo=0,
                 isLastLevel=False):
        self.params = params
        self.previousLevel = previousLevel
        if previousLevel is not None:
            assert not previousLevel.isLastLevel
        self.startLevelNo = startLevelNo
        self.nextLevel = None
        self.comm = comm
        self.label = label
        self.isLastLevel = isLastLevel

        label = '{}: '.format(self.levelID)
        self.Timer = TimerManager(LOGGER,
                                  comm=self.comm, prefix=label)

    def getLevelNo(self):
        if self.previousLevel is None:
            return self.startLevelNo
        else:
            return self.previousLevel.getLevelNo()+1

    levelNo = property(fget=getLevelNo)

    def getLevelID(self):
        if len(self.label) > 0:
            label = '{} {}'.format(self.label, self.levelNo)
        else:
            label = 'Level {}'.format(self.levelNo)
        return label

    levelID = property(fget=getLevelID)

    def __repr__(self):
        if len(self.label) > 0:
            label = '{} {}'.format(self.label, self.levelNo)
        else:
            label = '{}'.format(self.levelNo)
        s = '{} {}\n'.format(self.__class__.__name__, label)
        return s


######################################################################


class meshLevel(level):
    def __init__(self, mesh, params, previousLevel=None,
                 interfaces=None, meshOverlaps=None,
                 interiorBL=None,
                 comm=None,
                 label='', meshInformationPolicy=KEEP_MESH, startLevelNo=0,
                 isLastLevel=False):
        super(meshLevel, self).__init__(params, previousLevel, comm,
                                        label, startLevelNo, isLastLevel)
        self.mesh = mesh
        self.global_mesh = None
        self.interfaces = interfaces
        if self.interfaces is not None and self.params['debugOverlaps']:
            self.interfaces.validate(self.mesh, self.comm, label='Mesh interface \'{} {}\''.format(self.label, self.levelNo))
        self.meshOverlaps = meshOverlaps
        if self.meshOverlaps is not None and self.params['debugOverlaps']:
            self.meshOverlaps.check(self.mesh, self.comm, label='Mesh overlap \'{} {}\''.format(self.label, self.levelNo))
        self.interiorBL = interiorBL
        self.algebraicLevel = None
        self.meshInformationPolicy = meshInformationPolicy
        self._h = None
        self.algebraicLevelType = algebraicLevel

    def setAlgebraicLevelType(self, algLevelType):
        self.algebraicLevelType = algLevelType

    def refine(self, meshInformationPolicy):
        with self.Timer('Refined mesh'):
            newMesh, self.lookup = self.mesh.refine(returnLookup=True)
            if self.params['meshTransformation'] is not None:
                self.params['meshTransformation'](newMesh, self.lookup)
        if self.interfaces is not None:
            with self.Timer('Refined interfaces'):
                self.interfaces.refine(newMesh)
        if self.meshOverlaps is not None:
            with self.Timer('Refined mesh overlaps'):
                meshOverlaps = self.meshOverlaps.copy()
                meshOverlaps.refine(newMesh)
        if self.meshOverlaps is None:
            meshOverlaps = None
        if self.interiorBL is not None:
            with self.Timer('Refined boundary layers'):
                self.interiorBL.refine(newMesh)
        newMeshLevel = meshLevel(newMesh, self.params, self, self.interfaces, meshOverlaps, self.interiorBL, self.comm, self.label, meshInformationPolicy)
        if hasattr(self, 'numberCellsBeforeExtension'):
            newMeshLevel.numberCellsBeforeExtension = 2**self.mesh.dim * self.numberCellsBeforeExtension
        if hasattr(self, 'numberCellsLastLayer'):
            newMeshLevel.numberCellsLastLayer = 2**self.mesh.dim * self.numberCellsLastLayer
        newMeshLevel.setAlgebraicLevelType(self.algebraicLevelType)
        self.nextLevel = newMeshLevel
        return newMeshLevel

    def copy(self):
        newMeshLevel = meshLevel(self.mesh, self.params, self, self.interfaces, self.meshOverlaps,
                                 self.interiorBL, self.comm, self.label, self.meshInformationPolicy)
        return newMeshLevel

    def getIsDistributed(self):
        return self.interfaces is not None

    isDistributed = property(fget=getIsDistributed)

    def getAlgebraicLevel(self, buildType):
        self.algebraicLevel = self.algebraicLevelType(self, buildType)
        return self.algebraicLevel

    def clean(self):
        if self.meshInformationPolicy == DELETE_MESH:
            self.mesh = None
            self.meshOverlaps = None
        self.interfaces = None

    def getLevelDict(self):
        lvl = {}
        if self.mesh is not None:
            lvl['mesh'] = self.mesh
        if self.interfaces is not None:
            lvl['interfaces'] = self.interfaces
        if self.meshOverlaps is not None:
            lvl['meshOverlaps'] = self.meshOverlaps
        return lvl

    @staticmethod
    def fromLevelDict(lvl, params={}, previousLevel=None, comm=None, startLevelNo=0, label=''):
        alvl = meshLevel(None, params, previousLevel, comm=comm, startLevelNo=startLevelNo, label=label)
        if 'mesh' in lvl:
            alvl.mesh = lvl['mesh']
        if 'interfaces' in lvl:
            alvl.interfaces = lvl['interfaces']
        if 'meshOverlaps' in lvl:
            alvl.meshOverlaps = lvl['meshOverlaps']
        return alvl

    def __repr__(self):
        s = super(meshLevel, self).__repr__()
        if self.mesh is not None:
            s += ' mesh: '+self.mesh.__repr__()
        if self.interfaces is not None:
            s += self.interfaces.__repr__()
        return s

    def getH(self):
        if self._h is None:
            h = self.mesh.h
            if self.comm is not None:
                self._h = self.comm.allreduce(h, op=MPI.MAX)
        return self._h

    h = property(fget=getH)


######################################################################

class algebraicLevelBase(level):
    def __init__(self, meshLevel, buildType):
        if meshLevel.previousLevel is not None:
            previousLevel = meshLevel.previousLevel.algebraicLevel
        else:
            previousLevel = None
        super(algebraicLevelBase, self).__init__(meshLevel.params, previousLevel, meshLevel.comm, meshLevel.label, meshLevel.levelNo, meshLevel.isLastLevel)
        self.meshLevel = meshLevel
        self.P = None
        self.R = None
        self.DoFMap = None
        self.algebraicOverlaps = None
        self.build(buildType)

    def build(self, buildType):

        buildNeumann = self.params.get('buildNeumann', False)
        element = self.params['element']
        reorder = self.params['reorder']
        commType = self.params['commType']
        DoFMap_type = str2DoFMap(element)

        # Set DoFMap
        if buildType & DOFMAPS:
            if 'tag' in self.params:
                self.DoFMap = DoFMap_type(self.meshLevel.mesh, self.params['tag'])
            elif 'boundaryCondition' in self.params:
                if self.params['boundaryCondition'] in (HOMOGENEOUS_NEUMANN, DIRICHLET, NEUMANN):
                    self.DoFMap = DoFMap_type(self.meshLevel.mesh, NO_BOUNDARY)
                elif self.params['boundaryCondition'] == HOMOGENEOUS_DIRICHLET:
                    self.DoFMap = DoFMap_type(self.meshLevel.mesh, PHYSICAL)
                else:
                    raise NotImplementedError(boundaryConditions[self.params['boundaryCondition']])
            else:
                if self.isLastLevel and self.params['interiorBC'] == 'homogeneousDirichlet' and hasattr(self.meshLevel, 'numberCellsLastLayer'):
                    self.DoFMap = DoFMap_type(self.meshLevel.mesh, [PHYSICAL,
                                                                    INTERIOR],
                                              skipCellsAfter=self.meshLevel.mesh.num_cells-self.meshLevel.numberCellsLastLayer)
                elif not hasattr(self.meshLevel, 'numberCellsLastLayer') or not self.isLastLevel or self.params['interiorBC'] == 'homogeneousNeumann':
                    self.DoFMap = DoFMap_type(self.meshLevel.mesh, [PHYSICAL])
                else:
                    raise NotImplementedError()
            if buildNeumann:
                self.DoFMapNeumann = DoFMap_type(self.meshLevel.mesh, [PHYSICAL])

        if not reorder:
            if buildType & OVERLAPS:
                # build algebraic overlaps
                if self.meshLevel.meshOverlaps is not None:
                    with self.Timer('Build algebraic overlaps of type \'{}\''.format(commType)):
                        self.algebraicOverlaps = self.meshLevel.meshOverlaps.getDoFs(self.meshLevel.mesh, self.DoFMap, commType,
                                                                                     allowInteriorBoundary=self.params['interiorBC'] == 'homogeneousNeumann' or not self.isLastLevel)
                    if self.params['debugOverlaps']:
                        self.algebraicOverlaps.check(mesh=self.meshLevel.mesh,
                                                     dm=self.DoFMap,
                                                     label='algebraicOverlaps in \'{} {}\''.format(self.label, self.levelNo),
                                                     interfaces=self.meshLevel.meshOverlaps)
                elif self.meshLevel.interfaces is not None:
                    with self.Timer('Build algebraic overlaps of type \'{}\''.format(commType)):
                        self.algebraicOverlaps = self.meshLevel.interfaces.getDoFs(self.meshLevel.mesh, self.DoFMap, commType)
                    if self.params['debugOverlaps']:
                        self.algebraicOverlaps.check(mesh=self.meshLevel.mesh,
                                                     dm=self.DoFMap,
                                                     label='algebraicOverlaps in \'{} {}\''.format(self.label, self.levelNo),
                                                     interfaces=self.meshLevel.interfaces)

            if self.algebraicOverlaps is not None:
                self.inner = ip_distributed(self.algebraicOverlaps, 0)
                self.norm = norm_distributed(self.algebraicOverlaps, 0)
            else:
                self.inner = ip_serial()
                self.norm = norm_serial()
            if self.DoFMap is not None:
                self.DoFMap.set_ip_norm(self.inner, self.norm)
                self.DoFMap.set_complex_ip_norm(wrapRealInnerToComplex(self.inner),
                                                wrapRealNormToComplex(self.norm))

            if (buildType & RESTRICTION_PROLONGATION) and (self.previousLevel is not None):
                assert (self.previousLevel.DoFMap is not None) and (self.DoFMap is not None)
                # use reorder here, since reorder=False bugs out
                (self.R,
                 self.P) = buildRestrictionProlongation(self.previousLevel.DoFMap,
                                                        self.DoFMap)

    def buildCoarserMatrices(self):
        """
        Recursively build matrices on coarser levels
        """
        if self.previousLevel is not None:
            self.previousLevel.buildCoarserMatrices()

    def clean(self):
        if not self.params['keepAllDoFMaps'] and self.previousLevel is not None:
            self.DoFMap = None

    @classmethod
    def getKeys(cls):
        return ['P', 'R', 'DoFMap', 'algebraicOverlaps']

    def getLevelDict(self):
        lvl = {}
        for key in self.getKeys():
            if getattr(self, key) is not None:
                lvl[key] = getattr(self, key)
        return lvl

    @classmethod
    def fromLevelDict(cls, meshLevel, lvl):
        alvl = algebraicLevel(meshLevel, NO_BUILD)
        for key in cls.getKeys():
            if key in lvl:
                setattr(alvl, key, lvl[key])
        return alvl

    @property
    def accumulateOperator(self):
        if self.algebraicOverlaps is not None:
            return self.algebraicOverlaps.getAccumulateOperator()
        else:
            return None


class algebraicLevel(algebraicLevelBase):
    def __init__(self, meshLevel, buildType):
        self.A = None
        self.S = None
        self.D = None
        self.M = None
        self.surface_mass = None
        self.surface_stiffness = None
        super(algebraicLevel, self).__init__(meshLevel, buildType)

    def build(self, buildType):
        super(algebraicLevel, self).build(buildType)

        diffusivity = self.params['diffusivity']
        reaction = self.params['reaction']
        symmetric = self.params['symmetric']
        element = self.params['element']
        reorder = self.params['reorder']
        commType = self.params['commType']
        buildMass = self.params['buildMass'] or reaction is not None
        driftCoeff = self.params.get('driftCoeff', None)
        buildNeumann = self.params.get('buildNeumann', False)

        if buildType & SPARSITY_PATTERN:
            # set up sparsity patterns only
            DoFMap = self.DoFMap
            mesh = self.meshLevel.mesh
            self.fullyAssembled = False
            with self.Timer('Prepared sparsity patterns'):
                self.S = DoFMap.buildSparsityPattern(mesh.cells,
                                                     symmetric=symmetric,
                                                     reorder=reorder)
                if driftCoeff is not None:
                    self.D = self.S.copy()
                if buildMass:
                    self.M = self.S.copy()

        if buildType & ASSEMBLY:
            # fully build matrices
            DoFMap = self.DoFMap
            mesh = self.meshLevel.mesh
            self.fullyAssembled = True
            with self.Timer('Assembled matrices'):
                self.S = DoFMap.assembleStiffness(sss_format=symmetric,
                                                  reorder=reorder,
                                                  diffusivity=diffusivity)
                if buildMass:
                    self.M = DoFMap.assembleMass(sss_format=symmetric,
                                                 reorder=reorder)
                if driftCoeff is not None:
                    self.D = DoFMap.assembleDrift(driftCoeff)
                if buildNeumann:
                    self.neumannA = self.DoFMapNeumann.assembleStiffness(sss_format=symmetric,
                                                                         reorder=reorder,
                                                                         diffusivity=diffusivity)
                if isinstance(reaction, (float, REAL)):
                    self.A = self.S.copy()
                    for j in range(self.A.data.shape[0]):
                        self.A.data[j] += reaction*self.M.data[j]
                        if isinstance(self.A, SSS_LinearOperator):
                            for j in range(self.A.num_rows):
                                self.A.diagonal[j] += reaction*self.M.diagonal[j]
                elif isinstance(reaction, function):
                    self.A = self.S.copy()
                    dm = self.DoFMap
                    c = dm.interpolate(reaction)
                    for k in range(dm.num_dofs):
                        for j in range(self.A.indptr[k], self.A.indptr[k+1]):
                            self.A.data[j] += c[k]*self.M.data[j]
                    if isinstance(self.A, SSS_LinearOperator):
                        for k in range(self.A.num_rows):
                            self.A.diagonal[k] += c[k]*self.M.diagonal[k]
                elif reaction is None:
                    self.A = self.S
                else:
                    raise NotImplementedError()

            # surface mass matrix
            if self.isLastLevel and self.params['buildSurfaceMass']:
                with self.Timer('Build surface mass matrix'):
                    if self.params['depth'] > 0:
                        surface = mesh.get_surface_mesh(INTERIOR)
                    else:
                        surface = mesh.get_surface_mesh(INTERIOR_NONOVERLAPPING)
                    from PyNucleus_fem.femCy import assembleSurfaceMass
                    self.surface_mass = assembleSurfaceMass(mesh, surface,
                                                            self.DoFMap,
                                                            sss_format=symmetric,
                                                            reorder=reorder)
                    # ToDo: Don't just copy the sparsity pattern, this is a big waste of memory
                    # data = np.zeros((self.A.nnz), dtype=REAL)
                    # if symmetric:
                    #     diagonal = np.zeros(self.A.shape[0], dtype=REAL)
                    #     M = SSS_LinearOperator(self.A.indices, self.A.indptr, data, diagonal)
                    # else:
                    #     M = CSR_LinearOperator(self.A.indices, self.A.indptr, data)
                    # if element == 'P1':
                    #     dmS = P1_DoFMap(mesh, [PHYSICAL])
                    #     dmS.cells = surface.cells
                    # elif element == 'P2':
                    #     assert False, "Surface mass matrix not implemented for P2."
                    #     dmS = P2_DoFMap(mesh, [PHYSICAL])
                    #     cellOrig = dmS.mesh.cells
                    #     dmS.mesh.cells = surface.cells
                    # if mesh.dim == 1 and element == 'P1':
                    #     dmS.dofs_per_element = 1
                    #     self.surface_mass = assembleMatrix(surface, dmS, mass_0d_in_1d_sym_P1(), A=M,
                    #                                       sss_format=symmetric, reorder=reorder)
                    # elif mesh.dim == 2 and element == 'P1':
                    #     dmS.dofs_per_element = 2
                    #     self.surface_mass = assembleMatrix(surface, dmS, mass_1d_in_2d_sym_P1(), A=M,
                    #                                        sss_format=symmetric, reorder=reorder)
                    # elif mesh.dim == 2 and element == 'P2':
                    #     dmS.dofs_per_element = 3
                    #     self.surface_mass = assembleMatrix(surface, dmS, mass_1d_in_2d_sym_P2(), A=M,
                    #                                       sss_format=symmetric, reorder=reorder)
                    #     dmS.mesh.cells = cellOrig
                    # else:
                    #     raise NotImplementedError()

            # surface stiffness matrix
            if self.isLastLevel and self.params['buildSurfaceStiffness']:
                with self.Timer('Build surface stiffness matrix'):
                    if self.params['depth'] > 0:
                        surface = mesh.get_surface_mesh(INTERIOR)
                    else:
                        surface = mesh.get_surface_mesh(INTERIOR_NONOVERLAPPING)
                    # ToDo: Don't just copy the sparsity pattern, this is a big waste of memory
                    data = np.zeros((self.A.nnz), dtype=REAL)
                    if symmetric:
                        diagonal = np.zeros(self.A.shape[0], dtype=REAL)
                        AS = SSS_LinearOperator(self.A.indices, self.A.indptr, data, diagonal)
                    else:
                        AS = CSR_LinearOperator(self.A.indices, self.A.indptr, data)
                    assert element == 'P1', "Surface stiffness matrix only implemented for P1"
                    dmS = P1_DoFMap(mesh, [PHYSICAL])
                    dmS.cells = surface.cells
                    if mesh.dim == 2:
                        dmS.dofs_per_element = 2
                        self.surfaceStiffness = assembleMatrix(surface, dmS, stiffness_1d_in_2d_sym_P1(), A=AS,
                                                               sss_format=symmetric, reorder=reorder)
                    else:
                        raise NotImplementedError()

        if reorder and buildType & OVERLAPS:
            # build algebraic overlaps
            if self.meshLevel.meshOverlaps is not None:
                with self.Timer('Build algebraic overlaps of type \'{}\''.format(commType)):
                    self.algebraicOverlaps = self.meshLevel.meshOverlaps.getDoFs(self.meshLevel.mesh, self.DoFMap, commType,
                                                                                 allowInteriorBoundary=self.params['interiorBC'] == 'homogeneousNeumann' or not self.isLastLevel)
                if self.params['debugOverlaps']:
                    self.algebraicOverlaps.check(mesh=self.meshLevel.mesh,
                                                 dm=self.DoFMap,
                                                 label='algebraicOverlaps in \'{} {}\''.format(self.label, self.levelNo),
                                                 interfaces=self.meshLevel.meshOverlaps)
            elif self.meshLevel.interfaces is not None:
                with self.Timer('Build algebraic overlaps of type \'{}\''.format(commType)):
                    self.algebraicOverlaps = self.meshLevel.interfaces.getDoFs(self.meshLevel.mesh, self.DoFMap, commType)
                if self.params['debugOverlaps']:
                    self.algebraicOverlaps.check(mesh=self.meshLevel.mesh,
                                                 dm=self.DoFMap,
                                                 label='algebraicOverlaps in \'{} {}\''.format(self.label, self.levelNo),
                                                 interfaces=self.meshLevel.interfaces)

        if reorder and (buildType & RESTRICTION_PROLONGATION) and (self.previousLevel is not None):
            assert (self.previousLevel.DoFMap is not None) and (self.DoFMap is not None)
            # use reorder here, since reorder=False bugs out
            (self.R,
             self.P) = buildRestrictionProlongation(self.previousLevel.DoFMap,
                                                    self.DoFMap)

    def buildCoarserMatrices(self):
        """
        Recursively build matrices on coarser levels
        """
        if self.previousLevel is None:
            return
        if self.S is not None and self.P is not None and self.previousLevel.S is not None and not self.previousLevel.fullyAssembled:
            assert self.P.shape[0] == self.S.shape[0], (self.R.shape[1], self.S.shape[0])
            assert self.P.shape[1] == self.previousLevel.S.shape[0]
            with self.Timer('Restrict stiffness matrix'):
                self.P.restrictMatrix(self.S, self.previousLevel.S)
            if self.previousLevel.A is None:
                self.previousLevel.A = self.previousLevel.S
        if self.D is not None and self.P is not None and self.previousLevel.D is not None and not self.previousLevel.fullyAssembled:
            assert self.P.shape[0] == self.D.shape[0]
            assert self.P.shape[1] == self.previousLevel.D.shape[0]
            with self.Timer('Restrict drift matrix'):
                self.P.restrictMatrix(self.D, self.previousLevel.D)
        if self.M is not None and self.P is not None and self.previousLevel.M is not None and not self.previousLevel.fullyAssembled:
            assert self.P.shape[0] == self.M.shape[0]
            assert self.P.shape[1] == self.previousLevel.M.shape[0]
            with self.Timer('Restrict mass matrix'):
                self.P.restrictMatrix(self.M, self.previousLevel.M)
        if self.M is not None and self.A is not None and self.R is not None and self.previousLevel.A is not None and self.previousLevel.M is not None:
            reaction = self.params['reaction']
            if isinstance(reaction, (float, REAL)):
                for j in range(self.previousLevel.A.data.shape[0]):
                    self.previousLevel.A.data[j] += reaction*self.previousLevel.M.data[j]
                    if isinstance(self.previousLevel.A, SSS_LinearOperator):
                        for j in range(self.previousLevel.A.num_rows):
                            self.previousLevel.A.diagonal[j] += reaction*self.previousLevel.M.diagonal[j]
            elif isinstance(reaction, function):
                dm = self.previousLevel.DoFMap
                c = dm.interpolate(reaction)
                for k in range(dm.num_dofs):
                    for j in range(self.previousLevel.A.indptr[k], self.previousLevel.A.indptr[k+1]):
                        self.previousLevel.A.data[j] += c[k]*self.previousLevel.M.data[j]
                if isinstance(self.previousLevel.A, SSS_LinearOperator):
                    for k in range(self.previousLevel.A.num_rows):
                        self.previousLevel.A.diagonal[k] += c[k]*self.previousLevel.M.diagonal[k]
            elif reaction is None:
                pass
            else:
                raise NotImplementedError()
        if self.previousLevel is not None:
            self.previousLevel.fullyAssembled = True
            self.previousLevel.buildCoarserMatrices()

    @classmethod
    def getKeys(cls):
        return algebraicLevelBase.getKeys() + ['A', 'S', 'D', 'M', 'surface_mass', 'surface_stiffness']

    def getLevelDict(self):
        lvl = super(algebraicLevel, self).getLevelDict()
        if hasattr(self, ' neumannA'):
            lvl['neumannA'] = self.neumannA
        return lvl

    def getGlobalA(self, doDistribute=False, keepDistributedResult=False):
        if self.A is not None:
            if self.algebraicOverlaps is not None:
                if isinstance(self.A, CSR_LinearOperator):
                    return CSR_DistributedLinearOperator(self.A, self.algebraicOverlaps,
                                                         doDistribute=doDistribute,
                                                         keepDistributedResult=keepDistributedResult)
                else:
                    return DistributedLinearOperator(self.A, self.algebraicOverlaps,
                                                     doDistribute=doDistribute,
                                                     keepDistributedResult=keepDistributedResult)
            else:
                return self.A
        else:
            return None
