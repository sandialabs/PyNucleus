###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus_base import REAL
from PyNucleus_base.factory import factory
from PyNucleus_base.utilsFem import problem, generates
from PyNucleus_fem.mesh import (simpleInterval, intervalWithInteraction,
                                uniformSquare, squareWithInteractions,
                                discWithInteraction,
                                gradedDiscWithInteraction,
                                double_graded_interval,
                                double_graded_interval_with_interaction,
                                discWithIslands,
                                twinDisc,
                                box,
                                # boxWithInteractions,
                                ball)
from PyNucleus_fem.functions import (Lambda, constant,
                                     indicatorFunctor, squareIndicator, radialIndicator,
                                     solFractional1D, rhsFractional1D,
                                     solFractional, rhsFractional2D)
from PyNucleus_fem.DoFMaps import P1_DoFMap, str2DoFMapOrder
from PyNucleus_fem.mesh import meshFactory as meshFactoryClass
from PyNucleus_fem import (PHYSICAL, NO_BOUNDARY,
                           DIRICHLET, HOMOGENEOUS_DIRICHLET,
                           NEUMANN, HOMOGENEOUS_NEUMANN,
                           NORM, dofmapFactory)
from PyNucleus_fem.factories import functionFactory, rhsFractional2D_nonPeriodic
from scipy.special import gamma as Gamma, binom
from . twoPointFunctions import (constantTwoPoint,
                                 temperedTwoPoint,
                                 leftRightTwoPoint,
                                 interfaceTwoPoint,
                                 smoothedLeftRightTwoPoint,)
from . interactionDomains import (fullSpace,
                                  ball1,
                                  ball2,
                                  ballInf,
                                  ellipse)
from . fractionalOrders import (constFractionalOrder,
                                variableConstFractionalOrder,
                                constantNonSymFractionalOrder,
                                leftRightFractionalOrder,
                                linearLeftRightFractionalOrder,
                                smoothedLeftRightFractionalOrder,
                                innerOuterFractionalOrder,
                                smoothedInnerOuterFractionalOrder,
                                islandsFractionalOrder,
                                layersFractionalOrder,
                                singleVariableUnsymmetricFractionalOrder,
                                feFractionalOrder)
from . kernelsCy import (getKernelEnum,
                         FRACTIONAL, INDICATOR, PERIDYNAMIC, GAUSSIAN,
                         LOGINVERSEDISTANCE, MONOMIAL,
                         )
from . kernels import (getFractionalKernel,
                       getIntegrableKernel,
                       getKernel)
from copy import deepcopy


class fractionalOrderFactoryClass(factory):
    def build(self, name, *args, **kwargs):
        dm = None
        if 'dm' in kwargs:
            dm = kwargs.pop('dm')
        if dm is not None:
            s = self.build(name, *args, **kwargs)
            assert isinstance(s, (constFractionalOrder, variableConstFractionalOrder,
                                  constantNonSymFractionalOrder, singleVariableUnsymmetricFractionalOrder))
            sVec = dm.interpolate(s.fixedY(np.zeros((dm.mesh.dim), dtype=REAL)))
            return super().build('fe', sVec, s.min, s.max)
        else:
            return super().build(name, *args, **kwargs)


fractionalOrderFactory = fractionalOrderFactoryClass()
fractionalOrderFactory.register('constant', constFractionalOrder, aliases=['const'])
fractionalOrderFactory.register('varConst', variableConstFractionalOrder, aliases=['constVar', 'constantSym'])
fractionalOrderFactory.register('leftRight', leftRightFractionalOrder, aliases=['twoDomain'])
fractionalOrderFactory.register('linearLeftRightNonSym', linearLeftRightFractionalOrder)
fractionalOrderFactory.register('smoothedLeftRight', smoothedLeftRightFractionalOrder, params={'r': 0.1, 'slope': 200.}, aliases=['twoDomainNonSym'])
fractionalOrderFactory.register('constantNonSym', constantNonSymFractionalOrder)
fractionalOrderFactory.register('innerOuter', innerOuterFractionalOrder)
fractionalOrderFactory.register('innerOuterNonSym', smoothedInnerOuterFractionalOrder)
fractionalOrderFactory.register('islands', islandsFractionalOrder, params={'r': 0.1, 'r2': 0.6})
fractionalOrderFactory.register('layers', layersFractionalOrder)
fractionalOrderFactory.register('fe', feFractionalOrder)

twoPointFunctionFactory = factory()
twoPointFunctionFactory.register('constant', constantTwoPoint, aliases=['const', 'constantTwoPoint'])
twoPointFunctionFactory.register('tempered', temperedTwoPoint, aliases=['temperedTwoPoint'])
twoPointFunctionFactory.register('leftRight', leftRightTwoPoint, aliases=['leftRightTwoPoint'])
twoPointFunctionFactory.register('interface', interfaceTwoPoint, aliases=['interfaceTwoPoint'])

interactionFactory = factory()
interactionFactory.register('fullSpace', fullSpace, aliases=['full'])
interactionFactory.register('ball2', ball2, aliases=['2', 2])
interactionFactory.register('ball1', ball1, aliases=['1', 1])
interactionFactory.register('ballInf', ballInf, aliases=['inf', np.inf])

kernelFactory = factory()
kernelFactory.register('fractional', getFractionalKernel)
kernelFactory.register('indicator', getIntegrableKernel, params={'kernel': INDICATOR}, aliases=['constant'])
kernelFactory.register('inverseDistance', getIntegrableKernel, params={'kernel': PERIDYNAMIC}, aliases=['peridynamic', 'inverseOfDistance'])
kernelFactory.register('gaussian', getIntegrableKernel, params={'kernel': GAUSSIAN})
kernelFactory.register('logInverseDistance', getIntegrableKernel, params={'kernel': LOGINVERSEDISTANCE})
kernelFactory.register('monomial', getIntegrableKernel, params={'kernel': MONOMIAL})


class nonlocalMeshFactoryClass(factory):
    def __init__(self):
        super(nonlocalMeshFactoryClass, self).__init__()
        self.nonOverlappingMeshFactory = meshFactoryClass()
        self.overlappingMeshFactory = meshFactoryClass()

    def register(self, name, classTypeNoOverlap, classTypeOverlap, dim, indicators, paramsNoOverlap={}, paramsOverlap={}, aliases=[]):
        if classTypeNoOverlap is not None:
            self.nonOverlappingMeshFactory.register(name, classTypeNoOverlap, dim, paramsNoOverlap, aliases)
        if classTypeOverlap is not None:
            self.overlappingMeshFactory.register(name, classTypeOverlap, dim, paramsOverlap, aliases)
        super(nonlocalMeshFactoryClass, self).register(name, indicators)

    def build(self, name, kernel, boundaryCondition, noRef=0, useMulti=False, **kwargs):
        skipMesh = False
        if 'skipMesh' in kwargs:
            skipMesh = kwargs.pop('skipMesh')

        if kernel is None:
            horizonValue = 0.
        else:
            horizonValue = kernel.horizon.value

        domainIndicator, boundaryIndicator, interactionIndicator = super(nonlocalMeshFactoryClass, self).build(name, **kwargs)

        if boundaryCondition == HOMOGENEOUS_DIRICHLET:
            if horizonValue == np.inf:
                # if kernel.s.max < 0.5:
                #     tag = NO_BOUNDARY
                # else:
                #     tag = PHYSICAL
                tag = PHYSICAL
                zeroExterior = True
            else:
                tag = domainIndicator
                zeroExterior = False
            hasInteractionDomain = 0 < horizonValue < np.inf
        elif boundaryCondition == HOMOGENEOUS_NEUMANN:
            tag = NO_BOUNDARY
            zeroExterior = False
            hasInteractionDomain = False
        elif boundaryCondition == DIRICHLET:
            if horizonValue == np.inf:
                if kernel.s.max < 0.5:
                    tag = NO_BOUNDARY
                else:
                    tag = PHYSICAL
                raise NotImplementedError()
            else:
                tag = NO_BOUNDARY
            zeroExterior = False
            hasInteractionDomain = 0 < horizonValue < np.inf
        elif boundaryCondition == NEUMANN:
            if horizonValue == np.inf:
                assert False
            else:
                tag = NO_BOUNDARY
            zeroExterior = False
            hasInteractionDomain = True
        elif boundaryCondition == NORM:
            tag = PHYSICAL
            zeroExterior = kernel.s.max >= 0.5
            hasInteractionDomain = False
        else:
            raise NotImplementedError('Unknown boundary condition {}'.format(boundaryCondition))

        if not skipMesh:
            if hasInteractionDomain:
                assert 0 < horizonValue < np.inf, horizonValue
                kwargs['horizon'] = horizonValue
                mesh = self.overlappingMeshFactory.build(name, noRef, **kwargs)
            else:
                mesh = self.nonOverlappingMeshFactory.build(name, noRef, **kwargs)

            dmTest = P1_DoFMap(mesh, tag)
            while dmTest.num_dofs == 0:
                mesh = mesh.refine()
                dmTest = P1_DoFMap(mesh, tag)

        nonlocalInfo = {'domain': domainIndicator,
                        'boundary': boundaryIndicator,
                        'interaction': interactionIndicator,
                        'tag': tag,
                        'zeroExterior': zeroExterior}
        if not skipMesh:
            return mesh, nonlocalInfo
        else:
            return nonlocalInfo

    def getDim(self, name):
        return self.nonOverlappingMeshFactory.getDim(name)


def intervalIndicators(a=-1, b=1, **kwargs):
    eps = 1e-9
    domainIndicator = squareIndicator(np.array([a+eps], dtype=REAL),
                                      np.array([b-eps], dtype=REAL))
    interactionIndicator = Lambda(lambda x: 1. if ((x[0] < a-eps) or (b+eps < x[0])) else 0.)
    boundaryIndicator = Lambda(lambda x: 1. if ((a-eps < x[0] < a+eps) or (b-eps < x[0] < b+eps)) else 0.)
    return domainIndicator, boundaryIndicator, interactionIndicator


def squareIndicators(ax=-1., ay=-1., bx=1., by=1., **kwargs):
    domainIndicator = squareIndicator(np.array([ax+1e-9, ay+1e-9], dtype=REAL),
                                      np.array([bx-1e-9, by-1e-9], dtype=REAL))
    interactionIndicator = constant(1.)-squareIndicator(np.array([ax-1e-9, ay-1e-9], dtype=REAL),
                                                        np.array([bx+1e-9, by+1e-9], dtype=REAL))
    boundaryIndicator = constant(1.)-domainIndicator-interactionIndicator
    return domainIndicator, boundaryIndicator, interactionIndicator


def radialIndicators(*args, **kwargs):
    domainIndicator = radialIndicator(1.-1e-9)
    interactionIndicator = constant(1.)-radialIndicator(1.+1e-9)
    boundaryIndicator = radialIndicator(1.+1e-9)-radialIndicator(1.-1e-9)
    return domainIndicator, boundaryIndicator, interactionIndicator


def twinDiscIndicators(radius=1., sep=0.1, **kwargs):
    domainIndicator = (radialIndicator(radius-1e-9, np.array([sep/2+radius, 0.], dtype=REAL)) +
                       radialIndicator(radius-1e-9, np.array([-sep/2-radius, 0.], dtype=REAL)))
    interactionIndicator = constant(1.)-(radialIndicator(radius+1e-9, np.array([sep/2+radius, 0.], dtype=REAL)) +
                                         radialIndicator(radius+1e-9, np.array([-sep/2-radius, 0.], dtype=REAL)))
    boundaryIndicator = ((radialIndicator(radius+1e-9, np.array([sep/2+radius, 0.], dtype=REAL)) +
                          radialIndicator(radius+1e-9, np.array([-sep/2-radius, 0.], dtype=REAL))) -
                         (radialIndicator(radius-1e-9, np.array([sep/2+radius, 0.], dtype=REAL)) +
                          radialIndicator(radius-1e-9, np.array([-sep/2-radius, 0.], dtype=REAL))))
    return domainIndicator, boundaryIndicator, interactionIndicator


def boxIndicators(ax=-1., ay=-1., az=-1., bx=1., by=1., bz=1., **kwargs):
    domainIndicator = squareIndicator(np.array([ax+1e-9, ay+1e-9, az+1e-9], dtype=REAL),
                                      np.array([bx-1e-9, by-1e-9, bz-1e-9], dtype=REAL))
    interactionIndicator = constant(1.)-squareIndicator(np.array([ax-1e-9, ay-1e-9, az-1e-9], dtype=REAL),
                                                        np.array([bx+1e-9, by+1e-9, bz+1e-9], dtype=REAL))
    boundaryIndicator = constant(1.)-domainIndicator-interactionIndicator
    return domainIndicator, boundaryIndicator, interactionIndicator


def ballWithInteractions(*args, **kwargs):
    raise NotImplementedError()


nonlocalMeshFactory = nonlocalMeshFactoryClass()
nonlocalMeshFactory.register('interval', simpleInterval, intervalWithInteraction, 1, intervalIndicators,
                             {'a': -1, 'b': 1}, {'a': -1, 'b': 1})
nonlocalMeshFactory.register('gradedInterval', double_graded_interval, double_graded_interval_with_interaction, 1, intervalIndicators,
                             {'a': -1, 'b': 1, 'mu_ll': 2., 'mu_rr': 2.}, {'a': -1, 'b': 1, 'mu_ll': 2., 'mu_rr': 2.})
nonlocalMeshFactory.register('square', uniformSquare, squareWithInteractions, 2, squareIndicators,
                             {'N': 2, 'M': 2, 'ax': -1, 'ay': -1, 'bx': 1, 'by': 1}, {'ax': -1, 'ay': -1, 'bx': 1, 'by': 1}, aliases=['rectangle'])
nonlocalMeshFactory.register('disc', discWithInteraction, discWithInteraction, 2, radialIndicators,
                             {'horizon': 0., 'radius': 1.}, {'radius': 1.})
nonlocalMeshFactory.register('gradedDisc', gradedDiscWithInteraction, gradedDiscWithInteraction, 2, radialIndicators,
                             {'horizon': 0., 'radius': 1.}, {'radius': 1.})
nonlocalMeshFactory.register('discWithIslands', discWithIslands, discWithIslands, 2, radialIndicators,
                             {'horizon': 0., 'radius': 1., 'islandOffCenter': 0.35, 'islandDiam': 0.5},
                             {'radius': 1., 'islandOffCenter': 0.35, 'islandDiam': 0.5})
nonlocalMeshFactory.register('twinDisc', twinDisc, twinDisc, 2, radialIndicators,
                             {'radius': 1., 'sep': 0.1}, {'radius': 1., 'sep': 0.1})
# nonlocalMeshFactory.register('box', box, boxWithInteractions, 3, boxIndicators,
#                              {'Nx': 2, 'Ny': 2, 'Nz': 2, 'ax': -1, 'ay': -1, 'az': -1, 'bx': 1, 'by': 1, 'bz': 1}, {'Nx': 2, 'Ny': 2, 'Nz': 2, 'ax': -1, 'ay': -1, 'az': -1, 'bx': 1, 'by': 1, 'bz': 1})
nonlocalMeshFactory.register('ball', ball, ballWithInteractions, 3, radialIndicators,
                             {'radius': 1.}, {'radius': 1.})


class nonlocalBaseProblem(problem):
    def __init__(self, driver):
        super().__init__(driver)
        self.addProperty('sType')
        self.addProperty('sArgs')
        self.addProperty('phiType')
        self.addProperty('phiArgs')
        self.addProperty('admissibleParams')
        self.admissibleParams = None
        self.feFractionalOrder = None

    def setDriverArgs(self):
        p = self.driver.addGroup('kernel')
        self.setDriverFlag('kernelType', acceptedValues=['fractional', 'constant', 'inverseDistance', 'gaussian', 'local'], help='type of kernel', group=p)
        self.addParametrizedArg('const', [float])
        self.addParametrizedArg('varconst', [float])
        self.addParametrizedArg('constantNonSym', [float])
        self.addParametrizedArg('leftRight', [float, float, float, float])
        self.addParametrizedArg('twoDomain', [float, float, float, float])
        self.addParametrizedArg('twoDomainNonSym', [float, float])
        self.addParametrizedArg('linearLeftRightNonSym', [float, float, float])
        self.addParametrizedArg('innerOuterNonSym', [float, float])
        self.addParametrizedArg('layers', [float, float, int])
        self.addParametrizedArg('islands', [float, float])
        self.addParametrizedArg('islands4', [float, float, float, float])
        self.addParametrizedArg('tempered', [float])
        self.setDriverFlag('s', 'const(0.4)', argInterpreter=self.argInterpreter(['const', 'varconst', 'constantNonSym', 'twoDomain', 'twoDomainNonSym',
                                                                                  'linearLeftRightNonSym',
                                                                                  'innerOuterNonSym',
                                                                                  'layers', 'islands', 'islands4']), help='fractional order', group=p)
        self.setDriverFlag('horizon', 0.2, help='interaction horizon', group=p)
        self.addParametrizedArg('ellipse', [float, float])
        self.setDriverFlag('interaction', 'ball2', argInterpreter=self.argInterpreter(['ellipse'],
                                                                                      acceptedValues=['ball2', 'ellipse', 'fullSpace']),
                           help='interaction domain', group=p)
        self.setDriverFlag('phi', 'const(1.)', argInterpreter=self.argInterpreter(['const', 'twoDomain', 'twoDomainNonSym', 'tempered']),
                           help='kernel coefficient', group=p)
        self.setDriverFlag('normalized', True, help='kernel normalization', group=p)
        self.setDriverFlag('discretizedOrder', False, help='Use a FE function for the fractional order s.')

    def processCmdline(self, params):
        dim = nonlocalMeshFactory.getDim(params['domain'])
        if params['kernelType'] == 'fractional':
            s = params['s']
            for sName in ['const', 'varconst', 'constantNonSym', 'leftRight', 'twoDomain', 'twoDomainNonSym', 'linearLeftRightNonSym', 'innerOuterNonSym', 'islands']:
                if self.parametrizedArg(sName).match(s):
                    sType = sName
                    sArgs = self.parametrizedArg(sName).interpret(s)
                    break
            else:
                if self.parametrizedArg('layers').match(s):
                    t = np.linspace(*self.parametrizedArg('layers').interpret(s), dtype=REAL)
                    sVals = np.empty((t.shape[0], t.shape[0]), dtype=REAL)
                    for i in range(t.shape[0]):
                        for j in range(t.shape[0]):
                            sVals[i, j] = 0.5*(t[i]+t[j])
                    sType = 'layers'
                    # dim =
                    sArgs = (dim, np.linspace(-1., 1., sVals.shape[0]+1, dtype=REAL), s)
                elif self.parametrizedArg('islands4').match(s):
                    sType = 'islands'
                    sArgs = self.parametrizedArg('islands4').interpret(s)
                else:
                    raise NotImplementedError(s)
        else:
            sType = None
            sArgs = None
        self.sType = sType
        self.sArgs = sArgs

        phi = params['phi']
        if self.parametrizedArg('const').match(phi):
            c, = self.parametrizedArg('const').interpret(phi)
            if c == 1.:
                phiType = None
                phiArgs = None
            else:
                phiType = 'const'
                phiArgs = (c, )
        elif self.parametrizedArg('twoDomain').match(phi):
            phiType = 'twoDomain'
            phiArgs = self.parametrizedArg('twoDomain').interpret(phi)
        elif self.parametrizedArg('twoDomainNonSym').match(phi):
            phiType = 'twoDomainNonSym'
            phiArgs = self.parametrizedArg('twoDomainNonSym').interpret(phi)
        elif self.parametrizedArg('tempered').match(phi):
            lambdaCoeff, = self.parametrizedArg('tempered').interpret(phi)
            phiType = 'tempered'
            phiArgs = (lambdaCoeff, dim)
        else:
            raise NotImplementedError(phi)
        self.phiType = phiType
        self.phiArgs = phiArgs

        super().processCmdline(params)

    @generates('dim')
    def getDim(self, domain):
        self.dim = nonlocalMeshFactory.getDim(domain)

    @generates('dmAux')
    def constructAuxiliarySpace(self):
        self.dmAux = None

    @generates(['kernel', 'rangedKernel'])
    def processKernel(self, dim, kernelType, sType, sArgs, phiType, phiArgs, horizon, interaction, normalized, admissibleParams,
                      discretizedOrder, dmAux, feFractionalOrder):

        if kernelType == 'local':
            self.kernel = None
            return

        kType = getKernelEnum(kernelType)

        if admissibleParams is not None:
            assert kType == FRACTIONAL
            assert sType == 'const'
            from PyNucleus_nl.kernelsCy import RangedFractionalKernel
            rangedKernel = self.directlyGetWithoutChecks('rangedKernel')
            if rangedKernel is None or not isinstance(rangedKernel, RangedFractionalKernel):
                self.rangedKernel = RangedFractionalKernel(dim,
                                                           admissibleParams.subset({'sArgs'}),
                                                           functionFactory('constant', horizon),
                                                           normalized)
            else:
                self.rangedKernel = rangedKernel
            try:
                self.rangedKernel.setOrder(*sArgs)
                self.kernel = self.rangedKernel.getFrozenKernel(*sArgs)
            except TypeError:
                sArgs = (sArgs, )
                self.rangedKernel.setOrder(*sArgs)
                self.kernel = self.rangedKernel.getFrozenKernel(*sArgs)
            return
        else:
            self.rangedKernel = None

        if kType == FRACTIONAL:
            if feFractionalOrder is None:
                if isinstance(sArgs, dict):
                    if discretizedOrder:
                        sFun = fractionalOrderFactory(sType, dm=dmAux, **sArgs)
                    else:
                        sFun = fractionalOrderFactory(sType, **sArgs)
                else:
                    try:
                        if discretizedOrder:
                            sFun = fractionalOrderFactory(sType, *sArgs, dm=dmAux)
                        else:
                            sFun = fractionalOrderFactory(sType, *sArgs)
                    except TypeError:
                        sArgs = (sArgs, )
                        if discretizedOrder:
                            sFun = fractionalOrderFactory(sType, *sArgs, dm=dmAux)
                        else:
                                sFun = fractionalOrderFactory(sType, *sArgs)
            else:
                sFun = deepcopy(feFractionalOrder)
        else:
            sFun = None

        if phiType is not None:
            if phiType == 'const':
                phiFun = constantTwoPoint(*phiArgs)
            elif phiType == 'twoDomain':
                phiFun = leftRightTwoPoint(*phiArgs)
            elif phiType == 'twoDomainNonSym':
                phiFun = smoothedLeftRightTwoPoint(*phiArgs)
            elif phiType == 'tempered':
                phiFun = temperedTwoPoint(*phiArgs)
            else:
                raise NotImplementedError(phiType)
        else:
            phiFun = None

        if horizon == np.inf:
            interactionFun = fullSpace()
        elif interaction == 'ball2':
            interactionFun = ball2()
        elif self.parametrizedArg('ellipse').match(interaction):
            aFac, bFac = self.parametrizedArg('ellipse').interpret(interaction)
            interactionFun = ellipse(aFac, bFac)
        else:
            raise NotImplementedError(interaction)

        self.kernel = getKernel(dim=dim, kernel=kType, s=sFun, horizon=horizon, normalized=normalized, phi=phiFun,
                                interaction=interactionFun, piecewise=sFun.symmetric if sFun is not None else True)

    def report(self, group):
        group.add('kernel', self.kernel)
        if self.kernel is not None:
            if self.kernel.kernelType == FRACTIONAL:
                group.add('s', self.kernel.s)
            group.add('horizon', self.horizon)


class fractionalLaplacianProblem(nonlocalBaseProblem):
    def __init__(self, driver, useMulti=False):
        super().__init__(driver)
        self.useMulti = useMulti

    def setDriverArgs(self):
        super().setDriverArgs()
        if self.driver.isMaster:
            self.driver.parser.set_defaults(s='const(0.75)', horizon=np.inf, interaction='fullSpace')
        p = self.driver.addGroup('problem')
        self.setDriverFlag('domain', acceptedValues=['interval', 'disc', 'Lshape', 'square',
                                                     'cutoutCircle', 'disconnectedInterval', 'disconnectedDomain',
                                                     'ball'], group=p)
        self.setDriverFlag('problem', acceptedValues=['constant', 'notPeriodic', 'plateau',
                                                      'sin', 'cos', 3, 'source', 'zeroFlux', 'Greens', 'knownSolution'], group=p)
        self.setDriverFlag('element', acceptedValues=['P1', 'P2', 'P3', 'P0'], group=p)
        self.setDriverFlag('adaptive', acceptedValues=['residualMelenk', 'residualNochetto',
                                                       'residual', 'hierarchical', 'knownSolution', None],
                           argInterpreter=lambda v: None if v == 'None' else v, group=p)
        self.setDriverFlag('noRef', -1, group=p)

    def processCmdline(self, params):
        noRef = params['noRef']
        if noRef <= 0:
            domain = params['domain']
            element = params['element']
            adaptive = params['adaptive']
            if domain == 'interval':
                if adaptive is None:
                    if element == 'P0':
                        noRef = 6
                    elif element == 'P1':
                        noRef = 6
                    elif element == 'P2':
                        noRef = 5
                    elif element == 'P3':
                        noRef = 5
                    else:
                        raise NotImplementedError(element)
                else:
                    if element == 'P1':
                        noRef = 22
                    elif element == 'P2':
                        noRef = 21
                    else:
                        raise NotImplementedError(element)
            elif domain == 'disconnectedInterval':
                noRef = 40
            elif domain == 'disc':
                if adaptive is None:
                    noRef = 5
                else:
                    noRef = 7
            elif domain == 'square':
                noRef = 20
            elif domain == 'Lshape':
                noRef = 20
            elif domain == 'cutoutCircle':
                noRef = 30
            elif domain == 'ball':
                noRef = 2
            else:
                raise NotImplementedError(domain)
            params['noRef'] = noRef
        super().processCmdline(params)

    @generates('domainParams')
    def getDomainParams(self, domain):
        meshParams = {}
        if domain == 'interval':
            radius = 1.
            meshParams.update({'a': -radius, 'b': radius})
        elif domain == 'disconnectedInterval':
            meshParams['sep'] = 0.1
        elif domain == 'disc':
            radius = 1.
            meshParams.update({'h': 0.78, 'radius': radius})
        elif domain == 'square':
            meshParams.update({'N': 3, 'ax': -1, 'ay': -1, 'bx': 1, 'by': 1})
        elif domain == 'Lshape':
            pass
        elif domain == 'cutoutCircle':
            meshParams.update({'radius': 1., 'cutoutAngle': np.pi/2.})
        elif domain == 'ball':
            pass
        else:
            raise NotImplementedError(domain)
        self.domainParams = meshParams

    @generates(['analyticSolution', 'exactHsSquared', 'exactL2Squared', 'rhs',
                'mesh_domain', 'mesh_params', 'tag', 'boundaryCondition',
                'domainIndicator', 'interactionIndicator', 'fluxIndicator',
                'zeroExterior',
                'rhsData', 'dirichletData', 'fluxData'])
    def processProblem(self, kernel, dim, domain, domainParams, problem, normalized):
        s = kernel.s
        self.analyticSolution = None
        self.exactHsSquared = None
        L2_ex = None
        assert kernel.horizon.value == np.inf
        assert normalized

        boundaryCondition = HOMOGENEOUS_DIRICHLET
        if domain == 'interval':
            radius = 1.

            if problem == 'constant':
                self.rhs = constant(1.)
                if isinstance(s, (constFractionalOrder, variableConstFractionalOrder, constantNonSymFractionalOrder)):
                    C = 2.**(-2.*s.value)*Gamma(dim/2.)/Gamma((dim+2.*s.value)/2.)/Gamma(1.+s.value)
                    self.exactHsSquared = C * np.sqrt(np.pi)*Gamma(s.value+1)/Gamma(s.value+3/2)
                    L2_ex = np.sqrt(C**2 * np.sqrt(np.pi) * Gamma(1+2*s.value)/Gamma(3/2+2*s.value) * radius**2)
                    self.analyticSolution = solFractional(s.value, dim, radius)
            elif problem == 'sin':
                self.rhs = Lambda(lambda x: np.sin(np.pi*x[0]))
            elif problem == 'cos':
                self.rhs = Lambda(lambda x: np.cos(np.pi*x[0]/2.))
            elif problem == 'plateau':
                self.rhs = Lambda(np.sign)

                # def e(n):
                #     return (2*n+s+3/2)/2**(2*s)/np.pi / binom(n+s+1, n-1/2)**2/Gamma(s+5/2)**2

                # k = 10
                # exactHsSquared = sum([e(n) for n in range(1000000)])
                self.exactHsSquared = 2**(1-2*s) / (2*s+1) / Gamma(s+1)**2
            elif isinstance(problem, int):
                self.rhs = rhsFractional1D(s, problem)
                self.exactHsSquared = 2**(2*s)/(2*problem+s+0.5) * Gamma(1+s)**2 * binom(s+problem, problem)**2
                self.analyticSolution = solFractional1D(s, problem)
            elif problem == 'zeroFlux':
                boundaryCondition = HOMOGENEOUS_NEUMANN

                if kernel.variable:
                    assert isinstance(s, (variableConstFractionalOrder, smoothedLeftRightFractionalOrder))

                    def fun(x):
                        kernel.evalParams_py(x, x)
                        sVal = kernel.sValue
                        fac = 2*kernel.scalingValue
                        return fac/(2*sVal-1) * ((1-x[0])**(1-2*sVal) - (1+x[0])**(1-2*sVal))

                else:

                    sVal = s.value
                    fac = 2*kernel.scalingValue
                    assert sVal != 0.5

                    def fun(x):
                        return fac/(2*sVal-1) * ((1-x[0])**(1-2*sVal) - (1+x[0])**(1-2*sVal))

                self.rhs = functionFactory('Lambda', fun)
                self.analyticSolution = functionFactory('x0')
                L2_ex = np.sqrt(2/3)
            elif problem == 'knownSolution':
                from scipy.special import hyp2f1
                assert isinstance(s, (constFractionalOrder, variableConstFractionalOrder,
                                      constantNonSymFractionalOrder, singleVariableUnsymmetricFractionalOrder)), s

                beta = 0.7

                def fun(x):
                    kernel.evalParams_py(x, x)
                    sVal = kernel.sValue

                    return 2**(2*sVal) * Gamma(sVal+0.5)*Gamma(beta+1.)/np.sqrt(np.pi)/Gamma(beta+1.-sVal) * hyp2f1(sVal+0.5, -beta+sVal, 0.5, x[0]**2)

                self.rhs = functionFactory('Lambda', fun)
                self.analyticSolution = functionFactory('Lambda', lambda x: (1.-x[0]**2)**beta)
                L2_ex = np.sqrt(np.sqrt(np.pi) * Gamma(1+2*beta)/Gamma(3/2+2*beta) * radius**2)
            elif problem == 'Greens':
                boundaryCondition = HOMOGENEOUS_NEUMANN
                self.rhs = functionFactory('squareIndicator', np.array([-0.1]), np.array([0.1]))
            else:
                raise NotImplementedError(problem)
        elif domain == 'disconnectedInterval':
            if problem == 'constant':
                self.rhs = Lambda(lambda x: 1. if x[0] > 0.5 else 0.)
            else:
                raise NotImplementedError(problem)
        elif domain == 'disc':
            radius = 1.

            if problem == 'constant':
                self.rhs = constant(1.)
                if isinstance(s, (constFractionalOrder, variableConstFractionalOrder, constantNonSymFractionalOrder)):
                    C = 2.**(-2.*s.value)*Gamma(dim/2.)/Gamma((dim+2.*s.value)/2.)/Gamma(1.+s.value)
                    self.exactHsSquared = C * np.pi*radius**(2-2*s.value)/(s.value+1)
                    L2_ex = np.sqrt(C**2 * np.pi/(1+2*s.value)*radius**2)
                    self.analyticSolution = solFractional(s.value, dim, radius)
            elif problem == 'notPeriodic':
                n = 2
                l = 2
                self.exactHsSquared = 2**(2*s-1)/(2*n+s+l+1) * Gamma(1+s+n)**2/Gamma(1+n)**2 * (np.pi+np.sin(4*np.pi*l)/(4*l))

                n = 1
                l = 5
                self.exactHsSquared += 2**(2*s-1)/(2*n+s+l+1) * Gamma(1+s+n)**2/Gamma(1+n)**2 * (np.pi+np.sin(4*np.pi*l)/(4*l))
                self.rhs = rhsFractional2D_nonPeriodic(s)
            elif problem == 'plateau':
                self.rhs = Lambda(lambda x: x[0] > 0)
                try:
                    from mpmath import meijerg
                    self.exactHsSquared = np.pi/4*2**(-2*s) / (s+1) / Gamma(1+s)**2
                    self.exactHsSquared -= 2**(-2*s)/np.pi * meijerg([[1., 1.+s/2], [5/2+s, 5/2+s]],
                                                                     [[2., 1/2, 1/2], [2.+s/2]],
                                                                     -1., series=2)
                    self.exactHsSquared = float(self.exactHsSquared)
                except ImportError:
                    self.exactHsSquared = np.pi/4*2**(-2*s) / (s+1) / Gamma(1+s)**2
                    for k in range(100000):
                        self.exactHsSquared += 2**(-2*s) / Gamma(s+3)**2 / (2*np.pi) * (2*k+s+2) * (k+1) / binom(k+s+1.5, s+2)**2
            elif isinstance(problem, tuple):
                n, l = problem
                self.exactHsSquared = 2**(2*s-1)/(2*n+s+l+1) * Gamma(1+s+n)**2/Gamma(1+n)**2 * (np.pi+np.sin(4*np.pi*l)/(4*l))

                self.rhs = rhsFractional2D(s, n=n, l=l)
            elif problem == 'sin':
                self.rhs = Lambda(lambda x: np.sin(np.pi*(x[0]**2+x[1]**2)))
            elif problem == 'knownSolution':
                from scipy.special import hyp2f1
                assert isinstance(s, (constFractionalOrder, variableConstFractionalOrder,
                                      constantNonSymFractionalOrder, singleVariableUnsymmetricFractionalOrder)), s

                beta = 0.7

                def fun(x):
                    kernel.evalParams_py(x, x)
                    sVal = kernel.sValue

                    return 2**(2*sVal) * Gamma(sVal+1.0)*Gamma(beta+1.)/Gamma(beta+1.-sVal) * hyp2f1(sVal+1.0, -beta+sVal, 1.0, np.linalg.norm(x)**2)

                self.rhs = functionFactory('Lambda', fun)
                self.analyticSolution = functionFactory('Lambda', lambda x: max(1.-np.linalg.norm(x)**2, 0.)**beta)
                L2_ex = np.sqrt(np.pi/(1+2*beta)*radius**2)
            else:
                raise NotImplementedError(problem)
        elif domain == 'square':
            if problem == 'constant':
                self.rhs = constant(1.)
            elif problem == 'sin':
                self.rhs = Lambda(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
            elif problem == 'source':
                self.rhs = (functionFactory.build('radialIndicator', radius=0.3, center=np.array([0.2, 0.6], dtype=REAL)) -
                            functionFactory.build('radialIndicator', radius=0.3, center=np.array([-0.2, -0.6], dtype=REAL)))
            else:
                raise NotImplementedError(problem)
        elif domain == 'Lshape':
            if problem == 'constant':
                self.rhs = constant(1.)
            elif problem == 'sin':
                self.rhs = Lambda(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
            else:
                raise NotImplementedError(problem)
        elif domain == 'cutoutCircle':
            if problem == 'constant':
                self.rhs = constant(1.)
            elif problem == 'sin':
                self.rhs = Lambda(lambda x: np.sin(np.pi*(x[0]**2+x[1]**2)))
            else:
                raise NotImplementedError(problem)
        elif domain == 'ball':
            radius = 1.
            if problem == 'constant':
                self.rhs = constant(1.)
                if isinstance(s, (constFractionalOrder, variableConstFractionalOrder, constantNonSymFractionalOrder)):
                    C = 2.**(-2.*s.value)*Gamma(dim/2.)/Gamma((dim+2.*s.value)/2.)/Gamma(1.+s.value)
                    self.exactHsSquared = C * np.pi*radius**(2-2*s.value)/(s.value+1)
                    L2_ex = np.sqrt(C**2 * np.pi/(1+2*s.value)*radius**2)
                    self.analyticSolution = solFractional(s.value, dim, radius)
            else:
                raise NotImplementedError(problem)
        else:
            raise NotImplementedError(domain)

        mesh_domain = domain
        meshParams = {'kernel': kernel}
        meshParams.update(domainParams)
        self.boundaryCondition = meshParams['boundaryCondition'] = boundaryCondition
        meshParams['useMulti'] = self.useMulti
        self.mesh_domain = mesh_domain
        self.mesh_params = meshParams
        nI = nonlocalMeshFactory.build(mesh_domain, skipMesh=True, **meshParams)
        self.tag = nI['tag']
        self.domainIndicator = nI['domain']
        self.interactionIndicator = nI['interaction']+nI['boundary']
        if boundaryCondition in (NEUMANN, HOMOGENEOUS_NEUMANN):
            self.fluxIndicator = self.interactionIndicator
        else:
            self.fluxIndicator = functionFactory('constant', 0.)
        self.zeroExterior = nI['zeroExterior']
        self.dirichletData = None
        self.fluxData = None
        self.rhsData = self.rhs
        if L2_ex is not None:
            self.exactL2Squared = L2_ex**2
        else:
            self.exactL2Squared = None

    @generates(['eta', 'target_order'])
    def getApproximationParams(self, dim, kernel, element):
        s = kernel.s
        elementOrder = str2DoFMapOrder(element)
        if dim == 1:
            self.target_order = (1+elementOrder-s.min)/dim
        else:
            self.target_order = 1/dim

        # Picking bigger, say eta = 7, potentially speeds up assembly.
        # Not clear about impact on error.
        if dim == 1:
            self.eta = 1
        else:
            self.eta = 3.

    @generates('mesh')
    def buildMesh(self, mesh_domain, mesh_params):
        self.mesh, _ = nonlocalMeshFactory.build(mesh_domain, **mesh_params)

    @generates('dmAux')
    def constructAuxiliarySpace(self, dim, domain, domainParams, kernelType, horizon):
        # This is not the actual kernel that we use.
        # We just need something to get a mesh to support the fractional order.
        kType = getKernelEnum(kernelType)
        if kType == FRACTIONAL:
            sFun = fractionalOrderFactory('const', 0.75)
            kernel = getKernel(dim=dim, kernel=kType, s=sFun, horizon=horizon)
        else:
            kernel = getKernel(dim=dim, kernel=kType, horizon=horizon)
        mesh, _ = nonlocalMeshFactory(domain, kernel=kernel, boundaryCondition=HOMOGENEOUS_DIRICHLET, **domainParams)
        self.dmAux = dofmapFactory('P1', mesh, NO_BOUNDARY)

    def getIdentifier(self, params):
        keys = ['domain', 'problem', 's', 'noRef', 'element', 'adaptive']
        d = []
        for k in keys:
            try:
                d.append((k, str(getattr(self, k))))
            except KeyError:
                d.append((k, str(params[k])))
        return '-'.join(['fracLaplAdaptive'] + [key + '=' + v for key, v in d])


class nonlocalPoissonProblem(nonlocalBaseProblem):
    def setDriverArgs(self):
        super().setDriverArgs()
        self.setDriverFlag('domain', 'interval', acceptedValues=['gradedInterval', 'square', 'disc', 'gradedDisc', 'discWithIslands'], help='spatial domain')
        self.addParametrizedArg('indicator', [float, float])
        self.addParametrizedArg('polynomial', [int])
        self.setDriverFlag('problem', 'poly-Dirichlet',
                           argInterpreter=self.argInterpreter(['indicator', 'polynomial'], acceptedValues=['poly-Dirichlet', 'polynomial', 'poly-Dirichlet2', 'poly-Dirichlet3',
                                                                                                           'poly-Neumann', 'zeroFlux', 'source', 'constant',
                                                                                                           'exact-sin-Dirichlet', 'exact-sin-Neumann', 'discontinuous']),
                           help="select a problem to solve")
        self.setDriverFlag('noRef', argInterpreter=int)
        self.setDriverFlag('element', acceptedValues=['P1', 'P0', 'P2'], help="finite element space")
        self.setDriverFlag('target_order', -1., help="choose quadrature rule to allow convergence of order h^{target_order}")

    def processCmdline(self, params):
        noRef = params['noRef']
        if noRef is None or noRef <= 0:
            domain = params['domain']
            if domain in ('interval', 'gradedInterval'):
                noRef = 8
            elif domain == 'square':
                noRef = 2
            elif domain in ('disc', 'gradedDisc'):
                noRef = 4
            elif domain == 'discWithIslands':
                noRef = 4
            else:
                raise NotImplementedError(domain)
            params['noRef'] = noRef
        super().processCmdline(params)

    @generates(['mesh_domain', 'mesh_params',
                'tag', 'zeroExterior', 'boundaryCondition',
                'domainIndicator', 'fluxIndicator', 'interactionIndicator',
                'rhs', 'rhsData', 'dirichletData', 'fluxData',
                'analyticSolution', 'exactL2Squared', 'exactHsSquared'])
    def processProblem(self, kernel, domain, problem, normalized):
        kType = kernel.kernelType
        if kType == FRACTIONAL:
            sFun = kernel.s
        else:
            sFun = None
        phiFun = kernel.phi
        self.analyticSolution = None
        self.exactL2Squared = None
        self.exactHsSquared = None
        interaction = kernel.interaction

        if problem in ('poly-Neumann', 'exact-sin-Neumann', 'zeroFlux'):
            self.boundaryCondition = NEUMANN
        elif self.parametrizedArg('indicator').match(problem):
            self.boundaryCondition = HOMOGENEOUS_DIRICHLET
        elif problem in ('source', 'constant'):
            self.boundaryCondition = HOMOGENEOUS_DIRICHLET
        else:
            self.boundaryCondition = DIRICHLET

        mesh_params = {'kernel': kernel, 'boundaryCondition': self.boundaryCondition}
        if domain in ('interval', 'gradedInterval'):
            mesh_domain = domain
            nI = nonlocalMeshFactory.build(mesh_domain, **mesh_params, skipMesh=True)
            self.tag = nI['tag']
            self.zeroExterior = nI['zeroExterior']
            self.domainInteriorIndicator = domainIndicator = nI['domain']
            self.boundaryIndicator = boundaryIndicator = nI['boundary']
            self.interactionInteriorIndicator = interactionIndicator = nI['interaction']
            if problem == 'poly-Dirichlet':
                self.domainIndicator = domainIndicator
                self.fluxIndicator = constant(0)
                self.interactionIndicator = interactionIndicator+boundaryIndicator
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: 1-x[0]**2)
                if ((kType == FRACTIONAL and isinstance(sFun, constFractionalOrder)) or kType in (INDICATOR, PERIDYNAMIC, GAUSSIAN)) and phiFun is None and normalized:
                    self.analyticSolution = Lambda(lambda x: 1-x[0]**2)
            elif self.parametrizedArg('polynomial').match(problem):
                self.domainIndicator = domainIndicator
                self.fluxIndicator = constant(0)
                self.interactionIndicator = interactionIndicator+boundaryIndicator
                self.fluxData = constant(0)
                polyOrder = self.parametrizedArg('polynomial').interpret(problem)[0]
                knownSolution = (((kType == FRACTIONAL and isinstance(sFun, (constFractionalOrder, variableConstFractionalOrder, singleVariableUnsymmetricFractionalOrder))) or
                                  (kType in (INDICATOR, PERIDYNAMIC, GAUSSIAN))) and
                                 phiFun is None and
                                 normalized and
                                 0 <= polyOrder <= 3)
                if polyOrder == 0:
                    self.rhsData = functionFactory('constant', 0.)
                    self.dirichletData = functionFactory('constant', 1.)
                    if knownSolution:
                        self.analyticSolution = self.dirichletData
                elif polyOrder == 1:
                    self.rhsData = functionFactory('constant', 0.)
                    self.dirichletData = functionFactory('x0')
                    if knownSolution:
                        self.analyticSolution = self.dirichletData
                elif polyOrder == 2:
                    self.rhsData = functionFactory('constant', -2)
                    self.dirichletData = functionFactory('x0**2')
                    if knownSolution:
                        self.analyticSolution = self.dirichletData
                elif polyOrder == 3:
                    self.rhsData = -6*functionFactory('x0')
                    self.dirichletData = functionFactory('x0**3')
                    if knownSolution:
                        self.analyticSolution = self.dirichletData
                else:
                    self.rhsData = functionFactory('Lambda', lambda x: -polyOrder*(polyOrder-1)*x[0]**(polyOrder-2))
                    self.dirichletData = functionFactory('Lambda', lambda x: x[0]**polyOrder)
                    if knownSolution:
                        self.analyticSolution = self.dirichletData


            elif problem == 'exact-sin-Dirichlet':
                assert ((kType == INDICATOR) or (kType == FRACTIONAL)) and phiFun is None and normalized

                self.domainIndicator = domainIndicator
                self.fluxIndicator = constant(0)
                self.interactionIndicator = interactionIndicator+boundaryIndicator
                horizonValue = kernel.horizonValue
                scalingValue = kernel.scalingValue

                sin = functionFactory('sin1d')
                if kType == INDICATOR:
                    self.rhsData = -2.*scalingValue * 2*(np.sin(np.pi*horizonValue)/np.pi-horizonValue) * sin
                elif kType == FRACTIONAL:
                    from scipy.integrate import quad
                    assert isinstance(sFun, constFractionalOrder)
                    sBase = sFun.value
                    from scipy.special import gamma

                    def Phi(delta):
                        if delta > 0:
                            fac = delta**(-2*sBase)
                            integral = 0.
                            for k in range(1, 100):
                                integral += fac * (-1)**(k+1) * (np.pi*delta)**(2*k) / (2*k-2*sBase) / gamma(2*k+1)
                            return integral
                        else:
                            return 0.

                    Phi_delta = Phi(horizonValue)
                    self.rhsData = 4 * scalingValue * Phi_delta * sin
                self.fluxData = constant(0)
                self.dirichletData = sin
                self.analyticSolution = sin
            elif problem == 'exact-sin-Neumann':
                assert (kType == FRACTIONAL) and phiFun is None and normalized

                self.domainIndicator = domainIndicator
                self.fluxIndicator = boundaryIndicator+interactionIndicator
                self.interactionIndicator = constant(0.)
                horizonValue = self.kernel.horizonValue
                scalingValue = self.kernel.scalingValue

                sin = functionFactory('sin1d')
                cos = functionFactory('cos1d')
                if kType == FRACTIONAL:
                    from scipy.integrate import quad
                    assert isinstance(sFun, constFractionalOrder)
                    sBase = sFun.value
                    from scipy.special import gamma

                    def Phi(delta):
                        if delta > 0:
                            fac = delta**(-2*sBase)
                            integral = 0.
                            for k in range(1, 100):
                                integral += fac * (-1)**(k+1) * (np.pi*delta)**(2*k) / (2*k-2*sBase) / gamma(2*k+1)
                            return integral
                        else:
                            return 0.

                    Psi = lambda delta_min, delta_max: quad(lambda y: np.sin(np.pi*y)/y**(1+2*sBase), delta_min, delta_max)[0]
                    Phi_delta = Phi(horizonValue)
                    self.rhsData = 4 * scalingValue * Phi_delta * sin

                    def fluxFun(x):
                        dist = 1+horizonValue-abs(x[0])
                        assert dist >= 0
                        if x[0] > 0:
                            return 2 * scalingValue * ((Phi_delta + Phi(dist)) * sin(x) + (Psi(dist, horizonValue)) * cos(x))
                        else:
                            return 2 * scalingValue * ((Phi_delta + Phi(dist)) * sin(x) - (Psi(dist, horizonValue)) * cos(x))

                    self.fluxData = Lambda(fluxFun)
                self.dirichletData = sin
                self.analyticSolution = sin
            elif problem == 'poly-Neumann':
                if kType == FRACTIONAL:
                    assert sFun.max <= 0.5, "RHS is singular, need a special quadrature rule"
                self.domainIndicator = domainIndicator
                self.fluxIndicator = boundaryIndicator+interactionIndicator
                self.interactionIndicator = constant(0.)
                horizonBase = self.kernel.horizon.value

                if kType == FRACTIONAL:
                    sBase = sFun.value

                    def fluxFun(x):
                        # dist = 1+horizonBase-abs(x[0])
                        # assert dist >= 0
                        # return (1+(dist/horizonBase)**(2-2*sBase) - 2*abs(x[0]) * (2-2*sBase)/(1-2*sBase)/horizonBase * (1-(dist/horizonBase)**(1-2*sBase)))
                        dist = 1+horizonBase-abs(x[0])
                        assert dist >= 0
                        return 2*self.kernel.scalingValue * ((2*abs(x[0])/(1-2*sBase)) * (dist**(1-2*sBase)-horizonBase**(1-2*sBase)) + 1/(2-2*sBase) * (dist**(2-2*sBase)+horizonBase**(2-2*sBase)))
                elif kType == PERIDYNAMIC:
                    def fluxFun(x):
                        dist = 1+horizonBase-abs(x[0])
                        assert dist >= 0
                        return 2*self.kernel.scalingValue * (2*abs(x[0]) * (1-abs(x[0])) + 0.5 * (dist**2+horizonBase**2))
                elif kType == INDICATOR:
                    def fluxFun(x):
                        dist = 1+horizonBase-abs(x[0])
                        assert dist >= 0
                        return 2*self.kernel.scalingValue * (abs(x[0]) * (dist**2-horizonBase**2) + 1./3. * (dist**3+horizonBase**3))

                self.rhsData = constant(2)
                self.fluxData = Lambda(fluxFun)
                self.dirichletData = Lambda(lambda x: 1-x[0]**2)
                if ((kType == FRACTIONAL and isinstance(sFun, constFractionalOrder)) or kType != FRACTIONAL) and normalized:
                    self.analyticSolution = Lambda(lambda x: 1-x[0]**2)
            elif problem == 'zeroFlux':
                self.domainIndicator = domainIndicator
                self.fluxIndicator = Lambda(lambda x: 1. if (x[0] > 1) else 0.)
                self.interactionIndicator = interactionIndicator+boundaryIndicator
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: 1-x[0]**2)
            elif self.parametrizedArg('indicator').match(problem):
                self.domainIndicator = domainIndicator
                self.fluxIndicator = constant(0)
                self.interactionIndicator = interactionIndicator+boundaryIndicator
                # self.fluxIndicator = squareIndicator(np.array([1.], dtype=REAL),
                #                                      np.array([1.+horizon-1e-9], dtype=REAL))
                center, width = self.parametrizedArg('indicator').interpret(problem)
                self.rhsData = squareIndicator(np.array([center-width/2], dtype=REAL),
                                               np.array([center+width/2], dtype=REAL))
                self.fluxData = constant(0)
                self.dirichletData = constant(0.)
            elif problem == 'constant':
                self.domainIndicator = domainIndicator
                self.fluxIndicator = constant(0)
                self.interactionIndicator = interactionIndicator+boundaryIndicator
                self.rhsData = constant(1.)
                self.fluxData = constant(0)
                self.dirichletData = constant(0.)
                if (kType == FRACTIONAL) and (isinstance(self.kernel.s, constFractionalOrder) or
                                              isinstance(self.kernel.s, variableConstFractionalOrder)) and self.kernel.horizon == np.inf:
                    self.analyticSolution = functionFactory('solFractional', dim=1, s=self.kernel.s.value)
            elif problem == 'discontinuous':
                self.domainIndicator = domainIndicator
                self.fluxIndicator = constant(0)
                self.interactionIndicator = interactionIndicator+boundaryIndicator

                jumpPoint = 0.2
                horizonBase = self.kernel.horizon.value

                def f1lam(x):
                    v = x[0]-(jumpPoint-0.5)
                    if v < 0.5-horizonBase:
                        return 0.
                    elif v < 0.5:
                        return -(2/horizonBase**2) * (0.5*horizonBase**2-horizonBase+3/8 +
                                                      (2*horizonBase-3/2-np.log(horizonBase))*v +
                                                      (3/2+np.log(horizonBase))*v**2 - (v**2-v)*np.log(0.5-v))
                    elif v < 0.5+horizonBase:
                        return -(2/horizonBase**2) * (0.5*horizonBase**2-horizonBase-3/8 +
                                                      (2*horizonBase+3/2+np.log(horizonBase))*v -
                                                      (3/2+np.log(horizonBase))*v**2 + (v**2-v)*np.log(v-0.5))
                    else:
                        return -2.

                self.rhsData = functionFactory('Lambda', f1lam)
                self.fluxData = constant(0)

                self.dirichletData = functionFactory('Lambda', lambda x: x[0]+(0.5-jumpPoint) if x[0]+(0.5-jumpPoint) < 0.5 else (x[0]+(0.5-jumpPoint))**2)
                if kType == PERIDYNAMIC:
                    self.analyticSolution = self.dirichletData
            else:
                raise NotImplementedError(problem)
        elif domain == 'square':
            mesh_domain = domain
            # mesh_params['uniform'] = True
            if isinstance(sFun, layersFractionalOrder):
                t = np.array(sFun.layerBoundaries)[1:-1]
                mesh_params['preserveLinesHorizontal'] = t.tolist()
            elif isinstance(sFun, leftRightFractionalOrder) or isinstance(phiFun, leftRightTwoPoint):
                mesh_params['preserveLinesVertical'] = [0.]
            nI = nonlocalMeshFactory.build(mesh_domain, skipMesh=True, **mesh_params)
            self.tag = nI['tag']
            self.zeroExterior = nI['zeroExterior']
            self.domainInteriorIndicator = domainIndicator = nI['domain']
            self.boundaryIndicator = boundaryIndicator = nI['boundary']
            self.interactionInteriorIndicator = interactionIndicator = nI['interaction']
            self.domainIndicator = domainIndicator
            self.interactionIndicator = interactionIndicator+boundaryIndicator
            if problem == 'poly-Dirichlet' and isinstance(interaction, ball2):
                self.fluxIndicator = constant(0)
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: 1-x[0]**2)
                if (((kType == FRACTIONAL and isinstance(sFun, constFractionalOrder)) or
                     kType in (INDICATOR, PERIDYNAMIC, GAUSSIAN)) and
                    phiFun is None and
                    normalized):
                    self.analyticSolution = Lambda(lambda x: 1-x[0]**2)
            elif problem == 'poly-Dirichlet' and isinstance(interaction, ellipse):
                aFac = np.sqrt(self.kernel.interaction.aFac2)
                bFac = np.sqrt(self.kernel.interaction.bFac2)
                self.fluxIndicator = constant(0)
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: (1-x[0]**2) * 2/(np.pi*self.kernel.horizon.value**4/4 * aFac**3 * bFac))
                if (kType == INDICATOR and
                    phiFun is None and
                    not normalized):
                    self.analyticSolution = self.dirichletData
            elif problem == 'poly-Dirichlet2' and isinstance(interaction, ellipse):
                aFac = np.sqrt(self.kernel.interaction.aFac2)
                bFac = np.sqrt(self.kernel.interaction.bFac2)
                self.fluxIndicator = constant(0)
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: (1-x[1]**2) * 2/(np.pi*self.kernel.horizon.value**4/4 * aFac * bFac**3))
                if (kType == INDICATOR and
                    phiFun is None and
                    not normalized):
                    self.analyticSolution = self.dirichletData
            elif problem == 'poly-Dirichlet3' and isinstance(interaction, ellipse):
                aFac = np.sqrt(self.kernel.interaction.aFac2)
                bFac = np.sqrt(self.kernel.interaction.bFac2)
                self.fluxIndicator = constant(0)
                self.rhsData = constant(4)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: (1-x[0]**2) * 2/(np.pi*self.kernel.horizon.value**4/4 * aFac**3 * bFac) + (1-x[1]**2) * 2/(np.pi*self.kernel.horizon.value**4/4 * aFac * bFac**3))
                if (kType == INDICATOR and
                    phiFun is None and
                    not normalized):
                    self.analyticSolution = self.dirichletData
            elif problem == 'poly-Neumann':
                self.fluxIndicator = Lambda(lambda x: 1. if (x[0] > 1) else 0.)
                raise NotImplementedError(problem)
            elif problem == 'source':
                self.fluxIndicator = constant(0)
                self.rhsData = (functionFactory.build('radialIndicator', radius=0.3, center=np.array([0.2, 0.6], dtype=REAL)) -
                                functionFactory.build('radialIndicator', radius=0.3, center=np.array([-0.2, -0.6], dtype=REAL)))
                self.fluxData = constant(0)
                self.dirichletData = constant(0)
            elif problem == 'constant':
                self.fluxIndicator = constant(0)
                self.rhsData = constant(1.)
                self.fluxData = constant(0)
                self.dirichletData = constant(0)
            else:
                raise NotImplementedError(problem)
        elif domain in ('disc', 'gradedDisc'):
            mesh_domain = domain
            nI = nonlocalMeshFactory.build(mesh_domain, skipMesh=True, **mesh_params)
            self.tag = nI['tag']
            self.zeroExterior = nI['zeroExterior']
            self.domainInteriorIndicator = domainIndicator = nI['domain']
            self.boundaryIndicator = boundaryIndicator = nI['boundary']
            self.interactionInteriorIndicator = interactionIndicator = nI['interaction']
            self.domainIndicator = domainIndicator+boundaryIndicator
            self.interactionIndicator = interactionIndicator
            if problem == 'poly-Dirichlet':
                self.fluxIndicator = constant(0)
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: 1-x[0]**2)
                if isinstance(sFun, constFractionalOrder) and isinstance(phiFun, constantTwoPoint) and normalized:
                    self.analyticSolution = Lambda(lambda x: 1-x[0]**2)
            elif problem == 'poly-Neumann':
                self.fluxIndicator = Lambda(lambda x: 1. if (x[0] > 1) else 0.)
                raise NotImplementedError(problem)
            elif problem == 'source':
                self.fluxIndicator = constant(0)
                self.rhsData = (functionFactory.build('radialIndicator', radius=0.3, center=np.array([0.2, 0.6], dtype=REAL)) -
                                functionFactory.build('radialIndicator', radius=0.3, center=np.array([-0.2, -0.6], dtype=REAL)))
                self.fluxData = constant(0)
                self.dirichletData = constant(0)
            elif problem == 'constant':
                self.fluxIndicator = constant(0)
                self.rhsData = constant(1.)
                self.fluxData = constant(0)
                self.dirichletData = constant(0)
                if (kType == FRACTIONAL) and (isinstance(self.kernel.s, constFractionalOrder) or
                                              isinstance(self.kernel.s, variableConstFractionalOrder)):
                    self.analyticSolution = functionFactory('solFractional', dim=2, s=self.kernel.s.value)
            else:
                raise NotImplementedError(problem)
        elif domain == 'discWithIslands':
            mesh_domain = domain
            nI = nonlocalMeshFactory.build(mesh_domain, skipMesh=True, **mesh_params)
            self.tag = nI['tag']
            self.zeroExterior = nI['zeroExterior']
            self.domainInteriorIndicator = domainIndicator = nI['domain']
            self.boundaryIndicator = boundaryIndicator = nI['boundary']
            self.interactionInteriorIndicator = interactionIndicator = nI['interaction']
            self.domainIndicator = domainIndicator+boundaryIndicator
            self.interactionIndicator = interactionIndicator
            if problem == 'poly-Dirichlet':
                self.fluxIndicator = constant(0)
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: 1-x[0]**2)
                if isinstance(sFun, constFractionalOrder) and isinstance(phiFun, constantTwoPoint) and normalized:
                    self.analyticSolution = Lambda(lambda x: 1-x[0]**2)
            elif problem == 'poly-Neumann':
                self.fluxIndicator = Lambda(lambda x: 1. if (x[0] > 1) else 0.)
                raise NotImplementedError(problem)
            elif problem == 'source':
                self.fluxIndicator = constant(0)
                self.rhsData = (functionFactory.build('radialIndicator', radius=0.3, center=np.array([0.2, 0.6], dtype=REAL)) -
                                functionFactory.build('radialIndicator', radius=0.3, center=np.array([-0.2, -0.6], dtype=REAL)))
                self.fluxData = constant(0)
                self.dirichletData = constant(0)
            elif problem == 'constant':
                self.fluxIndicator = constant(0)
                self.rhsData = constant(1.)
                self.fluxData = constant(0)
                self.dirichletData = constant(0)
            else:
                raise NotImplementedError(problem)
        else:
            raise NotImplementedError(domain)

        self.mesh_domain = mesh_domain
        self.mesh_params = mesh_params

        # should be equal to the forcing term within the domain and equal to
        # the flux term in the interaction region
        self.rhs = (indicatorFunctor(self.rhsData, self.domainIndicator) +
                    indicatorFunctor(self.fluxData, self.fluxIndicator))

    @generates('eta')
    def getApproximationParams(self, dim, kernel, element, target_order):
        element = str2DoFMapOrder(element)
        kType = kernel.kernelType
        if kType == FRACTIONAL:
            s = kernel.s

            if dim == 1:
                if target_order <= 0.:
                    if s is not None:
                        target_order = (1+element-s.min)/dim
                    else:
                        target_order = 2.
            else:
                if self.target_order <= 0.:
                    target_order = 1/dim
                if element == 2:
                    raise NotImplementedError()
            self.directlySetWithoutChecks('target_order', target_order)
        else:
            if target_order <= 0.:
                target_order = 2/dim
                self.directlySetWithoutChecks('target_order', target_order)
        if dim == 1:
            self.eta = 1.
        else:
            self.eta = 3.

    @generates('mesh')
    def buildMesh(self, mesh_domain, mesh_params):
        self.mesh, _ = nonlocalMeshFactory.build(mesh_domain, **mesh_params)

    def getIdentifier(self, params):
        keys = ['domain', 'problem', 's', 'horizon', 'phi', 'noRef']
        d = []
        for k in keys:
            try:
                d.append((k, str(getattr(self, k))))
            except KeyError:
                d.append((k, str(params[k])))
        return '-'.join(['nonlocal'] + [key + '=' + v for key, v in d])


class transientFractionalProblem(fractionalLaplacianProblem):
    def __init__(self, driver, useMulti=False):
        super().__init__(driver, useMulti)

    def setDriverArgs(self):
        super().setDriverArgs()
        self.setDriverFlag('finalTime', 1.0, help='final time')

    @generates(['mesh_domain', 'mesh_params',
                'tag', 'zeroExterior', 'boundaryCondition',
                'domainIndicator', 'fluxIndicator', 'interactionIndicator',
                'rhs', 'rhsData', 'dirichletData',
                'analyticSolution', 'exactL2Squared', 'exactHsSquared',
                'initial'])
    def processProblem(self, kernel, dim, domain, domainParams, problem, normalized):
        super().processProblem(kernel, dim, domain, domainParams, problem, normalized)

        steadyStateRHS = self.rhs
        steadyStateRHSdata = self.rhsData
        steadyStateDirichletData = self.dirichletData
        steadyStateFluxData = self.fluxData
        steadyStateAnalyticSolution = self.analyticSolution
        steadyStateexactL2Squared = self.exactL2Squared
        steadyStateexactHsSquared = self.exactHsSquared

        if steadyStateAnalyticSolution is not None:
            self.analyticSolution = lambda t: np.cos(t)*steadyStateAnalyticSolution
            self.rhs = lambda t: -np.sin(t)*steadyStateAnalyticSolution + np.cos(t)*steadyStateRHS
            self.rhsData = lambda t: -np.sin(t)*steadyStateAnalyticSolution + np.cos(t)*steadyStateRHSdata
        else:
            self.analyticSolution = None
            self.rhs = lambda t: np.cos(t)*steadyStateRHS
            self.rhsData = lambda t: np.cos(t)*steadyStateRHSdata
        if steadyStateexactL2Squared is not None:
            self.exactL2Squared = lambda t: np.cos(t)**2 * steadyStateexactL2Squared
        else:
            self.exactL2Squared = None
        if steadyStateexactHsSquared is not None:
            self.exactHsSquared = lambda t: np.cos(t)**2 * steadyStateexactHsSquared
        else:
            self.exactHsSquared = None

        if self.analyticSolution is not None:
            self.initial = self.analyticSolution(0.)
        else:
            self.initial = functionFactory('constant', 0.)

        if steadyStateDirichletData is not None:
            self.dirichletData = lambda t: np.cos(t)*steadyStateDirichletData
        if steadyStateFluxData is not None:
            self.fluxData = lambda t: np.cos(t)*steadyStateFluxData

    def report(self, group):
        super().report(group)
        group.add('finalTime', self.finalTime)


class nonlocalInterfaceProblem(problem):
    def setDriverArgs(self):
        self.setDriverFlag('domain', acceptedValues=['doubleInterval', 'doubleSquare'])
        self.setDriverFlag('problem', acceptedValues=['polynomial-variableSolJump-fluxJump',
                                                      'polynomial-noSolJump-noFluxJump',
                                                      'exact-sin-variableSolJump-fluxJump',
                                                      'exact-sin1d-variableSolJump-fluxJump',
                                                      'sin',
                                                      'sin-fixedSolJump-fluxJump',
                                                      'sin-variableSolJump-fluxJump',
                                                      'sin-nojump',
                                                      'patch-test',
                                                      'sin1d-fixedSolJump-fluxJump',])
        self.setDriverFlag('element', acceptedValues=['P1', 'P0'])
        self.setDriverFlag('kernel1Type', acceptedValues=['fractional', 'indicator', 'peridynamic'])
        self.setDriverFlag('kernel2Type', acceptedValues=['fractional', 'indicator', 'peridynamic'])
        self.setDriverFlag('horizon1', 0.1)
        self.setDriverFlag('horizon2', 0.2)
        self.setDriverFlag('hTarget', 0.05)

        self.setDriverFlag('s11', 0.4)
        self.setDriverFlag('s12', 0.4)
        self.setDriverFlag('s21', 0.7)
        self.setDriverFlag('s22', 0.7)

        self.setDriverFlag('coeff11', 1.)
        self.setDriverFlag('coeff12', 1.)
        self.setDriverFlag('coeff21', 1.)
        self.setDriverFlag('coeff22', 1.)

    @generates(['dim',
                'kernel1',
                'kernel2',
                'horizon1',
                'horizon2',
                'mesh',
                'subdomainIndicator1',
                'subdomainIndicator2',
                'localSubdomainIndicator1',
                'localInterfaceIndicator',
                'localSubdomainIndicator2',
                'domainIndicator1',
                'domainIndicator2',
                'interfaceIndicator',
                'dirichletIndicator1',
                'dirichletIndicator2',
                'sol_1',
                'sol_2',
                'diri_left',
                'diri_right',
                'forcing_left',
                'forcing_right',
                'sol_jump',
                'flux_jump',
                'local_L2ex_left',
                'local_L2ex_right',
                'local_H10ex_left',
                'local_H10ex_right'])
    def processProblem(self, domain, problem, element, kernel1Type, kernel2Type, horizon1, horizon2, hTarget, s11, s12, s21, s22, coeff11, coeff12, coeff21, coeff22):
        if domain == 'doubleInterval':
            dim = 1
            a, b, c = 0, 2, 1
        elif domain == 'doubleSquare':
            dim = 2
            ax = 0
            ay = 0
            bx = 2
            by = 1
            cx = 1
        else:
            raise NotImplementedError()

        kType1 = getKernelEnum(kernel1Type)
        kType2 = getKernelEnum(kernel2Type)

        if (s11 == s12) and (s21 == s22):
            s1 = constFractionalOrder(s11)
            s2 = constFractionalOrder(s22)
        elif (s11 == s21) and (s12 == s22):
            assert dim == 1
            s1 = leftRightFractionalOrder(s11, s22, s11, s11, interface=c)
            s2 = leftRightFractionalOrder(s11, s22, s22, s22, interface=c)
        else:
            raise NotImplementedError()

        if (coeff11 == coeff12 == 1.0):
            phi1 = None
        elif (coeff11 == coeff12):
            phi1 = constantTwoPoint(coeff11)
        else:
            assert dim == 1
            phi1 = leftRightTwoPoint(coeff11, coeff12, coeff11, coeff11, interface=c)

        if (coeff22 == coeff21 == 1.0):
            phi2 = None
        elif (coeff22 == coeff21):
            phi2 = constantTwoPoint(coeff22)
        else:
            assert dim == 1
            phi2 = leftRightTwoPoint(coeff21, coeff22, coeff22, coeff22, interface=c)

        if dim == 1:
            phi1 = interfaceTwoPoint(horizon1, horizon2, True, interface=c)
            phi2 = interfaceTwoPoint(horizon1, horizon2, False, interface=c)
        elif dim == 2:
            phi1 = interfaceTwoPoint(horizon1, horizon2, True, interface=cx)
            phi2 = interfaceTwoPoint(horizon1, horizon2, False, interface=cx)

        kernel1 = getKernel(dim=dim, kernel=kType1, s=s1, horizon=constant(horizon1), phi=phi1)
        kernel2 = getKernel(dim=dim, kernel=kType2, s=s2, horizon=constant(horizon2), phi=phi2)

        self.mult = constant(1/(horizon1+horizon2))

        local_L2ex_left = None
        local_L2ex_right = None
        local_H10ex_left = None
        local_H10ex_right = None

        if domain == 'doubleInterval':
            from PyNucleus_fem.mesh import doubleIntervalWithInteractions
            mesh = doubleIntervalWithInteractions(horizon1=horizon1, horizon2=horizon2, h=hTarget)

            eps = 1e-9
            subdomainIndicator1 = squareIndicator(np.array([a-horizon1+eps], dtype=REAL),
                                                  np.array([c+horizon1-eps], dtype=REAL))
            subdomainIndicator2 = squareIndicator(np.array([c-horizon2+eps], dtype=REAL),
                                                  np.array([b+horizon2-eps], dtype=REAL))
            localSubdomainIndicator1 = squareIndicator(np.array([a+eps], dtype=REAL),
                                                       np.array([c-eps], dtype=REAL))
            localInterfaceIndicator = squareIndicator(np.array([c-eps], dtype=REAL),
                                                      np.array([c+eps], dtype=REAL))
            localSubdomainIndicator2 = squareIndicator(np.array([c+eps], dtype=REAL),
                                                       np.array([b-eps], dtype=REAL))
            domainIndicator1 = squareIndicator(np.array([a+eps], dtype=REAL),
                                               np.array([c-horizon2-eps], dtype=REAL))
            domainIndicator2 = squareIndicator(np.array([c+horizon1+eps], dtype=REAL),
                                               np.array([b-eps], dtype=REAL))
            interfaceIndicator = squareIndicator(np.array([c-horizon2-eps], dtype=REAL),
                                                 np.array([c+horizon1+eps], dtype=REAL))
            dirichletIndicator1 = constant(1.)-domainIndicator1-interfaceIndicator
            dirichletIndicator2 = constant(1.)-domainIndicator2-interfaceIndicator

            if problem == 'polynomial-noSolJump-noFluxJump':
                assert kType1 == INDICATOR
                assert kType2 == INDICATOR
                sol_1 = Lambda(lambda x: 1-(1-x[0])**2)
                sol_2 = Lambda(lambda x: 1-(1-x[0])**2)
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = constant(2*coeff11)
                forcing_right = constant(2*coeff22)
                # sol_jump = constant(-1.)
                sol_jump = sol_2-sol_1

                scaling1 = kernel1.scalingValue
                scaling2 = kernel2.scalingValue

                def flux_left_lam(x):
                    dist = 1+horizon1-x[0]
                    return 2*scaling1 * ((x[0]-1) * (dist**2-horizon1**2) + 1/3 * (horizon1**3 + dist**3))

                def flux_right_lam(x):
                    dist = x[0]-(1-horizon2)
                    return 2*scaling2 * ((x[0]-1) * (horizon2**2-dist**2) + 1/3 * (horizon2**3 + dist**3))

                flux_left = Lambda(flux_left_lam)
                flux_right = Lambda(flux_right_lam)
                flux_jump = (horizon1+horizon2)*(indicatorFunctor(flux_right, localSubdomainIndicator1) + indicatorFunctor(flux_left, localSubdomainIndicator2))
                # flux_jump = constant(0.)
            elif problem == 'patch-test':
                sol_1 = Lambda(lambda x: x[0])
                sol_2 = Lambda(lambda x: x[0])
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = constant(0.)
                forcing_right = constant(0.)
                sol_jump = sol_2-sol_1
                flux_left = constant(0.)
                flux_right = constant(0.)
                self.mult = constant(1.)
                flux_jump = constant(0.)
            elif problem == 'polynomial-variableSolJump-fluxJump':
                sol_1 = Lambda(lambda x: x[0]**2)
                sol_2 = Lambda(lambda x: (x[0]-1)**2)
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = constant(-2*coeff11)
                forcing_right = constant(-2*coeff22)
                # sol_jump = constant(-1.)
                sol_jump = sol_2-sol_1

                def flux_left_lam(x):
                    dist = 1+horizon1-x[0]
                    return -2*kernel1.scalingValue * (x[0] * (dist**2-horizon1**2) + 1/3 * (horizon1**3 + dist**3))

                def flux_right_lam(x):
                    dist = x[0]-1+horizon2
                    return -2*kernel2.scalingValue * ((x[0]-1) * (horizon2**2-dist**2) + 1/3 * (horizon2**3 + dist**3))

                flux_left = Lambda(flux_left_lam)
                flux_right = Lambda(flux_right_lam)
                flux_jump = (horizon1+horizon2)*(indicatorFunctor(flux_right, localSubdomainIndicator1) + indicatorFunctor(flux_left, localSubdomainIndicator2))
            elif problem == 'polynomial-nojump':
                sol_1 = Lambda(lambda x: (x[0]-1)**2)
                sol_2 = Lambda(lambda x: (x[0]-1)**2)
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = constant(-2*coeff11)
                forcing_right = constant(-2*coeff22)
                sol_jump = constant(0)
                flux_jump = constant(0)
            elif problem == 'exact-sin-variableSolJump-fluxJump':
                # the nonlocal problem has a know exact solution
                assert kType1 in (INDICATOR, FRACTIONAL)
                assert kType2 in (INDICATOR, FRACTIONAL)
                assert coeff11 == coeff12
                assert coeff21 == coeff22
                sin = functionFactory('sin1d')
                # cos = functionFactory('cos1d')
                one = functionFactory('constant', 1)
                sol_1 = sin
                sol_2 = one - sin
                diri_left = sol_1
                diri_right = sol_2

                sol_jump = sol_2-sol_1

                # Get scaling values for interactions within subdomains
                kernel1(np.array([0.5*(a+c)]),
                        np.array([0.6*a+0.4*c]))
                kernel2(np.array([0.5*(b+c)]),
                        np.array([0.6*b+0.4*c]))
                scaling1 = kernel1.scalingValue
                scaling2 = kernel2.scalingValue

                from scipy.integrate import quad

                if kType1 == INDICATOR:
                    forcing_left = -coeff11*(2.*scaling1) * 2*(np.sin(np.pi*horizon1)/np.pi-horizon1) * sin
                elif kType1 == FRACTIONAL:
                    assert isinstance(kernel1.s, constFractionalOrder)
                    sBase1 = kernel1.s.value
                    from scipy.special import gamma

                    def Phi1(delta):
                        if delta > 0:
                            fac = delta**(-2*sBase1)
                            integral = 0.
                            for k in range(1, 100):
                                integral += fac * (-1)**(k+1) * (np.pi*delta)**(2*k) / (2*k-2*sBase1) / gamma(2*k+1)
                            return integral
                        else:
                            return 0.

                    forcing_left = 4 * scaling1 * Phi1(horizon1) * sin

                def flux_left_lam(x):
                    # assert c < x[0] < c+horizon1
                    u1x = sol_1(x)
                    u2x = sol_2(x)
                    I = 0.
                    if x[0]-horizon1 < c-horizon2:
                        I += 2. * quad(lambda y: (u1x-sol_1(np.array([y]))) * kernel1(x, np.array([y])), x[0]-horizon1, c-horizon2)[0]
                    if max(c-horizon2, x[0]-horizon1) < c:
                        I += 2. * quad(lambda y: (u1x-sol_1(np.array([y]))) * kernel1(x, np.array([y])), max(c-horizon2, x[0]-horizon1), c)[0]
                    if max(c-horizon2, x[0]-horizon2) < c:
                        I -= 2. * quad(lambda y: (u2x-sol_2(np.array([y]))) * kernel2(x, np.array([y])), max(c-horizon2, x[0]-horizon2), c)[0]
                    return I

                if kType2 == INDICATOR:
                    forcing_right = -coeff22*(2.*scaling2) * 2*(np.sin(np.pi*horizon2)/np.pi-horizon2) * (-sin)
                elif kType2 == FRACTIONAL:
                    assert isinstance(kernel2.s, constFractionalOrder)
                    sBase2 = kernel2.s.value
                    from scipy.special import gamma

                    def Phi2(delta):
                        if delta > 0:
                            fac = delta**(-2*sBase2)
                            integral = 0.
                            for k in range(1, 100):
                                integral += fac * (-1)**(k+1) * (np.pi*delta)**(2*k) / (2*k-2*sBase2) / gamma(2*k+1)
                            return integral
                        else:
                            return 0.

                    forcing_right = 4 * scaling2 * Phi2(horizon2) * (-sin)

                def flux_right_lam(x):
                    # assert c-horizon2 < x[0] < c
                    u1x = sol_1(x)
                    u2x = sol_2(x)
                    I = 0.
                    if c+horizon1 < x[0]+horizon2:
                        I += 2. * quad(lambda y: (u2x-sol_2(np.array([y]))) * kernel2(x, np.array([y])), c+horizon1, x[0]+horizon2)[0]
                    if c < min(c+horizon1, x[0]+horizon2):
                        I += 2. * quad(lambda y: (u2x-sol_2(np.array([y]))) * kernel2(x, np.array([y])), c, min(c+horizon1, x[0]+horizon2))[0]
                    if c < min(c+horizon1, x[0]+horizon1):
                        I -= 2. * quad(lambda y: (u1x-sol_1(np.array([y]))) * kernel1(x, np.array([y])), c, min(c+horizon1, x[0]+horizon1))[0]
                    return I

                flux_left = Lambda(flux_left_lam)
                flux_right = Lambda(flux_right_lam)
                self.mult = constant(1.)
                flux_jump = indicatorFunctor(flux_right, localSubdomainIndicator1) + indicatorFunctor(flux_left, localSubdomainIndicator2)

                self.nonlocal_L2ex_left = 0.5
                self.nonlocal_L2ex_right = 1.5+4/np.pi
            elif problem == 'sin-fixedSolJump-fluxJump':
                # the local problem has a know exact solution
                sin = functionFactory('sin1d')
                one = functionFactory('constant', 1)
                sol_1 = sin
                sol_2 = one-2*sin
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = coeff11 * np.pi**2 * sin
                forcing_right = -2*coeff22 * np.pi**2 * sin
                sol_jump = one
                flux_jump = constant(-np.pi*coeff11 - 2*np.pi*coeff22)
                local_L2ex_left = 0.5
                local_L2ex_right = 3.+8/np.pi
                local_H10ex_left = np.pi**2 * coeff11 * 0.5
                local_H10ex_right = np.pi**2 * coeff22 * (2.0 + 4/np.pi)
            elif problem == 'sin-variableSolJump-fluxJump':
                # the local problem has a know exact solution
                sin = functionFactory('sin1d')
                one = functionFactory('constant', 1)
                sol_1 = sin
                sol_2 = one-2*sin
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = coeff11 * np.pi**2 * sin
                forcing_right = -2*coeff22 * np.pi**2 * sin
                sol_jump = sol_2-sol_1
                flux_jump = constant(-np.pi*coeff11 - 2*np.pi*coeff22)
                local_L2ex_left = 0.5
                local_L2ex_right = 3.+8/np.pi
                local_H10ex_left = np.pi**2 * coeff11 * 0.5
                local_H10ex_right = np.pi**2 * coeff22 * (2.0 + 4/np.pi)
            elif problem == 'sin-nojump':
                sin = functionFactory('sin1d')
                sol_1 = sin * (1./coeff11)
                sol_2 = sin * (1./coeff22)
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = np.pi**2 * sin
                forcing_right = np.pi**2 * sin
                sol_jump = constant(0)
                flux_jump = constant(0)
            else:
                raise NotImplementedError(problem)

        elif domain == 'doubleSquare':
            from PyNucleus_fem.mesh import doubleSquareWithInteractions
            mesh = doubleSquareWithInteractions(horizon1=horizon1, horizon2=horizon2, h=hTarget)

            eps = 1e-9
            subdomainIndicator1 = (squareIndicator(np.array([ax-horizon1-eps, ay-horizon1-eps], dtype=REAL),
                                                   np.array([cx+eps, by+horizon1+eps], dtype=REAL)) +
                                   squareIndicator(np.array([cx, ay], dtype=REAL),
                                                   np.array([cx+horizon1+eps, by], dtype=REAL)) +
                                   radialIndicator(horizon1+eps, np.array([cx, ay], dtype=REAL)) +
                                   radialIndicator(horizon1+eps, np.array([cx, by], dtype=REAL)))

            subdomainIndicator2 = (squareIndicator(np.array([cx-eps, -horizon2-eps], dtype=REAL),
                                                   np.array([bx+horizon2+eps, by+horizon2+eps], dtype=REAL)) +
                                   squareIndicator(np.array([cx-horizon2-eps, ay], dtype=REAL),
                                                   np.array([cx, by], dtype=REAL)) +
                                   radialIndicator(horizon2+eps, np.array([cx, ay], dtype=REAL)) +
                                   radialIndicator(horizon2+eps, np.array([cx, by], dtype=REAL)))

            localSubdomainIndicator1 = squareIndicator(np.array([ax+eps, ay+eps], dtype=REAL),
                                                       np.array([cx-eps, by-eps], dtype=REAL))
            localInterfaceIndicator = squareIndicator(np.array([cx-eps, ay+eps], dtype=REAL),
                                                      np.array([cx+eps, by-eps], dtype=REAL))
            localSubdomainIndicator2 = squareIndicator(np.array([cx+eps, ay+eps], dtype=REAL),
                                                       np.array([bx-eps, by-eps], dtype=REAL))
            domainIndicator1 = squareIndicator(np.array([ax+eps, ay+eps], dtype=REAL),
                                               np.array([cx-horizon2-eps, by-eps], dtype=REAL))
            domainIndicator2 = squareIndicator(np.array([cx+horizon1+eps, ay+eps], dtype=REAL),
                                               np.array([bx-eps, by-eps], dtype=REAL))
            interfaceIndicator = squareIndicator(np.array([cx-horizon2-eps, ay+eps], dtype=REAL),
                                                 np.array([cx+horizon1+eps, by-eps], dtype=REAL))
            dirichletIndicator1 = constant(1.)-domainIndicator1-interfaceIndicator
            dirichletIndicator2 = constant(1.)-domainIndicator2-interfaceIndicator

            def bnds(x1, x2, y1, horizon):
                r = horizon**2-(y1-x1)**2
                if r > 0:
                    r = np.sqrt(r)
                    lim = (max(x2-r, ay), min(x2+r, by))
                else:
                    lim = (x2, x2)
                return lim

            IJ1 = [lambda y1, x1, x2, horizon: bnds(x1, x2, y1, horizon),
                   lambda x1, x2, horizon: (cx, cx+horizon1)]

            IJ2 = [lambda y1, x1, x2, horizon: bnds(x1, x2, y1, horizon),
                   lambda x1, x2, horizon: (max(x1-horizon, cx-horizon2), min(x1+horizon, cx))]

            OmegaJ2 = [lambda y1, x1, x2, horizon: bnds(x1, x2, y1, horizon),
                       lambda x1, x2, horizon: (cx+horizon1, cx+horizon2)]

            if problem == 'polynomial-noSolJump-noFluxJump':
                assert kType1 == INDICATOR
                assert kType2 == INDICATOR
                sol_1 = Lambda(lambda x: 1-(1-x[0])**2)
                sol_2 = Lambda(lambda x: 1-(1-x[0])**2)
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = constant(2*coeff11)
                forcing_right = constant(2*coeff22)
                sol_jump = sol_2-sol_1

                scaling1 = kernel1.scalingValue
                scaling2 = kernel2.scalingValue

                def flux_left_lam(x):
                    dist = 1+horizon1-x[0]
                    return 4*scaling1 * (-2/3*(x[0]-1) * (horizon1**2-dist**2)**(3/2) + 1/8 * (np.sqrt(horizon1**2 - dist**2) * dist * (2*dist**2 - horizon1**2)) + horizon1**4/8 * (np.arcsin(dist/horizon1)-np.arcsin(-1)))

                def flux_right_lam(x):
                    dist = x[0]-(1-horizon2)
                    return 4*scaling2 * (-2/3*(x[0]-1) * (-1)*(horizon2**2-dist**2)**(3/2) + 1/8 * (np.sqrt(horizon2**2 - dist**2) * dist * (2*dist**2 - horizon2**2)) + horizon2**4/8 * (np.arcsin(1)-np.arcsin(-dist/horizon2)))

                flux_left = Lambda(flux_left_lam)
                flux_right = Lambda(flux_right_lam)
                self.mult = constant(1.)
                flux_jump = indicatorFunctor(flux_right, localSubdomainIndicator1) + indicatorFunctor(flux_left, localSubdomainIndicator2)
                # flux_jump = constant(0.)
            elif problem == 'polynomial':
                sol_1 = Lambda(lambda x: x[0]**2)
                sol_2 = Lambda(lambda x: (x[0]-1)**2)
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = constant(-2)
                forcing_right = constant(-2)
                # sol_jump = sol_2-sol_1
                sol_jump = constant(-1.)
                flux_jump = constant(2)
            elif problem == 'sin':
                sol_1 = Lambda(lambda x: np.sin(np.pi*x[0]))
                sol_2 = Lambda(lambda x: np.sin(np.pi*(x[0]-1)))
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = Lambda(lambda x: np.pi**2*np.sin(np.pi*x[0])*coeff11)
                forcing_right = Lambda(lambda x: np.pi**2*np.sin(np.pi*(x[0]-1))*coeff22)
                sol_jump = constant(0)
                flux_jump = constant(-np.pi*coeff11 - np.pi*coeff22)
            elif problem == 'sin1d-fixedSolJump-fluxJump':
                # the local problem has a know exact solution
                sin = functionFactory('sin1d')
                one = functionFactory('constant', 1)
                sol_1 = sin
                sol_2 = one-2*sin
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = coeff11 * np.pi**2 * sin
                forcing_right = -2*coeff22 * np.pi**2 * sin
                sol_jump = one
                flux_jump = constant(-np.pi*coeff11 - 2*np.pi*coeff22)
                local_L2ex_left = 0.5
                local_L2ex_right = 3.+8/np.pi
                local_H10ex_left = np.pi**2 * coeff11 * 0.5
                local_H10ex_right = np.pi**2 * coeff22 * (2.0 + 4/np.pi)
            elif problem == 'sin-fixedSolJump-fluxJump':
                # the local problem has a know exact solution
                sin2d = functionFactory('Lambda', lambda x: np.sin(np.pi*x[0])*np.sin(2*np.pi*x[1]))
                sin = functionFactory('sin2d')
                one = functionFactory('constant', 1)
                sol_1 = 2*one+2*sin2d
                sol_2 = one-sin
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = coeff11 * 2*5*np.pi**2 * sin2d
                forcing_right = -coeff22 * 2*np.pi**2 * sin
                sol_jump = -one
                flux_jump = -2*np.pi*coeff11 * functionFactory('Lambda', lambda x: np.sin(2*np.pi*x[1])) - np.pi*coeff22 * functionFactory('Lambda', lambda x: np.sin(np.pi*x[1]))
                local_L2ex_left = 5.
                local_L2ex_right = 1.25 + 8./np.pi**2
                local_H10ex_left = np.pi**2 * coeff11 * 5
                local_H10ex_right = np.pi**2 * coeff22 * 0.5
            elif problem == 'sin-variableSolJump-fluxJump':
                # the local problem has a know exact solution
                sin2d = functionFactory('Lambda', lambda x: np.sin(np.pi*x[0])*np.sin(2*np.pi*x[1]))
                sin = functionFactory('sin2d')
                one = functionFactory('constant', 1)
                sol_1 = 2*one+2*sin2d
                sol_2 = one-sin
                diri_left = sol_1
                diri_right = sol_2
                forcing_left = coeff11 * 2*5*np.pi**2 * sin2d
                forcing_right = -coeff22 * 2*np.pi**2 * sin
                sol_jump = -sin-one-2*sin2d
                flux_jump = -2*np.pi*coeff11 * functionFactory('Lambda', lambda x: np.sin(2*np.pi*x[1])) - np.pi*coeff22 * functionFactory('Lambda', lambda x: np.sin(np.pi*x[1]))
                local_L2ex_left = 5.
                local_L2ex_right = 1.25 + 8./np.pi**2
                local_H10ex_left = np.pi**2 * coeff11 * 5
                local_H10ex_right = np.pi**2 * coeff22 * 0.5
            elif problem == 'exact-sin1d-variableSolJump-fluxJump':
                # the nonlocal problem has a know exact solution
                assert kType1 in (INDICATOR, FRACTIONAL)
                assert kType2 in (INDICATOR, FRACTIONAL)
                assert coeff11 == coeff12
                assert coeff21 == coeff22
                sin = functionFactory('sin1d')
                # cos = functionFactory('cos1d')
                one = functionFactory('constant', 1)
                sol_1 = sin
                sol_2 = one - sin
                diri_left = sol_1
                diri_right = sol_2

                sol_jump = sol_2-sol_1

                # Get scaling values for interactions within subdomains
                kernel1(np.array([0.5*(ax+cx), 0.5]),
                        np.array([0.6*ax+0.4*cx, 0.5]))
                kernel2(np.array([0.5*(bx+cx), 0.5]),
                        np.array([0.6*bx+0.4*cx, 0.5]))
                scaling1 = kernel1.scalingValue
                scaling2 = kernel2.scalingValue

                from scipy.integrate import quad, nquad
                from scipy.special import jv

                from . strongForm import getStrongIntegrand

                fun1 = getStrongIntegrand(sol_1, kernel1, True)
                fun2 = getStrongIntegrand(sol_2, kernel2, True)

                epsabs = 1e-4
                epsrel = 1e-4

                def int2_1(x, y):
                    r = np.sqrt(horizon1**2-(x[0]-y)**2)
                    J = 0.
                    J += min(r, 1.-x[1])
                    J += min(r, x[1])
                    return J

                def int2_2(x, y):
                    r = np.sqrt(horizon2**2-(x[0]-y)**2)
                    J = 0.
                    J += min(r, 1.-x[1])
                    J += min(r, x[1])
                    return J

                if kType1 == INDICATOR:
                    forcing_left = coeff11*(2.*scaling1) * (np.pi*horizon1**2 - 2*horizon1*jv(1., horizon1*np.pi)) * sin

                    def flux_left_lam(x):
                        # x \in I^J_1
                        # assert cx < x[0] < cx+horizon1
                        u1x = sol_1(x)
                        u2x = sol_2(x)

                        I = 0.
                        if x[0]-horizon1 < cx-horizon2:
                            # Omega^J_1
                            I += 2. * quad(lambda y: (u1x-sol_1(np.array([y, x[1]]))) * int2_1(x, y) * kernel1(x, np.array([y, x[1]])), x[0]-horizon1, cx-horizon2)[0]
                        if max(cx-horizon2, x[0]-horizon1) < cx:
                            # I^J_2
                            I += 2. * quad(lambda y: (u1x-sol_1(np.array([y, x[1]]))) * int2_1(x, y) * kernel1(x, np.array([y, x[1]])), max(cx-horizon2, x[0]-horizon1), cx)[0]
                        if max(cx-horizon2, x[0]-horizon2) < cx:
                            # I^J_2
                            I -= 2. * quad(lambda y: (u2x-sol_2(np.array([y, x[1]]))) * int2_2(x, y) * kernel2(x, np.array([y, x[1]])), max(cx-horizon2, x[0]-horizon2), cx)[0]
                        return I

                elif kType1 == FRACTIONAL:
                    assert isinstance(kernel1.s, constFractionalOrder)
                    sBase1 = kernel1.s.value

                    fac1 = nquad(lambda rho, theta: (1-np.cos(np.pi*rho*np.cos(theta))) * rho**(-1-2*sBase1),
                                 [(0, horizon1), (0, 2*np.pi)])[0]
                    forcing_left = 2*scaling1 * fac1 * sin

                    def flux_left_lam(x):
                        # x in IJ1
                        # y in IJ2 \cap B(x, horizon1)
                        I1, err1 = nquad(fun1, IJ2, (x[0], x[1], horizon1), opts={'epsabs': epsabs, 'epsrel': epsrel})
                        # y in IJ2 \cap B(x, horizon2)
                        I2, err2 = nquad(fun2, IJ2, (x[0], x[1], horizon2), opts={'epsabs': epsabs, 'epsrel': epsrel})
                        return I1-I2

                if kType2 == INDICATOR:
                    forcing_right = coeff22*(2.*scaling2) * (np.pi*horizon2**2 - 2*horizon2*jv(1., horizon2*np.pi)) * (-sin)

                    def flux_right_lam(x):
                        # x \in I^J_2
                        # assert cx-horizon2 < x[0] < cx
                        u1x = sol_1(x)
                        u2x = sol_2(x)
                        I = 0.
                        if cx+horizon1 < x[0]+horizon2:
                            # Omega^J_2
                            I += 2. * quad(lambda y: (u2x-sol_2(np.array([y, x[1]]))) * int2_2(x, y) * kernel2(x, np.array([y, x[1]])), cx+horizon1, x[0]+horizon2)[0]
                        if cx < min(cx+horizon1, x[0]+horizon2):
                            # I^J_1
                            I += 2. * quad(lambda y: (u2x-sol_2(np.array([y, x[1]]))) * int2_2(x, y) * kernel2(x, np.array([y, x[1]])), cx, min(cx+horizon1, x[0]+horizon2))[0]
                        if cx < min(cx+horizon1, x[0]+horizon1):
                            # I^J_1
                            I -= 2. * quad(lambda y: (u1x-sol_1(np.array([y, x[1]]))) * int2_1(x, y) * kernel1(x, np.array([y, x[1]])), cx, min(cx+horizon1, x[0]+horizon1))[0]
                        return I

                elif kType2 == FRACTIONAL:
                    assert isinstance(kernel2.s, constFractionalOrder)
                    sBase2 = kernel2.s.value
                    from scipy.special import gamma

                    fac2 = nquad(lambda rho, theta: (1-np.cos(np.pi*rho*np.cos(theta))) * rho**(-1-2*sBase2),
                                 [(0, horizon2), (0, 2*np.pi)])[0]
                    forcing_right = 2*scaling2 * fac2 * (-sin)

                    def flux_right_lam(x):
                        # x in IJ2
                        # y in IJ1 \cap B(x, horizon2)
                        I1, err1 = nquad(fun2, IJ1, (x[0], x[1], horizon2), opts={'epsabs': epsabs, 'epsrel': epsrel})
                        # y in OmegaJ2 \cap B(x, horizon2)
                        I2, err2 = nquad(fun2, OmegaJ2, (x[0], x[1], horizon2), opts={'epsabs': epsabs, 'epsrel': epsrel})
                        # y in IJ1 \cap B(x, horizon1)
                        I3, err3 = nquad(fun1, IJ1, (x[0], x[1], horizon1), opts={'epsabs': epsabs, 'epsrel': epsrel})
                        return I1+I2-I3

                flux_left = Lambda(flux_left_lam)
                flux_right = Lambda(flux_right_lam)
                self.mult = constant(1.)
                flux_jump = indicatorFunctor(flux_right, localSubdomainIndicator1) + indicatorFunctor(flux_left, localSubdomainIndicator2)

                self.nonlocal_L2ex_left = 0.5
                self.nonlocal_L2ex_right = 1.5+4/np.pi

            elif problem == 'exact-sin-variableSolJump-fluxJump':
                # the nonlocal problem has a know exact solution
                assert kType1 in (INDICATOR, FRACTIONAL)
                assert kType2 in (INDICATOR, FRACTIONAL)
                assert coeff11 == coeff12
                assert coeff21 == coeff22
                sin2d = functionFactory('Lambda', lambda x: np.sin(np.pi*x[0])*np.sin(2*np.pi*x[1]))
                sin = functionFactory('sin2d')
                one = functionFactory('constant', 1)
                one = functionFactory('constant', 1)
                sol_1 = 2*one+2*sin2d
                sol_2 = one-sin
                diri_left = sol_1
                diri_right = sol_2

                sol_jump = sol_2-sol_1

                # Get scaling values for interactions within subdomains
                kernel1(np.array([0.5*(ax+cx), 0.5]),
                        np.array([0.6*ax+0.4*cx, 0.5]))
                kernel2(np.array([0.5*(bx+cx), 0.5]),
                        np.array([0.6*bx+0.4*cx, 0.5]))
                scaling1 = kernel1.scalingValue
                scaling2 = kernel2.scalingValue

                def evalRHSFac(alpha, beta, horizon, s, N=100):
                    from scipy.special import gamma as Gamma, binom
                    I = 0.
                    for k in range(N):
                        for m in range(N):
                            if k+m > 0:
                                I += (-1)**(k+m+1) * alpha**(2*k)/Gamma(2*k+1) * beta**(2*m)/Gamma(2*m+1) * horizon**(2*k+2*m-2*s) / (2*k+2*m-2*s) * 2. / binom(k+m, k-0.5) / (m+0.5)
                    return I

                if kType1 == INDICATOR:
                    sBase1 = -1.
                elif kType1 == FRACTIONAL:
                    assert isinstance(kernel1.s, constFractionalOrder)
                    sBase1 = kernel1.s.value
                forcing_left = 2*(2*scaling1*evalRHSFac(np.pi, 2.*np.pi, horizon1, sBase1))*sin2d

                if kType2 == INDICATOR:
                    sBase2 = -1.
                elif kType2 == FRACTIONAL:
                    assert isinstance(kernel2.s, constFractionalOrder)
                    sBase2 = kernel2.s.value
                forcing_right = -(2*scaling2*evalRHSFac(np.pi, np.pi, horizon2, sBase2))*sin

                from scipy.integrate import quad, nquad
                from . strongForm import getStrongIntegrand

                fun1 = getStrongIntegrand(sol_1, kernel1, True)
                fun2 = getStrongIntegrand(sol_2, kernel2, True)

                epsabs = 1e-4
                epsrel = 1e-4

                def flux_left_lam(x):
                    # x in IJ1
                    # y in IJ2 \cap B(x, horizon1)
                    I1, err1 = nquad(fun1, IJ2, (x[0], x[1], horizon1), opts={'epsabs': epsabs, 'epsrel': epsrel})
                    # y in IJ2 \cap B(x, horizon2)
                    I2, err2 = nquad(fun2, IJ2, (x[0], x[1], horizon2), opts={'epsabs': epsabs, 'epsrel': epsrel})
                    return I1-I2

                def flux_right_lam(x):
                    # x in IJ2
                    # y in IJ1 \cap B(x, horizon2)
                    I1, err1 = nquad(fun2, IJ1, (x[0], x[1], horizon2), opts={'epsabs': epsabs, 'epsrel': epsrel})
                    # y in OmegaJ2 \cap B(x, horizon2)
                    I2, err2 = nquad(fun2, OmegaJ2, (x[0], x[1], horizon2), opts={'epsabs': epsabs, 'epsrel': epsrel})
                    # y in IJ1 \cap B(x, horizon1)
                    I3, err3 = nquad(fun1, IJ1, (x[0], x[1], horizon1), opts={'epsabs': epsabs, 'epsrel': epsrel})
                    return I1+I2-I3

                flux_left = coeff11*Lambda(flux_left_lam)
                flux_right = coeff22*Lambda(flux_right_lam)
                self.mult = constant(1.)
                flux_jump = indicatorFunctor(flux_right, localSubdomainIndicator1) + indicatorFunctor(flux_left, localSubdomainIndicator2)

                self.nonlocal_L2ex_left = 5.
                self.nonlocal_L2ex_right = 1.25+8/np.pi**2

            else:
                raise NotImplementedError(problem)

        else:
            raise NotImplementedError(domain)

        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.mesh = mesh

        self.subdomainIndicator1 = subdomainIndicator1
        self.subdomainIndicator2 = subdomainIndicator2
        self.localSubdomainIndicator1 = localSubdomainIndicator1
        self.localInterfaceIndicator = localInterfaceIndicator
        self.localSubdomainIndicator2 = localSubdomainIndicator2
        self.domainIndicator1 = domainIndicator1
        self.domainIndicator2 = domainIndicator2
        self.interfaceIndicator = interfaceIndicator
        self.dirichletIndicator1 = dirichletIndicator1
        self.dirichletIndicator2 = dirichletIndicator2

        self.sol_1 = sol_1
        self.sol_2 = sol_2
        self.diri_left = diri_left
        self.diri_right = diri_right
        self.forcing_left = forcing_left
        self.forcing_right = forcing_right
        self.sol_jump = sol_jump
        self.flux_jump = flux_jump

        self.local_L2ex_left = local_L2ex_left
        self.local_L2ex_right = local_L2ex_right
        self.local_H10ex_left = local_H10ex_left
        self.local_H10ex_right = local_H10ex_right

    def getIdentifier(self, params):
        keys = ['domain', 'problem', 'kernel1', 'kernel2']
        d = []
        for k in keys:
            try:
                d.append((k, str(getattr(self, k))))
            except KeyError:
                d.append((k, str(params[k])))
        return '/'.join(['nonlocalInterface'] + [key + '=' + v for key, v in d])


class brusselatorProblem(problem):
    """
    Fractional order Brusselator system:

              \\partial_t U = -(-\\Delta)^\\alpha U + (B-1)*U + Q^2 V + B/Q * U**2 + 2*Q*U*V + U**2 * V
    \\eta**2 * \\partial_t V = -(-\\Delta)^\\beta  U - B*U     - Q^2 V - B/Q * U**2 - 2*Q*U*V - U**2 * V

    with zero flux conditions on U and V.

    s    = \\beta/\\alpha
    \\eta = \\sqrt(D_X**s / D_Y)
    Q    = A \\eta

    """

    def setDriverArgs(self):
        super().setDriverArgs()
        self.setDriverFlag('domain', acceptedValues=['disc', 'rectangle', 'twinDisc'], help='computational domain')
        self.setDriverFlag('bc', acceptedValues=['Neumann', 'Dirichlet'], help='type of boundary condition')
        self.setDriverFlag('noRef', 3, help='number of uniform mesh refinements')
        self.setDriverFlag('problem', acceptedValues=['spots', 'stripes'], help='pre-defined problems')
        self.setDriverFlag('T', 200., help='final time')

    @generates(['dim',
                'alpha',
                'beta',
                'eta',
                'initial_U',
                'initial_V',
                'Bcr',
                'kcr',
                'B',
                'Q',
                'A',
                'Dx',
                'Dy',
                'kernelU',
                'kernelV',
                'nonlinearity',
                'boundaryCondition',
                'mesh',
                'zeroExterior'])
    def processProblem(self, domain, bc, noRef, problem, T):
        from PyNucleus.fem.femCy import brusselator

        if problem == 'spots':
            self.alpha = self.beta = 0.75
            x = 0.1
            # eps = 0.1
            self.eta = 0.2

            if domain == 'disc':
                z1, z2 = 0., 0.
                R = 10.
            elif domain == 'twinDisc':
                z1, z2 = 11., 0.
                R = 10.

            def initial_U(x):
                r2 = (x[0]-z1)**2 + (x[1]-z2)**2
                if r2 < R**2:
                    return (R**2-r2)**2/R**4 * self.eta
                else:
                    return 0.

            def initial_V(x):
                r2 = (x[0]-z1)**2 + (x[1]-z2)**2
                if r2 < R**2:
                    return (R**2-r2)**2/R**4 / self.eta
                else:
                    return 0.

        elif problem == 'stripes':
            self.alpha = self.beta = 0.75
            x = 1.5
            # eps = 1.0
            self.eta = 0.2

            if domain == 'twinDisc':
                def initial_U(x):
                    if x[0] > 0.:
                        return np.random.rand() * self.eta
                    else:
                        return 0.

                def initial_V(x):
                    if x[0] > 0.:
                        return np.random.rand() / self.eta
                    else:
                        return 0.
            else:
                def initial_U(x):
                    return np.random.rand() * self.eta

                def initial_V(x):
                    return np.random.rand() / self.eta


        self.initial_U = functionFactory('Lambda', initial_U)
        self.initial_V = functionFactory('Lambda', initial_V)

        s = self.alpha/self.beta
        self.Bcr = (1+x)**2/(1+(1-s)*x)
        self.kcr = x**(1/self.alpha)
        self.B = self.Bcr + 0.01
        self.Q = np.sqrt(s*x**(1+1/s)/(1+(1-s)*x))
        self.A = self.Q/self.eta
        self.Dx = 1.
        self.Dy = 1/self.eta**2

        self.dim = nonlocalMeshFactory.getDim(domain)
        self.kernelU = kernelFactory('fractional', s=self.alpha, dim=self.dim, horizon=np.inf)
        self.kernelV = kernelFactory('fractional', s=self.beta, dim=self.dim, horizon=np.inf)
        self.nonlinearity = brusselator(self.B, self.Q)

        if bc == 'Neumann':
            self.boundaryCondition = HOMOGENEOUS_NEUMANN
        elif bc == 'Dirichlet':
            self.boundaryCondition = HOMOGENEOUS_DIRICHLET

        if domain == 'disc':
            self.mesh, nI = nonlocalMeshFactory('disc',
                                                h=10.,
                                                radius=50.,
                                                kernel=self.kernelU,
                                                boundaryCondition=self.boundaryCondition)
        elif domain == 'square':
            self.mesh = nonlocalMeshFactory('rectangle',
                                            ax=-50., ay=-50.,
                                            bx=50., by=50.,
                                            N=5, M=5,
                                            kernel=self.kernelU,
                                            boundaryCondition=self.boundaryCondition)

        elif domain == 'twinDisc':
            self.mesh, nI = nonlocalMeshFactory('twinDisc',
                                                h=10.,
                                                radius=50.,
                                                sep=2.,
                                                kernel=self.kernelU,
                                                boundaryCondition=self.boundaryCondition)
        self.zeroExterior = nI['zeroExterior']

    def getIdentifier(self, params):
        keys = ['domain', 'problem', 'alpha', 'beta', 'noRef', 'bc']
        d = []
        for k in keys:
            try:
                d.append((k, str(getattr(self, k))))
            except KeyError:
                d.append((k, str(params[k])))
        return '-'.join(['brusselator'] + [key + '=' + v for key, v in d])


