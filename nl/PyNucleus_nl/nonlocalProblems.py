###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from PyNucleus_base import REAL
from PyNucleus_base.factory import factory
from PyNucleus_base.utilsFem import problem
from PyNucleus_fem import (simpleInterval, intervalWithInteraction,
                           uniformSquare, squareWithInteractions,
                           discWithInteraction,
                           double_graded_interval,
                           double_graded_interval_with_interaction,
                           discWithIslands,
                           meshFactoryClass, PHYSICAL, NO_BOUNDARY,
                           Lambda, constant,
                           indicatorFunctor, squareIndicator, radialIndicator,
                           P1_DoFMap,
                           str2DoFMapOrder,
                           solFractional1D, rhsFractional1D,
                           solFractional, rhsFractional2D,
                           rhsFractional2D_nonPeriodic,
                           functionFactory,
                           
                           meshFactory, functionFactory,
                           DIRICHLET, HOMOGENEOUS_DIRICHLET,
                           NEUMANN, HOMOGENEOUS_NEUMANN,
                           NORM)
from PyNucleus_fem.mesh import twinDisc
from scipy.special import gamma as Gamma, binom
from . twoPointFunctions import (constantTwoPoint,
                                 temperedTwoPoint,
                                 leftRightTwoPoint,
                                 smoothedLeftRightTwoPoint,)
from . interactionDomains import (ball1,
                                  ball2,
                                  ballInf,
                                  ellipse)
from . fractionalOrders import (constFractionalOrder,
                                variableConstFractionalOrder,
                                leftRightFractionalOrder,
                                smoothedLeftRightFractionalOrder,
                                innerOuterFractionalOrder,
                                islandsFractionalOrder,
                                layersFractionalOrder,
                                variableFractionalLaplacianScaling)
from . kernelsCy import (getKernelEnum,
                         FRACTIONAL, INDICATOR, PERIDYNAMIC)
from . kernels import (getFractionalKernel,
                       getIntegrableKernel,
                       getKernel)


fractionalOrderFactory = factory()
fractionalOrderFactory.register('constant', constFractionalOrder, aliases=['const'])
fractionalOrderFactory.register('varConst', variableConstFractionalOrder, aliases=['constVar'])
fractionalOrderFactory.register('leftRight', leftRightFractionalOrder, aliases=['twoDomain'])
fractionalOrderFactory.register('smoothedLeftRight', smoothedLeftRightFractionalOrder, params={'r': 0.1, 'slope': 200.}, aliases=['twoDomainNonSym'])
fractionalOrderFactory.register('innerOuter', innerOuterFractionalOrder)
fractionalOrderFactory.register('islands', islandsFractionalOrder, params={'r': 0.1, 'r2': 0.6})
fractionalOrderFactory.register('layers', layersFractionalOrder)

interactionFactory = factory()
interactionFactory.register('ball2', ball2, aliases=['2', 2])
interactionFactory.register('ball1', ball1, aliases=['1', 1])
interactionFactory.register('ballInf', ballInf, aliases=['inf', np.inf])

kernelFactory = factory()
kernelFactory.register('fractional', getFractionalKernel)
kernelFactory.register('indicator', getIntegrableKernel, params={'kernel': INDICATOR}, aliases=['constant'])
kernelFactory.register('peridynamic', getIntegrableKernel, params={'kernel': PERIDYNAMIC})


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
        domainIndicator, boundaryIndicator, interactionIndicator = super(nonlocalMeshFactoryClass, self).build(name, **kwargs)

        if boundaryCondition == HOMOGENEOUS_DIRICHLET:
            if kernel.horizon.value == np.inf:
                if kernel.s.max < 0.5:
                    tag = NO_BOUNDARY
                else:
                    tag = PHYSICAL
                zeroExterior = True
            else:
                tag = domainIndicator
                zeroExterior = False
            hasInteractionDomain = kernel.horizon.value < np.inf
        elif boundaryCondition == HOMOGENEOUS_NEUMANN:
            tag = NO_BOUNDARY
            zeroExterior = False
            hasInteractionDomain = False
        elif boundaryCondition == DIRICHLET:
            if kernel.horizon.value == np.inf:
                if kernel.s.max < 0.5:
                    tag = NO_BOUNDARY
                else:
                    tag = PHYSICAL
                raise NotImplementedError()
            else:
                tag = NO_BOUNDARY
            zeroExterior = False
            hasInteractionDomain = True
        elif boundaryCondition == NEUMANN:
            if kernel.horizon.value == np.inf:
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

        if hasInteractionDomain:
            assert 0 < kernel.horizon.value < np.inf
            kwargs['horizon'] = kernel.horizon.value
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
        return mesh, nonlocalInfo

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


nonlocalMeshFactory = nonlocalMeshFactoryClass()
nonlocalMeshFactory.register('interval', simpleInterval, intervalWithInteraction, 1, intervalIndicators, {'a': -1, 'b': 1}, {'a': -1, 'b': 1})
nonlocalMeshFactory.register('gradedInterval', double_graded_interval, double_graded_interval_with_interaction, 1, intervalIndicators, {'a': -1, 'b': 1, 'mu_ll': 2., 'mu_rr': 2.}, {'a': -1, 'b': 1, 'mu_ll': 2., 'mu_rr': 2.})
nonlocalMeshFactory.register('square', uniformSquare, squareWithInteractions, 2, squareIndicators, {'N': 2, 'M': 2, 'ax': -1, 'ay': -1, 'bx': 1, 'by': 1}, {'ax': -1, 'ay': -1, 'bx': 1, 'by': 1}, aliases=['rectangle'])
nonlocalMeshFactory.register('disc', discWithInteraction, discWithInteraction, 2, radialIndicators, {'horizon': 0., 'radius': 1.}, {'radius': 1.})
nonlocalMeshFactory.register('discWithIslands', discWithIslands, discWithIslands, 2, radialIndicators, {'horizon': 0., 'radius': 1., 'islandOffCenter': 0.35, 'islandDiam': 0.5}, {'radius': 1., 'islandOffCenter': 0.35, 'islandDiam': 0.5})
nonlocalMeshFactory.register('twinDisc', twinDisc, twinDisc, 2, radialIndicators, {'radius': 1., 'sep': 0.1}, {'radius': 1., 'sep': 0.1})


class fractionalLaplacianProblem(problem):
    def setDriverArgs(self, driver):
        p = driver.addGroup('problem')
        p.add('domain', acceptedValues=['interval', 'disc', 'Lshape', 'square', 'cutoutCircle', 'disconnectedInterval', 'disconnectedDomain'])
        p.add('problem', acceptedValues=['constant', 'notPeriodic', 'plateau', 'sin', 'cos', 3, 'source'])
        self.addParametrizedArg('const', [float])
        self.addParametrizedArg('leftRight', [float, float])
        self.addParametrizedArg('genLeftRight', [float, float, float, float])
        self.addParametrizedArg('islands', [float, float, float])
        self.addParametrizedArg('layers', [float, float, int])
        p.add('s', 'const(0.75)', argInterpreter=self.argInterpreter(['const', 'leftRight', 'genLeftRight', 'islands', 'layers']))
        p.add('element', acceptedValues=[1, 2])
        p.add('adaptive', acceptedValues=['residualMelenk', 'residualNochetto', 'residual', 'hierarchical', 'knownSolution', None], argInterpreter=lambda v: None if v == 'None' else v)
        p.add('noRef', -1)

    def processImpl(self, params):
        element = params['element']
        self.dim = nonlocalMeshFactory.getDim(params['domain'])
        for sName in ['const', 'leftRight', 'genLeftRight', 'islands']:
            if self.parametrizedArg(sName).match(params['s']):
                s = fractionalOrderFactory.build(sName,
                                                 *self.parametrizedArg(sName).interpret(params['s']))
                break
        else:
            if self.parametrizedArg('layers').match(params['s']):
                t = np.linspace(*self.parametrizedArg('layers').interpret(params['s']), dtype=REAL)
                s = np.empty((t.shape[0], t.shape[0]), dtype=REAL)
                for i in range(t.shape[0]):
                    for j in range(t.shape[0]):
                        s[i, j] = 0.5*(t[i]+t[j])
                s = layersFractionalOrder(self.dim, np.linspace(-1., 1., s.shape[0]+1, dtype=REAL), s)
            else:
                raise NotImplementedError(params['s'])
        horizon = constant(np.inf)
        normalized = params.get('normalized', True)
        self.kernel = getFractionalKernel(self.dim, s, horizon=horizon, normalized=normalized)
        adaptive = params['adaptive']
        problem = params['problem']
        self.sol_ex = None
        self.Hs_ex = None
        self.L2_ex = None
        # Picking bigger, say eta = 7, potentially speeds up assembly.
        # Not clear about impact on error.
        self.eta = 3.

        if self.dim == 1:
            self.target_order = (1+element-s.min)/self.dim
        else:
            self.target_order = 1/self.dim
            if element == 2:
                raise NotImplementedError()

        if params['domain'] == 'interval':
            self.meshParams = {'a': -1., 'b': 1.}
            radius = 1.
            if self.noRef <= 0:
                if adaptive is None:
                    if element == 1:
                        self.noRef = 6
                    elif element == 2:
                        self.noRef = 5
                    else:
                        raise NotImplementedError(element)
                else:
                    if element == 1:
                        self.noRef = 22
                    elif element == 2:
                        self.noRef = 21
                    else:
                        raise NotImplementedError(element)
            self.eta = 1

            if problem == 'constant':
                self.rhs = constant(1.)
                if isinstance(s, constFractionalOrder):
                    C = 2.**(-2.*s.value)*Gamma(self.dim/2.)/Gamma((self.dim+2.*s.value)/2.)/Gamma(1.+s.value)
                    self.Hs_ex = C * np.sqrt(np.pi)*Gamma(s.value+1)/Gamma(s.value+3/2)
                    self.L2_ex = np.sqrt(C**2 * np.sqrt(np.pi) * Gamma(1+2*s.value)/Gamma(3/2+2*s.value) * radius**2)
                    self.sol_ex = solFractional(s.value, self.dim, radius)
            elif problem == 'sin':
                self.rhs = Lambda(lambda x: np.sin(np.pi*x[0]))
            elif problem == 'cos':
                self.rhs = Lambda(lambda x: np.cos(np.pi*x[0]/2.))
            elif problem == 'plateau':
                self.rhs = Lambda(np.sign)

                # def e(n):
                #     return (2*n+s+3/2)/2**(2*s)/np.pi / binom(n+s+1, n-1/2)**2/Gamma(s+5/2)**2

                # k = 10
                # Hs_ex = sum([e(n) for n in range(1000000)])
                self.Hs_ex = 2**(1-2*s) / (2*s+1) / Gamma(s+1)**2
            elif isinstance(problem, int):
                self.rhs = rhsFractional1D(s, problem)
                self.Hs_ex = 2**(2*s)/(2*problem+s+0.5) * Gamma(1+s)**2 * binom(s+problem, problem)**2
                self.sol_ex = solFractional1D(s, problem)
            else:
                raise NotImplementedError(params['problem'])
        elif params['domain'] == 'disconnectedInterval':
            if self.noRef <= 0:
                self.noRef = 40
            self.meshParams = {'sep': 0.1}

            if problem == 'constant':
                self.rhs = Lambda(lambda x: 1. if x[0] > 0.5 else 0.)
            else:
                raise NotImplementedError()
        elif params['domain'] == 'disc':
            if self.noRef <= 0:
                self.noRef = 15
                if adaptive is None:
                    self.noRef = 5
                else:
                    self.noRef = 7
            radius = 1.
            self.meshParams = {'h': 0.78, 'radius': radius}

            if problem == 'constant':
                self.rhs = constant(1.)
                if isinstance(s, constFractionalOrder):
                    C = 2.**(-2.*s.value)*Gamma(self.dim/2.)/Gamma((self.dim+2.*s.value)/2.)/Gamma(1.+s.value)
                    self.Hs_ex = C * np.pi*radius**(2-2*s.value)/(s.value+1)
                    self.L2_ex = np.sqrt(C**2 * np.pi/(1+2*s.value)*radius**2)
                    self.sol_ex = solFractional(s.value, self.dim, radius)
            elif problem == 'notPeriodic':
                n = 2
                l = 2
                self.Hs_ex = 2**(2*s-1)/(2*n+s+l+1) * Gamma(1+s+n)**2/Gamma(1+n)**2 * (np.pi+np.sin(4*np.pi*l)/(4*l))

                n = 1
                l = 5
                self.Hs_ex += 2**(2*s-1)/(2*n+s+l+1) * Gamma(1+s+n)**2/Gamma(1+n)**2 * (np.pi+np.sin(4*np.pi*l)/(4*l))
                self.rhs = rhsFractional2D_nonPeriodic(s)
            elif problem == 'plateau':
                self.rhs = Lambda(lambda x: x[0] > 0)
                try:
                    from mpmath import meijerg
                    self.Hs_ex = np.pi/4*2**(-2*s) / (s+1) / Gamma(1+s)**2
                    self.Hs_ex -= 2**(-2*s)/np.pi * meijerg([[1., 1.+s/2], [5/2+s, 5/2+s]],
                                                       [[2., 1/2, 1/2], [2.+s/2]],
                                                       -1., series=2)
                    self.Hs_ex = float(self.Hs_ex)
                except ImportError:
                    self.Hs_ex = np.pi/4*2**(-2*s) / (s+1) / Gamma(1+s)**2
                    for k in range(100000):
                        self.Hs_ex += 2**(-2*s) / Gamma(s+3)**2 / (2*np.pi) * (2*k+s+2) * (k+1) / binom(k+s+1.5, s+2)**2
            elif isinstance(problem, tuple):
                n, l = problem
                self.Hs_ex = 2**(2*s-1)/(2*n+s+l+1) * Gamma(1+s+n)**2/Gamma(1+n)**2 * (np.pi+np.sin(4*np.pi*l)/(4*l))

                self.rhs = rhsFractional2D(s, n=n, l=l)
            elif problem == 'sin':
                self.rhs = Lambda(lambda x: np.sin(np.pi*(x[0]**2+x[1]**2)))
            else:
                raise NotImplementedError()
        elif params['domain'] == 'square':
            if self.noRef <= 0:
                self.noRef = 20
            self.meshParams = {'N': 3, 'ax': -1, 'ay': -1, 'bx': 1, 'by': 1}

            if problem == 'constant':
                self.rhs = constant(1.)
            elif problem == 'sin':
                self.rhs = Lambda(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
            elif problem == 'source':
                self.rhs = (functionFactory.build('radialIndicator', radius=0.3, center=np.array([0.2, 0.6], dtype=REAL)) -
                            functionFactory.build('radialIndicator', radius=0.3, center=np.array([-0.2, -0.6], dtype=REAL)))
            else:
                raise NotImplementedError()
        elif params['domain'] == 'Lshape':
            if self.noRef <= 0:
                self.noRef = 20
            self.meshParams = {}

            if problem == 'constant':
                self.rhs = constant(1.)
            elif problem == 'sin':
                self.rhs = Lambda(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
            else:
                raise NotImplementedError()
        elif params['domain'] == 'cutoutCircle':
            if self.noRef <= 0:
                self.noRef = 30
            self.meshParams = {'radius': 1., 'cutoutAngle': np.pi/2.}

            if problem == 'constant':
                self.rhs = constant(1.)
            elif problem == 'sin':
                self.rhs = Lambda(lambda x: np.sin(np.pi*(x[0]**2+x[1]**2)))
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(params['domain'])

        self.mesh, nI = nonlocalMeshFactory.build(params['domain'], self.kernel, HOMOGENEOUS_DIRICHLET, useMulti=True, **self.meshParams)
        self.tag = nI['tag']

    def getIdentifier(self, params):
        keys = ['domain', 'problem', 's', 'noRef', 'element', 'adaptive']
        d = []
        for k in keys:
            try:
                d.append((k, str(self.__getattr__(k))))
            except KeyError:
                d.append((k, str(params[k])))
        return '-'.join(['fracLaplAdaptive'] + [key + '=' + v for key, v in d])


class nonlocalProblem(problem):
    def setDriverArgs(self, driver):
        driver.add('kernel', acceptedValues=['fractional', 'indicator', 'peridynamic'])
        driver.add('domain', 'interval', acceptedValues=['gradedInterval', 'square', 'disc', 'discWithIslands'])
        self.addParametrizedArg('indicator', [float, float])
        driver.add('problem', 'poly-Dirichlet',
                   argInterpreter=self.argInterpreter(['indicator'], acceptedValues=['poly-Dirichlet', 'poly-Dirichlet2', 'poly-Dirichlet3', 'poly-Neumann', 'zeroFlux', 'source', 'constant', 'exact-sin-Dirichlet', 'exact-sin-Neumann']))
        driver.add('noRef', argInterpreter=int)
        self.addParametrizedArg('const', [float])
        self.addParametrizedArg('varconst', [float])
        self.addParametrizedArg('leftRight', [float, float, float, float])
        self.addParametrizedArg('twoDomain', [float, float, float, float])
        self.addParametrizedArg('twoDomainNonSym', [float, float])
        self.addParametrizedArg('layers', [float, float, int])
        self.addParametrizedArg('islands', [float, float])
        self.addParametrizedArg('islands4', [float, float, float, float])
        self.addParametrizedArg('tempered', [float])
        driver.add('s', 'const(0.4)', argInterpreter=self.argInterpreter(['const', 'varconst', 'twoDomain', 'twoDomainNonSym', 'layers', 'islands', 'islands4']))
        driver.add('horizon', 0.2)
        self.addParametrizedArg('ellipse', [float, float])
        driver.add('interaction', 'ball2', argInterpreter=self.argInterpreter(['ellipse'], acceptedValues=['ball2', 'ellipse']))
        driver.add('phi', 'const(1.)', argInterpreter=self.argInterpreter(['const', 'twoDomain', 'twoDomainNonSym', 'tempered']))
        driver.add('normalized', True)
        driver.add('element', acceptedValues=['P1', 'P0'])
        driver.add('target_order', -1.)

    def processImpl(self, params):

        self.dim = nonlocalMeshFactory.getDim(params['domain'])

        self.kType = getKernelEnum(params['kernel'])
        if self.kType == FRACTIONAL:
            for sName in ['const', 'varconst', 'leftRight', 'twoDomain', 'twoDomainNonSym', 'islands']:
                if self.parametrizedArg(sName).match(params['s']):
                    s = fractionalOrderFactory.build(sName,
                                                     *self.parametrizedArg(sName).interpret(params['s']))
                    break
            else:
                if self.parametrizedArg('layers').match(params['s']):
                    t = np.linspace(*self.parametrizedArg('layers').interpret(params['s']), dtype=REAL)
                    s = np.empty((t.shape[0], t.shape[0]), dtype=REAL)
                    for i in range(t.shape[0]):
                        for j in range(t.shape[0]):
                            s[i, j] = 0.5*(t[i]+t[j])
                    s = layersFractionalOrder(self.dim, np.linspace(-1., 1., s.shape[0]+1, dtype=REAL), s)
                elif self.parametrizedArg('islands4').match(params['s']):
                    sii, soo, sio, soi = self.parametrizedArg('islands4').interpret(params['s'])
                    s = fractionalOrderFactory.build('islands', sii=sii, soo=soo, sio=sio, soi=soi)
                else:
                    raise NotImplementedError(params['s'])
            self.s = s
        else:
            self.s = None

        self.horizon = constant(params['horizon'])

        element = str2DoFMapOrder(params['element'])
        if self.dim == 1:
            self.eta = 1.
            if self.target_order < 0.:
                if self.s is not None:
                    self.target_order = (1+element-self.s.min)/self.dim
                else:
                    self.target_order = 2.
        else:
            self.eta = 3.
            if self.target_order < 0.:
                self.target_order = 1/self.dim
            if element == 2:
                raise NotImplementedError()

        if self.parametrizedArg('const').match(params['phi']):
            c, = self.parametrizedArg('const').interpret(params['phi'])
            if c == 1.:
                self.phi = None
            else:
                self.phi = constantTwoPoint(c)
        elif self.parametrizedArg('twoDomain').match(params['phi']):
            phill, phirr, philr, phirl = self.parametrizedArg('twoDomain').interpret(params['phi'])
            self.phi = leftRightTwoPoint(phill, phirr, philr, phirl)
        elif self.parametrizedArg('twoDomainNonSym').match(params['phi']):
            phil, phir = self.parametrizedArg('twoDomainNonSym').interpret(params['phi'])
            self.phi = smoothedLeftRightTwoPoint(phil, phir, r=0.1, slope=200.)
        elif self.parametrizedArg('tempered').match(params['phi']):
            lambdaCoeff, = self.parametrizedArg('tempered').interpret(params['phi'])
            self.phi = temperedTwoPoint(lambdaCoeff, self.dim)
        else:
            raise NotImplementedError(params['phi'])

        if params['interaction'] == 'ball2':
            interaction = ball2()
        elif self.parametrizedArg('ellipse').match(params['interaction']):
            aFac, bFac = self.parametrizedArg('ellipse').interpret(params['interaction'])
            interaction = ellipse(aFac, bFac)
        else:
            raise NotImplementedError(params['interaction'])

        normalized = params['normalized']
        self.kernel = getKernel(dim=self.dim, kernel=self.kType, s=self.s, horizon=self.horizon, normalized=normalized, phi=self.phi, interaction=interaction)
        self.scaling = self.kernel.scaling

        self.analyticSolution = None

        if params['problem'] in ('poly-Neumann', 'exact-sin-Neumann', 'zeroFlux'):
            self.boundaryCondition = NEUMANN
        elif self.parametrizedArg('indicator').match(params['problem']):
            self.boundaryCondition = HOMOGENEOUS_DIRICHLET
        elif params['problem'] in ('source', 'constant'):
            self.boundaryCondition = HOMOGENEOUS_DIRICHLET
        else:
            self.boundaryCondition = DIRICHLET

        if params['domain'] in ('interval', 'gradedInterval'):
            if params['noRef'] is None:
                self.noRef = 8
            self.mesh, nI = nonlocalMeshFactory.build(params['domain'], self.kernel, self.boundaryCondition)
            self.tag = nI['tag']
            self.zeroExterior = nI['zeroExterior']
            self.domainInteriorIndicator = domainIndicator = nI['domain']
            self.boundaryIndicator = boundaryIndicator = nI['boundary']
            self.interactionInteriorIndicator = interactionIndicator = nI['interaction']
            if params['problem'] == 'poly-Dirichlet':
                self.domainIndicator = domainIndicator
                self.fluxIndicator = constant(0)
                self.interactionIndicator = interactionIndicator+boundaryIndicator
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: 1-x[0]**2)
                if ((self.kType == FRACTIONAL and isinstance(self.s, constFractionalOrder)) or self.kType in (INDICATOR, PERIDYNAMIC)) and self.phi is None and normalized:
                    self.analyticSolution = Lambda(lambda x: 1-x[0]**2)
            elif params['problem'] == 'exact-sin-Dirichlet':
                assert ((self.kType == INDICATOR) or (self.kType == FRACTIONAL)) and self.phi is None and normalized

                self.domainIndicator = domainIndicator
                self.fluxIndicator = constant(0)
                self.interactionIndicator = interactionIndicator+boundaryIndicator
                horizonValue = self.kernel.horizonValue
                scalingValue = self.kernel.scalingValue

                sin = functionFactory('sin1d')
                if self.kType == INDICATOR:
                    self.rhsData = -2.*scalingValue * 2*(np.sin(np.pi*horizonValue)/np.pi-horizonValue) * sin
                elif self.kType == FRACTIONAL:
                    from scipy.integrate import quad
                    assert isinstance(self.s, constFractionalOrder)
                    sBase = self.s.value
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
            elif params['problem'] == 'exact-sin-Neumann':
                assert (self.kType == FRACTIONAL) and self.phi is None and normalized

                self.domainIndicator = domainIndicator
                self.fluxIndicator = boundaryIndicator+interactionIndicator
                self.interactionIndicator = constant(0.)
                horizonValue = self.kernel.horizonValue
                scalingValue = self.kernel.scalingValue

                sin = functionFactory('sin1d')
                cos = functionFactory('cos1d')
                if self.kType == FRACTIONAL:
                    from scipy.integrate import quad
                    assert isinstance(self.s, constFractionalOrder)
                    sBase = self.s.value
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
            elif params['problem'] == 'poly-Neumann':
                if self.kType == FRACTIONAL:
                    assert self.s.max <= 0.5, "RHS is singular, need a special quadrature rule"
                self.domainIndicator = domainIndicator
                self.fluxIndicator = boundaryIndicator+interactionIndicator
                self.interactionIndicator = constant(0.)
                horizonBase = self.horizon.value

                if self.kType == FRACTIONAL:
                    sBase = self.s.value

                    def fluxFun(x):
                        # dist = 1+horizonBase-abs(x[0])
                        # assert dist >= 0
                        # return (1+(dist/horizonBase)**(2-2*sBase) - 2*abs(x[0]) * (2-2*sBase)/(1-2*sBase)/horizonBase * (1-(dist/horizonBase)**(1-2*sBase)))
                        dist = 1+horizonBase-abs(x[0])
                        assert dist >= 0
                        return 2*self.kernel.scalingValue * ((2*abs(x[0])/(1-2*sBase)) * (dist**(1-2*sBase)-horizonBase**(1-2*sBase)) + 1/(2-2*sBase) * (dist**(2-2*sBase)+horizonBase**(2-2*sBase)))
                elif self.kType == PERIDYNAMIC:
                    def fluxFun(x):
                        dist = 1+horizonBase-abs(x[0])
                        assert dist >= 0
                        return 2*self.kernel.scalingValue * (2*abs(x[0]) * (1-abs(x[0])) + 0.5 * (dist**2+horizonBase**2))
                elif self.kType == INDICATOR:
                    def fluxFun(x):
                        dist = 1+horizonBase-abs(x[0])
                        assert dist >= 0
                        return 2*self.kernel.scalingValue * (abs(x[0]) * (dist**2-horizonBase**2) + 1./3. * (dist**3+horizonBase**3))

                self.rhsData = constant(2)
                self.fluxData = Lambda(fluxFun)
                self.dirichletData = Lambda(lambda x: 1-x[0]**2)
                if ((self.kType == FRACTIONAL and isinstance(self.s, constFractionalOrder)) or self.kType != FRACTIONAL) and normalized:
                    self.analyticSolution = Lambda(lambda x: 1-x[0]**2)
            elif params['problem'] == 'zeroFlux':
                self.domainIndicator = domainIndicator
                self.fluxIndicator = Lambda(lambda x: 1. if (x[0] > 1) else 0.)
                self.interactionIndicator = interactionIndicator+boundaryIndicator
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: 1-x[0]**2)
            elif self.parametrizedArg('indicator').match(params['problem']):
                self.domainIndicator = domainIndicator
                self.fluxIndicator = constant(0)
                self.interactionIndicator = interactionIndicator+boundaryIndicator
                # self.fluxIndicator = squareIndicator(np.array([1.], dtype=REAL),
                #                                      np.array([1.+params['horizon']-1e-9], dtype=REAL))
                center, width = self.parametrizedArg('indicator').interpret(params['problem'])
                self.rhsData = squareIndicator(np.array([center-width/2], dtype=REAL),
                                               np.array([center+width/2], dtype=REAL))
                self.fluxData = constant(0)
                self.dirichletData = constant(0.)
            elif params['problem'] == 'constant':
                self.domainIndicator = domainIndicator
                self.fluxIndicator = constant(0)
                self.interactionIndicator = interactionIndicator+boundaryIndicator
                self.rhsData = constant(1.)
                self.fluxData = constant(0)
                self.dirichletData = constant(0.)
                if (self.kType == FRACTIONAL) and isinstance(self.kernel.s, constFractionalOrder):
                    self.analyticSolution = functionFactory('solFractional', dim=1, s=self.kernel.s.value)
            else:
                raise NotImplementedError(params['problem'])
        elif params['domain'] == 'square':
            if params['noRef'] is None:
                self.noRef = 2
            meshParams = {}
            # meshParams['uniform'] = True
            if isinstance(self.s, layersFractionalOrder):
                t = np.array(self.s.layerBoundaries)[1:-1]
                meshParams['preserveLinesHorizontal'] = t.tolist()
            elif isinstance(self.s, leftRightFractionalOrder) or isinstance(self.phi, leftRightTwoPoint):
                meshParams['preserveLinesVertical'] = [0.]
            self.mesh, nI = nonlocalMeshFactory.build('square', self.kernel, self.boundaryCondition, **meshParams)
            self.tag = nI['tag']
            self.zeroExterior = nI['zeroExterior']
            self.domainInteriorIndicator = domainIndicator = nI['domain']
            self.boundaryIndicator = boundaryIndicator = nI['boundary']
            self.interactionInteriorIndicator = interactionIndicator = nI['interaction']
            self.domainIndicator = domainIndicator
            self.interactionIndicator = interactionIndicator+boundaryIndicator
            if params['problem'] == 'poly-Dirichlet' and isinstance(interaction, ball2):
                self.fluxIndicator = constant(0)
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: 1-x[0]**2)
                if (((self.kType == FRACTIONAL and isinstance(self.s, constFractionalOrder)) or
                     self.kType in (INDICATOR, PERIDYNAMIC)) and
                    self.phi is None and
                    normalized):
                    self.analyticSolution = Lambda(lambda x: 1-x[0]**2)
            elif params['problem'] == 'poly-Dirichlet' and isinstance(interaction, ellipse):
                aFac = np.sqrt(self.kernel.interaction.aFac2)
                bFac = np.sqrt(self.kernel.interaction.bFac2)
                self.fluxIndicator = constant(0)
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: (1-x[0]**2) * 2/(np.pi*self.kernel.horizon.value**4/4 * aFac**3 * bFac))
                if (self.kType == INDICATOR and
                    self.phi is None and
                    not normalized):
                    self.analyticSolution = self.dirichletData
            elif params['problem'] == 'poly-Dirichlet2' and isinstance(interaction, ellipse):
                aFac = np.sqrt(self.kernel.interaction.aFac2)
                bFac = np.sqrt(self.kernel.interaction.bFac2)
                self.fluxIndicator = constant(0)
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: (1-x[1]**2) * 2/(np.pi*self.kernel.horizon.value**4/4 * aFac * bFac**3))
                if (self.kType == INDICATOR and
                    self.phi is None and
                    not normalized):
                    self.analyticSolution = self.dirichletData
            elif params['problem'] == 'poly-Dirichlet3' and isinstance(interaction, ellipse):
                aFac = np.sqrt(self.kernel.interaction.aFac2)
                bFac = np.sqrt(self.kernel.interaction.bFac2)
                self.fluxIndicator = constant(0)
                self.rhsData = constant(4)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: (1-x[0]**2) * 2/(np.pi*self.kernel.horizon.value**4/4 * aFac**3 * bFac) + (1-x[1]**2) * 2/(np.pi*self.kernel.horizon.value**4/4 * aFac * bFac**3))
                if (self.kType == INDICATOR and
                    self.phi is None and
                    not normalized):
                    self.analyticSolution = self.dirichletData
            elif params['problem'] == 'poly-Neumann':
                self.fluxIndicator = Lambda(lambda x: 1. if (x[0] > 1) else 0.)
                raise NotImplementedError(params['problem'])
            elif params['problem'] == 'source':
                self.fluxIndicator = constant(0)
                self.rhsData = (functionFactory.build('radialIndicator', radius=0.3, center=np.array([0.2, 0.6], dtype=REAL)) -
                                functionFactory.build('radialIndicator', radius=0.3, center=np.array([-0.2, -0.6], dtype=REAL)))
                self.fluxData = constant(0)
                self.dirichletData = constant(0)
            elif params['problem'] == 'constant':
                self.fluxIndicator = constant(0)
                self.rhsData = constant(1.)
                self.fluxData = constant(0)
                self.dirichletData = constant(0)
            else:
                raise NotImplementedError(params['problem'])
        elif params['domain'] == 'disc':
            if params['noRef'] is None:
                self.noRef = 4
            meshParams = {}
            self.mesh, nI = nonlocalMeshFactory.build('disc', self.kernel, self.boundaryCondition, **meshParams)
            self.tag = nI['tag']
            self.zeroExterior = nI['zeroExterior']
            self.domainInteriorIndicator = domainIndicator = nI['domain']
            self.boundaryIndicator = boundaryIndicator = nI['boundary']
            self.interactionInteriorIndicator = interactionIndicator = nI['interaction']
            self.domainIndicator = domainIndicator+boundaryIndicator
            self.interactionIndicator = interactionIndicator
            if params['problem'] == 'poly-Dirichlet':
                self.fluxIndicator = constant(0)
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: 1-x[0]**2)
                if isinstance(self.s, constFractionalOrder) and isinstance(self.phi, constantTwoPoint) and normalized:
                    self.analyticSolution = Lambda(lambda x: 1-x[0]**2)
            elif params['problem'] == 'poly-Neumann':
                self.fluxIndicator = Lambda(lambda x: 1. if (x[0] > 1) else 0.)
                raise NotImplementedError(params['problem'])
            elif params['problem'] == 'source':
                self.fluxIndicator = constant(0)
                self.rhsData = (functionFactory.build('radialIndicator', radius=0.3, center=np.array([0.2, 0.6], dtype=REAL)) -
                                functionFactory.build('radialIndicator', radius=0.3, center=np.array([-0.2, -0.6], dtype=REAL)))
                self.fluxData = constant(0)
                self.dirichletData = constant(0)
            elif params['problem'] == 'constant':
                self.fluxIndicator = constant(0)
                self.rhsData = constant(1.)
                self.fluxData = constant(0)
                self.dirichletData = constant(0)
                if (self.kType == FRACTIONAL) and isinstance(self.kernel.s, constFractionalOrder):
                    self.analyticSolution = functionFactory('solFractional', dim=2, s=self.kernel.s.value)
            else:
                raise NotImplementedError(params['problem'])
        elif params['domain'] == 'discWithIslands':
            if params['noRef'] is None:
                self.noRef = 4
            meshParams = {}
            self.mesh, nI = nonlocalMeshFactory.build('discWithIslands', self.kernel, self.boundaryCondition, **meshParams)
            self.tag = nI['tag']
            self.zeroExterior = nI['zeroExterior']
            self.domainInteriorIndicator = domainIndicator = nI['domain']
            self.boundaryIndicator = boundaryIndicator = nI['boundary']
            self.interactionInteriorIndicator = interactionIndicator = nI['interaction']
            self.domainIndicator = domainIndicator+boundaryIndicator
            self.interactionIndicator = interactionIndicator
            if params['problem'] == 'poly-Dirichlet':
                self.fluxIndicator = constant(0)
                self.rhsData = constant(2)
                self.fluxData = constant(0)
                self.dirichletData = Lambda(lambda x: 1-x[0]**2)
                if isinstance(self.s, constFractionalOrder) and isinstance(self.phi, constantTwoPoint) and normalized:
                    self.analyticSolution = Lambda(lambda x: 1-x[0]**2)
            elif params['problem'] == 'poly-Neumann':
                self.fluxIndicator = Lambda(lambda x: 1. if (x[0] > 1) else 0.)
                raise NotImplementedError(params['problem'])
            elif params['problem'] == 'source':
                self.fluxIndicator = constant(0)
                self.rhsData = (functionFactory.build('radialIndicator', radius=0.3, center=np.array([0.2, 0.6], dtype=REAL)) -
                                functionFactory.build('radialIndicator', radius=0.3, center=np.array([-0.2, -0.6], dtype=REAL)))
                self.fluxData = constant(0)
                self.dirichletData = constant(0)
            elif params['problem'] == 'constant':
                self.fluxIndicator = constant(0)
                self.rhsData = constant(1.)
                self.fluxData = constant(0)
                self.dirichletData = constant(0)
            else:
                raise NotImplementedError(params['problem'])
        else:
            raise NotImplementedError(params['domain'])

        # should be equal to the forcing term within the domain and equal to
        # the flux term in the interaction region
        self.rhs = (indicatorFunctor(self.rhsData, self.domainIndicator) +
                    indicatorFunctor(self.fluxData, self.fluxIndicator))

    def getIdentifier(self, params):
        keys = ['domain', 'problem', 's', 'horizon', 'phi', 'noRef']
        d = []
        for k in keys:
            try:
                d.append((k, str(self.__getattr__(k))))
            except KeyError:
                d.append((k, str(params[k])))
        return '-'.join(['nonlocal'] + [key + '=' + v for key, v in d])



class brusselatorProblem(problem):
    """
    Fractional order Brusselator system:

              \partial_t U = -(-\Delta)^\alpha U + (B-1)*U + Q^2 V + B/Q * U**2 + 2*Q*U*V + U**2 * V
    \eta**2 * \partial_t V = -(-\Delta)^\beta  U - B*U     - Q^2 V - B/Q * U**2 - 2*Q*U*V - U**2 * V

    with zero flux conditions on U and V.

    s    = \beta/\alpha
    \eta = sqrt(D_X**s / D_Y)
    Q    = A \eta

    """

    def setDriverArgs(self, driver):
        driver.add('domain', acceptedValues=['disc', 'rectangle', 'twinDisc'], help='computational domain')
        driver.add('bc', acceptedValues=['Neumann', 'Dirichlet'], help='type of boundary condition')
        driver.add('noRef', 3, help='number of uniform mesh refinements')
        driver.add('problem', acceptedValues=['spots', 'stripes'], help='pre-defined problems')
        driver.add('T', 200., help='final time')

    def processImpl(self, params):
        from PyNucleus.fem.femCy import brusselator

        if params['problem'] == 'spots':
            self.alpha = self.beta = 0.75
            x = 0.1
            eps = 0.1
            self.eta = 0.2

            if params['domain'] == 'disc':
                z1, z2 = 0., 0.
                R = 10.
            elif params['domain'] == 'twinDisc':
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

        elif params['problem'] == 'stripes':
            self.alpha = self.beta = 0.75
            x = 1.5
            eps = 1.0
            self.eta = 0.2

            if params['domain'] == 'twinDisc':
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

        self.dim = nonlocalMeshFactory.getDim(params['domain'])
        self.kernelU = kernelFactory('fractional', s=self.alpha, dim=self.dim, horizon=np.inf)
        self.kernelV = kernelFactory('fractional', s=self.beta, dim=self.dim, horizon=np.inf)
        self.nonlinearity = brusselator(self.B, self.Q)

        if params['bc'] == 'Neumann':
            self.boundaryCondition = HOMOGENEOUS_NEUMANN
        elif params['bc'] == 'Dirichlet':
            self.boundaryCondition = HOMOGENEOUS_DIRICHLET

        if params['domain'] == 'disc':
            self.mesh, nI = nonlocalMeshFactory('disc',
                                                h=10.,
                                                radius=50.,
                                                kernel=self.kernelU,
                                                boundaryCondition=self.boundaryCondition)
        elif params['domain'] == 'square':
            self.mesh = nonlocalMeshFactory('rectangle',
                                            ax=-50., ay=-50.,
                                            bx=50., by=50.,
                                            N=5, M=5,
                                            kernel=self.kernelU,
                                            boundaryCondition=self.boundaryCondition)

        elif params['domain'] == 'twinDisc':
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
                d.append((k, str(self.__getattr__(k))))
            except KeyError:
                d.append((k, str(params[k])))
        return '-'.join(['brusselator'] + [key + '=' + v for key, v in d])
