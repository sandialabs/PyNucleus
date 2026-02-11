###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.factory import factory
import numpy as np
from PyNucleus_base import REAL

from PyNucleus_fem.mesh import (simpleInterval,
                                intervalWithInteraction,
                                uniformSquare, squareWithInteractions,
                                discWithInteraction,
                                gradedDiscWithInteraction,
                                graded_interval,
                                # double_graded_interval,
                                double_graded_interval_with_interaction,
                                discWithIslands,
                                twinDisc,
                                # box,
                                # boxWithInteractions,
                                ball)
from PyNucleus_fem.functions import (Lambda, constant,
                                     squareIndicator, radialIndicator)
from PyNucleus_fem.mesh import meshFactory as meshFactoryClass
from PyNucleus_fem.DoFMaps import P1_DoFMap
from PyNucleus_fem import (PHYSICAL, NO_BOUNDARY,
                           DIRICHLET, HOMOGENEOUS_DIRICHLET,
                           NEUMANN, HOMOGENEOUS_NEUMANN,
                           NORM)
from . twoPointFunctions import (constantTwoPoint,
                                 temperedTwoPoint,
                                 leftRightTwoPoint,
                                 interfaceTwoPoint,
                                 lambdaTwoPoint,
                                 lookupTwoPoint)
from . interactionDomains import (fullSpace,
                                  ball1_retriangulation,
                                  ball1_barycenter,
                                  ball2_retriangulation,
                                  ball2_barycenter,
                                  ballInf_retriangulation,
                                  ballInf_barycenter,
                                  ellipse_retriangulation,
                                  ellipse_barycenter)
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
from . kernels import (Kernel,
                       FractionalKernel,
                       FRACTIONAL, INDICATOR, PERIDYNAMIC, GAUSSIAN, EXPONENTIAL, POLYNOMIAL,
                       LOGINVERSEDISTANCE, MONOMIAL,
                       )
from PyNucleus_fem import functionFactory
from . nonlocal_functions import (solFractional,
                                  rhsFractional1D, solFractional1D,
                                  rhsFractional2D, solFractional2D,
                                  solFractional2Dcombination,
                                  rhsFractional2Dcombination,
                                  solutionArbitraryOrder,
                                  rhsArbitraryOrder)


def solFractional2D_nonPeriodic(s):
    import numpy as np
    return solFractional2Dcombination(s, [{'n': 2, 'l': 2, 'angular_shift': 0.},
                                          {'n': 1, 'l': 5, 'angular_shift': np.pi/3.}])


def rhsFractional2D_nonPeriodic(s):
    import numpy as np
    return rhsFractional2Dcombination(s, [{'n': 2, 'l': 2, 'angular_shift': 0.},
                                          {'n': 1, 'l': 5, 'angular_shift': np.pi/3.}])




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
twoPointFunctionFactory.register('lambda', lambdaTwoPoint)
twoPointFunctionFactory.register('lookup', lookupTwoPoint)

interactionFactory = factory()
interactionFactory.register('fullSpace', fullSpace, aliases=['full'])
interactionFactory.register('ball2_retriangulation', ball2_retriangulation, aliases=['ball2', '2', 2])
interactionFactory.register('ball2_barycenter', ball2_barycenter)
interactionFactory.register('ball1_retriangulation', ball1_retriangulation, aliases=['ball1', '1', 1])
interactionFactory.register('ball1_barycenter', ball1_barycenter)
interactionFactory.register('ballInf_retriangulation', ballInf_retriangulation, aliases=['ballInf', 'inf', np.inf])
interactionFactory.register('ballInf_barycenter', ballInf_barycenter)
interactionFactory.register('ellipse_retriangulation', ellipse_retriangulation, aliases=['ellipse'])
interactionFactory.register('ellipse_barycenter', ellipse_barycenter)


def getKernel(dim,
              s=None,
              horizon=None,
              scaling=None,
              interaction=None,
              normalized=True,
              piecewise=True,
              phi=None,
              kernel=FRACTIONAL,
              monomialPower=np.nan,
              exponentInverse=np.nan,
              max_horizon=np.nan,
              variance=1.,
              exponentialRate=1.0):
    from . kernels import (_getKernelType,
                           MANIFOLD_FRACTIONAL)
    kType = _getKernelType(kernel)
    if kType == FRACTIONAL:
        return FractionalKernel.build(dim=dim, s=s, horizon=horizon, interaction=interaction, scaling=scaling, normalized=normalized, piecewise=piecewise, phi=phi, max_horizon=max_horizon)
    else:
        return Kernel.build(dim=dim,
                            kernel=kType,
                            horizon=horizon,
                            scaling=scaling,
                            interaction=interaction,
                            normalized=normalized,
                            piecewise=piecewise,
                            phi=phi,
                            max_horizon=max_horizon,
                            monomialPower=monomialPower,
                            variance=variance,
                            exponentialRate=exponentialRate)
kernelFactory = factory()
kernelFactory.register('fractional', FractionalKernel.build)
kernelFactory.register('indicator', Kernel.build, params={'kernel': INDICATOR}, aliases=['constant'])
kernelFactory.register('inverseDistance', Kernel.build, params={'kernel': PERIDYNAMIC}, aliases=['peridynamic', 'inverseOfDistance'])
kernelFactory.register('gaussian', Kernel.build, params={'kernel': GAUSSIAN})
kernelFactory.register('exponential', Kernel.build, params={'kernel': EXPONENTIAL})
kernelFactory.register('polynomial', Kernel.build, params={'kernel': POLYNOMIAL})
kernelFactory.register('logInverseDistance', Kernel.build, params={'kernel': LOGINVERSEDISTANCE})
kernelFactory.register('monomial', Kernel.build, params={'kernel': MONOMIAL})


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
        elif isinstance(kernel.horizon, constant):
            horizonValue = kernel.horizon.value
        else:
            horizonValue = kernel.max_horizon

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
                raise NotImplementedError("Non-homogeneous Dirichlet conditions for infinite horizon kernels are not implemented.")
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
    eps = 1e-12
    domainIndicator = squareIndicator(np.array([a+eps], dtype=REAL),
                                      np.array([b-eps], dtype=REAL))
    interactionIndicator = Lambda(lambda x: 1. if ((x[0] < a-eps) or (b+eps < x[0])) else 0.)
    boundaryIndicator = Lambda(lambda x: 1. if ((a-eps < x[0] < a+eps) or (b-eps < x[0] < b+eps)) else 0.)
    return domainIndicator, boundaryIndicator, interactionIndicator


def squareIndicators(ax=-1., ay=-1., bx=1., by=1., **kwargs):
    eps = 1e-12
    domainIndicator = squareIndicator(np.array([ax+eps, ay+eps], dtype=REAL),
                                      np.array([bx-eps, by-eps], dtype=REAL))
    interactionIndicator = constant(1.)-squareIndicator(np.array([ax-eps, ay-eps], dtype=REAL),
                                                        np.array([bx+eps, by+eps], dtype=REAL))
    boundaryIndicator = constant(1.)-domainIndicator-interactionIndicator
    return domainIndicator, boundaryIndicator, interactionIndicator


def radialIndicators(*args, **kwargs):
    eps = 1e-12
    domainIndicator = radialIndicator(1.-eps)
    interactionIndicator = constant(1.)-radialIndicator(1.+eps)
    boundaryIndicator = radialIndicator(1.+eps)-radialIndicator(1.-eps)
    return domainIndicator, boundaryIndicator, interactionIndicator


def twinDiscIndicators(radius=1., sep=0.1, **kwargs):
    eps = 1e-9
    domainIndicator = (radialIndicator(radius-eps, np.array([sep/2+radius, 0.], dtype=REAL)) +
                       radialIndicator(radius-eps, np.array([-sep/2-radius, 0.], dtype=REAL)))
    interactionIndicator = constant(1.)-(radialIndicator(radius+eps, np.array([sep/2+radius, 0.], dtype=REAL)) +
                                         radialIndicator(radius+eps, np.array([-sep/2-radius, 0.], dtype=REAL)))
    boundaryIndicator = ((radialIndicator(radius+eps, np.array([sep/2+radius, 0.], dtype=REAL)) +
                          radialIndicator(radius+eps, np.array([-sep/2-radius, 0.], dtype=REAL))) -
                         (radialIndicator(radius-eps, np.array([sep/2+radius, 0.], dtype=REAL)) +
                          radialIndicator(radius-eps, np.array([-sep/2-radius, 0.], dtype=REAL))))
    return domainIndicator, boundaryIndicator, interactionIndicator


def boxIndicators(ax=-1., ay=-1., az=-1., bx=1., by=1., bz=1., **kwargs):
    eps = 1e-9
    domainIndicator = squareIndicator(np.array([ax+eps, ay+eps, az+eps], dtype=REAL),
                                      np.array([bx-eps, by-eps, bz-eps], dtype=REAL))
    interactionIndicator = constant(1.)-squareIndicator(np.array([ax-eps, ay-eps, az-eps], dtype=REAL),
                                                        np.array([bx+eps, by+eps, bz+eps], dtype=REAL))
    boundaryIndicator = constant(1.)-domainIndicator-interactionIndicator
    return domainIndicator, boundaryIndicator, interactionIndicator


def ballWithInteractions(*args, **kwargs):
    radius = kwargs.get('radius')
    horizon = kwargs.get('horizon')
    kwargs['radius'] = radius+horizon
    return ball(**kwargs)


nonlocalMeshFactory = nonlocalMeshFactoryClass()
nonlocalMeshFactory.register('interval', simpleInterval, intervalWithInteraction, 1, intervalIndicators,
                             {'a': -1, 'b': 1}, {'a': -1, 'b': 1})
nonlocalMeshFactory.register('gradedInterval', graded_interval, double_graded_interval_with_interaction, 1, intervalIndicators,
                             {'a': -1, 'b': 1, 'mu': 2., 'mu2': 2.}, {'a': -1, 'b': 1, 'mu_ll': 2., 'mu_rr': 2.})
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
#                              {'Nx': 2, 'Ny': 2, 'Nz': 2, 'ax': -1, 'ay': -1, 'az': -1, 'bx': 1, 'by': 1, 'bz': 1},
#                              {'Nx': 2, 'Ny': 2, 'Nz': 2, 'ax': -1, 'ay': -1, 'az': -1, 'bx': 1, 'by': 1, 'bz': 1})
nonlocalMeshFactory.register('ball', ball, ballWithInteractions, 3, radialIndicators,
                             {'radius': 1.}, {'radius': 1.})

functionFactory.register('solFractional', solFractional)
functionFactory.register('solFractional1D', solFractional1D)
functionFactory.register('solFractional2D', solFractional2D)
functionFactory.register('rhsFractional1D', rhsFractional1D)
functionFactory.register('rhsFractional2D', rhsFractional2D)
functionFactory.register('solutionArbitraryOrder', solutionArbitraryOrder)
functionFactory.register('rhsArbitraryOrder', rhsArbitraryOrder)
