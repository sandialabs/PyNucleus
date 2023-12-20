###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus_base.factory import factory
from . functions import (simpleAnisotropy, simpleAnisotropy2, inclusions, inclusionsHong,
                         motorPermeability)
from . functions import (_rhsFunSin1D, _solSin1D, _rhsFunSin2D, _cos1D, _cos2D, _rhsCos2D, _grad_cos2d_n,
                         _rhsFunSin3D, _solSin2D, _solSin3D, Lambda, constant,
                         monomial,
                         complexLambda,
                         real, imag,
                         _rhsFunSin3D_memoized,
                         _rhsFichera, _solFichera,
                         solCos1DHeat, rhsFunCos1DHeat,
                         rhsFunSource1D, rhsFunSource2D,
                         solCos2DHeat, rhsFunCos2DHeat,
                         solFractional, solFractionalDerivative,
                         rhsFractional1D, solFractional1D,
                         rhsFractional2D, solFractional2D,
                         solFractional2Dcombination,
                         rhsFractional2Dcombination,
                         rhsHr,
                         rhsHr2Ddisk,
                         indicatorFunctor,
                         shiftScaleFunctor,
                         squareIndicator,
                         radialIndicator,
                         fractalDiffusivity, expDiffusivity,
                         componentVectorFunction)
from . lookupFunction import lookupFunction, vectorLookupFunction


rhsFunSin1D = _rhsFunSin1D()
rhsFunSin2D = _rhsFunSin2D()
rhsFunSin3D = _rhsFunSin3D()
cos2D = _cos2D()
rhsCos2D = _rhsCos2D()
solSin1D = _solSin1D()
solSin2D = _solSin2D()
solSin3D = _solSin3D()
grad_cos2d_n = _grad_cos2d_n()
rhsFichera = _rhsFichera()
solFichera = _solFichera()


def solFractional2D_nonPeriodic(s):
    import numpy as np
    return solFractional2Dcombination(s, [{'n': 2, 'l': 2, 'angular_shift': 0.},
                                          {'n': 1, 'l': 5, 'angular_shift': np.pi/3.}])


def rhsFractional2D_nonPeriodic(s):
    import numpy as np
    return rhsFractional2Dcombination(s, [{'n': 2, 'l': 2, 'angular_shift': 0.},
                                          {'n': 1, 'l': 5, 'angular_shift': np.pi/3.}])


from . functions import (_rhsBoundaryLayer2D, _solBoundaryLayer2D,
                         _solCornerSingularity2D, rhsMotor,
                         rhsBoundarySingularity2D, solBoundarySingularity2D)


functionFactory = factory()
functionFactory.register('rhsFunSin1D', _rhsFunSin1D)
functionFactory.register('rhsFunSin2D', _rhsFunSin2D)
functionFactory.register('rhsFunSin3D', _rhsFunSin3D)
functionFactory.register('solSin1D', _solSin1D, aliases=['sin1d'])
functionFactory.register('solCos1D', _cos1D, aliases=['cos1d'])
functionFactory.register('solSin2D', _solSin2D, aliases=['sin2d'])
functionFactory.register('solCos2D', _cos2D, aliases=['cos2d'])
functionFactory.register('solSin3D', _solSin3D, aliases=['sin3d'])
functionFactory.register('solFractional', solFractional)
functionFactory.register('solFractionalDerivative', solFractionalDerivative)
functionFactory.register('solFractional1D', solFractional1D)
functionFactory.register('solFractional2D', solFractional2D)
functionFactory.register('rhsFractional1D', rhsFractional1D)
functionFactory.register('rhsFractional2D', rhsFractional2D)
functionFactory.register('constant', constant)
functionFactory.register('monomial', monomial)
functionFactory.register('x0', monomial, params={'exponent': np.array([1., 0., 0.])})
functionFactory.register('x1', monomial, params={'exponent': np.array([0., 1., 0.])})
functionFactory.register('x2', monomial, params={'exponent': np.array([0., 0., 1.])})
functionFactory.register('x0**2', monomial, params={'exponent': np.array([2., 0., 0.])})
functionFactory.register('x1**2', monomial, params={'exponent': np.array([0., 2., 0.])})
functionFactory.register('x2**2', monomial, params={'exponent': np.array([0., 0., 2.])})
functionFactory.register('x0*x1', monomial, params={'exponent': np.array([1., 1., 0.])})
functionFactory.register('x1*x2', monomial, params={'exponent': np.array([0., 1., 1.])})
functionFactory.register('x0*x2', monomial, params={'exponent': np.array([1., 0., 1.])})
functionFactory.register('x0**3', monomial, params={'exponent': np.array([3., 0., 0.])})
functionFactory.register('x1**3', monomial, params={'exponent': np.array([0., 3., 0.])})
functionFactory.register('x2**3', monomial, params={'exponent': np.array([0., 0., 3.])})
functionFactory.register('Lambda', Lambda)
functionFactory.register('complexLambda', complexLambda)
functionFactory.register('squareIndicator', squareIndicator)
functionFactory.register('radialIndicator', radialIndicator)
functionFactory.register('rhsBoundaryLayer2D', _rhsBoundaryLayer2D)
functionFactory.register('solBoundaryLayer2D', _solBoundaryLayer2D)
functionFactory.register('solCornerSingularity2D', _solCornerSingularity2D)
functionFactory.register('solBoundarySingularity2D', solBoundarySingularity2D)
functionFactory.register('rhsBoundarySingularity2D', rhsBoundarySingularity2D)
functionFactory.register('rhsMotor', rhsMotor)
functionFactory.register('simpleAnisotropy', simpleAnisotropy)
functionFactory.register('simpleAnisotropy2', simpleAnisotropy2)
functionFactory.register('inclusions', inclusions)
functionFactory.register('inclusionsHong', inclusionsHong)
functionFactory.register('motorPermeability', motorPermeability)
functionFactory.register('lookup', lookupFunction)
functionFactory.register('vectorLookup', vectorLookupFunction)
functionFactory.register('shiftScaleFunctor', shiftScaleFunctor)
functionFactory.register('componentVectorFunction', componentVectorFunction, aliases=['vector'])


# DoFMaps
from . DoFMaps import (P0_DoFMap, P1_DoFMap, P2_DoFMap, P3_DoFMap, N1e_DoFMap,
                       Product_DoFMap)


class vectorDoFMap:
    def __init__(self, dmType):
        self.dmType = dmType

    def __call__(self, mesh, *args, **kwargs):
        scalarDM = self.dmType(mesh, *args, **kwargs)
        return Product_DoFMap(scalarDM, mesh.dim)


dofmapFactory = factory()
dofmapFactory.register('P0d', P0_DoFMap, aliases=['P0'])
dofmapFactory.register('P1c', P1_DoFMap, aliases=['P1'])
dofmapFactory.register('P2c', P2_DoFMap, aliases=['P2'])
dofmapFactory.register('P3c', P3_DoFMap, aliases=['P3'])
for dmType, dmName in [(P0_DoFMap, 'P0d'), (P1_DoFMap, 'P1c'), (P2_DoFMap, 'P2c'), (P3_DoFMap, 'P3c')]:
    dmNameShort = dmName[:-1]
    dofmapFactory.register('vector'+dmName, vectorDoFMap(dmType), aliases=['vector'+dmNameShort, 'vector-'+dmNameShort, 'vector '+dmNameShort])
dofmapFactory.register('N1e', N1e_DoFMap)


# meshes
from . mesh import (simpleInterval, simpleSquare, simpleLshape, simpleBox, box,
                    circle, graded_circle, cutoutCircle, twinDisc, dumbbell, wrench,
                    Hshape, ball, rectangle, crossSquare,
                    gradedSquare, gradedBox,
                    squareWithCircularCutout, boxWithBallCutout,
                    disconnectedInterval, disconnectedDomain,
                    double_graded_interval,
                    simpleFicheraCube, uniformSquare,
                    standardSimplex2D, standardSimplex3D,
                    intervalWithInteraction,
                    double_graded_interval_with_interaction,
                    discWithIslands,
                    squareWithInteractions,
                    discWithInteraction,
                    gradedDiscWithInteraction,
                    plotFunctions)
from . mesh import meshFactory as meshFactoryClass

meshFactory = meshFactoryClass()
meshFactory.register('simpleInterval', simpleInterval, 1, aliases=['interval'])
meshFactory.register('unitInterval', simpleInterval, 1, params={'a': 0., 'b': 1.})
meshFactory.register('intervalWithInteraction', intervalWithInteraction, 1)
meshFactory.register('disconnectedInterval', disconnectedInterval, 1)
meshFactory.register('simpleSquare', simpleSquare, 2)
meshFactory.register('crossSquare', crossSquare, 2, aliases=['squareCross'])
meshFactory.register('unitSquare', uniformSquare, 2,
                     params={'N': 2, 'ax': 0., 'ay': 0., 'bx': 1., 'by': 1.},
                     aliases=['square'])
meshFactory.register('gradedSquare', gradedSquare, 2)
meshFactory.register('gradedBox', gradedBox, 3, aliases=['gradedCube'])
meshFactory.register('squareWithInteraction', squareWithInteractions, 2)
meshFactory.register('simpleLshape', simpleLshape, 2, aliases=['Lshape', 'L-shape'])
meshFactory.register('circle', circle, 2, aliases=['disc', 'unitDisc', 'ball2d', '2dball'])
meshFactory.register('graded_circle', graded_circle, 2, aliases=['gradedCircle'])
meshFactory.register('discWithInteraction', discWithInteraction, 2)
meshFactory.register('twinDisc', twinDisc, 2)
meshFactory.register('dumbbell', dumbbell, 2)
meshFactory.register('wrench', wrench, 2)
meshFactory.register('cutoutCircle', cutoutCircle, 2, aliases=['cutoutDisc'])
meshFactory.register('squareWithCircularCutout', squareWithCircularCutout, 2)
meshFactory.register('boxWithBallCutout', boxWithBallCutout, 3, aliases=['boxMinusBall'])
meshFactory.register('simpleBox', simpleBox, 3, aliases=['unitBox', 'cube', 'unitCube'])
meshFactory.register('box', box, 3)
meshFactory.register('ball', ball, 3)
meshFactory.register('simpleFicheraCube', simpleFicheraCube, 3, aliases=['fichera', 'ficheraCube'])
meshFactory.register('standardSimplex2D', standardSimplex2D, 2)
meshFactory.register('standardSimplex3D', standardSimplex3D, 3)
