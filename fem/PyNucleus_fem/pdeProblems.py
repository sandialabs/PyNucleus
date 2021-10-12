###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from PyNucleus_base import REAL
from PyNucleus_base.utilsFem import problem
from . functions import complexLambda, wrapRealToComplexFunction, waveFunction, radialIndicator


class diffusionProblem(problem):
    def setDriverArgs(self, driver):
        p = driver.addGroup('problem')
        p.add('domain', 'square', acceptedValues=['interval', 'cube', 'standardSimplex3D', 'fichera', 'gradedSquare', 'gradedCube'])
        p.add('problem', 'sin', acceptedValues=['reac-sin', 'diffusivity-sin', 'poly', 'fichera', 'cos'])
        p.add('noRef', argInterpreter=int)
        p.add('element', 'P1', acceptedValues=['P1', 'P2', 'P3'])
        p.add('symmetric', False)
        p.add('reorder', False)

    def processImpl(self, params):
        from . import (rhsFunSin1D, rhsFunSin2D, rhsFunSin3D,
                       solSin1D, solSin2D, solSin3D,
                       cos2D, rhsCos2D,
                       rhsFichera, solFichera,
                       constant, Lambda,
                       meshFactory)
        element = params['element']
        self.diffusivity = None
        self.reaction = None
        self.dim = meshFactory.getDim(params['domain'])
        if params['domain'] in ('interval', 'unitInterval'):

            if params['noRef'] is None:
                self.noRef = {'P1': 15, 'P2': 14, 'P3': 13}[element]
            if params['problem'] == 'sin':
                self.rhsFun = rhsFunSin1D
                self.exactSolution = solSin1D
                self.L2ex = 1/2
                self.H10ex = np.pi**2/2
                self.boundaryCond = None
            elif params['problem'] == 'reac-sin':
                self.rhsFun = Lambda(lambda x: (np.pi**2.0 + 10.)*np.sin(np.pi*x[0]))
                self.exactSolution = solSin1D
                self.L2ex = 1/2
                self.H10ex = (np.pi**2 + 10.)/2
                self.reaction = 10.
                self.boundaryCond = None
            else:
                raise NotImplementedError()
        elif params['domain'] in ('square', 'unitSquare', 'gradedSquare'):
            if params['noRef'] is None:
                self.noRef = {'P1': 9, 'P2': 8, 'P3': 7}[element]
            if params['problem'] == 'sin':
                self.rhsFun = rhsFunSin2D
                self.exactSolution = solSin2D
                self.L2ex = 1/4
                self.H10ex = 2*np.pi**2/4
                self.boundaryCond = None
            elif params['problem'] == 'cos':
                self.rhsFun = rhsCos2D
                self.exactSolution = cos2D
                self.L2ex = 1/4
                self.H10ex = 2*np.pi**2/4
                self.boundaryCond = cos2D
            elif params['problem'] == 'reac-sin':
                self.rhsFun = Lambda(lambda x: (2*np.pi**2.0 + 10.)*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
                self.exactSolution = solSin2D
                self.L2ex = 1/4
                self.H10ex = (2*np.pi**2 + 10.)/4
                self.boundaryCond = None
                self.reaction = 10.
            elif params['problem'] == 'diffusivity-sin':
                self.diffusivity = Lambda(lambda x: np.exp(np.sin(np.pi*x[0]) *
                                                           np.sin(np.pi*x[1])))
                self.rhsFun = Lambda(lambda x: -np.pi**2 *
                                     np.exp(np.sin(np.pi*x[0])*np.sin(np.pi*x[1])) *
                                     (np.sin(np.pi*x[0])**2 * np.cos(np.pi*x[1])**2 +
                                      np.cos(np.pi*x[0])**2 * np.sin(np.pi*x[1])**2 -
                                      2*np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])))
                self.exactSolution = solSin2D
                self.L2ex = 1/4
                self.H10ex = np.nan
                self.boundaryCond = None
            elif params['problem'] == 'poly':
                self.rhsFun = Lambda(lambda x: 32*x[0]*(1-x[0])+32*x[1]*(1-x[1]))
                self.exactSolution = Lambda(lambda x: 16*x[0]*x[1]*(1-x[0])*(1-x[1]))
                self.L2ex = 256/900
                self.H10ex = 256/45
                self.boundaryCond = None
            elif params['problem'] == 'variable-reac-sin':
                self.rhsFun = Lambda(lambda x: (2*np.pi**2.0 + 10.)*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
                self.exactSolution = solSin2D
                self.L2ex = 1/4
                self.H10ex = (2*np.pi**2 + 10.)/4
                self.boundaryCond = None
                self.reaction = Lambda(lambda x: 0. if x[0] < 0.5 else 2000.)
            else:
                raise NotImplementedError()
        elif params['domain'] == 'graded_disc':
            if params['noRef'] is None:
                self.noRef = {'P1': 5, 'P2': 4, 'P3': 3}[element]
            if params['problem'] == 'constant':
                self.rhsFun = constant(1.)
                self.exactSolution = None
                self.L2ex = None
                self.H10ex = None
                self.boundaryCond = None
            else:
                raise NotImplementedError()
        elif params['domain'] in ('cube', 'gradedCube'):
            if params['noRef'] is None:
                self.noRef = {'P1': 6, 'P2': 5, 'P3': 4}[element]
            if params['problem'] == 'sin':
                self.rhsFun = rhsFunSin3D
                self.exactSolution = solSin3D
                self.L2ex = 1/8
                self.H10ex = 3*np.pi**2/8
                self.boundaryCond = None
            elif params['problem'] == 'variable-reac-sin':
                self.rhsFun = constant(1.)
                self.exactSolution = None
                self.L2ex = np.nan
                self.H10ex = np.nan
                self.boundaryCond = None
                self.reaction = Lambda(lambda x: 0. if x[0] < 0.5 else 2000.)
            else:
                raise NotImplementedError()
        elif params['domain'] == 'standardSimplex3D':
            if params['noRef'] is None:
                self.noRef = {'P1': 2}[element]
            if params['problem'] == 'poly':
                self.rhsFun = Lambda(lambda x: 2*(x[1]*x[2]+x[0]*x[2]+x[0]*x[1]))
                self.L2ex = 1/8
                self.H10ex = 3*np.pi**2/8
                self.boundaryCond = None
            else:
                raise NotImplementedError()
        elif params['domain'] == 'fichera':
            if params['noRef'] is None:
                self.noRef = {'P1': 5, 'P2': 4}[element]
            if params['problem'] == 'fichera':
                self.rhsFun = rhsFichera
                self.exactSolution = solFichera
                self.L2ex = None
                # H10ex = 7/8**9.52031/4
                self.H10ex = None
                self.boundaryCond = solFichera
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()


class helmholtzProblem(problem):
    def setDriverArgs(self, driver):
        p = driver.addGroup('problem')
        p.add('domain', acceptedValues=['square', 'interval', 'cube'])
        p.add('problem', acceptedValues=['wave', 'greens'])
        p.add('element', 'P1', acceptedValues=['P1'])
        p.add('frequency', 40.)
        p.add('symmetric', False)
        p.add('reorder', False)

    def processImpl(self, params):
        from . import meshFactory
        self.dim = meshFactory.getDim(params['domain'])
        if params['domain'] == 'interval':
            self.noRef = 7

            def n(x):
                if x[0] == 0:
                    return np.array([-1.], dtype=REAL)
                elif x[0] == 1:
                    return np.array([1.], dtype=REAL)
                else:
                    raise NotImplementedError()

            if params['problem'] == 'wave':
                xi = np.array([0.5], dtype=REAL)
                self.solEx = complexLambda(lambda x: np.exp(1j*np.vdot(xi, x)))
                self.rhs = complexLambda(lambda x: (np.vdot(xi, xi)-self.frequency**2) * self.solEx(x))
                self.boundaryCond = complexLambda(lambda x: 1j*(np.vdot(xi, n(x))+self.frequency) * self.solEx(x))
            elif params['problem'] == 'greens':
                self.rhs = wrapRealToComplexFunction(radialIndicator(1e-2, np.array([0.5])))
                self.solEx = None
                self.boundaryCond = None
            else:
                raise NotImplementedError(params['problem'])
        elif params['domain'] == 'square':
            self.noRef = 8

            def n(x):
                if x[1] == 0:
                    return np.array([0., -1.], dtype=REAL)
                elif x[1] == 1.:
                    return np.array([0., 1.], dtype=REAL)
                elif x[0] == 0.:
                    return np.array([-1., 0.], dtype=REAL)
                elif x[0] == 1.:
                    return np.array([1., 0.], dtype=REAL)
                else:
                    raise NotImplementedError()

            if params['problem'] == 'wave':
                xi = np.array([0.5, 0.25], dtype=REAL)
                self.solEx = waveFunction(xi)
                self.rhs = (np.vdot(xi, xi)-self.frequency**2) * self.solEx
                self.boundaryCond = complexLambda(lambda x: 1j*(np.vdot(xi, n(x))+self.frequency) * self.solEx(x))
            elif params['problem'] == 'greens':
                self.rhs = wrapRealToComplexFunction(radialIndicator(1e-2, np.array([0.5, 0.5])))
                self.solEx = None
                self.boundaryCond = None
            else:
                raise NotImplementedError(params['problem'])
        elif params['domain'] == 'cube':
            self.noRef = 5

            def n(x):
                if x[2] == 0:
                    return np.array([0., 0., -1.], dtype=REAL)
                elif x[2] == 1.:
                    return np.array([0., 0., 1.], dtype=REAL)
                elif x[1] == 0:
                    return np.array([0., -1., 0.], dtype=REAL)
                elif x[1] == 1.:
                    return np.array([0., 1., 0.], dtype=REAL)
                elif x[0] == 0.:
                    return np.array([-1., 0., 0.], dtype=REAL)
                elif x[0] == 1.:
                    return np.array([1., 0., 0.], dtype=REAL)
                else:
                    raise NotImplementedError()

            if params['problem'] == 'wave':
                xi = np.array([0.75, 0.5, 0.25], dtype=REAL)
                self.solEx = waveFunction(xi)
                self.rhs = (np.vdot(xi, xi)-self.frequency**2) * self.solEx
                self.boundaryCond = complexLambda(lambda x: 1j*(np.vdot(xi, n(x))+self.frequency) * self.solEx(x))
            elif params['problem'] == 'greens':
                self.rhs = wrapRealToComplexFunction(radialIndicator(1e-1, np.array([0.5, 0.5, 0.5])))
                self.solEx = None
                self.boundaryCond = None
            else:
                raise NotImplementedError(params['problem'])
        else:
            raise NotImplementedError(params['domain'])
