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
    def setDriverArgs(self):
        p = self.driver.addGroup('problem')
        self.setDriverFlag('domain', 'square', acceptedValues=['interval', 'cube', 'standardSimplex3D', 'fichera', 'gradedSquare', 'gradedCube'], group=p)
        self.setDriverFlag('problem', 'sin', acceptedValues=['reac-sin', 'diffusivity-sin', 'poly', 'fichera', 'cos'], group=p)
        self.setDriverFlag('noRef', argInterpreter=int, group=p)
        self.setDriverFlag('element', 'P1', acceptedValues=['P1', 'P2', 'P3'], group=p)
        self.setDriverFlag('symmetric', False, group=p)
        self.setDriverFlag('reorder', False, group=p)

    def processCmdline(self, params):
        domain = params['domain']
        element = params['element']
        noRef = params['noRef']
        if domain in ('interval', 'unitInterval'):
            if noRef is None:
                noRef = {'P1': 15, 'P2': 14, 'P3': 13}[element]
        elif domain in ('square', 'unitSquare', 'gradedSquare'):
            if noRef is None:
                noRef = {'P1': 9, 'P2': 8, 'P3': 7}[element]
        elif domain in ('square', 'unitSquare', 'gradedSquare'):
            if noRef is None:
                noRef = {'P1': 9, 'P2': 8, 'P3': 7}[element]
        elif domain == 'graded_disc':
            if noRef is None:
                noRef = {'P1': 5, 'P2': 4, 'P3': 3}[element]
        elif domain in ('cube', 'gradedCube'):
            if noRef is None:
                noRef = {'P1': 6, 'P2': 5, 'P3': 4}[element]
        elif domain == 'standardSimplex3D':
            if noRef is None:
                noRef = {'P1': 2}[element]
        elif domain == 'fichera':
            if noRef is None:
                noRef = {'P1': 5, 'P2': 4}[element]
        params['noRef'] = noRef
        super().processCmdline(params)

    @problem.generates(['dim', 'diffusivity', 'reaction', 'rhsFun', 'exactSolution', 'L2ex', 'H10ex', 'boundaryCond'])
    def processProblem(self, domain, problem, noRef, element, symmetric, reorder):
        from . functions import constant, Lambda
        from . factories import (meshFactory, rhsFunSin1D,
                                 rhsFunSin2D, rhsFunSin3D, solSin1D, solSin2D, solSin3D, cos2D,
                                 rhsCos2D, rhsFichera, solFichera,)
        self.diffusivity = None
        self.reaction = None
        self.dim = meshFactory.getDim(domain)
        if domain in ('interval', 'unitInterval'):
            if problem == 'sin':
                self.rhsFun = rhsFunSin1D
                self.exactSolution = solSin1D
                self.L2ex = 1/2
                self.H10ex = np.pi**2/2
                self.boundaryCond = None
            elif problem == 'reac-sin':
                self.rhsFun = Lambda(lambda x: (np.pi**2.0 + 10.)*np.sin(np.pi*x[0]))
                self.exactSolution = solSin1D
                self.L2ex = 1/2
                self.H10ex = (np.pi**2 + 10.)/2
                self.reaction = 10.
                self.boundaryCond = None
            else:
                raise NotImplementedError()
        elif domain in ('square', 'unitSquare', 'gradedSquare'):
            if problem == 'sin':
                self.rhsFun = rhsFunSin2D
                self.exactSolution = solSin2D
                self.L2ex = 1/4
                self.H10ex = 2*np.pi**2/4
                self.boundaryCond = None
            elif problem == 'cos':
                self.rhsFun = rhsCos2D
                self.exactSolution = cos2D
                self.L2ex = 1/4
                self.H10ex = 2*np.pi**2/4
                self.boundaryCond = cos2D
            elif problem == 'reac-sin':
                self.rhsFun = Lambda(lambda x: (2*np.pi**2.0 + 10.)*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
                self.exactSolution = solSin2D
                self.L2ex = 1/4
                self.H10ex = (2*np.pi**2 + 10.)/4
                self.boundaryCond = None
                self.reaction = 10.
            elif problem == 'diffusivity-sin':
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
            elif problem == 'poly':
                self.rhsFun = Lambda(lambda x: 32*x[0]*(1-x[0])+32*x[1]*(1-x[1]))
                self.exactSolution = Lambda(lambda x: 16*x[0]*x[1]*(1-x[0])*(1-x[1]))
                self.L2ex = 256/900
                self.H10ex = 256/45
                self.boundaryCond = None
            elif problem == 'variable-reac-sin':
                self.rhsFun = Lambda(lambda x: (2*np.pi**2.0 + 10.)*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
                self.exactSolution = solSin2D
                self.L2ex = 1/4
                self.H10ex = (2*np.pi**2 + 10.)/4
                self.boundaryCond = None
                self.reaction = Lambda(lambda x: 0. if x[0] < 0.5 else 2000.)
            else:
                raise NotImplementedError()
        elif domain == 'graded_disc':
            if problem == 'constant':
                self.rhsFun = constant(1.)
                self.exactSolution = None
                self.L2ex = None
                self.H10ex = None
                self.boundaryCond = None
            else:
                raise NotImplementedError()
        elif domain in ('cube', 'gradedCube'):
            if problem == 'sin':
                self.rhsFun = rhsFunSin3D
                self.exactSolution = solSin3D
                self.L2ex = 1/8
                self.H10ex = 3*np.pi**2/8
                self.boundaryCond = None
            elif problem == 'variable-reac-sin':
                self.rhsFun = constant(1.)
                self.exactSolution = None
                self.L2ex = np.nan
                self.H10ex = np.nan
                self.boundaryCond = None
                self.reaction = Lambda(lambda x: 0. if x[0] < 0.5 else 2000.)
            else:
                raise NotImplementedError()
        elif domain == 'standardSimplex3D':
            if problem == 'poly':
                self.rhsFun = Lambda(lambda x: 2*(x[1]*x[2]+x[0]*x[2]+x[0]*x[1]))
                self.L2ex = 1/8
                self.H10ex = 3*np.pi**2/8
                self.boundaryCond = None
            else:
                raise NotImplementedError()
        elif domain == 'fichera':
            if problem == 'fichera':
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
    def setDriverArgs(self):
        p = self.driver.addGroup('problem')
        self.setDriverFlag('domain', acceptedValues=['square', 'interval', 'cube'], group=p)
        self.setDriverFlag('problem', acceptedValues=['wave', 'greens'], group=p)
        self.setDriverFlag('element', 'P1', acceptedValues=['P1'], group=p)
        self.setDriverFlag('frequency', 40., group=p)
        self.setDriverFlag('symmetric', False, group=p)
        self.setDriverFlag('reorder', False, group=p)

    @problem.generates(['dim', 'noRef', 'solEx', 'rhs', 'boundaryCond'])
    def processProblem(self, domain, problem, element, frequency, symmetric, reorder):
        from . import meshFactory
        self.dim = meshFactory.getDim(domain)
        if domain == 'interval':
            self.noRef = 7

            def n(x):
                if x[0] == 0:
                    return np.array([-1.], dtype=REAL)
                elif x[0] == 1:
                    return np.array([1.], dtype=REAL)
                else:
                    raise NotImplementedError()

            if problem == 'wave':
                xi = np.array([0.5], dtype=REAL)
                self.solEx = complexLambda(lambda x: np.exp(1j*np.vdot(xi, x)))
                self.rhs = complexLambda(lambda x: (np.vdot(xi, xi)-self.frequency**2) * self.solEx(x))
                self.boundaryCond = complexLambda(lambda x: 1j*(np.vdot(xi, n(x))+self.frequency) * self.solEx(x))
            elif problem == 'greens':
                self.rhs = wrapRealToComplexFunction(radialIndicator(1e-2, np.array([0.5])))
                self.solEx = None
                self.boundaryCond = None
            else:
                raise NotImplementedError(problem)
        elif domain == 'square':
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

            if problem == 'wave':
                xi = np.array([0.5, 0.25], dtype=REAL)
                self.solEx = waveFunction(xi)
                self.rhs = (np.vdot(xi, xi)-self.frequency**2) * self.solEx
                self.boundaryCond = complexLambda(lambda x: 1j*(np.vdot(xi, n(x))+self.frequency) * self.solEx(x))
            elif problem == 'greens':
                self.rhs = wrapRealToComplexFunction(radialIndicator(1e-2, np.array([0.5, 0.5])))
                self.solEx = None
                self.boundaryCond = None
            else:
                raise NotImplementedError(problem)
        elif domain == 'cube':
            self.noRef = 6

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

            if problem == 'wave':
                xi = np.array([0.75, 0.5, 0.25], dtype=REAL)
                self.solEx = waveFunction(xi)
                self.rhs = (np.vdot(xi, xi)-self.frequency**2) * self.solEx
                self.boundaryCond = complexLambda(lambda x: 1j*(np.vdot(xi, n(x))+self.frequency) * self.solEx(x))
            elif problem == 'greens':
                self.rhs = wrapRealToComplexFunction(radialIndicator(1e-1, np.array([0.5, 0.5, 0.5])))
                self.solEx = None
                self.boundaryCond = None
            else:
                raise NotImplementedError(problem)
        else:
            raise NotImplementedError(domain)
