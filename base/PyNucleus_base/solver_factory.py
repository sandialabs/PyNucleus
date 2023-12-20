###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import warnings
from . linear_operators import ComplexLinearOperator
from . factory import factory


class solverFactory(factory):
    def __init__(self):
        self.singleLevelSolverFactory = factory()
        self.multiLevelSolverFactory = factory()

    def isRegistered(self, name):
        return (self.singleLevelSolverFactory.isRegistered(name) or
                self.multiLevelSolverFactory.isRegistered(name) or
                self.isRegisteredComboSolver(name))

    def isRegisteredComboSolver(self, name):
        names = name.split('-')
        if len(names) <= 1:
            return False
        for name in names:
            if not self.isRegistered(name):
                return False
        return True

    def register(self, name, classType, isMultilevelSolver=False, aliases=[]):
        if not isMultilevelSolver:
            self.singleLevelSolverFactory.register(name, classType, aliases=aliases)
        else:
            self.multiLevelSolverFactory.register(name, classType, aliases=aliases)

    def build(self, name, **kwargs):
        setup = kwargs.pop('setup', False)
        if len(name.split('-')) == 1:
            name = self.getCanonicalName(name)

            if self.singleLevelSolverFactory.isRegistered(name):
                A = kwargs.pop('A', None)
                hierarchy = kwargs.pop('hierarchy', None)
                if A is None and hierarchy is not None:
                    if isinstance(hierarchy, list):
                        A = hierarchy[-1]['A']
                    else:
                        raise NotImplementedError()
                num_rows = kwargs.pop('num_rows', -1)
                if isinstance(A, ComplexLinearOperator) and self.singleLevelSolverFactory.isRegistered('complex_'+name):
                    name = 'complex_'+name
                solver = self.singleLevelSolverFactory.build(name, A, num_rows)
            elif self.multiLevelSolverFactory.isRegistered(name):
                kwargs.pop('A', None)
                hierarchy = kwargs.pop('hierarchy')
                smoother = kwargs.pop('smoother', 'jacobi')
                if (not isinstance(hierarchy, list) and
                        isinstance(hierarchy.builtHierarchies[-1].algebraicLevels[-1].A, ComplexLinearOperator) and
                        self.multiLevelSolverFactory.isRegistered('complex_'+name)):
                    name = 'complex_'+name
                solver = self.multiLevelSolverFactory.build(name, hierarchy, smoother, **kwargs)
            else:
                raise KeyError(name)
            for key in kwargs:
                if hasattr(solver, key):
                    solver.__setattr__(key, kwargs[key])
                elif key in ('tolerance', 'maxIter'):
                    pass
                else:
                    msg = '{} does not have attr \"{}\"'.format(solver, key)
                    warnings.warn(msg)
                    # raise NotImplementedError(msg)
            if setup:
                solver.setup()
            return solver
        else:
            names = name.split('-')
            solvers = []
            for name in names:
                params = kwargs.get(name, {})
                if 'A' in kwargs:
                    params['A'] = kwargs['A']
                if 'num_rows' in kwargs:
                    params['num_rows'] = kwargs['num_rows']
                if 'hierarchy' in kwargs:
                    params['hierarchy'] = kwargs['hierarchy']
                solvers.append(self.build(name, **params))
            if setup:
                for k in range(len(solvers)):
                    if not solvers[k].initialized:
                        solvers[k].setup()
            for k in range(len(solvers)-1):
                solvers[k].setPreconditioner(solvers[k+1].asPreconditioner())
            return solvers[0]

    def __repr__(self):
        s = ''
        if self.singleLevelSolverFactory.numRegistered() > 0:
            s += 'Single level solvers:\n'
            s += repr(self.singleLevelSolverFactory)
        if self.multiLevelSolverFactory.numRegistered() > 0:
            s += 'Multi level solvers:\n'
            s += repr(self.multiLevelSolverFactory)
        return s
