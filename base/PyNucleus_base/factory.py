###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from copy import deepcopy


class factory:
    def __init__(self):
        self.classes = {}
        self.aliases = {}

    def getCanonicalName(self, name):
        if isinstance(name, str):
            return name.lower()
        else:
            return name

    def register(self, name, classType, params={}, aliases=[]):
        canonical_name = self.getCanonicalName(name)
        self.classes[canonical_name] = (name, classType, params)
        for alias in aliases:
            canonical_alias = self.getCanonicalName(alias)
            self.aliases[canonical_alias] = (alias, canonical_name)

    def isRegistered(self, name):
        return self.getCanonicalName(name) in self.classes or name in self.aliases

    def __call__(self, name, *args, **kwargs):
        return self.build(name, *args, **kwargs)

    def build(self, name, *args, **kwargs):
        canonical_name = self.getCanonicalName(name)
        if canonical_name in self.aliases:
            canonical_name = self.aliases[canonical_name][1]
        if canonical_name not in self.classes:
            raise KeyError('\'{}\' not in factory. {}'.format(name, repr(self)))
        _, classType, params = self.classes[canonical_name]
        p = deepcopy(params)
        p.update(kwargs)
        obj = classType(*args, **p)
        return obj

    def numRegistered(self, countAliases=False):
        if not countAliases:
            return len(self.classes)
        else:
            return len(self.classes) + len(self.aliases)

    def __str__(self):
        s = ''
        for canonical_name in self.classes:
            a = [self.aliases[canonical_alias][0] for canonical_alias in self.aliases if self.aliases[canonical_alias][1] == canonical_name]
            s += '{} {} {}\n'.format(canonical_name, a, self.classes[canonical_name])
        return s

    def __repr__(self):
        s = 'Available:\n'
        for canonical_name in self.classes:
            name = self.classes[canonical_name][0]
            c = self.classes[canonical_name][1]
            a = [self.aliases[canonical_alias][0] for canonical_alias in self.aliases if self.aliases[canonical_alias][1] == canonical_name]
            sig = c.__doc__
            if sig is None:
                from inspect import signature
                try:
                    sig = signature(c)
                except ValueError:
                    pass
            if isinstance(sig, str) and sig.find('\n'):
                sig = sig.split('\n')[0]
            if len(a) > 0:
                s += '\'{}\' with aliases {}, signature: \'{}\'\n'.format(name, a, sig)
            else:
                s += '\'{}\', signature: \'{}\'\n'.format(name, sig)
        return s

    def print(self):
        print(repr(self))
