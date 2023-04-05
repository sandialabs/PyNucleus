###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import importlib
import pkgutil
import sys

subpackages = {}
__all__ = []
for finder, name, ispkg in pkgutil.iter_modules():
    if ispkg and name.find('PyNucleus_') == 0:
        importName = name[len('PyNucleus_'):]
        module = importlib.import_module(name, 'PyNucleus')
        sys.modules['PyNucleus.'+importName] = module
        subpackages[importName] = module
        names = [name for name in module.__dict__ if not name.startswith('_')]
        locals().update({name: getattr(module, name) for name in names})
