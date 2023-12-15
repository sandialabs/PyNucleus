###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import sys

subpackages = {}
__all__ = []

from PyNucleus_packageTools import *
import PyNucleus_packageTools as packageTools

sys.modules['PyNucleus.packageTools'] = packageTools
subpackages['packageTools'] = packageTools

from PyNucleus_base import *
import PyNucleus_base as base

sys.modules['PyNucleus.base'] = base
subpackages['base'] = base
__all__ += base.__all__

from PyNucleus_metisCy import *
import PyNucleus_metisCy as metisCy

sys.modules['PyNucleus.metisCy'] = metisCy
subpackages['metisCy'] = metisCy
__all__ += metisCy.__all__

from PyNucleus_fem import *
import PyNucleus_fem as fem

sys.modules['PyNucleus.fem'] = fem
subpackages['fem'] = fem
__all__ += fem.__all__

from PyNucleus_multilevelSolver import *
import PyNucleus_multilevelSolver as multilevelSolver

sys.modules['PyNucleus.multilevelSolver'] = multilevelSolver
subpackages['multilevelSolver'] = multilevelSolver
__all__ += multilevelSolver.__all__

try:
    from PyNucleus_nl import *
    import PyNucleus_nl as nl

    sys.modules['PyNucleus.nl'] = nl
    subpackages['nl'] = nl
    __all__ += nl.__all__
except ImportError:
    pass
