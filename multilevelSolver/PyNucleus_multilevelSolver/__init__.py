###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . hierarchies import EmptyHierarchy, hierarchy, hierarchyManager
from . connectors import (inputConnector, repartitionConnector,
                          )
from . multigrid import V, W, FMG_V, FMG_W
from . geometricMG import paramsForMG, paramsForSerialMG

from PyNucleus_base import solverFactory
from . multigrid import multigrid, Complexmultigrid

solverFactory.register('mg', multigrid, isMultilevelSolver=True)
solverFactory.register('complex_mg', Complexmultigrid, isMultilevelSolver=True)
