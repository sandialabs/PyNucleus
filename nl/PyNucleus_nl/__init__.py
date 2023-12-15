###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


"""
This module allows to assemble nonlocal operators.

It defines kernel functions, fractional orders, interaction domains and normalization constants.

It allows to assemble nonlocal operators as dense, sparse of H^2 matrices.
"""

from . kernelsCy import (Kernel,
                         FractionalKernel,
                         RangedFractionalKernel,
                         getKernelEnum,
                         FRACTIONAL, INDICATOR, PERIDYNAMIC, GAUSSIAN)
from . nonlocalAssembly import nonlocalBuilder
from . clusterMethodCy import H2Matrix
from . nonlocalProblems import (fractionalLaplacianProblem,
                                nonlocalPoissonProblem,
                                transientFractionalProblem,
                                twoPointFunctionFactory,
                                fractionalOrderFactory,
                                interactionFactory,
                                kernelFactory,
                                nonlocalMeshFactory)
from . discretizedProblems import (discretizedNonlocalProblem,
                                   discretizedTransientProblem)
__all__ = ['twoPointFunctionFactory', 'fractionalOrderFactory', 'interactionFactory', 'kernelFactory', 'nonlocalMeshFactory',
           'fractionalLaplacianProblem', 'nonlocalPoissonProblem', 'transientFractionalProblem',
           'discretizedNonlocalProblem', 'discretizedTransientProblem']
