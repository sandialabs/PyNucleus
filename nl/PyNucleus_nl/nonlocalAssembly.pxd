###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, COMPLEX_t, ENCODE_t, BOOL_t
from PyNucleus_base.tupleDict cimport indexSet, indexSetIterator, arrayIndexSet, unsortedArrayIndexSet, arrayIndexSetIterator
from PyNucleus_fem.quadrature cimport (simplexQuadratureRule, quadQuadratureRule,
                                       doubleSimplexQuadratureRule, GaussJacobi,
                                       simplexDuffyTransformation, simplexXiaoGimbutas)
from PyNucleus_fem.DoFMaps cimport DoFMap
from . bitset cimport tupleDictMASK
from . clusterMethodCy cimport (tree_node,
                                farFieldClusterPair,
                                H2Matrix,
                                DistributedH2Matrix_globalData,
                                DistributedH2Matrix_localData,
                                DistributedLinearOperator,
                                VectorH2Matrix)
from . nonlocalOperator cimport (double_local_matrix_t,
                                 Complexdouble_local_matrix_t,
                                 nonlocalOperator,
                                 ComplexnonlocalOperator,
                                 panelType,
                                 MASK_t)
from . fractionalLaplacian1D cimport (fractionalLaplacian1D,
                                      fractionalLaplacian1D_nonsym,
                                      fractionalLaplacian1D_boundary,
                                      )
from . fractionalLaplacian2D cimport (fractionalLaplacian2D,
                                      fractionalLaplacian2D_nonsym,
                                      fractionalLaplacian2D_boundary,
                                      )

import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py cimport MPI
from PyNucleus_base.performanceLogger cimport PLogger, FakePLogger, LoggingPLogger
from PyNucleus_base.linear_operators cimport LinearOperator, ComplexLinearOperator
from PyNucleus_base.linear_operators cimport VectorLinearOperator, ComplexVectorLinearOperator
from PyNucleus_fem.meshCy cimport meshBase
from PyNucleus_fem.DoFMaps cimport DoFMap
from . kernelsCy cimport (Kernel,
                          ComplexKernel,
                          FractionalKernel)


include "nonlocalAssembly_decl_REAL.pxi"
include "nonlocalAssembly_decl_COMPLEX.pxi"


cdef class nearFieldClusterPair:
    cdef:
        public tree_node n1, n2
        public indexSet cellsUnion, cellsInter
    cdef void set_cells(self)
    cdef void releaseCells(self)
