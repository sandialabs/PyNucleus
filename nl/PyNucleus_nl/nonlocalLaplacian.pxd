###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, ENCODE_t, BOOL_t
from PyNucleus_base.tupleDict cimport tupleDictMASK, indexSet, indexSetIterator, arrayIndexSet, unsortedArrayIndexSet, arrayIndexSetIterator
from PyNucleus_fem.quadrature cimport (simplexQuadratureRule, quadQuadratureRule,
                                       doubleSimplexQuadratureRule, GaussJacobi,
                                       simplexDuffyTransformation, simplexXiaoGimbutas)
from PyNucleus_fem.DoFMaps cimport DoFMap
from . clusterMethodCy cimport (tree_node,
                                farFieldClusterPair,
                                H2Matrix,
                                DistributedH2Matrix_globalData,
                                DistributedH2Matrix_localData,
                                DistributedLinearOperator)
from . nonlocalLaplacianBase cimport (double_local_matrix_t,
                                        nonlocalLaplacian,
                                        panelType,
                                        MASK_t)
from . fractionalLaplacian1D cimport (fractionalLaplacian1D_P1,
                                      fractionalLaplacian1D_P1_boundary,
                                      fractionalLaplacian1D_P0,
                                      fractionalLaplacian1D_P0_boundary)
from . fractionalLaplacian2D cimport (fractionalLaplacian2D_P1,
                                      fractionalLaplacian2D_P1_boundary,
                                      )
from . nonlocalLaplacianND cimport integrable1D, integrable2D

import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py cimport MPI
from PyNucleus_base.performanceLogger cimport PLogger, FakePLogger, LoggingPLogger
from PyNucleus_base.linear_operators cimport LinearOperator
from PyNucleus_fem.meshCy cimport meshBase
from PyNucleus_fem.DoFMaps cimport DoFMap
from . kernelsCy cimport (Kernel,
                          FractionalKernel)


include "config.pxi"


cdef class nonlocalBuilder:
    cdef:
        meshBase mesh
        public DoFMap dm
        public DoFMap dm2
        public Kernel kernel
        public double_local_matrix_t local_matrix
        public double_local_matrix_t local_matrix_zeroExterior
        public double_local_matrix_t local_matrix_surface
        bint zeroExterior
        REAL_t[::1] contrib, contribZeroExterior
        list _d2c
        public MPI.Comm comm
        public FakePLogger PLogger
        public dict params
    cdef inline double_local_matrix_t getLocalMatrix(self, dict params)
    cdef inline double_local_matrix_t getLocalMatrixBoundaryZeroExterior(self, dict params, BOOL_t infHorizon)
    cpdef REAL_t getEntry(self, INDEX_t I, INDEX_t J)
    cpdef REAL_t getEntryCluster(self, INDEX_t I, INDEX_t J)
    cpdef LinearOperator assembleClusters(self, list Pnear, bint forceUnsymmetric=*, LinearOperator Anear=*, dict jumps=*, str prefix=*, tree_node myRoot=*, BOOL_t doDistributedAssembly=*)


cdef class nearFieldClusterPair:
    cdef:
        public tree_node n1, n2
        public indexSet cellsUnion, cellsInter
    cdef void set_cells(self)


cdef LinearOperator getSparseNearField(DoFMap DoFMap, list Pnear, bint symmetric=*, tree_node myRoot=*)
