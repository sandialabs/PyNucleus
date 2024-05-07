###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cimport numpy as np
from mpi4py cimport MPI
from PyNucleus_fem.quadrature cimport (simplexDuffyTransformation,
                                       simplexQuadratureRule,
                                       simplexXiaoGimbutas)
from PyNucleus_fem.functions cimport function
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, BOOL_t
from PyNucleus_base.linear_operators cimport (LinearOperator,
                                              VectorLinearOperator,
                                              Dense_LinearOperator,
                                              CSR_LinearOperator,
                                              SSS_LinearOperator)
from PyNucleus_base.tupleDict cimport indexSet, indexSetIterator, arrayIndexSet, arrayIndexSetIterator, bitArray
from PyNucleus_base.performanceLogger cimport PLogger, FakePLogger
from PyNucleus_fem.DoFMaps cimport DoFMap
from PyNucleus_fem.meshCy cimport meshBase
from . fractionalOrders cimport fractionalOrderBase
from . kernelsCy cimport Kernel, FractionalKernel


cdef class transferMatrixBuilder:
    cdef:
        INDEX_t dim
        productIterator pit
        productIterator pit2

    cdef void build(self,
                    tree_node parent,
                    tree_node child)


cdef enum refinementType:
    GEOMETRIC
    BARYCENTER
    MEDIAN


cdef struct refinementParams:
    INDEX_t maxLevels
    INDEX_t maxLevelsMixed
    INDEX_t minSize
    INDEX_t minMixedSize
    refinementType refType
    BOOL_t splitEveryDim
    REAL_t eta
    INDEX_t farFieldInteractionSize
    INDEX_t interpolation_order
    BOOL_t attemptRefinement
    REAL_t targetOrder
    REAL_t meshDiam
    REAL_t maxSingularity


cdef class tree_node:
    cdef:
        public tree_node parent
        public list children
        public INDEX_t dim
        public INDEX_t id
        public INDEX_t _irregularLevelsOffset
        public INDEX_t distFromRoot
        public indexSet _dofs
        public indexSet _local_dofs
        public indexSet secondary_dofs
        INDEX_t _num_dofs
        public indexSet _cells
        public REAL_t[:, ::1] box
        public REAL_t[:, :, ::1] boxes
        public REAL_t[:, ::1] coords
        public REAL_t[::1] hVector
        public REAL_t[:, ::1] transferOperator
        public REAL_t[:, :, ::1] value
        public REAL_t[::1] coefficientsUp
        public REAL_t[::1] coefficientsDown
        public REAL_t[:, ::1] coefficientsDownVec
        public BOOL_t mixed_node
        public BOOL_t canBeAssembled
        public INDEX_t levelNo
        public INDEX_t coefficientsUpOffset
        public REAL_t hmin
        public INDEX_t interpolation_order
        public refinementParams refParams
    cdef void init(self, REAL_t[:, :, ::1] boxes, REAL_t[:, ::1] coords, REAL_t[::1] hVector)
    cdef void releaseData(self, BOOL_t recurse=*)
    cdef void releaseDoFs(self, BOOL_t recurse=*)
    cdef void releaseLocalDoFs(self, BOOL_t recurse=*)
    cdef void releaseCells(self, BOOL_t recurse=*)
    cdef indexSet get_dofs(self)
    cdef indexSet get_local_dofs(self)
    cdef INDEX_t get_num_dofs(self)
    cdef indexSet get_cells(self)
    cdef get_num_root_children(self)
    cdef get_num_children(self, INDEX_t levelOffset=*)
    cdef BOOL_t get_is_leaf(self)
    cdef INDEX_t _getLevels(self)
    cdef INDEX_t _getParentLevels(self)
    cdef void prepareTransferOperators(self, INDEX_t valueSize, transferMatrixBuilder tMB=*)
    cdef void upwardPass(self, REAL_t[::1] x, INDEX_t componentNo=*, BOOL_t skip_leaves=*, BOOL_t local=*)
    cdef void resetCoefficientsDown(self, BOOL_t vecValued=*)
    cdef void resetCoefficientsUp(self)
    cdef void downwardPass(self, REAL_t[::1] y, INDEX_t componentNo=*, BOOL_t local=*)
    cdef void downwardPassVec(self, REAL_t[:, ::1] y, INDEX_t componentNo=*, BOOL_t local=*)
    cpdef INDEX_t findCell(self, meshBase mesh, REAL_t[::1] vertex, REAL_t[:, ::1] simplex, REAL_t[::1] bary)
    cpdef set findCells(self, meshBase mesh, REAL_t[::1] vertex, REAL_t r, REAL_t[:, ::1] simplex)
    cdef BOOL_t trim(self, bitArray keep)
    cdef void upwardPassMatrix(self, dict coefficientsUp)
    cpdef void relabelDoFs(self, INDEX_t[::1] translate)


cdef class productIterator:
    cdef:
        INDEX_t m
        INDEX_t dim
        INDEX_t[::1] idx
    cdef void reset(self)
    cdef BOOL_t step(self)


cdef class farFieldClusterPair:
    cdef:
        public tree_node n1, n2
        public REAL_t[:, ::1] kernelInterpolant
        public REAL_t[:, :, ::1] kernelInterpolantVec
    cpdef void apply(self, REAL_t[::1] x, REAL_t[::1] y)
    cpdef void applyTrans(self, REAL_t[::1] x, REAL_t[::1] y)
    cpdef void applyVec(farFieldClusterPair self, REAL_t[::1] x, REAL_t[:, ::1] y)
    cpdef void applyVecTrans(farFieldClusterPair self, REAL_t[::1] x, REAL_t[:, ::1] y)


cdef class H2Matrix(LinearOperator):
    cdef:
        public LinearOperator Anear
        public dict Pfar
        public tree_node tree
        public FakePLogger PLogger
        public LinearOperator basis
        public BOOL_t skip_leaves_upward
        public REAL_t[::1] leafCoefficientsUp
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1
    cdef INDEX_t matvecTrans(self,
                             REAL_t[::1] x,
                             REAL_t[::1] y) except -1
    cdef INDEX_t matvec_submat(self,
                               REAL_t[::1] x,
                               REAL_t[::1] y,
                               list right_list,
                               tree_node left) except -1


cdef class VectorH2Matrix(VectorLinearOperator):
    cdef:
        public VectorLinearOperator Anear
        public dict Pfar
        public tree_node tree
        public FakePLogger PLogger
        public LinearOperator basis
        public BOOL_t skip_leaves_upward
        public REAL_t[::1] leafCoefficientsUp
    # cdef INDEX_t matvec(self,
    #                     REAL_t[::1] x,
    #                     REAL_t[:, ::1] y) except -1
    # cdef INDEX_t matvec_submat(self,
    #                            REAL_t[::1] x,
    #                            REAL_t[:, ::1] y,
    #                            list right_list,
    #                            tree_node left) except -1


cdef class DistributedH2Matrix_globalData(LinearOperator):
    cdef:
        public LinearOperator localMat
        comm
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1


cdef class DistributedLinearOperator(LinearOperator):
    cdef:
        public CSR_LinearOperator localMat
        public tree_node tree
        public list Pnear
        public MPI.Comm comm
        public DoFMap dm
        dict node_lookup
        dict lcl_node_lookup
        public tree_node lclRoot
        public DoFMap lcl_dm
        public LinearOperator lclR
        public LinearOperator lclP
        INDEX_t[::1] near_offsetReceives
        INDEX_t[::1] near_offsetSends
        INDEX_t[::1] near_remoteReceives
        INDEX_t[::1] near_remoteSends
        INDEX_t[::1] near_counterReceives
        INDEX_t[::1] near_counterSends
        REAL_t[::1] near_dataReceives
        REAL_t[::1] near_dataSends
        INDEX_t[::1] rowIdx
        INDEX_t[::1] colIdx
    cdef void setupNear(self)
    cdef void communicateNear(self, REAL_t[::1] src, REAL_t[::1] target)
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1
    cpdef tuple convert(self)


cdef class DistributedH2Matrix_localData(LinearOperator):
    cdef:
        public H2Matrix localMat
        public list Pnear
        public MPI.Comm comm
        public DoFMap dm
        dict node_lookup
        dict lcl_node_lookup
        public tree_node lclRoot
        public DoFMap lcl_dm
        public LinearOperator lclR
        public LinearOperator lclP
        INDEX_t[::1] near_offsetReceives
        INDEX_t[::1] near_offsetSends
        INDEX_t[::1] near_remoteReceives
        INDEX_t[::1] near_remoteSends
        INDEX_t[::1] near_counterReceives
        INDEX_t[::1] near_counterSends
        REAL_t[::1] near_dataReceives
        REAL_t[::1] near_dataSends
        INDEX_t[::1] far_offsetsReceives
        INDEX_t[::1] far_offsetsSends
        INDEX_t[::1] far_counterReceives
        INDEX_t[::1] far_counterSends
        INDEX_t[::1] far_remoteReceives
        INDEX_t[::1] far_remoteSends
        INDEX_t[::1] far_dataCounterReceives
        INDEX_t[::1] far_dataCounterSends
        INDEX_t[::1] far_dataOffsetReceives
        INDEX_t[::1] far_dataOffsetSends
        REAL_t[::1] far_dataReceives
        REAL_t[::1] far_dataSends
        INDEX_t[::1] rowIdx
        INDEX_t[::1] colIdx
    cdef void setupNear(self)
    cdef void communicateNear(self, REAL_t[::1] src, REAL_t[::1] target)
    cdef void setupFar(self)
    cdef void communicateFar(self)
    cdef INDEX_t matvec(self,
                        REAL_t[::1] x,
                        REAL_t[::1] y) except -1
    cpdef tuple convert(self)


cdef enum admissibilityType:
    ADMISSIBLE = 0
    INADMISSIBLE = 1
    ZERO = 2


cpdef admissibilityType queryAdmissibility(Kernel kernel,
                                           tree_node n1,
                                           tree_node n2)
