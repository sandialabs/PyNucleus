###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef class {SCALAR_label}nonlocalBuilder:
    cdef:
        meshBase mesh
        public DoFMap dm
        public DoFMap dm2
        public {SCALAR_label}Kernel kernel
        public {SCALAR_label}double_local_matrix_t local_matrix
        public {SCALAR_label}double_local_matrix_t local_matrix_zeroExterior
        public {SCALAR_label}double_local_matrix_t local_matrix_surface
        BOOL_t zeroExterior
        {SCALAR}_t[:, ::1] contrib, contribZeroExterior
        list _d2c
        public MPI.Comm comm
        public FakePLogger PLogger
        public dict params
    cdef inline {SCALAR_label}double_local_matrix_t getLocalMatrix(self, dict params)
    cdef inline {SCALAR_label}double_local_matrix_t getLocalMatrixBoundaryZeroExterior(self, {SCALAR_label}Kernel kernel, dict params)
    cpdef {SCALAR}_t getEntry(self, INDEX_t I, INDEX_t J)
    cpdef {SCALAR}_t getEntryCluster(self, INDEX_t I, INDEX_t J)
