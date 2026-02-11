###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

ctypedef {SCALAR}_t (*{SCALAR_label_lc_}kernel_fun_t)(REAL_t *x, REAL_t *y, void* user_data)


cdef class {SCALAR_label}Kernel({SCALAR_label}twoPointFunction):
    cdef:
        public INDEX_t dim
        public INDEX_t manifold_dim
        public kernelType kernelType
        public REAL_t min_singularity
        public REAL_t max_singularity
        public REAL_t min_log_singularity
        public REAL_t max_log_singularity
        public REAL_t max_horizon
        public function horizon
        public interactionDomain interaction
        public twoPointFunction scalingPrePhi
        public twoPointFunction scaling
        public twoPointFunction phi
        public BOOL_t variableSingularity
        public BOOL_t variableHorizon
        public BOOL_t finiteHorizon
        public BOOL_t complement
        public BOOL_t variableScaling
        public BOOL_t variable
        public BOOL_t piecewise
        {SCALAR_label_lc_}kernel_fun_t kernelFun
        void *c_kernel_params
        REAL_t[::1] n
    cdef setKernelFun(self, {SCALAR_label_lc_}kernel_fun_t kernelFun)
    cdef BOOL_t getBoundary(self)
    cdef void setBoundary(self, BOOL_t boundary)
    cdef REAL_t getSingularityValue(self)
    cdef void setSingularityValue(self, REAL_t log_singularity)
    cdef REAL_t getLogSingularityValue(self)
    cdef void setLogSingularityValue(self, REAL_t log_singularity)
    cdef REAL_t getHorizonValue(self)
    cdef void setHorizonValue(self, REAL_t horizon)
    cdef REAL_t getHorizonValue2(self)
    cdef REAL_t getScalingValue(self)
    cdef void setScalingValue(self, REAL_t scaling)
    cdef void evalParamsOnSimplicesPtr(self, INDEX_t dim, REAL_t* center1, REAL_t* center2, REAL_t* simplex1, REAL_t* simplex2)
    cdef void updateParams(self, INDEX_t dim, REAL_t* x, REAL_t* y)
    cdef void evalParamsPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y)
    cdef void evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y, {SCALAR}_t* value)
    cdef void setSimplices(self, REAL_t[:, ::1] simplex1, REAL_t[:, ::1] simplex2)


cdef class {SCALAR_label}BoundaryKernel({SCALAR_label}Kernel):
    cdef:
        public BOOL_t isSphericalManifold


cdef class {SCALAR_label}ErrorKernel({SCALAR_label}Kernel):
    pass


cdef class {SCALAR_label}MultiSingularityKernel({SCALAR_label}twoPointFunction):
    cdef:
        public list kernels
        {SCALAR}_t[::1] tempVec
