###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

cdef enum:
    OFFSET = sizeof(void*)

cdef enum:
    NUM_KERNEL_PARAMS = 12

cdef enum kernelParams:
    fS = 0*OFFSET
    fSINGULARITY = 1*OFFSET
    fHORIZON2 = 2*OFFSET
    fSCALING = 3*OFFSET
    fKDIM = 4*OFFSET
    fORDERFUN = 5*OFFSET
    fHORIZONFUN = 6*OFFSET
    fSCALINGFUN = 7*OFFSET
    fEVAL = 8*OFFSET
    fINTERACTION = 9*OFFSET
    fEXPONENTINVERSE = 10*OFFSET
    fTEMPERED=10*OFFSET
    fGREENS_LAMBDA=10*OFFSET
    fBOUNDARY=11*OFFSET


cdef inline BOOL_t isNull(void *c_params, size_t pos):
    return (<void**>((<char*>c_params)+pos))[0] == NULL

cdef inline BOOL_t getBOOL(void *c_params, size_t pos):
    return (<BOOL_t*>((<char*>c_params)+pos))[0]

cdef inline void setBOOL(void *c_params, size_t pos, BOOL_t val):
    (<BOOL_t*>((<char*>c_params)+pos))[0] = val

cdef inline INDEX_t getINDEX(void *c_params, size_t pos):
    return (<INDEX_t*>((<char*>c_params)+pos))[0]

cdef inline void setINDEX(void *c_params, size_t pos, INDEX_t val):
    (<INDEX_t*>((<char*>c_params)+pos))[0] = val

cdef inline REAL_t getREAL(void *c_params, size_t pos):
    return (<REAL_t*>((<char*>c_params)+pos))[0]

cdef inline void setREAL(void *c_params, size_t pos, REAL_t val):
    (<REAL_t*>((<char*>c_params)+pos))[0] = val

cdef inline COMPLEX_t getCOMPLEX(void *c_params, size_t pos):
    return (<COMPLEX_t*>((<char*>c_params)+pos))[0]

cdef inline void setCOMPLEX(void *c_params, size_t pos, COMPLEX_t val):
    (<COMPLEX_t*>((<char*>c_params)+pos))[0] = val

ctypedef REAL_t (*fun_t)(REAL_t *x, REAL_t *y, void *c_params)

cdef inline void setFun(void *c_params, size_t pos, fun_t val):
    (<fun_t*>((<char*>c_params)+pos))[0] = val

cdef inline fun_t getFun(void *c_params, size_t pos):
    return (<fun_t*>((<char*>c_params)+pos))[0]

ctypedef COMPLEX_t (*complex_fun_t)(REAL_t *x, REAL_t *y, void *c_params)

cdef inline void setComplexFun(void *c_params, size_t pos, complex_fun_t val):
    (<complex_fun_t*>((<char*>c_params)+pos))[0] = val

cdef inline complex_fun_t getComplexFun(void *c_params, size_t pos):
    return (<complex_fun_t*>((<char*>c_params)+pos))[0]

cdef inline REAL_t* getREALArray1D(void *c_params, size_t pos):
    return (<REAL_t**>((<char*>c_params)+pos))[0]

cdef inline void setREALArray1D(void *c_params, size_t pos, REAL_t[::1] val):
    (<REAL_t**>((<char*>c_params)+pos))[0] = &val[0]

cdef inline REAL_t* getREALArray2D(void *c_params, size_t pos):
    return (<REAL_t**>((<char*>c_params)+pos))[0]

cdef inline void setREALArray2D(void *c_params, size_t pos, REAL_t[:, ::1] val):
    (<REAL_t**>((<char*>c_params)+pos))[0] = &val[0, 0]


cpdef enum:
    FRACTIONAL = 0
    INDICATOR = 1
    PERIDYNAMIC = 2
    GAUSSIAN = 3
    LOGINVERSEDISTANCE = 4
    MONOMIAL = 5
    GREENS_2D = 6
    GREENS_3D = 7
    EXPONENTIAL = 8
    POLYNOMIAL = 9
    MANIFOLD_FRACTIONAL = 10
