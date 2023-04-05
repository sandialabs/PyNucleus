###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
cimport numpy as np
from libc.math cimport (sin, cos, sinh, cosh, tanh, sqrt, atan, atan2,
                        log, ceil,
                        fabs as abs, M_PI as pi, pow,
                        tgamma as gamma, exp, erf)
from PyNucleus_base.myTypes import INDEX, REAL, ENCODE, BOOL
from PyNucleus_fem.functions cimport constant
from PyNucleus_fem.meshCy cimport meshBase
from PyNucleus_fem.DoFMaps cimport DoFMap
from libc.stdlib cimport malloc
from libc.string cimport memcpy
import warnings
from . interactionDomains cimport ball1, ball2, ballInf

include "kernel_params.pxi"

cdef REAL_t inf = np.inf

######################################################################

cdef enum:
    NUM_FRAC_ORDER_PARAMS = 10


cdef enum fracOrderParams:
    fSFUN = 0
    fDIM = 2*OFFSET
    #
    fLAMBDA = 3*OFFSET
    #
    fSL = 3*OFFSET
    fSR = 4*OFFSET
    fR = 7*OFFSET
    fSLOPE = 8*OFFSET
    fFAC = 9*OFFSET
    #
    fSLL = 3*OFFSET
    fSRR = 4*OFFSET
    fSLR = 5*OFFSET
    fSRL = 6*OFFSET


cdef class fractionalOrderBase(twoPointFunction):
    def __init__(self, REAL_t smin, REAL_t smax, BOOL_t symmetric):
        super(fractionalOrderBase, self).__init__(symmetric)
        self.min = smin
        self.max = smax

    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        raise NotImplementedError()

    def __getstate__(self):
        return (self.min, self.max, self.symmetric)

    def __setstate__(self, state):
        fractionalOrderBase.__init__(self, state[0], state[1], state[2])


cdef class constFractionalOrder(fractionalOrderBase):
    def __init__(self, REAL_t s):
        super(constFractionalOrder, self).__init__(s, s, True)
        self.value = s

    def __getstate__(self):
        return self.value

    def __setstate__(self, state):
        constFractionalOrder.__init__(self, state)

    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.value

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        return self.value

    def __repr__(self):
        return '{}'.format(self.value)


cdef class variableFractionalOrder(fractionalOrderBase):
    def __init__(self, REAL_t smin, REAL_t smax, BOOL_t symmetric):
        super(variableFractionalOrder, self).__init__(smin, smax, symmetric)
        self.c_params = malloc(NUM_FRAC_ORDER_PARAMS*OFFSET)

    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            fun_t sFun = getFun(self.c_params, fSFUN)
        return sFun(&x[0], &y[0], self.c_params)

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            fun_t sFun = getFun(self.c_params, fSFUN)
        return sFun(x, y, self.c_params)

    cdef void setFractionalOrderFun(self, void* params):
        memcpy(params, self.c_params, NUM_FRAC_ORDER_PARAMS*OFFSET)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.symmetric)

    def __add__(self, variableFractionalOrder other):
        return sumFractionalOrder(self, 1., other, 1.)


cdef class extendedFunction(function):
    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x):
        pass


cdef class singleVariableUnsymmetricFractionalOrder(variableFractionalOrder):
    def __init__(self, extendedFunction sFun, REAL_t smin, REAL_t smax):
        super(singleVariableUnsymmetricFractionalOrder, self).__init__(smin, smax, False)
        self.sFun = sFun

    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.sFun.eval(x)

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        return self.sFun.evalPtr(dim, x)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.sFun)

    def __getstate__(self):
        return (self.sFun, self.min, self.max)

    def __setstate__(self, state):
        singleVariableUnsymmetricFractionalOrder.__init__(self, state[0], state[1], state[2])


cdef REAL_t lambdaFractionalOrderFun(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fDIM)
        REAL_t[::1] xA =<REAL_t[:dim]> x
        REAL_t[::1] yA =<REAL_t[:dim]> y
    return (<object>((<void**>(c_params+fLAMBDA))[0]))(xA, yA)


cdef class lambdaFractionalOrder(variableFractionalOrder):
    cdef:
        tuple fun

    def __init__(self, INDEX_t dim, REAL_t smin, REAL_t smax, BOOL_t symmetric, fun):
        super(lambdaFractionalOrder, self).__init__(smin, smax, symmetric)
        self.fun = (fun, )
        setINDEX(self.c_params, fDIM, dim)
        (<void**>(self.c_params+fLAMBDA))[0] = <void*>fun
        setFun(self.c_params, fSFUN, &lambdaFractionalOrderFun)


cdef REAL_t constFractionalOrderFun(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t s = getREAL(c_params, fSL)
    return s


cdef class variableConstFractionalOrder(variableFractionalOrder):
    cdef:
        public REAL_t value

    def __init__(self, REAL_t s):
        super(variableConstFractionalOrder, self).__init__(s, s, True)
        self.value = s
        setREAL(self.c_params, fSL, self.value)
        setFun(self.c_params, fSFUN, &constFractionalOrderFun)

    def __repr__(self):
        return '{}(s={},sym={})'.format(self.__class__.__name__, self.value, self.symmetric)

    def __getstate__(self):
        return self.value

    def __setstate__(self, state):
        variableConstFractionalOrder.__init__(self, state)


cdef REAL_t piecewiseConstantFractionalOrderFun(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fDIM)
        REAL_t[::1] xA =<REAL_t[:dim]> x
        REAL_t[::1] yA =<REAL_t[:dim]> y
        function blockIndicator = (<function>((<void**>(c_params+fLAMBDA))[0]))
        INDEX_t numBlocks = getINDEX(c_params, fSR)
        REAL_t* sVals = getREALArray2D(c_params, fR)
        INDEX_t blockX, blockY
    blockX = <INDEX_t>blockIndicator.eval(xA)
    blockY = <INDEX_t>blockIndicator.eval(yA)
    if blockX >= 0 and blockX < numBlocks and blockY >= 0 and blockY < numBlocks:
        return sVals[blockX*numBlocks+blockY]
    else:
        return -1.


cdef class piecewiseConstantFractionalOrder(variableFractionalOrder):
    def __init__(self, INDEX_t dim, function blockIndicator, REAL_t[:,::1] sVals):
        cdef:
            INDEX_t numBlocks = sVals.shape[0]
        assert sVals.shape[0] == sVals.shape[1]

        for i in range(numBlocks):
            for j in range(numBlocks):
                if i == j:
                    assert np.isfinite(sVals[i, j])
                else:
                    if not np.isfinite(sVals[i, j]):
                        sVals[i, j] = 0.5*(sVals[i, i]+sVals[j, j])
        smin = np.array(sVals, copy=False).min()
        smax = np.array(sVals, copy=False).max()
        symmetric = np.absolute(np.array(sVals, copy=False)-np.array(sVals, copy=False).T).max() < 1e-10

        self.blockIndicator = blockIndicator
        self.sVals = sVals

        super(piecewiseConstantFractionalOrder, self).__init__(smin, smax, symmetric)

        setFun(self.c_params, fSFUN, &piecewiseConstantFractionalOrderFun)
        setINDEX(self.c_params, fDIM, dim)
        setINDEX(self.c_params, fSR, numBlocks)
        setREALArray2D(self.c_params, fR, self.sVals)
        (<void**>(self.c_params+fLAMBDA))[0] = <void*>self.blockIndicator

    def __getstate__(self):
        cdef:
            INDEX_t dim = getINDEX(self.c_params, fDIM)
            function blockIndicator = (<function>((<void**>(self.c_params+fLAMBDA))[0]))
            INDEX_t numBlocks = getINDEX(self.c_params, fSR)
            REAL_t[:, ::1] sVals = <REAL_t[:numBlocks, :numBlocks]>getREALArray2D(self.c_params, fR)
        return (dim, blockIndicator, sVals)

    def __setstate__(self, state):
        piecewiseConstantFractionalOrder.__init__(self, state[0], state[1], state[2])

    def __repr__(self):
        cdef:
            INDEX_t numBlocks = getINDEX(self.c_params, fSR)
        return '{}(numBlocks={},sym={})'.format(self.__class__.__name__, numBlocks, self.symmetric)

    @property
    def numBlocks(self):
        cdef:
            INDEX_t numBlocks = getINDEX(self.c_params, fSR)
        return numBlocks


cdef REAL_t leftRightFractionalOrderFun(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        REAL_t sll, srr, slr, srl
        REAL_t interface = getREAL(c_params, fR)
    if x[0] < interface:
        if y[0] < interface:
            sll = getREAL(c_params, fSLL)
            return sll
        else:
            slr = getREAL(c_params, fSLR)
            return slr
    else:
        if y[0] < interface:
            srl = getREAL(c_params, fSRL)
            return srl
        else:
            srr = getREAL(c_params, fSRR)
            return srr


cdef class leftRightFractionalOrder(variableFractionalOrder):
    def __init__(self, REAL_t sll, REAL_t srr, REAL_t slr=np.nan, REAL_t srl=np.nan, REAL_t interface=0.):
        if not np.isfinite(slr):
            slr = 0.5*(sll+srr)
        if not np.isfinite(srl):
            srl = 0.5*(sll+srr)
        symmetric = (slr == srl)
        super(leftRightFractionalOrder, self).__init__(min([sll, srr, slr, srl]), max([sll, srr, slr, srl]), symmetric)

        setFun(self.c_params, fSFUN, &leftRightFractionalOrderFun)
        setREAL(self.c_params, fSLL, sll)
        setREAL(self.c_params, fSRR, srr)
        setREAL(self.c_params, fSLR, slr)
        setREAL(self.c_params, fSRL, srl)
        setREAL(self.c_params, fR, interface)

    def __getstate__(self):
        sll = getREAL(self.c_params, fSLL)
        srr = getREAL(self.c_params, fSRR)
        slr = getREAL(self.c_params, fSLR)
        srl = getREAL(self.c_params, fSRL)
        interface = getREAL(self.c_params, fR)
        return (sll, srr, slr, srl, interface)

    def __setstate__(self, state):
        leftRightFractionalOrder.__init__(self, state[0], state[1], state[2], state[3], state[4])

    def __repr__(self):
        sll = getREAL(self.c_params, fSLL)
        srr = getREAL(self.c_params, fSRR)
        slr = getREAL(self.c_params, fSLR)
        srl = getREAL(self.c_params, fSRL)
        interface = getREAL(self.c_params, fR)
        return '{}(ll={},rr={},lr={},rl={},interface={},sym={})'.format(self.__class__.__name__, sll, srr, slr, srl, interface, self.symmetric)


cdef class constantExtended(extendedFunction):
    cdef:
        REAL_t value

    def __init__(self, REAL_t value):
        self.value = value

    cdef REAL_t eval(self, REAL_t[::1] x):
        return self.value

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x):
        return self.value

    def __repr__(self):
        return '{}'.format(self.value)


cdef class smoothLeftRight(extendedFunction):
    cdef:
        REAL_t sl, sr, r, slope, fac

    def __init__(self, REAL_t sl, REAL_t sr, REAL_t r=0.1, REAL_t slope=200.):
        self.sl = sl
        self.sr = sr
        self.r = r
        self.slope = slope
        self.fac = 1./atan(r*slope)

    cdef REAL_t eval(self, REAL_t[::1] x):
        if x[0] < -self.r:
            return self.sl
        elif x[0] > self.r:
            return self.sr
        return 0.5*(self.sl+self.sr)+0.5*(self.sr-self.sl)*atan(x[0]*self.slope) * self.fac

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x):
        if x[0] < -self.r:
            return self.sl
        elif x[0] > self.r:
            return self.sr
        return 0.5*(self.sl+self.sr)+0.5*(self.sr-self.sl)*atan(x[0]*self.slope) * self.fac

    def __repr__(self):
        return '{}(sl={},sr={},r={},slope={})'.format(self.__class__.__name__, self.sl, self.sr, self.r, self.slope)


cdef class smoothStep(extendedFunction):
    cdef:
        REAL_t sl, sr, r, slope, interface

    def __init__(self, REAL_t sl, REAL_t sr, REAL_t r, REAL_t interface=0.):
        self.sl = sl
        self.sr = sr
        self.r = r
        self.slope = 0.5/r
        self.interface = interface

    cdef REAL_t eval(self, REAL_t[::1] x):
        if x[0] < self.interface-self.r:
            return self.sl
        elif x[0] > self.interface+self.r:
            return self.sr
        return self.sl + (self.sr-self.sl) * (3.0*pow((x[0]-self.interface)*self.slope+0.5, 2.0) - 2.0*pow((x[0]-self.interface)*self.slope+0.5, 3.0))

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x):
        if x[0] < self.interface-self.r:
            return self.sl
        elif x[0] > self.interface+self.r:
            return self.sr
        return self.sl + (self.sr-self.sl) * (3.0*pow((x[0]-self.interface)*self.slope+0.5, 2.0) - 2.0*pow((x[0]-self.interface)*self.slope+0.5, 3.0))

    def __repr__(self):
        return '{}(sl={},sr={},r={},interface={})'.format(self.__class__.__name__, self.sl, self.sr, self.r, self.interface)


cdef class linearStep(extendedFunction):
    cdef:
        REAL_t sl, sr, r, slope

    def __init__(self, REAL_t sl, REAL_t sr, REAL_t r):
        self.sl = sl
        self.sr = sr
        self.r = r
        self.slope = 0.5*(sr-sl)/r

    cdef REAL_t eval(self, REAL_t[::1] x):
        if x[0] < -self.r:
            return self.sl
        elif x[0] > self.r:
            return self.sr
        return self.sl + self.slope*(x[0]+self.r)

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x):
        if x[0] < -self.r:
            return self.sl
        elif x[0] > self.r:
            return self.sr
        return self.sl + self.slope*(x[0]+self.r)

    def __repr__(self):
        return '{}(sl={},sr={},r={})'.format(self.__class__.__name__, self.sl, self.sr, self.r)


cdef class constantNonSymFractionalOrder(singleVariableUnsymmetricFractionalOrder):
    def __init__(self, REAL_t s):
        sFun = constantExtended(s)
        super(constantNonSymFractionalOrder, self).__init__(sFun, s, s)


cdef class smoothedLeftRightFractionalOrder(singleVariableUnsymmetricFractionalOrder):
    def __init__(self, REAL_t sl, REAL_t sr, REAL_t r=0.1, REAL_t slope=200., REAL_t interface=0.):
        sFun = smoothStep(sl, sr, r, interface)
        # sFun = smoothLeftRight(sl, sr, r, slope)
        super(smoothedLeftRightFractionalOrder, self).__init__(sFun, min(sl, sr), max(sl, sr))


cdef class linearLeftRightFractionalOrder(singleVariableUnsymmetricFractionalOrder):
    def __init__(self, REAL_t sl, REAL_t sr, REAL_t r=0.1):
        sFun = linearStep(sl, sr, r)
        super(linearLeftRightFractionalOrder, self).__init__(sFun, min(sl, sr), max(sl, sr))


cdef REAL_t innerOuterFractionalOrderFun(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fDIM)
        REAL_t sii = getREAL(c_params, fSLL)
        REAL_t soo = getREAL(c_params, fSRR)
        REAL_t sio = getREAL(c_params, fSLR)
        REAL_t soi = getREAL(c_params, fSRL)
        REAL_t r2 = getREAL(c_params, fR)
        REAL_t r2x = 0., r2y = 0.
        INDEX_t i
        REAL_t[::1] center =<REAL_t[:dim]> (<REAL_t*>(c_params+fSLOPE))
    for i in range(dim):
        r2x += (x[i]-center[i])**2
        r2y += (y[i]-center[i])**2
    if r2x < r2 and r2y < r2:
        return sii
    elif r2x >= r2 and r2y >= r2:
        return soo
    elif r2x < r2 and r2y >= r2:
        return sio
    elif r2x >= r2 and r2y < r2:
        return soi
    else:
        raise NotImplementedError()


cdef class innerOuterFractionalOrder(variableFractionalOrder):
    cdef:
        REAL_t sii, soo, sio, soi
        REAL_t r2
        REAL_t[::1] center

    def __init__(self, INDEX_t dim, REAL_t sii, REAL_t soo, REAL_t r, REAL_t[::1] center, REAL_t sio=np.nan, REAL_t soi=np.nan):
        if not np.isfinite(sio):
            sio = 0.5*(sii+soo)
        if not np.isfinite(soi):
            soi = 0.5*(sii+soo)
        super(innerOuterFractionalOrder, self).__init__(min([sii, soo, sio, soi]), max([sii, soo, sio, soi]), sio == soi)
        setINDEX(self.c_params, fDIM, dim)
        setFun(self.c_params, fSFUN, &innerOuterFractionalOrderFun)
        setREAL(self.c_params, fSLL, sii)
        setREAL(self.c_params, fSRR, soo)
        setREAL(self.c_params, fSLR, sio)
        setREAL(self.c_params, fSRL, soi)
        setREAL(self.c_params, fR, r*r)
        setREAL(self.c_params, fSLOPE, center[0])

    def __getstate__(self):
        dim = getINDEX(self.c_params, fDIM)
        sii = getREAL(self.c_params, fSLL)
        soo = getREAL(self.c_params, fSRR)
        sio = getREAL(self.c_params, fSLR)
        soi = getREAL(self.c_params, fSRL)
        r = sqrt(getREAL(self.c_params, fR))
        center =<REAL_t[:dim]> (<REAL_t*>(self.c_params+fSLOPE))
        return (dim, sii, soo, r, np.array(center), sio, soi)

    def __setstate__(self, state):
        innerOuterFractionalOrder.__init__(self, state[0], state[1], state[2], state[3], state[4], state[5], state[6])

    def __repr__(self):
        return '{}(ii={},oo={},io={},oi={},r={},sym={})'.format(self.__class__.__name__, self.sii, self.soo, self.sio, self.soi, np.sqrt(self.r2), self.symmetric)


cdef class sumFractionalOrder(variableFractionalOrder):
    cdef:
        variableFractionalOrder s1, s2
        REAL_t fac1, fac2

    def __init__(self, variableFractionalOrder s1, REAL_t fac1, variableFractionalOrder s2, REAL_t fac2):
        super(sumFractionalOrder, self).__init__(min(s1.min, s2.min), max(s1.max, s2.max), s1.symmetric and s2.symmetric)
        self.s1 = s1
        self.fac1 = fac1
        self.s2 = s2
        self.fac2 = fac2

    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        return self.s1.eval(x, y) + self.s2.eval(x, y)


cdef REAL_t islandsFractionalOrderFun(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fDIM)
        REAL_t sii = getREAL(c_params, fSLL)
        REAL_t soo = getREAL(c_params, fSRR)
        REAL_t sio = getREAL(c_params, fSLR)
        REAL_t soi = getREAL(c_params, fSRL)
        REAL_t r = getREAL(c_params, fR)
        REAL_t r2 = getREAL(c_params, fSLOPE)
        REAL_t p
        INDEX_t i
        BOOL_t xInIsland = True, yInIsland = True
    for i in range(dim):
        p = abs(x[i])
        if not(p >= r and p <= r2):
            xInIsland = False
            break

    for i in range(dim):
        p = abs(y[i])
        if not(p >= r and p <= r2):
            yInIsland = False
            break

    if xInIsland:
        if yInIsland:
            return sii
        else:
            return sio
    else:
        if yInIsland:
            return soi
        else:
            return soo


cdef class islandsFractionalOrder(variableFractionalOrder):
    def __init__(self, REAL_t sii, REAL_t soo, REAL_t r, REAL_t r2, REAL_t sio=np.nan, REAL_t soi=np.nan):
        if not np.isfinite(sio):
            sio = 0.5*(sii+soo)
        if not np.isfinite(soi):
            soi = 0.5*(sii+soo)
        super(islandsFractionalOrder, self).__init__(min([sii, soo, sio, soi]), max([sii, soo, sio, soi]), sio == soi)
        setINDEX(self.c_params, fDIM, 2)
        setFun(self.c_params, fSFUN, &islandsFractionalOrderFun)
        setREAL(self.c_params, fSLL, sii)
        setREAL(self.c_params, fSRR, soo)
        setREAL(self.c_params, fSLR, sio)
        setREAL(self.c_params, fSRL, soi)
        setREAL(self.c_params, fR, r)
        setREAL(self.c_params, fSLOPE, r2)

    def __getstate__(self):
        sii = getREAL(self.c_params, fSLL)
        soo = getREAL(self.c_params, fSRR)
        sio = getREAL(self.c_params, fSLR)
        soi = getREAL(self.c_params, fSRL)
        r = getREAL(self.c_params, fR)
        r2 = getREAL(self.c_params, fSLOPE)
        return (sii, soo, r, r2, sio, soi)

    def __setstate__(self, state):
        islandsFractionalOrder.__init__(self, state[0], state[1], state[2], state[3], state[4], state[5])

    def __repr__(self):
        sii = getREAL(self.c_params, fSLL)
        soo = getREAL(self.c_params, fSRR)
        sio = getREAL(self.c_params, fSLR)
        soi = getREAL(self.c_params, fSRL)
        r = getREAL(self.c_params, fR)
        r2 = getREAL(self.c_params, fSLOPE)
        return '{}(ii={},oo={},io={},oi={},r={},r2={},sym={})'.format(self.__class__.__name__, sii, soo, sio, soi, r, r2, self.symmetric)


cdef REAL_t layersFractionalOrderFun(REAL_t *x, REAL_t *y, void *c_params):
    cdef:
        INDEX_t dim = getINDEX(c_params, fDIM)
        INDEX_t numLayers = getINDEX(c_params, fR)
        REAL_t* layerBoundaries = getREALArray1D(c_params, fSL)
        REAL_t* layerOrders = getREALArray2D(c_params, fSR)
        INDEX_t i, j, I = 0, J = 0
        REAL_t c

    c = x[dim-1]
    if c <= layerBoundaries[0]:
        I = 0
    elif c >= layerBoundaries[numLayers]:
        I = numLayers-1
    else:
        for i in range(numLayers):
            if (layerBoundaries[i] <= c) and (c <= layerBoundaries[i+1]):
                I = i
                break
    c = y[dim-1]
    if c <= layerBoundaries[0]:
        J = 0
    elif c >= layerBoundaries[numLayers]:
        J = numLayers-1
    else:
        for j in range(numLayers):
            if (layerBoundaries[j] <= c) and (c <= layerBoundaries[j+1]):
                J = j
                break
    return layerOrders[I*numLayers+J]


cdef class layersFractionalOrder(variableFractionalOrder):
    cdef:
        public REAL_t[::1] layerBoundaries
        public REAL_t[:, ::1] layerOrders

    def __init__(self, INDEX_t dim, REAL_t[::1] layerBoundaries, REAL_t[:, ::1] layerOrders):
        cdef:
            REAL_t smin, smax
            INDEX_t i, j, numLayers
            BOOL_t sym

        smin = np.inf
        smax = 0.
        sym = True
        numLayers = layerBoundaries.shape[0]-1
        assert layerOrders.shape[0] == numLayers
        assert layerOrders.shape[1] == numLayers
        for i in range(numLayers):
            for j in range(numLayers):
                smin = min(smin, layerOrders[i, j])
                smax = max(smax, layerOrders[i, j])
                if layerOrders[i, j] != layerOrders[j, i]:
                    sym = False
        super(layersFractionalOrder, self).__init__(smin, smax, sym)
        self.layerBoundaries = layerBoundaries
        self.layerOrders = layerOrders
        setINDEX(self.c_params, fDIM, dim)
        setFun(self.c_params, fSFUN, &layersFractionalOrderFun)
        setINDEX(self.c_params, fR, numLayers)
        setREALArray1D(self.c_params, fSL, self.layerBoundaries)
        setREALArray2D(self.c_params, fSR, self.layerOrders)

    def __getstate__(self):
        dim = getINDEX(self.c_params, fDIM)
        return (dim, np.array(self.layerBoundaries), np.array(self.layerOrders))

    def __setstate__(self, state):
        layersFractionalOrder.__init__(self, state[0], state[1], state[2])

    def __repr__(self):
        numLayers = getINDEX(self.c_params, fR)
        return '{}(numLayers={})'.format(self.__class__.__name__, numLayers)


######################################################################

cdef class constantFractionalLaplacianScaling(constantTwoPoint):
    def __init__(self, INDEX_t dim, REAL_t s, REAL_t horizon):
        self.dim = dim
        if 1. < s and s < 2.:
            s = s-1.
        self.s = s
        self.horizon = horizon
        if (self.horizon <= 0.) or (self.s <= 0.) or (self.s >= 1.):
            value = np.nan
        else:
            if dim == 1:
                if horizon < inf:
                    value = (2.-2*s) * pow(horizon, 2*s-2.) * 0.5
                else:
                    value = 2.0**(2.0*s) * s * gamma(s+0.5)/sqrt(pi)/gamma(1.0-s) * 0.5
            elif dim == 2:
                if horizon < inf:
                    value = (2.-2*s)*pow(horizon, 2*s-2.) * 2./pi * 0.5
                else:
                    value = 2.0**(2.0*s) * s * gamma(s+1.0)/pi/gamma(1.-s) * 0.5
            else:
                raise NotImplementedError()
        super(constantFractionalLaplacianScaling, self).__init__(value)

    def __getstate__(self):
        return (self.dim, self.s, self.horizon)

    def __setstate__(self, state):
        constantFractionalLaplacianScaling.__init__(self, state[0], state[1], state[2])

    def __repr__(self):
        return '{}({},{} -> {})'.format(self.__class__.__name__, self.s, self.horizon, self.value)


cdef class constantIntegrableScaling(constantTwoPoint):
    def __init__(self, kernelType kType, interactionDomain interaction, INDEX_t dim, REAL_t horizon):
        self.kType = kType
        self.dim = dim
        self.interaction = interaction
        self.horizon = horizon
        if self.horizon <= 0.:
            value = np.nan
        else:
            if kType == INDICATOR:
                if dim == 1:
                    value = 3./horizon**3 / 2.
                elif dim == 2:
                    if isinstance(self.interaction, ball2):
                        value = 8./pi/horizon**4 / 2.
                    elif isinstance(self.interaction, ballInf):
                        value = 3./4./horizon**4 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif kType == PERIDYNAMIC:
                if dim == 1:
                    value = 2./horizon**2 / 2.
                elif dim == 2:
                    if isinstance(self.interaction, ball2):
                        value = 6./pi/horizon**3 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            elif kType == GAUSSIAN:
                if dim == 1:
                    # value = 4.0/sqrt(pi)/(horizon/3.)**3 / 2.
                    value = 4.0/sqrt(pi)/(erf(3.0)-6.0*exp(-9.0)/sqrt(pi))/(horizon/3.0)**3 / 2.
                elif dim == 2:
                    if isinstance(self.interaction, ball2):
                        # value = 4.0/pi/(horizon/3.0)**4 / 2.
                        value = 4.0/pi/(1.0-10.0*exp(-9.0))/(horizon/3.0)**4 / 2.
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        super(constantIntegrableScaling, self).__init__(value)

    def __getstate__(self):
        return (self.kType, self.interaction, self.dim, self.horizon)

    def __setstate__(self, state):
        constantIntegrableScaling.__init__(self, state[0], state[1], state[2], state[3])

    def __repr__(self):
        return '{}({} -> {})'.format(self.__class__.__name__, self.horizon, self.value)


cdef class variableFractionalLaplacianScaling(parametrizedTwoPointFunction):
    def __init__(self, BOOL_t symmetric):
        super(variableFractionalLaplacianScaling, self).__init__(symmetric)

    cdef void setParams(self, void *params):
        parametrizedTwoPointFunction.setParams(self, params)
        self.dim = getINDEX(self.params, fKDIM)

    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            REAL_t s = getREAL(self.params, fS)
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)

        if self.dim == 1:
            if horizon2 < inf:
                return (2.-2*s) * pow(horizon2, s-1.) * 0.5
            else:
                return 2.0**(2.0*s) * s * gamma(s+0.5)/sqrt(pi)/gamma(1.0-s) * 0.5
        elif self.dim == 2:
            if horizon2 < inf:
                return (2.-2*s)*pow(horizon2, s-1.) * 2./pi * 0.5
            else:
                return 2.0**(2.0*s) * s * gamma(s+1.0)/pi/gamma(1.-s) * 0.5
        else:
            raise NotImplementedError()

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            REAL_t s = getREAL(self.params, fS)
            REAL_t horizon2 = getREAL(self.params, fHORIZON2)

        if self.dim == 1:
            if horizon2 < inf:
                return (2.-2*s) * pow(horizon2, s-1.) * 0.5
            else:
                return 2.0**(2.0*s) * s * gamma(s+0.5)/sqrt(pi)/gamma(1.0-s) * 0.5
        elif self.dim == 2:
            if horizon2 < inf:
                return (2.-2*s)*pow(horizon2, s-1.) * 2./pi * 0.5
            else:
                return 2.0**(2.0*s) * s * gamma(s+1.0)/pi/gamma(1.-s) * 0.5
        else:
            raise NotImplementedError()

    def getScalingWithDifferentHorizon(self):
        cdef:
            variableFractionalLaplacianScalingWithDifferentHorizon scaling
            function horizonFun
            BOOL_t horizonFunNull = isNull(self.params, fHORIZONFUN)
        if not horizonFunNull:
            horizonFun = <function>((<void**>(self.params+fHORIZONFUN))[0])
        else:
            horizonFun = constant(sqrt(getREAL(self.params, fHORIZON2)))
        scaling = variableFractionalLaplacianScalingWithDifferentHorizon(self.symmetric, horizonFun)
        return scaling

    def __repr__(self):
        return 'variableFractionalLaplacianScaling'


######################################################################


cdef class variableFractionalLaplacianScalingWithDifferentHorizon(variableFractionalLaplacianScaling):
    def __init__(self, BOOL_t symmetric, function horizonFun):
        super(variableFractionalLaplacianScalingWithDifferentHorizon, self).__init__(symmetric)
        self.horizonFun = horizonFun

    cdef REAL_t eval(self, REAL_t[::1] x, REAL_t[::1] y):
        cdef:
            void* params
            void* paramsModified = malloc(NUM_KERNEL_PARAMS*OFFSET)
            REAL_t horizon
        horizon = self.horizonFun.eval(x)
        params = self.getParams()
        memcpy(paramsModified, params, NUM_KERNEL_PARAMS*OFFSET)
        setREAL(paramsModified, fHORIZON2, horizon**2)
        self.setParams(paramsModified)
        scalingValue = variableFractionalLaplacianScaling.eval(self, x, y)
        self.setParams(params)
        return scalingValue

    cdef REAL_t evalPtr(self, INDEX_t dim, REAL_t* x, REAL_t* y):
        cdef:
            void* params
            void* paramsModified = malloc(NUM_KERNEL_PARAMS*OFFSET)
            REAL_t horizon
            REAL_t[::1] xA = <REAL_t[:dim]> x
        horizon = self.horizonFun.eval(xA)
        params = self.getParams()
        memcpy(paramsModified, params, NUM_KERNEL_PARAMS*OFFSET)
        setREAL(paramsModified, fHORIZON2, horizon**2)
        self.setParams(paramsModified)
        scalingValue = variableFractionalLaplacianScaling.evalPtr(self, dim, x, y)
        self.setParams(params)
        return scalingValue

    def __getstate__(self):
        return (self.symmetric, self.horizonFun)

    def __setstate__(self, state):
        variableFractionalLaplacianScalingWithDifferentHorizon.__init__(self, state[0], state[1])
