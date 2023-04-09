###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from libc.math cimport (sqrt, log, ceil, fabs as abs, M_PI as pi, pow)
import numpy as np
cimport numpy as np

from PyNucleus_base.myTypes import INDEX, REAL
from PyNucleus_base import uninitialized, uninitialized_like
from PyNucleus_fem.meshCy cimport meshBase
from PyNucleus_fem.quadrature cimport (simplexQuadratureRule,
                             doubleSimplexQuadratureRule,
                             GaussJacobi,
                             simplexXiaoGimbutas)
from PyNucleus_fem.DoFMaps cimport DoFMap, P1_DoFMap, P2_DoFMap, P0_DoFMap
from PyNucleus_fem.functions cimport function
from . nonlocalLaplacianBase import ALL
from scipy.special import gamma
from scipy.integrate import nquad
from cpython.pycapsule cimport PyCapsule_New
from scipy import LowLevelCallable
from libc.stdlib cimport malloc
include "panelTypes.pxi"
include "kernel_params.pxi"

cdef enum:
    INTEGRAND_OFFSET = sizeof(void*)

cdef enum:
    NUM_INTEGRAND_PARAMS = 9


cdef enum packType:
    fDOF1
    fDOF2 = INTEGRAND_OFFSET
    fNR1 = 2*INTEGRAND_OFFSET
    fNC1 = 3*INTEGRAND_OFFSET
    fNR2 = 4*INTEGRAND_OFFSET
    fNC2 = 5*INTEGRAND_OFFSET
    fSIMPLEX1 = 6*INTEGRAND_OFFSET
    fSIMPLEX2 = 7*INTEGRAND_OFFSET
    fKERNEL = 8*INTEGRAND_OFFSET


cdef inline Kernel getKernel(void *c_params, size_t pos):
    return <Kernel>((<void**>(c_params+pos))[0])

cdef inline void setKernel(void *c_params, size_t pos, Kernel kernel):
    (<void**>(c_params+pos))[0] = <void*>kernel


cdef REAL_t symIntegrandId1D(int n, REAL_t *xx, void *c_params):
    cdef:
        REAL_t* lx = xx
        REAL_t* ly = xx+n//2
        REAL_t l1x = lx[0]
        REAL_t l1y = ly[0]
        REAL_t l0x = 1.-l1x
        REAL_t l0y = 1.-l1y
        REAL_t x
        REAL_t y
        INDEX_t i = getINDEX(c_params, fDOF1)
        INDEX_t j = getINDEX(c_params, fDOF2)
        REAL_t *simplex1 = getREALArray2D(c_params, fSIMPLEX2)
        # REAL_t* simplex2 = getREALArray2D(c_params, fSIMPLEX2)
        REAL_t psi1, psi2
        Kernel kernel = getKernel(c_params, fKERNEL)

    if i == 0:
        psi1 = l0x-l0y
    else:
        psi1 = l1x-l1y

    if j == 0:
        psi2 = l0x-l0y
    else:
        psi2 = l1x-l1y

    x = l0x*simplex1[0]+l1x*simplex1[1]
    y = l0y*simplex1[0]+l1y*simplex1[1]

    return psi1 * psi2 * kernel.evalPtr(1, &x, &y)


cdef REAL_t symIntegrandVertex1D(int n, REAL_t *xx, void *c_params):
    cdef:
        REAL_t l1x = xx[0]
        REAL_t l1y = xx[n//2]
        REAL_t l0x = 1.-l1x
        REAL_t l0y = 1.-l1y
        REAL_t x
        REAL_t y
        INDEX_t i = getINDEX(c_params, fDOF1)
        INDEX_t j = getINDEX(c_params, fDOF2)
        # INDEX_t nr1 = getINDEX(c_params, fNR1)
        # INDEX_t nc1 = getINDEX(c_params, fNC1)
        # INDEX_t nr2 = getINDEX(c_params, fNR2)
        # INDEX_t nc2 = getINDEX(c_params, fNC2)
        REAL_t* simplex1 = getREALArray2D(c_params, fSIMPLEX1)
        REAL_t* simplex2 = getREALArray2D(c_params, fSIMPLEX2)
        REAL_t psi1, psi2
        Kernel kernel = getKernel(c_params, fKERNEL)

    if i == 0:
        psi1 = l0x
    elif (i == 1) or (i == 2):
        psi1 = l1x-l0y
    else:
        psi1 = -l1y

    if j == 0:
        psi2 = l0x
    elif (j == 1) or (j == 2):
        psi2 = l1x-l0y
    else:
        psi2 = -l1y

    x = l0x*simplex1[0]+l1x*simplex1[1]
    y = l0y*simplex2[0]+l1y*simplex2[1]

    return psi1 * psi2 * kernel.evalPtr(1, &x, &y)


cdef REAL_t symIntegrandDistant1D(int n, REAL_t *xx, void *c_params):
    cdef:
        REAL_t l1x = xx[0]
        REAL_t l1y = xx[n//2]
        REAL_t l0x = 1.-l1x
        REAL_t l0y = 1.-l1y
        REAL_t x
        REAL_t y
        INDEX_t i = getINDEX(c_params, fDOF1)
        INDEX_t j = getINDEX(c_params, fDOF2)
        # INDEX_t nr1 = getINDEX(c_params, fNR1)
        # INDEX_t nc1 = getINDEX(c_params, fNC1)
        # INDEX_t nr2 = getINDEX(c_params, fNR2)
        # INDEX_t nc2 = getINDEX(c_params, fNC2)
        REAL_t* simplex1 = getREALArray2D(c_params, fSIMPLEX1)
        REAL_t* simplex2 = getREALArray2D(c_params, fSIMPLEX2)
        REAL_t psi1, psi2
        Kernel kernel = getKernel(c_params, fKERNEL)

    if i == 0:
        psi1 = l0x
    elif i == 1:
        psi1 = l1x
    elif i == 2:
        psi1 = -l0y
    else:
        psi1 = -l1y

    if j == 0:
        psi2 = l0x
    elif j == 1:
        psi2 = l1x
    elif j == 2:
        psi2 = -l0y
    else:
        psi2 = -l1y

    x = l0x*simplex1[0]+l1x*simplex1[1]
    y = l0y*simplex2[0]+l1y*simplex2[1]

    return psi1 * psi2 * kernel.evalPtr(1, &x, &y)


cdef class fractionalLaplacian1D_P1_automaticQuadrature(nonlocalLaplacian1D):
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 num_dofs=None,
                 abstol=1e-4,
                 reltol=1e-4,
                 target_order=None,
                 **kwargs):
        assert isinstance(DoFMap, P1_DoFMap)
        super(fractionalLaplacian1D_P1_automaticQuadrature, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        if target_order is None:
            if isinstance(self.kernel, FractionalKernel):
                smin, smax = self.kernel.s.min, self.kernel.s.max
                # this is the desired local quadrature error
                target_order = 2.-smin
            else:
                target_order = 5
        self.target_order = target_order

        self.user_ptr = malloc(NUM_INTEGRAND_PARAMS*INTEGRAND_OFFSET)
        setINDEX(self.user_ptr, fNR1, 2)
        setINDEX(self.user_ptr, fNC1, 1)
        setINDEX(self.user_ptr, fNR2, 2)
        setINDEX(self.user_ptr, fNC2, 1)
        c_params = PyCapsule_New(self.user_ptr, NULL, NULL)
        func_type = b"double (int, double *, void *)"
        func_capsule_id = PyCapsule_New(<void*>symIntegrandId1D, func_type, NULL)
        func_capsule_vertex = PyCapsule_New(<void*>symIntegrandVertex1D, func_type, NULL)
        func_capsule_distant = PyCapsule_New(<void*>symIntegrandDistant1D, func_type, NULL)
        self.integrandId = LowLevelCallable(func_capsule_id, c_params, func_type)
        self.integrandVertex = LowLevelCallable(func_capsule_vertex, c_params, func_type)
        self.integrandDistant = LowLevelCallable(func_capsule_distant, c_params, func_type)
        self.abstol = abstol
        self.reltol = reltol
        setKernel(self.user_ptr, fKERNEL, self.kernel)

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        return DISTANT

    cdef void getNearQuadRule(self, panelType panel):
        pass

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J, t = 0
            REAL_t val, err, vol1 = self.vol1, vol2 = self.vol2
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            REAL_t horizon = self.kernel.getHorizonValue()

        setREALArray2D(self.user_ptr, fSIMPLEX1, simplex1)
        setREALArray2D(self.user_ptr, fSIMPLEX2, simplex2)

        contrib[:] = 0.

        if panel == COMMON_EDGE:
            for I in range(2):
                for J in range(I, 2):
                    k = 4*I-(I*(I+1) >> 1) + J
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, I)
                        setINDEX(self.user_ptr, fDOF2, J)
                        val, err = nquad(self.integrandId,
                                         (lambda y: (0., y),
                                          (0., 1.)),
                                         opts=[lambda y: {'points': [y], 'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol}])
                        contrib[k] = val
                        val, err = nquad(self.integrandId,
                                         (lambda y: (y, 1.),
                                          (0., 1.)),
                                         opts=[lambda y: {'points': [y], 'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol}])
                        contrib[k] += val
                        contrib[k] *= vol1*vol1
        elif panel == COMMON_VERTEX:
            for i in range(2):
                for j in range(2):
                    if simplex1[i, 0] == simplex2[j, 0]:
                        if (i == 1) and (j == 0):
                            t = 2
                            break
                        elif (i == 0) and (j == 1):
                            t = 3
                            break
                        else:
                            raise IndexError()

            # loop over all local DoFs
            for I in range(3):
                for J in range(I, 3):
                    i = 3*(I//t)+(I%t)
                    j = 3*(J//t)+(J%t)
                    if j < i:
                        i, j = j, i
                    k = 4*i-(i*(i+1) >> 1) + j
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, i)
                        setINDEX(self.user_ptr, fDOF2, j)
                        val, err = nquad(self.integrandVertex,
                                         ((0., 1.), (0., 1.)),
                                         opts={'epsabs': self.abstol, 'epsrel': self.reltol})
                        contrib[k] = val*vol1*vol2
        elif panel == DISTANT:
            k = 0
            for I in range(4):
                for J in range(I, 4):
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, I)
                        setINDEX(self.user_ptr, fDOF2, J)
                        val, err = nquad(self.integrandDistant,
                                         ((0., 1.), (0., 1.)),
                                         opts=[lambda y: {'points': [((1-y)*simplex2[0, 0]+y*simplex2[1, 0]-horizon-simplex1[0, 0])/(simplex1[1, 0]-simplex1[0, 0]),
                                                                     ((1-y)*simplex2[0, 0]+y*simplex2[1, 0]+horizon-simplex1[0, 0])/(simplex1[1, 0]-simplex1[0, 0])],
                                                          'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol}])
                        contrib[k] = val*vol1*vol2
                    k += 1
        else:
            print(np.array(simplex1), np.array(simplex2))
            raise NotImplementedError('Unknown panel type: {}'.format(panel))


cdef REAL_t nonsymIntegrandId(int n, REAL_t *xx, void *c_params):
    cdef:
        REAL_t l1x = xx[0]
        REAL_t l1y = xx[1]
        REAL_t l0x = 1.-l1x
        REAL_t l0y = 1.-l1y
        REAL_t x
        REAL_t y
        INDEX_t i = getINDEX(c_params, fDOF1)
        INDEX_t j = getINDEX(c_params, fDOF2)
        # INDEX_t nr1 = getINDEX(c_params, fNR1)
        # INDEX_t nc1 = getINDEX(c_params, fNC1)
        # INDEX_t nr2 = getINDEX(c_params, fNR2)
        # INDEX_t nc2 = getINDEX(c_params, fNC2)
        REAL_t* simplex1 = getREALArray2D(c_params, fSIMPLEX1)
        # REAL_t* simplex2 = getREALArray2D(c_params, fSIMPLEX2)
        REAL_t phi1x, phi1y
        REAL_t psi2
        Kernel kernel = getKernel(c_params, fKERNEL)

    if i == 0:
        phi1x = l0x
        phi1y = l0y
    else:
        phi1x = l1x
        phi1y = l1y

    if j == 0:
        psi2 = l0x-l0y
    else:
        psi2 = l1x-l1y

    x = l0x*simplex1[0]+l1x*simplex1[1]
    y = l0y*simplex1[0]+l1y*simplex1[1]

    return (phi1x * kernel.evalPtr(1, &x, &y) - phi1y * kernel.evalPtr(1, &y, &x)) * psi2


cdef REAL_t nonsymIntegrandVertex1(int n, REAL_t *xx, void *c_params):
    cdef:
        REAL_t l1x = xx[0]
        REAL_t l1y = xx[1]
        REAL_t l0x = 1.-l1x
        REAL_t l0y = 1.-l1y
        REAL_t x
        REAL_t y
        INDEX_t i = getINDEX(c_params, fDOF1)
        INDEX_t j = getINDEX(c_params, fDOF2)
        # INDEX_t nr1 = getINDEX(c_params, fNR1)
        # INDEX_t nc1 = getINDEX(c_params, fNC1)
        # INDEX_t nr2 = getINDEX(c_params, fNR2)
        # INDEX_t nc2 = getINDEX(c_params, fNC2)
        REAL_t* simplex1 = getREALArray2D(c_params, fSIMPLEX1)
        REAL_t* simplex2 = getREALArray2D(c_params, fSIMPLEX2)
        REAL_t phi1x, phi1y
        REAL_t psi2
        Kernel kernel = getKernel(c_params, fKERNEL)

    if i == 0:
        phi1x = l0x
        phi1y = 0.
    elif (i == 1) or (i == 2):
        phi1x = l1x
        phi1y = l0y
    else:
        phi1x = 0.
        phi1y = l1y

    if j == 0:
        psi2 = l0x
    elif (j == 1) or (j == 2):
        psi2 = l1x - l0y
    else:
        psi2 = -l1y

    x = l0x*simplex1[0]+l1x*simplex1[1]
    y = l0y*simplex2[0]+l1y*simplex2[1]

    return (phi1x * kernel.evalPtr(1, &x, &y) - phi1y * kernel.evalPtr(1, &y, &x)) * psi2


cdef REAL_t nonsymIntegrandVertex2(int n, REAL_t *xx, void *c_params):
    assert n == 2
    cdef:
        REAL_t l1x = xx[0]
        REAL_t l1y = xx[1]
        REAL_t l0x = 1.-l1x
        REAL_t l0y = 1.-l1y
        REAL_t x
        REAL_t y
        INDEX_t i = getINDEX(c_params, fDOF1)
        INDEX_t j = getINDEX(c_params, fDOF2)
        # INDEX_t nr1 = getINDEX(c_params, fNR1)
        # INDEX_t nc1 = getINDEX(c_params, fNC1)
        # INDEX_t nr2 = getINDEX(c_params, fNR2)
        # INDEX_t nc2 = getINDEX(c_params, fNC2)
        REAL_t* simplex1 = getREALArray2D(c_params, fSIMPLEX1)
        REAL_t* simplex2 = getREALArray2D(c_params, fSIMPLEX2)
        REAL_t phi1x, phi1y
        REAL_t psi2
        Kernel kernel = getKernel(c_params, fKERNEL)

    if (i == 0) or (i == 3):
        phi1x = l0x
        phi1y = l1y
    elif i == 1:
        phi1x = l1x
        phi1y = 0.
    else:
        phi1x = 0.
        phi1y = l0y

    if (j == 0) or (j == 3):
        psi2 = l0x-l1y
    elif j == 1:
        psi2 = l1x
    else:
        psi2 = -l0y

    x = l0x*simplex1[0]+l1x*simplex1[1]
    y = l0y*simplex2[0]+l1y*simplex2[1]

    return (phi1x * kernel.evalPtr(1, &x, &y) - phi1y * kernel.evalPtr(1, &y, &x)) * psi2


cdef REAL_t nonsymIntegrandDistant(int n, REAL_t *xx, void *c_params):
    cdef:
        REAL_t l1x = xx[0]
        REAL_t l1y = xx[1]
        REAL_t l0x = 1.-l1x
        REAL_t l0y = 1.-l1y
        REAL_t x
        REAL_t y
        INDEX_t i = getINDEX(c_params, fDOF1)
        INDEX_t j = getINDEX(c_params, fDOF2)
        # INDEX_t nr1 = getINDEX(c_params, fNR1)
        # INDEX_t nc1 = getINDEX(c_params, fNC1)
        # INDEX_t nr2 = getINDEX(c_params, fNR2)
        # INDEX_t nc2 = getINDEX(c_params, fNC2)
        REAL_t* simplex1 = getREALArray2D(c_params, fSIMPLEX1)
        REAL_t* simplex2 = getREALArray2D(c_params, fSIMPLEX2)
        REAL_t phi1x, phi1y
        REAL_t psi2
        Kernel kernel = getKernel(c_params, fKERNEL)

    if i == 0:
        phi1x = l0x
        phi1y = 0.
    elif i == 1:
        phi1x = l1x
        phi1y = 0.
    elif i == 2:
        phi1x = 0.
        phi1y = l0y
    else:
        phi1x = 0.
        phi1y = l1y

    if j == 0:
        psi2 = l0x
    elif j == 1:
        psi2 = l1x
    elif j == 2:
        psi2 = -l0y
    else:
        psi2 = -l1y

    x = l0x*simplex1[0]+l1x*simplex1[1]
    y = l0y*simplex2[0]+l1y*simplex2[1]

    return (phi1x * kernel.evalPtr(1, &x, &y) - phi1y * kernel.evalPtr(1, &y, &x)) * psi2


cdef class fractionalLaplacian1D_P1_nonsymAutomaticQuadrature(nonlocalLaplacian1D):
    """
    This implements the operator

    \int_{R} (u(x)-u(y)) * k(x,y)

    for unsymmetric k(x,y).

    The adjoint of this operator is

    \int_{R} (u(x) * k(x,y) - u(y) * k(y,x))

    """

    def __init__(self,
                 FractionalKernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 num_dofs=None,
                 abstol=1e-4,
                 reltol=1e-4,
                 target_order=None,
                 **kwargs):
        assert isinstance(DoFMap, P1_DoFMap)
        super(fractionalLaplacian1D_P1_nonsymAutomaticQuadrature, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        smin, smax = self.kernel.s.min, self.kernel.s.max
        if target_order is None:
            # this is the desired local quadrature error
            target_order = 2.-smin
        self.target_order = target_order

        self.user_ptr = malloc(NUM_INTEGRAND_PARAMS*INTEGRAND_OFFSET)
        setINDEX(self.user_ptr, fNR1, 2)
        setINDEX(self.user_ptr, fNC1, 1)
        setINDEX(self.user_ptr, fNR2, 2)
        setINDEX(self.user_ptr, fNC2, 1)
        c_params = PyCapsule_New(self.user_ptr, NULL, NULL)
        func_type = b"double (int, double *, void *)"
        func_capsule_id = PyCapsule_New(<void*>nonsymIntegrandId, func_type, NULL)
        func_capsule_vertex1 = PyCapsule_New(<void*>nonsymIntegrandVertex1, func_type, NULL)
        func_capsule_vertex2 = PyCapsule_New(<void*>nonsymIntegrandVertex2, func_type, NULL)
        func_capsule_distant = PyCapsule_New(<void*>nonsymIntegrandDistant, func_type, NULL)
        self.integrandId = LowLevelCallable(func_capsule_id, c_params, func_type)
        self.integrandVertex1 = LowLevelCallable(func_capsule_vertex1, c_params, func_type)
        self.integrandVertex2 = LowLevelCallable(func_capsule_vertex2, c_params, func_type)
        self.integrandDistant = LowLevelCallable(func_capsule_distant, c_params, func_type)
        self.abstol = abstol
        self.reltol = reltol
        self.symmetricCells = False
        self.symmetricLocalMatrix = False
        setKernel(self.user_ptr, fKERNEL, self.kernel)

        self.x = uninitialized((0, self.dim), dtype=REAL)
        self.y = uninitialized((0, self.dim), dtype=REAL)
        self.temp = uninitialized((0), dtype=REAL)
        self.temp2 = uninitialized_like(self.temp)
        self.temp3 = uninitialized_like(self.temp)
        self.distantPSI = {}
        self.distantPHIx = {}
        self.distantPHIy = {}

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        cdef:
            panelType panel, panel2
            REAL_t logdh1 = log(d/h1), logdh2 = log(d/h2)
            REAL_t s = self.kernel.sValue
        if d < 0.05:
            return DISTANT
        else:
            panel = <panelType>max(ceil(((self.target_order+2.)*log(self.num_dofs*self.H0) + (2.*s-1.)*abs(log(h2/self.H0)) - 2.*s*logdh2) /
                                        (max(logdh1, 0) + 0.8)),
                                   2)
            panel2 = <panelType>max(ceil(((self.target_order+2.)*log(self.num_dofs*self.H0) + (2.*s-1.)*abs(log(h1/self.H0)) - 2.*s*logdh1) /
                                         (max(logdh2, 0) + 0.8)),
                                    2)
            panel = max(panel, panel2)
            try:
                self.distantQuadRules[panel]
            except KeyError:
                self.addQuadRule(panel)
            return panel

    cdef void addQuadRule(self, panelType panel):
        cdef:
            simplexQuadratureRule qr
            doubleSimplexQuadratureRule qr2
            REAL_t[:, ::1] PSI
            INDEX_t I, k, i, j
        qr = simplexXiaoGimbutas(panel, self.dim)
        qr2 = doubleSimplexQuadratureRule(qr, qr)
        self.distantQuadRules[panel] = qr2
        PHIx = np.zeros((2*self.DoFMap.dofs_per_element,
                         qr2.num_nodes), dtype=REAL)
        PHIy = np.zeros((2*self.DoFMap.dofs_per_element,
                         qr2.num_nodes), dtype=REAL)
        PSI = uninitialized((2*self.DoFMap.dofs_per_element,
                         qr2.num_nodes), dtype=REAL)
        # phi_i(x) - phi_i(y) = phi_i(x) for i = 0,1
        for I in range(self.DoFMap.dofs_per_element):
            k = 0
            for i in range(qr2.rule1.num_nodes):
                for j in range(qr2.rule2.num_nodes):
                    PSI[I, k] = self.getLocalShapeFunction(I)(qr2.rule1.nodes[:, i])
                    PHIx[I, k] = self.getLocalShapeFunction(I)(qr2.rule1.nodes[:, i])
                    k += 1
        # phi_i(x) - phi_i(y) = -phi_i(y) for i = 2,3
        for I in range(self.DoFMap.dofs_per_element):
            k = 0
            for i in range(qr2.rule1.num_nodes):
                for j in range(qr2.rule2.num_nodes):
                    PSI[I+self.DoFMap.dofs_per_element, k] = -self.getLocalShapeFunction(I)(qr2.rule2.nodes[:, j])
                    PHIy[I+self.DoFMap.dofs_per_element, k] = self.getLocalShapeFunction(I)(qr2.rule2.nodes[:, j])
                    k += 1
        self.distantPSI[panel] = PSI
        self.distantPHIx[panel] = PHIx
        self.distantPHIy[panel] = PHIy

        if qr2.rule1.num_nodes > self.x.shape[0]:
            self.x = uninitialized((qr2.rule1.num_nodes, self.dim), dtype=REAL)
        if qr2.rule2.num_nodes > self.y.shape[0]:
            self.y = uninitialized((qr2.rule2.num_nodes, self.dim), dtype=REAL)
        if qr2.num_nodes > self.temp.shape[0]:
            self.temp = uninitialized((qr2.num_nodes), dtype=REAL)
            self.temp2 = uninitialized_like(self.temp)
            self.temp3 = uninitialized_like(self.temp)

    cdef void getNearQuadRule(self, panelType panel):
        pass

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J, t
            REAL_t val, vol1 = self.vol1, vol2 = self.vol2
            set K1, K2
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            REAL_t[:, ::1] PSI
            doubleSimplexQuadratureRule qr2
            REAL_t horizon = self.kernel.getHorizonValue()

        setREALArray2D(self.user_ptr, fSIMPLEX1, simplex1)
        setREALArray2D(self.user_ptr, fSIMPLEX2, simplex2)

        contrib[:] = 0.

        if panel == COMMON_EDGE:
            for I in range(2):
                for J in range(2):
                    # k = 4*I-(I*(I+1) >> 1) + J
                    k = 4*I+J
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, I)
                        setINDEX(self.user_ptr, fDOF2, J)
                        val, err = nquad(self.integrandId,
                                         (lambda y: (0., y),
                                          (0., 1.)),
                                         opts=[lambda y: {'points': [y], 'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol}])
                        contrib[k] = val
                        val, err = nquad(self.integrandId,
                                         (lambda y: (y, 1.),
                                          (0., 1.)),
                                         opts=[lambda y: {'points': [y], 'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol}])
                        contrib[k] += val
                        contrib[k] *= vol1*vol1
        elif panel == COMMON_VERTEX:
            K1 = set()
            K2 = set()
            for i in range(2):
                for j in range(2):
                    if simplex1[i, 0] == simplex2[j, 0]:
                        K1.add(i)
                        K2.add(j)
            if K1 == set([1]) and K2 == set([0]):
                t = 2
            elif K1 == set([0]) and K2 == set([1]):
                t = 3
            else:
                raise IndexError()

            # loop over all local DoFs
            for I in range(3):
                if (I == 2) and (t == 2):
                    I = 3
                for J in range(3):
                    if (J == 2) and (t == 2):
                        J = 3
                    k = 4*I+J
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, I)
                        setINDEX(self.user_ptr, fDOF2, J)
                        if t == 2:
                            val, err = nquad(self.integrandVertex1,
                                             ((0., 1.), (0., 1.)),
                                             opts={'epsabs': self.abstol, 'epsrel': self.reltol})
                        else:
                            val, err = nquad(self.integrandVertex2,
                                             ((0., 1.), (0., 1.)),
                                             opts={'epsabs': self.abstol, 'epsrel': self.reltol})
                        contrib[k] = val*vol1*vol2
        elif panel == DISTANT:
            vol = vol1*vol2
            k = 0
            for I in range(4):
                for J in range(4):
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, I)
                        setINDEX(self.user_ptr, fDOF2, J)
                        val, err = nquad(self.integrandDistant,
                                         ((0., 1.), (0., 1.)),
                                         opts=[lambda y: {'points': [((1-y)*simplex2[0, 0]+y*simplex2[1, 0]-horizon-simplex1[0, 0])/(simplex1[1, 0]-simplex1[0, 0]),
                                                                     ((1-y)*simplex2[0, 0]+y*simplex2[1, 0]+horizon-simplex1[0, 0])/(simplex1[1, 0]-simplex1[0, 0])],
                                                          'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol}])
                        contrib[k] = val*vol
                    k += 1
        elif panel >= 1:
            qr2 = self.distantQuadRules[panel]
            PSI = self.distantPSI[panel]
            PHIx = self.distantPHIx[panel]
            PHIy = self.distantPHIy[panel]
            qr2.rule1.nodesInGlobalCoords(simplex1, self.x)
            qr2.rule2.nodesInGlobalCoords(simplex2, self.y)
            k = 0
            for i in range(qr2.rule1.num_nodes):
                for j in range(qr2.rule2.num_nodes):
                    self.temp[k] = self.kernel.evalPtr(1, &self.x[i, 0], &self.y[j, 0])
                    self.temp3[k] = self.kernel.evalPtr(1, &self.y[j, 0], &self.x[i, 0])
                    k += 1

            # ( phi1x * kernel(x, y) - phi1y * kernel(y, x) ) * psi2

            vol = (vol1 * vol2)
            k = 0
            for I in range(4):
                for J in range(4):
                    if mask & (1 << k):
                        for i in range(qr2.num_nodes):
                            self.temp2[i] = (PHIx[I, i]*self.temp[i] - PHIy[I, i]*self.temp3[i])*PSI[J, i]
                        contrib[k] = qr2.eval(self.temp2, vol)
                    k += 1
        else:
            print(np.array(simplex1), np.array(simplex2))
            raise NotImplementedError('Unknown panel type: {}'.format(panel))


cdef REAL_t symIntegrand1D_boundary(int n, REAL_t *xx, void *c_params):
    cdef:
        REAL_t l1x = xx[0]
        REAL_t l0x = 1.-l1x
        REAL_t x
        REAL_t y
        INDEX_t i = getINDEX(c_params, fDOF1)
        INDEX_t j = getINDEX(c_params, fDOF2)
        # INDEX_t nr1 = getINDEX(c_params, fNR1)
        # INDEX_t nc1 = getINDEX(c_params, fNC1)
        # INDEX_t nr2 = getINDEX(c_params, fNR2)
        # INDEX_t nc2 = getINDEX(c_params, fNC2)
        REAL_t* simplex1 = getREALArray2D(c_params, fSIMPLEX1)
        REAL_t* simplex2 = getREALArray2D(c_params, fSIMPLEX2)
        REAL_t phi1, phi2
        Kernel kernel = getKernel(c_params, fKERNEL)

    if i == 0:
        phi1 = l0x
    else:
        phi1 = l1x

    if j == 0:
        phi2 = l0x
    else:
        phi2 = l1x

    x = l0x*simplex1[0]+l1x*simplex1[1]
    y = simplex2[0]

    return phi1 * phi2 * kernel.evalPtr(1, &x, &y)


cdef class fractionalLaplacian1D_boundary(nonlocalLaplacian1D):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap DoFMap, num_dofs=None, **kwargs):
        manifold_dim2 = mesh.dim-1
        super(fractionalLaplacian1D_boundary, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2=manifold_dim2, **kwargs)
        self.symmetricCells = False


cdef class fractionalLaplacian1D_P1_boundary_automaticQuadrature(fractionalLaplacian1D_boundary):
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 num_dofs=None,
                 abstol=1e-4,
                 reltol=1e-4,
                 target_order=None,
                 **kwargs):
        assert isinstance(DoFMap, P1_DoFMap)
        super(fractionalLaplacian1D_P1_boundary_automaticQuadrature, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        if target_order is None:
            if isinstance(self.kernel, FractionalKernel):
                smin, smax = self.kernel.s.min, self.kernel.s.max
                # this is the desired local quadrature error
                target_order = 2.-smin
            else:
                target_order = 5
        self.target_order = target_order

        self.user_ptr = malloc(NUM_INTEGRAND_PARAMS*INTEGRAND_OFFSET)
        setINDEX(self.user_ptr, fNR1, 2)
        setINDEX(self.user_ptr, fNC1, 1)
        setINDEX(self.user_ptr, fNR2, 2)
        setINDEX(self.user_ptr, fNC2, 1)
        c_params = PyCapsule_New(self.user_ptr, NULL, NULL)
        func_type = b"double (int, double *, void *)"
        func_capsule = PyCapsule_New(<void*>symIntegrand1D_boundary, func_type, NULL)
        self.integrand = LowLevelCallable(func_capsule, c_params, func_type)
        self.abstol = abstol
        self.reltol = reltol
        setKernel(self.user_ptr, fKERNEL, self.kernel)

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        return DISTANT

    cdef void getNearQuadRule(self, panelType panel):
        pass

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J, t = 0
            REAL_t val, err, vol1 = self.vol1, vol2 = self.vol2
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            REAL_t horizon = self.kernel.getHorizonValue()

        setREALArray2D(self.user_ptr, fSIMPLEX1, simplex1)
        setREALArray2D(self.user_ptr, fSIMPLEX2, simplex2)

        contrib[:] = 0.

        if panel == COMMON_VERTEX:
            for i in range(2):
                if simplex1[i, 0] == simplex2[0, 0]:
                    t = i
                    break

            # loop over all local DoFs
            for I in range(2):
                for J in range(I, 2):
                    i = (t+I)%2
                    j = (t+J)%2
                    if j < i:
                        i, j = j, i
                    k = 2*i-(i*(i+1) >> 1) + j
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, i)
                        setINDEX(self.user_ptr, fDOF2, j)
                        val, err = nquad(self.integrand,
                                         ((0., 1.), ),
                                         opts={'epsabs': self.abstol, 'epsrel': self.reltol, 'points': [simplex2[0, 0]]})
                        contrib[k] = val*vol1*vol2
        elif panel == DISTANT:
            k = 0
            for I in range(2):
                for J in range(I, 2):
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, I)
                        setINDEX(self.user_ptr, fDOF2, J)
                        val, err = nquad(self.integrand,
                                         ((0., 1.), ),
                                         opts={'epsabs': self.abstol, 'epsrel': self.reltol, 'points': [simplex2[0, 0]]})
                        contrib[k] = val*vol1*vol2
                    k += 1
        else:
            print(np.array(simplex1), np.array(simplex2))
            raise NotImplementedError('Unknown panel type: {}'.format(panel))


cdef REAL_t symIntegrandId2D(int n, REAL_t *xx, void *c_params):
    cdef:
        REAL_t* lx = xx
        REAL_t* ly = xx+n//2
        REAL_t l1x = lx[0]
        REAL_t l2x = lx[1]
        REAL_t l1y = ly[0]
        REAL_t l2y = ly[1]
        REAL_t l0x = 1.-l1x-l2x
        REAL_t l0y = 1.-l1y-l2y
        REAL_t x[2]
        REAL_t y[2]
        INDEX_t i = getINDEX(c_params, fDOF1)
        INDEX_t j = getINDEX(c_params, fDOF2)
        REAL_t *simplex1 = getREALArray2D(c_params, fSIMPLEX2)
        # REAL_t* simplex2 = getREALArray2D(c_params, fSIMPLEX2)
        REAL_t psi1, psi2
        Kernel kernel = getKernel(c_params, fKERNEL)

    if i == 0:
        psi1 = l0x-l0y
    elif i == 1:
        psi1 = l1x-l1y
    else:
        psi1 = l2x-l2y

    if j == 0:
        psi2 = l0x-l0y
    elif j == 1:
        psi2 = l1x-l1y
    else:
        psi2 = l2x-l2y

    x[0] = l0x*simplex1[0]+l1x*simplex1[2]+l2x*simplex1[4]
    x[1] = l0x*simplex1[1]+l1x*simplex1[3]+l2x*simplex1[5]
    y[0] = l0y*simplex1[0]+l1y*simplex1[2]+l2y*simplex1[4]
    y[1] = l0y*simplex1[1]+l1y*simplex1[3]+l2y*simplex1[5]

    return psi1 * psi2 * kernel.evalPtr(2, x, y)


cdef REAL_t symIntegrandVertex2D(int n, REAL_t *xx, void *c_params):
    cdef:
        REAL_t* lx = xx
        REAL_t* ly = xx+n//2
        REAL_t l1x = lx[0]
        REAL_t l2x = lx[1]
        REAL_t l1y = ly[0]
        REAL_t l2y = ly[1]
        REAL_t l0x = 1.-l1x-l2x
        REAL_t l0y = 1.-l1y-l2y
        REAL_t x[2]
        REAL_t y[2]
        INDEX_t i = getINDEX(c_params, fDOF1)
        INDEX_t j = getINDEX(c_params, fDOF2)
        # INDEX_t nr1 = getINDEX(c_params, fNR1)
        # INDEX_t nc1 = getINDEX(c_params, fNC1)
        # INDEX_t nr2 = getINDEX(c_params, fNR2)
        # INDEX_t nc2 = getINDEX(c_params, fNC2)
        REAL_t* simplex1 = getREALArray2D(c_params, fSIMPLEX1)
        REAL_t* simplex2 = getREALArray2D(c_params, fSIMPLEX2)
        REAL_t psi1, psi2
        Kernel kernel = getKernel(c_params, fKERNEL)

    if i == 0:
        psi1 = l0x
    elif (i == 1) or (i == 2):
        psi1 = l1x-l0y
    else:
        psi1 = -l1y

    if j == 0:
        psi2 = l0x
    elif (j == 1) or (j == 2):
        psi2 = l1x-l0y
    else:
        psi2 = -l1y

    x[0] = l0x*simplex1[0]+l1x*simplex1[2]+l2x*simplex1[4]
    x[1] = l0x*simplex1[1]+l1x*simplex1[3]+l2x*simplex1[5]
    y[0] = l0y*simplex2[0]+l1y*simplex2[2]+l2y*simplex2[4]
    y[1] = l0y*simplex2[1]+l1y*simplex2[3]+l2y*simplex2[5]

    return psi1 * psi2 * kernel.evalPtr(2, x, y)


cdef REAL_t symIntegrandDistant2D(int n, REAL_t *xx, void *c_params):
    cdef:
        # REAL_t* lx = xx
        # REAL_t* ly = xx[2]
        REAL_t l1x = xx[0]
        REAL_t l2x = xx[1]
        REAL_t l1y = xx[2]
        REAL_t l2y = xx[3]
        REAL_t l0x = 1.-l1x-l2x
        REAL_t l0y = 1.-l1y-l2y
        REAL_t x[2]
        REAL_t y[2]
        INDEX_t i = getINDEX(c_params, fDOF1)
        INDEX_t j = getINDEX(c_params, fDOF2)
        # INDEX_t nr1 = getINDEX(c_params, fNR1)
        # INDEX_t nc1 = getINDEX(c_params, fNC1)
        # INDEX_t nr2 = getINDEX(c_params, fNR2)
        # INDEX_t nc2 = getINDEX(c_params, fNC2)
        REAL_t* simplex1 = getREALArray2D(c_params, fSIMPLEX1)
        REAL_t* simplex2 = getREALArray2D(c_params, fSIMPLEX2)
        REAL_t psi1, psi2
        Kernel kernel = getKernel(c_params, fKERNEL)

    if i == 0:
        psi1 = l0x
    elif i == 1:
        psi1 = l1x
    elif i == 2:
        psi1 = l2x
    elif i == 3:
        psi1 = -l0y
    elif i == 4:
        psi1 = -l1y
    else:
        psi1 = -l2y

    if j == 0:
        psi2 = l0x
    elif j == 1:
        psi2 = l1x
    elif j == 2:
        psi2 = l2x
    elif j == 3:
        psi2 = -l0y
    elif j == 4:
        psi2 = -l1y
    else:
        psi2 = -l2y

    x[0] = l0x*simplex1[0]+l1x*simplex1[2]+l2x*simplex1[4]
    x[1] = l0x*simplex1[1]+l1x*simplex1[3]+l2x*simplex1[5]
    y[0] = l0y*simplex2[0]+l1y*simplex2[2]+l2y*simplex2[4]
    y[1] = l0y*simplex2[1]+l1y*simplex2[3]+l2y*simplex2[5]

    return psi1 * psi2 * kernel.evalPtr(2, x, y)


cdef class fractionalLaplacian2D_P1_automaticQuadrature(nonlocalLaplacian2D):
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 num_dofs=None,
                 abstol=1e-4,
                 reltol=1e-4,
                 **kwargs):
        assert isinstance(DoFMap, P1_DoFMap)
        super(fractionalLaplacian2D_P1_automaticQuadrature, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)
        self.user_ptr = malloc(NUM_INTEGRAND_PARAMS*INTEGRAND_OFFSET)
        setINDEX(self.user_ptr, fNR1, 3)
        setINDEX(self.user_ptr, fNC1, 2)
        setINDEX(self.user_ptr, fNR2, 3)
        setINDEX(self.user_ptr, fNC2, 2)
        c_params = PyCapsule_New(self.user_ptr, NULL, NULL)
        func_type = b"double (int, double *, void *)"
        func_capsule_id = PyCapsule_New(<void*>symIntegrandId2D, func_type, NULL)
        func_capsule_vertex = PyCapsule_New(<void*>symIntegrandVertex2D, func_type, NULL)
        func_capsule_distant = PyCapsule_New(<void*>symIntegrandDistant2D, func_type, NULL)
        self.integrandId = LowLevelCallable(func_capsule_id, c_params, func_type)
        self.integrandVertex = LowLevelCallable(func_capsule_vertex, c_params, func_type)
        self.integrandDistant = LowLevelCallable(func_capsule_distant, c_params, func_type)
        self.abstol = abstol
        self.reltol = reltol
        setKernel(self.user_ptr, fKERNEL, self.kernel)

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        return DISTANT

    cdef void getNearQuadRule(self, panelType panel):
        pass

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J, t = 0
            REAL_t val, err, vol1 = self.vol1, vol2 = self.vol2
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2

        setREALArray2D(self.user_ptr, fSIMPLEX1, simplex1)
        setREALArray2D(self.user_ptr, fSIMPLEX2, simplex2)

        contrib[:] = 0.

        if panel == COMMON_EDGE:
            for I in range(2):
                for J in range(I, 2):
                    k = 4*I-(I*(I+1) >> 1) + J
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, I)
                        setINDEX(self.user_ptr, fDOF2, J)
                        val, err = nquad(self.integrandId,
                                         (lambda y: (0., y),
                                          (0., 1.)),
                                         opts=[lambda y: {'points': [y], 'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol}])
                        contrib[k] = val
                        val, err = nquad(self.integrandId,
                                         (lambda y: (y, 1.),
                                          (0., 1.)),
                                         opts=[lambda y: {'points': [y], 'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol}])
                        contrib[k] += val
                        contrib[k] *= vol1*vol1
        elif panel == COMMON_VERTEX:
            for i in range(2):
                for j in range(2):
                    if simplex1[i, 0] == simplex2[j, 0]:
                        if (i == 1) and (j == 0):
                            t = 2
                            break
                        elif (i == 0) and (j == 1):
                            t = 3
                            break
                        else:
                            raise IndexError()

            # loop over all local DoFs
            for I in range(3):
                for J in range(I, 3):
                    i = 3*(I//t)+(I%t)
                    j = 3*(J//t)+(J%t)
                    if j < i:
                        i, j = j, i
                    k = 4*i-(i*(i+1) >> 1) + j
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, i)
                        setINDEX(self.user_ptr, fDOF2, j)
                        val, err = nquad(self.integrandVertex,
                                         ((0., 1.),
                                          (0., 1.)),
                                         opts={'epsabs': self.abstol, 'epsrel': self.reltol})
                        contrib[k] = val*vol1*vol2
        elif panel == DISTANT:
            k = 0
            for I in range(6):
                for J in range(I, 6):
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, I)
                        setINDEX(self.user_ptr, fDOF2, J)
                        val, err = nquad(self.integrandDistant,
                                         (lambda  l2x, l1y, l2y: (0., 1.-l2x),
                                          (0., 1.),
                                          lambda l2y: (0., 1.-l2y),
                                          (0., 1.)),
                                         opts=[{'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol}])
                        contrib[k] = val*vol1*vol2*4.
                    k += 1
        else:
            print(np.array(simplex1), np.array(simplex2))
            raise NotImplementedError('Unknown panel type: {}'.format(panel))


cdef REAL_t symIntegrand2D_boundary(int n, REAL_t *xx, void *c_params):
    cdef:
        REAL_t l1x = xx[0]
        REAL_t l2x = xx[1]
        REAL_t l1y = xx[2]
        REAL_t l0x = 1.-l1x-l2x
        REAL_t l0y = 1.-l1y
        REAL_t x[2]
        REAL_t y[2]
        REAL_t w[2]
        REAL_t normal[2]
        INDEX_t i = getINDEX(c_params, fDOF1)
        INDEX_t j = getINDEX(c_params, fDOF2)
        # INDEX_t nr1 = getINDEX(c_params, fNR1)
        # INDEX_t nc1 = getINDEX(c_params, fNC1)
        # INDEX_t nr2 = getINDEX(c_params, fNR2)
        # INDEX_t nc2 = getINDEX(c_params, fNC2)
        REAL_t* simplex1 = getREALArray2D(c_params, fSIMPLEX1)
        REAL_t* simplex2 = getREALArray2D(c_params, fSIMPLEX2)
        REAL_t phi1, phi2, fac
        Kernel kernel = getKernel(c_params, fKERNEL)

    if i == 0:
        phi1 = l0x
    elif i == 1:
        phi1 = l1x
    else:
        phi1 = l2x

    if j == 0:
        phi2 = l0x
    elif j == 1:
        phi2 = l1x
    else:
        phi2 = l2x

    x[0] = l0x*simplex1[0]+l1x*simplex1[2]+l2x*simplex1[4]
    x[1] = l0x*simplex1[1]+l1x*simplex1[3]+l2x*simplex1[5]
    y[0] = l0y*simplex2[0]+l1y*simplex2[2]
    y[1] = l0y*simplex2[1]+l1y*simplex2[3]
    w[0] = y[0]-x[0]
    w[1] = y[1]-x[1]
    fac = 1./sqrt(w[0]*w[0] + w[1]*w[1])
    w[0] *= fac
    w[1] *= fac

    normal[0] = simplex2[3] - simplex2[1]
    normal[1] = simplex2[0] - simplex2[2]
    fac = 1./sqrt(normal[0]*normal[0] + normal[1]*normal[1])
    normal[0] *= fac
    normal[1] *= fac


    assert 0 <= l1x+l2x, (l1x, l2x)
    assert l1x+l2x <= 1., (l1x, l2x)
    assert 0 <= l1y, l1y
    assert l1y <= 1., l1y


    return phi1 * phi2 * (w[0]*normal[0] + w[1]*normal[1]) * kernel.evalPtr(2, &x[0], &y[0])


cdef class fractionalLaplacian2D_boundary(nonlocalLaplacian2D):
    def __init__(self, Kernel kernel, meshBase mesh, DoFMap DoFMap, num_dofs=None, **kwargs):
        manifold_dim2 = mesh.dim-1
        super(fractionalLaplacian2D_boundary, self).__init__(kernel, mesh, DoFMap, num_dofs, manifold_dim2=manifold_dim2, **kwargs)
        self.symmetricCells = False


cdef class fractionalLaplacian2D_P1_boundary_automaticQuadrature(fractionalLaplacian2D_boundary):
    def __init__(self,
                 Kernel kernel,
                 meshBase mesh,
                 DoFMap DoFMap,
                 num_dofs=None,
                 abstol=1e-4,
                 reltol=1e-4,
                 target_order=None,
                 **kwargs):
        assert isinstance(DoFMap, P1_DoFMap)
        super(fractionalLaplacian2D_P1_boundary_automaticQuadrature, self).__init__(kernel, mesh, DoFMap, num_dofs, **kwargs)

        if target_order is None:
            if isinstance(self.kernel, FractionalKernel):
                smin, smax = self.kernel.s.min, self.kernel.s.max
                # this is the desired local quadrature error
                target_order = 1.
            else:
                target_order = 5
        self.target_order = target_order

        self.user_ptr = malloc(NUM_INTEGRAND_PARAMS*INTEGRAND_OFFSET)
        setINDEX(self.user_ptr, fNR1, 3)
        setINDEX(self.user_ptr, fNC1, 2)
        setINDEX(self.user_ptr, fNR2, 3)
        setINDEX(self.user_ptr, fNC2, 2)
        c_params = PyCapsule_New(self.user_ptr, NULL, NULL)
        func_type = b"double (int, double *, void *)"
        func_capsule = PyCapsule_New(<void*>symIntegrand2D_boundary, func_type, NULL)
        self.integrand = LowLevelCallable(func_capsule, c_params, func_type)
        self.abstol = abstol
        self.reltol = reltol
        setKernel(self.user_ptr, fKERNEL, self.kernel)

    cdef panelType getQuadOrder(self,
                                const REAL_t h1,
                                const REAL_t h2,
                                REAL_t d):
        return DISTANT

    cdef void getNearQuadRule(self, panelType panel):
        pass

    cdef void eval(self,
                   REAL_t[::1] contrib,
                   panelType panel,
                   MASK_t mask=ALL):
        cdef:
            INDEX_t k, i, j, I, J, t = 0
            REAL_t val, err, vol1 = self.vol1, vol2 = self.vol2
            REAL_t[:, ::1] simplex1 = self.simplex1
            REAL_t[:, ::1] simplex2 = self.simplex2
            REAL_t horizon = self.kernel.getHorizonValue()

        setREALArray2D(self.user_ptr, fSIMPLEX1, simplex1)
        setREALArray2D(self.user_ptr, fSIMPLEX2, simplex2)

        contrib[:] = 0.

        if panel == COMMON_EDGE:
            pass
            # for i in range(3):
            #     for j in range(2):
            #         if (simplex1[i, 0] == simplex2[j, 0]) and (simplex1[i, 1] == simplex2[j, 1]):
            #             t = i
            #             break

            # loop over all local DoFs
            for I in range(3):
                for J in range(I, 3):
                    # i = (t+I)%3
                    # j = (t+J)%3
                    # if j < i:
                    #     i, j = j, i
                    # k = 3*i-(i*(i+1) >> 1) + j
                    i = I
                    j = J
                    k = 3*i-(i*(i+1) >> 1) + j
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, i)
                        setINDEX(self.user_ptr, fDOF2, j)
                        val, err = nquad(self.integrand,
                                         (lambda  l2x, l1y: (0., 1.-l2x),
                                          (0., 1.),
                                          (0., 1.)),
                                         opts=[{'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol}])
                        contrib[k] = val*vol1*vol2*2.0
        elif panel == COMMON_VERTEX:
            for i in range(3):
                for j in range(2):
                    if (simplex1[i, 0] == simplex2[j, 0]) and (simplex1[i, 1] == simplex2[j, 1]):
                        t = i
                        break

            # loop over all local DoFs
            for I in range(3):
                for J in range(I, 3):
                    i = (t+I)%3
                    j = (t+J)%3
                    if j < i:
                        i, j = j, i
                    k = 3*i-(i*(i+1) >> 1) + j
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, i)
                        setINDEX(self.user_ptr, fDOF2, j)
                        val, err = nquad(self.integrand,
                                         (lambda  l2x, l1y: (0., 1.-l2x),
                                          (0., 1.),
                                          (0., 1.)),
                                         opts=[{'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol}])
                        contrib[k] = val*vol1*vol2*2.0
        elif panel == DISTANT:
            k = 0
            for I in range(3):
                for J in range(I, 3):
                    if mask & (1 << k):
                        setINDEX(self.user_ptr, fDOF1, I)
                        setINDEX(self.user_ptr, fDOF2, J)
                        val, err = nquad(self.integrand,
                                         (lambda  l2x, l1y: (0., 1.-l2x),
                                          (0., 1.),
                                          (0., 1.)),
                                         opts=[{'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol},
                                               {'epsabs': self.abstol, 'epsrel': self.reltol}])
                        contrib[k] = val*vol1*vol2*2.0
                    k += 1
        else:
            print(np.array(simplex1), np.array(simplex2))
            raise NotImplementedError('Unknown panel type: {}'.format(panel))
