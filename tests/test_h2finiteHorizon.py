###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus.fem.mesh import intervalWithInteraction
from PyNucleus.fem.DoFMaps import P1_DoFMap
from PyNucleus.fem.functions import Lambda, constant
from PyNucleus.nl.fractionalOrders import (constFractionalOrder,
                                           variableConstFractionalOrder)
from PyNucleus.nl.nonlocalAssembly import nonlocalBuilder
from PyNucleus.nl.kernelNormalization import variableFractionalLaplacianScaling
from PyNucleus.nl.kernels import getFractionalKernel
from scipy.linalg import solve
import pytest


def idfunc(param):
    S = [str(p) for p in param]
    return '-'.join(S)


@pytest.fixture(scope='module',
                params=[(1, constFractionalOrder(0.25), 1.0, 0.5, True),
                        (1, constFractionalOrder(0.75), 1.0, 0.5, True),
                        (1, constFractionalOrder(0.25), 1.0, 0.5, False),
                        (1, constFractionalOrder(0.75), 1.0, 0.5, False),
                        (1, constFractionalOrder(0.25), 1.0, 2.5, False),
                        (1, constFractionalOrder(0.75), 1.0, 2.5, False),
                        ],
                ids=idfunc)
def kernels(request):
    dim, s, horizon1, horizon2, normalized = request.param
    kernel1 = getFractionalKernel(dim, s, constant(horizon1), normalized=normalized)
    kernel2 = getFractionalKernel(dim, s, constant(horizon2), normalized=normalized)
    return dim, kernel1, kernel2


def meshNoOverlap(dim):
    if dim == 1:
        mesh = intervalWithInteraction(a=-1, b=1, h=2**-8, horizon=0.)
    else:
        raise NotImplementedError()
    return mesh


def meshOverlap(dim, horizon):
    if dim == 1:
        mesh = intervalWithInteraction(a=-1, b=1, h=2**-8, horizon=horizon)
    else:
        raise NotImplementedError()
    return mesh


def test_h2_finite(kernels):
    dim, kernel1, kernel2 = kernels

    mesh1 = meshOverlap(dim, kernel1.horizon.value)
    dm1 = P1_DoFMap(mesh1)

    mesh2 = meshNoOverlap(dim)
    dm2 = P1_DoFMap(mesh2)

    ind = dm1.interpolate(Lambda(lambda x: abs(x[0]) < 1-1e-12))
    idx = ind.toarray() > 0

    builder1 = nonlocalBuilder(dm1, kernel1, zeroExterior=False, logging=True)
    print('\nDENSE\n')
    A1 = builder1.getDense()

    print('\nH2\n')
    A1_h2, Pnear = builder1.getH2(returnNearField=True)
    # A2.setKernel(kernel1)

    # mass = assembleMass(mesh1, dm1)
    # if mesh1.dim == 1:
    #     vol = 2
    # elif mesh1.dim == 2:
    #     vol = 2*np.pi * horizon
    # else:
    #     raise NotImplementedError()
    # C = kernel1.scalingValue
    # s = kernel1.s.value
    # M = (-vol*C*pow(kernel1.horizon.value, 1-mesh1.dim-2*s)/s) * mass
    # A3 = A2+M

    # err3 = np.log10(np.absolute(A1.toarray()-A3.toarray()))
    # print(err3.max())

    # plt.figure()
    # Adnear = A2.Anear.copy()
    # for i in range(Adnear.num_rows):
    #     Adnear.diagonal[i] = A1.data[i, i]
    #     for jj in range(Adnear.indptr[i], Adnear.indptr[i+1]):
    #         j = Adnear.indices[jj]
    #         Adnear.data[jj] = A1.data[i, j]
    # err2 = np.log10(np.absolute(A2.Anear.toarray()-Adnear.toarray()))
    # plt.pcolormesh(err2)
    # plt.colorbar()
    # print(err2.max())

    # plt.figure()
    # err2 = np.log10(np.absolute((A2-A2.Anear).toarray()-(A1-Adnear).toarray()))
    # plt.pcolormesh(err2, vmin=-6)
    # plt.colorbar()
    # print(err2.max())


    buildCorrected = not isinstance(kernel1.scaling, variableFractionalLaplacianScaling)

    if buildCorrected:
        print('\nCORRECTED\n')
        builder2 = nonlocalBuilder(dm2, kernel1, zeroExterior=False)
        A2 = builder2.getH2FiniteHorizon()
        A2.setKernel(kernel1)

    A1d = A1.toarray()[np.ix_(idx, idx)]
    A1_h2d = A1_h2.toarray()[np.ix_(idx, idx)]
    A1_h2_neard = A1_h2.Anear.toarray()[np.ix_(idx, idx)]
    A1_neard = A1d.copy()
    A1_neard[np.where(np.absolute(A1_h2_neard) == 0.)] = 0.

    if buildCorrected:
        A2d = A2.toarray()

    nn = np.absolute(A1d)
    nn[nn < 1e-16] = 1.

    errDenseH2 = np.absolute(A1d-A1_h2d)
    errDenseH2_rel = errDenseH2/nn
    print('errDenseH2', errDenseH2.max(), errDenseH2_rel.max())

    errDenseH2_near = np.absolute(A1_neard-A1_h2_neard)
    errDenseH2_near_rel = errDenseH2_near/nn
    print('errDenseH2_near', errDenseH2_near.max(), errDenseH2_near_rel.max())

    if buildCorrected:
        errDenseCor = np.absolute(A1d-A2d)
        errDenseCor_rel = errDenseCor/nn
        print('errDenseCor', errDenseCor.max(), errDenseCor_rel.max())

        errH2Cor = np.absolute(A1_h2d-A2d)
        errH2Cor_rel = errDenseCor/nn
        print('errH2Cor', errH2Cor.max(), errH2Cor_rel.max())

    # c = dm1.getDoFCoordinates()[idx, 0]
    # X, Y = np.meshgrid(c, c)

    # if errDenseH2.max() > -3:
    #     plt.figure()
    #     plt.pcolormesh(X, Y, np.log10(np.maximum(errDenseH2, 1e-12)), vmin=-6)
    #     plt.colorbar()
    #     plt.title('errDenseH2 absolute')

    #     plt.figure()
    #     plt.pcolormesh(X, Y, np.log10(np.maximum(errDenseH2_rel, 1e-12)), vmin=-6)
    #     plt.colorbar()
    #     plt.title('errDenseH2 relative')

    # if errDenseCor.max() > -3:
    #     plt.figure()
    #     plt.pcolormesh(X, Y, np.log10(np.maximum(errDenseCor, 1e-12)), vmin=-6)
    #     plt.colorbar()
    #     plt.title('errDenseCor absolute')

    #     plt.figure()
    #     plt.pcolormesh(X, Y, np.log10(np.maximum(errDenseCor_rel, 1e-12)), vmin=-6)
    #     plt.colorbar()
    #     plt.title('errDenseCor relative')

    #     # plt.figure()
    #     # A1_h2.plot(Pnear)

    #     plt.show()

    # assert errDenseH2.max() < 1e-4
    # assert errDenseCor.max() < 1e-4
    # assert errH2Cor.max() < 1e-4

    rhs = Lambda(lambda x: 1. if abs(x[0]) < 1. else 0.)
    b1 = dm1.assembleRHS(rhs).toarray()

    y1 = solve(A1d, b1[idx])
    x1 = np.zeros((A1.shape[0]))
    x1[idx] = y1

    y1_h2 = solve(A1_h2d, b1[idx])
    x1_h2 = np.zeros((A1.shape[0]))
    x1_h2[idx] = y1_h2

    if buildCorrected:
        b2 = dm2.assembleRHS(rhs).toarray()
        y2 = solve(A2d, b2)
        x2 = np.zeros((A1.shape[0]))
        x2[idx] = y2

    # assert np.absolute(A1d[np.ix_(idx, idx)]-A2d).max() < 1e-5

    M = dm1.assembleMass()
    L2_denseH2 = np.sqrt(abs(np.vdot(M*(x1-x1_h2), x1-x1_h2)))
    if buildCorrected:
        L2_denseCor = np.sqrt(abs(np.vdot(M*(x1-x2), x1-x2)))
        L2_H2Cor = np.sqrt(abs(np.vdot(M*(x1_h2-x2), x1_h2-x2)))
    L2_dense = np.sqrt(abs(np.vdot(M*x1, x1)))
    # L2_cor = np.sqrt(abs(np.vdot(M*x2, x2)))

    # if not (L2/L2_1 < mesh2.h**(0.5+min(kernel1.s.min, 0.5))):
    print('L2 errDenseH2', L2_denseH2)
    if buildCorrected:
        print('L2 errDenseCor', L2_denseCor)
        print('L2 errH2Cor', L2_H2Cor)

    # mesh1.plotFunction(x1, DoFMap=dm1, label='dense')
    # mesh1.plotFunction(x1_h2, DoFMap=dm1, label='h2')
    # mesh1.plotFunction(x2, DoFMap=dm1, label='corrected')
    # plt.legend()
    # plt.show()

    if buildCorrected:
        assert L2_denseCor/L2_dense < mesh2.h**(0.5+min(kernel1.s.min, 0.5)), (L2_denseCor, L2_dense, L2_denseCor/L2_dense, mesh2.h**(0.5+min(kernel1.s.min, 0.5)))

    mesh3 = meshOverlap(dim, kernel2.horizon.value)
    dm3 = P1_DoFMap(mesh3)

    ind = dm3.interpolate(Lambda(lambda x: abs(x[0]) < 1-1e-12))
    idx = ind.toarray() > 0

    if buildCorrected:
        print('\nCORRECTED\n')
        A2.setKernel(kernel2)

        print('\nDENSE\n')
        builder3 = nonlocalBuilder(dm3, kernel2, zeroExterior=False)
        A3 = builder3.getDense()

        A2d = A2.toarray()
        A3d = A3.toarray()[np.ix_(idx, idx)]

        nn = np.absolute(A3d)
        nn[nn < 1e-16] = 1.

        errDenseCor = np.absolute(A3d-A2d)
        errDenseCor_rel = errDenseCor/nn
        print('errDenseCor', errDenseCor.max(), errDenseCor_rel.max())

        y2 = solve(A2d, b2)
        x2 = np.zeros((A3.shape[0]))
        x2[idx] = y2

        b3 = dm3.assembleRHS(rhs).toarray()
        y3 = solve(A3d, b3[idx])
        x3 = np.zeros((A3.shape[0]))
        x3[idx] = y3

        # assert np.absolute(A3d[np.ix_(idx, idx)]-A2d).max() < 1e-5

        M = dm3.assembleMass()

        L2_denseCor = np.sqrt(abs(np.vdot(M*(x3-x2), x3-x2)))
        L2_dense = np.sqrt(abs(np.vdot(M*x3, x3)))
        # L2_cor = np.sqrt(abs(np.vdot(M*x2, x2)))

        print('L2 errDenseCor', L2_denseCor)

        # mesh3.plotFunction(x2, DoFMap=dm3, label='corrected')
        # mesh3.plotFunction(x3, DoFMap=dm3, label='dense')
        # plt.legend()
        # plt.show()

        # if not (L2 < mesh2.h**(0.5+min(kernel2.s.value, 0.5))):
        #     mesh3.plotFunction(x3, DoFMap=dm3)
        #     mesh3.plotFunction(x2, DoFMap=dm3)
        #     plt.figure()
        #     for lvl in A2.Ainf.Pfar:
        #         for fCP in A2.Ainf.Pfar[lvl]:
        #             fCP.plot()
        #     plt.figure()
        #     diff = np.absolute((A3d[np.ix_(idx, idx)]-A2d))
        #     plt.pcolormesh(np.log10(diff))
        #     plt.colorbar()
        #     plt.figure()
        #     diffRel = np.absolute((A3d[np.ix_(idx, idx)]-A2d)/A2d)
        #     diffRel[diff < 1e-12] = 0.
        #     plt.pcolormesh(np.log10(diffRel))
        #     print(diffRel[np.isfinite(diffRel)].max(), diffRel[np.isfinite(diffRel)].mean(), np.median(diffRel[np.isfinite(diffRel)]))
        #     plt.colorbar()
        #     plt.show()

        assert L2_denseCor/L2_dense < mesh2.h**(0.5+min(kernel2.s.min, 0.5)), (L2_denseCor, L2_dense, L2_denseCor/L2_dense, mesh2.h**(0.5+min(kernel1.s.min, 0.5)))
