###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
from PyNucleus_fem import functionFactory, meshFactory, dofmapFactory
from numpy.testing import assert_allclose
import pytest

rtol = 2e-3
atol = 1e-10


@pytest.fixture(scope='module', params=['square'])
def mesh(request):
    if request.param == 'square':
        mesh = meshFactory('square', N=2, noRef=5)
    else:
        raise NotImplementedError()
    return mesh


def test_integrals_drift(mesh):
    dm = dofmapFactory('P1', mesh, -1)

    c1 = functionFactory('vector', [functionFactory('constant', 1.),
                                    functionFactory('constant', 0.)])
    c2 = functionFactory('vector', [functionFactory('constant', 0.),
                                    functionFactory('constant', 1.)])
    D1 = dm.assembleDrift(c1)
    D2 = dm.assembleDrift(c2)

    x = dm.getDoFCoordinates()[:, 0]
    y = dm.getDoFCoordinates()[:, 1]

    u = dm.zeros()
    v = dm.zeros()
    for i in range(3):
        for j in range(3):
            u.assign(x**i*y**j)
            for l in range(3):
                for m in range(3):
                    v.assign(x**l*y**m)
                    if i+l > 0:
                        assert_allclose(np.vdot(u, D1*v), l/(i+l)/(j+m+1), rtol=rtol, atol=atol)
                    else:
                        assert_allclose(np.vdot(u, D1*v), 0., atol=atol)
                    if j+m > 0:
                        assert_allclose(np.vdot(u, D2*v), m/(i+l+1)/(j+m), rtol=rtol, atol=atol)
                    else:
                        assert_allclose(np.vdot(u, D2*v), 0., atol=atol)


def test_integrals_grad(mesh):
    dm = dofmapFactory('P1', mesh, -1)

    c1 = functionFactory('vector', [functionFactory('constant', 1.),
                                    functionFactory('constant', 0.)])
    c2 = functionFactory('vector', [functionFactory('constant', 0.),
                                    functionFactory('constant', 1.)])
    x = dm.getDoFCoordinates()[:, 0]
    y = dm.getDoFCoordinates()[:, 1]

    v = dm.zeros()
    for i in range(3):
        for j in range(3):
            b1 = dm.assembleRHSgrad(functionFactory('Lambda', lambda x: x[0]**i*x[1]**j), c1)
            b2 = dm.assembleRHSgrad(functionFactory('Lambda', lambda x: x[0]**i*x[1]**j), c2)
            for l in range(3):
                for m in range(3):
                    v.assign(x**l*y**m)
                    if i+l > 0:
                        assert_allclose(np.vdot(b1, v), l/(i+l)/(j+m+1), rtol=rtol, atol=atol)
                    else:
                        assert_allclose(np.vdot(b1, v), 0., atol=atol)
                    if j+m > 0:
                        assert_allclose(np.vdot(b2, v), m/(i+l+1)/(j+m), rtol=rtol, atol=atol)
                    else:
                        assert_allclose(np.vdot(b2, v), 0., atol=atol)
