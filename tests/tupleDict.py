###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus.base import INDEX
from PyNucleus.base.tupleDict import tupleDictMASK
from copy import deepcopy


def test_tupleDict():
    N = 15
    masks = tupleDictMASK(N, deleteHits=False, logicalAndHits=True)
    e = np.empty((2), dtype=INDEX)
    for k in range(N-2, -1, -1):
        e[0] = k
        e[1] = k+1
        masks.enterValue_py(e, 1)
    assert masks.nnz == N-1, (masks.nnz, N-1)
    assert masks.isCorrect()
    for k in range(N-1):
        e[0] = k
        e[1] = k+1
        masks.enterValue_py(e, 2)
    assert masks.nnz == N-1
    masks2 = deepcopy(masks)
    assert masks2.nnz == N-1
    for k in range(N-1):
        e[0] = k
        e[1] = k+1
        assert masks2[e] == 3
    masks3 = tupleDictMASK(N, deleteHits=False, logicalAndHits=True)
    for k in range(N-1):
        e[0] = k
        e[1] = k+1
        masks3.enterValue_py(e, 4)
    masks2.merge(masks3)
    assert masks2.nnz == N-1
    for k in range(N-1):
        e[0] = k
        e[1] = k+1
        assert masks2[e] == 7

    masks = tupleDictMASK(1, deleteHits=False, logicalAndHits=True)
    for k in range(30, -1, -1):
        e[0] = 0
        e[1] = k
        masks.enterValue_py(e, k)
    for k in range(31):
        e[0] = 0
        e[1] = k
        masks.enterValue_py(e, k)
    assert masks.nnz == 31, (masks.nnz, 31)
    assert masks.isCorrect()
    for k in range(31):
        e[0] = 0
        e[1] = k
        assert masks[e] == k

    masks = tupleDictMASK(1, deleteHits=False, logicalAndHits=True)
    for k in range(31):
        e[0] = 0
        e[1] = k
        masks.enterValue_py(e, k)
        assert masks.isCorrect(), k
    # for k in range(30, -1, -1):
    #     e[0] = 0
    #     e[1] = k
    #     masks.enterValue_py(e, k)
    assert masks.nnz == 31, (masks.nnz, 31)
    assert masks.isCorrect()
    for k in range(31):
        e[0] = 0
        e[1] = k
        assert masks[e] == k
