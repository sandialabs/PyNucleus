###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
from PyNucleus_base import INDEX
from PyNucleus_base.tupleDict import arrayIndexSet, bitArray


def test_arrayIndexSet():
    I = np.empty((3), dtype=INDEX)
    I[0] = 0
    I[1] = 3
    I[2] = 65
    aIS = arrayIndexSet(I, sorted=True)
    print(aIS.toSet())
    assert aIS.inSet_py(0)
    assert not aIS.inSet_py(1)
    assert aIS.inSet_py(3)
    assert not aIS.inSet_py(64)
    assert aIS.inSet_py(65)
    assert len(aIS) == 3
    for i in aIS:
        print(i)

    I[0] = 3
    I[1] = 0
    I[2] = 65
    aIS = arrayIndexSet(I, sorted=False)
    print(aIS.toSet())
    assert aIS.inSet_py(0)
    assert not aIS.inSet_py(1)
    assert aIS.inSet_py(3)
    assert not aIS.inSet_py(64)
    assert aIS.inSet_py(65)
    assert len(aIS) == 3
    for i in aIS:
        print(i)


    aIS.fromSet({759, 760, 761, 762, 763, 764, 765, 766, 767})
    aIS2 = arrayIndexSet()
    aIS2.fromSet({751, 752, 753, 754, 755, 756, 757, 758, 759})
    aIS3 = aIS.union(aIS2)
    assert len(aIS3) == 17
    aIS3 = aIS.inter(aIS2)
    assert len(aIS3) == 1

    aIS = arrayIndexSet()
    aIS.fromSet({759, 760, 761})
    aIS2 = arrayIndexSet()
    aIS2.fromSet({760})
    aIS3 = aIS.setminus(aIS2)
    assert len(aIS3) == 2
    assert aIS3.inSet_py(759)
    assert not aIS3.inSet_py(760)
    assert aIS3.inSet_py(761)


def test_bitArray():
    bA = bitArray()

    print(bA.toSet())
    bA.set_py(65)
    print(bA.toSet())
    bA.set_py(0)
    print(bA.toSet())
    bA.set_py(3)
    print(bA.toSet())


    assert bA.inSet_py(0)
    assert bA.inSet_py(3)
    assert bA.inSet_py(65)
    assert not bA.inSet_py(1)
    assert not bA.inSet_py(4)
    assert not bA.inSet_py(66)
    assert not bA.inSet_py(129)

    assert len(bA) == 3
    print(bA.length)

    bA.empty()

    assert len(bA) == 0

    bA.fromSet(set([0, 128]))

    assert bA.inSet_py(0)
    assert bA.inSet_py(128)
    assert len(bA) == 2

    bA.empty()
    for k in range(64):
        bA.set_py(k)
    print(bA.toSet())
    assert len(bA) == 64


    bA2 = bitArray()
    print(bA2.toSet())
    bA2.set_py(32)
    print(bA2.toSet())
    for k in range(32, 96):
        bA2.set_py(k)
    print(bA2.toSet())
    assert len(bA2) == 64

    print(bA.union(bA2).toSet())
    assert len(bA.union(bA2)) == 96
    print(bA.inter(bA2).toSet())
    assert len(bA.inter(bA2)) == 32
