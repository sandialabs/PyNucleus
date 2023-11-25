###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from PyNucleus_base.myTypes import REAL
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t
from PyNucleus_base import uninitialized
from PyNucleus_base.linear_operators cimport CSR_LinearOperator
from PyNucleus_base.sparsityPattern cimport sparsityPattern
from PyNucleus_fem.DoFMaps cimport (DoFMap,
                                    P0_DoFMap, P1_DoFMap, P2_DoFMap, P3_DoFMap)


def buildRestrictionProlongation(DoFMap coarse_DoFMap,
                                 DoFMap fine_DoFMap):
    if isinstance(coarse_DoFMap, P0_DoFMap):
        if isinstance(fine_DoFMap, P0_DoFMap):
            if coarse_DoFMap.mesh.manifold_dim == 1:
                R = buildRestriction_1D_P0(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 2:
                R = buildRestriction_2D_P0(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 3:
                R = buildRestriction_3D_P0(coarse_DoFMap, fine_DoFMap)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    elif isinstance(coarse_DoFMap, P1_DoFMap):
        if isinstance(fine_DoFMap, P1_DoFMap):
            if coarse_DoFMap.mesh.manifold_dim == 1:
                R = buildRestriction_1D_P1(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 2:
                R = buildRestriction_2D_P1(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 3:
                R = buildRestriction_3D_P1(coarse_DoFMap, fine_DoFMap)
            else:
                raise NotImplementedError()
        elif isinstance(fine_DoFMap, P2_DoFMap):
            if coarse_DoFMap.mesh.manifold_dim == 1:
                R = buildRestriction_1D_P1_P2(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 2:
                R = buildRestriction_2D_P1_P2(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 3:
                R = buildRestriction_3D_P1_P2(coarse_DoFMap, fine_DoFMap)
            else:
                raise NotImplementedError()
        elif isinstance(fine_DoFMap, P3_DoFMap):
            if coarse_DoFMap.mesh.manifold_dim == 1:
                R = buildRestriction_1D_P1_P3(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 2:
                R = buildRestriction_2D_P1_P3(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 3:
                R = buildRestriction_3D_P1_P3(coarse_DoFMap, fine_DoFMap)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    elif isinstance(coarse_DoFMap, P2_DoFMap):
        if isinstance(fine_DoFMap, P2_DoFMap):
            if coarse_DoFMap.mesh.manifold_dim == 1:
                R = buildRestriction_1D_P2(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 2:
                R = buildRestriction_2D_P2(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 3:
                R = buildRestriction_3D_P2(coarse_DoFMap, fine_DoFMap)
            else:
                raise NotImplementedError()
        elif isinstance(fine_DoFMap, P3_DoFMap):
            if coarse_DoFMap.mesh.manifold_dim == 1:
                R = buildRestriction_1D_P2_P3(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 2:
                R = buildRestriction_2D_P2_P3(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 3:
                R = buildRestriction_3D_P2_P3(coarse_DoFMap, fine_DoFMap)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    elif isinstance(coarse_DoFMap, P3_DoFMap):
        if isinstance(fine_DoFMap, P3_DoFMap):
            if coarse_DoFMap.mesh.manifold_dim == 1:
                R = buildRestriction_1D_P3(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 2:
                R = buildRestriction_2D_P3(coarse_DoFMap, fine_DoFMap)
            elif coarse_DoFMap.mesh.manifold_dim == 3:
                R = buildRestriction_3D_P3(coarse_DoFMap, fine_DoFMap)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError('Unknown DoFMap: {}'.format(coarse_DoFMap))
    P = R.transpose()
    return R, P


cdef inline void add(sparsityPattern sPat,
                     const INDEX_t dof,
                     const INDEX_t dofF):
    if dofF >= 0:
        sPat.add(dof, dofF)


cdef inline void enterData(CSR_LinearOperator R,
                           const INDEX_t dof,
                           const INDEX_t dofF,
                           const REAL_t val):
    if dofF >= 0:
        R.setEntry(dof, dofF, val)


include "restriction_1D_P0.pxi"
include "restriction_2D_P0.pxi"
include "restriction_3D_P0.pxi"

include "restriction_1D_P1.pxi"
include "restriction_2D_P1.pxi"
include "restriction_3D_P1.pxi"

include "restriction_1D_P1_P2.pxi"
include "restriction_2D_P1_P2.pxi"
include "restriction_3D_P1_P2.pxi"

include "restriction_1D_P1_P3.pxi"
include "restriction_2D_P1_P3.pxi"
include "restriction_3D_P1_P3.pxi"

include "restriction_1D_P2.pxi"
include "restriction_2D_P2.pxi"
include "restriction_3D_P2.pxi"

include "restriction_1D_P2_P3.pxi"
include "restriction_2D_P2_P3.pxi"
include "restriction_3D_P2_P3.pxi"

include "restriction_1D_P3.pxi"
include "restriction_2D_P3.pxi"
include "restriction_3D_P3.pxi"
