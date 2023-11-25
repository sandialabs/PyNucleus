###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
cimport numpy as np
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py cimport MPI
from mpi4py cimport libmpi as mpi

from PyNucleus_base import uninitialized

ctypedef mpi.MPI_Comm MPI_Comm

include "metisTypes_decl.pxi"
include "metisTypes.pxi"


cdef extern from "parmetis.h":
    int ParMETIS_V3_PartMeshKway(
        idx_t *elmdist, idx_t *eptr, idx_t *eind, idx_t *elmwgt,
        idx_t *wgtflag, idx_t *numflag, idx_t *ncon, idx_t *ncommonnodes, idx_t *nparts,
        real_t *tpwgts, real_t *ubvec, idx_t *options, idx_t *edgecut, idx_t *part,
        MPI_Comm *comm)

    int ParMETIS_V3_Mesh2Dual(
        idx_t *elmdist, idx_t *eptr, idx_t *eind, idx_t *numflag,
        idx_t *ncommonnodes, idx_t **xadj, idx_t **adjncy, MPI_Comm *comm)

    int ParMETIS_V3_RefineKway(
        idx_t *vtxdist, idx_t *xadj, idx_t *adjncy, idx_t *vwgt,
        idx_t *adjwgt, idx_t *wgtflag, idx_t *numflag, idx_t *ncon, idx_t *nparts,
        real_t *tpwgts, real_t *ubvec, idx_t *options, idx_t *edgecut,
        idx_t *part, MPI_Comm *comm)

    int METIS_Free(void *ptr)

    idx_t METIS_OK, METIS_ERROR_INPUT, METIS_ERROR_MEMORY, METIS_ERROR
    idx_t PARMETIS_PSR_COUPLED, PARMETIS_PSR_UNCOUPLED
    idx_t PARMETIS_DBGLVL_TIME, PARMETIS_DBGLVL_INFO, PARMETIS_DBGLVL_PROGRESS, PARMETIS_DBGLVL_REFINEINFO, PARMETIS_DBGLVL_MATCHINFO, PARMETIS_DBGLVL_RMOVEINFO, PARMETIS_DBGLVL_REMAP


cpdef process_return(returnVal):
    if returnVal == METIS_OK:
        return
    elif returnVal == METIS_ERROR_INPUT:
        raise Exception("METIS_ERROR_INPUT")
    elif returnVal == METIS_ERROR_MEMORY:
        raise Exception("METIS_ERROR_MEMORY")
    elif returnVal == METIS_ERROR:
        raise Exception("METIS_ERROR")
    else:
        raise Exception("Unknown METIS error")


cpdef PartMeshKway(idx_t[::1] elemdist,
                   idx_t[::1] eptr,
                   idx_t[::1] eind,
                   idx_t ncommonnodes,
                   idx_t nparts,
                   MPI.Comm comm,
                   idx_t[::1] elemwgt=None,
                   idx_t[::1] adjwgt=None,
                   real_t[:, ::1] tpwgts=None,
                   real_t[::1] ubvec=None,
                   idx_t[::1] options=None):
    cdef:
        np.ndarray[idx_t, ndim=1] part = uninitialized((eptr.shape[0]-1), dtype=idx)
        idx_t[::1] part_mv = part
        idx_t *elemwgtPtr
        int returnVal
        idx_t wgtflag, numflag = 0, ncon, edgecut = 0
        MPI_Comm *commPtr = &comm.ob_mpi

    if options is None:
        options = np.zeros((3), dtype=idx)
    elemwgtPtr = NULL if elemwgt is None else &elemwgt[0]

    if ubvec is None:
        ncon = 1
        ubvec = real(1.05)*np.ones((ncon), dtype=real)
    else:
        ncon = ubvec.shape[0]

    if tpwgts is None:
        tpwgts = 1/real(nparts)*np.ones((ncon, nparts), dtype=real)

    if elemwgt is None:
        wgtflag = 0
    else:
        wgtflag = 2

    returnVal = ParMETIS_V3_PartMeshKway(&elemdist[0],
                                         &eptr[0],
                                         &eind[0],
                                         elemwgtPtr,
                                         &wgtflag,
                                         &numflag,
                                         &ncon,
                                         &ncommonnodes,
                                         &nparts,
                                         &tpwgts[0, 0],
                                         &ubvec[0],
                                         &options[0],
                                         &edgecut,
                                         &part_mv[0],
                                         commPtr)
    process_return(returnVal)
    return part


cpdef Mesh2Dual(idx_t[::1] elemdist,
                idx_t[::1] eptr,
                idx_t[::1] eind,
                idx_t ncommonnodes,
                MPI.Comm comm):
    cdef:
        int returnVal
        idx_t numflag = 0
        MPI_Comm *commPtr = &comm.ob_mpi
        idx_t *xadjPtr
        idx_t *adjncyPtr
        idx_t[::1] xadj = np.zeros((eptr.shape[0]), dtype=idx)
        idx_t[::1] adjncy

    returnVal = ParMETIS_V3_Mesh2Dual(&elemdist[0],
                                      &eptr[0],
                                      &eind[0],
                                      &numflag,
                                      &ncommonnodes,
                                      &xadjPtr,
                                      &adjncyPtr,
                                      commPtr)
    process_return(returnVal)

    for i in range(xadj.shape[0]):
        xadj[i] = xadjPtr[i]
    returnVal = METIS_Free(xadjPtr)
    process_return(returnVal)
    adjncy = np.zeros((xadj[xadj.shape[0]-1]), dtype=idx)
    for i in range(adjncy.shape[0]):
        adjncy[i] = adjncyPtr[i]
    returnVal = METIS_Free(adjncyPtr)
    process_return(returnVal)

    return np.array(xadj, copy=False, dtype=idx), np.array(adjncy, copy=False, dtype=idx)


cpdef RefineKway(idx_t[::1] vtxdist,
                 idx_t[::1] xadj,
                 idx_t[::1] adjncy,
                 idx_t[::1] part,
                 idx_t nparts,
                 MPI.Comm comm,
                 idx_t[::1] vwgt=None,
                 idx_t[::1] adjwgt=None,
                 real_t[:, ::1] tpwgts=None,
                 real_t[::1] ubvec=None,
                 idx_t[::1] options=None):
    cdef:
        idx_t *vwgtPtr
        idx_t *adjwgtPtr
        int returnVal
        idx_t wgtflag, numflag = 0, ncon, edgecut = 0
        MPI_Comm *commPtr = &comm.ob_mpi

    if options is None:
        options = np.zeros((4), dtype=idx)
    vwgtPtr = NULL if vwgt is None else &vwgt[0]
    adjwgtPtr = NULL if adjwgt is None else &adjwgt[0]

    if ubvec is None:
        ncon = 1
        ubvec = real(1.05)*np.ones((ncon), dtype=real)
    else:
        ncon = ubvec.shape[0]

    if tpwgts is None:
        tpwgts = 1/real(nparts)*np.ones((ncon, nparts), dtype=real)

    if vwgt is None:
        if adjwgt is None:
            wgtflag = 0
        else:
            wgtflag = 1
    else:
        if adjwgt is None:
            wgtflag = 2
        else:
            wgtflag = 3

    returnVal = ParMETIS_V3_RefineKway(&vtxdist[0],
                                       &xadj[0],
                                       &adjncy[0],
                                       vwgtPtr,
                                       adjwgtPtr,
                                       &wgtflag,
                                       &numflag,
                                       &ncon,
                                       &nparts,
                                       &tpwgts[0, 0],
                                       &ubvec[0],
                                       &options[0],
                                       &edgecut,
                                       &part[0],
                                       commPtr)
    process_return(returnVal)
    # print(edgecut)
    return part
