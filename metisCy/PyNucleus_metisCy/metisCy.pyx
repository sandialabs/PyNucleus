###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import numpy as np
cimport numpy as np

include "config.pxi"

from PyNucleus_base import uninitialized

IF IDXTYPEWIDTH == 32:
    idx = np.int32
ELIF IDXTYPEWIDTH == 64:
    idx = np.int64

IF REALTYPEWIDTH == 32:
    real = np.float32
ELIF REALTYPEWIDTH == 64:
    real = np.float64


cdef extern from "metis.h":
    int METIS_PartGraphRecursive(idx_t *nvtxs, idx_t *ncon, idx_t *xadj, idx_t *adjncy,
                                 idx_t *vwgt, idx_t *vsize, idx_t *adjwgt, idx_t *nparts, real_t *tpwgts,
                                 real_t *ubvec, idx_t *options, idx_t *objval, idx_t *part)

    int METIS_PartGraphKway(idx_t *nvtxs, idx_t *ncon, idx_t *xadj, idx_t *adjncy,
                            idx_t *vwgt, idx_t *vsize, idx_t *adjwgt, idx_t *nparts, real_t *tpwgts,
                            real_t *ubvec, idx_t *options, idx_t *objval, idx_t *part)

    int METIS_MeshToDual(idx_t *ne, idx_t *nn, idx_t *eptr, idx_t *eind,
                         idx_t *ncommon, idx_t *numflag, idx_t **r_xadj, idx_t **r_adjncy);

    int METIS_MeshToNodal(idx_t *ne, idx_t *nn, idx_t *eptr, idx_t *eind,
                          idx_t *numflag, idx_t **r_xadj, idx_t **r_adjncy);

    int METIS_PartMeshNodal(idx_t *ne, idx_t *nn, idx_t *eptr, idx_t *eind,
                            idx_t *vwgt, idx_t *vsize, idx_t *nparts, real_t *tpwgts,
                            idx_t *options, idx_t *objval, idx_t *epart, idx_t *npart);

    int METIS_PartMeshDual(idx_t *ne, idx_t *nn, idx_t *eptr, idx_t *eind,
                           idx_t *vwgt, idx_t *vsize, idx_t *ncommon, idx_t *nparts,
                           real_t *tpwgts, idx_t *options, idx_t *objval, idx_t *epart,
                           idx_t *npart);

    int METIS_NodeND(idx_t *nvtxs, idx_t *xadj, idx_t *adjncy, idx_t *vwgt,
                     idx_t *options, idx_t *perm, idx_t *iperm);

    int METIS_Free(void *ptr);

    idx_t METIS_NOPTIONS

    idx_t METIS_OK, METIS_ERROR_INPUT, METIS_ERROR_MEMORY, METIS_ERROR

    # Options codes
    idx_t METIS_OPTION_PTYPE
    idx_t METIS_OPTION_OBJTYPE
    idx_t METIS_OPTION_CTYPE
    idx_t METIS_OPTION_IPTYPE
    idx_t METIS_OPTION_RTYPE
    idx_t METIS_OPTION_DBGLVL
    idx_t METIS_OPTION_NITER
    idx_t METIS_OPTION_NCUTS
    idx_t METIS_OPTION_SEED
    idx_t METIS_OPTION_NO2HOP
    idx_t METIS_OPTION_MINCONN
    idx_t METIS_OPTION_CONTIG
    idx_t METIS_OPTION_COMPRESS
    idx_t METIS_OPTION_CCORDER
    idx_t METIS_OPTION_PFACTOR
    idx_t METIS_OPTION_NSEPS
    idx_t METIS_OPTION_UFACTOR
    idx_t METIS_OPTION_NUMBERING

    # Partitioning Schemes
    idx_t METIS_PTYPE_RB
    idx_t METIS_PTYPE_KWAY

    # Graph types for meshes
    idx_t METIS_GTYPE_DUAL
    idx_t METIS_GTYPE_NODAL

    # Coarsening Schemes
    idx_t METIS_CTYPE_RM
    idx_t METIS_CTYPE_SHEM

    # Initial partitioning schemes
    idx_t METIS_IPTYPE_GROW
    idx_t METIS_IPTYPE_RANDOM
    idx_t METIS_IPTYPE_EDGE
    idx_t METIS_IPTYPE_NODE
    idx_t METIS_IPTYPE_METISRB

    # Refinement schemes
    idx_t METIS_RTYPE_FM
    idx_t METIS_RTYPE_GREEDY
    idx_t METIS_RTYPE_SEP2SIDED
    idx_t METIS_RTYPE_SEP1SIDED

    # Debug Levels
    idx_t METIS_DBG_INFO        # Shows various diagnostic messages
    idx_t METIS_DBG_TIME        # Perform timing analysis
    idx_t METIS_DBG_COARSEN     # Show the coarsening progress
    idx_t METIS_DBG_REFINE      # Show the refinement progress
    idx_t METIS_DBG_IPART       # Show info on initial partitioning
    idx_t METIS_DBG_MOVEINFO    # Show info on vertex moves during refinement
    idx_t METIS_DBG_SEPINFO     # Show info on vertex moves during sep refinement
    idx_t METIS_DBG_CONNINFO    # Show info on minimization of subdomain connectivity
    idx_t METIS_DBG_CONTIGINFO  # Show info on elimination of connected components
    idx_t METIS_DBG_MEMORY      # Show info related to wspace allocation

    # Types of objectives
    idx_t METIS_OBJTYPE_CUT
    idx_t METIS_OBJTYPE_VOL
    idx_t METIS_OBJTYPE_NODE

    int METIS_SetDefaultOptions(idx_t *options)


NOPTIONS = METIS_NOPTIONS

# Options codes
OPTION_PTYPE	 = METIS_OPTION_PTYPE
OPTION_OBJTYPE	 = METIS_OPTION_OBJTYPE
OPTION_CTYPE	 = METIS_OPTION_CTYPE
OPTION_IPTYPE	 = METIS_OPTION_IPTYPE
OPTION_RTYPE	 = METIS_OPTION_RTYPE
OPTION_DBGLVL	 = METIS_OPTION_DBGLVL
OPTION_NITER	 = METIS_OPTION_NITER
OPTION_NCUTS	 = METIS_OPTION_NCUTS
OPTION_SEED	 = METIS_OPTION_SEED
OPTION_NO2HOP	 = METIS_OPTION_NO2HOP
OPTION_MINCONN	 = METIS_OPTION_MINCONN
OPTION_CONTIG	 = METIS_OPTION_CONTIG
OPTION_COMPRESS	 = METIS_OPTION_COMPRESS
OPTION_CCORDER	 = METIS_OPTION_CCORDER
OPTION_PFACTOR	 = METIS_OPTION_PFACTOR
OPTION_NSEPS	 = METIS_OPTION_NSEPS
OPTION_UFACTOR	 = METIS_OPTION_UFACTOR
OPTION_NUMBERING = METIS_OPTION_NUMBERING

# Partitioning Schemes
PTYPE_RB   = METIS_PTYPE_RB
PTYPE_KWAY = METIS_PTYPE_KWAY

# Graph types for meshes
GTYPE_DUAL  = METIS_GTYPE_DUAL
GTYPE_NODAL = METIS_GTYPE_NODAL

# Coarsening Schemes
CTYPE_RM   = METIS_CTYPE_RM
CTYPE_SHEM = METIS_CTYPE_SHEM

# Initial partitioning schemes
IPTYPE_GROW    = METIS_IPTYPE_GROW
IPTYPE_RANDOM  = METIS_IPTYPE_RANDOM
IPTYPE_EDGE    = METIS_IPTYPE_EDGE
IPTYPE_NODE    = METIS_IPTYPE_NODE
IPTYPE_METISRB = METIS_IPTYPE_METISRB

# Refinement schemes
RTYPE_FM	= METIS_RTYPE_FM
RTYPE_GREEDY	= METIS_RTYPE_GREEDY
RTYPE_SEP2SIDED = METIS_RTYPE_SEP2SIDED
RTYPE_SEP1SIDED = METIS_RTYPE_SEP1SIDED

# Debug Levels
DBG_INFO       = METIS_DBG_INFO
DBG_TIME       = METIS_DBG_TIME
DBG_COARSEN    = METIS_DBG_COARSEN
DBG_REFINE     = METIS_DBG_REFINE
DBG_IPART      = METIS_DBG_IPART
DBG_MOVEINFO   = METIS_DBG_MOVEINFO
DBG_SEPINFO    = METIS_DBG_SEPINFO
DBG_CONNINFO   = METIS_DBG_CONNINFO
DBG_CONTIGINFO = METIS_DBG_CONTIGINFO
DBG_MEMORY     = METIS_DBG_MEMORY

# Types of objectives
OBJTYPE_CUT  = METIS_OBJTYPE_CUT
OBJTYPE_VOL  = METIS_OBJTYPE_VOL
OBJTYPE_NODE = METIS_OBJTYPE_NODE


cpdef SetDefaultOptions():
    cdef:
        np.ndarray[idx_t, ndim=1] options = uninitialized((NOPTIONS), dtype=idx)
        idx_t[::1] options_mv = options

    METIS_SetDefaultOptions(&options_mv[0])
    return options


cpdef process_return(returnVal):
    if returnVal == METIS_OK:
        return
    else:
        raise Exception


cpdef PartGraphRecursive(idx_t[::1] xadj,
                         idx_t[::1] adjncy,
                         idx_t nparts,
                         idx_t[::1] vwgt=None,
                         idx_t[::1] vsize=None,
                         idx_t[::1] adjwgt=None,
                         real_t[::1] tpwgts=None,
                         real_t[::1] ubvec=None,
                         idx_t[::1] options=None):
    cdef:
        idx_t nvtxs = xadj.shape[0]-1, ncon, objval
        np.ndarray[idx_t, ndim=1] part = uninitialized(nvtxs, dtype=idx)
        idx_t[::1] part_mv = part
        idx_t *vwgtPtr
        idx_t *vsizePtr
        idx_t *adjwgtPtr
        idx_t *optionsPtr
        real_t *tpwgtsPtr
        real_t *ubvecPtr
        int returnVal

    optionsPtr = NULL if options is None else &options[0]
    vwgtPtr = NULL if vwgt is None else &vwgt[0]
    vsizePtr = NULL if vsize is None else &vsize[0]
    adjwgtPtr = NULL if adjwgt is None else &adjwgt[0]
    tpwgtsPtr = NULL if tpwgts is None else &tpwgts[0]

    if ubvec is None:
        ncon = 1
        ubvecPtr = NULL
    else:
        ncon = ubvec.shape[0]
        ubvecPtr = &ubvec[0]

    returnVal = METIS_PartGraphRecursive(&nvtxs, &ncon, &xadj[0], &adjncy[0],
                                         vwgtPtr, vsizePtr, adjwgtPtr, &nparts,
                                         tpwgtsPtr, ubvecPtr, optionsPtr,
                                         &objval, &part_mv[0])
    process_return(returnVal)
    return part, objval


cpdef PartGraphKway(idx_t[::1] xadj,
                    idx_t[::1] adjncy,
                    idx_t nparts,
                    idx_t[::1] vwgt=None,
                    idx_t[::1] vsize=None,
                    idx_t[::1] adjwgt=None,
                    real_t[::1] tpwgts=None,
                    real_t[::1] ubvec=None,
                    idx_t[::1] options=None):
    cdef:
        idx_t nvtxs = xadj.shape[0]-1, ncon, objval
        np.ndarray[idx_t, ndim=1] part = uninitialized(nvtxs, dtype=idx)
        idx_t[::1] part_mv = part
        idx_t *vwgtPtr
        idx_t *vsizePtr
        idx_t *adjwgtPtr
        idx_t *optionsPtr
        real_t *tpwgtsPtr
        real_t *ubvecPtr
        int returnVal

    optionsPtr = NULL if options is None else &options[0]
    vwgtPtr = NULL if vwgt is None else &vwgt[0]
    vsizePtr = NULL if vsize is None else &vsize[0]
    adjwgtPtr = NULL if adjwgt is None else &adjwgt[0]
    tpwgtsPtr = NULL if tpwgts is None else &tpwgts[0]

    if ubvec is None:
        ncon = 1
        ubvecPtr = NULL
    else:
        ncon = ubvec.shape[0]
        ubvecPtr = &ubvec[0]

    returnVal = METIS_PartGraphKway(&nvtxs, &ncon, &xadj[0], &adjncy[0],
                                    vwgtPtr, vsizePtr, adjwgtPtr, &nparts,
                                    tpwgtsPtr, ubvecPtr, optionsPtr,
                                    &objval, &part_mv[0])
    process_return(returnVal)
    return part, objval


cpdef PartMeshDual(idx_t[::1] eptr,
                   idx_t[::1] eind,
                   idx_t ncommon,
                   idx_t nparts,
                   idx_t[::1] vwgt=None,
                   idx_t[::1] vsize=None,
                   idx_t[::1] adjwgt=None,
                   real_t[::1] tpwgts=None,
                   real_t[::1] ubvec=None,
                   idx_t[::1] options=None):
    cdef:
        idx_t ne = eptr.shape[0]-1, objval
        idx_t nn = max(eind)+1
        np.ndarray[idx_t, ndim=1] npart = uninitialized(nn, dtype=idx)
        np.ndarray[idx_t, ndim=1] epart = uninitialized(ne, dtype=idx)
        idx_t[::1] npart_mv = npart
        idx_t[::1] epart_mv = epart
        idx_t *vwgtPtr
        idx_t *vsizePtr
        idx_t *optionsPtr
        real_t *tpwgtsPtr
        int returnVal

    optionsPtr = NULL if options is None else &options[0]
    vwgtPtr = NULL if vwgt is None else &vwgt[0]
    vsizePtr = NULL if vsize is None else &vsize[0]
    tpwgtsPtr = NULL if tpwgts is None else &tpwgts[0]

    returnVal = METIS_PartMeshDual(&ne, &nn, &eptr[0], &eind[0],
                                   vwgtPtr, vsizePtr, &ncommon, &nparts,
                                   tpwgtsPtr, optionsPtr,
                                   &objval, &epart_mv[0], &npart_mv[0])
    process_return(returnVal)
    return epart, npart, objval


cpdef PartMeshNodal(idx_t[::1] eptr,
                    idx_t[::1] eind,
                    idx_t nparts,
                    idx_t[::1] vwgt=None,
                    idx_t[::1] vsize=None,
                    idx_t[::1] adjwgt=None,
                    real_t[::1] tpwgts=None,
                    real_t[::1] ubvec=None,
                    idx_t[::1] options=None):
    cdef:
        idx_t ne = eptr.shape[0]-1, objval
        idx_t nn = max(eind)+1
        np.ndarray[idx_t, ndim=1] npart = uninitialized(nn, dtype=idx)
        np.ndarray[idx_t, ndim=1] epart = uninitialized(ne, dtype=idx)
        idx_t[::1] npart_mv = npart
        idx_t[::1] epart_mv = epart
        idx_t *vwgtPtr
        idx_t *vsizePtr
        idx_t *optionsPtr
        real_t *tpwgtsPtr
        int returnVal

    optionsPtr = NULL if options is None else &options[0]
    vwgtPtr = NULL if vwgt is None else &vwgt[0]
    vsizePtr = NULL if vsize is None else &vsize[0]
    tpwgtsPtr = NULL if tpwgts is None else &tpwgts[0]

    returnVal = METIS_PartMeshNodal(&ne, &nn, &eptr[0], &eind[0],
                                    vwgtPtr, vsizePtr, &nparts,
                                    tpwgtsPtr, optionsPtr,
                                    &objval, &epart_mv[0], &npart_mv[0])
    process_return(returnVal)
    return epart, npart, objval



cpdef NodeND(idx_t[::1] xadj,
             idx_t[::1] adjncy,
             idx_t[::1] vwgt=None,
             idx_t[::1] options=None):
    cdef:
        idx_t nvtxs = xadj.shape[0]-1
        np.ndarray[idx_t, ndim=1] perm = uninitialized(nvtxs, dtype=idx)
        np.ndarray[idx_t, ndim=1] iperm = uninitialized(nvtxs, dtype=idx)
        idx_t[::1] perm_mv = perm, iperm_mv = iperm
        idx_t *vwgtPtr
        idx_t *optionsPtr
        int returnVal

    optionsPtr = NULL if options is None else &options[0]
    vwgtPtr = NULL if vwgt is None else &vwgt[0]

    returnVal = METIS_NodeND(&nvtxs, &xadj[0], &adjncy[0],
                             vwgtPtr, optionsPtr,
                             &perm_mv[0], &iperm_mv[0])
    process_return(returnVal)
    return perm, iperm
