###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from . myTypes cimport INDEX_t, REAL_t, BOOL_t


cdef class Map:
    cdef:
        public INDEX_t[:, ::1] GID_PID
        dict gbl_to_lcl
        dict lcl_to_gbl
        dict localNumElements
        INDEX_t[::1] lcl_to_gbl_offsets, lcl_to_gbl_array

    cpdef INDEX_t getGlobalElement(self, INDEX_t pid, INDEX_t lid)
    cpdef INDEX_t getLocalElement(self, INDEX_t pid, INDEX_t gid)
    cpdef INDEX_t getLocalNumElements(self, INDEX_t pid)
    cpdef INDEX_t getGlobalNumElements(self)
    cpdef INDEX_t numRanks(self)


cdef class DistributedMap:
    cdef:
        public comm
        public INDEX_t[:, ::1] GIDs
        dict GID2LID
    cpdef INDEX_t getGlobalElement(self, INDEX_t lid)
    cpdef INDEX_t getOwner(self, INDEX_t lid)
    cpdef BOOL_t isOwned(self, INDEX_t lid)
    cpdef INDEX_t getLocalElement(self, INDEX_t gid)
    cpdef INDEX_t getLocalNumElements(self)
    cpdef INDEX_t getGlobalNumElements(self)
    cpdef INDEX_t numRanks(self)


cdef class Import:
    cdef:
        DistributedMap ovMap
        DistributedMap oneToOneMap
        INDEX_t[::1] receiveLIDs
        INDEX_t[::1] sendLIDs
        INDEX_t[::1] countsReceive
        INDEX_t[::1] countsSends
        INDEX_t[::1] offsetsReceives
        INDEX_t[::1] offsetsSends
