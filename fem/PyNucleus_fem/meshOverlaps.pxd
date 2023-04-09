###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from mpi4py cimport MPI
from PyNucleus_base.myTypes cimport INDEX_t, REAL_t, ENCODE_t, BOOL_t


cdef class sharedMesh:
    cdef:
        public INDEX_t mySubdomainNo, otherSubdomainNo, dim
        public INDEX_t[:, ::1] vertices, edges, faces
        public INDEX_t[::1] cells


cdef class meshInterface(sharedMesh):
    pass


cdef class sharedMeshManager:
    cdef:
        public INDEX_t numSubdomains
        public dict sharedMeshes
        public list requests
        public MPI.Comm comm
        public dict _rank2subdomain
        public dict _subdomain2rank
    cdef inline INDEX_t rank2subdomain(self, INDEX_t rank)
    cdef inline INDEX_t subdomain2rank(self, INDEX_t subdomainNo)


cdef class interfaceManager(sharedMeshManager):
    pass


cdef class meshOverlap(sharedMesh):
    pass

cdef class overlapManager(sharedMeshManager):
    pass
