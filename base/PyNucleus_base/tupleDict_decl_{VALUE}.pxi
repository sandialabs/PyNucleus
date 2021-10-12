###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


cdef class tupleDict{VALUE}:
    cdef:
        INDEX_t ** indexL
        {VALUE_t} ** vals
        {LENGTH_t}[::1] counts
        {LENGTH_t} initial_length
        {LENGTH_t} length_inc
        {LENGTH_t}[::1] lengths
        INDEX_t num_dofs
        readonly INDEX_t nnz
        BOOL_t deleteHits, logicalAndHits
        INDEX_t i, jj
        public {VALUE_t} invalid
        {LENGTH_t} invalidIndex
        {LENGTH_t} index
    cdef inline BOOL_t findIndex(self, INDEX_t I, INDEX_t J)
    cdef inline void increaseSize(self, INDEX_t I, {LENGTH_t} increment)
    cdef {VALUE_t} enterValue(self, const INDEX_t[::1] e, {VALUE_t} val)
    cdef {VALUE_t} removeValue(self, const INDEX_t[::1] e)
    cpdef {VALUE_t} enterValue_py(self, const INDEX_t[::1] e, {VALUE_t} val)
    cpdef {VALUE_t} removeValue_py(self, const INDEX_t[::1] e)
    cdef {VALUE_t} getValue(self, const INDEX_t[::1] e)
    cdef void startIter(self)
    cdef BOOL_t next(self, INDEX_t[::1] e, {VALUE_t} * val)
    cdef tuple getData(self)
    cpdef void merge(self, tupleDict{VALUE} other)
    cpdef void mergeData(self, {LENGTH_t}[::1] counts, INDEX_t[::1] indexL, {VALUE_t}[::1] vals)
    cdef INDEX_t getSizeInBytes(self)
