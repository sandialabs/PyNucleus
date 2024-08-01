/////////////////////////////////////////////////////////////////////////////////////
// Copyright 2021 National Technology & Engineering Solutions of Sandia,           //
// LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           //
// U.S. Government retains certain rights in this software.                        //
// If you want to use this code, please refer to the README.rst and LICENSE files. //
/////////////////////////////////////////////////////////////////////////////////////

// Matches demo/wrap-cython/mpi-compat.h from mpi4py

#ifndef MPI_COMPAT_H
#define MPI_COMPAT_H

#include <mpi.h>

#if (MPI_VERSION < 3) && !defined(PyMPI_HAVE_MPI_Message)
typedef void *PyMPI_MPI_Message;
#define MPI_Message PyMPI_MPI_Message
#endif

#if (MPI_VERSION < 4) && !defined(PyMPI_HAVE_MPI_Session)
typedef void *PyMPI_MPI_Session;
#define MPI_Session PyMPI_MPI_Session
#endif

#endif
