###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


"""
A Cython interface to METIS and ParMETIS.

http://glaros.dtc.umn.edu/gkhome/metis/metis/overview
http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview

"""
from . metisCy import (PartGraphRecursive,
                       PartGraphKway,
                       PartMeshNodal,
                       PartMeshDual,
                       NodeND,
                       SetDefaultOptions)

from . metisCy import NOPTIONS

# Options codes
from . metisCy import (OPTION_PTYPE,
                       OPTION_OBJTYPE,
                       OPTION_CTYPE,
                       OPTION_IPTYPE,
                       OPTION_RTYPE,
                       OPTION_DBGLVL,
                       OPTION_NITER,
                       OPTION_NCUTS,
                       OPTION_SEED,
                       OPTION_NO2HOP,
                       OPTION_MINCONN,
                       OPTION_CONTIG,
                       OPTION_COMPRESS,
                       OPTION_CCORDER,
                       OPTION_PFACTOR,
                       OPTION_NSEPS,
                       OPTION_UFACTOR,
                       OPTION_NUMBERING)

# Partitioning Schemes
from . metisCy import (PTYPE_RB,
                       PTYPE_KWAY)

# Graph types for meshes
from . metisCy import (GTYPE_DUAL,
                       GTYPE_NODAL)

# Coarsening Schemes
from . metisCy import (CTYPE_RM,
                       CTYPE_SHEM)

# Initial partitioning schemes
from . metisCy import (IPTYPE_GROW,
                       IPTYPE_RANDOM,
                       IPTYPE_EDGE,
                       IPTYPE_NODE,
                       IPTYPE_METISRB)

# Refinement schemes
from . metisCy import (RTYPE_FM,
                       RTYPE_GREEDY,
                       RTYPE_SEP2SIDED,
                       RTYPE_SEP1SIDED)

# Debug Levels
from . metisCy import (DBG_INFO,
                       DBG_TIME,
                       DBG_COARSEN,
                       DBG_REFINE,
                       DBG_IPART,
                       DBG_MOVEINFO,
                       DBG_SEPINFO,
                       DBG_CONNINFO,
                       DBG_CONTIGINFO,
                       DBG_MEMORY)

# Types of objectives
from . metisCy import (OPTION_OBJTYPE,
                       OBJTYPE_CUT,
                       OBJTYPE_NODE,
                       OBJTYPE_VOL)

from . import _version
__version__ = _version.get_versions()['version']
