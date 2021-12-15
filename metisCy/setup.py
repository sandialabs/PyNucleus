###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from subprocess import Popen, PIPE, STDOUT
import re
try:
    from PyNucleus_base.setupUtils import package
except ImportError as e:
    raise ImportError('\'PyNucleus_base\' needs to be installed first.') from e

try:
    import cython
except ImportError as e:
    raise ImportError('PyNucleus requires \'Cython\'. Please install it.') from e


p = package('PyNucleus_metisCy')

p.parseConfig()
idx, real = cython.inline("""
cdef extern from "metis.h":
    int IDXTYPEWIDTH
    int REALTYPEWIDTH
return IDXTYPEWIDTH, REALTYPEWIDTH""",
                          cython_include_dirs=p.config['includeDirs'])

p.addOption('IDXTYPEWIDTH', 'METIS_idx_width', idx)
p.addOption('REALTYPEWIDTH', 'METIS_real_width', real)
p.loadConfig()

p.addExtension("metisCy",
               sources=[p.folder+"metisCy.pyx"],
               libraries=["metis"])
p.addExtension("parmetisCy",
               sources=[p.folder+"parmetisCy.pyx"],
               libraries=["parmetis", "metis"])

p.setup(description="Cython wrapper for METIS.",
        install_requires=['cython', 'numpy'])
