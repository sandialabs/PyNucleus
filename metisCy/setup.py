###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from pathlib import Path
from shutil import copy
try:
    from PyNucleus_packageTools import package
except ImportError as e:
    raise ImportError('\'PyNucleus_packageTools\' needs to be installed first.') from e

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
idx = int(idx)
real = int(real)
python_def = ''
decl = ''
if idx == 32:
    python_def += 'idx = np.int32\n'
    decl += 'ctypedef np.int32_t idx_t\n'
elif idx == 64:
    python_def += 'idx = np.int64\n'
    decl += 'ctypedef np.int64_t idx_t\n'
else:
    raise NotImplementedError()
if real == 32:
    python_def += 'real = np.float32\n'
    decl += 'ctypedef float real_t\n'
elif real == 64:
    python_def += 'real = np.float64\n'
    decl += 'ctypedef np.float64_t real_t\n'
else:
    raise NotImplementedError()
with open(p.folder+'metisTypesTemp.pxi', 'w') as f:
    f.write(python_def)
with open(p.folder+'metisTypesTemp_decl.pxi', 'w') as f:
    f.write(decl)
if not (p.hash_file(p.folder+'metisTypesTemp.pxi') ==
        p.hash_file(p.folder+'metisTypes.pxi')):
    copy(p.folder+'metisTypesTemp.pxi', p.folder+'metisTypes.pxi')
Path(p.folder+'metisTypesTemp.pxi').unlink()
if not (p.hash_file(p.folder+'metisTypesTemp_decl.pxi') ==
        p.hash_file(p.folder+'metisTypes_decl.pxi')):
    copy(p.folder+'metisTypesTemp_decl.pxi', p.folder+'metisTypes_decl.pxi')
Path(p.folder+'metisTypesTemp_decl.pxi').unlink()
p.loadConfig()
p.addPackageInclude('PyNucleus_base')

p.addExtension("metisCy",
               sources=[p.folder+"metisCy.pyx"],
               libraries=["metis"])
p.addExtension("parmetisCy",
               sources=[p.folder+"parmetisCy.pyx"],
               libraries=["parmetis", "metis"])

p.setup(description="Cython wrapper for METIS.",
        install_requires=['Cython>=0.29.32', 'numpy'])
