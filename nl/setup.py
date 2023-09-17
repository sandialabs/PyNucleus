###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from shutil import move
from os import remove
from importlib.metadata import version
from distutils.errors import CompileError

try:
    import cython
except ImportError as e:
    raise ImportError('PyNucleus requires \'Cython\'. Please install it.') from e

try:
    from PyNucleus_packageTools import package
except ImportError as e:
    raise ImportError('\'PyNucleus_packageTools\' needs to be installed first.') from e
from PyNucleus_packageTools import fillTemplate
from pathlib import Path

p = package('PyNucleus_nl')
try:
    cython.inline("""
    cdef extern from "malloc.h" nogil:
        int malloc_trim(size_t pad)
    """)
    have_malloc_h = True
except CompileError as e:
    print('malloc.h not found, error was \"{}\". Depending on the system, this might be normal.'.format(e))
    have_malloc_h = False
p.addOption('HAVE_MALLOC_H', 'have_malloc_h', have_malloc_h)
p.addOption(None, 'mask_size', 256)
cython_directives = {'initializedcheck': False,
                     'boundscheck': False,
                     'cdivision': True,
                     'wraparound': False}
if version('cython') >= '3.0':
    cython_directives['cpow'] = True
p.loadConfig(extra_config={'annotate': True,
                           'cythonDirectives': cython_directives})
p.addPackageInclude('PyNucleus_base')
p.addPackageInclude('PyNucleus_fem')
p.addPackageInclude('PyNucleus_multilevelSolver')


# process 'mask_size' option
with open(p.folder+'bitset.pxd.in', 'r') as f:
    lines = ''.join(f.readlines())
with open(p.folder+'bitset.pxd.temp', 'w') as f:
    f.write(lines.format(MASK_SIZE=p.config['mask_size']))
if p.hash_file(p.folder+'bitset.pxd.temp') != p.hash_file(p.folder+'bitset.pxd'):
    move(p.folder+'bitset.pxd.temp', p.folder+'bitset.pxd')
else:
    remove(p.folder+'bitset.pxd.temp')

print('Generating templates')
templates = [
    'twoPointFunctions_{SCALAR}.pxi', 'twoPointFunctions_decl_{SCALAR}.pxi',
    'nonlocalOperator_{SCALAR}.pxi', 'nonlocalOperator_decl_{SCALAR}.pxi',
    'nonlocalAssembly_{SCALAR}.pxi', 'nonlocalAssembly_decl_{SCALAR}.pxi',
]
replacementGroups = [[('{SCALAR}', 'REAL'),
                      ('{SCALAR_label}', ''),
                      ('{SCALAR_label_lc}', ''),
                      ('{SCALAR_label_lc_}', ''),
                      ('{IS_REAL}', 'True'),
                      ('{IS_COMPLEX}', 'False'),
                      ('{function_type}', 'function')],
                     [('{SCALAR}', 'COMPLEX'),
                      ('{SCALAR_label}', 'Complex'),
                      ('{SCALAR_label_lc}', 'complex'),
                      ('{SCALAR_label_lc_}', 'complex_'),
                      ('{IS_REAL}', 'False'),
                      ('{IS_COMPLEX}', 'True'),
                      ('{function_type}', 'complexFunction'),
                      # for some reason, complex cannot handle += etc
                      ('\s([^\s]+\[[^\]]*\])\s([\*\+-])=', ' \\1 = \\1 \\2'),
                      ('\s([^\s]+)\s([\*\+-])=', ' \\1 = \\1 \\2')]]
fillTemplate(Path(p.folder), templates, replacementGroups)


p.addExtension("bitset",
               sources=[p.folder+"bitset.pyx"],
               language='c++')
p.addExtension("nonlocalOperator",
               sources=[p.folder+"nonlocalOperator.pyx"],
               language='c++')
p.addExtension("nonlocalAssembly",
               sources=[p.folder+"nonlocalAssembly.pyx"],
               language='c++')
p.addExtension("fractionalLaplacian1D",
               sources=[p.folder+"fractionalLaplacian1D.pyx"],
               language='c++')
p.addExtension("fractionalLaplacian2D",
               sources=[p.folder+"fractionalLaplacian2D.pyx"],
               language='c++')
p.addExtension("twoPointFunctions",
               sources=[p.folder+"twoPointFunctions.pyx"])
p.addExtension("interactionDomains",
               sources=[p.folder+"interactionDomains.pyx"])
p.addExtension("kernelNormalization",
               sources=[p.folder+"kernelNormalization.pyx"])
p.addExtension("kernelsCy",
               sources=[p.folder+"kernelsCy.pyx"])
p.addExtension("fractionalOrders",
               sources=[p.folder+"fractionalOrders.pyx"])
p.addExtension("clusterMethodCy",
               sources=[p.folder+"clusterMethodCy.pyx"],
               language='c++')

p.setup(description="Nonlocal operator assembly",
        python_requires='>=3.10',
        install_requires=['Cython>=0.29.32', 'numpy', 'scipy',
                          'mpi4py>=2.0.0',
                          'PyNucleus_base', 'PyNucleus_fem', 'PyNucleus_multilevelSolver'])
