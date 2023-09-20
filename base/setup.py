###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

from shutil import copy
from pathlib import Path
from distutils.errors import CompileError

try:
    import cython
except ImportError as e:
    raise ImportError('PyNucleus requires \'Cython\'. Please install it.') from e

try:
    import numpy
except ImportError as e:
    raise ImportError('PyNucleus requires \'numpy\'. Please install it.') from e

try:
    import scipy
except ImportError as e:
    raise ImportError('PyNucleus requires \'scipy\'. Please install it.') from e

try:
    import mpi4py
except ImportError as e:
    raise ImportError('PyNucleus requires \'mpi4py\'. Please install it.') from e

try:
    from PyNucleus_packageTools import package, fillTemplate
except ImportError as e:
    raise ImportError('\'PyNucleus_packageTools\' needs to be installed first.') from e

p = package('PyNucleus_base')
p.addOption(None, 'useBLAS', True)
p.addOption(None, 'useMKL', False)
p.addOption(None, 'use_cholmod', True, ['scikit-sparse'])
p.addOption(None, 'use_pyamg', False, ['pyamg'])
p.addOption(None, 'use_pypardiso', False, ['pypardiso'])
p.addOption(None, 'mklLibrary', 'mkl_rt')
p.addOption(None, 'useMKL_trisolve', False)
p.addOption(None, 'fillUninitialized', True)

try:
    cython.inline("""
    cdef extern from "malloc.h" nogil:
        int malloc_trim(size_t pad)
    """)
    have_malloc_h = True
    if not (p.hash_file(p.folder+'malloc_linux.pxi') ==
            p.hash_file(p.folder+'malloc.pxi')):
        copy(p.folder+'malloc_linux.pxi', p.folder+'malloc.pxi')
except CompileError as e:
    print('malloc.h not found, error was \"{}\". Depending on the system, this might be normal.'.format(e))
    if not (p.hash_file(p.folder+'malloc_mac.pxi') ==
            p.hash_file(p.folder+'malloc.pxi')):
        copy(p.folder+'malloc_mac.pxi', p.folder+'malloc.pxi')
p.loadConfig(extra_config={'annotate': True,
                           'cythonDirectives': {'initializedcheck': False,
                                                'boundscheck': False,
                                                'cdivision': True,
                                                'wraparound': False}})

# set up variable types
if cython.inline('return sizeof(a)', a=1) == 4:
    archDetected = '32bit'
else:
    archDetected = '64bit'
print('Arch detected: {}'.format(archDetected))
if p.config['arch'] == 'detect':
    p.config['arch'] = archDetected
if p.config['arch'] == '32bit':
    if not (p.hash_file(p.folder+'myTypes32.pyx') ==
            p.hash_file(p.folder+'myTypes.pyx') and
            p.hash_file(p.folder+'myTypes32.pxd') ==
            p.hash_file(p.folder+'myTypes.pxd') and
            p.hash_file(p.folder+'myTypes32.h') ==
            p.hash_file(p.folder+'myTypes.h')):
        print('Configuring for 32 bit system')
        copy(p.folder+'myTypes32.pyx', p.folder+'myTypes.pyx')
        copy(p.folder+'myTypes32.pxd', p.folder+'myTypes.pxd')
        copy(p.folder+'myTypes32.h', p.folder+'myTypes.h')
elif p.config['arch'] == '64bit':
    if not (p.hash_file(p.folder+'myTypes64.pyx') ==
            p.hash_file(p.folder+'myTypes.pyx') and
            p.hash_file(p.folder+'myTypes64.pxd') ==
            p.hash_file(p.folder+'myTypes.pxd') and
            p.hash_file(p.folder+'myTypes64.h') ==
            p.hash_file(p.folder+'myTypes.h')):
        print('Configuring for 64 bit system')
        copy(p.folder+'myTypes64.pyx', p.folder+'myTypes.pyx')
        copy(p.folder+'myTypes64.pxd', p.folder+'myTypes.pxd')
        copy(p.folder+'myTypes64.h', p.folder+'myTypes.h')
else:
    raise NotImplementedError()

p.addExtension("myTypes",
               sources=[p.folder+"myTypes.pyx"])
p.addExtension("blas",
               sources=[p.folder+"blas.pyx"],
               libraries=[p.config['mklLibrary']] if p.config['useMKL'] else [])
p.addExtension("performanceLogger",
               sources=[p.folder+"performanceLogger.pyx"])
p.addExtension("utilsCy",
               sources=[p.folder+"utilsCy.pyx"])


print('Filling templates')

templates = ['LinearOperator_{SCALAR}.pxi', 'LinearOperator_decl_{SCALAR}.pxi',
             'LinearOperatorWrapper_{SCALAR}.pxi', 'LinearOperatorWrapper_decl_{SCALAR}.pxi',
             'DenseLinearOperator_{SCALAR}.pxi', 'DenseLinearOperator_decl_{SCALAR}.pxi',
             'CSR_LinearOperator_{SCALAR}.pxi', 'CSR_LinearOperator_decl_{SCALAR}.pxi',
             'SSS_LinearOperator_{SCALAR}.pxi', 'SSS_LinearOperator_decl_{SCALAR}.pxi',
             'DiagonalLinearOperator_{SCALAR}.pxi', 'DiagonalLinearOperator_decl_{SCALAR}.pxi',
             'IJOperator_{SCALAR}.pxi', 'IJOperator_decl_{SCALAR}.pxi',
             'SchurComplement_{SCALAR}.pxi', 'SchurComplement_decl_{SCALAR}.pxi',]
replacementGroups = [[('{SCALAR}', 'REAL'),
                      ('{SCALAR_label}', ''),
                      ('{SCALAR_label_lc}', ''),
                      ('{SCALAR_label_lc_}', '')],
                     [('{SCALAR}', 'COMPLEX'),
                      ('{SCALAR_label}', 'Complex'),
                      ('{SCALAR_label_lc}', 'complex'),
                      ('{SCALAR_label_lc_}', 'complex_'),
                      # for some reason, complex cannot handle += etc
                      ('\s([^\s]+\[[^\]]*\])\s([\*\+-])=', ' \\1 = \\1 \\2'),
                      ('\s([^\s]+)\s([\*\+-])=', ' \\1 = \\1 \\2')]]
fillTemplate(Path(p.folder), templates, replacementGroups)

templates = [
    'tupleDict_{VALUE}.pxi', 'tupleDict_decl_{VALUE}.pxi'
]
replacementGroups = [[('{VALUE}', 'INDEX'),
                      ('{VALUE_dtype}', 'INDEX'),
                      ('{VALUE_t}', 'INDEX_t'),
                      ('{LENGTH_dtype}', 'np.uint8'),
                      ('{LENGTH_t}', 'np.uint8_t'),
                      ('{INVALID}', str(numpy.iinfo(numpy.int32).max))]]
fillTemplate(Path(p.folder), templates, replacementGroups)


# allocation
p.conditionalCopy(p.folder+'allocation.pxi', p.config['fillUninitialized'], p.folder+'opt_true_allocation.pxi', p.folder+'opt_false_allocation.pxi')
p.conditionalCopy(p.folder+'blas_routines.pxi', p.config['useBLAS'], p.folder+'opt_true_blas.pxi', p.folder+'opt_false_blas.pxi')
p.conditionalCopy(p.folder+'mkl_routines.pxi', p.config['useMKL'], p.folder+'opt_true_mkl.pxi', p.folder+'opt_false_mkl.pxi')

# solvers
p.conditionalCopy(p.folder+'solver_chol.pxi', p.config['use_cholmod'], p.folder+'opt_true_solver_cholmod.pxi', p.folder+'opt_false_solver_cholmod.pxi')
p.conditionalCopy(p.folder+'solver_ichol.pxi', p.config['useMKL_trisolve'], p.folder+'opt_true_mkl_trisolve.pxi', p.folder+'opt_false_mkl_trisolve.pxi')
p.conditionalCopy(p.folder+'solver_pypardiso.pxi', p.config['use_pypardiso'], p.folder+'opt_true_solver_pypardiso.pxi', None)
p.conditionalCopy(p.folder+'solver_pypardiso_decl.pxi', p.config['use_pypardiso'], p.folder+'opt_true_solver_pypardiso_decl.pxi', None)
p.conditionalCopy(p.folder+'solver_pyamg.pxi', p.config['use_pyamg'], p.folder+'opt_true_solver_pyamg.pxi', None)
p.conditionalCopy(p.folder+'solver_pyamg_decl.pxi', p.config['use_pyamg'], p.folder+'opt_true_solver_pyamg_decl.pxi', None)

p.addExtension("linear_operators",
               sources=[p.folder+"linear_operators.pyx"])
p.addExtension("sparseGraph",
               sources=[p.folder+"sparseGraph.pyx"])
p.addExtension("solvers",
               sources=[p.folder+"solvers.pyx"])
p.addExtension("linalg",
               sources=[p.folder+"linalg.pyx"],
               libraries=[p.config['mklLibrary']] if p.config['useMKL'] else [])
p.addExtension("sparsityPattern",
               sources=[p.folder+"sparsityPattern.pyx"])
p.addExtension("convergence",
               sources=[p.folder+"convergence.pyx"])
p.addExtension("ip_norm",
               sources=[p.folder+"ip_norm.pyx"])
p.addExtension("intTuple",
               sources=[p.folder+"intTuple.pyx"])
p.addExtension("tupleDict",
               sources=[p.folder+"tupleDict.pyx"])
p.addExtension("SchurComplement",
               sources=[p.folder+"SchurComplement.pyx"])
p.addExtension("io",
               sources=[p.folder+"io.pyx"])


p.setup(description="Helper functions for PyNucleus.",
        install_requires=['numpy', 'scipy', 'Cython>=0.29.32', 'mpi4py>=2.0.0', 'matplotlib', 'tabulate', 'h5py', 'pyyaml', 'psutil'],
        )
