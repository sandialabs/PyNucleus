###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from pathlib import Path
try:
    from PyNucleus_packageTools import package
except ImportError as e:
    raise ImportError('\'PyNucleus_packageTools\' needs to be installed first.') from e
from PyNucleus_packageTools import fillTemplate


p = package('PyNucleus_multilevelSolver')

p.loadConfig()
p.addPackageInclude('PyNucleus_base')
p.addPackageInclude('PyNucleus_fem')

print('Generating templates')
templates = [
    'smoothers_{SCALAR}.pxi', 'smoothers_decl_{SCALAR}.pxi',
    'coarseSolvers_{SCALAR}.pxi', 'coarseSolvers_decl_{SCALAR}.pxi',
    'multigrid_{SCALAR}.pxi', 'multigrid_decl_{SCALAR}.pxi'
]
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

p.addExtension("smoothers",
               sources=[p.folder+"smoothers.pyx"])
p.addExtension("restrictionProlongation",
               sources=[p.folder+"restrictionProlongation.pyx"])
p.addExtension("multigrid",
               sources=[p.folder+"multigrid.pyx"])

p.addExtension("coarseSolvers",
               sources=[p.folder+"coarseSolvers.pyx"])


p.setup(description="An implementation of geometric multigrid",
        install_requires=['Cython>=0.29.32', 'mpi4py>=2.0.0', 'numpy', 'scipy',
                          'tabulate', 'PyNucleus_fem', 'PyNucleus_metisCy'])
