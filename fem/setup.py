###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


try:
    from PyNucleus_packageTools import package
except ImportError as e:
    raise ImportError('\'PyNucleus_packageTools\' needs to be installed first.') from e
from PyNucleus_packageTools import fillTemplate
from pathlib import Path

p = package('PyNucleus_fem')
p.addOption('USE_METIS', 'use_metis', True, ['PyNucleus_metisCy'])

p.loadConfig(extra_config={'annotate': True})
p.addPackageInclude('PyNucleus_base')

print('Generating templates')
templates = [
    'distributed_operators_{SCALAR}.pxi', 'distributed_operators_decl_{SCALAR}.pxi',
    'vector_{SCALAR}.pxi', 'vector_decl_{SCALAR}.pxi'
]
replacementGroups = [[('{SCALAR}', 'REAL'),
                      ('{SCALAR_label}', ''),
                      ('{SCALAR_label_lc}', ''),
                      ('{SCALAR_label_lc_}', ''),
                      ('{IS_REAL}', 'True'),
                      ('{IS_COMPLEX}', 'False')],
                     [('{SCALAR}', 'COMPLEX'),
                      ('{SCALAR_label}', 'Complex'),
                      ('{SCALAR_label_lc}', 'complex'),
                      ('{SCALAR_label_lc_}', 'complex_'),
                      ('{IS_REAL}', 'False'),
                      ('{IS_COMPLEX}', 'True'),
                      # for some reason, complex cannot handle += etc
                      ('\s([^\s]+\[[^\]]*\])\s([\*\+-])=', ' \\1 = \\1 \\2'),
                      ('\s([^\s]+)\s([\*\+-])=', ' \\1 = \\1 \\2')]]
fillTemplate(Path(p.folder), templates, replacementGroups)

p.addExtension("meshCy",
               sources=[p.folder+"meshCy.pyx"])

p.addExtension("meshPartitioning",
               sources=[p.folder+"meshPartitioning.pyx"])
p.addExtension("functions",
               sources=[p.folder+"functions.pyx"])
p.addExtension("femCy",
               sources=[p.folder+"femCy.pyx"])
p.addExtension("repartitioner",
               sources=[p.folder+"repartitioner.pyx"])
p.addExtension("DoFMaps",
               sources=[p.folder+"DoFMaps.pyx"])
p.addExtension("quadrature",
               sources=[p.folder+"quadrature.pyx"])
p.addExtension("meshOverlaps",
               sources=[p.folder+"meshOverlaps.pyx"])
p.addExtension("algebraicOverlaps",
               sources=[p.folder+"algebraicOverlaps.pyx"])
p.addExtension("distributed_operators",
               sources=[p.folder+"distributed_operators.pyx"])
p.addExtension("boundaryLayerCy",
               sources=[p.folder+"boundaryLayerCy.pyx"])
p.addExtension("simplexMapper",
               sources=[p.folder+"simplexMapper.pyx"])
p.addExtension("splitting",
               sources=[p.folder+"splitting.pyx"])

p.setup(description="A finite element code.",
        install_requires=['Cython>=0.29.32', 'numpy', 'scipy', 'matplotlib', 'meshpy', 'modepy',
                          'mpi4py>=2.0.0',
                          'PyNucleus_base'])
