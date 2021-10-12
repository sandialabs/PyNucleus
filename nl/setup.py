###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


try:
    from PyNucleus_base.setupUtils import package
except ImportError as e:
    raise ImportError('\'PyNucleus_base\' needs to be installed first.') from e

p = package('PyNucleus_nl')

p.loadConfig(extra_config={'annotate': False})


p.addExtension("nonlocalLaplacianBase",
               sources=[p.folder+"nonlocalLaplacianBase.pyx"])
p.addExtension("nonlocalLaplacian",
               sources=[p.folder+"nonlocalLaplacian.pyx"])
p.addExtension("fractionalLaplacian1D",
               sources=[p.folder+"fractionalLaplacian1D.pyx"])
p.addExtension("fractionalLaplacian2D",
               sources=[p.folder+"fractionalLaplacian2D.pyx"])
p.addExtension("twoPointFunctions",
               sources=[p.folder+"twoPointFunctions.pyx"])
p.addExtension("interactionDomains",
               sources=[p.folder+"interactionDomains.pyx"])
p.addExtension("kernels2",
               sources=[p.folder+"kernels2.pyx"])
p.addExtension("fractionalOrders",
               sources=[p.folder+"fractionalOrders.pyx"])
p.addExtension("kernelsCy",
               sources=[p.folder+"kernelsCy.pyx",
                        p.folder+"kernels.cpp"],
               depends=[p.folder+"kernels.hpp"],
               language="c++")
p.addExtension("clusterMethodCy",
               sources=[p.folder+"clusterMethodCy.pyx"])

p.addExtension("nonlocalLaplacianND",
               sources=[p.folder+"nonlocalLaplacianND.pyx"])

p.setup(description="Nonlocal operator assembly",
        install_requires=['cython', 'numpy', 'scipy',
                          'mpi4py>=2.0.0',
                          'PyNucleus_base', 'PyNucleus_fem', 'PyNucleus_multilevelSolver'])
