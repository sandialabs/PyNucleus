###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../packageTools'))

from PyNucleus_packageTools import package

p = package('PyNucleus_packageTools')
p.loadConfig()

p.setup(description='tools for setting up Python packages',
        install_requires=['numpy', 'scipy', 'matplotlib', 'Cython>=0.29.32', 'mpi4py>=2.0.0', 'tabulate', 'PyYAML', 'H5py', 'modepy', 'meshpy'])
