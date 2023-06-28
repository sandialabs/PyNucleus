###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import os
import importlib
from setuptools import setup
from pathlib import Path

requirements = ['numpy', 'Cython>=0.29.32']

# We do this dance, so that we can install everything in editable mode
# as well. Otherwise installing PyNucleus in editable mode replaces
# the subpackages with their non-editable installs.
lclDir = os.getcwd().replace('\\', '/')
for pkg, srcLocation in [
        # These are just in the same git repo.
        ('packageTools', 'file://localhost/{lclDir}/packageTools'.format(lclDir=lclDir)),
        ('base', 'file://localhost/{lclDir}/base'.format(lclDir=lclDir)),
        ('metisCy', 'file://localhost/{lclDir}/metisCy'.format(lclDir=lclDir)),
        ('fem', 'file://localhost/{lclDir}/fem'.format(lclDir=lclDir)),
        ('multilevelSolver', 'file://localhost/{lclDir}/multilevelSolver'.format(lclDir=lclDir)),
        ('nl', 'file://localhost/{lclDir}/nl'.format(lclDir=lclDir)),
]:
    fullPkgName = 'PyNucleus_'+pkg
    try:
        importlib.import_module(fullPkgName)
        requirements += [fullPkgName]
    except ImportError:
        requirements += ['{} @ {}'.format(fullPkgName, srcLocation)]

version = '0.0.0'
if Path('VERSION').exists():
    with open('VERSION', 'r') as f:
        for line in f.readlines():
            if not line[0].isnumeric():
                continue
            version = line
            break

setup(name='PyNucleus',
      version=version,
      packages=['PyNucleus'],
      data_files=[('drivers', [str(p) for p in Path('drivers').glob('*.py')])],
      description='A finite element code that specifically targets nonlocal operators.',
      long_description=''.join(open('README.rst').readlines()),
      long_description_content_type='text/x-rst',
      author="Christian Glusa",
      author_email='caglusa@sandia.gov',
      platforms='any',
      license='MIT',
      license_files=['LICENSE'],
      python_requires='>=3.10',
      install_requires=requirements,
      zip_safe=False)
