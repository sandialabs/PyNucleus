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


p = package('PyNucleus_metisCy')

######################################################################
# Attempt to detect the types used for indices and reals in Metis
cmd = "echo '#include <metis.h>' | cpp -H -o /dev/null 2>&1 | head -n1"
proc = Popen(cmd,
             stdout=PIPE, stderr=STDOUT,
             shell=True,
             universal_newlines=True)
out, _ = proc.communicate()
metisHeader = out[2:-1]

idx = re.compile(r'\s*#define\s*IDXTYPEWIDTH\s*([0-9]+)')
real = re.compile(r'\s*#define\s*REALTYPEWIDTH\s*([0-9]+)')

idxDefault = 32
realDefault = 32
with open(metisHeader, 'r') as f:
    for line in f:
        match = idx.match(line)
        if match:
            idxDefault = int(match.group(1))
        match = real.match(line)
        if match:
            realDefault = int(match.group(1))

p.addOption('IDXTYPEWIDTH', 'METIS_idx_width', idxDefault)
p.addOption('REALTYPEWIDTH', 'METIS_real_width', realDefault)
p.loadConfig()

p.addExtension("metisCy",
               sources=[p.folder+"metisCy.pyx"],
               libraries=["metis"])
p.addExtension("parmetisCy",
               sources=[p.folder+"parmetisCy.pyx"],
               libraries=["parmetis", "metis"])

p.setup(description="Cython wrapper for METIS.",
        install_requires=['cython', 'numpy'])
