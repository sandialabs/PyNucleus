#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


from sys import argv
from subprocess import Popen, PIPE
from pathlib import Path
from datetime import datetime

assert len(argv) == 3

moduleFile = argv[1]
installLocation = Path(argv[2]).resolve()

proc = Popen('git describe --always --dirty --abbrev=40', shell=True, stdout=PIPE)
proc.wait()
sha = proc.stdout.read()
sha = sha[:-1].decode('utf-8')

with open(moduleFile, 'w') as f:
    f.write('whatis("Version: {}")\n'.format(sha))
    f.write('whatis("BuildDate: {}")\n'.format(datetime.now()))
    f.write('\n')
    f.write('prepend_path(\"PYTHONPATH\", \"{}\")\n'.format(installLocation))
