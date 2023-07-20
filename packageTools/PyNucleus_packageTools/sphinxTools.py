###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


class codeRegion:
    def __init__(self, mgr, label, isFinalTarget, codeTarget=''):
        self.mgr = mgr
        self.isTarget = isFinalTarget
        self.codeTarget = codeTarget

    def __enter__(self):
        from inspect import getframeinfo, stack
        import sys
        from io import StringIO
        caller = getframeinfo(stack()[1][0])
        self.startLine = caller.lineno
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()

        return self

    def __exit__(self, type, value, traceback):
        from inspect import getframeinfo, stack
        import sys
        import matplotlib.pyplot as plt

        sys.stdout = self._stdout

        caller = getframeinfo(stack()[1][0])

        if hasattr(caller, 'positions'):
            self.endLine = caller.positions.end_lineno
        else:
            self.endLine = caller.lineno

        if self.codeTarget != '':
            with open(caller.filename, 'r') as f:
                lines = f.readlines()
            from textwrap import dedent
            code = dedent(''.join(lines[self.startLine:self.endLine]))
            code += '\n'
            with open(self.codeTarget, 'a') as f:
                f.writelines(code)

        if self.isTarget:
            print(self._stringio.getvalue())


class codeRegionManager:
    def __init__(self):
        from argparse import ArgumentParser

        parser = ArgumentParser()
        parser.add_argument('--export', help='filename for code export')
        parser.add_argument('--finalTarget', default='', help='code up to this code region should be executed')
        args = parser.parse_args()

        if args.export is not None:
            self.codeTarget = args.export
            from pathlib import Path
            try:
                Path(self.codeTarget).unlink()
            except FileNotFoundError:
                pass
        else:
            self.codeTarget = ''
        self.finalTarget = args.finalTarget
        self.finalTargetHit = False

        if self.finalTarget == '' and self.codeTarget != '':
            with open(self.codeTarget, 'w') as f:
                f.write('#!/usr/bin/env python3\n')

    def add(self, label, onlyIfFinal=False):
        if self.finalTarget == label:
            self.finalTargetHit = True
        else:
            if self.finalTargetHit:
                exit(0)
        return codeRegion(self,
                          label,
                          isFinalTarget=(self.finalTarget == label) or (self.finalTarget == ''),
                          codeTarget=self.codeTarget if (not onlyIfFinal or self.finalTargetHit or self.finalTarget == '') else '')
