###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


import os
import multiprocessing
import re
from copy import copy
from pathlib import Path
from collections.abc import Mapping


###############################################################################
# from
# https://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils

from distutils.ccompiler import CCompiler
from distutils.command.build_ext import build_ext
try:
    from concurrent.futures import ThreadPoolExecutor as Pool
except ImportError:
    from multiprocessing.pool import ThreadPool as LegacyPool

    # To ensure the with statement works. Required for some older 2.7.x releases
    class Pool(LegacyPool):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
            self.join()


def build_extensions_multithreaded(self):
    """Function to monkey-patch
    distutils.command.build_ext.build_ext.build_extensions

    """
    self.check_extensions_list(self.extensions)

    try:
        num_jobs = os.cpu_count()
    except AttributeError:
        num_jobs = multiprocessing.cpu_count()

    with Pool(num_jobs) as pool:
        pool.map(self.build_extension, self.extensions)


def compile_multithreaded(
        self, sources, output_dir=None, macros=None, include_dirs=None,
        debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    """Function to monkey-patch distutils.ccompiler.CCompiler"""
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
    )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    for obj in objects:
        try:
            src, ext = build[obj]
        except KeyError:
            continue
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # Return *all* object filenames, not just the ones we just built.
    return objects


###############################################################################


def update(d, u):
    d = d.copy()
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class package:
    def __init__(self, name, namespace=''):
        import multiprocessing

        self.name = name
        self.namespace = namespace
        if self.namespace != '':
            self.full_name = self.namespace+'.'+self.name
            self.folder = self.namespace+'/'+self.name + '/'
        else:
            self.full_name = self.name
            self.folder = self.name + '/'
        self.configLoaded = False
        self.extensions = []
        self.options = []
        self.defaults = {'compileArgs': ['-O3', '-pipe', '-Wno-cpp'],  # '-fdiagnostics-color=always'
                         'linkArgs': ['-O3', '-pipe'],
                         'includeDirs': [],
                         'macros': [],
                         'use_ccache': True,
                         'setupProfiling': False,
                         'cythonDirectives': {'language_level': '2',
                                              'embedsignature': True,
                                              'binding': True},
                         'annotate': False,
                         'arch': 'detect',
                         'compiler_c': 'detect',
                         'compiler_c++': 'detect',
                         'mpi': 'openmpi',
                         'threads': 1}
        self.addOption('USE_OPENMP', 'useOpenMP', False)
        self.addOption(None, 'gitSHA', self.getGitSHA())

    def addOption(self, optionCy, optionPy, default, pkgDependencies=[]):
        if isinstance(pkgDependencies, str):
            pkgDependencies = [pkgDependencies]
        self.options.append((optionCy, optionPy, pkgDependencies))
        self.defaults[optionPy] = default

    def addPackageInclude(self, packageName):
        assert self.configLoaded
        try:
            import importlib

            module = importlib.import_module(packageName)
            self.config['includeDirs'].append(str(Path(module.__file__).parent))
        except ImportError as e:
            raise ImportError('\'{}\' needs to be installed first.'.format(packageName)) from e

    def parseConfig(self, filename=None, extra_config={}):
        if filename is None:
            filename = os.getcwd()+'/../config.yaml'
        defaults = self.defaults
        if Path(filename).exists():
            import yaml
            config = yaml.load(open(filename, 'r'), Loader=yaml.FullLoader)
            self.config = update(defaults, config)
        else:
            self.config = defaults
        self.config = update(self.config, extra_config)
        if 'PYNUCLEUS_BUILD_PARALLELISM' in os.environ:
            try:
                self.config['threads'] = int(os.environ['PYNUCLEUS_BUILD_PARALLELISM'])
            except:
                pass
        self.configLoaded = True

    def loadConfig(self, filename=None, extra_config={}):
        if filename is None:
            filename = os.getcwd()+'/../config.yaml'
        self.parseConfig(filename, extra_config)
        self.setCompiler()
        self.setInclude()
        self.setProfiling()
        self.setupOpenMP()
        self.setOptions()

    def setCompiler(self):
        assert self.configLoaded
        assert self.config['mpi'] in ('openmpi', 'generic'), "Currently, only OpenMPI is properly supported. Setting 'mpi'='generic' might work for other implementations."
        # set compiler
        if self.config['compiler_c'] == 'detect':
            if 'MPICC' in os.environ:
                self.config['compiler_c'] = os.environ['MPICC']
            else:
                try:
                    import mpi4py
                    self.config['compiler_c'] = str(mpi4py.get_config()['mpicc'])
                except:
                    self.config['compiler_c'] = 'mpicc'
        os.environ['CC'] = self.config['compiler_c']
        if self.config['compiler_c++'] == 'detect':
            if 'MPICXX' in os.environ:
                self.config['compiler_c++'] = os.environ['MPICXX']
            else:
                try:
                    import mpi4py
                    self.config['compiler_c++'] = str(mpi4py.get_config()['mpicxx'])
                except:
                    self.config['compiler_c++'] = 'mpicxx'
        os.environ['CXX'] = self.config['compiler_c++']
        from shutil import which
        from subprocess import Popen, PIPE
        if self.config['use_ccache'] and which('ccache') is not None:
            if self.config['mpi'] == 'openmpi':
                out, err = Popen([self.config['compiler_c'], '--showme:command'], stdout=PIPE, stderr=PIPE).communicate()
                assert len(err) == 0, err
                underlying_c_compiler = out.decode()[:-1]
                print(underlying_c_compiler)
                os.environ['OMPI_CC'] = 'ccache {}'.format(underlying_c_compiler)

                out, err = Popen([self.config['compiler_c++'], '--showme:command'], stdout=PIPE, stderr=PIPE).communicate()
                assert len(err) == 0, err
                underlying_cxx_compiler = out.decode()[:-1]
                os.environ['OMPI_CXX'] = 'ccache {}'.format(underlying_cxx_compiler)

        if self.config['mpi'] == 'openmpi':
            out, err = Popen([self.config['compiler_c'], '--version'], stdout=PIPE, stderr=PIPE).communicate()
            assert len(err) == 0, err
            print('C compiler \'{}\' description:\n{}\n'.format(self.config['compiler_c'], out.decode()[:-1]))

            out, err = Popen([self.config['compiler_c++'], '--version'], stdout=PIPE, stderr=PIPE).communicate()
            assert len(err) == 0, err
            print('C++ compiler \'{}\' description:\n{}\n'.format(self.config['compiler_c'], out.decode()[:-1]))

    def setInclude(self):
        assert self.configLoaded
        try:
            import numpy
            self.config['includeDirs'] += [numpy.get_include()]
        except ImportError:
            pass
        try:
            import mpi4py
            self.config['includeDirs'] += [mpi4py.get_include()]
        except ImportError:
            pass

    def setupOpenMP(self):
        assert self.configLoaded
        if self.config['useOpenMP']:
            self.config['compileArgs'] += ['-fopenmp']
            self.config['linkArgs'] += ['-fopenmp']
            self.config['macros'] += [('USE_OPENMP', 1)]
        else:
            self.config['macros'] += [('USE_OPENMP', 0)]

    def setProfiling(self):
        assert self.configLoaded
        # set up profiling
        if self.config['setupProfiling']:
            print('Building with profiling')
            self.config['cythonDirectives']['linetrace'] = True
            self.config['cythonDirectives']['binding'] = True
            self.config['macros'] += [('CYTHON_TRACE', '1')]

    def updateFile(self, filename, content):
        try:
            with open(filename, 'r') as f:
                contentOld = f.read(-1)
        except:
            contentOld = ''
        if content != contentOld:
            with open(filename, 'w') as f:
                f.write(content)

    def setOptions(self):
        assert self.configLoaded
        cy = ''
        py = ''
        for optionCy, optionPy, _ in self.options:
            if isinstance(self.config[optionPy], str):
                value = '\"{}\"'.format(self.config[optionPy])
            else:
                value = self.config[optionPy]
            if optionCy is not None:
                cy += 'DEF {} = {}\n'.format(optionCy, value)
            if optionPy is not None:
                py += '{} = {}\n'.format(optionPy, value)
        self.updateFile(self.folder+'/config.pxi', cy)
        self.updateFile(self.folder+'/config.py', py)

    def addExtension(self, ext_name, **kwargs):
        assert self.configLoaded
        from setuptools import Extension
        if 'extra_compile_args' in kwargs:
            kwargs['extra_compile_args'] += self.config['compileArgs']
        else:
            kwargs['extra_compile_args'] = self.config['compileArgs']
        kwargs['extra_link_args'] = self.config['linkArgs']
        kwargs['define_macros'] = self.config['macros']
        kwargs['include_dirs'] = self.config['includeDirs']
        self.extensions.append(Extension(self.full_name+'.'+ext_name, **kwargs))

    def setup(self, **kwargs):
        assert self.configLoaded
        from setuptools import setup

        if 'install_requires' not in kwargs:
            kwargs['install_requires'] = []
        for _, optionPy, pkgDependencies in self.options:
            if self.config[optionPy]:
                kwargs['install_requires'] += pkgDependencies
        for includeDir in self.config['includeDirs']:
            if not Path(includeDir).exists():
                import warnings
                warnings.warn('The include path \'{}\' does not exist.'.format(includeDir))

        from sys import platform
        if platform == 'darwin':
            warnings.warn('Multithreaded builds currently do not work on MacOS. Falling back to serial build.')
            self.config['threads'] = 1

        if self.config['threads'] > 1:
            build_ext.build_extensions = build_extensions_multithreaded
            CCompiler.compile = compile_multithreaded

        if len(self.extensions) > 0:
            from Cython.Build import cythonize
            kwargs['ext_modules'] = cythonize(self.extensions,
                                              include_path=self.config['includeDirs'],
                                              compiler_directives=self.config['cythonDirectives'],
                                              annotate=self.config['annotate'],
                                              nthreads=self.config['threads'])
        kwargs['name'] = self.name
        version = '0.0.0'
        possibleVersionFiles = [Path('../VERSION'),
                                Path('VERSION')]
        for versionFile in possibleVersionFiles:
            if versionFile.exists():
                with open(versionFile, 'r') as f:
                    for line in f.readlines():
                        if not line[0].isnumeric():
                            continue
                        version = line
                        break
                break

        kwargs['version'] = version
        # kwargs['version'] = self.getGitDate()

        if self.namespace != '':
            kwargs['namespace_packages'] = [self.namespace]
        if self.namespace != '':
            from setuptools import find_namespace_packages
            kwargs['packages'] = find_namespace_packages(include=[self.namespace+'.*'])
        else:
            kwargs['packages'] = [self.full_name]
        kwargs['package_data'] = {self.name: ['*.pxd', '*_decl*.pxi', '*config.pxi', '*.h']}
        kwargs['zip_safe'] = False
        if 'author' not in kwargs:
            kwargs['author'] = 'Christian Glusa'
        if 'author_email' not in kwargs:
            kwargs['author_email'] = 'caglusa@sandia.gov'
        if 'platforms' not in kwargs:
            kwargs['platforms'] = 'any'
        if 'license' not in kwargs:
            kwargs['license'] = 'MIT'
        if 'license_files' not in kwargs:
            kwargs['license_files'] = ['../LICENSE']
        setup(**kwargs)

    def getGitDate(self):
        # import datetime
        # return datetime.datetime.today().strftime('%Y.%-m.%-d')
        try:
            from subprocess import Popen, PIPE
            proc = Popen('git log -1 --format=%cd --date="format:%Y.%-m.%-d"', shell=True, stdout=PIPE)
            proc.wait()
            sha = proc.stdout.read()
            return sha[:-1].decode('utf-8')
        except:
            return ''

    def getGitSHA(self):
        try:
            from subprocess import Popen, PIPE
            proc = Popen('git describe --always --dirty --abbrev=40', shell=True, stdout=PIPE)
            proc.wait()
            sha = proc.stdout.read()
            return sha[:-1].decode('utf-8')
        except:
            return ''

    def hash_file(self, filename):
        import hashlib
        hasher = hashlib.md5()
        try:
            with open(filename, 'rb') as afile:
                buf = afile.read()
                hasher.update(buf)
            file_hash = hasher.hexdigest()
            return file_hash
        except:
            return


def fillTemplate(basedir, templates, replacements):
    for tmp in templates:
        with open(str(basedir/tmp), 'r') as f:
            lines = ''.join(f.readlines())
        for i in range(len(replacements)):
            newLines = copy(lines)
            newFileName = tmp
            for key, value in replacements[i]:
                r = re.compile(key)
                newLines = r.subn(value, newLines)[0]
                newFileName = r.sub(value, newFileName)
            if (basedir/newFileName).exists():
                with open(str(basedir/newFileName), 'r') as f:
                    oldLines = ''.join(f.readlines())
                if oldLines == newLines:
                    print('Skipping {}'.format(newFileName))
                    continue
            print('Generating {}'.format(newFileName))
            with open(str(basedir/newFileName), 'w') as f:
                f.write(newLines)
