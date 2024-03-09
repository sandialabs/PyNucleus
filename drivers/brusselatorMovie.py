#!/usr/bin/env python3
###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################

import numpy as np
import matplotlib.pyplot as plt
from path import Path
from shutil import rmtree
import h5py
from subprocess import Popen
from PyNucleus_base import driver
from PyNucleus_fem import meshNd
from PyNucleus_fem.DoFMaps import DoFMap
from PyNucleus_nl.nonlocalProblems import brusselatorProblem

d = driver()
brusselatorProblem(d)
d.add('zoomIn', False)
d.add('shading', acceptedValues=['gouraud', 'flat'])
d.process()

filename = d.identifier+'.hdf5'
resultFile = h5py.File(str(filename), 'r')

mesh = meshNd.HDF5read(resultFile['mesh'])
dm = DoFMap.HDF5read(resultFile['dm'])
dm.mesh = mesh

folder = Path('brusselatorMovie')/Path(filename).basename()
try:
    rmtree(str(folder))
except:
    pass
folder.makedirs_p()

if d.zoomIn:
    folderZoom = Path('brusselatorMovie')/(Path(filename).basename()+'-zoomIn')
    try:
        rmtree(str(folderZoom))
    except:
        pass
    folderZoom.makedirs_p()

l = sorted([int(i) for i in resultFile['U']])

u = dm.zeros()
# v = dm.zeros()
u.assign(np.array(resultFile['U'][str(l[-1])]))
vmin = u.min()
vmax = u.max()
vmin, vmax = -0.1*(vmax-vmin)+vmin, 1.1*(vmax-vmin)+vmin

plt.figure()
upd = u.plot(flat=True, vmin=vmin, vmax=vmax, shading=d.shading)

# for i in reversed(l):
for i in l:
    u.assign(np.array(resultFile['U'][str(i)]))
    # v.assign(np.array(resultFile['V'][str(i)]))
    d.logger.info('ts={}: min={}, max={}'.format(i, u.min(), u.max()))

    u.plot(flat=True, vmin=vmin, vmax=vmax, shading=d.shading, update=upd)
    # plt.title(r'$\alpha={:.3}, \beta={:.3}, \varepsilon={:.3}, \phi={:.3}$'.format(f.attrs['alpha'], f.attrs['beta']))
    plt.savefig(folder/'{:05}.png'.format(i), dpi=300)
    if d.zoomIn:
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.savefig(folderZoom/'{:05}.png'.format(i), dpi=300)
    # plt.close()
resultFile.close()

Popen(['mencoder', 'mf://*.png', '-mf', 'fps=10', '-o',
       '../{}.avi'.format(Path(filename).basename().stripext()), '-ovc', 'lavc',
       '-lavcopts', 'vcodec=msmpeg4v2:vbitrate=800'],
      cwd=folder).wait()
if d.zoomIn:
    Popen(['mencoder', 'mf://*.png', '-mf', 'fps=10', '-o',
           '../{}-zoom.avi'.format(Path(filename).basename().stripext()), '-ovc', 'lavc',
           '-lavcopts', 'vcodec=msmpeg4v2:vbitrate=800'],
          cwd=folderZoom).wait()

d.finish()
