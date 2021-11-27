import matplotlib.pyplot as plt
from numpy import sqrt

######################################################################
# Get a mesh and refine it

from PyNucleus import meshFactory

mesh = meshFactory('square', ax=0., ay=0., bx=1., by=1.)
for _ in range(3):
    mesh = mesh.refine()

######################################################################
# Construct a finite element space
from PyNucleus import dofmapFactory

# show available options
dofmapFactory.print()

# We use piecewise linears
dm = dofmapFactory('P1', mesh)

print('DoFMap:', dm)

plt.figure().gca().set_title('DoFMap')
dm.plot()

