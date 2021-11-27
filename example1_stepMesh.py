import matplotlib.pyplot as plt
from numpy import sqrt

######################################################################
# Get a mesh and refine it

from PyNucleus import meshFactory

# show available options
meshFactory.print()

mesh = meshFactory('square', ax=0., ay=0., bx=1., by=1.)
for _ in range(3):
    mesh = mesh.refine()

print('Mesh:', mesh)
plt.figure().gca().set_title('Mesh')
mesh.plot()

