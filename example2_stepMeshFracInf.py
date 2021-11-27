import matplotlib.pyplot as plt
from time import time

######################################################################
# Get a fractional kernel
from PyNucleus import kernelFactory

from numpy import inf
kernelFracInf = kernelFactory('fractional', dim=2, s=0.75, horizon=inf)

######################################################################
# Generate an appropriate mesh
from PyNucleus import nonlocalMeshFactory, HOMOGENEOUS_DIRICHLET

# Get a mesh that is appropriate for the problem, i.e. with the required interaction domain.
meshFracInf, _ = nonlocalMeshFactory('disc', kernel=kernelFracInf, boundaryCondition=HOMOGENEOUS_DIRICHLET, hTarget=0.15)

print(meshFracInf)
plt.figure().gca().set_title('Mesh for fractional kernel')
meshFracInf.plot()

