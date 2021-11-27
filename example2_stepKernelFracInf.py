import matplotlib.pyplot as plt
from time import time

######################################################################
# Get a fractional kernel
from PyNucleus import kernelFactory

# show available options
kernelFactory.print()

from numpy import inf
kernelFracInf = kernelFactory('fractional', dim=2, s=0.75, horizon=inf)

print(kernelFracInf)
plt.figure().gca().set_title('Fractional kernel')
kernelFracInf.plot()

