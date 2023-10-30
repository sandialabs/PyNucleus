
Drivers
=======

The code repository contains several sample problems that can be found in the `drivers <https://github.com/sandialabs/PyNucleus/blob/master/drivers>`_ subfolder.
The drivers take several command line parameters that can be used to change the problem and the type of outputs.
A listing of parameters can be displayed by passing the `--help` flag:

.. code-block:: shell

   drivers/runFractional.py --help

.. program-output:: python3 ../drivers/runFractional.py --help

Some of the drivers can be run in parallel using MPI, e.g.

.. code-block:: shell

   mpiexec -n 4 drivers/runFractional.py --domain=disc


runFractional.py
----------------

Assembles and solves fractional Poisson problems with infinite horizon.

runNonlocal.py
----------------

Assembles and solves nonlocal Poisson problems with finite horizon.

runFractionalHeat.py
--------------------

Solves a fractional heat equation with infinite horizon.

runNonlocalInterface.py
-----------------------

A two domain interface problem with jumps in solution and flux for finite horizon kernels.

brusselator.py
--------------

Solves a fractional-order Brusselator system.

runParallelGMG.py
-----------------

Assembles and solves a classical local Poisson problem using geometric multigrid.

runHelmholtz.py
-----------------

Assembles and solves a classical local Helmholtz problem.
