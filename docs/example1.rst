
Example 1 - A simple PDE problem
================================

In this first example, we will construct a finite element discretization of a classical PDE problem and solve it.
The full code of this example can be found in `examples/example1.py <https://github.com/sandialabs/PyNucleus/blob/master/examples/example1.py>`_.

Factories
---------

The creation of different groups of objects, such as finite element spaces or meshes, use factories.
The available classes that a factory provides can be displayed by calling the ``print()`` method of the factory.
An object is built by passing the name of the desired class and additional parameters to the factory.
If this sounds vague now, don't worry, the examples below will make it clear.

Meshes
------

The first object we need to create is a mesh to support the finite element discretization.
We start by construction a mesh for a square domain :math:`\Omega=[0, 1] \times [0, 1]` and refining it uniformly three times:

.. literalinclude:: ../examples/example1.py
   :start-after: Get a mesh
   :end-before: #################
   :lineno-match:

The output of the above code snippet is given below.
In particular, we see what other meshes we could have constructed using the ``meshFactory``, apart from 'square', and what parameters we can pass to the factory,
We also see that we created a 2d mesh with 289 vertices and 512 cells.

.. program-output:: python3 example1.py --finalTarget mesh

.. plot:: example1_stepMesh.py

Many PyNucleus objects have a ``plot`` method, similar to the mesh that we just created.

DoFMaps
-------

In the next step, we create a finite element space on the mesh.
By default, we assume a Dirichlet condition on the entire boundary of the domain.
We build a piecewise linear finite element space.

.. literalinclude:: ../examples/example1.py
   :start-after: Construct a finite element space
   :end-before: #################
   :lineno-match:

.. program-output:: python3 example1.py --finalTarget dofmap

.. plot:: example1_stepDoFMap.py

Functions and vectors
---------------------

Functions can either be defined in Python, or in Cython.
The advantage of the latter is that their code is compiled, which speeds up evaluation significantly.
A couple of compiled functions are already available via the ``functionFactory``.
A generic Python function can be used via the ``Lambda`` function class.

We will be solving the problem

.. math::

   -\Delta u &= f & \text{ in } \Omega, \\
   u &= 0 & \text{ on } \partial \Omega,

for two different forcing functions :math:`f`.

We assemble the right-hand side

.. math::

   \int_\Omega f v

of the linear system by calling the ``assembleRHS`` method of the DoFMap object, and interpolate the exact solutions into the finite element space.


.. literalinclude:: ../examples/example1.py
   :start-after: Construct some simple functions
   :end-before: #################
   :lineno-match:

.. program-output:: python3 example1.py --finalTarget functions

.. plot:: example1_stepFunctions.py

Matrices
--------

We assemble two matrices, the mass matrix

.. math::

   \int_\Omega u v

and the stiffness matrix associated with the Laplacian

.. math::

   \int_\Omega \nabla u \cdot \nabla v

.. literalinclude:: ../examples/example1.py
   :start-after: Assemble mass
   :end-before: #######
   :lineno-match:

.. program-output:: python3 example1.py --finalTarget matrices

Solvers
-------

Now that we have assembled our linear system, we want to solve it.
We choose to solve one system using an LU solver, and the other one using a CG solver.

.. literalinclude:: ../examples/example1.py
   :start-after: Construct solvers
   :end-before: #################
   :lineno-match:

.. program-output:: python3 example1.py --finalTarget solvers

.. plot:: example1_stepSolvers.py

Norms and inner products
------------------------

Finally, we want to check that we actually solved the system by computing residual errors.
We also compute errors in :math:`H^1_0` and :math:`L^2` norms.

.. literalinclude:: ../examples/example1.py
   :start-after: Inner products
   :end-before: plt.show
   :lineno-match:

.. program-output:: python3 example1.py --finalTarget innerNorm

This concludes our first example.
Next, we turn to nonlocal equations.
