
Example 2 - Nonlocal problems
=============================

I this second example, we will assemble and solve several nonlocal equations.
The full code of this example can be found in `examples/example2.py <https://github.com/sandialabs/PyNucleus/blob/master/examples/example2.py>`_.

PyNucleus can assemble operators of the form

.. math::

   \mathcal{L}u(x) = \int_{\mathbb{R}^d} [u(y)-u(x)] \gamma(x, y) dy.

The kernel :math:`\gamma` is of the form

.. math::

   \gamma(x,y) = \phi(x, y) |x-y|^{-\beta(x,y)} \chi_{V_\delta(x)}(y).

Here, :math:`\phi` is a positive function, and :math:`\chi` is the indicator function.
:math:`0<\delta\le\infty` is called the horizon and determines the size of the kernel support :math:`V_\delta(x) \subset \mathbb{R}^d`.
The singularity :math:`\beta` of the kernel depends on the family of kernels:

- fractional type: :math:`\beta(x,y)=d+2s(x,y)`, where :math:`d` is the spatial dimension and :math:`s(x,y)` is the fractional order.
- constant type :math:`\beta(x,y)=0`
- peridynamic type :math:`\beta(x,y)=1`

At present, the only implemented interaction regions are balls in the 2-norm:

.. math::

   V_{\delta}^{(2)}(x) = \{y \in \mathbb{R}^d | ||x-y||_2<\delta\}.


A fractional kernel
-------------------

We start off by creating a fractional kernel with infinite horizon and constant fractional order :math:`s=0.75`.

.. literalinclude:: ../examples/example2.py
   :start-after: Get a fractional kernel
   :end-before: #################
   :lineno-match:

.. program-output:: python3 example2.py --finalTarget kernelFracInf

.. plot:: example2_stepKernelFracInf.py

By default, kernels are normalized. This can be disabled by passing `normalized=False`.


Nonlocal assembly
-----------------

We will be solving the problem

.. math::

   -\mathcal{L} u &= f && \text{ in } \Omega=B(0,1)\subset\mathbb{R}^2, \\
   u &= 0 && \text{ in } \mathbb{R}^2 \setminus \Omega,

for constant forcing function :math:`f=1`.

First, we generate a mesh.
Instead of the `meshFactory` used in the previous example, we now use the `nonlocalMeshFactory`.
The advantage is that this factory can generate meshes with appropriate interaction domains.
For this particular example, the factory will not generate any interaction domain, since the homogeneous Dirichlet condition on :math:`\mathbb{R}^2\setminus\Omega` can be enforced via a boundary integral.

.. literalinclude:: ../examples/example2.py
   :start-after: Generate an appropriate mesh
   :end-before: #################
   :lineno-match:

.. program-output:: python3 example2.py --finalTarget meshFracInf

.. plot:: example2_stepMeshFracInf.py

Next, we obtain a piecewise linear, continuous DoFMap on the mesh, assemble the RHS and interpolate the known analytic solution.
We assemble the nonlocal operator by passing the kernel to the `assembleNonlocal` method of the DoFMap object.
The optional parameter `matrixFormat` determines what kind of linear operator is assembled.
We time the assembly of the operator as a dense matrix, and as a hierarchical matrix, and inspect the resulting objects.

.. literalinclude:: ../examples/example2.py
   :start-after: Assemble the operator
   :end-before: #################
   :lineno-match:

.. program-output:: python3 example2.py --finalTarget assemblyFracInf

It can be observed that both assembly routines take roughly the same amount of time.
The reason for this is that the operator itself has quite small dimensions.
For larger number of unknowns, we expect the hierarchical assembly scale like :math:`\mathcal{O}(N \log^{2d} N)`, whereas the dense assembly will scale at best like :math:`\mathcal{O}(N^2)`.

Similar to the local PDE example, we can then solve the resulting linear equation and compute the error in energy norm.

.. literalinclude:: ../examples/example2.py
   :start-after: Solve the linear system
   :end-before: #################
   :lineno-match:

.. program-output:: python3 example2.py --finalTarget solveFracInf

.. plot:: example2_stepSolveFracInf.py


A finite horizon case
---------------------

Next, we solve a nonlocal Poisson problem involving a constant kernel with finite horizon.
We will choose :math:`\gamma(x,y) \sim \chi_{V_{\delta}^{(2)}(x)}(y)` for :math:`\delta=0.2`, and solve

.. math::

   -\mathcal{L} u &= f && \text{ in } \Omega=[0,1]^2, \\
   u &= -(x_1^2 + x_2^2)/4 && \text{ in } \mathcal{I},

where :math:`\mathcal{I}:=\{y\in\mathbb{R}^2\setminus\Omega | \exists x\in\Omega: \gamma(x,y)\neq 0\}` is the interaction domain.

.. literalinclude:: ../examples/example2.py
   :start-after: Solve a problem with finite horizon
   :end-before: #################
   :lineno-match:

.. program-output:: python3 example2.py --finalTarget finiteHorizon

.. plot:: example2_stepFiniteHorizon.py
