
Example 3 - Operator interpolation
==================================

This example demostrates the construction of a family of fractional
Laplacians parametrized by the fractional order using operator
interpolation. This can reduce the cost compared to assembling a new
matrix for each value.

The fractional Laplacian

.. math::

  (-\Delta)^{s} \text{ for } s \in [s_{\min}, s_{\max}] \subset (0, 1)

is approximated by

.. math::

   (-\Delta)^{s} \approx \sum_{m=0}^{M} \Theta_{k,m}(s) (-\Delta)^{s_{k,m}} \text{ for } s \in \mathcal{S}_{k}

for a sequence of intervals :math:`\mathcal{S}_{k}` that cover :math:`[s_{\min}, s_{\max}]` and scalar coefficients :math:`\Theta_{k,m}(s)`.
The number of intervals and interpolation nodes is picked so that the interpolation error is dominated by the discretization error.

The following example can be found at `examples/example3.py <https://github.com/sandialabs/PyNucleus/blob/master/examples/example3.py>`_.

We set up a mesh, a dofmap and a fractional kernel.
Instead of specifying a single value for the fractional order, we allow a range of values :math:`[s_{\min}, s_{\max}]=[0.05, 0.95]`.

.. literalinclude:: ../examples/example3.py
   :start-after: preamble
   :end-before: #################
   :lineno-match:

Next, we call the assembly of a nonlocal operator as before.
Since the operator for a particular value of the fractional order will be constructed on demand, this operation is fast.

.. literalinclude:: ../examples/example3.py
   :start-after: The operator is set up to be constructed on-demand
   :end-before: #################
   :lineno-match:

.. program-output:: python3 example3.py --finalTarget operator

Next, we choose the value of the fractional order. This needs to be within the range that we specified earlier.
We then solve a linear system involving the operator.

.. literalinclude:: ../examples/example3.py
   :start-after: Set s = 0.75
   :end-before: #################
   :lineno-match:

.. program-output:: python3 example3.py --finalTarget firstSolve

This solve is relatively slow, as it involves the assembly of the nonlocal operators that are needed for the interpolation.
We select a different value for the fractional order that is close to the first.
Solving a linear system with this value is faster as we have already assembled the operator needed for the interpolation.

.. literalinclude:: ../examples/example3.py
   :start-after: This should be faster
   :end-before: #################
   :lineno-match:

.. program-output:: python3 example3.py --finalTarget secondSolve

Next, we save the operator to file.
This first triggers the assembly of all operators nescessary to represent every value in :math:`s\in[0.05,0.95]`.

.. literalinclude:: ../examples/example3.py
   :start-after: We can save the operator
   :end-before: #################
   :lineno-match:

.. program-output:: python3 example3.py --finalTarget saveToFile

Next, we read the operator back in and solve another linear system.

.. literalinclude:: ../examples/example3.py
   :start-after: Now we can read them
   :lineno-match:

.. program-output:: python3 example3.py --finalTarget thirdSolve
