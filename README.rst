
Welcome to PyNucleus!
=====================================

PyNucleus is a finite element code that specifically targets nonlocal operators of the form

.. math::

   \int_{\mathbb{R}^d} [u(x)-u(y)] \gamma(x, y) dy

for nonlocal kernels :math:`\gamma` with finite or infinite horizon and of integrable or fractional type.
Specific examples of such operators include the integral and regional fractional Laplacians, their truncated and tempered variants, and operators arising from peridynamics.

The package aims to provide efficient discretization and assembly routines with :math:`O(N \log N)` quasi-optimal complexity.
The resulting sets of equations can be solved using optimal linear solvers.
The code is fully NumPy/SciPy compatible, allowing easy integration into application codes.


Features
========

* Simplical meshes in 1D, 2D, 3D

* Finite Elements:

  * continuous P1, P2, P3 spaces,
  * discontinuous P0 space

* Assembly of local operators

* Nonlocal kernels:

  * Finite and infinite horizon
  * Singularities: fractional, peridynamic, constant, Gaussian kernel
  * spatially variable kernels: variable fractional order and variable coefficients

* Nonlocal assembly (1D and 2D) into dense, sparse and hierarchical matrices

* Solvers/preconditioners:

  * LU,
  * Cholesky,
  * incomplete LU & Cholesky,
  * Jacobi,
  * CG,
  * BiCGStab,
  * GMRES,
  * geometric multigrid

* Distributed computing using MPI

* Computationally expensive parts of the code are compiled via Cython.

* Partitioning using METIS / ParMETIS


Documentation
=============

The documentation is available `here <https://sandialabs.github.io/PyNucleus/index.html>`_.

To generate the Sphinx documentation locally, run

.. code-block:: shell

   make docs

and open ``docs/build/index.html`` in your browser.




Possible ways to install and use PyNucleus
==================================

* container image
* Spack installation
* manual installation


Container image
----------------

The simplest way to use PyNucleus is to pull a container image from the GitHub Container Registry.
This requires an installation of either

* podman (https://podman.io/) and podman-compose (https://github.com/containers/podman-compose) or
* Docker (https://www.docker.com/) and Docker Compose (https://docs.docker.com/compose/install/).

For many Linux distributions these can be installed from the package repositories.
In what follows we will assume that we are using podman.
All commands for Docker should be identical up to the substitution of `podman` with `docker`.

For example, on Ubuntu podman can be installed with

.. code-block:: shell

   sudo apt-get install podman podman-compose

Instructions for other platforms can be found here: https://podman.io/docs/installation

Once podman is installed, we can download a copy of https://github.com/sandialabs/PyNucleus/compose.yaml and save it to an empty directory.
In that directory we then run

.. code-block:: shell

   podman compose run pynucleus

This launches a shell on the container with PyNucleus.
A simple way to test if things work is to run

.. code-block:: shell

   drivers/runFractional.py

This should print some information about the solution of a fractional Laplacian problem and open up several plots.

For development using PyNucleus it can be useful to launch a Jupyter notebook server:

.. code-block:: shell

   podman compose up pynucleus-jupyter

and then open the access the Jupyter notebook interface at https://localhost:8889


Spack install
-------------

In order to install Spack itself, follow the instructions at https://github.com/spack/spack.

Install PyNucleus and all its dependencies with the command

.. code-block:: shell

   spack install py-pynucleus

To then load PyNucleus

.. code-block:: shell

   spack load py-pynucleus

The examples can be found in the install directory. In order to get there:

.. code-block:: shell

   spack cd -i py-pynucleus


Manual installation
-------------------

In order to install PyNucleus, you will need

* Python 3,
* MPI,
* METIS,
* ParMETIS,
* SuiteSparse,
* make.

On Debian, Ubuntu etc, the required dependencies can be installed with

.. code-block:: shell

   sudo apt-get install python3 mpi-default-bin mpi-default-dev libmetis-dev libparmetis-dev libsuitesparse-dev

On MacOS the required dependencies can be installed with

.. code-block:: shell

   brew install python open-mpi
   brew tap brewsci/num
   brew install brewsci-metis brewsci-parmetis brewsci-suite-sparse

After cloning the source code, PyNucleus is installed via

.. code-block:: shell

   make

The compilation of PyNucleus can be configured by modifying the file `config.yaml <https://github.com/sandialabs/PyNucleus/blob/master/config.yaml>`_ in the root folder.
This allows for example to set paths for libraries that are installed in non-standard directories.

If you want to easily modify the source code without re-installing the package every time, and editable install is available as

.. code-block:: shell

   make dev

PyNucleus depends on other Python packages that will be installed automatically:

* NumPy
* SciPy
* Matplotlib
* Cython
* mpi4py
* tabulate
* PyYAML
* H5py
* modepy
* meshpy
* scikit-sparse


Funding
=======

PyNucleus' development is funded through the FOMSI project (PI: Christian Glusa, FY23-FY25) of the LDRD program at Sandia National Laboratories.

PyNucleus' development was previously funded through the MATNIP project (PI: Marta D'Elia, FY20-22).

.. image:: data/matnip.png
   :height: 100px

*The MATNIP project develops for the first time a rigorous nonlocal interface theory based on physical principles that is consistent with the classical theory of partial differential equations when the nonlocality vanishes and is mathematically well-posed.
This will improve the predictive capability of nonlocal models and increase their usability at Sandia and, more in general, in the computational-science and engineering community.
Furthermore, this theory will provide the groundwork for the development of nonlocal solvers, reducing the burden of prohibitively expensive computations.*
