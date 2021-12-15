

Prerequisites
=============

In order to install PyNucleus, you will need

* Python 3,
* MPI,
* METIS,
* ParMETIS,
* SuiteSparse,
* make.

On Debian, Ubuntu etc, the required dependecies can be installed with

.. code-block:: shell

   sudo apt-get install python3 mpi-default-bin mpi-default-dev libmetis-dev libparmetis-dev libsuitesparse-dev

Installation
============

PyNucleus is installed via

.. code-block:: shell

   make

The compilation of PyNucleus can be configured by modifying the file `config.yaml` in the root folder.
This allows for example to set paths for libraries that are installed in non-standard directories.
The defaults are as follows:

.. literalinclude:: ../config.yaml
   :language: yaml

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


Docker container
================

A Docker container that contains all the required dependecies can be built as well:

.. code-block:: shell

   make docker

Once the build is done, it can be launched as

.. code-block:: shell

   make docker-linux

or

.. code-block:: shell

   make docker-mac
