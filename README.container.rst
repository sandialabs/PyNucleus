
This is a container image for PyNucleus.

The directory from which the container was launched on the host system is mapped to /root.
PyNucleus is installed at /pynucleus.
A copy of the drivers and examples for PyNucleus can be found in /root/drivers and /root/examples.
The Jupyter notebook interface is available at https://localhost:8889 on the host.
A quick way to check that everything works is to run

  /root/drivers/runFractional.py

This should print some information about the solution of a fractional Laplacian problem and show several plots.
