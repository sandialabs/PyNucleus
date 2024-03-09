# VERSION:        1.0
# DESCRIPTION:    Dockerized PyNucleus build
# AUTHOR:         Christian Glusa

# Base docker image
FROM debian:testing
LABEL maintainer Christian Glusa

# install packages needed for build
RUN sed -i 's/Components: main/Components: main contrib non-free/' /etc/apt/sources.list.d/debian.sources \
  && apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
        gcc g++ make gfortran \
        libssl-dev ca-certificates \
        git less nano \
        libopenblas0-serial \
        libmetis-dev libparmetis-dev \
        hdf5-tools \
        libsuitesparse-dev \
        libarpack2-dev \
        mpi-default-bin mpi-default-dev \
        python3 python3-dev python-is-python3 python3-pip \
        python3-numpy python3-scipy python3-matplotlib python3-mpi4py cython3 python3-yaml python3-h5py python3-tk jupyter-notebook \
  --no-install-recommends \
  && rm -rf /var/lib/apt/lists/*

# allow running MPI as root in the container
# bind MPI ranks to hwthreads
ENV OMPI_MCA_hwloc_base_binding_policy=hwthread \
    MPIEXEC_FLAGS=--allow-run-as-root \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

COPY . /pynucleus

WORKDIR /pynucleus

ARG PYNUCLEUS_BUILD_PARALLELISM=1

# Build PyNucleus
RUN make prereq PIP_FLAGS="--no-cache-dir --break-system-packages" && \
    make prereq-extra PIP_FLAGS="--no-cache-dir --break-system-packages" && \
    make install PIP_INSTALL_FLAGS="--no-cache-dir --break-system-packages" && \
    make docs && \
    find . -type f -name '*.c' -exec rm {} + && \
    find . -type f -name '*.cpp' -exec rm {} + && \
    rm -rf build packageTools/build base/build metisCy/build fem/build multilevelSolver/build nl/build

# Set up Jupyter notebooks, greeting, some bash things
RUN python -m pip install --no-cache-dir --break-system-packages ipykernel && \
    python -m ipykernel install --name=PyNucleus && \
    echo '[ ! -z "$TERM" -a -r /pynucleus/README.container.rst ] && printf "\e[32m" && cat /pynucleus/README.container.rst && printf "\e[0m"' >> /etc/bash.bashrc && \
    echo "alias ls='ls --color=auto -FN'" >> /etc/bash.bashrc && \
    echo "set completion-ignore-case On" >> /etc/inputrc

WORKDIR /root

# Copy examples and drivers to user home, launch Jupyter notebook server
ENTRYPOINT mkdir -p /root/examples && \
           mkdir -p /root/drivers && \
           cp -r --update=none /pynucleus/examples/* /root/examples && \
           cp -r --update=none /pynucleus/drivers/* /root/drivers && \
           jupyter notebook --port=8889 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password='' --notebook-dir=/root/ --KernelSpecManager.ensure_native_kernel=False --KernelSpecManager.allowed_kernelspecs=pynucleus > /dev/null 2>&1 & \
           /bin/bash

EXPOSE 8889
