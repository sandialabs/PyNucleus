# VERSION:        1.0
# DESCRIPTION:    Dockerized PyNucleus build
# AUTHOR:         Christian Glusa

# Base docker image
FROM docker.io/library/debian:unstable
LABEL maintainer Christian Glusa

# install packages needed for build
RUN sed -i 's/Components: main/Components: main contrib non-free/' /etc/apt/sources.list.d/debian.sources \
  && apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade -y \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        gcc g++ make gfortran \
        libssl-dev ca-certificates \
        ccache \
        git less nano \
        libopenblas0-serial \
        libmetis-dev libparmetis-dev \
        hdf5-tools \
        libsuitesparse-dev \
        libarpack2-dev \
        mpi-default-bin mpi-default-dev \
        python3 python3-dev python-is-python3 python3-pip \
        python3-numpy python3-scipy python3-matplotlib python3-mpi4py cython3 python3-yaml python3-h5py python3-tk jupyter-notebook python3-meshio python3-gmsh \
        mencoder \
  --no-install-recommends \
  && rm -rf /var/lib/apt/lists/*

# allow running MPI as root in the container
# bind MPI ranks to hwthreads
ENV OMPI_MCA_hwloc_base_binding_policy=hwthread \
    PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe \
    MPIEXEC_FLAGS=--allow-run-as-root \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

COPY . /pynucleus

WORKDIR /pynucleus

ARG PYNUCLEUS_BUILD_PARALLELISM=1

# Install dependencies
# RUN --mount=type=cache,target=/root/.ccache --mount=type=cache,target=/root/.cache/pip \
RUN \
    make prereq PIP_FLAGS=" --break-system-packages" \
    && make prereq-extra PIP_FLAGS=" --break-system-packages"

# Build PyNucleus
# RUN --mount=type=cache,target=/root/.ccache --mount=type=cache,target=/root/.cache/pip \
RUN \
    make install PIP_INSTALL_FLAGS=" --break-system-packages" \
    && find . -type f -name '*.c' -exec rm {} + \
    && find . -type f -name '*.cpp' -exec rm {} + \
    && rm -rf build packageTools/build base/build metisCy/build fem/build multilevelSolver/build nl/build \
    && ccache -s

# Generate documentation and examples
RUN make docs \
    && rm examples/test.hdf5

# Set up greeting, settings
RUN \
    echo '[ ! -z "$TERM" -a -r /pynucleus/README.container.rst ] && printf "\e[32m" && cat /pynucleus/README.container.rst && printf "\e[0m"' >> /etc/bash.bashrc \
    && echo "alias ls='ls --color=auto -FN'" >> /etc/bash.bashrc \
    && echo "set completion-ignore-case On" >> /etc/inputrc

# Set up entrypoint with jupyter notebook
COPY entrypoint.sh /usr/local/bin/
ENTRYPOINT ["entrypoint.sh"]
WORKDIR /root
EXPOSE 8889
