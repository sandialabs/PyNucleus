# VERSION:        0.1
# DESCRIPTION:    Dockerized PyNucleus build
# AUTHOR:         Christian Glusa

# Base docker image
FROM debian:testing
LABEL maintainer Christian Glusa

ENV LANG en_US.UTF-8

# install packages needed for build
RUN sed -i 's/Components: main/Components: main contrib non-free/' /etc/apt/sources.list.d/debian.sources \
  && apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
        locales \
        autoconf automake gcc g++ make gfortran wget zlib1g-dev libffi-dev \
        tk-dev \
        libssl-dev ca-certificates cmake \
        git less \
        libboost-dev  \
        hdf5-tools \
        libsuitesparse-dev \
        libarpack2-dev \
        mpi-default-bin mpi-default-dev \
        python3 python3-dev python-is-python3 python3-pip python3-mpi4py cython3 python3-numpy python3-scipy python3-matplotlib python3-tk python3-venv \
        libmetis-dev libparmetis-dev \
        texlive texlive-extra-utils texlive-latex-extra ttf-staypuft dvipng cm-super \
        jupyter-notebook \
        emacs-nox vim \
  --no-install-recommends \
  && rm -rf /var/lib/apt/lists/* \
  && sed -i -e "s/# $LANG.*/$LANG UTF-8/" /etc/locale.gen \
  && dpkg-reconfigure --frontend=noninteractive locales \
  && update-locale LANG=$LANG

# allow running MPI as root in the container
# bind MPI ranks to hwthreads
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    VIRTUAL_ENV=/pynucleus/venv \
    PATH="/pynucleus/venv/bin:$PATH" \
    OMPI_MCA_hwloc_base_binding_policy=hwthread \
    MPIEXEC_FLAGS=--allow-run-as-root \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

COPY . /pynucleus

WORKDIR /pynucleus

RUN python3 -m venv $VIRTUAL_ENV && \
    make prereq PIP_FLAGS=--no-cache-dir && \
    make prereq-extra PIP_FLAGS=--no-cache-dir && \
    make install && \
    make docs && \
    python -m pip install --no-cache-dir ipykernel && \
    rm -rf build packageTools/build base/build metisCy/build fem/build multilevelSolver/build nl/build && \
    python -m ipykernel install --name=PyNucleus && \
    echo '[ ! -z "$TERM" -a -r /pynucleus/README.container.rst ] && printf "\e[32m" && cat /pynucleus/README.container.rst && printf "\e[0m"' >> /etc/bash.bashrc

WORKDIR /root
