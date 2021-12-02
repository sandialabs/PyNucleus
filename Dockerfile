
# VERSION:        0.1
# DESCRIPTION:    Dockerized PyNucleus build
# AUTHOR:         Christian Glusa

# Base docker image
FROM debian:testing
LABEL maintainer Christian Glusa

ENV LANG C.UTF-8

# based on recommendations from
# https://docs.nersc.gov/development/shifter/how-to-use/

# add contrib and non-free debian repos
RUN sed -i "s#deb http://deb.debian.org/debian testing main#deb http://deb.debian.org/debian testing main contrib non-free#g" /etc/apt/sources.list

# install packages needed for build
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
        autoconf automake gcc g++ make gfortran wget zlib1g-dev libffi-dev \
        tk-dev \
        libssl-dev ca-certificates cmake \
        git less \
        libboost-dev  \
        hdf5-tools \
        libsuitesparse-dev \
        libarpack2-dev \
        libmkl-avx2 libmkl-dev \
        mpi-default-bin mpi-default-dev \
        python3 python3-dev python3-pip python3-mpi4py cython3 python3-numpy python3-scipy python3-matplotlib python3-tk \
        libmetis-dev libparmetis-dev \
        texlive texlive-extra-utils texlive-latex-extra ttf-staypuft dvipng cm-super \
  --no-install-recommends \
  && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/lib

RUN echo "alias ls='ls --color=auto -FN'" >> /root/.bashrc


RUN /sbin/ldconfig



# copy code to container and build
# we copy only the packages over, not any run scripts

COPY PyNucleus /home/pynucleus-build/PyNucleus
COPY packageTools /home/pynucleus-build/packageTools
COPY base /home/pynucleus-build/base
COPY metisCy /home/pynucleus-build/metisCy
COPY fem /home/pynucleus-build/fem
COPY multilevelSolver /home/pynucleus-build/multilevelSolver
COPY nl /home/pynucleus-build/nl
COPY setup.py setup.cfg versioneer.py MANIFEST.in Makefile README.rst LICENSE /home/pynucleus-build/
RUN cd /home/pynucleus-build && make
