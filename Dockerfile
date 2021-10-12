
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
  --no-install-recommends \
  && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/lib

RUN echo "alias ls='ls --color=auto -FN'" >> /root/.bashrc

RUN mkdir /build/

# install python
# Consider adding configure flags:
# --enable-optimizations
# --with-lto
# --build="$gnuArch"
# --enable-shared
# --with-system-expat
# --with-system-ffi
ARG pythonVersion=3.8.2
RUN cd /build && wget --no-check-certificate https://www.python.org/ftp/python/${pythonVersion}/Python-${pythonVersion}.tgz \
  && tar xvzf Python-${pythonVersion}.tgz && cd /build/Python-${pythonVersion} \
  && ./configure --enable-optimizations --with-pymalloc --enable-shared && make -j4 && make install && make clean && rm /build/Python-${pythonVersion}.tgz && rm -rf /build/Python-${pythonVersion}

# install mpich
ARG mpichVersion=3.2
RUN cd /build && wget --no-check-certificate https://www.mpich.org/static/downloads/${mpichVersion}/mpich-${mpichVersion}.tar.gz \
  && tar xvzf mpich-${mpichVersion}.tar.gz && cd /build/mpich-${mpichVersion} \
  && ./configure && make -j4 && make install && make clean && rm /build/mpich-${mpichVersion}.tar.gz && rm -rf /build/mpich-${mpichVersion}

# install mpi4py
ARG mpi4pyVersion=3.0.3
RUN cd /build && wget --no-check-certificate https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-${mpi4pyVersion}.tar.gz \
  && tar xvzf mpi4py-${mpi4pyVersion}.tar.gz && cd /build/mpi4py-${mpi4pyVersion} \
  && python3 setup.py build && python3 setup.py install && rm -rf /build/mpi4py-${mpi4pyVersion}

# install parmetis
ARG parmetisVersion=4.0.3
RUN cd /build && wget --no-check-certificate http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-${parmetisVersion}.tar.gz \
  && tar xvzf parmetis-${parmetisVersion}.tar.gz && cd /build/parmetis-${parmetisVersion} \
  && make config shared=1 && make -j4 && make install && make clean && rm /build/parmetis-${parmetisVersion}.tar.gz && rm -rf /build/parmetis-${parmetisVersion}

# install metis
ARG metisVersion=5.1.0
RUN cd /build && wget --no-check-certificate http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-${metisVersion}.tar.gz \
  && tar xvzf metis-${metisVersion}.tar.gz && cd /build/metis-${metisVersion} \
  && make config shared=1 && make -j4 && make install && make clean && rm /build/metis-${metisVersion}.tar.gz && rm -rf /build/metis-${metisVersion}

# delete build directory
RUN rm -rf /build/

RUN /sbin/ldconfig



# copy code to container and build
# we copy only the packages over, not any run scripts

COPY PyNucleus_* /home/pynucleus-build/
COPY setup.py /home/pynucleus-build/
COPY Makefile /home/pynucleus-build/
RUN cd /home/pynucleus-build && python -m pip install .
