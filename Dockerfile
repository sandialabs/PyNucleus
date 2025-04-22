# VERSION:        1.0
# DESCRIPTION:    PyNucleus for Binder
# AUTHOR:         Christian Glusa

# Base docker image
FROM ghcr.io/sandialabs/pynucleus:880679799281ed3b4b60f7f75954f8218305742a
LABEL maintainer Christian Glusa

# Uninstall jupyter notebook server
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get remove -y \
        jupyter-core jupyter-notebook python3-traitlets python3-jsonschema \
  --no-install-recommends \
  && rm -rf /var/lib/apt/lists/*

# Install Jupyterlab
# RUN --mount=type=cache,target=/root/.cache/pip \
RUN \
    pip install --break-system-packages notebook jupyterlab && \
    jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

# Set up user for binder
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} \
    && mkdir ${HOME}/examples && cp -r /pynucleus/examples/* ${HOME}/examples \
    && mkdir ${HOME}/drivers && cp -r /pynucleus/drivers/* ${HOME}/drivers \
    && chown -R ${NB_UID} ${HOME}

USER ${NB_USER}
WORKDIR ${HOME}
ENTRYPOINT []
ENV SHELL /bin/bash
