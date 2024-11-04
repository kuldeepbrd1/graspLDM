FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

# ENV
ENV HOME_DIR=/root/
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# REQUIREMENTS & CERTS
ADD requirements.txt /tmp/

SHELL ["/bin/bash", "-c"]

# hotfix- cuda source error on ubuntu 20.04
RUN  echo "deb [by-hash=no] http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list

# APT
RUN apt-get update -y\
    && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive \
    apt-get install -q -y --no-install-recommends \
    build-essential \
    cmake \
    dirmngr \
    gnupg2 \
    git \
    iputils-ping \
    ca-certificates \
    nano \
    net-tools \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-opengl \
    tree \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates \
    && echo "alias python=python3" >> /root/.bashrc\
    && echo "alias pip=pip3" >> /root/.bashrc

# PIP
ENV ACRONYM_INSTALL_PATH=/tmp/acronym
RUN git clone https://github.com/NVlabs/acronym.git ${ACRONYM_INSTALL_PATH} \
    && pip install -r ${ACRONYM_INSTALL_PATH}/requirements.txt \
    && pip install ${ACRONYM_INSTALL_PATH} \
    && rm -r ${ACRONYM_INSTALL_PATH} \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

CMD ["/bin/bash"]
