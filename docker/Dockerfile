# Base container that includes all dependencies but not the actual repo
# Modified from the softlearning Docker file

ARG UBUNTU_VERSION=16.04
ARG ARCH=
ARG CUDA=10.0

##### GPU stuff
FROM nvidia/cudagl${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

ARG UBUNTU_VERSION
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.5.32-1

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND="noninteractive"
# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# ========== Anaconda ==========
# https://github.com/ContinuumIO/docker-images/blob/master/anaconda/Dockerfile
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget  --no-check-certificate --quiet https://repo.anaconda.com/archive/Anaconda2-2019.03-Linux-x86_64.sh && \
    /bin/bash Anaconda2-2019.03-Linux-x86_64.sh -b -p /opt/conda && \
    rm Anaconda2-2019.03-Linux-x86_64.sh
RUN sed -i'' 's/archive\.ubuntu\.com/us\.archive\.ubuntu\.com/' /etc/apt/sources.list
RUN sed -i~orig -e 's/# deb-src/deb-src/' /etc/apt/sources.list && apt-get update --fix-missing
RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH

###### GPU STUFF
# ========== Tensorflow dependencies ==========
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        cuda-cublas-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        zip \
        unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN [ ${ARCH} = ppc64le ] || (apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu${UBUNTU_VERSION/./}-5.0.2-ga-cuda${CUDA} \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*)

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
########

# ========== Special Deps ==========
RUN apt-get update
RUN apt-get -y install git make cmake unzip
# RUN pip install awscli
# ALE requires zlib
RUN apt-get -y install zlib1g-dev
# MUJOCO requires graphics stuff (Why?)
RUN apt-get -y build-dep glfw
RUN apt-get -y install libxrandr2 libxinerama-dev libxi6 libxcursor-dev
# copied from requirements.txt
#RUN pip install imageio tabulate nose
RUN apt-get install -y vim ack-grep
RUN pip install --upgrade pip
# usual pip install pygame will fail
RUN apt-get build-dep -y python-pygame
RUN pip install Pillow
RUN apt install -y libopenmpi-dev

# ========== OpenAI Gym ==========
RUN apt-get -y install libgtk2.0-0
#RUN apt-get -y install ffmpeg
RUN apt-get -y install libav-tools
CMD alias ffmpeg="avconv"

# ========== Setup mujoco ========
CMD mkdir /root/.mujoco
RUN wget https://www.roboti.us/download/mjpro150_linux.zip
RUN wget https://www.roboti.us/download/mjpro131_linux.zip
RUN unzip mjpro150_linux.zip -d /root/.mujoco
RUN unzip mjpro131_linux.zip -d /root/.mujoco
ADD mjkey.txt /root/.mujoco/mjkey.txt
RUN apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev

# ========= Cassie Environment ===========
RUN apt-get install -y llvm-5.0
RUN apt-get install -y libglfw3-dev
RUN apt-get install -y clang-5.0

# ========== Add codebase stub ==========
CMD mkdir /root/code
ADD environment.yml /root/code/environment.yml
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH
# This is a hack required to make mujocopy to compile in gpu mode
RUN mkdir -p /usr/lib/nvidia-000
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/lib/nvidia-000

# make the conda environment
RUN conda env create -f /root/code/environment.yml

ENV PYTHONPATH /root/code/meta_mb:$PYTHONPATH
ENV PATH /opt/conda/envs/meta_mb/bin:$PATH
RUN echo "source activate meta_mb" >> /root/.bashrc
ENV BASH_ENV /root/.bashrc
WORKDIR /root/code

RUN python -c 'import mujoco_py'
