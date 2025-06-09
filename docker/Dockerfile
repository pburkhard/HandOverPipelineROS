##############
# Base Image #
##############
FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04 AS base

# Path to project root folder inside the container
ARG PROJECT_ROOT_PATH

# Set default RUN shell to bash
SHELL ["/bin/bash", "-c"]

# Adds labels to the Dockerfile
LABEL version="0.1"
LABEL description="Dockerfile for building and running the ROS container image."

# Set environment variables
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and clean up apt cache
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    ncurses-term && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Install ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update \
    && apt-get install -y --no-install-recommends ros-noetic-desktop-full
RUN apt-get install -y --no-install-recommends python3-rosdep
RUN rosdep init \
    && rosdep fix-permissions \
    && rosdep update
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Copy the project directory (files to exclude are defined in .dockerignore)
COPY ../ ${PROJECT_ROOT_PATH}

# Automatically source the catkin workspace if it built
RUN echo 'if [ -f /handover_pipeline_ws/devel/setup.bash ]; then' >> ~/.bashrc && \
    echo '  source handover_pipeline_ws/devel/setup.bash' >> ~/.bashrc && \
    echo 'fi' >> ~/.bashrc

##############
# Full Image #
##############
FROM base AS full

# Path to conda installation directory
ARG CONDA_PATH=/opt/conda

# Set environment variables for conda
ENV CONDA_PATH=/opt/conda
ENV PATH=$CONDA_PATH/bin:$PATH

# Install further dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download and Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_PATH \
    && rm /tmp/miniconda.sh \
    && conda clean -ay \
    && conda init bash

# make working directory as the project root
# this is the default directory when the container is run
WORKDIR ${PROJECT_ROOT_PATH}