#!/bin/bash

# Script to install docker and nvidia-docker(2)
set -x

sudo yum update -y

if ! command -v docker &> /dev/null; then
  sudo amazon-linux-extras install -y docker
  sudo systemctl --now enable docker
  # Test your Docker installation
  # sudo docker run --rm hello-world
else
  echo "Docker already installed, skipping installation"
fi

# Install nvidia-docker
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-amazon-linux
if ! command -v nvidia-container-toolkit &> /dev/null; then
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
     && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
  sudo yum clean expire-cache
  sudo yum install nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  # At this point, a working setup can be tested by running a base CUDA container:
  # sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
else
  echo "Nvidia-container-toolkit already installed, skipping installation"
fi