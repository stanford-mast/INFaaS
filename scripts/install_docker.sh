#!/bin/bash

# Script to install docker and nvidia-docker(2)
set -x
# Delete older version of docker
sudo apt-get remove docker docker-engine docker.io

sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

# Check OS
OS=`lsb_release -i -s`
if [[ $OS == "Debian" ]]; then
  curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
	sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/debian \
    $(lsb_release -cs) \
    stable"
else
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"
fi

sudo apt-get update

sudo apt-get install docker-ce

# Install nvidia-docker

# First delete old versions
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker

# Add package
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
       sudo apt-key add -

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update

# Install
sudo apt-get install -y nvidia-docker2

sudo pkill -SIGHUP dockerd

# Test...
nvidia-container-cli --load-kmods info
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi

