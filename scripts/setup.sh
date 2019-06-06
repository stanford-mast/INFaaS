#!/bin/bash

# Script to install necessary dependencies for INFaaS
set -xe

sudo apt-get update
sudo apt-get install -y python-pip
pip install --user --upgrade pip

sudo apt-get install build-essential libcurl3-dev libopencv-dev \
    libopencv-core-dev python-pil software-properties-common autoconf automake \
    libtool pkg-config

#./install_docker.sh
./install_grpc.sh
