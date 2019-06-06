#!/bin/bash

# Script to install gRPC C++ and Python
set -x
set -e

# Install gRPC Python via pip
sudo python -m pip install grpcio grpcio-tools

# Install gRPC C++
BASE_DIR=$HOME/grpc
sudo apt-get install -y build-essential autoconf libtool pkg-config

# Check the latest release version at: https://grpc.io/release
# In order to be reproducible, we chose v1.16.0
git clone -b v1.16.0 https://github.com/grpc/grpc ${BASE_DIR}
cd ${BASE_DIR}
git submodule update --init

make -j8
sudo make install

# Install protocol buffer
pushd third_party/protobuf
sudo make install
popd

rm -rf ${BASE_DIR}
