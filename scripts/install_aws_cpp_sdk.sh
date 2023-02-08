#!/bin/bash
set -x
set -e

cmake_inst_path="/opt/INFaaS/thirdparty/"
if [ $# -ne 1 ]; then
  echo "Using "${cmake_inst_path}" for cmake"
else
  cmake_inst_path=$1
fi

sudo yum install -y gcc-c++ openssl-devel curl-devel
pushd ${cmake_inst_path}
rm -rf aws-sdk-cpp
git clone https://github.com/aws/aws-sdk-cpp.git
cd aws-sdk-cpp
git checkout main
git pull origin main
git submodule update --init --recursive # https://github.com/aws/aws-sdk-cpp/issues/1770
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j $(nproc)
sudo make install
popd

