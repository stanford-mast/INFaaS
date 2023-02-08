#!/bin/bash
set -xe

# Need to install higher version cmake
sudo yum remove cmake

version=3.13
build=2
rm -rf ~/temp
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
tar -xzvf cmake-$version.$build.tar.gz
cd cmake-$version.$build/

./bootstrap
make
sudo make install
cmake --version
