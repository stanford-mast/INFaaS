#!/bin/bash
set -xe

# Need to install higher version cmake
sudo apt remove --purge --auto-remove cmake

version=3.7
build=2
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
tar -xzvf cmake-$version.$build.tar.gz
cd cmake-$version.$build/

./bootstrap
make -j4
sudo make install

cmake --version
