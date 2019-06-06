#!/bin/bash
set -x
set -e

cmake_inst_path="/opt/INFaaS/thirdparty/aws-cpp-sdk"
if [ $# -ne 1 ]; then
  echo "Using "${cmake_inst_path}" for cmake"
else
  cmake_inst_path=$1
fi

# Need to install higher version cmake
install_path="${HOME}/aws-cpp-sdk"
mkdir -p ${install_path}
cd ${install_path}

cmake -DCMAKE_BUILD_TYPE=Release  ${cmake_inst_path}
make -j8
sudo make install

