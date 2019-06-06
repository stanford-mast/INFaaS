#!/bin/bash

set -ex

SCRIPT_DIR=$(dirname $(readlink -f $0))
INFAAS_HOME=${SCRIPT_DIR} # Know relative path

echo "Executing setup script"

# Check if Redis is installed
if command -v redis-server >/dev/null; then
  echo "Redis detected"
else
  echo "Installing Redis"
  wget http://download.redis.io/redis-stable.tar.gz
  tar xvzf redis-stable.tar.gz
  pushd redis-stable
  make
  sudo make install
  popd
  rm -rf redis-stable.tar.gz redis-stable
fi

# Ensure dependencies are installed for master server and metadata store
if command -v grpc_cpp_plugin >/dev/null; then
  echo "gRPC detected"
else
  echo "Installing gRPC"
  pushd ${INFAAS_HOME}"/scripts"
  bash install_grpc.sh
  popd
fi

# Check if redox is installed
if [[ -d ${HOME}/redox && -f /usr/local/lib64/libredox_static.a ]]; then
  echo "Redox detected"
else
  echo "Installing Redox"
  sudo apt update
  sudo apt install -y cmake build-essential libhiredis-dev libev-dev
  pushd ${HOME}
  git clone https://github.com/hmartiro/redox.git
  cd redox
  mkdir build && cd build
  cmake ..
  make -j$(nproc)
  sudo make install
  # https://stackoverflow.com/a/9631350
  export LD_LIBRARY_PATH="/usr/local/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  popd
fi

# Check if OpenCV is installed
opencv_state="installed"
pkg-config --modversion opencv | grep -q "was not found" && opencv_state=""
if [[ -z "${opencv_state}" ]]; then
  echo "Installing OpenCV"
  sudo apt-get install libopencv-dev
fi
