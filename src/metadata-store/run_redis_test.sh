#!/bin/bash

set -ex
set -u
set -o pipefail

SCRIPT_DIR=$(dirname $(readlink -f $0))
cd ${SCRIPT_DIR}

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

# Create the metadata server: Redis
echo "Creating metadata store for test"
if [[ ! -z $(pidof redis-server) ]]; then
  sudo kill $(pidof redis-server)
  rm -rf appendonly.aof
  sleep 2
fi

redis-server ${SCRIPT_DIR}/redis-test.conf &

# Build metadata store
pushd ${SCRIPT_DIR}"/../../"
rm -rf build_md
mkdir build_md && cd build_md
cmake -DBUILD_ONLY_MD=ON ..
make -j$(nproc)

echo "Running redis test"
bin/redis_md_test
popd

# Remove build_md
rm -rf ${SCRIPT_DIR}"/../../build_md"

echo "All tests finished. Deleting test server..."
pkill redis-server
rm -rf appendonly.aof

