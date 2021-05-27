#!/bin/bash

set -xe
set -u
set -o pipefail

REDIS_IP="0.0.0.0"
AUTOSCALER="0"
if [[ $# -lt 1 ]]; then
  echo "Usage: ./start_worker.sh <redis-ip> <infaas-bucket> [<autoscaler type>]"
  echo "Using default ip: 0.0.0.0"
else
  REDIS_IP="$1"
fi

MODELDB="$2"

if [[ $# -ge 3 ]]; then
  AUTOSCALER="$3"
else
  echo "Using default autoscaler: 0 (None)"
fi

# Constants. Some may eventually become command-line inputs
USER=`whoami`
if [[ $USER == "root" ]]; then
  export HOME="/root"
else
  export HOME="/home/${USER}"
fi
echo "HOME directory is: ${HOME}"

REGION='us-west-2'
WORKER_ID=`curl -s http://169.254.169.254/latest/meta-data/instance-id`
WORKER_NAME=$(aws ec2 describe-tags --region $REGION --filters "Name=resource-id,Values=$WORKER_ID" "Name=key,Values=Name" --output text | cut -f5)

# TODO: the $INFAAS_HOME is dedicated to /opt for workers. 
INFAAS_HOME="/opt/INFaaS/"
INFAAS_URL="https://github.com/stanford-mast/INFaaS.git"
WORKERDB='/tmp/trtmodels'
max_try=60  # At most try 60 times.
WORKER_IP=`curl -s http://169.254.169.254/latest/meta-data/local-ipv4`
LOCAL_INPUT="/tmp/infaas_input"  # For offline queries
LOCAL_OUTPUT="/tmp/infaas_output"
INFAAS_LOG_DIR="${INFAAS_HOME}logs/worker"

# Create INFaaS log directory
mkdir -p "${INFAAS_LOG_DIR}"

# For metadata store
REDIS_PORT=`grep "port" ${INFAAS_HOME}/src/metadata-store/redis-serv.conf \
            | awk '{print $NF}'`

echo "Executing INFaaS worker setup script for worker: ${WORKER_NAME}"

# Check INFAAS_HOME exiss
if [[ ! -d ${INFAAS_HOME} ]]; then
  echo "INFaaS doesn't exist! Downloading..."
  git clone ${INFAAS_URL} ${INFAAS_HOME}
fi

# Update docker image
docker pull qianl15/infaaspytorch:latest
docker pull nvcr.io/nvidia/tensorrtserver:19.03-py3
docker pull qianl15/gnmt-infaas:latest

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

# Check whether gRPC is installed.
if command -v grpc_cpp_plugin >/dev/null; then
  echo "gRPC detected"
else
  echo "Installing gRPC"
  pushd ${INFAAS_HOME}"/scripts"
  bash install_grpc.sh
  popd
fi

# Check whether OpenCV is installed.
opencv_state="installed"
pkg-config --modversion opencv | grep -q "was not found" && opencv_state=""
if [[ -z "${opencv_state}" ]]; then
  echo "Installing OpenCV"
  sudo apt-get install libopencv-dev
fi

# Wait until redis server is ready
if [[ ${REDIS_IP} == "0.0.0.0" ]]; then
  echo "Start local redis server!"
  redis-server ${INFAAS_HOME}src/metadata-store/redis-serv.conf &
fi

# For all-in-a-box test (worker master on the same machine.)
if [[ "${WORKER_IP}" == "${REDIS_IP}" ]]; then
  REDIS_IP="0.0.0.0"
fi

cnt=0
ready_state=""
while [[ -z "${ready_state}" ]]; do
  redis-cli -h ${REDIS_IP} -p ${REDIS_PORT} PING | \
    grep -q "PONG" && ready_state="ready"
  cnt=$[$cnt+1]
  if [[ $cnt -eq ${max_try} ]]; then
    echo "Redis server failed to start."
    exit 1
  fi
  sleep 1 # avoid busy looping
done
echo "Redis server ${REDIS_IP}:${REDIS_PORT} is connected!"

# Clear previous modeldb
if [[ -d "${WORKERDB}" ]]; then
  rm -rf "${WORKERDB}"
fi
mkdir -p ${WORKERDB}

# Remove old input/output
if [[ -d "${LOCAL_INPUT}" ]]; then
  rm -rf "$LOCAL_INPUT"
fi
mkdir "$LOCAL_INPUT"

if [[ -d "${LOCAL_OUTPUT}" ]]; then
  rm -rf "$LOCAL_OUTPUT"
fi
mkdir "$LOCAL_OUTPUT"

# Build query executor

### First, write out the constants file ###
cd ${INFAAS_HOME}

sed -e "s@const std::string infaas_bucket = .*;@const std::string infaas_bucket = \"${MODELDB}\";@g" \
    -e "s@const std::string region = .*;@const std::string region = \"${REGION}\";@g" \
    src/include/constants.h.templ > src/include/constants.h

mkdir -p build && cd build
cmake .. -DBUILD_ONLY_WORKER=ON
make -j$(nproc)

# Check whether this machine has GPU. If so, start tensorRT Inference server.
if [[ -f "/proc/driver/nvidia/version" ]]; then
  echo "Worker has GPU!"

  # Check if nvidia-docker is installed
  if command -v nvidia-docker >/dev/null; then
    echo "nvidia-docker detected"
  else
    echo "Installing nvidia-docker"
    pushd ${INFAAS_HOME}"/scripts"
    bash install_docker.sh
    popd
  fi
  TRTLOGS="${INFAAS_LOG_DIR}/trtserver.log"
  # Start tensorRT Inference server, initially no model loaded.
  nvidia-docker run --rm --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 \
    -v${WORKERDB}:/models \
    nvcr.io/nvidia/tensorrtserver:19.03-py3 trtserver \
    --model-store='/models' \
    --exit-on-error=false \
    --strict-model-config=false \
    --tf-gpu-memory-fraction=0.5 \
    --repository-poll-secs=5  > ${TRTLOGS} 2>&1 & # Reduce the polling interval.
  # Wait until the server is up
  ready_state=""
  cnt=0
  while [[ -z "${ready_state}" ]]; do
    curl -s localhost:8000/api/status/ | grep -q "SERVER_READY" && ready_state="ready"
    cnt=$[$cnt+1]
    if [[ $cnt -eq ${max_try} ]]; then
      echo "TensorRT server failed to start."
      exit 1
    fi
    sleep 1 # avoid busy looping
  done
  echo "TensorRT Inference Server is Ready!"

  echo "Start GPU monitoring Daemon!"
  GPU_LOG="${INFAAS_LOG_DIR}/gpu_daemon.log"
  ${INFAAS_HOME}/build/bin/gpu_daemon ${WORKER_NAME} ${REDIS_IP} ${REDIS_PORT} \
              > ${GPU_LOG} 2>&1 &
else
  echo "No GPU detected!"
fi

# Check if this machine has Inferentia. If so, prepare the rtd.
if [[ -d "/run/infa" ]]; then
  echo "Worker has Inferentia (Neuron)!"
  docker pull qianl15/neuron-rtd:latest
  docker pull qianl15/neuron-tf:latest

  # Check the neuron-rtd on the host and stop if needed.
  rtd_running=""
  sudo service neuron-rtd status | grep "running" && rtd_running="yes"
  if [[ ! -z ${rtd_running} ]]; then
    sudo service neuron-rtd stop
  else
    echo "Host neuron-rtd is not running."
  fi

  # Start the neuron-rtd docker.
  if [[ ! -d "/tmp/neuron_rtd_sock" ]]; then
    mkdir -p /tmp/neuron_rtd_sock
    chmod o+rwx /tmp/neuron_rtd_sock
  fi
  docker run --rm -it -d --env AWS_NEURON_VISIBLE_DEVICES="0" \
    --cap-add SYS_ADMIN --cap-add IPC_LOCK -v /tmp/neuron_rtd_sock/:/sock \
    --name=neuron-rtd qianl15/neuron-rtd:latest

  # Wait until the rtd is running
  rtd_running=""
  cnt=0
  while [[ -z "${rtd_running}" ]]; do
    docker logs neuron-rtd | grep "Server listening on unix:/sock/neuron.sock" && rtd_running="yes"
    cnt=$[$cnt+1]
    if [[ $cnt -eq ${max_try} ]]; then
      echo "Neuron-rtd docker container failed to start."
      exit 1
    fi
    sleep 1 # avoid busy looping
  done
  echo "Neuron-rtd container is Ready!"
else
  echo "No Inferentia (Neuron) chips detected!"
fi

# Create query executor
QUERY_LOG="${INFAAS_LOG_DIR}/query_executor.log"
${INFAAS_HOME}build/bin/query_executor ${WORKER_NAME} ${REDIS_IP} ${REDIS_PORT} \
 ${AUTOSCALER} > ${QUERY_LOG} 2>&1 &

# Check that query_executor is successfully running
cnt=0
ready_state=""
while [[ -z "${ready_state}" ]]; do
  ${INFAAS_HOME}build/bin/query_heartbeat | grep -q "SUCCEEDED" && ready_state="ready"
  cnt=$[$cnt+1]
  if [[ $cnt -eq ${max_try} ]]; then
    echo "Query executor failed to start."
    exit 1
  fi
  sleep 1 # avoid busy looping
done

echo "Query Executor has successfully launched, log is available at ${QUERY_LOG}"

echo "INFaaS worker is all set up!"

exit 0


