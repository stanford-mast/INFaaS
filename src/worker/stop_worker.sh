#!/bin/bash

set -xe
set -u
set -o pipefail

# Constants. Some may eventually become command-line inputs
USER=`whoami`
if [[ $USER == "root" ]]; then
  export HOME="/root"
else
  export HOME="/home/${USER}"
fi
echo "HOME directory is: ${HOME}"

REDIS_IP="0.0.0.0"
if [[ $# -ne 1 ]]; then
  echo "Usage: ./stop_worker.sh <redis-ip>"
  echo "Using default ip: 0.0.0.0"
else
  REDIS_IP="$1"
fi

INFAAS_HOME="/opt/INFaaS/"

echo "Stopping INFaaS worker"

# 1. Kill this worker
if [[ -z $(pidof query_executor) ]]; then
  echo "query_executor has already stopped."
else
  sudo kill $(pidof query_executor)
fi

# Kill the gpu daemon
if [[ -z $(pidof gpu_daemon) ]]; then
  echo "GPU daemon has already stopped or not started."
else
  sudo kill $(pidof gpu_daemon)
fi

# Check that query_executor is not running
# If still running, force quit.
if [[ -f "${INFAAS_HOME}build/bin/query_heartbeat" ]]; then
  cd "${INFAAS_HOME}build"

  ./bin/query_heartbeat | grep -q "SUCCEEDED" && echo "Force kill..." && \
    sudo kill -9 $(pidof query_executor)
  # Check second time
  ./bin/query_heartbeat | grep -q "SUCCEEDED" && echo "Failed to stop!" && exit 1
fi

# Terminate all containers.
# First terminate them gracefully
if [[ ! -z $(docker ps -q) ]]; then
  # If we have neuron-rtd docker, need to stop others first. Otherwise, memory
  # leak.
  if [[ ! -z $(docker ps -f "name=neuron-rtd" -q) ]]; then
    echo "Stop neuron-rtd till the end."
    if [[ ! -z $(docker ps -f "since=neuron-rtd" -q) ]]; then
      docker stop -t 30 $(docker ps -f "since=neuron-rtd" -q)
    fi
  fi
  docker stop -t 30 $(docker ps -q)
fi

# Then force them to stop.
if [[ ! -z $(docker ps -q) ]]; then
  docker kill --signal=9 $(docker ps -q)
fi

# Delete metadata store only if we started our local one.
if [[ "${REDIS_IP}" == "0.0.0.0" ]]; then
  if [[ ! -z $(pidof redis-server) ]]; then
    sudo kill $(pidof redis-server)
  fi
fi

echo "INFaaS worker is shutdown!"

exit 0


