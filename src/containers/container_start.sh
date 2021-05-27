#!/bin/bash

if [ $# -ne 4 ]; then
  echo "Usage: ./container_start.sh <scale> <model-file> <port>"
	echo "<scale>: input size dimensions (e.g., 224)"
  exit 1
fi

set -e
set -u
set -o pipefail

container_script=$1
scale=$2
model=$3
port=$4

/bin/bash -c "exec python ${container_script} ${scale} ${model} ${port}"

