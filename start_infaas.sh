#!/bin/bash
set -ex

### General variables ###
SCRIPT_DIR=$(dirname $(readlink -f $0))
INFAAS_HOME=${SCRIPT_DIR} # Know relative path
CMAKE_VERSION='3.7'
LOG_DIR=${INFAAS_HOME}"/logs/master_logs/" # Logging directory for Master
MASTER_IP=`curl -s http://169.254.169.254/latest/meta-data/local-ipv4` # IP of the machine that runs this script

### Modes ###
# Different modes used to configure INFaaS like other existing systems.
# Defaults should be sufficient for most users, but more details
## about each can be found by running ./queryfe_server with no inputs

# Master Decision Mode
# 0: INFAAS_ALL, 1: INFAAS_NOQPSLAT, 2: ROUNDROBIN,
## 3: ROUNDROBIN_STATIC, 4: GPUSHARETRIGGER,
## 5: CPUBLISTCHECK, 6: GPUSHARETRIGGER_SKIPBLIST,
## 7: ROUNDROBIN_DYNAMIC
MASTER_DECISION_MODE="6"

# Whether to use VM_DAEMON or not. ON: true, OFF: false
# VM_DAEMON is disabled for single worker autoscaling test.
VM_DAEMON_MODE="ON"

# Worker Autoscaling Mode
WORKER_AUTOSCALER="3"  # 0: NONE, 1: Static, 2: Individual, 3: INFaaS

# Utilization threshold and maximum number of tries when running this script
MAX_TRY=40  # Number of tries before failing to setup
CPUGPU_UTIL_THRESH=80 # CPU/GPU utilization threshold (out of 100)
INFERENTIA_UTIL_THRESH=70 # Inferentia utilization threshold (out of 100)

###### UPDATE THESE VALUES BEFORE RUNNING ######
REGION='<REGION>'
ZONE='<ZONE>'
SECURITY_GROUP='<SECURITYGROUP>'
IAM_ROLE='<IAMROLE>'
MODELDB='<MYMODELDB>' # Model repository bucket (do not include s3://)
CONFIGDB='<MYCONFIGDB>' # Configuration bucket (do not include s3://)
WORKER_IMAGE='ami-<INFAASAMI>'
NUM_INIT_CPU_WORKERS=1
NUM_INIT_GPU_WORKERS=0
NUM_INIT_INFERENTIA_WORKERS=0
MAX_CPU_WORKERS=1
MAX_GPU_WORKERS=0
MAX_INFERENTIA_WORKERS=0
# Used for making popular GPU variants exclusive
# Set to 0 for no GPU to be used as exclusive
# IMPORTANT: if NUM_INIT_GPU_WORKERS > 0, SLACK_GPU should be less than this (i.e. at least one GPU should be available for sharing)
SLACK_GPU=0
KEY_NAME='worker_key'
MACHINE_TYPE_GPU='p3.2xlarge'
MACHINE_TYPE_CPU='m5.2xlarge'
MACHINE_TYPE_INFERENTIA='inf1.2xlarge'
DELETE_MACHINES='2' # 0: VM daemon stops machines; 1: VM daemon deletes machines; 2: VM daemon persists machines, but removes them from INFaaS's view

### Values for VMs that don't need user configuration ###
MIN_WORKERS=$[$NUM_INIT_CPU_WORKERS + $NUM_INIT_GPU_WORKERS + $NUM_INIT_INFERENTIA_WORKERS] # Used for VM daemon
EXECUTOR_PREFIX='infaas-worker'
EXECUTOR_PORT='50051'
STARTUP_SCRIPT='/opt/INFaaS/src/worker/start_worker.sh'


#########Beginning of setup script#########

echo "=============Welcome to INFaaS============="
echo ""
echo "Executing setup script"

# Check if user has put in their credentials via aws configure
if [ ! -f ${HOME}/.aws/credentials ]; then
  echo "AWS credentials not found! Please put them in by calling: aws configure"
  exit 1
fi

# Checks if 1) cmake is installed and 2) if the right version of cmake is installed
cmake_installed=`cmake --version 2> /dev/null | grep -q ${CMAKE_VERSION} && echo "installed"`
if [ -z "$cmake_installed" ]; then
  echo "Installing cmake, version "${CMAKE_VERSION}
  pushd ${INFAAS_HOME}/scripts
  bash install_cmake.sh
  popd
else
  echo "cmake, version "${CMAKE_VERSION}" detected"
fi

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
echo "Creating metadata store"
if [[ ! -z $(pidof redis-server) ]]; then
  sudo kill $(pidof redis-server)
  sleep 5  # May not be able to restart instantly.
fi

if [[ -f "appendonly.aof" ]]; then
  rm appendonly.aof  # remove old data.
fi

redis-server ${INFAAS_HOME}"/src/metadata-store/redis-serv.conf" &

# Get port from redis-serv.conf for passing to modelreg_server
REDIS_PORT=`grep "port" src/metadata-store/redis-serv.conf | awk '{print $NF}'`
echo "Metadata store created"

# Wait until redis server is ready
cnt=0
ready_state=""
while [[ -z "${ready_state}" ]]; do
  redis-cli -p ${REDIS_PORT} PING | grep -q "PONG" && ready_state="ready"
  cnt=$[$cnt+1]
  if [[ $cnt -eq ${MAX_TRY} ]]; then
    echo "Redis server failed to start."
    exit 1
  fi
  sleep 1 # avoid busy looping
done

#########Ensure dependencies are installed vor master and metadata store#########

# Check if gRPC cpp is installed
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
  popd
fi
# Redox-specific (https://stackoverflow.com/a/9631350)
export LD_LIBRARY_PATH="/usr/local/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Check if OpenCV is installed
opencv_state="installed"
pkg-config --modversion opencv | grep -q "was not found" && opencv_state=""
if [[ -z "${opencv_state}" ]]; then
  echo "Installing OpenCV"
  sudo apt-get install libopencv-dev
else
  echo "OpenCV detected"
fi

# Check if AWS CPP is installed
if [[ -d /usr/local/include/aws ]]; then
  echo "AWS cpp detected"
else
  echo "Installing AWS cpp"
  pushd ${INFAAS_HOME}/scripts
  bash install_aws_cpp_sdk.sh ${INFAAS_HOME}/thirdparty/aws-cpp-sdk
  popd
fi

#########Build INFaaS#########

### First, write out the constants file ###
sed -e "s@const std::string infaas_bucket = .*;@const std::string infaas_bucket = \"${MODELDB}\";@g" \
    -e "s@const std::string infaas_config_bucket = .*@const std::string infaas_config_bucket = \"${CONFIGDB}\";@g" \
    -e "s@const std::string region = .*;@const std::string region = \"${REGION}\";@g" \
    src/include/constants.h.templ > src/include/constants.h

echo "Now building INFaaS from source"

mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..

# Create model database/bucket
# If bucket already exists, command will simply skip the creation.
aws s3api head-bucket --bucket ${MODELDB} 2>&1 | grep -q "Not Found" && \
      aws s3api create-bucket --bucket ${MODELDB} --acl private \
      --create-bucket-configuration LocationConstraint=${REGION}

# Create all initial CPU workers
if [[ $[${NUM_INIT_CPU_WORKERS}] -le 0 ]]; then
  echo "NO CPU Workers!"
else
  actual_num_init_cpu=$[$NUM_INIT_CPU_WORKERS-1] # Subtract one since we enumerate from zero
  for iworker in $( eval echo {0..$actual_num_init_cpu} ); do
    next_executor_name=${EXECUTOR_PREFIX}"-cpu-"${iworker}
    echo "Starting CPU worker: "${next_executor_name}

    exec_ip=`bash scripts/start_vm.sh ${REGION} ${ZONE} \
                      ${KEY_NAME} ${next_executor_name} ${WORKER_IMAGE} \
                      ${MACHINE_TYPE_CPU} ${STARTUP_SCRIPT} ${SECURITY_GROUP} \
                      ${MAX_TRY} ${IAM_ROLE} ${MASTER_IP} ${WORKER_AUTOSCALER} \
                      ${MODELDB}`

    # Set to last word in line
    exec_ip=`echo ${exec_ip} | awk '{print $NF}'`

    if [[ "$exec_ip" == "FAIL" ]]; then
      echo "Failure in creating worker "${next_executor_name}
      exit 1
    fi

    echo "Executor "${next_executor_name}" created with IP "${exec_ip}

    # Wait until the worker is ready.
    cnt=0
    ready_state=""
    while [[ -z "${ready_state}" ]]; do
      ./build/bin/query_heartbeat ${exec_ip}:${EXECUTOR_PORT} \
        | grep -q "SUCCEEDED" && ready_state="ready"
      cnt=$[$cnt+1]
      if [[ $cnt -eq ${MAX_TRY} ]]; then
        echo ${next_executor_name}" query executor failed to start."
        exit 1
      fi
      sleep 5 # avoid busy looping
    done

    echo ${next_executor_name}" query executor is ready!"

    # Register worker in shared memory (both IP and instance-id)
    worker_inst_id=`aws ec2 describe-instances --filters Name=tag:Name,Values=${next_executor_name} --query 'Reservations[*].Instances[*].InstanceId' --output text`
    build/bin/redis_startup_helper localhost ${REDIS_PORT} ${next_executor_name} \
                                   ${exec_ip} ${EXECUTOR_PORT} "0" "0" \
                                   ${worker_inst_id}
  done
fi

# Create all initial GPU workers
if [[ $[${NUM_INIT_GPU_WORKERS}] -le 0 ]]; then
  echo "NO GPU Workers!"
else
  slack_gpu_counter=${SLACK_GPU}
  actual_num_init_gpu=$[$NUM_INIT_GPU_WORKERS-1] # Subtract one since we enumerate from zero
  for iworker in $( eval echo {0..$actual_num_init_gpu} ); do
    next_executor_name=${EXECUTOR_PREFIX}"-gpu-"${iworker}
    echo "Starting GPU worker: "${next_executor_name}

    exec_ip=`bash scripts/start_vm.sh ${REGION} ${ZONE} \
                      ${KEY_NAME} ${next_executor_name} ${WORKER_IMAGE} \
                      ${MACHINE_TYPE_GPU} ${STARTUP_SCRIPT} ${SECURITY_GROUP} \
                      ${MAX_TRY} ${IAM_ROLE} ${MASTER_IP} ${WORKER_AUTOSCALER} \
                      ${MODELDB}`

    # Set to last word in line
    exec_ip=`echo ${exec_ip} | awk '{print $NF}'`

    if [[ "$exec_ip" == "FAIL" ]]; then
      echo "Failure in creating worker "${next_executor_name}
      exit 1
    fi

    echo "Executor "${next_executor_name}" created with IP "${exec_ip}

    # Wait until the worker is ready.
    cnt=0
    ready_state=""
    while [[ -z "${ready_state}" ]]; do
      ./build/bin/query_heartbeat ${exec_ip}:${EXECUTOR_PORT} \
        | grep -q "SUCCEEDED" && ready_state="ready"
      cnt=$[$cnt+1]
      if [[ $cnt -eq ${MAX_TRY} ]]; then
        echo ${next_executor_name}" query executor failed to start."
        exit 1
      fi
      sleep 5 # avoid busy looping
    done

    echo ${next_executor_name}" query executor is ready!"

    # Register worker in shared memory (both IP and instance-id)
    worker_inst_id=`aws ec2 describe-instances --filters Name=tag:Name,Values=${next_executor_name} --query 'Reservations[*].Instances[*].InstanceId' --output text`

    # Make slack if necessary
    is_slack="0"
    if [[ ${slack_gpu_counter} -gt 0 ]]; then
      is_slack="1"
      slack_gpu_counter=$[$slack_gpu_counter-1]
      echo "Executor "${next_executor_name}" is slack"
    fi
    build/bin/redis_startup_helper localhost ${REDIS_PORT} ${next_executor_name} \
                                   ${exec_ip} ${EXECUTOR_PORT} "2" ${is_slack} \
                                   ${worker_inst_id}
  done
fi

# Create all initial Inferentia workers
if [[ $[${NUM_INIT_INFERENTIA_WORKERS}] -le 0 ]]; then
  echo "NO Inferentia Workers!"
else
  actual_num_init_inferentia=$[$NUM_INIT_INFERENTIA_WORKERS-1] # Subtract one since we enumerate from zero
  for iworker in $( eval echo {0..$actual_num_init_inferentia} ); do
    next_executor_name=${EXECUTOR_PREFIX}"-inf-"${iworker}
    echo "Starting Inferentia worker: "${next_executor_name}

    exec_ip=`bash scripts/start_vm.sh ${REGION} ${ZONE} \
                      ${KEY_NAME} ${next_executor_name} ${WORKER_IMAGE} \
                      ${MACHINE_TYPE_INFERENTIA} ${STARTUP_SCRIPT} ${SECURITY_GROUP} \
                      ${MAX_TRY} ${IAM_ROLE} ${MASTER_IP} ${WORKER_AUTOSCALER} \
                      ${MODELDB}`

    # Set to last word in line
    exec_ip=`echo ${exec_ip} | awk '{print $NF}'`

    if [[ "$exec_ip" == "FAIL" ]]; then
      echo "Failure in creating worker "${next_executor_name}
      exit 1
    fi

    echo "Executor "${next_executor_name}" created with IP "${exec_ip}

    # Wait until the worker is ready.
    cnt=0
    ready_state=""
    while [[ -z "${ready_state}" ]]; do
      ./build/bin/query_heartbeat ${exec_ip}:${EXECUTOR_PORT} \
        | grep -q "SUCCEEDED" && ready_state="ready"
      cnt=$[$cnt+1]
      if [[ $cnt -eq ${MAX_TRY} ]]; then
        echo ${next_executor_name}" query executor failed to start."
        exit 1
      fi
      sleep 5 # avoid busy looping
    done

    echo ${next_executor_name}" query executor is ready!"

    # Register worker in shared memory (both IP and instance-id)
    worker_inst_id=`aws ec2 describe-instances --filters Name=tag:Name,Values=${next_executor_name} --query 'Reservations[*].Instances[*].InstanceId' --output text`
    build/bin/redis_startup_helper localhost ${REDIS_PORT} ${next_executor_name} \
                                   ${exec_ip} ${EXECUTOR_PORT} "1" "0" \
                                   ${worker_inst_id}
  done
fi

# Make LOG_DIR if it does not exist
mkdir -p ${LOG_DIR}

### Create model register frontend ###
if [[ ! -z $(pidof modelreg_server ) ]]; then
  pkill -f modelreg_server
fi

MODELREG_LOG=${LOG_DIR}"modelreg_server.log"
build/bin/modelreg_server localhost ${REDIS_PORT} >${MODELREG_LOG} 2>&1 &

# Wait one second for server to launch
sleep 1

# Check that modelreg_server is successfully running
build/bin/modelreg_heartbeat | grep -q "FAILED\|No such file or directory" && \
    echo "Heartbeat failed, check Model Registration Server" && exit 1

echo "Model Registration server successfully launched"

### Create query frontend ###
if [[ ! -z $(pidof queryfe_server) ]]; then
  pkill -f queryfe_server
fi

QUERYFE_LOG=${LOG_DIR}"queryfe_server.log"
build/bin/queryfe_server localhost ${REDIS_PORT} ${MASTER_DECISION_MODE} >${QUERYFE_LOG} 2>&1 &

# Wait one second for server to launch
sleep 1

# Check that queryfe_server is successfully running
build/bin/queryfe_heartbeat 2>&1 | grep -q "FAILED\|No such file or directory" && \
    echo "Heartbeat failed, check Query Frontend Server" && exit 1

echo "Query Frontend server successfully launched"

### Create VM scaling daemon ###
if [[ ! -z $(pidof master_vm_daemon) ]]; then
  pkill -f master_vm_daemon
fi

if [[ "${VM_DAEMON_MODE}" == "ON" ]]; then
  VMDAEMON_LOG=${LOG_DIR}"vmdaemon.log"
  build/bin/master_vm_daemon localhost ${REDIS_PORT} \
                             ${CPUGPU_UTIL_THRESH} ${INFERENTIA_UTIL_THRESH} \
                             ${ZONE} ${KEY_NAME} ${WORKER_IMAGE} \
                             ${MACHINE_TYPE_GPU} ${MACHINE_TYPE_CPU} \
                             ${MACHINE_TYPE_INFERENTIA} \
                             ${STARTUP_SCRIPT} ${SECURITY_GROUP} \
                             ${MAX_TRY} ${IAM_ROLE} ${EXECUTOR_PORT} \
                             ${EXECUTOR_PREFIX} ${MIN_WORKERS} \
                             ${MAX_CPU_WORKERS} ${MAX_GPU_WORKERS} \
                             ${MAX_INFERENTIA_WORKERS} \
                             ${MASTER_IP} ${WORKER_AUTOSCALER} \
                             ${DELETE_MACHINES} >${VMDAEMON_LOG} 2>&1 &
  echo "Master VM daemon successfully launched"
else
  echo "Not using Master VM daemon."
fi

echo "INFaaS is all set up!"

exit 0

