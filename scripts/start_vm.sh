#!/bin/bash

# Script to start a new VM. This is AWS-specific, and is called by both
# start_infaas.sh and the master daemon

set -x
set -e
set -u
set -o pipefail

if [ $# -ne 13 ]; then
	echo "Usage: ./start_vm.sh <region> <zone> <key-name> <executor-name> <worker-image> <machine-type> <startup-script> <security-group> <max-try> <iam-role> <master-ip> <worker-autoscaler> <infaas_bucket>"
  exit 1
fi

REGION=$1
ZONE=$2
KEY_NAME=$3
EXECUTOR_NAME=$4
WORKER_IMAGE=$5
MACHINE_TYPE=$6
STARTUP_SCRIPT=$7
SECURITY_GROUP=$8
MAX_TRY=$9
IAM_ROLE=${10}
MASTER_IP=${11}
WORKER_AUTOSCALER=${12}
MODELDB=${13}

# Check if key-pair already exists. IMPORTANT: this will only check in the current directory!
KEY_FULL=${KEY_NAME}".pem"
if [ ! -e ./"${KEY_FULL}" ]; then
  new_kp=`aws ec2 create-key-pair --key-name ${KEY_NAME} --query KeyMaterial`
  echo ${new_kp} | cut -d "\"" -f 2 | sed -e 's/\\n/\n/g' > ${KEY_FULL}
  chmod 400 ${KEY_FULL}
  sleep 2 # Time for it to register in AWS
fi

# Get the instance ID to see if the worker already exists
worker_inst_id=`aws ec2 describe-instances --filters Name=tag:Name,Values=${EXECUTOR_NAME} --query 'Reservations[*].Instances[*].InstanceId' --output text`

# If worker_inst_id is empty, the instance needs to be created
if [ -z "$worker_inst_id" ]; then
  aws ec2 run-instances \
    --image-id ${WORKER_IMAGE} \
    --instance-type ${MACHINE_TYPE} \
    --placement AvailabilityZone=${REGION}${ZONE} \
    --count 1 \
    --key-name ${KEY_NAME} \
    --security-groups ${SECURITY_GROUP} \
    --tag-specifications 'ResourceType=instance,Tags=[{Key="Name",Value='$EXECUTOR_NAME'}]' \
    --iam-instance-profile Name=${IAM_ROLE}

    # Now wait for the machine to be started
    state="starting"
    count=0
    while [[ "running" != "${state}" ]]; do
      sleep 2
      state=`aws ec2 describe-instances --filters Name=tag:Name,Values=${EXECUTOR_NAME} --query 'Reservations[*].Instances[*].State.Name' --output text`
      count=$[${count}+1]
      if [[ ${count} == ${MAX_TRY} ]]; then
        echo "MAX LIMIT REACHED!"
        exit 1
      fi
    done
    # Sleep a while to reduce connection refuse error.
    sleep 5
else
  # The worker exists. Now check if it is stopped
  state=`aws ec2 describe-instances --filters Name=tag:Name,Values=${EXECUTOR_NAME} --query 'Reservations[*].Instances[*].State.Name' --output text`
  if [[ "stopped" == "${state}" ]]; then
    # Boot and wait for it to load
    aws ec2 start-instances --instance-ids ${worker_inst_id}

    count=0
    while [[ "running" != "${state}" ]]; do
      state=`aws ec2 describe-instances --filters Name=tag:Name,Values=${EXECUTOR_NAME} --query 'Reservations[*].Instances[*].State.Name' --output text`
      count=$[${count}+1]
      if [[ ${count} == ${MAX_TRY} ]]; then
        echo "MAX LIMIT REACHED!"
        exit 1
      fi
      sleep 2
    done
  fi
fi

# The worker is running at this point. Get its IP address
exec_ip=`aws ec2 describe-instances --filters Name=tag:Name,Values=${EXECUTOR_NAME} --query 'Reservations[*].Instances[*].NetworkInterfaces[*].PrivateIpAddress' --output text`

# Then run the start script.
# Stop existing executors.
PUBLIC_DNS=${exec_ip}
if [[ "${MASTER_IP}" == "${exec_ip}" ]]; then
  # Stop the executor locally.
  bash /opt/INFaaS/src/worker/stop_worker.sh ${MASTER_IP} \
       > /tmp/stop_worker.log 2>&1 < /dev/null
else
  ssh -o StrictHostKeyChecking=no -i ${KEY_FULL} "ubuntu@"${PUBLIC_DNS} \
    -p 22 \
    "bash /opt/INFaaS/src/worker/stop_worker.sh ${MASTER_IP} > /tmp/stop_worker.log 2>&1 < /dev/null"
fi

# nohup to avoid busy waiting
if [[ "${MASTER_IP}" == "${exec_ip}" ]]; then
  # Start the executor locally.
  bash ${STARTUP_SCRIPT} ${MASTER_IP} ${MODELDB} ${WORKER_AUTOSCALER} \
       > /tmp/start_worker.log 2>&1 < /dev/null &
else
  ssh -o StrictHostKeyChecking=no -i ${KEY_FULL} "ubuntu@"${PUBLIC_DNS} \
    -p 22 \
    "nohup bash ${STARTUP_SCRIPT} ${MASTER_IP} ${MODELDB} ${WORKER_AUTOSCALER} \
     > /tmp/start_worker.log 2>&1 < /dev/null &"
fi

# Echo for caller to grab IP
echo ${exec_ip}

exit 0

