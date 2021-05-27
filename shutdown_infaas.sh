#!/bin/bash -e

# Constants
REGION='<REGION>'
ZONE='<ZONE>'
WORKER_PREFIX="infaas-worker"

# Ask user if they want workers to be deleted
echo -n "Do you want workers to be deleted? [y/N]: "
read yn_del
if [[ $yn_del == y ]] || [[ $yn_del == Y ]]; then
  # First get all workers
  all_workers=`aws ec2 describe-instances --filter 'Name=tag:Name,Values='${WORKER_PREFIX}'*' --query 'Reservations[*].Instances[*].InstanceId' --output text`
  echo "Now deleting: "${all_workers}
  aws ec2 terminate-instances \
    --instance-ids ${all_workers}
else
  echo -n "Do you want workers to be stopped/paused? [y/N]: "
  read yn_stop
  if [[ $yn_stop == y ]] || [[ $yn_stop == Y ]]; then
    # First get all workers
    all_workers=`aws ec2 describe-instances --filter 'Name=tag:Name,Values='${WORKER_PREFIX}'*' --query 'Reservations[*].Instances[*].InstanceId' --output text`
    echo "Now stopping: "${all_workers}
    aws ec2 stop-instances \
      --instance-ids ${all_workers}
  fi
fi

# Kill modelreg_server
pkill -f modelreg_server
echo "Model registration server shut down"

# Kill queryfe_server
pkill -f queryfe_server
echo "Query Frontend server shut down"

# Kill monitoring daemon
pkill -f master_vm_daemon
echo "VM Scaling daemon shut down"

# Kill redis-server
pkill -f redis-server
echo "Redis server shut down"

echo "INFaaS has successfully shut down!"

exit 0

