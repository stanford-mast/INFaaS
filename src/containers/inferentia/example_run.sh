#!/bin/bash

# first, start the rtd container
# docker run --env AWS_NEURON_VISIBLE_DEVICES="0" --cap-add SYS_ADMIN --cap-add IPC_LOCK -v /tmp/neuron_rtd_sock/:/sock -it qianl15/neuron-rtd:latest

# then, start the app container
docker run -t  --env NEURON_RTD_ADDRESS=unix:/sock/neuron.sock \
      -v /tmp/neuron_rtd_sock/:/sock \
      --env AWS_NEURON_VISIBLE_DEVICES="0" neuron-tf ./infer_resnet50.py

