### INFaaS DockerFiles

To build *from within this directory*:
```
docker build -t infaaspytorch -f PyTorchDocker ../
```

To build *from the INFaaS root directory*
```
docker build -t infaaspytorch -f dockerfiles/PyTorchDocker ./
```

To run using the docker cli:
```
docker run --rm -it -p 8000:8000 --name=<name> --ipc=host -v/tmp/model:/tmp/model infaaspytorch /workspace/container_start.sh pytorch_container.py <scale> <serialized-model-filename> <port>
<scale>: image scale size, (e.g., 224)
<serialized-model-filename>: the name of the serialized model file, with the extension. The model should sit in /tmp/model, as this is the mapped volume between the host and container.
```

Example:
```
docker run --rm -it -p 8000:8000 --name=mymodel --ipc=host -v/tmp/model:/tmp/model infaaspytorch /workspace/container_start.sh pytorch_container.py 224 mymodel 8000
```

### Inferentia Docker

To build the Neuron runtime daemon container from within this directory:
```
docker build -t neuron-rtd -f Dockerfile.neuron-rtd  ./
```

To build the Neuron tensorflow container from within this directory:
```
docker build -t neuron-tf -f Dockerfile.tf-python ../
```

I have pre-built and make the container images available on docker hub:
```
docker pull qianl15/neuron-rtd:latest
```

Similar to `qianl15/neuron-tf:latest`

#### Run Inferentia apps in Docker
1. Follow steps to set up environment. Stop host neuron-rtd.
https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-container-tools/tutorial-docker.md

2. Follow these steps to start two containers. https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-container-tools/docker-example/README.md

An example to run (note that you cannot start more app containers than the number of Neuron cores):
```bash
  mkdir -p /tmp/neuron_rtd_sock
  chmod o+rwx /tmp/neuron_rtd_sock
  docker run --rm -it -d --env AWS_NEURON_VISIBLE_DEVICES="0" --cap-add SYS_ADMIN --cap-add IPC_LOCK -v /tmp/neuron_rtd_sock/:/sock -it qianl15/neuron-rtd:latest
  docker run --rm -it -d  --env NEURON_RTD_ADDRESS=unix:/sock/neuron.sock \
         -v /tmp/neuron_rtd_sock/:/sock \
         --env AWS_NEURON_VISIBLE_DEVICES="0" qianl15/neuron-tf:latest ./infer_resnet50.py
```

### GNMT-nvpy Docker

This docker container is used to run translate model, [GNMT-nvpy](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT), a RNN trained with NVIDIA optimized PyTorch.

To build *from within this directory*:
```
docker build -t gnmt-infaas -f Dockerfile.gnmt-nvpy ../
```

We have pre-built and make the container images available on docker hub:
```
docker pull qianl15/gnmt-infaas:latest
```

This container can be used with or without GPU; to use GPU, simply replacing `docker` with `nvidia-docker`.
To run using the docker cli:
```
docker run --init -it --rm --ipc=host -v /tmp/models:/tmp/model  -p <port>:<port> gnmt-infaas:latest ./gnmt_container.py [arguments]

optional arguments:
  -h, --help           show this help message and exit
  --cuda               Use cuda (GPU) for inference.
  --no-cuda            Use CPU for inference.
  --math {fp16,fp32}   Precision. FP16 only supported on GPU.
  --beam-size {1,2,5}  Beam size (e.g., 1, 2, 5)
  --model MODEL        Model name to load
  --port PORT          Port to listen on
```

CPU Example:
```
docker run --init -it --rm --ipc=host -v /tmp/models:/tmp/model  -p 9001:9001 gnmt-infaas:latest ./gnmt_container.py --model gnmt_ende4_cpu_fp32_2 --no-cuda --math fp32 --port 9001 --beam-size 2
```
Note: 

GPU Example:
```
nvidia-docker run --init -it --rm --ipc=host -v /tmp/models:/tmp/model  -p 9001:9001 gnmt-infaas:latest ./gnmt_container.py --model gnmt_ende4_gpu_fp16_2 --cuda --math fp16 --port 9001 --beam-size 2
```
Note: gnmt_ende4_cpu_fp32_2 means a CPU model with FP32 and beam size 2; gnmt_ende4_gpu_fp32_2 means a GPU model with FP16 and beam size 2. Although both variants have the same checkpoint file: model_best.pth.
gnmt = Google neural machine translation;
ende4 = English to German, 4-layer LSTM.
The model has BLEU score 24.45 (FP32)	and 24.48 (FP16).

