INFaaS DockerFiles

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
docker run --rm -it -p 8000:8000 --name=mymodel --ipc=host -v/tmp/model:/tmp/model infaaspytorch /workspace/container_start.sh pytorch_container.py 224 mymodel.pt 8000
```

