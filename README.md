<img src="https://infaas-logo.s3-us-west-2.amazonaws.com/infaas_logo.png" width="200">

INFaaS is an inference-as-a-service platform that makes inference accessible and easy-to-use by abstracting resource management and model selection.
Users simply specify their inference task along with any performance and accuracy requirements for queries.

## Features
- A simple-to-use API for submitting queries and performance objectives
- A fully managed infrastructure for sharing across both models *and* users
- Automatic scaling in response to user demand
- Support for models built with TensorFlow, PyTorch, Caffe2, and TensorRT
- Support for inference on CPUs, GPUs, and [Inferentia](https://aws.amazon.com/machine-learning/inferentia/) (with support for other hardware platforms coming soon)

## Getting Started
INFaaS runs on [AWS](https://aws.amazon.com/) (with other provider platform support coming soon).

### One-time Setup
There are a few AWS-specific setup steps, all of which can be accomplished from the AWS dashboard:
1. Create an IAM Role.
Go to IAMs -> Roles, and create an EC2 role with policies for `AmazonEC2FullAccess`, `AmazonS3FullAccess` and `AdministratorAccess`.
2. Create a Security Group.
Go to EC2 -> Security Groups, and allow for all inbound traffic *from your desired trusted IP addresses or domains* and all outbound traffic.
3. Create an INFaaS Model Repository.
Go to S3, and create a bucket.
We recommend to keep this bucket private, since only the trusted INFaaS infrastructure will have access to it.
4. Create a model configuration bucket.
Again, go to S3, and create a bucket.
This bucket will hold profiled configurations used when models are registered (details in the [**Model Registration**](#model-registration) section).
INFaaS will need access to this bucket when registering a model.

### General Setup
1. Create an [AWS EC2](https://aws.amazon.com/ec2/) instance, which will serve as the master node.
In our experiments, we use an `m5.2xlarge` instance.
We provide a public AMI (ami-036de08e2e59b4abc) in *us-west-2* (that you can [copy to your region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/CopyingAMIs.html)) that contains the pre-installed dependencies.
The instance should have the IAM Role and Security Group you created in the [**One-time Setup**](#one-time-setup) attached to it.
2. If you don't use our AMI (which already has INFaaS's directory set up), clone the INFaaS repository: `git clone https://github.com/stanford-mast/INFaaS.git`.
3. Open `start_infaas.sh` and fill in the following entries. Entries between <> must be filled in prior to using INFaaS; the rest are set to defaults which can be changed based on your desired configuration.
    ```
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
    SLACK_GPU=0 # Used for making popular GPU variants exclusive, set to 0 for no GPU to be used as exclusive
    KEY_NAME='worker_key'
    MACHINE_TYPE_GPU='p3.2xlarge'
    MACHINE_TYPE_CPU='m5.2xlarge'
    MACHINE_TYPE_INFERENTIA='inf1.2xlarge'
    DELETE_MACHINES='2' # 0: VM daemon stops machines; 1: VM daemon deletes machines; 2: VM daemon persists machines, but removes them from INFaaS's view
    ```
    **Note:** If you would like to run the example below, you can either set CONFIGDB to be *infaas-sample-public/configs* or copy its contents over to your own configuration bucket using the [AWS CLI](https://docs.aws.amazon.com/cli/latest/reference/s3/cp.html):
    ```
    aws s3 sync s3://infaas-sample-public/ s3://your-config-bucket/ --exclude "resnet*"
    ```
3. Run `./start_infaas.sh` from the INFaaS home directory (i.e., the directory that `start_infaas.sh` is located in).
This will set up all INFaaS components and initial workers, as well as run some basic tests to check that the system is properly set up.
If a worker fails to start after the timeout period, you can simply rerun the script.
All executables can be found in `build/bin`.

### Model Registration
Currently, users must profile their models and generate a configuration file that can be passed to `infaas_modelregistration`.
We plan to make this process more automated in the future, but for now:
- Go to `src/profiler`
- Run `./profile_model.sh <frozen-model-path> <accuracy> <dataset> <task> [cpus]`
The script is interactive and will prompt you for information needed to profile your model.
Once complete, it will output a configuration (.config) file.
Upload this configuration file to your configuration bucket configured in the [**One-time Setup**](#one-time-setup). Here is how you would do this with the [AWS CLI](https://docs.aws.amazon.com/cli/latest/reference/s3/cp.html):
  ```
  aws s3 cp mymodel.config s3://your-config-bucket/mymodel.config
  ```
- Pass the .config (e.g., *mymodel.config*, not *my-config-bucket/mymodel.config*) to `infaas_modelregistration` as the second parameter.

**Example**

In `infaas-sample-public`, we have provided a CPU TensorFlow model, an equivalent TensorRT model optimized for batch-4, and an equivalent Inferentia model optimized for batch-1 on a single Inferentia core. We also provide their respective configuration files that were generated as specified above using `src/profiler/profile_model.sh`. Register these models as follows:
```
./infaas_modelregistration resnet_v1_50_4.config infaas-sample-public/resnet_v1_50_4/
./infaas_modelregistration resnet50_tensorflow-cpu_4.config infaas-sample-public/resnet50_tensorflow-cpu_4/
./infaas_modelregistration resnet50_inferentia_1_1.config infaas-sample-public/resnet50_inferentia_1_1/
```
If INFaaS is set up correctly, all of these commands should output a *SUCCEEDED* message.

### Inference Query
Information about registered models:
- To see the available model architectures for a given task and dataset, use `infaas_modarch`.
- To get more information on the model-variants for a model architecture, use `infaas_modinfo`.

**Example**

To see information about the models you registered in the Model Registration example, run `./infaas_modarch classification imagenet`, which should show that *resnet50* is the only registered model architecture.

Running `./infaas_modinfo resnet50` should show the three model-variants you registered: *resnet_v1_50_4*, *resnet50_tensorflow-cpu_4*, *resnet50_inferentia_1_1*.

Running queries:
- To run an online query, use `infaas_online_query`.
Running this with no parameters describes the valid input configurations (corresponding with the *model-less* abstraction, which you can read about more in the second reference paper below).
INFaaS returns the raw output from the model (e.g., output probabilities for each class).
- To run an offline query, use `infaas_offline_query`.
INFaaS returns whether the job scheduling was successful.
If successfully scheduled, the job can be monitored by checking the `output_url` bucket.

**Example**

Note: to run this example, you must have called `./start_infaas.sh` with at least one GPU worker (i.e., *NUM_INIT_GPU_WORKERS* >= 1 and *MAX_GPU_WORKERS* >= *NUM_INIT_GPU_WORKERS*) and one Inferentia worker (i.e., *NUM_INIT_INFERENTIA_WORKERS* >= 1 and *MAX_INFERENTIA_WORKERS* >= *NUM_INIT_INFERENTIA_WORKERS*).

Let's send an online image classification query to INFaaS and specify the model architecture with a latency constraint. After you have registered the three ResNet50 models from the above Model Registration example, we can first send the request with a relaxed latency constraint (assuming you are running in `build/bin` for the image path to work):
```
./infaas_online_query -d 224 -i ../../data/mug_224.jpg -a resnet50 -l 300
```
The first time you run the query, the latency will be on the order of seconds, since the model needs to be loaded before it can be run. If you rerun the query, it should complete much faster (in hundreds of milliseconds). INFaaS uses *resnet50_tensorflow-cpu_4* to service this query since it is sufficient to the latency requirements.

Now, let's send a query with a stricter latency requirement:
```
./infaas_online_query -d 224 -i ../../data/mug_224.jpg -a resnet50 -l 50
```
Again, the first time you run this query, the latency will be on the order of seconds (or you may even get a Deadline Exceeded message if it's longer than 10 seconds). Inferentia models can take longer to load and set up, which INFaaS accounts for in its scaling algorithm. If you rerun the query, it should complete in milliseconds. INFaaS uses *resnet50_inferentia_1_1* to service this query, since, despite being loaded, *resnet50_tensorflow-cpu_4* cannot meet the performance requirements you specified.

Finally, let's send a batch-2 query with a strict latency requirements:
```
./infaas_online_query -d 224 -i ../../data/mug_224.jpg -i ../../data/mug_224.jpg -a resnet50 -l 20
```
Again, the first time you run this query, the latency will be on the order of seconds (or you may even get a Deadline Exceeded message if it's longer than 10 seconds). Similar to Inferentia models, GPU models take longer to load and set up. If you rerun the query, it should complete in milliseconds. INFaaS uses *resnet_v1_50_4* to service this query, since (a) *resnet50_tensorflow-cpu_4* supports the batch size but not the latency requirement, and (b) *resnet50_inferentia_1_1* only supports batch-1 and cannot meet the latency requirement.

You can also simply specify a use-case to INFaaS with a latency and accuracy requirement. For example:
```
./infaas_online_query -d 224 -i ../../data/mug_224.jpg -t classification -D imagenet -A 70 -l 50
```

### Clean Up
Update the following two parameters in `shutdown_infaas.sh`:
```
REGION='<REGION>'
ZONE='<ZONE>'
```
Then, run `./shutdown_infaas.sh`.
You will be prompted on whether you would like to delete or shut down existing worker nodes.
Once this completes, all running INFaaS processes will be shut down on the master, in addition to workers being shut down or deleted (depending on what you inputted).

### Contributing
To file a bug, ask a question, or request a feature, please file a GitHub issue.
Pull requests are welcome.

### Reference
For details about INFaaS, please refer to the following two papers. We kindly ask that you cite them should you use INFaaS in your work.
- Neeraja J. Yadwadkar, Francisco Romero, Qian Li, Christos Kozyrakis. [A Case for *Managed* and *Model-less* Inference Serving](https://dl.acm.org/citation.cfm?id=3321443). In *Workshop on Hot Topics in Operating Systems* (HotOS), 2019.
- Francisco Romero*, Qian Li*, Neeraja J. Yadwadkar, Christos Kozyrakis. [INFaaS: Automated *Model-less* Inference Serving](https://www.usenix.org/conference/atc21/presentation/romero). In *USENIX Annual Technical Conference* (ATC), 2021.
