This readme shows the process to create and deploy a Pravega cluster, with the Milvus VM instance, a VM with a GPU to run the neural network and the client VM. 

## Prerequisites

### AWS credentials
AWS credentials are required to create the cluster. The credentials must be stored in the file `~/.aws/credentials`.


### SSH key
You’ll need to create both a public and a private SSH key at `~/.ssh/benchmark_key` (private) and `~/.ssh/benchmark_key.pub` (public), respectively.

``` bash
ssh-keygen -f ~/.ssh/benchmark_key
```


## Docker image with dependencies

To deploy the cluster we use the following image: `raulgracia/pravega-deploy-ec2:2.0`.

It is recommended to mount some folders to the container to have access to the scripts and the credentials.

``` bash
sudo docker run -it \
	-v /path/to/repo/:/root/benchmark \
	-v /home/$USER/.aws/:/root/.aws/ \
	-v /home/$USER/.ssh:/root/.ssh \
	raulgracia/pravega-deploy-ec2:2.0 
```

## Check the Terraform configuration
The number of nodes and the instance type can be configured in the file [terraform.tfvars](terraform.tfvars). Open the file and configure the number of nodes and the instance type.

``` bash
nano terraform.tfvars
```

## Create the cluster using Terraform

Once the container is running, we can create the cluster using Terraform.

``` bash
cd /root/benchmark/deploy
terraform init
echo "yes" | terraform apply
```


## Deploy Pravega cluster

Once the cluster is created, we can deploy Pravega using Ansible.

``` bash
# Fixes "terraform-inventory had an execution error: Error reading tfstate file: 0.12 format error"
export TF_STATE=./
ansible-playbook \
  --user ubuntu \
  --inventory `which terraform-inventory` \
  deploy-pravega-all-in-one.yaml
```


## Deploy the benchmark

We can deploy the remaining of the setup using the following command:

``` bash
ansible-playbook \
  --inventory `which terraform-inventory` \
  deploy-benchmark.yaml
```


## Tear down the cluster
Once you’re finished running your benchmarks, you should tear down the AWS infrastructure you deployed for the sake of saving costs. You can do that with one command:

``` bash
terraform destroy --force
```

Make sure to let the process run to completion (it could take several minutes). Once the tear down is complete, all AWS resources that you created for the Pravega benchmarking suite will have been removed.