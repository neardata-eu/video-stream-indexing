public_key_path   = "~/.ssh/benchmark_key.pub"
region            = "us-east-1"
availability_zone = "us-east-1a"
ami_ubuntu        = "ami-053b0d53c279acc90" // Ubuntu 22.04
ami_gpu           = "ami-078d2afa1e02a4aa6" // Ubuntu 20.04 with NVIDIA drivers
# ami = "ami-004fac3d4533a2541" // RHEL-7.9

instance_types = {
  "bookkeeper" = "i3en.2xlarge"
  "client"     = "c5.4xlarge" //"m5.large" 
  "milvus"     = "m5.2xlarge"
  "gpu"        = "p3.2xlarge"

# Not currently used
  "controller" = "m5.large"
  "zookeeper"  = "t2.small"
  "metrics"    = "t2.micro"
}

num_instances = {
  "bookkeeper" = 1
  "client"     = 1
  "gpu"        = 1
  "milvus"     = 1
  "metrics"    = 1

# Not currently used
  "controller" = 0
  "zookeeper"  = 0
}
