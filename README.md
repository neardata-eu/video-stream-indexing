# StreamSense
A policy-driven semantic video search solution that exploits tiered storage in streaming systems.

StreamSense has been presented at the 25th ACM/IFIP International Middleware Conference. The paper can be found [here](https://dl.acm.org/doi/10.1145/3700824.3701097).

## Abstract
Streaming systems are an increasingly appealing substrate for managing video data via the stream abstraction. However, if we consider a large stream collection, it can be hard for data scientists to discover and locate relevant videos, let alone specific video fragments. In this paper, we propose StreamSense: a policy-driven, semantic video search solution for streaming systems. StreamSense allows users to deploy AI models that generate embeddings from video frames via policies. Our system uses such embeddings for building a two-level index in a vector DB that efficiently handles inter/intra video queries. StreamSense abstracts users from vector DB interactions so they can perform semantic search using images as input and visualize the results. We built our prototype on top of a tiered streaming storage system (Pravega) and validated it on a health-related use case. We show that StreamSense allows data scientists to search for video fragments in real surgery datasets in < 30ms. StreamSense also reduces data ingestion related to AI training data loading in +80% compared to simple bulk loading video streams.

## StreamSense CLI (a.k.a. Indexer Controller)

To launch the StreamSense CLI, run the following command:

``` bash
python3 streamsense/streamsense-cli.py
```

The CLI will prompt you and you can start entering commands:

![CLI image](/media/cli.png)

When generating embeddings, the CLI will automatically launch a pod on your previously configured Kubernetes cluster. In that pod, the video will be read from the Pravega stream, generating the two-level index and storing the embeddings in the Milvus database.

When querying the system, the CLI will run the query and return the results.

# Evaluation
To reproduce the evaluation results, you need to run StreamSense manually. The CLI is designed to be user-friendly and may not collect/output all the evaluation metrics.

## Setup

The evaluation was performed on a AWS EC2 cluster of 4 nodes

* 1 VM for Pravega (i3en.2xlarge).
* 1 VM for Milvus (m5.2xlarge).
* 1 VM with GPU to perform the inference (p3.2xlarge).
* 1 VM for the client to generate the video stream (c5.4xlarge). 

## Installation

In order to deploy the environment to run this project, plase refer to ['/deploy'](/deploy/README.md).

Notice that StreamSense can be deployed locally on a single machine to develop and test the system. To deploy it in a local environment, plase refer to ['/deploy/local'](/deploy/local/README.md).

# Instructions

### 0. Prepare environment
 - Prepare the setup as described previously. You can either use the full deployment or the local deployment.
 - Make sure to assign the correct IPs in `streamsense/policies/constants.py`.

### 1. Prepare Docker container
 - We recommend running all the scripts with the following Docker image. 
 - This container contains all the required dependencies to run all of the steps. 
 - It is recommended to add gpu to the Inference/Indexing container via the ```--gpus all``` flag.
 - The image can also be built or modified in the `/docker` folder.

```
docker run -it -v /{path-to-repo}/video-stream-indexing/:/project --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" arnaugabriel/video-indexing:2.0 bash
```

### 2. Video Ingestion
 - This step performs a simulation of a real surgery by reading a local mp4 file and sending it as a stream to Pravega. 
 - This process should run on the client VM (low hardware requirements).
 - Navigate to `/project/streamsense/ingestion` and then run the following script:
```
bash ingestion.sh /project/videos/<video_name>.mp4 <stream_name> <fps>
```

 - Optionally, once the video is written we can read and visualize the video stream.
 - Navigate to `/project/streamsense/ingestion` and then run the following script:
```
bash read.sh <stream_name>
```

### 3. Inference and Indexing

 - This step reads the stream from the previous step, generates the embeddings from the key video frames and sends them to Milvus. 
 - This process can be run at the same time as the previous one to simulate a live surgery or afterwards for batch indexing. 
 - This process should run on the VM with GPU support.
 - Navigate to `/project/streamsense/indexing` and then run the following script:

```
GST_PLUGIN_PATH=/gstreamer-pravega/target/debug:${GST_PLUGIN_PATH} python3 inference.py --stream <stream_name>
```

### 4. Perform a query to the system

 - This steps showcases the query capabilities of our system. We recommend adding multiple videos from different surgerys in order to appreciate the results.
 - This process can run on any VM, but it is recommended to run on the node with GPU support.
 - The following example performs an inter-video and intra-video query to our system.
 - Navigate to `/project/streamsense/benchmarks/experiment4` and then run `inter_video_search.py` and `intra_video_search.py`.

 - The `/project/streamsense/pytorch_examples` folder contains examples showcasing the use within a DataLoader class in order to generate a DataSet to train a PyTorch model.
   
 - The video fragments generated can be visualized with the following command:
```
vlc <fragment_name>.h264 --demux h264
```

# Video Demo

The following demo showcases all of the pipeline steps.

![Video Demo](/media/demo.mp4)

![Surveillance Demo](/media/surveillance.mp4)
