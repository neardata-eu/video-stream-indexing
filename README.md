# StreamSense
A policy-driven semantic video search solution that exploits tiered storage in streaming systems.

# Setup

While everything can be run on the same machine, we recommend the following 4 machine setup:

* 1 VM for Pravega.
* 1 VM for Milvus.
* 1 VM with GPU to perform the inference.
* 1 VM for the client to generate the video stream. 

## Installation

In order to deploy the environment to run this project, plase refer to ['/deploy'](https://github.com/neardata-eu/video-stream-indexing/tree/main/deploy/README.md).

Alternatively, to deploy it in a local environment, plase refer to ['/deploy/local'](https://github.com/neardata-eu/video-stream-indexing/tree/main/deploy/local/README.md).

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
 - Navigate to `/project/streamsense/query` and then run the following script:
```
python3 milvus_demo.py
```

 - The following example showcases the use withing a DataLoader class in order to generate a DataSet to train a PyTorch model.
 - Navigate to `/project/streamsense/query` and then run the following script:
```
python3 pytorch_example.py
```

 - The video fragments generated can be visualized with the following command:
```
vlc <fragment_name>.h264 --demux h264
```

# Video Demo

The following demo showcases all of the pipeline steps.

[Video Demo](https://github.com/ArnauGabrielAtienza/video-stream-indexing/blob/main/media/demo.mp4)