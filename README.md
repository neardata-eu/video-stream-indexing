# StreamSense
Integration of Pravega and Milvus for NCT computer-assisted surgery use-case.

# Setup

While everything can be run on the same machine, we recommend the following 4 machine setup:

* 1 VM for Pravega.
* 1 VM for Milvus.
* 1 VM with GPU to perform the inference.
* 1 VM to read the surgery video. 

## Pravega Installation

This shows how to start the local Docker Compose version of Pravega. For other types of deployments, refer to [here](https://github.com/pravega/pravega/blob/master/documentation/src/docs/deployment/deployment.md).

Please install docker & docker-compose as prerequisite and run the following commands on Linux host to start/stop pravega instance. 

Start pravega:
```
PRAVEGA_LTS_PATH=/opt/docker/pravega_lts ./pravega-docker/up.sh
```

Stop pravega:
```
./pravega-docker/down.sh
```

## Milvus Installation

Install Milvus
```
wget https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh
```

Start Milvus
```
bash standalone_embed.sh start
```

Stop Milvus
```
bash standalone_embed.sh stop
```

Delete Milvus
```
bash standalone_embed.sh delete
```

To start the ATTU GUI. This is a browser interface to easily manage the Milvus collections and indexes (optional):
```
docker run -p 8000:3000 -e MILVUS_URL=172.17.0.1:19530 zilliz/attu:v2.3.10
```

# Instructions
### 1. Start docker container
Setup docker container. This container contains all the required dependencies to run the entire pipeline. It is recommended to add gpu to the Inference container via the ```--gpus all``` flag.

```
xhost +
docker run -it -v /{path-to-repo}/video-stream-indexing/:/project --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" arnaugabriel/video-indexing:2.0 bash
```

### 2. Video Ingestion
Perform a simulation o a real surgery by reading a local mp4 file and sending it as a stream to Pravega. This process should run on the VM that reads the surgery video.
```
cd /project/scripts/ingestion
bash ingestion.sh /project/videos/<video_name>.mp4 <stream_name> <fps>
```

Display pravega stream to screen (optional):
```
cd /project/scripts/ingestion
bash read.sh <stream_name>
```

### 3. Inference and Indexing
Read the previously created stream, generate the embeddings from the key frames and send them to Milvus. This process should run on the VM with GPU support.

```
cd /project/scripts/inference
GST_PLUGIN_PATH=/gstreamer-pravega/target/debug:${GST_PLUGIN_PATH} python3 inference.py --stream <stream_name>
```

### 4. Perform a query to the system

Perform a query to our system by giving a sample image. This process can run on any VM, but it is recommended to run on the node with GPU support.
```
cd /project/scripts/query
python3 milvus_demo.py
```

To run the PyTorch/DataLoader example:
```
cd /project/scripts/query
python3 pytorch_example.py
```

Read video segment (to read the generated video segments):
```
vlc <stream_name>.h264 --demux h264
```

# Video Demo

[Video Demo](https://github.com/ArnauGabrielAtienza/video-stream-indexing/blob/main/media/demo.mp4)