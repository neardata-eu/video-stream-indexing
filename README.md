## To start/stop pravega

Please install docker & docker-compose as prerequisite and run the following commands on Linux host to start/stop pravega instance.

Start pravega:
```
PRAVEGA_LTS_PATH=/opt/docker/pravega_lts ./pravega-docker/up.sh
```

Stop pravega:
```
./pravega-docker/down.sh
```

## To start/stop Milvus

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

To start the ATTU GUI. This is a browser interface to easily manage the collections (optional):
```
docker run -p 8000:3000 -e MILVUS_URL=172.17.0.1:19530 zilliz/attu:v2.3.10
```

## Run pipeline
### 1) Start docker container
Setup docker container. Ideally, start one to perform the Video Ingestion and another one to perform the Inference+Indexing. It is also recommended to add gpu to the Inference container via ```--gpus all```:
```
xhost +
docker run -it -v /{path-to-repo}/video-stream-indexing/:/project --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" arnaugabriel/video-indexing:latest bash
```

### 2) Video Ingestion
Perform a simulation o a real surgery by reading a local mp4 file and sending it as a stream to Pravega:
```
cd /project/scripts/ingestion
bash ingestion.sh /project/videos/<video_name>.mp4 <stream_name> <fps>
```

Display pravega stream to screen (optional):
```
cd /project/scripts/ingestion
bash read.sh <stream_name>
```

### 3) Inference and Indexing
Read the previously created stream, generate the embeddings from the key frames and send them to Milvus.

```
cd /project/scripts/inference
GST_PLUGIN_PATH=/gstreamer-pravega/target/debug:${GST_PLUGIN_PATH} python3 inference.py --stream <stream_name>
```

### 4) Perform a Milvus Query

Perform a query to our system by giving a sample image. 
```
cd /project/scripts/query
python3 milvus_demo.py
```

Read video segment (to read the generated video segments):
```
vlc <stream_name>.h264 --demux h264
```