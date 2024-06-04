## To start/stop pravega

Please install docker & docker-compose as prerequisite and run the following commands on Linux host to start/stop pravega instance.
Consider to set PRAVEGA_LTS_PATH env variable if to change the pravega tier 2 storage path.

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

To start the ATTU GUI:
```
docker run -p 8000:3000 -e MILVUS_URL=172.17.0.1:19530 zilliz/attu:v2.3.10
```

## Run pipeline
Setup docker container (change `/mnt/data/projects/video-stream-indexing/` accordingly to the location of this repository):
```
xhost +
docker run -it -v /mnt/data/projects/video-stream-indexing/:/project --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" arnaugabriel/video-indexing:latest bash
```

Ingest local video to pravega
```
bash ingestion.sh /project/videos/sample.mp4 <stream_name>
```

Display pravega video to screen
```
bash read.sh <stream_name>
```

Perform inference
```
GST_PLUGIN_PATH=/gstreamer-pravega/target/debug:${GST_PLUGIN_PATH} python3 inference.py --stream <stream_name>
```

Export video segment:
```
BEGIN_OFFSET=474795 END_OFFSET=1651706 bash export.sh <stream_name>
```

Read video segment:
```
vlc <stream_name>.h264 --demux h264
```