## To start/stop pravega

Please install docker & docker-compose as prerequisite and run the following commands on Linux host to start/stop pravega instance.
Consider to set PRAVEGA_LTS_PATH env variable if to change the pravega tier 2 storage path.

```
PRAVEGA_LTS_PATH=/opt/docker/pravega_lts ./pravega-docker/up.sh
./pravega-docker/down.sh
```

## Run scripts in the docker container
```
# change `/mnt/data/projects/gstreamer-pravega/` accordingly to the location of this repository
xhost +
docker run -it -v /mnt/data/projects/gstreamer-pravega/:/scripts --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" ghcr.io/streamstorage/gstreamer:22.04-1.22.6-0.11.1-dev bash
cd /scripts

# install dependencies
./install_dependencies

# test video source to pravega
./ingestion.sh

# pravega to screen
./read.sh

# python inference job
GST_PLUGIN_PATH=../target/debug:${GST_PLUGIN_PATH} ./inference.py

# export video clips with offset
BEGIN_OFFSET=6986522 END_OFFSET=7498906 ./export.sh

# play the exported test.h264 file with vlc
# vlc test.h264 --demux h264
```