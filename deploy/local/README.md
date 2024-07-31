## Local Pravega Installation

Please install docker & docker-compose as prerequisite and run the following commands on Linux host to start/stop pravega instance. 

Start pravega:
```
PRAVEGA_LTS_PATH=/opt/docker/pravega_lts ./up.sh
```

Stop pravega:
```
./down.sh
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