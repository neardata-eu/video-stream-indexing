#!/usr/bin/env bash

#
# Copyright (c) Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

set -ex

if [ $# -ne 1 ]; then
    echo "Usage: $0 <pravega_stream>"
    exit 1
fi

eval "$(python3 /project/policies/constants.py)"
ROOT_DIR=/gstreamer-pravega
pushd ${ROOT_DIR}/gst-plugin-pravega
export GST_PLUGIN_PATH=${ROOT_DIR}/target/debug:${GST_PLUGIN_PATH}
export GST_DEBUG=pravegasink:DEBUG,basesink:INFO
export RUST_BACKTRACE=1
export TZ=UTC
PRAVEGA_STREAM=$1
SIZE_SEC=120
FPS=30
KEY_FRAME_INTERVAL=$((1*$FPS))

gst-launch-1.0 \
-v \
videotestsrc name=src is-live=true do-timestamp=true num-buffers=$(($SIZE_SEC*$FPS)) \
! "video/x-raw,format=RGB,width=800,height=600,framerate=${FPS}/1" \
! videoconvert \
! x264enc tune=fastdecode speed-preset=ultrafast key-int-max=${KEY_FRAME_INTERVAL} \
! timestampcvt input-timestamp-mode=start-at-current-time \
! pravegasink stream=examples/${PRAVEGA_STREAM} controller=${PRAVEGA_CONTROLLER} allow-create-scope=true seal=true sync=false timestamp-mode=tai buffer-size=1024