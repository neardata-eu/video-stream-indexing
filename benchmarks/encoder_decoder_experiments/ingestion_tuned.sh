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

if [ $# -ne 3 ]; then
    echo "Usage: $0 <video_path> <pravega_stream> <fps>"
    exit 1
fi
FILESRC_PATH="$1"
PRAVEGA_STREAM="$2"
FPS="$3"

eval "$(python3 /project/policies/constants.py)"
ROOT_DIR=/gstreamer-pravega
pushd ${ROOT_DIR}/gst-plugin-pravega
export GST_PLUGIN_PATH=${ROOT_DIR}/target/debug:${GST_PLUGIN_PATH}
export GST_DEBUG=pravegasink:DEBUG,basesink:INFO
export RUST_BACKTRACE=1
export TZ=UTC
KEY_FRAME_INTERVAL=$((1*$FPS))

gst-launch-1.0 \
-v \
filesrc location="${FILESRC_PATH}" \
! qtdemux \
! h264parse \
! avdec_h264 \
! videoconvert \
! videoscale \
! video/x-raw,width=940,height=560 \
! x264enc tune=fastdecode speed-preset=ultrafast key-int-max=${KEY_FRAME_INTERVAL} \
! timestampcvt input-timestamp-mode=start-at-current-time \
! pravegasink stream=${PRAVEGA_SCOPE}/${PRAVEGA_STREAM} controller=${PRAVEGA_CONTROLLER} allow-create-scope=true seal=true sync=false timestamp-mode=tai buffer-size=1024
