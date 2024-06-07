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
ROOT_DIR=/gstreamer-pravega
pushd ${ROOT_DIR}/gst-plugin-pravega
cargo build
ls -lh ${ROOT_DIR}/target/debug/*.so
export GST_PLUGIN_PATH=${ROOT_DIR}/target/debug:${GST_PLUGIN_PATH}
# log level can be INFO, DEBUG, or LOG (verbose)
export GST_DEBUG=pravegasink:DEBUG,basesink:INFO
export RUST_BACKTRACE=1
export TZ=UTC
PRAVEGA_CONTROLLER_URI=${PRAVEGA_CONTROLLER_URI:-172.28.1.1:9090}
PRAVEGA_STREAM="videotest"
SIZE_SEC=600
FPS=25
KEY_FRAME_INTERVAL=$((1*$FPS))

gst-launch-1.0 \
-v \
videotestsrc name=src is-live=true do-timestamp=true num-buffers=$(($SIZE_SEC*$FPS)) \
! "video/x-raw,format=YUY2,width=940,height=540,framerate=${FPS}/1" \
! videoconvert \
! timeoverlay valignment=bottom "font-desc=Aria 48px" shaded-background=true \
! clockoverlay "font-desc=Aria 48px" "time-format=%F %T" shaded-background=true \
! videoconvert \
! queue \
! x264enc tune=zerolatency key-int-max=${KEY_FRAME_INTERVAL} \
! queue \
! timestampcvt input-timestamp-mode=start-at-current-time \
! pravegasink stream=examples/${PRAVEGA_STREAM} controller=${PRAVEGA_CONTROLLER_URI} allow-create-scope=true seal=true sync=false timestamp-mode=tai