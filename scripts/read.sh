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

if [ $# -eq 0 ]; then
    echo "Usage: $0 <pravega_stream>"
    exit 1
fi
PRAVEGA_STREAM="$1"

ROOT_DIR=/gstreamer-pravega
pushd ${ROOT_DIR}/gst-plugin-pravega
#cargo build
ls -lh ${ROOT_DIR}/target/debug/*.so
export GST_PLUGIN_PATH=${ROOT_DIR}/target/debug:${GST_PLUGIN_PATH}
# log level can be INFO, DEBUG, or LOG (verbose)
export GST_DEBUG=pravegasrc:DEBUG,basesink:INFO
export RUST_BACKTRACE=1
export TZ=UTC
PRAVEGA_CONTROLLER_URI=${PRAVEGA_CONTROLLER_URI:-172.28.1.1:9090}
FPS=30


gst-launch-1.0 \
-v \
pravegasrc stream=examples/${PRAVEGA_STREAM} controller=${PRAVEGA_CONTROLLER_URI} allow-create-scope=true start-mode=earliest end-mode=latest \
! decodebin \
! autovideosink
