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

if [ $# -ne 4 ]; then
    echo "Usage: $0 <pravega_stream> <output_file> <begin_offset> <end_offset>"
    exit 1
fi
PRAVEGA_STREAM="$1"
FILE_NAME="$2"
BEGIN_OFFSET="$3"
END_OFFSET="$4"

eval "$(python3 /project/streamsense/policies/constants.py)"
ROOT_DIR=/gstreamer-pravega
pushd ${ROOT_DIR}/apps
export RUST_BACKTRACE=1

cargo run --bin pravega_stream_exporter -- \
--controller ${PRAVEGA_CONTROLLER} \
--scope ${PRAVEGA_SCOPE} \
--stream ${PRAVEGA_STREAM} \
--begin-offset ${BEGIN_OFFSET} \
--end-offset ${END_OFFSET} \
--file-path ${FILE_NAME}
