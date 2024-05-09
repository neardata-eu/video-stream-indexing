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
ROOT_DIR=$(readlink -f $(dirname $0)/..)
pushd ${ROOT_DIR}/apps
cargo build

#export RUST_LOG=${RUST_LOG:-info}
export RUST_BACKTRACE=1
PRAVEGA_CONTROLLER_URI=${PRAVEGA_CONTROLLER_URI:-172.28.1.1:9090}
PRAVEGA_SCOPE=${PRAVEGA_SCOPE:-examples}
PRAVEGA_STREAM=${PRAVEGA_STREAM:-urv6}

BEGIN_OFFSET=${BEGIN_OFFSET:-0}
END_OFFSET=${END_OFFSET:-0}

cargo run --bin pravega_stream_exporter -- \
--controller ${PRAVEGA_CONTROLLER_URI} \
--scope ${PRAVEGA_SCOPE} \
--stream ${PRAVEGA_STREAM} \
--begin-offset ${BEGIN_OFFSET} \
--end-offset ${END_OFFSET} \
--file-path ${ROOT_DIR}/urv/test.h264
