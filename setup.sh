#!/bin/bash
set -ex

export DEBIAN_FRONTEND=noninteractive

sed -i 's/# deb-src/deb-src/g' /etc/apt/sources.list
apt-get update
apt-get dist-upgrade -y

apt-get install -y --no-install-recommends \
    bison \
    flex \
    g++ \
    git \
    libgirepository1.0-dev \
    libgl-dev \
    libpciaccess-dev \
    libpython3.10-dev \
    libsodium-dev \
    libsrtp2-dev \
    libssl-dev \
    libx11-xcb-dev \
    libx265-dev \
    make \
    nasm \
    pkg-config \
    python3-pip \
    wget \
    libgdk-pixbuf2.0-dev \
    libgtk-3-dev

pip3 install -r ./requirements.txt
apt-get clean
rm -rf /var/lib/apt/lists/*
git clone -b flexible-index https://github.com/pravega/gstreamer-pravega /gstreamer-pravega
rustup component add rustfmt