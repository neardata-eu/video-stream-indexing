FROM ghcr.io/streamstorage/gstreamer:22.04-1.22.6-0.11.1-dev

# Set non-interactive mode
# ENV DEBIAN_FRONTEND=noninteractive

# Update apt sources and upgrade packages
RUN sed -i 's/# deb-src/deb-src/g' /etc/apt/sources.list
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install -y --no-install-recommends --option=Dpkg::Options::=--force-confdef \
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
        libgtk-3-dev \
        ffmpeg

RUN rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# Clone repository
RUN git clone -b flexible-index https://github.com/pravega/gstreamer-pravega /gstreamer-pravega

# Install Rustup
RUN rustup component add rustfmt

# Cargo builds
ENV ROOT_DIR=/gstreamer-pravega
WORKDIR ${ROOT_DIR}/gst-plugin-pravega
RUN cargo build
WORKDIR ${ROOT_DIR}/apps
RUN cargo build

# Clean up
RUN apt-get clean

# Set working directory
WORKDIR /project/scripts
