#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# supress outputs from pushd and popd
function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

echo "This script requires elevated privileges."
echo "It will ask for password on the first call to sudo."

base_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/../)

# Install dependencies
sudo apt install -y \
    autoconf automake build-essential ccache cmake cpufrequtils \
    doxygen ethtool g++ git inetutils-tools libboost-all-dev \
    libncurses5 libncurses5-dev libusb-1.0-0 libusb-1.0-0-dev \
    libusb-dev python3-dev python3-mako python3-numpy python3-requests \
    python3-scipy python3-setuptools python3-ruamel.yaml ninja-build

# ensure target dir path exist
mkdir -p "${base_dir}/ext/"

# Clone UHD repository
git clone https://github.com/EttusResearch/uhd.git "${base_dir}/ext/uhd"

pushd  "${base_dir}/ext/uhd/host"
mkdir build && cd build
cmake -DCMAKE_POLICY_DEFAULT_CMD0167=NEW -GNinja ..
ninja
ninja test
sudo ninja install
popd

# Download firmware images
sudo /usr/local/lib/uhd/utils/uhd_images_downloader.py

# Refresh the linker cache
sudo ldconfig

# Set USRP permission to non-root mode
sudo cp /usr/local/lib/uhd/utils/uhd-usrp.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
