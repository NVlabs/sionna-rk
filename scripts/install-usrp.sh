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

function usage() {
    echo "Usage: $0 [-h|--help] [--force] [--ci] [--verbose] [--dry-run]"
    exit 1
}

function execute() {
    # Always show the command
    if [ "$VERBOSE" == "1" ]; then
        echo "Executing: $@"
    fi

    if [ "$DRYRUN" == "1" ]; then
        # only print the command
        echo "[DRY-RUN] $@"
    else
        # actually execute the command
        eval "$@"
        ret_val=$?
        if [ $ret_val -ne 0 ]; then
            echo "Command failed with exit code $ret_val"
            exit $ret_val
        fi
    fi
}

echo "This script requires elevated privileges."
echo "It will ask for password on the first call to sudo."

base_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/../)

# default values
FORCE=0
CI=0
VERBOSE=0
DRYRUN=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        --force) FORCE=1 ; shift ;;
        --ci) CI=1 ; shift ;;
        --verbose) VERBOSE=1 ; shift ;;
        --dry-run) DRYRUN=1 ; shift ;;
        *) shift ;; # ignore other arguments
    esac
done

if [ "$(which uhd_find_devices)" != "" ] && [ "$FORCE" == "0" ] && [ "$CI" == "0" ]; then
  echo "Found 'uhd_find_devices'. Assuming UHD is already installed. Skipping..."
  exit 0
fi

# source distro info
source /etc/lsb-release

LIBNCURSES=""
# check if we are on jammy
if [ "$DISTRIB_CODENAME" == "jammy" ]; then
    LIBNCURSES="libncurses5 libncurses5-dev"
fi

# check if we are on noble
if [ "$DISTRIB_CODENAME" == "noble" ]; then
    LIBNCURSES="libncurses6 libncurses-dev"
fi

if [ -z "$LIBNCURSES" ]; then
    echo "This script is only supported on Ubuntu 22.04 (Jammy Jellyfish) and Ubuntu 24.04 (Noble Narwhal)."
    usage
fi

# Install dependencies
execute sudo apt install -y \
    autoconf automake build-essential ccache cmake cpufrequtils \
    doxygen ethtool g++ git inetutils-tools libboost-all-dev \
    ${LIBNCURSES} libusb-1.0-0 libusb-1.0-0-dev \
    libusb-dev python3-dev python3-mako python3-numpy python3-requests \
    python3-scipy python3-setuptools python3-ruamel.yaml ninja-build

# ensure target dir path exist
execute mkdir -p "${base_dir}/ext/"

# Clone UHD repository
execute git clone https://github.com/EttusResearch/uhd.git "${base_dir}/ext/uhd"

execute pushd  "${base_dir}/ext/uhd/host"
execute mkdir build
execute cd build
execute cmake -DCMAKE_POLICY_DEFAULT_CMD0167=NEW -GNinja ..
execute ninja
execute ninja test
execute sudo ninja install
execute popd

# Download firmware images
execute sudo /usr/local/lib/uhd/utils/uhd_images_downloader.py

# Refresh the linker cache
execute sudo ldconfig

# Set USRP permission to non-root mode
if [ "$CI" == "0" ]; then
    execute sudo cp /usr/local/lib/uhd/utils/uhd-usrp.rules /etc/udev/rules.d/
    execute sudo udevadm control --reload-rules
    execute sudo udevadm trigger
fi
