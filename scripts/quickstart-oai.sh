#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

source "$(dirname "${BASH_SOURCE[0]}")/license-checks.inc"

check-license

# functions
function usage() {
    echo "Usage: $0 [-h|--help] [--clean] [--debug] [--no-tutorials] [--no-build] [--tag <tagname>] [--arch (x86|arm64|cuda)] --source <kit-rootdir> --dest <openairinterface5g_dir>"
    exit 1
}

# supress outputs from pushd and popd
function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

# defaults
default_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/../)
source_dir=${source_dir:-"$default_dir"}
dest_dir=${dest_dir:-$(realpath -sm "./ext/openairinterface5g")}
arch=${arch:-$(uname -m)}
tag="latest"
clean_dest=0
no_build=0
no_tutorials=0
debug=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            case $2 in
                arm64|x86|cuda)
                    arch="$2"
                    shift
                    ;;
                *)
                    echo "Invalid architecture. Use arm64, x86, or cuda."
                    usage
                    exit 1
                    ;;
            esac
            ;;
        -h|--help) usage ;;
        --source) source_dir="$2"; shift ;;
        --dest) dest_dir="$2"; shift ;;
        --tag) tag="$2"; shift ;;
        --clean) clean_dest=1 ;;
        --no-build) no_build=1 ;;
        --no-tutorials) no_tutorials=1 ;;
        --debug) debug=1 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

# Check if both source and destination directories are provided
if [ -z "$source_dir" ] || [ -z "$dest_dir" ]; then
    usage
fi

# tranform arch parameters
case $arch in
    x86_64) arch="x86" ;;
    aarch64) arch="cuda" ;; # cuda images are extended arm64 images. arm64 images are non-gpu
    arm64 | x86 | cuda) ;;	# valid options
    *)
        echo "Arch $arch is not supported. exiting."
        exit 1
        ;;
esac

# Convert source_dir and dest_dir to absolute paths
source_dir=$(realpath -sm "$source_dir")
dest_dir=$(realpath -sm "$dest_dir")

echo "source: $source_dir"
echo "dest: $dest_dir"
echo "arch: $arch"

# If clean install, remove destination directory
if [ "$clean_dest" = "1" ] && [ -d "$dest_dir" ]; then
    echo "Removing directory $dest_dir ..."
    rm -rf "$dest_dir"
fi

# create destination directory if needed
if [ ! -d "$dest_dir" ]; then
    mkdir -p $(dirname "$dest_dir")
else
    echo "Destination directory $dest_dir already exists. Use the --clean option or remove it before proceeding."
    usage
    exit 1
fi

# checkout openairinterface5g
echo "Fetch OpenAirInterface..."
git clone --branch 2024.w34 https://gitlab.eurecom.fr/oai/openairinterface5g.git "$dest_dir"

# if we are on Jetson, apply patches
if [ "$arch" = "arm64" ] || [ "$arch" = "cuda" ]; then
    echo "Apply ARM64 patches..."
    pushd "$dest_dir"
    git apply < "${source_dir}/patches/openairinterface5g.patch"
    popd

    if [ "$no_tutorials" = "0" ]; then
        echo "Add tutorials..."
        pushd "$dest_dir"
        git apply < "${source_dir}/patches/tutorials.patch"
        popd
    fi
fi

if [ "$no_build" = "0" ]; then
    echo "Build OAI images..."
    debug_opts=""
    if [ "$debug" = "1" ]; then
        debug_opts="--debug"
    fi
    "${source_dir}/scripts/build-oai-images.sh" $debug_opts --tag "$tag" --arch "$arch" "$dest_dir"
fi
