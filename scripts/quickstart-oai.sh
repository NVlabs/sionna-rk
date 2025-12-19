#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

source "$(dirname "${BASH_SOURCE[0]}")/license-checks.inc"

check-license

# functions
function usage() {
    echo "Usage: $0 [-h|--help] [--clean] [--debug] [--no-build] [--tag <tagname>] [--ci] [--oai-version <oai-version>] --source <kit-rootdir> --dest <openairinterface5g_dir>"
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

TAG="latest"
OAI_VERSION="2025.w34"

clean_dest=0
no_build=0
no_tutorials=0
debug=0
ci=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        --source) source_dir="$2"; shift ;;
        --dest) dest_dir="$2"; shift ;;
        --tag) TAG="$2"; shift ;;
        --oai-version) OAI_VERSION="$2"; shift ;;
        --clean) clean_dest=1 ;;
        --no-build) no_build=1 ;;
        --debug) debug=1 ;;
        --ci) ci=1 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

# Check if both source and destination directories are provided
if [ -z "$source_dir" ] || [ -z "$dest_dir" ]; then
    usage
fi

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

# checkout openairinterface5g (submodules will be initialized after patching)
echo "Fetch OpenAirInterface..."
echo "Cloning version: $OAI_VERSION"
git clone --branch "$OAI_VERSION" https://gitlab.eurecom.fr/oai/openairinterface5g.git "$dest_dir"

echo "Applying SRK patches..."
pushd "$dest_dir"
git apply --index < "${source_dir}/patches/openairinterface5g.patch"
popd

# initialize submodules
echo "Initializing submodules..."
pushd "$dest_dir"
git submodule update --init --recursive
popd

if [ "$no_build" = "0" ]; then
    echo "Build OAI images..."
    extra_opts=""
    if [ "$debug" = "1" ]; then
        extra_opts="$extra_opts --debug"
    fi
    if [ "$ci" = "1" ]; then
        extra_opts="$extra_opts --ci"
    fi
    "${source_dir}/scripts/build-oai-images.sh" $extra_opts --tag "$TAG" "$dest_dir"
fi
