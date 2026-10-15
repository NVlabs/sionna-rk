#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/license-checks.inc"

check-license

# functions
function usage() {
    echo "Usage: $0 [-h|--help] [--clean] [--use-existing] [--debug] [--no-build] [--tag <tagname>] [--ci] [--oai-version <oai-version>] --source <kit-rootdir> --dest <openairinterface5g_dir>"
    exit 1
}

# suppress outputs from pushd and popd
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
use_existing=0
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
        --use-existing) use_existing=1 ;;
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

if [ "$clean_dest" = "1" ] && [ "$use_existing" = "1" ]; then
    echo "The --clean and --use-existing options cannot be used together."
    exit 1
fi

# If clean install, remove destination directory
if [ "$clean_dest" = "1" ] && [ -d "$dest_dir" ]; then
    echo "Removing directory $dest_dir ..."
    rm -rf "$dest_dir"
fi

# create destination directory if needed
if [ ! -d "$dest_dir" ]; then
    if [ "$use_existing" = "1" ]; then
        echo "Destination directory $dest_dir does not exist. The --use-existing option requires an existing OpenAirInterface checkout."
        exit 1
    fi
    mkdir -p $(dirname "$dest_dir")
else
    if [ "$use_existing" = "1" ]; then
        if ! git -C "$dest_dir" rev-parse --is-inside-work-tree > /dev/null 2>&1 || \
           [ ! -f "$dest_dir/CMakeLists.txt" ] || \
           [ ! -f "$dest_dir/executables/nr-softmodem.c" ]; then
            echo "Destination directory $dest_dir is not a valid OpenAirInterface checkout."
            exit 1
        fi
    else
        echo "Destination directory $dest_dir already exists. Use the --clean option, --use-existing, or remove it before proceeding."
        usage
        exit 1
    fi
fi

if [ "$use_existing" = "0" ]; then
    # checkout openairinterface5g (submodules will be initialized after patching)
    # Default to the public upstream repository. In CI, an internal mirror can be
    # provided via the OAI_REPO_URL environment variable to avoid rate limits and
    # keep the URL out of this script.
    oai_repo_url="https://gitlab.eurecom.fr/oai/openairinterface5g.git"
    if [ "$ci" = "1" ] && [ -n "${OAI_REPO_URL:-}" ]; then
        oai_repo_url="$OAI_REPO_URL"
    fi

    echo "Fetch OpenAirInterface..."
    echo "Cloning version: $OAI_VERSION"
    git clone --branch "$OAI_VERSION" "$oai_repo_url" "$dest_dir"

    echo "Applying SRK patches..."
    pushd "$dest_dir"
    git apply --index < "${source_dir}/patches/openairinterface5g.patch"
    popd
else
    echo "Using existing OpenAirInterface checkout at $dest_dir"
fi

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
