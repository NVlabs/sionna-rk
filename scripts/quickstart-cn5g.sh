#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

source "$(dirname "${BASH_SOURCE[0]}")/license-checks.inc"

check-license

# functions
function usage() {
    echo "Usage: $0 [-h|--help] [--clean] [--debug] [--branch <branchname>] [--no-build] [--tag <tagname>] [--arch (x86|arm64)] --source <kit-rootdir> --dest <oai-cn5g-fed_dir>"
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
dest_dir=${dest_dir:-$(realpath -sm "./ext/oai-cn5g-fed")}
arch=${arch:-$(uname -m)}
branch=${branch:-"v2.1.0-1.2"}
tag=${tag:-"$branch"}
clean_dest=0
no_build=0
debug=0
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            case $2 in
                arm64|x86)
                    arch="$2"
                    shift
                    ;;
                *)
                    echo "Invalid architecture. Use arm64 or x86."
                    usage
                    exit 1
                    ;;
            esac
            ;;
        -h|--help) usage ;;
        --source) source_dir="$2"; shift ;;
        --dest) dest_dir="$2"; shift ;;
        --branch) branch="$2"; tag="$2"; shift ;;   # order is important here. must be before --tag.
        --tag) tag="$2"; shift ;;
        --clean) clean_dest=1 ;;
        --no-build) no_build=1 ;;
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
    aarch64) arch="arm64" ;;
    arm64 | x86) ;;	# valid options
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

# checkout oai-cn5g-fed
echo "Fetch OAI 5G Core Network..."
git clone --branch "$branch" https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed.git "${dest_dir}"

echo "Sync Components from OAI-CN5G..."
pushd "$dest_dir"
./scripts/syncComponents.sh \
    --nrf-branch "$branch" \
    --amf-branch "$branch" \
    --smf-branch "$branch" \
    --upf-branch "$branch" \
    --ausf-branch "$branch" \
    --udm-branch "$branch" \
    --udr-branch "$branch" \
    --upf-vpp-branch "$branch" \
    --nssf-branch "$branch" \
    --nef-branch "$branch" \
    --pcf-branch "$branch" \
    --lmf-branch "$branch"
popd
echo "Completed Sync."

# if we are on Jetson, apply patches
if [ "$arch" = "arm64" ]; then
    echo "Apply ARM64 patches..."
    "${source_dir}/scripts/patch-oai-cn5g.sh" --patch "${source_dir}/patches/oai-cn5g.patch" --dest "$dest_dir"
else
    echo "Arch is '$arch', nothing to patch."
fi

if [ "$no_build" = "0" ]; then
    echo "Build OAI 5G Core Network images..."
    debug_opts=""
    if [ "$debug" = "1" ]; then
        debug_opts="-d"
    fi
    "${source_dir}/scripts/build-cn5g-images.sh" $debug_opts --tag "$tag" "$dest_dir"
else
    echo "Skipping build of docker images."
fi

echo "Quickstart OAI Core Network done."
