#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# functions
usage() {
    echo "Usage: $0 [-h|--help] [--tag <tagname>] [--arch (x86|arm64|cuda)] [-d|--debug] <openairinterface5g_dir>"
    exit 1
}

build_cuda_images() {
    echo "Building CUDA images"
    echo "CUDA base image"
    docker build $debug_opts --target ran-base-cuda --tag ran-base-cuda:${tag} --file docker/Dockerfile.base.cuda.aarch64 .
    echo "CUDA build image"
    docker build $debug_opts --target ran-build-cuda --tag ran-build-cuda:${tag} --file docker/Dockerfile.build.cuda.aarch64 .
    echo "CUDA gNB"
    docker build $debug_opts --target oai-gnb-cuda --tag oai-gnb-cuda:${tag} --file docker/Dockerfile.gNB.cuda.aarch64 .
    echo "build UE (cuda version)"
    docker build $debug_opts --target oai-nr-ue-cuda --tag oai-nr-ue-cuda:${tag} --file docker/Dockerfile.nrUE.cuda.aarch64 .
}

build_arm_images() {
    echo "Building images for AARCH64"
    echo "base image"
    docker build $debug_opts --target ran-base --tag ran-base:${tag} --file docker/Dockerfile.base.ubuntu22.aarch64 .
    echo "build image"
    docker build $debug_opts --target ran-build --tag ran-build:${tag} --file docker/Dockerfile.build.ubuntu22.aarch64 .
    echo "build gNB"
    docker build $debug_opts --target oai-gnb --tag oai-gnb:${tag} --file docker/Dockerfile.gNB.ubuntu22.aarch64 .
    echo "build UE"
    docker build $debug_opts --target oai-nr-ue --tag oai-nr-ue:${tag} --file docker/Dockerfile.nrUE.ubuntu22.aarch64 .
}

build_x86_images() {
    echo "Building images for x86"
    echo "base image"
    docker build $debug_opts --target ran-base --tag ran-base:${tag} --file docker/Dockerfile.base.ubuntu22 .
    echo "build image"
    docker build $debug_opts --target ran-build --tag ran-build:${tag} --file docker/Dockerfile.build.ubuntu22 .
    echo "build gNB"
    docker build $debug_opts --target oai-gnb --tag oai-gnb:${tag} --file docker/Dockerfile.gNB.ubuntu22 .
    echo "build UE"
    docker build $debug_opts --target oai-nr-ue --tag oai-nr-ue:${tag} --file docker/Dockerfile.nrUE.ubuntu22 .
}

check_docker_group() {
    if id -nG "$USER" | grep -qw docker; then
        echo "Checking: $USER is a member of the docker group"
    else
        echo "$USER needs to be a member of the docker group"
        echo "Execute 'sudo usermod -aG docker $USER' and logout and log back in to refresh membership."
        exit 1
    fi
}

# Default values
path="$(pwd)"
tag="latest"
arch="cuda"
debug_opts=""

check_docker_group

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        --tag)
            tag="$2"
            shift 2
            ;;
        --arch)
            case $2 in
                arm64|x86|cuda)
                    arch="$2"
                    shift 2
                    ;;
                *)
                    echo "Invalid architecture. Use arm64, x86, or cuda."
                    usage
                    exit 1
                    ;;
            esac
            ;;
        -d|--debug)
            debug_opts="--progress plain"
            shift
            ;;
        *)
            path="$1"
            shift
            ;;
    esac
done

# Convert relative path to absolute path
oai_path=$(realpath -sm "$path")

if [ ! -d "$oai_path" ]; then
    echo "Error: $oai_path does not exist."
    usage
    exit 1
fi

# Use the parsed values
echo "Path: ${oai_path}"
echo "Tag: ${tag}"
echo "Arch: ${arch}"

# switch to the OAI path
pushd "$oai_path"

case "$arch" in
    x86)
        build_x86_images
        ;;
    arm64)
        build_arm_images
        ;;
    cuda)
        build_cuda_images
        ;;
esac

popd
