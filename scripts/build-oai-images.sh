#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# functions
function usage() {
    echo "Usage: $0 [-h|--help] [--debug] [--tag <tagname>] [--no-cache] [--ci] [--force-platform <platform>] <openairinterface5g_dir>"
    exit 1
}

function log() {
    echo "$@" | tee -a ${project_root}/build.log
}

function log_separator() {
    log "================================================================================================"
}

function log_check_status() {
    echo "Running: $@" | tee -a ${project_root}/build.log
    eval "$@" 2>&1 | tee -a ${project_root}/build.log
    ret_val=$?
    if [ $ret_val -ne 0 ]; then
        echo "Command failed with exit code $ret_val"
        exit $ret_val
    fi
    return $ret_val
}

function build_cuda_images() {
    extra_opts=""
    if [ "${platform}" == "Orin Nano Super" ] || [ "${platform}" == "AGX Orin" ]; then
        export BASE_IMAGE="nvcr.io/nvidia/l4t-jetpack:r36.3.0"
        export BOOST_VERSION="1.74.0"
        export EXTRA_DEB_PKGS="gcc-12 g++-12"
        export BUILD_OPTION="--cmake-opt -DCMAKE_C_COMPILER=gcc-12 --cmake-opt -DCMAKE_CXX_COMPILER=g++-12 --cmake-opt -DCMAKE_CUDA_ARCHITECTURES=87 --cmake-opt -DAVX2=OFF --cmake-opt -DAVX512=OFF"
        export FLEXRIC_BUILD_OPTIONS="-DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12"
        extra_opts="--build-arg BASE_IMAGE --build-arg BOOST_VERSION --build-arg EXTRA_DEB_PKGS --build-arg BUILD_OPTION --build-arg FLEXRIC_BUILD_OPTIONS"
    fi

    echo "$platform" > ${oai_path}/host_product_family
    echo "Building CUDA images"
    log_separator
    log "CUDA base image"
    # Base image builds from OAI context only
    pushd "$oai_path"
    log_check_status docker build --progress plain \
        $cache_opts $extra_opts --build-arg DOCKER_CUSTOM_IMAGE_TAG=${tag} --target ran-base-cuda --tag ran-base-cuda:${tag} \
        --file docker/Dockerfile.base.ubuntu.cuda .
    popd

    # Build image uses broader context (sionna-rk root) to include tutorials
    log_separator
    log "CUDA build image (with tutorials)"
    log_check_status docker build --progress plain \
        $cache_opts $extra_opts --build-arg DOCKER_CUSTOM_IMAGE_TAG=${tag} --target ran-build-cuda --tag ran-build-cuda:${tag} \
        --file ${oai_path}/docker/Dockerfile.build.ubuntu.cuda ${project_root}

    # Remaining images build from OAI context
    pushd "$oai_path"
    log_separator
    log "CUDA gNB"
    log_check_status docker build --progress plain \
        $cache_opts $extra_opts --build-arg DOCKER_CUSTOM_IMAGE_TAG=${tag} --target oai-gnb-cuda --tag oai-gnb-cuda:${tag} \
        --file docker/Dockerfile.gNB.ubuntu.cuda .

    log_separator
    log "build UE (cuda version)"
    log_check_status docker build --progress plain \
        $cache_opts $extra_opts --build-arg DOCKER_CUSTOM_IMAGE_TAG=${tag} --target oai-nr-ue-cuda --tag oai-nr-ue-cuda:${tag} \
        --file docker/Dockerfile.nrUE.ubuntu.cuda .

    log_separator
    log "build FlexRIC"
    log_check_status docker build \
        $cache_opts $extra_opts --build-arg DOCKER_CUSTOM_IMAGE_TAG=${tag} --target oai-flexric-fixed --tag oai-flexric:${tag} \
        --file docker/Dockerfile.flexric.ubuntu .
    popd
}

check_docker_group() {
    if [ "$ci" == "1" ]; then
        return
    fi
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
cache_opts=""
debug_opts=""
ci="0"

# Determine project root (parent of the script location)
project_root=$(realpath $(dirname "${BASH_SOURCE[0]}")/../)
platform=$( ${project_root}/scripts/detect_host.sh )

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
        --debug)
            # do nothing, progress plain is now default
            shift
            ;;
        --no-cache)
            cache_opts="--no-cache"
            shift
            ;;
        --force-platform)
            platform="$2"
            shift 2
            ;;
        --ci)
            ci="1"
            shift
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

check_docker_group

# Convert relative path to absolute path
oai_path=$(realpath -sm "$path")

if [ ! -d "$oai_path" ]; then
    echo "Error: $oai_path does not exist."
    usage
    exit 1
fi

# Use the parsed values
echo "OAI Path: ${oai_path}"
echo "Project Root: ${project_root}"
echo "Tag: ${tag}"
echo "Arch: ${arch}"
echo "Platform: ${platform}"

# Clear build.log
echo "" > ${project_root}/build.log

# Build CUDA images
build_cuda_images
