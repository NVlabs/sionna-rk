#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# functions
usage() {
    echo "Usage: $0 [-h|--help] [--tag <tagname>] <oai-cn5g_dir>"
    exit 1
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
tag="v2.0.1"

check_docker_group

if id -nG "$USER" | grep -qw docker; then
    echo "Checking: $USER is a member of the docker group"
else
    echo "$USER needs to be a member of the docker group"
    echo "Execute 'sudo usermod -aG docker $USER' and logout and log back in to refresh membership."
    exit 1
fi

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            tag="$2"
            shift 2
            ;;
        -h|--help)
            usage
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

# switch to the OAI path
pushd "$oai_path"

# build docker images

echo "Building OAI 5G Core network docker images...."
echo "AMF"
docker build --target oai-amf --tag oai-amf:${tag} --file component/oai-amf/docker/Dockerfile.amf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-amf
echo "SMF"
docker build --target oai-smf --tag oai-smf:${tag} --file component/oai-smf/docker/Dockerfile.smf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-smf
echo "NRF"
docker build --target oai-nrf --tag oai-nrf:${tag} --file component/oai-nrf/docker/Dockerfile.nrf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-nrf
echo "AUSF"
docker build --target oai-ausf --tag oai-ausf:${tag} --file component/oai-ausf/docker/Dockerfile.ausf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-ausf
echo "UDM"
docker build --target oai-udm --tag oai-udm:${tag} --file component/oai-udm/docker/Dockerfile.udm.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-udm
echo "UDR"
docker build --target oai-udr --tag oai-udr:${tag} --file component/oai-udr/docker/Dockerfile.udr.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-udr
echo "NSSF"
docker build --target oai-nssf --tag oai-nssf:${tag} --file component/oai-nssf/docker/Dockerfile.nssf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-nssf
echo "UPF"
docker build --target oai-upf --tag oai-upf:${tag} --file component/oai-upf/docker/Dockerfile.upf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-upf
echo "Traffic Generator"
cd ci-scripts
docker build --target trf-gen-cn5g --tag trf-gen-cn5g:latest --file Dockerfile.traffic.generator.ubuntu .

popd
