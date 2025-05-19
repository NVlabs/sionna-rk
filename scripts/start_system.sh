#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

set -e  # Stop script on any error

# supress outputs from pushd and popd
function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

# defaults
CONFIG_NAME=${1:-b200_arm64}
configs_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/../configs)

# change into config directory
pushd ${configs_dir}/$CONFIG_NAME

echo "Starting 5G Core network"
docker compose up -d mysql oai-amf oai-smf oai-upf oai-ext-dn

# Function to wait until a container is healthy
wait_for_container() {
    container_name=$1
    timeout=60  # Set timeout in seconds
    start_time=$(date +%s)
    echo "Waiting for $container_name to be healthy (Timeout: ${timeout}s)..."
    while true; do
        status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "not_found")

        if [[ "$status" == "healthy" ]]; then
            echo "$container_name is ready!"
            return 0
        elif [[ "$status" == "unhealthy" ]]; then
            echo "Error: $container_name became unhealthy! Exiting..."
            exit 1
        elif [[ "$status" == "not_found" ]]; then
            echo "Error: Container $container_name not found! Exiting..."
            exit 1
        fi

        # Check if timeout is reached
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        if [[ $elapsed_time -ge $timeout ]]; then
            echo "Error: Timeout reached while waiting for $container_name! Exiting..."
            exit 1
        fi

        sleep 2
    done
}

# Wait for each service to be healthy
wait_for_container "oai-mysql"
wait_for_container "oai-amf"
wait_for_container "oai-smf"
wait_for_container "oai-upf"
wait_for_container "oai-ext-dn"

echo "All services are up and healthy!"
#sleep 30

echo "Starting gNB"
docker compose up -d oai-gnb

wait_for_container "oai-gnb"

# Check if CONFIG_NAME contains "rfsim" and start "nr-ue" service if needed
if [[ "$CONFIG_NAME" == *"rfsim"* ]]; then
    echo "gNB ready to connect"
    echo "Starting nr-ue"
    docker compose up -d oai-nr-ue
    wait_for_container "oai-nr-ue"
fi

echo "5G network is ready to connect!"

# back to original directory
popd
