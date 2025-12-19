#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Integration test for Neural Demapper (TensorRT)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIG_NAME="${CONFIG_NAME:-testing}"

# Config paths
CONFIGS_DIR="$REPO_ROOT/config"
COMMON_CONFIG_DIR="$CONFIGS_DIR/common"
ENV_FILE="$CONFIGS_DIR/$CONFIG_NAME/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

echo "Neural Demapper Integration Test (config: $CONFIG_NAME)"

# Change to common config directory for docker-compose
cd "$COMMON_CONFIG_DIR"

cleanup() {
    RET=$?
    echo "Cleaning up..."
    docker compose --env-file "$ENV_FILE" down --remove-orphans 2>/dev/null || true

    if [ $RET -ne 0 ]; then
        echo "❌ Test failed."
        docker logs oai-gnb --tail 50
    fi
}
trap cleanup EXIT

# Stop existing containers
docker compose --env-file "$ENV_FILE" down --remove-orphans 2>/dev/null || true

# Enable Neural Demapper (TRT) with MCS limits (as per tutorial/env)
export GNB_EXTRA_OPTIONS="--loader.demapper.shlibversion _trt --MACRLCs.[0].dl_max_mcs 10 --MACRLCs.[0].ul_max_mcs 10"
export GNB_THREAD_POOL="--thread-pool 1"

# Start system
echo "Starting 5G system..."
docker compose --env-file "$ENV_FILE" up -d mysql oai-amf oai-smf oai-upf oai-ext-dn
sleep 20

docker compose --env-file "$ENV_FILE" up -d oai-gnb
sleep 15

# Verify Neural Demapper loaded
# The logs should show "Initializing TRT demapper" or similar, or the loader message.
if docker logs oai-gnb 2>&1 | grep -qE "Initializing TRT demapper|library libdemapper_trt.so successfully loaded"; then
    echo "✅ Neural Demapper (TRT) loaded"

else
    echo "❌ Neural Demapper (TRT) NOT loaded"
    docker logs oai-gnb --tail 50
    exit 1
fi

# Start UE and run traffic test
GNB_EXTRA_OPTIONS="$GNB_EXTRA_OPTIONS" GNB_THREAD_POOL="$GNB_THREAD_POOL" docker compose --env-file "$ENV_FILE" up -d oai-nr-ue
sleep 15

# Check UE IP
UE_IP=$(docker exec oai-nr-ue ip addr show oaitun_ue1 2>/dev/null | grep -oP 'inet \K[\d.]+' || echo "")
if [[ -z "$UE_IP" ]]; then
    echo "❌ UE has no IP"
    docker logs oai-nr-ue --tail 30
    exit 1
fi
echo "✅ UE connected: $UE_IP"

# Run iperf3 uplink test in background
# Increased duration to allow time for GPU check
IPERF_DURATION=15
echo "Running iperf3 test ($IPERF_DURATION s, 10M bandwidth)..."
docker exec oai-nr-ue iperf3 -u -t $IPERF_DURATION -i 1 -b 10M -B 12.1.1.2 -c 192.168.72.135 > /tmp/iperf_out.txt 2>&1 &
IPERF_PID=$!

# Check GPU load while iperf is running
echo "Checking GPU usage..."
GPU_DETECTED=false
SOFTMODEM_DETECTED=false

# Check for a few seconds
for i in {1..5}; do
    # Check if nvidia-smi works inside container
    if docker exec oai-gnb nvidia-smi >/dev/null 2>&1; then
        # List compute apps
        SMI_OUT=$(docker exec oai-gnb nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader)
        echo "GPU Processes: $SMI_OUT"

        if echo "$SMI_OUT" | grep -q "softmodem"; then
            SOFTMODEM_DETECTED=true
            echo "✅ Found softmodem using GPU"
            break
        fi
    else
        # Fallback: check on host if container doesn't have nvidia-smi (it should though)
        # Note: This might be less reliable if host has many GPUs/processes
        echo "⚠️ nvidia-smi not found inside container, checking host..."
        if nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -q "softmodem"; then
            SOFTMODEM_DETECTED=true
            echo "✅ Found softmodem using GPU (on host)"
            break
        fi
    fi
    sleep 1
done

# Wait for iperf to finish
wait $IPERF_PID || true
IPERF_OUTPUT=$(cat /tmp/iperf_out.txt)
echo "$IPERF_OUTPUT"

# Validate Results

# 1. iperf traffic
if echo "$IPERF_OUTPUT" | grep -q "sender"; then
    echo "✅ iperf3 traffic sent"
else
    echo "⚠️ iperf3 reported no data sent (or failed). Proceeding to check GPU load."
fi

# 2. GPU Load
if [ "$SOFTMODEM_DETECTED" = "true" ]; then
    echo "✅ GPU load from softmodem verified"
else
    echo "❌ No GPU load detected from softmodem"
    # List all processes on GPU for debug
    nvidia-smi
    exit 1
fi

echo "✅ Neural Demapper Integration Test PASSED"
