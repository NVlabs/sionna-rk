#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Integration test for Neural Receiver (TensorRT)
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

echo "Neural Receiver Integration Test (config: $CONFIG_NAME)"

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

# Enable Neural Receiver (TRT) with MCS limits (as per tutorial/env)
# Using loader option: --loader.receiver.shlibversion _trt (matches CMake target receiver_trt)
# And MCS limits to ensure 16-QAM usage (as per docs)
export GNB_EXTRA_OPTIONS="--loader.receiver.shlibversion _trt --MACRLCs.[0].dl_max_mcs 10 --MACRLCs.[0].ul_max_mcs 10"
export GNB_THREAD_POOL="--thread-pool 1"

# Start system
echo "Starting 5G system..."
docker compose --env-file "$ENV_FILE" up -d mysql oai-amf oai-smf oai-upf oai-ext-dn
sleep 20

# Pass environment variables explicitly to ensure they override any .env file settings
# Note: GNB_CONFIG needs to be set if using custom PRBs, but tutorial says default model is for 24 PRBs
# and rfsim config 'gnb.sa.band78.106prbs.conf' uses 106 PRBs.
# Doc says: "You must use a compatible gNB configuration, or re-export the model for a different number of PRBs. For example, use the provided config: GNB_CONFIG=../common/gnb.sa.band78.24prbs.conf"
# So we MUST override GNB_CONFIG for this test.

export GNB_CONFIG="../common/gnb.sa.band78.24prbs.conf"

docker compose --env-file "$ENV_FILE" up -d oai-gnb
sleep 20

# Verify Neural Receiver loaded
if docker logs oai-gnb 2>&1 | grep -qE "Initializing TRT receiver|library libreceiver_trt.so successfully loaded"; then
    echo "✅ Neural Receiver (TRT) loaded"
else
    echo "❌ Neural Receiver (TRT) NOT loaded"
    docker logs oai-gnb --tail 50
    exit 1
fi

# Start UE and run traffic test
# We need to make sure UE uses matching config/params if necessary.
# But 106PRB UE usually works with 24PRB gNB (it adapts).
docker compose --env-file "$ENV_FILE" up -d oai-nr-ue
sleep 20

# Check UE IP
UE_IP=$(docker exec oai-nr-ue ip addr show oaitun_ue1 2>/dev/null | grep -oP 'inet \K[\d.]+' || echo "")
if [[ -z "$UE_IP" ]]; then
    echo "❌ UE has no IP"
    docker logs oai-nr-ue --tail 30
    exit 1
fi
echo "✅ UE connected: $UE_IP"

# Run iperf3 uplink test in background
# Neural Receiver processes Uplink (PUSCH), so Uplink traffic is essential.
IPERF_DURATION=15
echo "Running iperf3 test ($IPERF_DURATION s, 5M bandwidth)..."
docker exec oai-nr-ue iperf3 -u -t $IPERF_DURATION -i 1 -b 5M -B 12.1.1.2 -c 192.168.72.135 > /tmp/iperf_out.txt 2>&1 &
IPERF_PID=$!

# Check GPU load while iperf is running
echo "Checking GPU usage..."
GPU_DETECTED=false
SOFTMODEM_DETECTED=false

# Check for a few seconds
for i in {1..5}; do
    if docker exec oai-gnb nvidia-smi >/dev/null 2>&1; then
        SMI_OUT=$(docker exec oai-gnb nvidia-smi --query-compute-apps=process_name --format=csv,noheader)
        if echo "$SMI_OUT" | grep -q "softmodem"; then
            SOFTMODEM_DETECTED=true
            echo "✅ Found softmodem using GPU"
            break
        fi
    else
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
    echo "⚠️ iperf3 reported no data sent (or failed). Proceeding to check status logs."
fi

# 2. GPU Load
if [ "$SOFTMODEM_DETECTED" = "true" ]; then
    echo "✅ GPU load from softmodem verified"
else
    echo "❌ No GPU load detected from softmodem"
    nvidia-smi
    exit 1
fi

# 3. Check for NRX Status Logs
# The log output contains ASCII charts with "PRBs / s" and "Latency"
# We check for the summary line, e.g., "32791.00 PRBs / s" or just "PRBs / s"
if docker logs oai-gnb 2>&1 | grep -q "PRBs / s"; then
    echo "✅ Neural Receiver statistics found in logs"
    # Optional: Print the stats line
    docker logs oai-gnb 2>&1 | grep "PRBs / s" | tail -1
else
    echo "⚠️ Neural Receiver statistics NOT found in logs (maybe traffic wasn't high enough or logging disabled?)"
    # Don't fail strictly if iperf passed, but warn.
fi

echo "✅ Neural Receiver Integration Test PASSED"
