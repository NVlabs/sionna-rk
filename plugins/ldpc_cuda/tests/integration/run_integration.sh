#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Integration test for LDPC CUDA tutorial

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

echo "LDPC CUDA Integration Test (config: $CONFIG_NAME)"

# Change to common config directory for docker-compose
cd "$COMMON_CONFIG_DIR"

cleanup() {
    echo "Cleaning up..."
    docker compose --env-file "$ENV_FILE" down --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

# Stop existing containers
docker compose --env-file "$ENV_FILE" down --remove-orphans 2>/dev/null || true

# Enable CUDA LDPC; use threadpool that works on all devices
export GNB_EXTRA_OPTIONS="--loader.ldpc.shlibversion _cuda"

# Start system
echo "Starting 5G system..."
docker compose --env-file "$ENV_FILE" up -d mysql oai-amf oai-smf oai-upf oai-ext-dn
sleep 20

docker compose --env-file "$ENV_FILE" up -d oai-gnb
sleep 15

# Verify CUDA LDPC initialized
if docker logs oai-gnb 2>&1 | grep -q "Initializing LDPC runtime"; then
    echo "✅ CUDA LDPC decoder initialized"
else
    echo "❌ CUDA LDPC not initialized"
    docker logs oai-gnb --tail 50
    exit 1
fi

# Start UE and run traffic test
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

# Run iperf3 uplink test
echo "Running iperf3 test..."
IPERF_OUTPUT=$(docker exec oai-nr-ue iperf3 -u -t 10 -i 1 -b 1M -B 12.1.1.2 -c 192.168.72.135 2>&1 || true)
echo "$IPERF_OUTPUT"

# Check iperf3 transferred data (uplink triggers LDPC decoding at gNB)
if echo "$IPERF_OUTPUT" | grep -q "sender"; then
    SENT=$(echo "$IPERF_OUTPUT" | grep "sender" | awk '{print $5, $6}')
    echo "✅ iperf3 sent: $SENT"
else
    echo "❌ iperf3 failed - no data sent"
    exit 1
fi

# Verify CUDA LDPC decoder was actually used during traffic
if docker logs oai-gnb 2>&1 | grep -q "CUDA LDPC decoder:"; then
    LDPC_TIME=$(docker logs oai-gnb 2>&1 | grep "CUDA LDPC decoder:" | tail -1 | grep -oP '\d+\.\d+ us' | head -1)
    echo "✅ CUDA LDPC decoder active: $LDPC_TIME"
else
    echo "❌ CUDA LDPC decoder not used during traffic"
    docker logs oai-gnb --tail 50
    exit 1
fi

# Shutdown
echo "Shutting down..."
docker compose --env-file "$ENV_FILE" down --remove-orphans

echo "✅ LDPC CUDA Integration Test PASSED"
