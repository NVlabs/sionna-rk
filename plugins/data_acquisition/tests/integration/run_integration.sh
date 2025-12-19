#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Integration test for Demapper Capture (Data Acquisition)
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

# Log files path
LOG_DIR="$REPO_ROOT/plugins/data_acquisition/logs"
LOG_IN="$LOG_DIR/demapper_in.txt"
LOG_OUT="$LOG_DIR/demapper_out.txt"

echo "Demapper Capture Integration Test (config: $CONFIG_NAME)"

# Ensure log directory and files exist with correct permissions
mkdir -p "$LOG_DIR"
touch "$LOG_IN" "$LOG_OUT"
chmod 666 "$LOG_IN" "$LOG_OUT"

# 1. Check that log files are empty at the beginning (after explicit clear)
echo "Clearing log files..."
truncate -s 0 "$LOG_IN"
truncate -s 0 "$LOG_OUT"

if [ -s "$LOG_IN" ] || [ -s "$LOG_OUT" ]; then
    echo "❌ Log files are not empty after clearing"
    exit 1
fi
echo "✅ Log files cleared and empty"

# Change to common config directory for docker-compose
cd "$COMMON_CONFIG_DIR"

cleanup() {
    RET=$?
    echo "Cleaning up..."
    docker compose --env-file "$ENV_FILE" down --remove-orphans 2>/dev/null || true

    if [ $RET -eq 0 ]; then
        # Reset log files after test
        echo "Resetting log files..."
        truncate -s 0 "$LOG_IN"
        truncate -s 0 "$LOG_OUT"
    else
        echo "❌ Test failed. Log files preserved for inspection at:"
        echo "  - $LOG_IN"
        echo "  - $LOG_OUT"
    fi
}
trap cleanup EXIT

# Stop existing containers
docker compose --env-file "$ENV_FILE" down --remove-orphans 2>/dev/null || true

# Enable Demapper Capture
export GNB_EXTRA_OPTIONS="--loader.demapper.shlibversion _capture"
export GNB_THREAD_POOL="--thread-pool 1"

# Start system
echo "Starting 5G system..."
docker compose --env-file "$ENV_FILE" up -d mysql oai-amf oai-smf oai-upf oai-ext-dn
sleep 20

docker compose --env-file "$ENV_FILE" up -d oai-gnb
sleep 15

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
# Short run to generate traffic
IPERF_OUTPUT=$(docker exec oai-nr-ue iperf3 -u -t 10 -i 1 -b 1M -B 12.1.1.2 -c 192.168.72.135 2>&1 || true)
echo "$IPERF_OUTPUT"

if echo "$IPERF_OUTPUT" | grep -q "sender"; then
    echo "✅ iperf3 traffic sent"
else
    echo "⚠️ iperf3 reported no data sent (or failed), but checking logs for samples anyway..."
fi

# Verify samples recorded in log files
# We expect files to be non-empty and contain data lines
if [ ! -s "$LOG_IN" ] || [ ! -s "$LOG_OUT" ]; then
    echo "❌ Log files are empty after traffic"
    exit 1
fi

# Check for specific content headers (QPSK or QAM16)
if grep -qE "QPSK|QAM16" "$LOG_IN"; then
    echo "✅ Samples found in input log"
else
    echo "❌ No modulation headers found in input log"
    head -n 20 "$LOG_IN"
    exit 1
fi

if grep -qE "QPSK|QAM16" "$LOG_OUT"; then
    echo "✅ Samples found in output log"
else
    echo "❌ No modulation headers found in output log"
    head -n 20 "$LOG_OUT"
    exit 1
fi

echo "✅ Demapper Capture Integration Test PASSED"
