#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Integration test for RIC xApps
# Runs the system in rfsim config, verifies monitor_xapp is running,
# connects a ZeroMQ client, and runs traffic to check for MCS updates.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIG_NAME="${CONFIG_NAME:-testing}"
# The docker-compose.yaml is in config/common, but we need the .env from config/rfsim
CONFIG_DIR="$REPO_ROOT/config/common"
ENV_FILE="$REPO_ROOT/config/$CONFIG_NAME/.env"

echo "RIC xApp Integration Test (config: $CONFIG_NAME)"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file $ENV_FILE not found!"
    exit 1
fi

cd "$CONFIG_DIR"

cleanup() {
    echo "Cleaning up..."
    # Kill background client if running
    if [ -n "$CLIENT_PID" ]; then
        kill $CLIENT_PID 2>/dev/null || true
    fi
    docker compose --env-file "$ENV_FILE" down --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

# Stop existing containers
docker compose --env-file "$ENV_FILE" down --remove-orphans 2>/dev/null || true

# Start 5G Core, gNB, RIC, and monitor_xapp
echo "Starting 5G system..."
docker compose --env-file "$ENV_FILE" up -d mysql oai-amf oai-smf oai-upf oai-ext-dn nearRT-RIC
sleep 20

docker compose --env-file "$ENV_FILE" up -d oai-gnb
sleep 15

docker compose --env-file "$ENV_FILE" up -d monitor_xapp
sleep 15
# Verify monitor_xapp is running
if docker ps | grep -q monitor_xapp; then
    echo "✅ monitor_xapp container is running"
else
    echo "❌ monitor_xapp container is NOT running"
    docker logs monitor_xapp --tail 50
    exit 1
fi

# Start UE
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

# Start ZeroMQ client in background and capture output
CLIENT_LOG="/tmp/zmq_client_$$.log"
echo "Starting ZeroMQ client logging to $CLIENT_LOG"
# Run client for enough messages to capture traffic
python3 "$REPO_ROOT/plugins/ric_xapps/src/zmq_stats_client.py" --max-messages 20 > "$CLIENT_LOG" 2>&1 &
CLIENT_PID=$!

# Run iperf3 uplink traffic to generate load and MCS changes
echo "Running iperf3 test to generate traffic..."
# Send traffic to the CN5G traffic generator (oai-ext-dn)
docker exec oai-nr-ue iperf3 -u -t 10 -i 1 -b 1M -B 12.1.1.2 -c 192.168.72.135 || true

# Wait for client to finish or timeout
wait $CLIENT_PID 2>/dev/null || true

# Check if client received stats
if grep -q "UE Stats" "$CLIENT_LOG"; then
    echo "✅ ZeroMQ client received stats"

    # Check for MCS activity
    # Look for non-zero MCS or just presence of MCS fields
    if grep -q "MCS" "$CLIENT_LOG"; then
         echo "✅ MCS values detected in output:"
         grep "MCS" "$CLIENT_LOG" | head -n 5

         # Optional: Check if we see some traffic related stats
         if grep -q "PRBs" "$CLIENT_LOG"; then
             echo "✅ PRB usage detected"
         fi
    else
         echo "⚠️ No MCS values found in stats"
         cat "$CLIENT_LOG"
         exit 1
    fi
else
    echo "❌ ZeroMQ client received no stats"
    echo "Client Log Output:"
    cat "$CLIENT_LOG"
    docker logs monitor_xapp --tail 50
    exit 1
fi

echo "✅ RIC xApp Integration Test PASSED"
