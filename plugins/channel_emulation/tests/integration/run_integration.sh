#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Integration test for the CUDA channel emulator.
#
# Starts the full 5G system in rfsim mode with the channel emulator
# (file-based CIR, pass-through), verifies the UE connects, runs
# iperf3 traffic, and checks that the emulator was active throughout.
#
# This test requires a CUDA-capable GPU (designed for DGX Spark).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIG_NAME="${CONFIG_NAME:-rfsim}"
CONFIG_DIR="$REPO_ROOT/config/common"
ENV_FILE="$REPO_ROOT/config/$CONFIG_NAME/.env"

IPERF_DURATION=15
IPERF_SERVER="192.168.72.135"

echo "Channel Emulation Integration Test (config: $CONFIG_NAME)"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

cd "$CONFIG_DIR"

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------

cleanup() {
    RET=$?
    echo "Cleaning up..."
    docker compose --env-file "$ENV_FILE" down --remove-orphans 2>/dev/null || true

    if [ $RET -ne 0 ]; then
        echo "❌ Test FAILED — dumping logs"
        echo "--- oai-gnb (last 80 lines) ---"
        docker logs oai-gnb --tail 80 2>&1 || true
        echo "--- oai-nr-ue (last 30 lines) ---"
        docker logs oai-nr-ue --tail 30 2>&1 || true
    fi
}
trap cleanup EXIT

docker compose --env-file "$ENV_FILE" down --remove-orphans 2>/dev/null || true

# ---------------------------------------------------------------------------
# Configure: channel emulator with file-based pass-through CIR
# ---------------------------------------------------------------------------

export GNB_EXTRA_OPTIONS="--cir-folder /opt/oai-gnb/plugins/channel_emulation/data/pass_through_cir"

# ---------------------------------------------------------------------------
# Start 5G Core
# ---------------------------------------------------------------------------

echo "Starting 5G Core..."
docker compose --env-file "$ENV_FILE" up -d mysql oai-amf oai-smf oai-upf oai-ext-dn
sleep 20

# ---------------------------------------------------------------------------
# Start gNB with channel emulator
# ---------------------------------------------------------------------------

echo "Starting gNB (with channel emulator)..."
docker compose --env-file "$ENV_FILE" up -d oai-gnb
sleep 15

# Verify channel emulator initialized
if docker logs oai-gnb 2>&1 | grep -q "Channel Emulator initialized"; then
    echo "✅ Channel emulator initialized"
else
    echo "❌ Channel emulator NOT initialized"
    exit 1
fi

# Verify CIR file loader loaded
if docker logs oai-gnb 2>&1 | grep -q "CIR_FILE: Loaded"; then
    echo "✅ CIR file loader active (pass-through CIR)"
else
    echo "❌ CIR file loader NOT active"
    exit 1
fi

# ---------------------------------------------------------------------------
# Start UE
# ---------------------------------------------------------------------------

echo "Starting UE..."
docker compose --env-file "$ENV_FILE" up -d oai-nr-ue
sleep 20

UE_IP=$(docker exec oai-nr-ue ip addr show oaitun_ue1 2>/dev/null \
        | grep -oP 'inet \K[\d.]+' || echo "")
if [[ -z "$UE_IP" ]]; then
    echo "❌ UE has no IP — did not attach"
    exit 1
fi
echo "✅ UE connected: $UE_IP"

# ---------------------------------------------------------------------------
# Verify gNB is still healthy (no crash after UE attach)
# ---------------------------------------------------------------------------

if ! docker ps --format '{{.Names}}' | grep -q oai-gnb; then
    echo "❌ gNB container crashed after UE attach"
    exit 1
fi
echo "✅ gNB still running"

# ---------------------------------------------------------------------------
# Run iperf3 traffic (downlink via -R)
# ---------------------------------------------------------------------------

echo "Running iperf3 downlink test (${IPERF_DURATION}s)..."
IPERF_OUTPUT=$(docker exec oai-nr-ue \
    iperf3 -t "$IPERF_DURATION" -i 1 -B "$UE_IP" -c "$IPERF_SERVER" -R 2>&1 || true)
echo "$IPERF_OUTPUT"

if echo "$IPERF_OUTPUT" | grep -q "receiver"; then
    RECV=$(echo "$IPERF_OUTPUT" | grep "receiver" | awk '{print $5, $6}')
    echo "✅ iperf3 received: $RECV"
else
    echo "❌ iperf3 failed — no data received"
    exit 1
fi

# ---------------------------------------------------------------------------
# Verify containers survived the traffic
# ---------------------------------------------------------------------------

for c in oai-gnb oai-nr-ue; do
    if docker ps --format '{{.Names}}' | grep -q "$c"; then
        echo "✅ $c still running after traffic"
    else
        echo "❌ $c crashed during traffic"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Run iperf3 uplink (exercises channel emulator in UL direction)
# ---------------------------------------------------------------------------

echo "Running iperf3 uplink test (${IPERF_DURATION}s)..."
IPERF_UL=$(docker exec oai-nr-ue \
    iperf3 -u -t "$IPERF_DURATION" -i 1 -b 5M -B "$UE_IP" -c "$IPERF_SERVER" 2>&1 || true)

if echo "$IPERF_UL" | grep -q "sender"; then
    SENT=$(echo "$IPERF_UL" | grep "sender" | awk '{print $5, $6}')
    echo "✅ iperf3 uplink sent: $SENT"
else
    echo "⚠️  iperf3 uplink reported no data (non-fatal)"
fi

# Final container health check
for c in oai-gnb oai-nr-ue; do
    if ! docker ps --format '{{.Names}}' | grep -q "$c"; then
        echo "❌ $c crashed during uplink traffic"
        exit 1
    fi
done

echo ""
echo "✅ Channel Emulation Integration Test PASSED"
