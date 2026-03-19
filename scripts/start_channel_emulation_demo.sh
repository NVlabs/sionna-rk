#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end channel emulation demo (rfsim mode).
#
# Starts the 5G network with the CUDA channel emulator (ZMQ CIR source),
# waits for the UE to connect, launches iperf3 traffic, and starts the
# Sionna RT GUI with MPS.
#
# Prerequisites:
#   - Docker images built (oai-gnb-cuda, oai-nr-ue-cuda, CN5G, FlexRIC)
#   - Sionna RT GUI installed (https://github.com/NVlabs/sionna-rt-gui)
#
# Usage:
#   ./scripts/start_channel_emulation_demo.sh [GUI_CONFIG]
#
# Environment variables:
#   SIONNA_RT_GUI_DIR   Path to the Sionna RT GUI (default: ext/sionna-rt-gui)
#   MPS_ACTIVE_THREAD_PCT  MPS thread percentage (default: 50.0)
#   CIR_ZMQ_NUM_TAPS   Number of CIR taps (default: 48)
#   SKIP_GUI            Set to 1 to skip the GUI (default: 0)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIGS_DIR="$REPO_ROOT/config"
ENV_FILE="$CONFIGS_DIR/rfsim/.env"
GUI_CONFIG="${1:-}"

CIR_ZMQ_NUM_TAPS="${CIR_ZMQ_NUM_TAPS:-48}"
IPERF_SERVER="192.168.72.135"
UE_TIMEOUT=120

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo -e "\033[1;32m[demo]\033[0m $*"; }
warn() { echo -e "\033[1;33m[demo]\033[0m $*"; }
err()  { echo -e "\033[1;31m[demo]\033[0m $*" >&2; }

wait_for_container() {
    local name=$1 timeout=${2:-60} elapsed=0
    log "Waiting for $name to be healthy (timeout: ${timeout}s)..."
    while true; do
        local status
        status=$(docker inspect --format='{{.State.Health.Status}}' "$name" 2>/dev/null || echo "not_found")
        case "$status" in
            healthy)    log "$name is ready."; return 0 ;;
            unhealthy)  err "$name became unhealthy!"; exit 1 ;;
            not_found)  err "Container $name not found!"; exit 1 ;;
        esac
        if (( elapsed >= timeout )); then
            err "Timeout waiting for $name!"
            exit 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
}

wait_for_ue_ip() {
    local timeout=$1 elapsed=0
    log "Waiting for UE to obtain an IP address (timeout: ${timeout}s)..."
    while true; do
        UE_IP=$(docker exec oai-nr-ue ip -4 addr show oaitun_ue1 2>/dev/null \
                | grep -oP 'inet\s+\K[\d.]+' || true)
        if [ -n "$UE_IP" ]; then
            log "UE connected with IP: $UE_IP"
            return 0
        fi
        if (( elapsed >= timeout )); then
            err "Timeout waiting for UE IP!"
            exit 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
}

# ---------------------------------------------------------------------------
# 1. Validate environment
# ---------------------------------------------------------------------------

if [[ ! -f "$ENV_FILE" ]]; then
    err "rfsim .env not found at $ENV_FILE"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Stop any running system
# ---------------------------------------------------------------------------

log "Stopping any running 5G network..."
cd "$CONFIGS_DIR/common"
docker compose down 2>/dev/null || true
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 3. Configure the channel emulator (ZMQ mode) in the env file
# ---------------------------------------------------------------------------

# Ensure channel emulation ZMQ line is active
if ! grep -q '^GNB_EXTRA_OPTIONS=.*cir-zmq-num-taps' "$ENV_FILE"; then
    warn "Activating channel emulator (ZMQ mode) in $ENV_FILE"

    # Comment out any existing active GNB_EXTRA_OPTIONS
    sed -i 's/^GNB_EXTRA_OPTIONS=/#GNB_EXTRA_OPTIONS=/' "$ENV_FILE"

    # Uncomment or append the ZMQ line
    if grep -q '#.*GNB_EXTRA_OPTIONS=.*cir-zmq-num-taps' "$ENV_FILE"; then
        sed -i '0,/#.*GNB_EXTRA_OPTIONS=.*cir-zmq-num-taps/{s/^#\s*//}' "$ENV_FILE"
    else
        echo "GNB_EXTRA_OPTIONS=\"--cir-zmq-num-taps ${CIR_ZMQ_NUM_TAPS}\"" >> "$ENV_FILE"
    fi
fi
log "Using config: rfsim (env: $ENV_FILE)"

# ---------------------------------------------------------------------------
# 4. Start the 5G network
# ---------------------------------------------------------------------------

cd "$CONFIGS_DIR/common"

log "Starting 5G Core..."
docker compose --env-file "$ENV_FILE" up -d mysql oai-amf oai-smf oai-upf oai-ext-dn nearRT-RIC

wait_for_container "oai-mysql"
wait_for_container "oai-amf"
wait_for_container "oai-smf"
wait_for_container "oai-upf"
wait_for_container "oai-ext-dn"

log "Starting gNB (with channel emulator)..."
docker compose --env-file "$ENV_FILE" up -d oai-gnb
wait_for_container "oai-gnb"

log "Starting UE..."
docker compose --env-file "$ENV_FILE" up -d oai-nr-ue
wait_for_container "oai-nr-ue"

log "Starting stats xApp..."
docker compose --env-file "$ENV_FILE" up -d monitor_xapp

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 5. Wait for UE connectivity
# ---------------------------------------------------------------------------

wait_for_ue_ip "$UE_TIMEOUT"

# Quick connectivity check
log "Testing connectivity..."
if docker exec oai-nr-ue ping -c 2 -W 2 "$IPERF_SERVER" > /dev/null 2>&1; then
    log "UE can reach the traffic generator."
else
    warn "Ping to $IPERF_SERVER failed (may still work for iperf)."
fi

# ---------------------------------------------------------------------------
# 6. Start iperf3 (DL traffic via the UE)
# ---------------------------------------------------------------------------

log "Starting iperf3 (downlink, continuous)..."
docker exec -d oai-nr-ue iperf3 -t 86400 -i 1 -B "$UE_IP" -c "$IPERF_SERVER" -R

sleep 2
if docker exec oai-nr-ue pgrep -f "iperf3.*$IPERF_SERVER" > /dev/null 2>&1; then
    log "iperf3 running."
else
    warn "iperf3 may not have started. Check: docker exec oai-nr-ue pgrep iperf3"
fi

# ---------------------------------------------------------------------------
# 7. Start MPS + Sionna RT GUI
# ---------------------------------------------------------------------------

if [[ "${SKIP_GUI:-0}" == "1" ]]; then
    log "Skipping GUI (SKIP_GUI=1)."
    log "Channel emulator ZMQ is available on host port 5556."
    log "Stats ZMQ is available on host port 5555."
else
    if [ -z "$GUI_CONFIG" ]; then
        warn "No GUI config provided."
        echo ""
        echo "  To start the GUI manually:"
        echo "    plugins/channel_emulation/scripts/start_mps_gui.sh <config.yaml>"
        echo ""
        echo "  The channel emulator ZMQ is available on host port 5556."
        echo "  The stats ZMQ is available on host port 5555."
    else
        log "Starting MPS and Sionna RT GUI..."
        exec "$REPO_ROOT/plugins/channel_emulation/scripts/start_mps_gui.sh" "$GUI_CONFIG"
    fi
fi

echo ""
log "Demo is running!"
echo ""
echo "  Useful commands:"
echo "    docker logs -f oai-gnb          # gNB logs"
echo "    docker logs -f oai-nr-ue        # UE logs"
echo "    docker exec oai-nr-ue iperf3 -c $IPERF_SERVER -R -t 10  # quick DL test"
echo ""
echo "  To stop:"
echo "    ./scripts/stop_system.sh"
echo ""
