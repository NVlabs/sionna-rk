#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Live test script for the CIR ZMQ plugin.
#
# Run this while the OAI gNB is running with cir_zmq enabled. The script
# connects to the gNB's ZMQ REP socket (default: localhost:5556 when gNB is
# in Docker, since container port 5555 is mapped to host 5556), exercises
# all message types (config_req, cir, nrx), and runs a timed sequence of
# CIR updates using a scaled Dirac to vary effective SNR.
#
# Usage:
#   python test_cir_zmq_live.py [--host HOST] [--port PORT] [--wait-secs N] [--steps N] [--no-nrx]
#
# Example (gNB in Docker):
#   python test_cir_zmq_live.py --host localhost --port 5556 --wait-secs 10
#

import argparse
import json
import sys
import time

import zmq


DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5556
DEFAULT_WAIT_SECS = 10
DEFAULT_STEPS = 4
SOCKET_TIMEOUT_MS = 10000


def parse_args():
    p = argparse.ArgumentParser(
        description="Live test CIR ZMQ plugin on a running OAI gNB"
    )
    p.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"gNB ZMQ host (default: {DEFAULT_HOST})",
    )
    p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"gNB ZMQ port (default: {DEFAULT_PORT}, use 5556 when gNB is in Docker)",
    )
    p.add_argument(
        "--wait-secs",
        type=float,
        default=DEFAULT_WAIT_SECS,
        help=f"Seconds to wait between CIR steps (default: {DEFAULT_WAIT_SECS})",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of CIR update steps in the SNR sequence (default: {DEFAULT_STEPS})",
    )
    p.add_argument(
        "--no-nrx",
        action="store_true",
        help="Skip NRX (custom receiver) message tests",
    )
    p.add_argument(
        "--no-tap-scale",
        action="store_true",
        help="Skip CIR tap-scaling sequence (tap_scale 1, 0.5, 0.25, noise constant)",
    )
    p.add_argument(
        "--expect-num-taps",
        type=int,
        default=None,
        help="If set, fail config_req when gNB num_taps does not match (e.g. 10)",
    )
    p.add_argument(
        "--expect-fft-size",
        type=int,
        default=None,
        help="If set, fail config_req when gNB fft_size does not match (e.g. 2048)",
    )
    return p.parse_args()


def send_request(socket, req):
    """Send a JSON request and return the parsed JSON response."""
    socket.send_string(json.dumps(req))
    try:
        raw = socket.recv_string()
    except zmq.error.Again as exc:
        raise TimeoutError(
            f"ZMQ receive timed out after {SOCKET_TIMEOUT_MS} ms waiting for server reply"
        ) from exc
    return json.loads(raw)


def build_scaled_dirac_cir(S, T, tap_scale, sigma_scaling, sigma_max):
    """Build a CIR message with a scaled Dirac: one tap per symbol at delay 0.

    - norms[s] = 1.0 for all s
    - taps: for each symbol s, taps[s*2*T + 0] = tap_scale (I), taps[s*2*T + 1] = 0 (Q); rest zero
    - tap_indices[s*T : (s+1)*T] = [0, 1, ..., T-1]
    """
    norms = [1.0] * S
    taps = [0.0] * (S * 2 * T)
    for s in range(S):
        taps[s * 2 * T + 0] = tap_scale  # I
        taps[s * 2 * T + 1] = 0.0        # Q
    tap_indices = list(range(T)) * S
    return {
        "msg_type": "cir",
        "sigma_scaling": sigma_scaling,
        "sigma_max": sigma_max,
        "norms": norms,
        "taps": taps,
        "tap_indices": tap_indices,
    }


def run_config_test(socket, expect_num_taps=None, expect_fft_size=None):
    """Send config_req, return (num_taps, num_ofdm_symbols_per_slot)."""
    print("Sending config_req ...")
    resp = send_request(socket, {"msg_type": "config_req"})
    if resp.get("msg_type") != "config_res":
        print(f"Unexpected response: {resp}", file=sys.stderr)
        sys.exit(1)
    T = int(resp["num_taps"])
    S = int(resp["num_ofdm_symbols_per_slot"])
    fft_size = resp.get("fft_size", "?")
    scs = resp.get("subcarrier_spacing", "?")
    freq = resp.get("frequency", "?")
    print(f"  num_taps={T}, num_ofdm_symbols_per_slot={S}, fft_size={fft_size}, scs={scs}, frequency={freq}")
    if expect_num_taps is not None and T != expect_num_taps:
        print(f"  ERROR: expected num_taps={expect_num_taps}, got {T}", file=sys.stderr)
        sys.exit(1)
    if expect_fft_size is not None:
        got_fft = resp.get("fft_size")
        if got_fft != expect_fft_size:
            print(f"  ERROR: expected fft_size={expect_fft_size}, got {got_fft}", file=sys.stderr)
            sys.exit(1)
    return T, S


def run_cir_sequence(socket, S, T, wait_secs, steps):
    """Run a timed sequence of CIR updates: baseline -> lower SNR -> lower SNR -> restore."""
    # Sequence: (sigma_scaling, sigma_max, tap_scale) to decrease then restore SNR
    if steps >= 4:
        sequence = [
            (1.0, 1.0, 1.0),   # baseline
            (2.0, 2.0, 1.0),   # lower SNR
            (3.0, 3.0, 1.0),   # even lower SNR
            (1.0, 1.0, 1.0),   # restore
        ]
        # pad or trim to match --steps
        sequence = (sequence * ((steps + 3) // 4))[:steps]
    else:
        sequence = [(1.0 + 0.5 * i, 1.0 + 0.5 * i, 1.0) for i in range(steps)]

    for i, (sigma_scaling, sigma_max, tap_scale) in enumerate(sequence):
        req = build_scaled_dirac_cir(S, T, tap_scale, sigma_scaling, sigma_max)
        print(f"CIR step {i + 1}/{steps}: sigma_scaling={sigma_scaling}, sigma_max={sigma_max}, tap_scale={tap_scale}")
        resp = send_request(socket, req)
        if resp.get("msg_type") != "cir_ack":
            print(f"  Unexpected response: {resp}", file=sys.stderr)
            sys.exit(1)
        print("  -> cir_ack")
        if i < steps - 1:
            print(f"  Waiting {wait_secs}s ...")
            time.sleep(wait_secs)


def run_cir_tap_sequence(socket, S, T, wait_secs):
    """Run CIR updates with constant noise, tap_scale 1 -> 0.5 -> 0.25 -> 1 (restore)."""
    sequence = [
        (1.0, 1.0, 1.0),    # baseline
        (1.0, 1.0, 0.5),    # half tap
        (1.0, 1.0, 0.25),   # quarter tap
        (1.0, 1.0, 1.0),    # restore
    ]
    for i, (sigma_scaling, sigma_max, tap_scale) in enumerate(sequence):
        req = build_scaled_dirac_cir(S, T, tap_scale, sigma_scaling, sigma_max)
        print(f"CIR tap step {i + 1}/{len(sequence)}: sigma_scaling={sigma_scaling}, sigma_max={sigma_max}, tap_scale={tap_scale} (noise constant)")
        resp = send_request(socket, req)
        if resp.get("msg_type") != "cir_ack":
            print(f"  Unexpected response: {resp}", file=sys.stderr)
            sys.exit(1)
        print("  -> cir_ack")
        if i < len(sequence) - 1:
            print(f"  Waiting {wait_secs}s ...")
            time.sleep(wait_secs)


def run_nrx_tests(socket):
    """Send nrx enabled=true, then enabled=false; check nrx_ack."""
    print("Sending nrx enabled=true ...")
    resp = send_request(socket, {"msg_type": "nrx", "enabled": True})
    if resp.get("msg_type") != "nrx_ack" or resp.get("enabled") != 1:
        print(f"  Unexpected response: {resp}", file=sys.stderr)
        sys.exit(1)
    print("  -> nrx_ack enabled=1")

    print("Sending nrx enabled=false ...")
    resp = send_request(socket, {"msg_type": "nrx", "enabled": False})
    if resp.get("msg_type") != "nrx_ack" or resp.get("enabled") != 0:
        print(f"  Unexpected response: {resp}", file=sys.stderr)
        sys.exit(1)
    print("  -> nrx_ack enabled=0")


def run_unknown_message_test(socket):
    """Send unknown msg_type, expect error response."""
    print("Sending unknown msg_type (expect error) ...")
    resp = send_request(socket, {"msg_type": "unknown_type"})
    if resp.get("msg_type") != "error":
        print(f"  Expected error response, got: {resp}", file=sys.stderr)
        sys.exit(1)
    err = resp.get("error", "")
    details = resp.get("details", "")
    print(f"  -> expected error response: {err}" + (f" ({details})" if details else ""))


def main():
    args = parse_args()
    endpoint = f"tcp://{args.host}:{args.port}"
    print(f"Connecting to {endpoint} ...")

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, SOCKET_TIMEOUT_MS)
    socket.setsockopt(zmq.SNDTIMEO, SOCKET_TIMEOUT_MS)
    socket.connect(endpoint)

    try:
        T, S = run_config_test(
            socket,
            expect_num_taps=args.expect_num_taps,
            expect_fft_size=args.expect_fft_size,
        )
        print()

        run_cir_sequence(socket, S, T, args.wait_secs, args.steps)
        print()

        if not args.no_tap_scale:
            run_cir_tap_sequence(socket, S, T, args.wait_secs)
            print()

        if not args.no_nrx:
            run_nrx_tests(socket)
            print()

        run_unknown_message_test(socket)
        print()

        print("All tests passed (ZMQ protocol and replies).")
    except (TimeoutError, json.JSONDecodeError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
