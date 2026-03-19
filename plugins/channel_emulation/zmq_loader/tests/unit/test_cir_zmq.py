#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for the CIR ZMQ plugin.

These tests compile the plugin as a Python extension (via nanobind) and
exercise every ZMQ message type, error path, and public API function.
"""

import json
import threading
import time

import numpy as np
import pytest
import zmq


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

NUM_TAPS = 10
NUM_SYMBOLS = 14  # num_ofdm_symbols_per_slot
FFT_SIZE = 512
SUBCARRIER_SPACING = 30000.0
FREQUENCY = 3500000000.0
ZMQ_ENDPOINT = "tcp://localhost:5555"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def initialized_server(compiled_cir_zmq):
    """Initialise and yield the CIR ZMQ server, shut down after the test."""
    result = compiled_cir_zmq.init(NUM_TAPS, NUM_SYMBOLS, FFT_SIZE,
                                   SUBCARRIER_SPACING, FREQUENCY)
    assert result == 0, "Failed to initialise CIR ZMQ server"
    # Give the server socket and ZMQ thread time to bind and block in recv
    time.sleep(0.15)
    yield compiled_cir_zmq
    # Let the server thread finish any in-flight reply and return to recv before shutdown
    time.sleep(0.2)
    compiled_cir_zmq.shutdown()
    # Give time for socket cleanup before next test
    time.sleep(0.1)


@pytest.fixture
def zmq_client(initialized_server):
    """Create a ZMQ REQ socket connected to the server.

    Depends on *initialized_server* to guarantee the REP socket is ready.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)       # Don't block on close
    socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 s receive timeout
    socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 s send timeout
    socket.connect(ZMQ_ENDPOINT)
    time.sleep(0.05)  # Allow connection handshake
    yield socket
    socket.close()
    context.term()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def send_and_receive(module, zmq_client, request):
    """Send *request* via *zmq_client* while *module*.receive() runs in a
    background thread.  Returns the parsed JSON response.

    The ZMQ REQ/REP protocol requires that:
      1. The server calls recv (blocking) before the client sends.
      2. The server sends a reply before the client can recv.

    We therefore start ``module.receive()`` in a thread, give it a moment
    to block on zmq_recv, then send the client request and collect the
    response.
    """
    error_holder = [None]

    def do_receive():
        try:
            module.receive()
        except Exception as e:
            error_holder[0] = e

    # Start receive thread (blocks until a message arrives)
    recv_thread = threading.Thread(target=do_receive)
    recv_thread.start()
    time.sleep(0.05)  # Ensure receive is blocking before we send

    # Send request
    zmq_client.send_string(
        json.dumps(request) if isinstance(request, dict) else request
    )

    # Receive response
    try:
        raw_response = zmq_client.recv_string()
    except zmq.error.Again:
        raise TimeoutError("ZMQ receive timed out waiting for server reply")

    recv_thread.join(timeout=2)
    if error_holder[0]:
        raise error_holder[0]

    return json.loads(raw_response)


def send_request_server_loop(zmq_client, request):
    """Send *request* and return the parsed JSON response.

    Does NOT start a thread that calls module.receive(). Use this only when
    the server was started with init() and thus has its own ZMQ receive thread
    (cir_zmq_run). This exercises the production code path and would fail
    (timeout) if init() did not start the receive thread.
    """
    zmq_client.send_string(
        json.dumps(request) if isinstance(request, dict) else request
    )
    try:
        raw_response = zmq_client.recv_string()
    except zmq.error.Again:
        raise TimeoutError(
            "ZMQ receive timed out; server may not have a running receive thread"
        )
    return json.loads(raw_response)


def cir_message(norms=None, taps=None, tap_indices=None,
                sigma_scaling=1.0, sigma_max=0.5):
    """Build a CIR message dict with per-symbol flat arrays.

    Defaults produce valid arrays sized for NUM_SYMBOLS x NUM_TAPS.
    """
    S = NUM_SYMBOLS
    T = NUM_TAPS
    if norms is None:
        norms = [1.0] * S
    if taps is None:
        taps = [0.0] * (S * 2 * T)
    if tap_indices is None:
        tap_indices = list(range(T)) * S
    return {
        "msg_type": "cir",
        "sigma_scaling": sigma_scaling,
        "sigma_max": sigma_max,
        "norms": norms,
        "taps": taps,
        "tap_indices": tap_indices,
    }


# ---------------------------------------------------------------------------
# Init / Shutdown
# ---------------------------------------------------------------------------

def test_init_shutdown(compiled_cir_zmq):
    """Init followed by shutdown should both return 0."""
    assert compiled_cir_zmq.init(NUM_TAPS, NUM_SYMBOLS, FFT_SIZE,
                                  SUBCARRIER_SPACING, FREQUENCY) == 0
    assert compiled_cir_zmq.shutdown() == 0
    time.sleep(0.1)


def test_double_init_shutdown(compiled_cir_zmq):
    """Calling init twice (re-init) and shutdown twice must not crash."""
    assert compiled_cir_zmq.init(NUM_TAPS, NUM_SYMBOLS, FFT_SIZE,
                                  SUBCARRIER_SPACING, FREQUENCY) == 0
    # Re-initialise with different parameters
    assert compiled_cir_zmq.init(NUM_TAPS * 2, NUM_SYMBOLS, FFT_SIZE,
                                  SUBCARRIER_SPACING, FREQUENCY) == 0
    assert compiled_cir_zmq.get_num_taps() == NUM_TAPS * 2
    assert compiled_cir_zmq.shutdown() == 0
    # Second shutdown is safe (idempotent)
    assert compiled_cir_zmq.shutdown() == 0
    time.sleep(0.1)


# ---------------------------------------------------------------------------
# Config request
# ---------------------------------------------------------------------------

def test_config_request(initialized_server, zmq_client):
    """config_req should return all configuration parameters."""
    response = send_and_receive(
        initialized_server, zmq_client, {"msg_type": "config_req"}
    )

    assert response["msg_type"] == "config_res"
    assert response["num_taps"] == NUM_TAPS
    assert response["num_ofdm_symbols_per_slot"] == NUM_SYMBOLS
    assert response["fft_size"] == FFT_SIZE
    assert response["subcarrier_spacing"] == SUBCARRIER_SPACING
    assert response["frequency"] == FREQUENCY


@pytest.mark.skip(reason="In-process build does not start ZMQ thread; use live script against gNB")
def test_config_req_via_server_loop(initialized_server, zmq_client):
    """config_req must be answered by the server's own ZMQ thread (no manual receive).

    In the OAI build, init() starts cir_zmq_run() in a thread. In the Python
    (nanobind) build the thread is not started, so this test would timeout.
    Run scripts/test_cir_zmq_live.py against a running gNB to verify the thread.
    """
    response = send_request_server_loop(zmq_client, {"msg_type": "config_req"})
    assert response["msg_type"] == "config_res"
    assert response["num_taps"] == NUM_TAPS
    assert response["num_ofdm_symbols_per_slot"] == NUM_SYMBOLS


# ---------------------------------------------------------------------------
# CIR messages
# ---------------------------------------------------------------------------

def test_cir_message(initialized_server, zmq_client):
    """Sending valid CIR data should return a cir_ack."""
    taps = [float(i) * 0.1 for i in range(NUM_SYMBOLS * 2 * NUM_TAPS)]
    tap_indices = list(range(NUM_TAPS)) * NUM_SYMBOLS
    norms = [1.0] * NUM_SYMBOLS

    response = send_and_receive(
        initialized_server, zmq_client,
        cir_message(norms=norms, taps=taps, tap_indices=tap_indices),
    )

    assert response["msg_type"] == "cir_ack"


def test_cir_read(initialized_server, zmq_client):
    """After pushing CIR data, read() must return matching values."""
    expected_taps = [float(i) * 0.1 for i in range(NUM_SYMBOLS * 2 * NUM_TAPS)]
    expected_indices = list(range(NUM_TAPS)) * NUM_SYMBOLS
    expected_norms = [0.5 + 0.1 * s for s in range(NUM_SYMBOLS)]
    expected_sigma_scaling = 1.2
    expected_sigma_max = 0.8

    response = send_and_receive(
        initialized_server, zmq_client,
        cir_message(
            norms=expected_norms,
            taps=expected_taps,
            tap_indices=expected_indices,
            sigma_scaling=expected_sigma_scaling,
            sigma_max=expected_sigma_max,
        ),
    )
    assert response["msg_type"] == "cir_ack"

    sigma_scaling, sigma_max, norms, taps, tap_indices = initialized_server.read()

    assert abs(sigma_scaling - expected_sigma_scaling) < 1e-6
    assert abs(sigma_max - expected_sigma_max) < 1e-6
    np.testing.assert_array_almost_equal(norms, expected_norms, decimal=5)
    np.testing.assert_array_almost_equal(taps, expected_taps, decimal=5)
    np.testing.assert_array_equal(tap_indices, expected_indices)


def test_default_cir_after_init(initialized_server):
    """After init (no CIR message sent), read() should return meaningful defaults:
    sigma_scaling=1, sigma_max=1, norms all 1.0,
    first tap per symbol I=1.0 rest zero, tap_indices sequential."""
    sigma_scaling, sigma_max, norms, taps, tap_indices = initialized_server.read()

    S = NUM_SYMBOLS
    T = NUM_TAPS

    # Scalar defaults
    assert abs(sigma_scaling - 1.0) < 1e-6
    assert abs(sigma_max - 1.0) < 1e-6

    # norms: all 1.0
    assert len(norms) == S
    np.testing.assert_array_almost_equal(norms, np.ones(S), decimal=6)

    # taps: for each symbol, first tap I=1.0, Q=0.0, rest zero
    assert len(taps) == S * 2 * T
    taps_2d = np.array(taps).reshape(S, 2 * T)
    for s in range(S):
        assert abs(taps_2d[s, 0] - 1.0) < 1e-6, f"symbol {s}: tap[0] I should be 1.0"
        np.testing.assert_array_almost_equal(
            taps_2d[s, 1:], 0.0, decimal=6,
            err_msg=f"symbol {s}: remaining taps should be 0.0"
        )

    # tap_indices: [0, 1, 2, ..., T-1] for each symbol
    assert len(tap_indices) == S * T
    indices_2d = np.array(tap_indices).reshape(S, T)
    expected_row = np.arange(T, dtype=np.uint16)
    for s in range(S):
        np.testing.assert_array_equal(
            indices_2d[s], expected_row,
            err_msg=f"symbol {s}: tap_indices should be 0..{T-1}"
        )


def test_cir_roundtrip(initialized_server, zmq_client):
    """Send a CIR message with random values and verify read() returns them."""
    rng = np.random.default_rng(seed=42)

    S = NUM_SYMBOLS
    T = NUM_TAPS

    expected_sigma_scaling = float(rng.uniform(0.1, 10.0))
    expected_sigma_max = float(rng.uniform(0.1, 10.0))
    expected_norms = rng.uniform(0.1, 5.0, size=S).tolist()
    expected_taps = rng.uniform(-1.0, 1.0, size=S * 2 * T).tolist()
    expected_indices = rng.integers(0, 256, size=S * T).tolist()

    response = send_and_receive(
        initialized_server, zmq_client,
        cir_message(
            norms=expected_norms,
            taps=expected_taps,
            tap_indices=expected_indices,
            sigma_scaling=expected_sigma_scaling,
            sigma_max=expected_sigma_max,
        ),
    )
    assert response["msg_type"] == "cir_ack"

    sigma_scaling, sigma_max, norms, taps, tap_indices = initialized_server.read()

    assert abs(sigma_scaling - expected_sigma_scaling) < 1e-5
    assert abs(sigma_max - expected_sigma_max) < 1e-5
    np.testing.assert_array_almost_equal(norms, expected_norms, decimal=5)
    np.testing.assert_array_almost_equal(taps, expected_taps, decimal=5)
    np.testing.assert_array_equal(tap_indices, expected_indices)


def test_multiple_cir_updates(initialized_server, zmq_client):
    """Multiple sequential CIR updates should each be stored correctly."""
    S = NUM_SYMBOLS
    T = NUM_TAPS

    for i in range(3):
        norms = [1.0 + 0.1 * i] * S
        taps = [float(i + j) * 0.01 for j in range(S * 2 * T)]
        tap_indices = [(j + i) % 256 for j in range(S * T)]
        sigma_scaling = 0.5 + 0.2 * i
        sigma_max = 0.3 + 0.1 * i

        response = send_and_receive(
            initialized_server, zmq_client,
            cir_message(norms=norms, taps=taps, tap_indices=tap_indices,
                       sigma_scaling=sigma_scaling, sigma_max=sigma_max),
        )
        assert response["msg_type"] == "cir_ack"

        # Verify the latest data is stored
        r_sigma_scaling, r_sigma_max, r_norms, r_taps, r_indices = initialized_server.read()
        assert abs(r_sigma_scaling - sigma_scaling) < 1e-6
        assert abs(r_sigma_max - sigma_max) < 1e-6
        np.testing.assert_array_almost_equal(r_norms, norms, decimal=5)
        np.testing.assert_array_almost_equal(r_taps, taps, decimal=5)
        np.testing.assert_array_equal(r_indices, tap_indices)


# ---------------------------------------------------------------------------
# NRX (custom receiver) messages
# ---------------------------------------------------------------------------

def test_initial_nrx_state(initialized_server):
    """Custom receiver should be disabled by default after init."""
    assert initialized_server.receiver_symbols_requested() == 0


def test_nrx_message_enable(initialized_server, zmq_client):
    """Enabling the custom receiver should be acknowledged and take effect."""
    response = send_and_receive(
        initialized_server, zmq_client, {"msg_type": "nrx", "enabled": True}
    )

    assert response["msg_type"] == "nrx_ack"
    assert response["enabled"] == 1
    assert initialized_server.receiver_symbols_requested() == -1


def test_nrx_message_disable(initialized_server, zmq_client):
    """Disabling the custom receiver after enabling it should work."""
    # Enable first
    response = send_and_receive(
        initialized_server, zmq_client, {"msg_type": "nrx", "enabled": True}
    )
    assert response["msg_type"] == "nrx_ack"

    # Then disable
    response = send_and_receive(
        initialized_server, zmq_client, {"msg_type": "nrx", "enabled": False}
    )
    assert response["msg_type"] == "nrx_ack"
    assert response["enabled"] == 0
    assert initialized_server.receiver_symbols_requested() == 0


def test_nrx_missing_enabled_field(initialized_server, zmq_client):
    """NRX message without the 'enabled' field should return an error."""
    response = send_and_receive(
        initialized_server, zmq_client, {"msg_type": "nrx"}
    )
    assert response["msg_type"] == "error"


def test_nrx_invalid_enabled_field(initialized_server, zmq_client):
    """NRX message with a non-boolean 'enabled' should return an error."""
    response = send_and_receive(
        initialized_server, zmq_client, {"msg_type": "nrx", "enabled": "yes"}
    )
    assert response["msg_type"] == "error"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_invalid_json(initialized_server, zmq_client):
    """Malformed JSON should return an error response."""
    response = send_and_receive(
        initialized_server, zmq_client, "not valid json {{{"
    )

    assert response["msg_type"] == "error"
    assert "Invalid JSON" in response["error"]


def test_unknown_message_type(initialized_server, zmq_client):
    """Unknown msg_type should return an error response."""
    response = send_and_receive(
        initialized_server, zmq_client, {"msg_type": "unknown_type"}
    )

    assert response["msg_type"] == "error"
    assert "Unknown message type" in response["error"]


def test_missing_msg_type(initialized_server, zmq_client):
    """Missing msg_type field should return an error response."""
    response = send_and_receive(
        initialized_server, zmq_client, {"data": "some data"}
    )

    assert response["msg_type"] == "error"
    assert "msg_type" in response["error"]


def test_cir_taps_size_mismatch(initialized_server, zmq_client):
    """Wrong number of taps should return an error response."""
    response = send_and_receive(
        initialized_server, zmq_client,
        cir_message(taps=[0.1] * 5),  # taps length wrong
    )

    assert response["msg_type"] == "error"
    assert "mismatch" in response["error"].lower()


def test_cir_tap_indices_size_mismatch(initialized_server, zmq_client):
    """Wrong number of tap_indices should return an error response."""
    response = send_and_receive(
        initialized_server, zmq_client,
        cir_message(tap_indices=[0, 1, 2]),  # tap_indices length wrong
    )

    assert response["msg_type"] == "error"
    assert "mismatch" in response["error"].lower()


def test_cir_norms_size_mismatch(initialized_server, zmq_client):
    """Wrong number of norms should return an error response."""
    response = send_and_receive(
        initialized_server, zmq_client,
        cir_message(norms=[1.0, 2.0]),  # norms length wrong (should be NUM_SYMBOLS)
    )

    assert response["msg_type"] == "error"
    assert "mismatch" in response["error"].lower()


def test_cir_missing_fields(initialized_server, zmq_client):
    """CIR message missing required fields should return an error."""
    # Missing tap_indices, sigma_scaling, sigma_max, norms
    response = send_and_receive(
        initialized_server, zmq_client,
        {"msg_type": "cir", "taps": [0.1] * (NUM_SYMBOLS * 2 * NUM_TAPS)},
    )

    assert response["msg_type"] == "error"


def test_cir_missing_sigma_fields(initialized_server, zmq_client):
    """CIR message without sigma_scaling, sigma_max, or norms should error."""
    taps = [0.1] * (NUM_SYMBOLS * 2 * NUM_TAPS)
    tap_indices = list(range(NUM_TAPS)) * NUM_SYMBOLS
    # Omit sigma_scaling, sigma_max, norms
    response = send_and_receive(
        initialized_server, zmq_client,
        {
            "msg_type": "cir",
            "taps": taps,
            "tap_indices": tap_indices,
        },
    )

    assert response["msg_type"] == "error"
    assert ("sigma" in response["error"].lower()
            or "norms" in response["error"].lower()
            or "Invalid" in response["error"])


# ---------------------------------------------------------------------------
# Getter functions
# ---------------------------------------------------------------------------

def test_get_num_taps(initialized_server):
    """get_num_taps should return the value passed to init."""
    assert initialized_server.get_num_taps() == NUM_TAPS


def test_128_taps_supported(compiled_cir_zmq):
    """Initialise with 128 taps and send a full CIR message (buffer and logic support)."""
    T128 = 128
    result = compiled_cir_zmq.init(T128, NUM_SYMBOLS, FFT_SIZE,
                                  SUBCARRIER_SPACING, FREQUENCY)
    assert result == 0, "Failed to initialise CIR ZMQ with 128 taps"
    time.sleep(0.15)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.setsockopt(zmq.SNDTIMEO, 5000)
    sock.connect(ZMQ_ENDPOINT)
    time.sleep(0.05)

    try:
        # config_req must report 128 taps
        resp = send_and_receive(
            compiled_cir_zmq, sock, {"msg_type": "config_req"}
        )
        assert resp["msg_type"] == "config_res"
        assert resp["num_taps"] == T128

        # Full CIR message with 128 taps must be accepted (fits in 128 KB buffer)
        S, T = NUM_SYMBOLS, T128
        norms = [1.0] * S
        taps = [float(i % 10) * 0.1 for i in range(S * 2 * T)]
        tap_indices = list(range(T)) * S
        cir_req = cir_message(
            norms=norms, taps=taps, tap_indices=tap_indices,
            sigma_scaling=1.0, sigma_max=1.0
        )
        r = send_and_receive(compiled_cir_zmq, sock, cir_req)
        assert r["msg_type"] == "cir_ack"

        assert compiled_cir_zmq.get_num_taps() == T128
    finally:
        sock.close()
        ctx.term()
        compiled_cir_zmq.shutdown()
    time.sleep(0.1)


def test_get_sigma_scaling_default(initialized_server):
    """get_sigma_scaling should return 1.0 after init (before any CIR message)."""
    assert abs(initialized_server.get_sigma_scaling() - 1.0) < 1e-6


def test_get_sigma_max_default(initialized_server):
    """get_sigma_max should return 1.0 after init (before any CIR message)."""
    assert abs(initialized_server.get_sigma_max() - 1.0) < 1e-6


def test_get_sigma_after_cir(initialized_server, zmq_client):
    """After sending a CIR message, getters should reflect the new values."""
    response = send_and_receive(
        initialized_server, zmq_client,
        cir_message(sigma_scaling=2.5, sigma_max=3.7),
    )
    assert response["msg_type"] == "cir_ack"

    assert abs(initialized_server.get_sigma_scaling() - 2.5) < 1e-6
    assert abs(initialized_server.get_sigma_max() - 3.7) < 1e-6
