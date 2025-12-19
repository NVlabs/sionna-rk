#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np
import os

@pytest.fixture
def trt_receiver_module(compiled_modules):
    """Get the TRT receiver module from compiled modules."""
    return compiled_modules["trt_receiver"]


@pytest.mark.parametrize("num_prbs", [(24)])
def test_trt_receiver_running(trt_receiver_module, num_prbs, batch_size=1, num_tx=1, num_antennas=1, num_ofdm_symbols=13, num_subcarriers_per_prb=12, num_pilots_per_prb=18, num_dmrs_symbols=3):
    """Test that the TRT receiver runs without crashing and produces non-zero output."""
    trt_nr = trt_receiver_module

    symbols_i = np.random.randint(-32768, 32767, size=(batch_size, num_prbs * num_subcarriers_per_prb, num_ofdm_symbols, num_antennas, 2), dtype=np.int16)
    h_i = np.random.randint(-32768, 32767, size=(batch_size, num_prbs * num_pilots_per_prb, num_tx, num_antennas, 2), dtype=np.int16)
    dmrs_port_mask_i = np.ones((batch_size, num_tx), dtype=np.int16)
    dmrs_ofdm_pos = np.array([[2, 7, 11]], dtype=np.int32)  # Typical DMRS positions for additional_position=2
    dmrs_subcarrier_pos = np.array([[0, 2, 4, 6, 8, 10]], dtype=np.int32)  # Typical subcarrier positions

    symbols_h = np.ldexp(symbols_i.astype(np.float32), -8).astype(np.float16)
    h_h = np.ldexp(h_i.astype(np.float32), -8).astype(np.float16)
    dmrs_port_mask_h = np.ldexp(dmrs_port_mask_i.astype(np.float32), -8).astype(np.float16)

    llrs_h = trt_nr.run_nrx(symbols_h, h_h, dmrs_port_mask_h, dmrs_ofdm_pos, dmrs_subcarrier_pos, 4)

    assert np.any(llrs_h != 0), "TRT receiver output should contain non-zero values"


@pytest.mark.parametrize("num_prbs", [(24)])
def test_trt_decode(trt_receiver_module, num_prbs, batch_size=1, num_tx=1, num_antennas=1, num_ofdm_symbols=13, num_subcarriers_per_prb=12, num_pilots_per_prb=18, num_bits_per_symbol=4):
    """Test the decode function with random inputs."""
    trt_nr = trt_receiver_module

    symbols_i = np.random.randint(-32768, 32767, size=(batch_size, num_prbs * num_subcarriers_per_prb, num_ofdm_symbols, num_antennas, 2), dtype=np.int16)
    h_i = np.random.randint(-32768, 32767, size=(batch_size, num_prbs * num_pilots_per_prb, num_tx, num_antennas, 2), dtype=np.int16)
    dmrs_port_mask_i = np.ones((batch_size, num_tx), dtype=np.int16)
    dmrs_ofdm_pos = np.array([[2, 7, 11]], dtype=np.int32)
    prb_pilot_pos = np.array([[0, 2, 4, 6, 8, 10]], dtype=np.int32)
    scaling_factor = 10.0

    llrs_i = trt_nr.decode(symbols_i, h_i, 1/scaling_factor, dmrs_port_mask_i, dmrs_ofdm_pos, prb_pilot_pos, num_bits_per_symbol)

    assert llrs_i is not None, "decode() should return a result"
    assert np.any(llrs_i != 0), "decode() output should contain non-zero values"
