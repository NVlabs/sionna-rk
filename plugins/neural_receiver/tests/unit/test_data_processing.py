#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np

@pytest.fixture
def data_processing_module(compiled_modules):
    """Get the data processing module from compiled modules."""
    return compiled_modules["data_processing"]


@pytest.mark.parametrize("num_symbols", [(1024), (32), (24), (2), (1), (0)])
def test_symbol_conversion(data_processing_module, num_symbols):
    dp = data_processing_module
    symbols_i = np.random.randint(-32768, 32767, size=(num_symbols, 2), dtype=np.int16)
    scale = np.random.uniform(0.1, 10.0)

    symbols_h = dp.int16_symbols_to_float16(symbols_i, scale)

    symbols_h_ref = (scale * np.ldexp(symbols_i.astype(np.float32), -8)).astype(np.float16)
    assert np.all(symbols_h == symbols_h_ref)

@pytest.mark.parametrize("num_symbols", [(1024), (32), (24), (2), (1), (0)])
@pytest.mark.parametrize("num_bits", [(2), (4)])
def test_llr_conversion(data_processing_module, num_symbols, num_bits):
    dp = data_processing_module
    llrs_h = np.floor(np.random.random_sample(size=(num_symbols, num_bits)) * 256 - 128).clip(min=-128, max=127).astype(np.float16)
    llrs_i_ref = np.rint(np.ldexp(llrs_h.astype(np.float32), 8)).astype(np.int16)

    llrs_i = dp.float16_llrs_to_int16(llrs_h)

    assert np.all(llrs_i == llrs_i_ref)

@pytest.mark.parametrize("num_symbols", [(1024), (32), (24), (2), (1), (0)])
@pytest.mark.parametrize("num_bits", [(2), (4)])
def test_inversion(data_processing_module, num_symbols, num_bits):
    dp = data_processing_module
    llrs_h = np.floor(np.random.random_sample(size=(num_symbols, num_bits)) * 256 - 128).clip(min=-128, max=127).astype(np.float16)

    llrs_i = dp.float16_llrs_to_int16(llrs_h)
    llrs_h_rev = dp.int16_symbols_to_float16(llrs_i.copy().reshape(-1, 2), 1.0).reshape(-1, num_bits)

    assert np.all(llrs_h == llrs_h_rev)

    llrs_i_ref = np.rint(np.ldexp(llrs_h.astype(np.float32), 8)).astype(np.int16)
    assert np.all(llrs_i == llrs_i_ref)

@pytest.mark.parametrize("num_symbols", [(1024), (288), (32), (24), (2), (1)])
@pytest.mark.parametrize("num_ofdm", [(2), (13), (15), (16)])
@pytest.mark.parametrize("num_antennas", [(1), (2), (4)])
def test_pad_reshape(data_processing_module, num_symbols, num_ofdm, num_antennas):
    dp = data_processing_module
    values = np.random.randint(0, 0xffffffff, size=(1, num_symbols, num_ofdm, num_antennas)).astype(np.uint32)
    out_shape = [1, max(1584, num_symbols), max(14, num_ofdm), max(4, num_antennas)]
    if values.size > 0:
        padded_ref = np.pad(values, ((0, out_shape[0]-values.shape[0]), (0,0), (0,0), (0, out_shape[3]-values.shape[3])), mode='edge')
        padded_ref = np.pad(padded_ref, ((0,0), (0, 0), (0, out_shape[2]-values.shape[2]), (0, 0)), mode='wrap')
        padded_ref = np.pad(padded_ref, ((0,0), (0, out_shape[1]-values.shape[1]), (0, 0), (0, 0)), mode='reflect')
    else:
        padded_ref = np.pad(values, tuple((0, os-values.shape[i]) for i, os in enumerate(out_shape)))

    padded = dp.reshape_and_pad_32bit(values, out_shape)

    assert np.all(padded == padded_ref)

@pytest.mark.parametrize("num_symbols", [(1024), (288), (32), (24), (2), (1), (0)])
@pytest.mark.parametrize("num_ofdm", [(2), (13), (15)])
@pytest.mark.parametrize("num_bits", [(2), (4)])
def test_gather_llrs(data_processing_module, num_symbols, num_ofdm, num_bits):
    dp = data_processing_module
    values = np.random.randint(-32768, 32767, size=(num_bits, num_symbols, num_ofdm), dtype=np.int16)
    out_shape = [ min(23, num_symbols), min(14, num_ofdm) ]
    llrs_ref = values[:,:out_shape[0],:out_shape[1]].transpose((1,2,0))

    llrs = dp.gather_transposed_llrs(values, out_shape)

    assert np.all(llrs == llrs_ref)

