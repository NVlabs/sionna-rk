#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

import numpy as np
import data_processing as dp

@pytest.mark.parametrize("num_symbols", [(1024), (32), (24), (2), (1), (0)])
def test_norm_symbol_conversion(num_symbols):
    symbols_i = np.random.randint(-32768, 32767, size=(num_symbols, 2), dtype=np.int16)
    magnitudes_i = np.random.randint(np.abs(symbols_i) / 2 + 1, np.abs(symbols_i), size=(num_symbols, 2), dtype=np.int16)
    symbols_h_ref = (symbols_i.astype(np.float32) / magnitudes_i.astype(np.float32)).astype(np.float16)

    symbols_h = dp.norm_int16_symbols_to_float16(symbols_i, magnitudes_i)

    assert np.all(symbols_h == symbols_h_ref)

@pytest.mark.parametrize("num_symbols", [(1024), (32), (24), (2), (1), (0)])
def test_symbol_conversion(num_symbols):
    symbols_i = np.random.randint(-32768, 32767, size=(num_symbols, 2), dtype=np.int16)
    symbols_h_ref = np.ldexp(symbols_i.astype(np.float32), -8).astype(np.float16)

    symbols_h = dp.int16_symbols_to_float16(symbols_i)

    assert np.all(symbols_h == symbols_h_ref)

@pytest.mark.parametrize("num_symbols", [(1024), (32), (24), (2), (1), (0)])
@pytest.mark.parametrize("num_bits", [(2), (3), (4)])
def test_llr_conversion(num_symbols, num_bits):
    llrs_h = np.floor(np.random.random_sample(size=(num_symbols, num_bits)) * 256 - 128).clip(min=-128, max=127).astype(np.float16)
    llrs_i_ref = np.rint(np.ldexp(llrs_h.astype(np.float32), 8)).astype(np.int16)

    llrs_i = dp.float16_llrs_to_int16(llrs_h)

    assert np.all(llrs_i == llrs_i_ref)

@pytest.mark.parametrize("num_symbols", [(1024), (32), (24), (2), (1), (0)])
@pytest.mark.parametrize("num_bits", [(2), (4)])
def test_inversion(num_symbols, num_bits):
    llrs_h = np.floor(np.random.random_sample(size=(num_symbols, num_bits)) * 256 - 128).clip(min=-128, max=127).astype(np.float16)

    llrs_i = dp.float16_llrs_to_int16(llrs_h)
    llrs_h_rev = dp.int16_symbols_to_float16(llrs_i.copy().reshape(-1, 2)).reshape(-1, num_bits)

    assert np.all(llrs_h == llrs_h_rev)

    llrs_i_ref = np.rint(np.ldexp(llrs_h.astype(np.float32), 8)).astype(np.int16)
    assert np.all(llrs_i == llrs_i_ref)
