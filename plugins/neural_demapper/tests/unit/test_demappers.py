#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

import numpy as np

@pytest.fixture
def configured_trt_dm(compiled_modules):
    trt_dm = compiled_modules["trt_demapper"]
    import os
    # path to models relative to this test file
    # __file__ is .../tests/unit/test_demappers.py
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Always use base filename - the C++ runtime will apply PLAN_POSTFIX if set
    model_file = "neural_demapper_qam16_2.plan"
    model_path = os.path.join(base_dir, "models", model_file)
    
    # Check if the actual file exists (C++ will apply postfix at runtime)
    # For validation, check if the expected file with postfix exists if PLAN_POSTFIX is set
    plan_postfix = os.getenv("PLAN_POSTFIX", "")
    if plan_postfix:
        # Check that the postfixed file exists
        postfixed_path = os.path.join(base_dir, "models", f"neural_demapper_qam16_2{plan_postfix}.plan")
        if not os.path.exists(postfixed_path):
            pytest.skip(f"Model file not found: {postfixed_path}")
    else:
        # Check that the base file exists
        if not os.path.exists(model_path):
            pytest.skip(f"Model file not found: {model_path}")

    trt_dm.configure(model_path, 1) # 1 for normalized inputs
    return trt_dm

NUM_BITS_PER_SYMBOL = 4
QAM_THRESHOLD_MAG = 2 * 0.3162278

use_sionna_reference = True

if use_sionna_reference:
    import sionna as sn
    import tensorflow as tf

    # reference implementation
    mapper = sn.phy.mapping.Mapper(num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                               constellation_type="qam")
    demapper = sn.phy.mapping.Demapper(demapping_method="app",
                                   num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                   constellation_type="qam")

    EBN0_DB = 17.0 # Eb/N0 in dB
    # data sources
    binary_source = sn.phy.mapping.BinarySource()
    awgn_channel = sn.phy.channel.AWGN()
    no = sn.phy.utils.ebnodb2no(ebno_db=EBN0_DB,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)

def generate_symbols(num_symbols):
    magnitudes_i = np.random.randint(1, 32767, size=(num_symbols, 2), dtype=np.int16)
    magnitudes_h = np.ldexp(magnitudes_i.astype(np.float32), -8).astype(np.float16)

    if use_sionna_reference:
        no = tf.random.uniform([num_symbols,1], minval=0., maxval=1., dtype=tf.float32)
        bits = binary_source([num_symbols, NUM_BITS_PER_SYMBOL])
        x = mapper(bits)
        symbols_cx = awgn_channel(x, no).numpy()
        symbols_h = np.concatenate((np.real(symbols_cx), np.imag(symbols_cx)), axis=-1).astype(np.float16)
    else:
        bits = None
        symbols_h = np.random.uniform(-1.1, 1.1, size=magnitudes_i.shape).astype(np.float16)
        no = 0.4

    symbols_i = (symbols_h.astype(np.float32) * magnitudes_i / QAM_THRESHOLD_MAG).clip(min=-32768, max=32767).astype(np.int16)
    symbols_h = np.ldexp(symbols_i.astype(np.float32), -8).astype(np.float16)

    if use_sionna_reference:
        tf_symbols = tf.convert_to_tensor(symbols_h, dtype=tf.dtypes.float32)
        tf_magnitudes = tf.convert_to_tensor(magnitudes_h, dtype=tf.dtypes.float32)
        tf_no = tf.convert_to_tensor(no, dtype=tf.dtypes.float32)
        tf_symbols = tf_symbols / tf_magnitudes * QAM_THRESHOLD_MAG
        llrs_ref = demapper(tf.complex(tf_symbols[...,0:1],
                                       tf_symbols[...,1:2]),
                            tf_no)
        llrs_ref = -llrs_ref.numpy()#.reshape(-1, NUM_BITS_PER_SYMBOL)
    else:
        llrs_ref = None

    return symbols_i, magnitudes_i, bits, llrs_ref


@pytest.mark.parametrize("num_symbols", [(512), (32), (24), (2), (1)])
def test_trt_demapper_running(configured_trt_dm, num_symbols):
    symbols_i, magnitudes_i, _, _ = generate_symbols(num_symbols)

    symbols_h = np.ldexp(symbols_i.astype(np.float32), -8).astype(np.float16)
    magnitudes_h = np.ldexp(magnitudes_i.astype(np.float32), -8).astype(np.float16)

    unnormalized_symbols_h = np.concatenate((symbols_h, magnitudes_h), axis=-1)
    normalized_symbols_h = symbols_h / magnitudes_h

    llrs_h = configured_trt_dm.run_qam(normalized_symbols_h, NUM_BITS_PER_SYMBOL)

    assert np.any(llrs_h != 0)

@pytest.mark.parametrize("num_symbols", [(1024), (32), (24), (2), (1), (5000000)])
def test_trt_decode(configured_trt_dm, num_symbols):
    symbols_i, magnitudes_i, bits, llrs_ref = generate_symbols(num_symbols)

    llrs_i = configured_trt_dm.decode_qam(symbols_i, magnitudes_i, NUM_BITS_PER_SYMBOL)
    assert np.any(llrs_i != 0)

    llrs_h = np.ldexp(llrs_i.astype(np.float32), -8).astype(np.float16)

    if bits is not None:
        bit_signs = 1 - 2 * bits
        correct_mask = bit_signs == np.sign(llrs_h)
        llrs_h = np.where(correct_mask, bit_signs, llrs_h)
        llrs_ref = np.where(correct_mask, bit_signs, llrs_ref)

        assert np.count_nonzero(correct_mask) / llrs_h.size >= 0.5

    if llrs_ref is not None:
        decision_threshold = 0.66
        llrs_ref = (llrs_ref / decision_threshold).clip(-1, 1)
        llrs_h = (llrs_h / decision_threshold).clip(-1, 1)

        assert np.count_nonzero(np.abs(llrs_h - llrs_ref) < 1.0) / llrs_h.size >= 0.98
