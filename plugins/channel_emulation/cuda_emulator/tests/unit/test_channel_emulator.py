#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import numpy as np


# Test configurations
NUM_TAPS = 10
NUM_SYMBOLS = 14
SAMPLES_PER_SLOT = 7680
SAMPLES_PER_FRAME = 153600 # 20 slots per frame
FFT_SIZE = 512
CP_LEN0 = 44  # CP length for first symbol
CP_LEN = 36   # CP length for other symbols
MAX_TAP_DELAY = 256
SIGMA_MAX = 1e2  # No cap on noise std

# Test with various tap counts to ensure alignment works correctly
# Entry size = 4 + num_taps*8 + num_taps*2 = 4 + num_taps*10
# For proper alignment, entry size must be multiple of 4:
#   num_taps=1: 14 bytes -> needs 2 bytes padding
#   num_taps=3: 34 bytes -> needs 2 bytes padding
#   num_taps=5: 54 bytes -> needs 2 bytes padding
#   num_taps=10: 104 bytes -> already aligned
TAP_COUNTS_TO_TEST = [1, 3, 5, 10]


def generate_test_cir(num_taps, num_symbols, reverse_tap_indices=False):
    """
    Generate a simple test CIR with decaying taps.
    Returns cir_taps as [num_taps*num_symbols, 2] and tap_indices.
    """
    np.random.seed(42)

    cir_taps = np.zeros((num_taps * num_symbols, 2), dtype=np.float32)
    tap_indices = np.zeros(num_taps * num_symbols, dtype=np.uint16)

    for sym in range(num_symbols):
        for i in range(num_taps):
            idx = sym * num_taps + i
            a = np.random.normal(0, np.sqrt(0.5)) + 1j * np.random.normal(0, np.sqrt(0.5))
            cir_taps[idx, 0] = a.real
            cir_taps[idx, 1] = a.imag
            tap_indices[idx] = i if not reverse_tap_indices else num_taps - i - 1

    return cir_taps, tap_indices

def reference_channel(x, cir_taps, tap_indices, num_taps, num_symbols, samples_per_slot,
                        samples_first_symbol, samples_other_symbols, data_offset):
    """
    Reference implementation of the tapped delay line channel in Python.
    """
    y = np.zeros(samples_per_slot, dtype=np.complex64)
    x = np.roll(x, -data_offset + MAX_TAP_DELAY - 1, axis=0)

    # Compute symbol boundaries
    symbol_boundary = samples_first_symbol

    sym_idx = 0
    for s in range(samples_per_slot):
        # Find which symbol this sample belongs to
        if s >= symbol_boundary:
            sym_idx += 1
            symbol_boundary += samples_other_symbols

        cir_offset = sym_idx * num_taps

        # Apply tapped delay line
        for l in range(num_taps):
            tap_idx = cir_offset + l
            delay = tap_indices[tap_idx]
            assert delay <= MAX_TAP_DELAY

            h_real = cir_taps[tap_idx, 0]
            h_imag = cir_taps[tap_idx, 1]
            h = complex(h_real, h_imag)

            # To avoid overflow when subtracting unsigned (delay) from signed (s), cast delay to int
            x_idx = s - int(delay) + MAX_TAP_DELAY - 1
            x_val = complex(x[x_idx, 0], x[x_idx, 1])
            y[s] += h * x_val

    return y


@pytest.mark.parametrize("num_taps", TAP_COUNTS_TO_TEST)
def test_init_shutdown(compiled_emulator, num_taps):
    """Test that init and shutdown work correctly with various tap counts."""
    result = compiled_emulator.init(num_taps, NUM_SYMBOLS, 1.0, SIGMA_MAX)
    assert result == 0, f"Init should return 0 on success for num_taps={num_taps}"

    result = compiled_emulator.shutdown()
    assert result == 0, f"Shutdown should return 0 on success for num_taps={num_taps}"

@pytest.mark.parametrize("num_taps", TAP_COUNTS_TO_TEST)
def test_identity_channel(compiled_emulator, num_taps):
    """
    Test with an identity-like channel (first tap = 1, others = 0).
    The output should approximately match the input.
    Tests various tap counts to verify 4-byte alignment fix.
    """
    compiled_emulator.init(num_taps, NUM_SYMBOLS, 0.0, SIGMA_MAX)

    try:
        # Create test data - a simple sine wave
        data = np.zeros((SAMPLES_PER_FRAME, 2), dtype=np.int16)
        t = np.arange(SAMPLES_PER_SLOT)
        # Generate a test signal
        data[:SAMPLES_PER_SLOT, 0] = (1000 * np.sin(2 * np.pi * t / 100)).astype(np.int16)
        data[:SAMPLES_PER_SLOT, 1] = (1000 * np.cos(2 * np.pi * t / 100)).astype(np.int16)

        # Identity channel: first tap = 1, delay = 0
        cir_taps = np.zeros((num_taps * NUM_SYMBOLS, 2), dtype=np.float32)
        for sym in range(NUM_SYMBOLS):
            cir_taps[sym * num_taps, 0] = 1.0  # Real part = 1

        tap_indices = np.zeros(num_taps * NUM_SYMBOLS, dtype=np.uint16)

        # Store original data for comparison
        original_data = data.copy()

        cir_norms = np.square(cir_taps[:, 0]) + np.square(cir_taps[:, 1])
        cir_norms = np.reshape(cir_norms, (NUM_SYMBOLS, num_taps))
        cir_norms = np.sqrt(np.sum(cir_norms, axis=1))

        samples_first_symbol = FFT_SIZE + CP_LEN0
        samples_other_symbols = FFT_SIZE + CP_LEN

        compiled_emulator.compute(
            data, cir_norms, cir_taps, tap_indices,
            SAMPLES_PER_SLOT, SAMPLES_PER_FRAME,
            samples_first_symbol, samples_other_symbols,
            0
        )

        # Check that output is approximately the input
        # Allow some tolerance for float->int16 conversion
        output_i = data[:SAMPLES_PER_SLOT, 0].astype(np.float32)
        output_q = data[:SAMPLES_PER_SLOT, 1].astype(np.float32)
        input_i = original_data[:SAMPLES_PER_SLOT, 0].astype(np.float32)
        input_q = original_data[:SAMPLES_PER_SLOT, 1].astype(np.float32)

        # Compute error (skip first few samples due to edge effects)
        error_i = np.abs(output_i - input_i)
        error_q = np.abs(output_q - input_q)

        max_error = max(np.max(error_i), np.max(error_q))
        print(f"Max error with identity channel (num_taps={num_taps}): {max_error}")

        assert max_error == 0, f"Identity channel error too large: {max_error}"

    finally:
        compiled_emulator.shutdown()

def test_delayed_channel(compiled_emulator):
    """
    Test with a single-tap delayed channel.
    """
    compiled_emulator.init(NUM_TAPS, NUM_SYMBOLS, 0.0, SIGMA_MAX)

    try:
        delay = 5

        # Create test data - impulse at a specific position
        data = np.zeros((SAMPLES_PER_FRAME, 2), dtype=np.int16)
        impulse_pos = 100
        data[impulse_pos, 0] = 10000
        data[impulse_pos, 1] = 5000

        # Single tap with delay
        cir_taps = np.zeros((NUM_TAPS * NUM_SYMBOLS, 2), dtype=np.float32)
        tap_indices = np.zeros(NUM_TAPS * NUM_SYMBOLS, dtype=np.uint16)

        for sym in range(NUM_SYMBOLS):
            cir_taps[sym * NUM_TAPS, 0] = 1.0  # Real part = 1
            tap_indices[sym * NUM_TAPS] = delay

        cir_norms = np.square(cir_taps[:, 0]) + np.square(cir_taps[:, 1])
        cir_norms = np.reshape(cir_norms, (NUM_SYMBOLS, NUM_TAPS))
        cir_norms = np.sqrt(np.sum(cir_norms, axis=1))

        samples_first_symbol = FFT_SIZE + CP_LEN0
        samples_other_symbols = FFT_SIZE + CP_LEN

        compiled_emulator.compute(
            data, cir_norms, cir_taps, tap_indices,
            SAMPLES_PER_SLOT, SAMPLES_PER_FRAME,
            samples_first_symbol, samples_other_symbols,
            0
        )

        # The impulse should appear at position (impulse_pos + delay)
        expected_pos = impulse_pos + delay

        # Find the peak in output
        output_magnitude = np.sqrt(data[:SAMPLES_PER_SLOT, 0].astype(float)**2 +
                                   data[:SAMPLES_PER_SLOT, 1].astype(float)**2)
        peak_pos = np.argmax(output_magnitude)

        print(f"Expected peak at {expected_pos}, found at {peak_pos}")
        assert abs(peak_pos - expected_pos) <= 1, f"Peak position mismatch: expected {expected_pos}, got {peak_pos}"

    finally:
        compiled_emulator.shutdown()

def test_multi_tap_channel(compiled_emulator):
    """
    Test with multiple taps and verify the convolution is correct.
    """
    compiled_emulator.init(NUM_TAPS, NUM_SYMBOLS, 0.0, SIGMA_MAX)

    try:
        # Generate test CIR
        cir_taps, tap_indices = generate_test_cir(NUM_TAPS, NUM_SYMBOLS)

        # Create test data with random IQ samples
        np.random.seed(123)
        data = np.zeros((SAMPLES_PER_FRAME, 2), dtype=np.int16)
        data[:, 0] = np.random.randint(-1000, 1000, SAMPLES_PER_FRAME).astype(np.int16)
        data[:, 1] = np.random.randint(-1000, 1000, SAMPLES_PER_FRAME).astype(np.int16)

        # Keep copy for reference computation
        input_data = data.copy()

        cir_norms = np.square(cir_taps[:, 0]) + np.square(cir_taps[:, 1])
        cir_norms = np.reshape(cir_norms, (NUM_SYMBOLS, NUM_TAPS))
        cir_norms = np.sqrt(np.sum(cir_norms, axis=1))

        samples_first_symbol = FFT_SIZE + CP_LEN0
        samples_other_symbols = FFT_SIZE + CP_LEN

        compiled_emulator.compute(
            data, cir_norms, cir_taps, tap_indices,
            SAMPLES_PER_SLOT, SAMPLES_PER_FRAME,
            samples_first_symbol, samples_other_symbols,
            0
        )
        # Get CUDA output
        cuda_output = data[:SAMPLES_PER_SLOT, 0].astype(float)\
            + 1j * data[:SAMPLES_PER_SLOT, 1].astype(float)

        # Apply convolution to the input data
        reference_output = reference_channel(input_data, cir_taps, tap_indices, NUM_TAPS, NUM_SYMBOLS, SAMPLES_PER_SLOT,
                                            FFT_SIZE + CP_LEN0, FFT_SIZE + CP_LEN, 0)
        error = np.abs(cuda_output - reference_output)
        # Allow some tolerance for float->int16 conversion
        assert np.max(error) < 10, f"Error too large: {np.max(error)}"

    finally:
        compiled_emulator.shutdown()

def test_multi_tap_channel_with_offset(compiled_emulator):
    """
    Test with multiple taps and verify the convolution is correct.
    """
    compiled_emulator.init(NUM_TAPS, NUM_SYMBOLS, 0.0, SIGMA_MAX)

    offset = 1234

    try:
        # Generate test CIR
        cir_taps, tap_indices = generate_test_cir(NUM_TAPS, NUM_SYMBOLS)

        # Create test data with random IQ samples
        np.random.seed(456)
        data = np.zeros((SAMPLES_PER_FRAME, 2), dtype=np.int16)
        data[:, 0] = np.random.randint(-1000, 1000, SAMPLES_PER_FRAME).astype(np.int16)
        data[:, 1] = np.random.randint(-1000, 1000, SAMPLES_PER_FRAME).astype(np.int16)

        # Keep copy for reference computation
        input_data = data.copy()

        cir_norms = np.square(cir_taps[:, 0]) + np.square(cir_taps[:, 1])
        cir_norms = np.reshape(cir_norms, (NUM_SYMBOLS, NUM_TAPS))
        cir_norms = np.sqrt(np.sum(cir_norms, axis=1))

        samples_first_symbol = FFT_SIZE + CP_LEN0
        samples_other_symbols = FFT_SIZE + CP_LEN

        compiled_emulator.compute(
            data, cir_norms, cir_taps, tap_indices,
            SAMPLES_PER_SLOT, SAMPLES_PER_FRAME,
            samples_first_symbol, samples_other_symbols,
            offset
        )
        # Get CUDA output
        cuda_output = data[offset:offset+SAMPLES_PER_SLOT, 0].astype(float)\
            + 1j * data[offset:offset+SAMPLES_PER_SLOT, 1].astype(float)

        # Apply convolution to the input data
        reference_output = reference_channel(input_data, cir_taps, tap_indices, NUM_TAPS, NUM_SYMBOLS, SAMPLES_PER_SLOT,
                                            FFT_SIZE + CP_LEN0, FFT_SIZE + CP_LEN, offset)
        error = np.abs(cuda_output - reference_output)
        assert np.max(error) < 10, f"Error too large: {np.max(error)}"

    finally:
        compiled_emulator.shutdown()

def test_multi_tap_channel_with_varying_tap_indices(compiled_emulator):
    """
    Test with multiple taps and verify the convolution is correct.
    """
    compiled_emulator.init(NUM_TAPS, NUM_SYMBOLS, 0.0, SIGMA_MAX)

    try:
        # Generate test CIR
        cir_taps, tap_indices = generate_test_cir(NUM_TAPS, NUM_SYMBOLS, reverse_tap_indices=False)

        # Create test data with random IQ samples
        np.random.seed(789)
        data = np.zeros((SAMPLES_PER_FRAME, 2), dtype=np.int16)
        data[:, 0] = np.random.randint(-1000, 1000, SAMPLES_PER_FRAME).astype(np.int16)
        data[:, 1] = np.random.randint(-1000, 1000, SAMPLES_PER_FRAME).astype(np.int16)

        # Keep copy for reference computation
        input_data = data.copy()

        cir_norms = np.square(cir_taps[:, 0]) + np.square(cir_taps[:, 1])
        cir_norms = np.reshape(cir_norms, (NUM_SYMBOLS, NUM_TAPS))
        cir_norms = np.sqrt(np.sum(cir_norms, axis=1))

        samples_first_symbol = FFT_SIZE + CP_LEN0
        samples_other_symbols = FFT_SIZE + CP_LEN

        compiled_emulator.compute(
            data, cir_norms, cir_taps, tap_indices,
            SAMPLES_PER_SLOT, SAMPLES_PER_FRAME,
            samples_first_symbol, samples_other_symbols,
            0
        )
        # Get CUDA output
        cuda_output = data[:SAMPLES_PER_SLOT, 0].astype(float)\
            + 1j * data[:SAMPLES_PER_SLOT, 1].astype(float)

        # Apply convolution to the input data
        reference_output = reference_channel(input_data, cir_taps, tap_indices, NUM_TAPS, NUM_SYMBOLS, SAMPLES_PER_SLOT,
                                            FFT_SIZE + CP_LEN0, FFT_SIZE + CP_LEN, 0)
        error = np.abs(cuda_output - reference_output)
        assert np.max(error) < 10, f"Error too large: {np.max(error)}"

    finally:
        compiled_emulator.shutdown()

def test_cir_norms(compiled_emulator):
    """
    Test that noise is properly added to the signal.
    """
    sigma_scaling = 1e5
    compiled_emulator.init(NUM_TAPS, NUM_SYMBOLS, sigma_scaling, np.inf)

    try:
        # Generate test CIR
        cir_taps = np.zeros((NUM_TAPS * NUM_SYMBOLS, 2), dtype=np.float32)
        tap_indices = np.zeros(NUM_TAPS * NUM_SYMBOLS, dtype=np.uint16)

        # Create test data with random IQ samples
        np.random.seed(789)
        data = np.zeros((SAMPLES_PER_FRAME, 2), dtype=np.int16)
        data[:, 0] = np.zeros(SAMPLES_PER_FRAME).astype(np.int16)
        data[:, 1] = np.zeros(SAMPLES_PER_FRAME).astype(np.int16)

        cir_norms = np.random.uniform(100.0, 1000.0, NUM_SYMBOLS).astype(np.float32)
        samples_first_symbol = FFT_SIZE + CP_LEN0
        samples_other_symbols = FFT_SIZE + CP_LEN

        # Run multiple times to get an estimate of the noise std
        cuda_outputs = []
        for _ in range(10000):
            compiled_emulator.compute(
                data, cir_norms, cir_taps, tap_indices,
                SAMPLES_PER_SLOT, SAMPLES_PER_FRAME,
                samples_first_symbol, samples_other_symbols,
                0
            )
            # Get CUDA output
            cuda_output = data[:SAMPLES_PER_SLOT, 0].astype(float)\
                + 1j * data[:SAMPLES_PER_SLOT, 1].astype(float)
            cuda_outputs.append(cuda_output)
        cuda_outputs = np.stack(cuda_outputs, axis=1)

        # Average over the samples of OFDM symbols
        var_estimates_ = np.mean(np.square(np.abs(cuda_outputs)), axis=1)
        next_symbol_start = samples_first_symbol
        symbol_index = 0
        var_estimates = np.zeros(NUM_SYMBOLS)
        for s in range(SAMPLES_PER_SLOT):
            if s >= next_symbol_start:
                if symbol_index == 0:
                    var_estimates[symbol_index] /= samples_first_symbol
                else:
                    var_estimates[symbol_index] /= samples_other_symbols
                symbol_index += 1
                next_symbol_start += samples_other_symbols
            var_estimates[symbol_index] += var_estimates_[s]
        var_estimates[-1] /= samples_other_symbols
        noise_std_estimate = np.sqrt(var_estimates)

        # Compute relative error
        expected_noise_std = sigma_scaling/cir_norms
        rel_err = np.abs(noise_std_estimate - expected_noise_std) / expected_noise_std
        assert np.max(rel_err) < 0.05, f"Error too large: {np.max(rel_err)}"

    finally:
        compiled_emulator.shutdown()

def test_sigma_scaling(compiled_emulator):
    """
    Test that noise is properly added to the signal.
    """
    sigma_scaling = 1e4
    compiled_emulator.init(NUM_TAPS, NUM_SYMBOLS, sigma_scaling, np.inf)

    try:
        # Generate test CIR
        cir_taps = np.zeros((NUM_TAPS * NUM_SYMBOLS, 2), dtype=np.float32)
        tap_indices = np.zeros(NUM_TAPS * NUM_SYMBOLS, dtype=np.uint16)

        # Create test data with random IQ samples
        np.random.seed(789)
        data = np.zeros((SAMPLES_PER_FRAME, 2), dtype=np.int16)
        data[:, 0] = np.zeros(SAMPLES_PER_FRAME).astype(np.int16)
        data[:, 1] = np.zeros(SAMPLES_PER_FRAME).astype(np.int16)

        cir_norms = np.ones(NUM_SYMBOLS, dtype=np.float32)
        samples_first_symbol = FFT_SIZE + CP_LEN0
        samples_other_symbols = FFT_SIZE + CP_LEN

        # Run multiple times to get an estimate of the noise std
        cuda_outputs = []
        for _ in range(10000):
            compiled_emulator.compute(
                data, cir_norms, cir_taps, tap_indices,
                SAMPLES_PER_SLOT, SAMPLES_PER_FRAME,
                samples_first_symbol, samples_other_symbols,
                0
            )
            # Get CUDA output
            cuda_output = data[:SAMPLES_PER_SLOT, 0].astype(float)\
                + 1j * data[:SAMPLES_PER_SLOT, 1].astype(float)
            cuda_outputs.append(cuda_output)
        cuda_outputs = np.stack(cuda_outputs, axis=1)

        # Average over the samples of OFDM symbols
        var_estimates_ = np.mean(np.square(np.abs(cuda_outputs)), axis=1)
        next_symbol_start = samples_first_symbol
        symbol_index = 0
        var_estimates = np.zeros(NUM_SYMBOLS)
        for s in range(SAMPLES_PER_SLOT):
            if s >= next_symbol_start:
                if symbol_index == 0:
                    var_estimates[symbol_index] /= samples_first_symbol
                else:
                    var_estimates[symbol_index] /= samples_other_symbols
                symbol_index += 1
                next_symbol_start += samples_other_symbols
            var_estimates[symbol_index] += var_estimates_[s]
        var_estimates[-1] /= samples_other_symbols
        noise_std_estimate = np.sqrt(var_estimates)

        # Compute relative error
        rel_err = np.abs(noise_std_estimate - sigma_scaling) / sigma_scaling
        assert np.max(rel_err) < 0.05, f"Error too large: {np.max(rel_err)}"

    finally:
        compiled_emulator.shutdown()

def test_sigma_max(compiled_emulator):
    """
    Test that noise is properly added to the signal.
    """
    # sigma_scaling > sigma_max
    sigma_scaling = 1e4
    sigma_max = 1e3
    compiled_emulator.init(NUM_TAPS, NUM_SYMBOLS, sigma_scaling, sigma_max)

    try:
        # Generate test CIR
        cir_taps = np.zeros((NUM_TAPS * NUM_SYMBOLS, 2), dtype=np.float32)
        tap_indices = np.zeros(NUM_TAPS * NUM_SYMBOLS, dtype=np.uint16)

        # Create test data with random IQ samples
        np.random.seed(789)
        data = np.zeros((SAMPLES_PER_FRAME, 2), dtype=np.int16)
        data[:, 0] = np.zeros(SAMPLES_PER_FRAME).astype(np.int16)
        data[:, 1] = np.zeros(SAMPLES_PER_FRAME).astype(np.int16)

        cir_norms = np.ones(NUM_SYMBOLS, dtype=np.float32)
        samples_first_symbol = FFT_SIZE + CP_LEN0
        samples_other_symbols = FFT_SIZE + CP_LEN

        # Run multiple times to get an estimate of the noise std
        cuda_outputs = []
        for _ in range(10000):
            compiled_emulator.compute(
                data, cir_norms, cir_taps, tap_indices,
                SAMPLES_PER_SLOT, SAMPLES_PER_FRAME,
                samples_first_symbol, samples_other_symbols,
                0
            )
            # Get CUDA output
            cuda_output = data[:SAMPLES_PER_SLOT, 0].astype(float)\
                + 1j * data[:SAMPLES_PER_SLOT, 1].astype(float)
            cuda_outputs.append(cuda_output)
        cuda_outputs = np.stack(cuda_outputs, axis=1)

        # Average over the samples of OFDM symbols
        var_estimates_ = np.mean(np.square(np.abs(cuda_outputs)), axis=1)
        next_symbol_start = samples_first_symbol
        symbol_index = 0
        var_estimates = np.zeros(NUM_SYMBOLS)
        for s in range(SAMPLES_PER_SLOT):
            if s >= next_symbol_start:
                if symbol_index == 0:
                    var_estimates[symbol_index] /= samples_first_symbol
                else:
                    var_estimates[symbol_index] /= samples_other_symbols
                symbol_index += 1
                next_symbol_start += samples_other_symbols
            var_estimates[symbol_index] += var_estimates_[s]
        var_estimates[-1] /= samples_other_symbols
        noise_std_estimate = np.sqrt(var_estimates)
        print("noise std estimate: ", noise_std_estimate)

        # Compute relative error
        rel_err = np.abs(noise_std_estimate - sigma_max) / sigma_max
        assert np.max(rel_err) < 0.05, f"Error too large: {np.max(rel_err)}"

    finally:
        compiled_emulator.shutdown()
