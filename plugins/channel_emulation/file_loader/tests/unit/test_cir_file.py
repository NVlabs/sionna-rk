#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path


# Test configurations
NUM_TAPS = 10
NUM_SYMBOLS_PER_SLOT = 14
NUM_SLOTS = 5
NUM_CIRS = NUM_SYMBOLS_PER_SLOT * NUM_SLOTS
SIGMA_SCALING = 1.0
SIGMA_MAX = 10.0


def create_test_cir_files(folder_path, num_taps, num_cirs, sigma_scaling, sigma_max,
                          seed=42, reverse_tap_indices=False):
    """
    Create test CIR files in a folder structure.

    Folder structure:
      - folder_path/config.json: JSON config with channel_emulation section
      - folder_path/cirs.bin: Binary file with packed CIR entries

    JSON format:
      {
        "channel_emulation": {
          "num_taps": N,
          "num_cirs": M,
          "sigma_scaling": F,
          "sigma_max": F
        }
      }

    Packed CIR entry format (per symbol):
      - float32: norm (channel norm for noise std computation)
      - float32[num_taps * 2]: Interleaved real/imag CIR tap values
      - uint16_t[num_taps]: Tap delay indices

    Returns the generated norms, taps, and tap_indices arrays for verification.
    """
    np.random.seed(seed)

    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Generate random CIR data for each symbol
    norms = np.zeros(num_cirs, dtype=np.float32)
    taps = np.zeros((num_cirs * num_taps, 2), dtype=np.float32)
    tap_indices = np.zeros(num_cirs * num_taps, dtype=np.uint16)

    for cir in range(num_cirs):
        # Generate norm: random value between 0.5 and 2.0
        norms[cir] = np.random.uniform(0.5, 2.0)

        for tap in range(num_taps):
            idx = cir * num_taps + tap
            # Generate random complex tap with decaying amplitude
            amplitude = np.exp(-0.2 * tap)
            real = np.random.normal(0, amplitude * np.sqrt(0.5))
            imag = np.random.normal(0, amplitude * np.sqrt(0.5))
            taps[idx, 0] = real
            taps[idx, 1] = imag
            # Tap indices (delay) - increasing with tap number
            tap_indices[idx] = tap if not reverse_tap_indices else num_taps - tap - 1

    # Create JSON config file with channel_emulation section
    json_path = os.path.join(folder_path, "config.json")
    config = {
        "channel_emulation": {
            "num_taps": num_taps,
            "num_cirs": num_cirs,
            "sigma_scaling": sigma_scaling,
            "sigma_max": sigma_max
        }
    }
    with open(json_path, 'w') as f:
        json.dump(config, f)

    # Calculate padded entry size (must match C code: 4-byte aligned)
    raw_entry_size = 4 + num_taps * 4 * 2 + num_taps * 2  # float32 + taps + uint16_t indices
    padded_entry_size = (raw_entry_size + 3) & ~3  # Align to 4 bytes
    padding_bytes = padded_entry_size - raw_entry_size

    # Create packed CIR binary data file
    cirs_bin_path = os.path.join(folder_path, "cirs.bin")
    with open(cirs_bin_path, 'wb') as f:
        for cir in range(num_cirs):
            # Write norm
            np.array([norms[cir]], dtype=np.float32).tofile(f)
            # Write taps (interleaved real/imag)
            start_tap = cir * num_taps
            end_tap = start_tap + num_taps
            taps[start_tap:end_tap].flatten().tofile(f)
            # Write tap indices
            tap_indices[start_tap:end_tap].tofile(f)
            # Write padding bytes for 4-byte alignment
            if padding_bytes > 0:
                f.write(b'\x00' * padding_bytes)

    return norms, taps, tap_indices


def test_init_shutdown(compiled_cir_file):
    """Test that init and shutdown work correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder_path = os.path.join(tmpdir, "test_cir_folder")
        create_test_cir_files(folder_path, NUM_TAPS, NUM_CIRS, SIGMA_SCALING, SIGMA_MAX)

        result = compiled_cir_file.init(folder_path, NUM_SYMBOLS_PER_SLOT)
        assert result == 0, f"Init should return 0 on success, got {result}"

        result = compiled_cir_file.shutdown()
        assert result == 0, f"Shutdown should return 0 on success, got {result}"


def test_get_num_taps(compiled_cir_file):
    """Test that get_num_taps returns the correct value after init."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder_path = os.path.join(tmpdir, "test_cir_folder")
        create_test_cir_files(folder_path, NUM_TAPS, NUM_CIRS, SIGMA_SCALING, SIGMA_MAX)

        compiled_cir_file.init(folder_path, NUM_SYMBOLS_PER_SLOT)

        try:
            num_taps = compiled_cir_file.get_num_taps()
            assert num_taps == NUM_TAPS, f"Expected {NUM_TAPS} taps, got {num_taps}"
        finally:
            compiled_cir_file.shutdown()


def test_get_sigma_parameters(compiled_cir_file):
    """Test that sigma_scaling and sigma_max are loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder_path = os.path.join(tmpdir, "test_cir_folder")
        create_test_cir_files(folder_path, NUM_TAPS, NUM_CIRS, SIGMA_SCALING, SIGMA_MAX)

        compiled_cir_file.init(folder_path, NUM_SYMBOLS_PER_SLOT)

        try:
            sigma_scaling = compiled_cir_file.get_sigma_scaling()
            sigma_max = compiled_cir_file.get_sigma_max()
            assert sigma_scaling == pytest.approx(SIGMA_SCALING), \
                f"Expected sigma_scaling={SIGMA_SCALING}, got {sigma_scaling}"
            assert sigma_max == pytest.approx(SIGMA_MAX), \
                f"Expected sigma_max={SIGMA_MAX}, got {sigma_max}"
        finally:
            compiled_cir_file.shutdown()


def test_read_slots(compiled_cir_file):
    """Test reading multiple slots and verify data progression."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder_path = os.path.join(tmpdir, "test_cir_folder")
        expected_norms, expected_taps, expected_indices = create_test_cir_files(
            folder_path, NUM_TAPS, NUM_CIRS, SIGMA_SCALING, SIGMA_MAX
        )

        compiled_cir_file.init(folder_path, NUM_SYMBOLS_PER_SLOT)

        try:
            taps_per_slot = NUM_TAPS * NUM_SYMBOLS_PER_SLOT

            for slot in range(NUM_SLOTS):
                norms, taps, tap_indices = compiled_cir_file.read(NUM_SYMBOLS_PER_SLOT)

                # Calculate expected range for this slot
                start_idx = slot * taps_per_slot
                end_idx = start_idx + taps_per_slot
                norm_start = slot * NUM_SYMBOLS_PER_SLOT
                norm_end = norm_start + NUM_SYMBOLS_PER_SLOT

                slot_expected_norms = expected_norms[norm_start:norm_end]
                slot_expected_taps = expected_taps[start_idx:end_idx]
                slot_expected_indices = expected_indices[start_idx:end_idx]

                returned_norms = np.array(norms)
                returned_taps = np.array(taps)
                returned_indices = np.array(tap_indices)

                np.testing.assert_allclose(returned_norms, slot_expected_norms, rtol=1e-5,
                    err_msg=f"Slot {slot} norms don't match expected values")
                np.testing.assert_allclose(returned_taps, slot_expected_taps, rtol=1e-5,
                    err_msg=f"Slot {slot} taps don't match expected values")
                np.testing.assert_array_equal(returned_indices, slot_expected_indices,
                    err_msg=f"Slot {slot} tap indices don't match expected values")
        finally:
            compiled_cir_file.shutdown()


def test_read_slots_reverse_tap_indices(compiled_cir_file):
    """Test reading multiple slots with reversed tap indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder_path = os.path.join(tmpdir, "test_cir_folder")
        expected_norms, expected_taps, expected_indices = create_test_cir_files(
            folder_path, NUM_TAPS, NUM_CIRS, SIGMA_SCALING, SIGMA_MAX, reverse_tap_indices=True
        )

        compiled_cir_file.init(folder_path, NUM_SYMBOLS_PER_SLOT)

        try:
            taps_per_slot = NUM_TAPS * NUM_SYMBOLS_PER_SLOT

            for slot in range(NUM_SLOTS):
                norms, taps, tap_indices = compiled_cir_file.read(NUM_SYMBOLS_PER_SLOT)

                # Calculate expected range for this slot
                start_idx = slot * taps_per_slot
                end_idx = start_idx + taps_per_slot
                norm_start = slot * NUM_SYMBOLS_PER_SLOT
                norm_end = norm_start + NUM_SYMBOLS_PER_SLOT

                slot_expected_norms = expected_norms[norm_start:norm_end]
                slot_expected_taps = expected_taps[start_idx:end_idx]
                slot_expected_indices = expected_indices[start_idx:end_idx]

                returned_norms = np.array(norms)
                returned_taps = np.array(taps)
                returned_indices = np.array(tap_indices)

                np.testing.assert_allclose(returned_norms, slot_expected_norms, rtol=1e-5,
                    err_msg=f"Slot {slot} norms don't match expected values")
                np.testing.assert_allclose(returned_taps, slot_expected_taps, rtol=1e-5,
                    err_msg=f"Slot {slot} taps don't match expected values")
                np.testing.assert_array_equal(returned_indices, slot_expected_indices,
                    err_msg=f"Slot {slot} tap indices don't match expected values")
        finally:
            compiled_cir_file.shutdown()


def test_slot_wraparound(compiled_cir_file):
    """Test that reading wraps around after all slots are consumed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder_path = os.path.join(tmpdir, "test_cir_folder")
        expected_norms, expected_taps, expected_indices = create_test_cir_files(
            folder_path, NUM_TAPS, NUM_CIRS, SIGMA_SCALING, SIGMA_MAX
        )

        compiled_cir_file.init(folder_path, NUM_SYMBOLS_PER_SLOT)

        try:
            taps_per_slot = NUM_TAPS * NUM_SYMBOLS_PER_SLOT

            # Read all slots once
            for _ in range(NUM_SLOTS):
                compiled_cir_file.read(NUM_SYMBOLS_PER_SLOT)

            # Now read again - should wrap around to slot 0
            norms, taps, tap_indices = compiled_cir_file.read(NUM_SYMBOLS_PER_SLOT)

            # This should be the same as slot 0
            slot0_expected_norms = expected_norms[:NUM_SYMBOLS_PER_SLOT]
            slot0_expected_taps = expected_taps[:taps_per_slot]
            slot0_expected_indices = expected_indices[:taps_per_slot]

            returned_norms = np.array(norms)
            returned_taps = np.array(taps)
            returned_indices = np.array(tap_indices)

            np.testing.assert_allclose(returned_norms, slot0_expected_norms, rtol=1e-5,
                err_msg="Wraparound: Norms don't match slot 0")
            np.testing.assert_allclose(returned_taps, slot0_expected_taps, rtol=1e-5,
                err_msg="Wraparound: Taps don't match slot 0")
            np.testing.assert_array_equal(returned_indices, slot0_expected_indices,
                err_msg="Wraparound: Tap indices don't match slot 0")

        finally:
            compiled_cir_file.shutdown()


def test_init_missing_file(compiled_cir_file):
    """Test that init fails gracefully with missing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Non-existent folder
        folder_path = os.path.join(tmpdir, "nonexistent_folder")
        result = compiled_cir_file.init(folder_path, NUM_SYMBOLS_PER_SLOT)
        assert result != 0, "Init should fail with missing folder/files"


def test_different_tap_counts(compiled_cir_file):
    """Test with different numbers of taps."""
    test_tap_counts = [1, 5, 20, 50]

    for num_taps in test_tap_counts:
        num_cirs = NUM_SYMBOLS_PER_SLOT * 2  # 2 slots

        with tempfile.TemporaryDirectory() as tmpdir:
            folder_path = os.path.join(tmpdir, f"test_cir_folder_{num_taps}")
            _, _, _ = create_test_cir_files(
                folder_path, num_taps, num_cirs, SIGMA_SCALING, SIGMA_MAX
            )

            compiled_cir_file.init(folder_path, NUM_SYMBOLS_PER_SLOT)

            try:
                assert compiled_cir_file.get_num_taps() == num_taps, \
                    f"get_num_taps should return {num_taps}"

                norms, taps, tap_indices = compiled_cir_file.read(NUM_SYMBOLS_PER_SLOT)
                expected_total = num_taps * NUM_SYMBOLS_PER_SLOT

                assert taps.shape == (expected_total, 2), \
                    f"Expected taps shape ({expected_total}, 2), got {taps.shape}"
                assert tap_indices.shape == (expected_total,), \
                    f"Expected tap_indices shape ({expected_total},), got {tap_indices.shape}"

                # Verify norms array has correct shape
                returned_norms = np.array(norms)
                assert returned_norms.shape == (NUM_SYMBOLS_PER_SLOT,), \
                    f"Expected norms shape ({NUM_SYMBOLS_PER_SLOT},), got {returned_norms.shape}"

            finally:
                compiled_cir_file.shutdown()

def test_pass_through_cir(compiled_cir_file):
    """Test loading pass-through CIR from the pre-generated folder if it exists."""
    # Path to the pass-through CIR folder (relative to the test file location)
    test_dir = Path(__file__).parent
    default_cir_path = test_dir.parent.parent.parent / "data" / "pass_through_cir"
    if not default_cir_path.exists():
        pytest.skip(f"Pass-through CIR folder not found at {default_cir_path}.")

    # Read config to get expected values
    config_path = default_cir_path / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    num_cirs = config["channel_emulation"]["num_cirs"]

    # Initialize the CIR file loader
    result = compiled_cir_file.init(str(default_cir_path), num_cirs)
    assert result == 0, f"Init should return 0 on success, got {result}"

    try:
        # Verify configuration
        assert compiled_cir_file.get_sigma_scaling() == 0.0
        assert compiled_cir_file.get_sigma_max() == 0.0
        assert compiled_cir_file.get_num_taps() == 1

        # Read CIR data for one slot
        norms, taps, tap_indices = compiled_cir_file.read(num_cirs)

        returned_norms = np.array(norms)
        returned_taps = np.array(taps)
        returned_indices = np.array(tap_indices)

        # For pass-through: norms should be 1.0
        expected_norms = np.ones(num_cirs, dtype=np.float32)
        np.testing.assert_allclose(returned_norms, expected_norms, rtol=1e-5,
            err_msg="Pass-through CIR norms should be 1.0")

        # For pass-through: taps should be I=1, Q=0
        expected_taps = np.zeros((num_cirs, 2), dtype=np.float32)
        expected_taps[:, 0] = 1.0
        np.testing.assert_allclose(returned_taps, expected_taps, rtol=1e-5,
            err_msg="Pass-through CIR tap coefficients should be I=1, Q=0")

        # For pass-through: tap indices should be 0
        expected_indices = np.zeros(num_cirs, dtype=np.uint16)
        np.testing.assert_array_equal(returned_indices, expected_indices,
            err_msg="Pass-through CIR tap indices should all be 0")

    finally:
        compiled_cir_file.shutdown()
