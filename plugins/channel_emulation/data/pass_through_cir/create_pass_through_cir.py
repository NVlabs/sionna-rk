#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Generate a default CIR for the channel emulator that corresponds to
a pass-through channel with no distortion.

The CIR consists of:
- Single tap with I=1, Q=0 (unity gain, no phase shift)
- Tap delay = 0 (no delay)
- Channel norm = 1.0 (unity norm)
- sigma_scaling = 0, sigma_max = 0 (no added noise)

This effectively passes the signal through unchanged: y[n] = x[n]

Output files:
- config.json: JSON configuration with channel_emulation section
- cirs.bin: Binary file with packed CIR entries
"""

import numpy as np
import json
import os
import argparse


def create_default_cir(output_folder: str,
                       num_symbols_per_slot: int = 14,
                       num_slots: int = 10) -> None:
    """
    Create a default pass-through CIR file set.

    The folder will contain:
      - config.json: Configuration with channel_emulation section containing
                     num_taps, num_cirs, sigma_scaling, sigma_max
      - cirs.bin: Binary file with packed CIR entries (norm + taps + tap_indices per entry)

    JSON format:
      {
        "channel_emulation": {
          "num_taps": N,
          "num_cirs": M,
          "sigma_scaling": F,
          "sigma_max": F
        }
      }

    Binary format for cirs.bin (packed entries, one per symbol):
      For each entry:
        - float32: norm (channel norm for noise std computation)
        - float32[num_taps * 2]: Interleaved real/imag CIR tap values
        - uint16_t[num_taps]: Tap delay indices

    Args:
        output_folder: Path to the output folder
        num_symbols_per_slot: Number of OFDM symbols per slot (default: 14 for 5G NR)
        num_slots: Number of slots to generate CIRs for (default: 10)
    """
    # Constants for pass-through channel
    NUM_TAPS = 1
    TAP_REAL = 1.0   # I component (unity gain)
    TAP_IMAG = 0.0   # Q component (no phase shift)
    TAP_INDEX = 0    # No delay
    NORM = 1.0       # Unity channel norm
    SIGMA_SCALING = 0.0  # No noise scaling
    SIGMA_MAX = 0.0      # No maximum noise

    num_cirs = num_symbols_per_slot * num_slots

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Write config.json with channel_emulation section
    config = {
        "channel_emulation": {
            "num_taps": NUM_TAPS,
            "num_cirs": num_cirs,
            "sigma_scaling": SIGMA_SCALING,
            "sigma_max": SIGMA_MAX
        }
    }
    config_path = os.path.join(output_folder, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Created: {config_path}")

    # Write cirs.bin with packed format
    # Each entry: [norm (float32)] [taps (float32 * num_taps * 2)] [tap_indices (uint16_t * num_taps)]
    cirs_path = os.path.join(output_folder, "cirs.bin")
    with open(cirs_path, 'wb') as f:
        for _ in range(num_cirs):
            # Write norm (float32)
            np.array([NORM], dtype=np.float32).tofile(f)

            # Write taps (float32 interleaved real/imag)
            taps = np.array([TAP_REAL, TAP_IMAG], dtype=np.float32)
            taps.tofile(f)

            # Write tap indices (uint16_t)
            tap_indices = np.array([TAP_INDEX], dtype=np.uint16)
            tap_indices.tofile(f)

            # Write padding bytes for 4-byte alignment
            raw_size = 4 + NUM_TAPS * 4 * 2 + NUM_TAPS * 2
            padded_size = (raw_size + 3) & ~3
            padding_bytes = padded_size - raw_size
            if padding_bytes > 0:
                f.write(b'\x00' * padding_bytes)

    print(f"Created: {cirs_path}")

    # Calculate and print entry size for verification
    entry_size = 4 + NUM_TAPS * 4 * 2 + NUM_TAPS * 2  # float32 + taps + uint16_t indices
    # Pad to 4-byte alignment
    entry_size = (entry_size + 3) & ~3
    total_size = entry_size * num_cirs

    print(f"\nDefault CIR created successfully in: {output_folder}")
    print(f"  - Number of taps: {NUM_TAPS}")
    print(f"  - Number of CIRs: {num_cirs}")
    print(f"  - Symbols per slot: {num_symbols_per_slot}")
    print(f"  - Number of slots: {num_slots}")
    print(f"  - Channel norm: {NORM}")
    print(f"  - Tap coefficient: {TAP_REAL} + {TAP_IMAG}j (pass-through)")
    print(f"  - Tap delay: {TAP_INDEX} samples")
    print(f"  - Sigma scaling: {SIGMA_SCALING} (no noise)")
    print(f"  - Sigma max: {SIGMA_MAX} (no noise)")
    print(f"  - Entry size: {entry_size} bytes")
    print(f"  - Total cirs.bin size: {total_size} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a default pass-through CIR for the channel emulator"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./",
        help="Output folder path (default: ./)"
    )
    parser.add_argument(
        "-s", "--symbols-per-slot",
        type=int,
        default=14,
        help="Number of OFDM symbols per slot (default: 14 for 5G NR)"
    )
    parser.add_argument(
        "-n", "--num-slots",
        type=int,
        default=1,
        help="Number of slots to generate CIRs for (default: 1)"
    )

    args = parser.parse_args()

    create_default_cir(
        output_folder=args.output,
        num_symbols_per_slot=args.symbols_per_slot,
        num_slots=args.num_slots
    )


if __name__ == "__main__":
    main()
