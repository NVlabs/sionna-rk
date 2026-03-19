/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#ifndef __CIR_FILE_H__
#define __CIR_FILE_H__

#include <stdint.h>
#include "cir_file_defs.h"

/**
 * Initialize the CIR file plugin by loading CIR data from a folder.
 *
 * Expects a folder containing two files:
 *   - config.json: JSON file containing config
 *   - cirs.bin: Binary file with packed CIR entries
 *
 * JSON format:
 *   {
 *     "channel_emulation": {
 *       "num_taps": N,
 *       "num_cirs": M,
 *       "sigma_scaling": F,
 *       "sigma_max": F
 *     }
 *   }
 *
 * cirs.bin format (packed entries, one per symbol):
 *   For each CIR entry:
 *     - float32: norm (channel norm ||h_s|| for noise std computation)
 *     - float32[num_taps * 2]: Interleaved real/imag CIR tap values
 *     - uint16_t[num_taps]: Tap delay indices
 *
 * @param folder_path Path to the folder containing the CIR files
 * @param num_symbols_per_slot Number of OFDM symbols per slot
 * @return 0 on success, -1 on failure
 */
int32_t cir_file_init(const char* folder_path, int num_symbols_per_slot);

/**
 * Thread-specific initialization (not needed for file-based CIR).
 * @return Always returns 0
 */
int32_t cir_file_init_thread(void);

/**
 * Shutdown the CIR file plugin and free allocated memory.
 * @return 0 on success
 */
int32_t cir_file_shutdown(void);

/**
 * Read CIR data for the next slot.
 *
 * Returns packed CIR data for the current slot and advances an internal counter.
 * The counter wraps around to 0 when reaching the end of the CIR bank.
 *
 * The returned pointer points to packed CIR entries for num_symbols_per_slot symbols.
 * Each entry contains:
 *   - float norm: Channel norm ||h_s||
 *   - float taps[num_taps * 2]: Complex taps as interleaved [Re, Im, ...]
 *   - uint16_t tap_indices[num_taps]: Tap delay indices
 *
 * @return Pointer to packed CIR data for the slot
 */
const void* cir_file_read(void);

/**
 * Get the number of taps per CIR.
 * @return Number of taps
 */
int cir_file_get_num_taps(void);

/**
 * Get the sigma_scaling parameter for noise std computation.
 * Noise std is computed as: min(sigma_scaling / norm, sigma_max)
 * @return sigma_scaling value
 */
float cir_file_get_sigma_scaling(void);

/**
 * Get the sigma_max parameter (maximum noise standard deviation).
 * @return sigma_max value
 */
float cir_file_get_sigma_max(void);

#endif
