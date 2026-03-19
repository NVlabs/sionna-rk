/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
// START marker-plugins

#include "plugins/neural_demapper/src/nr_demapper_extern.h"
#include "plugins/neural_receiver/src/nr_receiver_extern.h"
#include "plugins/channel_emulation/common/channel_emulator.h"

/**
 * @brief Initialise all plugins.
 *
 * Channel emulation is enabled by one of two mutually exclusive CLI options
 * (read via the config module in init_channel_emulator_libs):
 *   - --cir-folder <path>     -> use cir_file (file-based CIR)
 *   - --cir-zmq-num-taps <n>   -> use cir_zmq (ZMQ-based CIR)
 *
 * @param fp Pointer to frame parameters (used to derive OFDM params for channel emulation).
 */
void init_plugins(const NR_DL_FRAME_PARMS *fp);
void free_plugins(void);

void worker_thread_plugin_init(void);

// Check if channel emulation is loaded and initialized
int is_channel_emulation_enabled(void);

/**
 * @brief Read CIR data from the active source and update channel emulator sigma.
 *
 * Calls the active CIR source's read(), then pushes the current
 * sigma_scaling / sigma_max values to the channel emulator.
 *
 * @return Pointer to packed CIR data, or NULL if channel emulation is disabled.
 */
const void *plugins_cir_read_and_apply(void);

// END marker-plugins
