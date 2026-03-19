/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#ifndef __CIR_ZMQ_DEFS_H__
#define __CIR_ZMQ_DEFS_H__

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

/**
 * @brief Function pointer types for the CIR ZMQ plugin API.
 *
 * cir_zmq_read_t returns a const void* pointing to packed CIR data
 * (same layout as cir_file_read), suitable for chn_emu_interface.compute().
 */

typedef int32_t(cir_zmq_initfunc_t)(int num_taps_param,
                                     int num_ofdm_symbols_per_slot,
                                     int fft_size_param,
                                     float subcarrier_spacing_param,
                                     float frequency_param);
typedef int32_t(cir_zmq_init_threadfunc_t)(void);
typedef int32_t(cir_zmq_shutdownfunc_t)(void);
typedef int(cir_zmq_receive_t)(void);
typedef int(cir_zmq_receiver_symbols_requested_t)(void*);
typedef const void *(cir_zmq_read_t)(void);
typedef void *(cir_zmq_run_t)(void *);

/** @brief Getter function pointer types (mirror cir_file API). */
typedef int(cir_zmq_get_num_taps_t)(void);
typedef float(cir_zmq_get_sigma_scaling_t)(void);
typedef float(cir_zmq_get_sigma_max_t)(void);

#endif
