/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
/**
 * @file cir_zmq.h
 * @brief Public API for the CIR ZMQ plugin.
 *
 * This plugin exposes a ZMQ REP socket (tcp:// *:5555) that accepts JSON
 * messages to configure and update channel impulse response (CIR) data
 * at runtime.  A control application connects with a ZMQ REQ socket and
 * can:
 *   - query the current configuration  (msg_type: "config_req")
 *   - push new CIR taps and noise      (msg_type: "cir")
 *   - enable/disable a custom receiver  (msg_type: "nrx")
 *
 * CIR data is stored internally as clean flat arrays:
 *   - norms[S]           (one float per OFDM symbol)
 *   - taps[S * 2*T]      (interleaved I/Q, row-major by symbol)
 *   - tap_indices[S * T]  (delay-bin indices, row-major by symbol)
 *
 * cir_zmq_read() packs these arrays into the binary layout expected
 * by the CUDA channel emulator (same format as cir_file_read).
 *
 * Thread-safety model:
 *   - cir_zmq_run / cir_zmq_receive are meant to execute on a single
 *     dedicated thread (the "ZMQ thread").
 *   - cir_zmq_read, cir_zmq_receiver_symbols_requested, and the getter
 *     functions may be called from any thread; shared state is protected
 *     by an internal mutex.
 */

#ifndef __CIR_ZMQ_H__
#define __CIR_ZMQ_H__

#include <stdint.h>
#include <pthread.h>
#include "cir_zmq_defs.h"

/**
 * @brief Initialise the CIR ZMQ plugin.
 *
 * Creates the ZMQ context and REP socket, and allocates internal CIR
 * storage for num_ofdm_symbols_per_slot symbols, each with num_taps taps.
 * All arrays are initialised to meaningful defaults:
 *   - sigma_scaling = 1.0, sigma_max = 1.0
 *   - norms[s] = 1.0 for all s
 *   - taps: first tap I=1.0, Q=0.0; remaining taps zero
 *   - tap_indices: [0, 1, 2, ..., num_taps-1] for each symbol
 *
 * Safe to call more than once (re-initialises arrays).
 *
 * @param num_taps_param               Number of channel taps.
 * @param num_ofdm_symbols_per_slot    Number of OFDM symbols per slot.
 * @param fft_size_param               FFT size used by the OFDM system.
 * @param subcarrier_spacing_param     Subcarrier spacing in Hz.
 * @param frequency_param              Carrier frequency in Hz.
 * @return 0 on success, -1 on failure.
 */
int32_t cir_zmq_init(int num_taps_param,
                     int num_ofdm_symbols_per_slot,
                     int fft_size_param,
                     float subcarrier_spacing_param,
                     float frequency_param);

/** @brief Per-thread initialisation (currently a no-op). */
int32_t cir_zmq_init_thread(void);

/**
 * @brief Shut down the CIR ZMQ plugin and free all resources.
 * @return 0 on success.
 */
int32_t cir_zmq_shutdown(void);

/**
 * @brief Receive and process one ZMQ message (blocking).
 *
 * Blocks until a message arrives on the REP socket, parses the JSON
 * payload, dispatches to the appropriate handler, and sends the reply.
 *
 * @return 0 on success, 1 on ZMQ receive error.
 */
int cir_zmq_receive(void);

/**
 * @brief Read the current CIR data as a packed binary blob (thread-safe).
 *
 * Packs the internal norms, taps, and tap_indices arrays into the
 * per-symbol binary layout expected by the CUDA channel emulator:
 *   [float norm | float taps[2*T] | uint16_t tap_indices[T] | padding]
 *
 * The returned pointer is valid until the next call to cir_zmq_read
 * or cir_zmq_shutdown.
 *
 * @return Pointer to packed CIR data, or NULL if not initialised.
 */
const void *cir_zmq_read(void);

/**
 * @brief Blocking event loop -- calls cir_zmq_receive in a loop.
 *
 * Designed to be the entry point for a dedicated ZMQ thread
 * (e.g. via pthread_create).  Exits when the REP socket is closed
 * by cir_zmq_shutdown.
 *
 * @param arg Unused (pass NULL).
 * @return Always returns NULL.
 */
void *cir_zmq_run(void *arg);

/**
 * @brief Check whether the custom (neural) receiver is enabled.
 *
 * @param arg Unused (pass NULL).
 * @return -1 if the custom receiver is enabled, 0 otherwise.
 */
int cir_zmq_receiver_symbols_requested(void *arg);

/* ---- Getter functions (mirror cir_file API) ---- */

/** @brief Get the configured number of taps. */
int cir_zmq_get_num_taps(void);

/** @brief Get the current sigma_scaling value (thread-safe). */
float cir_zmq_get_sigma_scaling(void);

/** @brief Get the current sigma_max value (thread-safe). */
float cir_zmq_get_sigma_max(void);

#endif /* __CIR_ZMQ_H__ */
