/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/

#include "channel_emulator.h"
#include "common/config/config_userapi.h"
#include "common/utils/utils.h"

// Help strings for channel emulation CLI options (used by config paramdef_t)
#define CHNEMU_HLP_CIRFOLDER  "Path to CIR folder for channel emulation (enables channel emulation when set)\n"
#define CHNEMU_HLP_CIRZMQTAPS "Number of CIR taps for ZMQ-based channel emulation (enables cir_zmq when set)\n"

// Channel emulation parameters read from command line via config module
static char *cir_folder_path_param = NULL;
static int cir_zmq_num_taps_param = 0;

// Flag to track if channel emulation is enabled
static int channel_emulation_enabled = 0;

// Flag to track which CIR source is active (for shutdown)
static int cir_zmq_active = 0;

// Abstracted CIR source function pointers (set during init)
static const void *(*cir_read_fn)(void) = NULL;
static float (*cir_get_sigma_scaling_fn)(void) = NULL;
static float (*cir_get_sigma_max_fn)(void) = NULL;

/**
 * @brief Try to load and init channel emulation with the cir_file source.
 */
 static int init_channel_emu_cir_file(const char *cir_folder_path, int num_symbols_per_slot)
 {
     int ret = load_cir_file_lib(NULL, &cir_file_interface);
     if (ret != 0 || cir_file_interface.init == NULL) return -1;
 
     ret = load_chn_emu_lib(NULL, &chn_emu_interface);
     if (ret != 0 || chn_emu_interface.init == NULL) return -1;
 
     ret = cir_file_interface.init(cir_folder_path, num_symbols_per_slot);
     if (ret != 0) return -1;
 
     int num_taps = cir_file_interface.get_num_taps();
     float sigma_scaling = cir_file_interface.get_sigma_scaling();
     float sigma_max = cir_file_interface.get_sigma_max();
     ret = chn_emu_interface.init(num_taps, num_symbols_per_slot, sigma_scaling, sigma_max);
     if (ret != 0) return -1;
 
     // Wire up abstracted CIR source
     cir_read_fn = cir_file_interface.read;
     cir_get_sigma_scaling_fn = cir_file_interface.get_sigma_scaling;
     cir_get_sigma_max_fn = cir_file_interface.get_sigma_max;
     cir_zmq_active = 0;
 
     printf("Channel emulation enabled: CIR source=cir_file, folder=%s, num_taps=%d\n",
            cir_folder_path, num_taps);
     return 0;
 }
 
 /**
  * @brief Try to load and init channel emulation with the cir_zmq source.
  */
 static int init_channel_emu_cir_zmq(int num_taps, const NR_DL_FRAME_PARMS *fp)
 {
     int ret = load_cir_zmq_lib(NULL, &cir_zmq_interface);
     if (ret != 0 || cir_zmq_interface.init == NULL) return -1;
 
     ret = load_chn_emu_lib(NULL, &chn_emu_interface);
     if (ret != 0 || chn_emu_interface.init == NULL) return -1;
 
     int num_symbols_per_slot = fp->symbols_per_slot;
     int fft_size = fp->ofdm_symbol_size;
     float subcarrier_spacing = (float)fp->subcarrier_spacing;
     float frequency = (float)fp->dl_CarrierFreq;
 
     ret = cir_zmq_interface.init(num_taps, num_symbols_per_slot,
                                   fft_size, subcarrier_spacing, frequency);
     if (ret != 0) return -1;
 
     // Init channel emulator with default sigma values (cir_zmq defaults: 1.0, 1.0)
     float sigma_scaling = cir_zmq_interface.get_sigma_scaling();
     float sigma_max = cir_zmq_interface.get_sigma_max();
     ret = chn_emu_interface.init(num_taps, num_symbols_per_slot, sigma_scaling, sigma_max);
     if (ret != 0) return -1;
 
     // Wire up abstracted CIR source
     cir_read_fn = cir_zmq_interface.read;
     cir_get_sigma_scaling_fn = cir_zmq_interface.get_sigma_scaling;
     cir_get_sigma_max_fn = cir_zmq_interface.get_sigma_max;
     cir_zmq_active = 1;
 
     printf("Channel emulation enabled: CIR source=cir_zmq, num_taps=%d, "
            "symbols_per_slot=%d, fft_size=%d, scs=%.0f, freq=%.0f\n",
            num_taps, num_symbols_per_slot, fft_size, subcarrier_spacing, frequency);
     return 0;
 }

int is_channel_emulation_enabled(void)
{
    return channel_emulation_enabled;
}

void init_channel_emulator_libs(const NR_DL_FRAME_PARMS *fp)
{
    configmodule_interface_t *cfg = config_get_if();
    if (cfg != NULL) {
        paramdef_t channel_emulator_params[] = {
            {"cir-folder",       CHNEMU_HLP_CIRFOLDER,  0, .strptr = &cir_folder_path_param, .defstrval = NULL, TYPE_STRING, 0},
            {"cir-zmq-num-taps", CHNEMU_HLP_CIRZMQTAPS, 0, .iptr = &cir_zmq_num_taps_param,  .defintval = 0,   TYPE_INT,    0},
        };
        config_get(cfg, channel_emulator_params, sizeofArray(channel_emulator_params), NULL);
    }

    const char *cir_folder_path = cir_folder_path_param;
    int cir_zmq_num_taps = cir_zmq_num_taps_param;

    printf("Channel emulation enabled: CIR source=cir_file, folder=%s, num_taps=%d\n",
           cir_folder_path, cir_zmq_num_taps);

    // Channel emulation plugins - two mutually exclusive sources
    if (cir_zmq_num_taps > 0 && fp != NULL) {
        // ZMQ-based CIR source (takes priority over cir_folder)
        if (init_channel_emu_cir_zmq(cir_zmq_num_taps, fp) == 0) {
            channel_emulation_enabled = 1;
        } else {
            printf("Channel emulation plugins not loaded (cir_zmq init failed)\n");
        }
    } else if (cir_folder_path != NULL && strlen(cir_folder_path) > 0) {
        // File-based CIR source
        int num_symbols_per_slot = fp ? fp->symbols_per_slot : 14;
        if (init_channel_emu_cir_file(cir_folder_path, num_symbols_per_slot) == 0) {
            channel_emulation_enabled = 1;
        } else {
            printf("Channel emulation plugins not loaded (cir_file init failed)\n");
        }
    } else {
        printf("Channel emulation disabled (--cir-folder and --cir-zmq-num-taps not set)\n");
    }
}

void free_channel_emulator_libs()
{
    if (channel_emulation_enabled) {
        if (chn_emu_interface.shutdown)
            free_chn_emu_lib(&chn_emu_interface);
        if (cir_zmq_active) {
            if (cir_zmq_interface.shutdown)
                cir_zmq_interface.shutdown();
        } else {
            if (cir_file_interface.shutdown)
                free_cir_file_lib(&cir_file_interface);
        }
        cir_read_fn = NULL;
        cir_get_sigma_scaling_fn = NULL;
        cir_get_sigma_max_fn = NULL;
        channel_emulation_enabled = 0;
        cir_zmq_active = 0;
    }
}

void init_channel_emulator_worker_thread(void)
{
    if (chn_emu_interface.init_thread)
        chn_emu_interface.init_thread();

    if (cir_zmq_interface.init_thread)
        cir_zmq_interface.init_thread();

    if (cir_file_interface.init_thread)
        cir_file_interface.init_thread();
}

const void *channel_emulator_cir_read_and_apply(void)
{
    if (!cir_read_fn) return NULL;

    // Read CIR data from the active source
    const void *cir_data = cir_read_fn();

    // Update channel emulator sigma values from the CIR source
    if (cir_get_sigma_scaling_fn && chn_emu_interface.set_sigma_scaling)
        chn_emu_interface.set_sigma_scaling(cir_get_sigma_scaling_fn());
    if (cir_get_sigma_max_fn && chn_emu_interface.set_sigma_max)
        chn_emu_interface.set_sigma_max(cir_get_sigma_max_fn());

    return cir_data;
}
