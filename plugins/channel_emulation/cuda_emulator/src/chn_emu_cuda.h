/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#ifndef __CHN_EMU_CUDA_H__
#define __CHN_EMU_CUDA_H__

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_SAMPLES_PER_SLOT 200000
#define NOISE_POOL_SIZE 1000000
// Maximum delay of the channel impulse response (number of taps)
#define MAX_TAP_DELAY 256

int32_t chn_emu_cuda_init(int num_taps_param,
                          int num_symbols_per_slot_param,
                          float sigma_scaling_param,
                          float sigma_max_param);

int32_t chn_emu_cuda_init_thread(void);

int32_t chn_emu_cuda_shutdown(void);

void chn_emu_cuda_compute(void *data,
                          int samples_per_slot,
                          int samples_per_frame,
                          int samples_first_symbol,
                          int samples_other_symbols,
                          int data_offset,
                          const char *direction,
                          const void* cir);

void chn_emu_cuda_set_sigma_scaling(float val);
void chn_emu_cuda_set_sigma_max(float val);

#ifdef __cplusplus
}
#endif

#endif // __CHN_EMU_CUDA_H__
