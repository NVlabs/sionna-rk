/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include<string.h>
#include<stdio.h>
#include<math.h>


#include "chn_emu.h"
#include "chn_emu_cuda.h"

#include "openair1/PHY/defs_nr_common.h"
#include "openair1/PHY/defs_RU.h"

#ifdef __aarch64__
#define USE_128BIT
#endif


// START marker-plugin
int32_t chn_emu_init(int num_taps_param,
                     int num_symbols_per_slot_param,
                     float sigma_scaling_param,
                     float sigma_max_param)
{
  chn_emu_cuda_init(num_taps_param, num_symbols_per_slot_param, sigma_scaling_param, sigma_max_param);

  printf("\nChannel Emulator initialized\n");
  printf("  num_taps_param        = %d\n", num_taps_param);
  printf("  num_symbols_per_slot_param = %d\n", num_symbols_per_slot_param);
  printf("  sigma_scaling_param        = %f\n", sigma_scaling_param);
  printf("  sigma_max_param            = %f\n", sigma_max_param);
  printf("\n");

  return 0;
}

int32_t chn_emu_init_thread(void)
{
    chn_emu_cuda_init_thread();
    return 0;
}

int32_t chn_emu_shutdown(void)
{
    chn_emu_cuda_shutdown();
    return 0;
}

void chn_emu_compute(RU_t *ru,
                     int slot,
                     NR_DL_FRAME_PARMS *fp,
                     int samples_first_symbol,
                     int samples_other_symbols,
                     const char *direction,
                     int data_offset,
                     const void* cir)
{
    c16_t *data = NULL;
    if (strcmp(direction, "rx") == 0) {
        data = (c16_t *)(ru->common.rxdata[0]);
    } else {
        data = (c16_t *)(ru->common.txdata[0]);
    }

    chn_emu_cuda_compute(data,
                         fp->get_samples_per_slot(slot, fp),
                         fp->samples_per_frame,
                         samples_first_symbol,
                         samples_other_symbols,
                         data_offset,
                         direction,
                         cir);
}

void chn_emu_set_sigma_scaling(float val)
{
    chn_emu_cuda_set_sigma_scaling(val);
}

void chn_emu_set_sigma_max(float val)
{
    chn_emu_cuda_set_sigma_max(val);
}
