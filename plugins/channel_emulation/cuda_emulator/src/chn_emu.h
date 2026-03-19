/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#ifndef __CHN_EMU_H__
#define __CHN_EMU_H__


#include <stdint.h>

#include "openair1/PHY/defs_nr_common.h"
#include "openair1/PHY/defs_RU.h"


int32_t chn_emu_init(int num_taps_param,
                     int num_symbols_per_slot_param,
                     float sigma_scaling_param,
                     float sigma_max_param);

int32_t chn_emu_init_thread(void);

int32_t chn_emu_shutdown(void);

void chn_emu_compute(RU_t *ru,
                     int slot,
                     NR_DL_FRAME_PARMS *fp,
                     int samples_first_symbol,
                     int samples_other_symbols,
                     const char *direction,
                     int data_offset,
                     const void* cir);

#endif // __CHN_EMU_H__