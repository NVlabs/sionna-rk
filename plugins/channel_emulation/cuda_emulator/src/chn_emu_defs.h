/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#ifndef __CHN_EMU_DEFS_H__
#define __CHN_EMU_DEFS_H__

#include <stdint.h>

#include "openair1/PHY/defs_nr_common.h"
#include "openair1/PHY/defs_RU.h"

// START marker-plugin-defs
typedef int32_t(chn_emu_initfunc_t)(int, int, float, float);
typedef int32_t(chn_emu_init_threadfunc_t)(void);
typedef int32_t(chn_emu_shutdownfunc_t)(void);
typedef void(chn_emu_compute_t)(RU_t*, int, NR_DL_FRAME_PARMS*, int, int, const char*, int, const void*);
typedef void(chn_emu_set_sigma_scaling_t)(float);
typedef void(chn_emu_set_sigma_max_t)(float);
// END marker-plugin-defs

#endif // __CHN_EMU_DEFS_H__