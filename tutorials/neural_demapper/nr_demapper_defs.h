/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#ifndef __NR_DEMAPPER_DEFS_H__
#define __NR_DEMAPPER_DEFS_H__

#include <stdint.h>

// START marker-plugin-defs
typedef int32_t(demapper_initfunc_t)(void);
typedef int32_t(demapper_shutdownfunc_t)(void);

typedef int(demapper_compute_llrfunc_t)( int32_t *rxdataF_comp,
                          int32_t *ul_ch_mag,
                          int32_t *ul_ch_magb,
                          int32_t *ul_ch_magc,
                          int16_t *ulsch_llr,
                          uint32_t nb_re,
                          uint8_t  symbol,
                          uint8_t  mod_order );
// END marker-plugin-defs

#endif // _NR_DEMAPPER_DEFS_H__
