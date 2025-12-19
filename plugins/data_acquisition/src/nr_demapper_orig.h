/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#ifndef __NR_DEMAPPER_ORIG_H__
#define __NR_DEMAPPER_ORIG_H__

#include <stdint.h>

// The function definitions must match the names in the plugin load and the function signatures in openair1/PHY/NR_TRANSPORT/nr_demapper_defs.h

// Plugin API functions

int32_t demapper_init( void );

int32_t demapper_shutdown( void );

int demapper_compute_llr( int32_t *rxdataF_comp,
                          c16_t *ul_ch_mag,
                          c16_t *ul_ch_magb,
                          c16_t *ul_ch_magc,
                          int16_t *ulsch_llr,
                          uint32_t nb_re,
                          uint8_t  symbol,
                          uint8_t  mod_order );

// Internal functions

#endif // __NR_DEMAPPER_ORIG_H__

