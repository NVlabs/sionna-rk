/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include "openair1/PHY/TOOLS/tools_defs.h"
#include "openair1/PHY/sse_intrin.h"
#include "nr_demapper_orig.h"

#ifdef __aarch64__
#define USE_128BIT
#endif

// Plugin Init / Shutdown

// START marker-plugin-orig
int32_t demapper_init( void )
{
    printf("Original/pass-through demapping initialized\n");
    // do nothing
    return 0;
}

int32_t demapper_shutdown( void )
{
    // do nothing
    return 0;
}

// No custom code, trigger default handling by returning 0 for unhandled

int demapper_compute_llr(int32_t *rxdataF_comp,
                         int32_t *ul_ch_mag,
                         int32_t *ul_ch_magb,
                         int32_t *ul_ch_magc,
                         int16_t *ulsch_llr,
                         uint32_t nb_re,
                         uint8_t  symbol,
                         uint8_t  mod_order) {
    // do nothing
    return 0;
}
// END marker-plugin-orig
