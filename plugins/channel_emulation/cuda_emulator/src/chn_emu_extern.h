/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#ifndef _CHN_EMU_EXTERN_H__
#define _CHN_EMU_EXTERN_H__

// START marker-plugin-extern
#include "chn_emu_defs.h"

typedef struct chn_emu_interface_s {
    chn_emu_initfunc_t              *init;
    chn_emu_init_threadfunc_t       *init_thread;
    chn_emu_shutdownfunc_t          *shutdown;
    chn_emu_compute_t               *compute;
    chn_emu_set_sigma_scaling_t     *set_sigma_scaling;
    chn_emu_set_sigma_max_t         *set_sigma_max;
} chn_emu_interface_t;

// global access point for the plugin interface
extern chn_emu_interface_t chn_emu_interface;

int load_chn_emu_lib( char *version, chn_emu_interface_t * );
int free_chn_emu_lib( chn_emu_interface_t *chn_emu_interface );

chn_emu_initfunc_t              chn_emu_init;
chn_emu_init_threadfunc_t       chn_emu_init_thread;
chn_emu_shutdownfunc_t          chn_emu_shutdown;
chn_emu_compute_t               chn_emu_compute;
chn_emu_set_sigma_scaling_t     chn_emu_set_sigma_scaling;
chn_emu_set_sigma_max_t         chn_emu_set_sigma_max;
// END marker-plugin-extern

#endif // _CHN_EMU_EXTERN_H__
