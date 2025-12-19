/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#ifndef _NR_DEMAPPER_EXTERN_H__
#define _NR_DEMAPPER_EXTERN_H__

// START marker-plugin-extern
#include "nr_demapper_defs.h"

typedef struct demapper_interface_s {
    demapper_initfunc_t        *init;
    demapper_initfunc_t        *init_thread;
    demapper_shutdownfunc_t    *shutdown;
    demapper_compute_llrfunc_t *compute_llr;
} demapper_interface_t;

// global access point for the plugin interface
extern demapper_interface_t demapper_interface;

int load_demapper_lib( char *version, demapper_interface_t * );
int free_demapper_lib( demapper_interface_t *demapper_interface );

demapper_initfunc_t        demapper_init;
demapper_shutdownfunc_t    demapper_shutdown;
demapper_compute_llrfunc_t demapper_compute_llr;
// END marker-plugin-extern

#endif // _NR_DEMAPPER_EXTERN_H__
