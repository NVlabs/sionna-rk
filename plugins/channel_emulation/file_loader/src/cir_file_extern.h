/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#ifndef _CIR_FILE_EXTERN_H__
#define _CIR_FILE_EXTERN_H__

#include "cir_file_defs.h"

/**
 * Interface structure for the CIR file plugin.
 * Provides function pointers for all plugin operations.
 */
typedef struct cir_file_interface_s {
    cir_file_initfunc_t              *init;
    cir_file_init_threadfunc_t       *init_thread;
    cir_file_shutdownfunc_t          *shutdown;
    cir_file_read_t                  *read;
    cir_file_get_num_taps_t          *get_num_taps;
    cir_file_get_sigma_scaling_t     *get_sigma_scaling;
    cir_file_get_sigma_max_t         *get_sigma_max;
} cir_file_interface_t;

extern cir_file_interface_t cir_file_interface;

/**
 * Load the CIR file library.
 * @param version Version string (can be NULL)
 * @param interface Pointer to interface structure to populate
 * @return 0 on success, negative on failure
 */
int load_cir_file_lib(char *version, cir_file_interface_t *interface);

/**
 * Free the CIR file library resources.
 * @param cir_file_interface Pointer to interface structure
 * @return 0 on success
 */
int free_cir_file_lib(cir_file_interface_t *cir_file_interface);

/* Function declarations */
cir_file_initfunc_t              cir_file_init;
cir_file_init_threadfunc_t       cir_file_init_thread;
cir_file_shutdownfunc_t          cir_file_shutdown;
cir_file_read_t                  cir_file_read;
cir_file_get_num_taps_t          cir_file_get_num_taps;
cir_file_get_sigma_scaling_t     cir_file_get_sigma_scaling;
cir_file_get_sigma_max_t         cir_file_get_sigma_max;

#endif
