/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#ifndef __CIR_FILE_DEFS_H__
#define __CIR_FILE_DEFS_H__

#include <stdint.h>
#include <stddef.h>

/**
 * Function pointer types for the cir_file interface.
 */
typedef int32_t(cir_file_initfunc_t)(const char* folder_path, int num_symbols_per_slot);
typedef int32_t(cir_file_init_threadfunc_t)(void);
typedef int32_t(cir_file_shutdownfunc_t)(void);
typedef const void*(cir_file_read_t)(void);
typedef int(cir_file_get_num_taps_t)(void);
typedef float(cir_file_get_sigma_scaling_t)(void);
typedef float(cir_file_get_sigma_max_t)(void);

#endif
