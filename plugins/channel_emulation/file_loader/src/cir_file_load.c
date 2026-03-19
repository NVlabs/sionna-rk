/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include "common/config/config_userapi.h"
#include "common/utils/LOG/log.h"
#include "common/utils/load_module_shlib.h"
#include "cir_file_extern.h"

/* cir_file_arg is used to initialize the config module so that the loader works as expected */
static char *cir_file_arg[64] = {"cir_file_test", NULL};

int load_cir_file_lib(char *version, cir_file_interface_t *interface)
{
    char *ptr = (char*)config_get_if();
    char libname[64] = "cir_file";

    if (ptr == NULL) {  // config module possibly not loaded
        uniqCfg = load_configmodule(1, cir_file_arg, CONFIG_ENABLECMDLINEONLY);
        logInit();
    }

    // Function description array for the shlib loader
    loader_shlibfunc_t shlib_fdesc[] = {
        {.fname = "cir_file_init"},
        {.fname = "cir_file_init_thread"},
        {.fname = "cir_file_shutdown"},
        {.fname = "cir_file_read"},
        {.fname = "cir_file_get_num_taps"},
        {.fname = "cir_file_get_sigma_scaling"},
        {.fname = "cir_file_get_sigma_max"}
    };

    int ret;
    ret = load_module_version_shlib(libname, version, shlib_fdesc, sizeofArray(shlib_fdesc), NULL);
    // Gracefully handle missing library - channel emulation is optional
    if (ret < 0) {
        return ret;
    }

    // Assign loaded functions to the interface
    interface->init = (cir_file_initfunc_t *)shlib_fdesc[0].fptr;
    interface->init_thread = (cir_file_init_threadfunc_t *)shlib_fdesc[1].fptr;
    interface->shutdown = (cir_file_shutdownfunc_t *)shlib_fdesc[2].fptr;
    interface->read = (cir_file_read_t *)shlib_fdesc[3].fptr;
    interface->get_num_taps = (cir_file_get_num_taps_t *)shlib_fdesc[4].fptr;
    interface->get_sigma_scaling = (cir_file_get_sigma_scaling_t *)shlib_fdesc[5].fptr;
    interface->get_sigma_max = (cir_file_get_sigma_max_t *)shlib_fdesc[6].fptr;
    // Note: init() should be called by the user with appropriate parameters

    return 0;
}

int free_cir_file_lib(cir_file_interface_t *cir_file_interface)
{
    return cir_file_interface->shutdown();
}
