/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include "common/config/config_userapi.h"
#include "common/utils/LOG/log.h"
#include "common/utils/load_module_shlib.h"
#include "chn_emu_extern.h"


// TODO: Q: can this be inside the loader function?
/* demapper_arg is used to initialize the config module so that the loader works as expected */
static char *chn_emu_arg[64]={"chn_emu_test",NULL};

// START marker-plugin-load
int load_chn_emu_lib( char *version, chn_emu_interface_t *interface )
{
    char *ptr = (char*)config_get_if();
    char libname[64] = "chn_emu";

    if (ptr == NULL) {  // config module possibly not loaded
        uniqCfg = load_configmodule( 1, chn_emu_arg, CONFIG_ENABLECMDLINEONLY );
        logInit();
    }

    // function description array for the shlib loader
    loader_shlibfunc_t shlib_fdesc[] = { {.fname = "chn_emu_init" },
                                         {.fname = "chn_emu_init_thread" },
                                         {.fname = "chn_emu_shutdown" },
                                         {.fname = "chn_emu_compute" },
                                         {.fname = "chn_emu_set_sigma_scaling" },
                                         {.fname = "chn_emu_set_sigma_max" }};

    int ret;
    ret = load_module_version_shlib( libname, version, shlib_fdesc, sizeofArray(shlib_fdesc), NULL );
    // Gracefully handle missing library - channel emulation is optional
    if (ret < 0) {
        return ret;
    }

    // assign loaded functions to the interface
    interface->init = (chn_emu_initfunc_t *)shlib_fdesc[0].fptr;
    interface->init_thread = (chn_emu_init_threadfunc_t *)shlib_fdesc[1].fptr;
    interface->shutdown = (chn_emu_shutdownfunc_t *)shlib_fdesc[2].fptr;
    interface->compute = (chn_emu_compute_t *)shlib_fdesc[3].fptr;
    interface->set_sigma_scaling = (chn_emu_set_sigma_scaling_t *)shlib_fdesc[4].fptr;
    interface->set_sigma_max = (chn_emu_set_sigma_max_t *)shlib_fdesc[5].fptr;

    // Note: init() should be called by the user with appropriate parameters

    return 0;
}

int free_chn_emu_lib( chn_emu_interface_t *chn_emu_interface )
{
    return chn_emu_interface->shutdown();
}
// END marker-plugin-load
