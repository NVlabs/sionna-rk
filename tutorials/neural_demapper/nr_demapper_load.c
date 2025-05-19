/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include "common/config/config_userapi.h"
#include "common/utils/LOG/log.h"
#include "common/utils/load_module_shlib.h"
#include "nr_demapper_extern.h"

// TODO: Q: can this be inside the loader function?
/* demapper_arg is used to initialize the config module so that the loader works as expected */
static char *demapper_arg[64]={"demappertest",NULL};

// START marker-plugin-load
static int32_t demapper_no_thread_init() {
    return 0;
}

int load_demapper_lib( char *version, demapper_interface_t *interface )
{
    char *ptr = (char*)config_get_if();
    char libname[64] = "demapper";

    if (ptr == NULL) {  // config module possibly not loaded
        uniqCfg = load_configmodule( 1, demapper_arg, CONFIG_ENABLECMDLINEONLY );
        logInit();
    }

    // function description array for the shlib loader
    loader_shlibfunc_t shlib_fdesc[] = { {.fname = "demapper_init" },
                                         {.fname = "demapper_init_thread", .fptr = &demapper_no_thread_init },
                                         {.fname = "demapper_shutdown" },
                                         {.fname = "demapper_compute_llr" }};

    int ret;
    ret = load_module_version_shlib( libname, version, shlib_fdesc, sizeofArray(shlib_fdesc), NULL );
    AssertFatal((ret >= 0), "Error loading demapper library");

    // assign loaded functions to the interface
    interface->init = (demapper_initfunc_t *)shlib_fdesc[0].fptr;
    interface->init_thread = (demapper_initfunc_t *)shlib_fdesc[1].fptr;
    interface->shutdown = (demapper_shutdownfunc_t *)shlib_fdesc[2].fptr;
    interface->compute_llr = (demapper_compute_llrfunc_t *)shlib_fdesc[3].fptr;

    AssertFatal( interface->init() == 0, "Error starting Demapper library %s %s\n", libname, version );

    return 0;
}

int free_demapper_lib( demapper_interface_t *demapper_interface )
{
    return demapper_interface->shutdown();
}
// END marker-plugin-load
