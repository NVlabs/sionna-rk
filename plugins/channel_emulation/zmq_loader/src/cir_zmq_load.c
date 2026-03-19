/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include "common/config/config_userapi.h"
#include "common/utils/LOG/log.h"
#include "common/utils/load_module_shlib.h"
#include "cir_zmq_extern.h"

// START marker-plugin-load
int load_cir_zmq_lib( char *version, cir_zmq_interface_t *interface )
{
    char *ptr = (char*)config_get_if();
    char libname[64] = "cir_zmq";

    if (ptr == NULL) {  // config module possibly not loaded
        // Fake argv to bootstrap the OAI config module
        char *cir_zmq_argv[2] = {"cir_zmq", NULL};
        uniqCfg = load_configmodule( 1, cir_zmq_argv, CONFIG_ENABLECMDLINEONLY );
        logInit();
    }

    // function description array for the shlib loader
    loader_shlibfunc_t shlib_fdesc[] = { {.fname = "cir_zmq_init" },
                                         {.fname = "cir_zmq_init_thread" },
                                         {.fname = "cir_zmq_shutdown" },
                                         {.fname = "cir_zmq_receive" },
                                         {.fname = "cir_zmq_read" },
                                         {.fname = "cir_zmq_run" },
                                         {.fname = "cir_zmq_receiver_symbols_requested" },
                                         {.fname = "cir_zmq_get_num_taps" },
                                         {.fname = "cir_zmq_get_sigma_scaling" },
                                         {.fname = "cir_zmq_get_sigma_max" } };

    int ret;
    ret = load_module_version_shlib( libname, version, shlib_fdesc, sizeofArray(shlib_fdesc), NULL );
    AssertFatal((ret >= 0), "Error loading cir_zmq library");

    // assign loaded functions to the interface
    interface->init = (cir_zmq_initfunc_t *)shlib_fdesc[0].fptr;
    interface->init_thread = (cir_zmq_init_threadfunc_t *)shlib_fdesc[1].fptr;
    interface->shutdown = (cir_zmq_shutdownfunc_t *)shlib_fdesc[2].fptr;
    interface->receive = (cir_zmq_receive_t *)shlib_fdesc[3].fptr;
    interface->read = (cir_zmq_read_t *)shlib_fdesc[4].fptr;
    interface->run = (cir_zmq_run_t *)shlib_fdesc[5].fptr;
    interface->receiver_symbols_requested = (cir_zmq_receiver_symbols_requested_t *)shlib_fdesc[6].fptr;
    interface->get_num_taps = (cir_zmq_get_num_taps_t *)shlib_fdesc[7].fptr;
    interface->get_sigma_scaling = (cir_zmq_get_sigma_scaling_t *)shlib_fdesc[8].fptr;
    interface->get_sigma_max = (cir_zmq_get_sigma_max_t *)shlib_fdesc[9].fptr;

    // Note: init() should be called by the user with appropriate parameters

    return 0;
}

int free_cir_zmq_lib( cir_zmq_interface_t *cir_zmq_interface )
{
    return cir_zmq_interface->shutdown();
}
