/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#ifndef __CIR_ZMQ_EXTERN_H__
#define __CIR_ZMQ_EXTERN_H__

#include "cir_zmq_defs.h"

typedef struct cir_zmq_interface_s {
    cir_zmq_initfunc_t        *init;
    cir_zmq_init_threadfunc_t *init_thread;
    cir_zmq_shutdownfunc_t    *shutdown;
    cir_zmq_receive_t         *receive;
    cir_zmq_read_t            *read;
    cir_zmq_run_t             *run;
    cir_zmq_receiver_symbols_requested_t *receiver_symbols_requested;
    cir_zmq_get_num_taps_t       *get_num_taps;
    cir_zmq_get_sigma_scaling_t  *get_sigma_scaling;
    cir_zmq_get_sigma_max_t      *get_sigma_max;
} cir_zmq_interface_t;

extern cir_zmq_interface_t cir_zmq_interface;

int load_cir_zmq_lib( char *version, cir_zmq_interface_t * );
int free_cir_zmq_lib( cir_zmq_interface_t *cir_zmq_interface );

cir_zmq_initfunc_t        cir_zmq_init;
cir_zmq_init_threadfunc_t cir_zmq_init_thread;
cir_zmq_shutdownfunc_t    cir_zmq_shutdown;
cir_zmq_receive_t         cir_zmq_receive;
cir_zmq_read_t            cir_zmq_read;
cir_zmq_run_t             cir_zmq_run;
cir_zmq_receiver_symbols_requested_t cir_zmq_receiver_symbols_requested;
cir_zmq_get_num_taps_t       cir_zmq_get_num_taps;
cir_zmq_get_sigma_scaling_t  cir_zmq_get_sigma_scaling;
cir_zmq_get_sigma_max_t      cir_zmq_get_sigma_max;

#endif
