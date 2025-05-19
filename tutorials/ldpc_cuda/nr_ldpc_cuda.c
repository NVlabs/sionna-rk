/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#define _GNU_SOURCE
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/nrLDPC_decoder/nrLDPC_types.h"
#include "PHY/CODING/nrLDPC_decoder/nrLDPCdecoder_defs.h"
#include "assertions.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#ifdef __aarch64__
#define USE_128BIT
#endif

//#define PRINT_TIMES

#define TIMESTAMP_CLOCK_SOURCE CLOCK_MONOTONIC

struct ThreadContext;
struct ThreadContext* ldpc_decoder_init(int make_stream);
void ldpc_decoder_shutdown();

uint32_t ldpc_decode(struct ThreadContext* context_, cudaStream_t stream, uint32_t BG, uint32_t Z,
                     int8_t const* llr_in, uint32_t block_length,
                     uint8_t* llr_bits, uint32_t num_iter,
                     uint32_t perform_syndrome_check);

// Plugin Init / Shutdown

static __thread struct ThreadContext* context;

#ifdef PRINT_TIMES
struct TimeMeasurements {
    unsigned long long total_ns;
    unsigned long long max_ns;
    unsigned count;
};
static __thread struct TimeMeasurements trt_time;

static unsigned add_measurement(struct TimeMeasurements* time, unsigned long long time_ns) {
    time->total_ns += time_ns;
    if (time_ns > time->max_ns)
        time->max_ns = time_ns;
    return time->count++;
}
#endif

int32_t LDPCinit( void )
{
    context = ldpc_decoder_init(1);
    if (context) {
        printf("Initialized LDPC decoder\n");
        fflush(stdout);
        return 0;
    }
    else
        return -1;
}

int32_t LDPCinitthread( void ) {
    if (!context) { // only run initialization for new threads
        context = ldpc_decoder_init(1);
        printf("Initialized new LDPC decoder context\n");
        fflush(stdout);
    }
    if (!context)
        return -1;
    return 0;
}

int32_t LDPCshutdown( void )
{
    printf("Shutting down LDPC decoder\n");
    fflush(stdout);
    ldpc_decoder_shutdown();
    return 0;
}

int32_t LDPCdecoder(t_nrLDPC_dec_params *p_decParams,
                    uint8_t harq_pid,
                    uint8_t ulsch_id,
                    uint8_t C,
                    int8_t *p_llr,
                    int8_t *p_out,
                    t_nrLDPC_time_stats *time_stats,
                    decode_abort_t *ab) {
    if (!context) // in case no explicit pre-initialization was performed for this thread
        context = ldpc_decoder_init(1);

    uint32_t num_iter = ldpc_decode(context, 0, p_decParams->BG, p_decParams->Z,
                                    p_llr, p_decParams->E,
                                    p_out, p_decParams->numMaxIter,
                                    !p_decParams->check_crc); // additional syndrome check required?

    // verify reconstructed output using custom CRC check, if given
    if (p_decParams->check_crc && num_iter <= p_decParams->numMaxIter
     && !p_decParams->check_crc((uint8_t*)p_out, p_decParams->E, p_decParams->crc_type)) {
      LOG_D(PHY, "set abort: CRC after %d\n", num_iter);
      set_abort(ab, true);
      return p_decParams->numMaxIter+1;
    }
    // otherwise communicate failed reconstructions detected using syndrome
    else if (!p_decParams->check_crc && num_iter > p_decParams->numMaxIter) {
      LOG_D(PHY, "set abort: syndrome after %d\n", num_iter);
      set_abort(ab, true);
      return p_decParams->numMaxIter+1;
    }

    return num_iter;
}
