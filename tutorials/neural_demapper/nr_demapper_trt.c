/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#define _GNU_SOURCE
#include "openair1/PHY/TOOLS/tools_defs.h"
#include "PHY/sse_intrin.h"
#include "nr_demapper_orig.h"
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <cuda_runtime_api.h>
#ifdef __aarch64__
#define USE_128BIT
#endif

//#define PRINT_TIMES

#define TIMESTAMP_CLOCK_SOURCE CLOCK_MONOTONIC

void trt_demapper_configure(char const* weight_file, int normalized_inputs);

struct TRTContext;
struct TRTContext* trt_demapper_init(int make_stream);
void trt_demapper_shutdown();

void trt_demapper_decode(struct TRTContext* context, cudaStream_t stream, int16_t const* in_symbols, int16_t const* in_mags, size_t num_symbols,
                         int16_t* outputs, uint32_t num_bits_per_symbol);

// Plugin Init / Shutdown

static __thread struct TRTContext* context;

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

int32_t demapper_init( void )
{
    printf("Initializing TRT demapper (TID %d)\n", (int) gettid());
    FILE* cf = fopen("demapper_trt.config", "r");
    if (cf) {
        char weights_buf[1024] = { 0 };
        fgets(weights_buf, sizeof(weights_buf), cf);
        if (weights_buf[0]) // remove newline
            weights_buf[strlen(weights_buf)-1] = 0;
        int normalized_input = 0;
        fscanf(cf, "%d\n", &normalized_input);
        trt_demapper_configure(weights_buf, normalized_input);
        fclose(cf);
    }

    context = trt_demapper_init(1);
    if (context) {
        printf("Initialized TRT demapper\n");
        return 0;
    }
    else
        return -1;
}

int32_t demapper_init_thread( void ) {
    printf("Initializing TRT demapper context (TID %d)\n", (int) gettid());
    if (!context) {
        context = trt_demapper_init(1);
        fflush(stdout);
    }
    if (!context)
        return -1;
    return 0;
}

int32_t demapper_shutdown( void )
{
    trt_demapper_shutdown();
    return 0;
}

// QAM16

int demapper_qam16( int32_t *rxdataF_comp,
                    int32_t *ul_ch_mag,
                    int16_t *ulsch_llr,
                    uint32_t nb_re,
                    uint8_t symbol )
{
  struct timespec ts_begin;
  clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_begin );

  if (!context) { // check for new threads in thread pool
    context = trt_demapper_init(1);
    fflush(stdout);
  }

  if (nb_re > 0)
    trt_demapper_decode(context, 0, (int16_t const*) rxdataF_comp, (int16_t const*) ul_ch_mag, nb_re,
                        ulsch_llr, 4);

  struct timespec ts_end;
  clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_end );

#ifdef PRINT_TIMES
  unsigned long long time_ns = ts_end.tv_nsec - ts_begin.tv_nsec + 1000000000ll * (ts_end.tv_sec - ts_begin.tv_sec);
  if (add_measurement(&trt_time, time_ns) % 500 == 499) {
    time_ns = trt_time.total_ns / trt_time.count;
    printf("TRT demapper avg runtime: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
    printf("TRT demapper max runtime: %llu us %llu ns\n", trt_time.max_ns / 1000, trt_time.max_ns - trt_time.max_ns / 1000 * 1000);
    fflush(stdout);
    memset(&trt_time, 0, sizeof(trt_time));
  }
#endif

  return 1; // handled
}

// demapper handler

int demapper_compute_llr(int32_t *rxdataF_comp,
                         int32_t *ul_ch_mag,
                         int32_t *ul_ch_magb,
                         int32_t *ul_ch_magc,
                         int16_t *ulsch_llr,
                         uint32_t nb_re,
                         uint8_t  symbol,
                         uint8_t  mod_order) {
  struct timespec ts;
  clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts );

  switch(mod_order){
    case 2:
      return 0; // defer to default handling
      break;
    case 4:
      return demapper_qam16(rxdataF_comp,
                            ul_ch_mag,
                            ulsch_llr,
                            nb_re,
                            symbol);
      break;
/*
    case 6:
      nr_ulsch_64qam_llr(rxdataF_comp,
                         ul_ch_mag,
                         ul_ch_magb,
                         ulsch_llr,
                         nb_re,
                         symbol);
      break;
    case 8:
      nr_ulsch_256qam_llr(rxdataF_comp,
                          ul_ch_mag,
                          ul_ch_magb,
                          ul_ch_magc,
                          ulsch_llr,
                          nb_re,
                          symbol);
      break;
*/
    default:
      printf("Neural demapper not implemented for mod_order > 4\n");
      fflush(stdout);
      AssertFatal(1==0,"trt_demapper_compute_llr: invalid/unhandled Qm value, symbol = %d, Qm = %d\n",symbol, mod_order);
  }
  return 0;
}
