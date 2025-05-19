/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include "openair1/PHY/TOOLS/tools_defs.h"
#include "openair1/PHY/sse_intrin.h"
#include "nr_demapper_capture.h"
#include <pthread.h>
#include <stdio.h>
#include <time.h>

#ifdef __aarch64__
#define USE_128BIT
#endif

// START marker-capture-full
static char *filename_in = "demapper_in.txt";
static char *filename_out = "demapper_out.txt";
static FILE *f_in;
static FILE *f_out;
static pthread_mutex_t capture_lock = PTHREAD_MUTEX_INITIALIZER;

// import default implementations
void nr_ulsch_qpsk_llr(int32_t *rxdataF_comp,
                      int16_t  *ulsch_llr,
                      uint32_t nb_re,
                      uint8_t  symbol);
void nr_ulsch_16qam_llr(int32_t *rxdataF_comp, int32_t *ul_ch_mag, int16_t *ulsch_llr, uint32_t nb_re, uint8_t symbol);

/* configure which clock source to use, will affect resolution */
#define TIMESTAMP_CLOCK_SOURCE CLOCK_MONOTONIC

/* time processing functions */
void fprint_time( FILE* file, struct timespec *ts )
{
  fprintf( file, "%jd.%09ld\n", ts->tv_sec, ts->tv_nsec );
  return;
}

// START marker-capture-input-format
/*
 * Input file format (f_in):
 * time source resolution (sec.nanosec) - only first line
 * timestamp (sec.nanosec)
 * modulation scheme (string: QPSK, QAM16)
 * nb_re (int32)
 * real imag (for QPSK: int16 <space> int16)
 * real imag ch_mag.r ch_mag.i (for QAM16: int16 <space> int16 <space> int16 <space> int16)̦̦̦̦̦̦
 * ... (done nb_re times)
 */
// END marker-capture-input-format

// START marker-capture-output-format
/*
 * Output file format (f_out):
 * time source resolution (sec.nanosec) - only first line
 * timestamp (sec.nanosec)
 * modulation scheme (string: QPSK, QAM16)
 * nb_re (int32)
 * real imag llr.r llr.i (int16 <space> int16 <space> int16 <space> int16 )
 * --->number of elements in each row depend on the modulation scheme: QPSK: 2 ; QAM16: 4
 * ... (done nb_re times)
 */
// END marker-capture-output-format


void capture_qpsk(
            int32_t *rxdataF_comp,
            int16_t *ulsch_llr,
            uint32_t nb_re,
            struct timespec *ts )
{
    pthread_mutex_lock( &capture_lock );

    c16_t *rxF = (c16_t *)rxdataF_comp;

    fprint_time( f_in, ts );
    fprintf( f_in, "QPSK\n" );
    fprintf( f_in, "%d\n", nb_re );
    for(int i = 0; i < nb_re; i++ )
      fprintf( f_in, "%hd %hd\n", rxF[i].r, rxF[i].i );
    fflush( f_in );

    fprint_time( f_out, ts );
    fprintf( f_out, "QPSK\n" );
    fprintf( f_out, "%d\n", nb_re );
    for(int i = 0; i < nb_re; i++ )
      fprintf( f_out, "%hd %hd\n", ulsch_llr[2*i+0], ulsch_llr[2*i+1] );
    fflush( f_out );

    pthread_mutex_unlock( &capture_lock );
}

/*
 * from the reference implementation of test_llr.cpp
 * in: rxdataF_comp[nb_re] , each element is a complex number with .r and .i components. Each component is int16_t
 * in: ul_ch_mag[nb_re] , casted then to int16_t it becomes an array of [2*nb_re]. first component is the mag_real, second is mag_imag. both in int16_t
 * out: ulsch_llr[4*nb_re]. each group of 4 are: .r, .i , saturating_sub(mag_real, std::abs(.r) , saturating_sub(mag_imag, std::abs(.i)
 */

void capture_qam16(
            int32_t *rxdataF_comp,
            int32_t *ul_ch_mag,
            int16_t *ulsch_llr,
            uint32_t nb_re,
            struct timespec *ts )
{
    pthread_mutex_lock( &capture_lock );

    c16_t *rxF = (c16_t *)rxdataF_comp;
    int16_t *ul_ch_mag_i16 = (int16_t *)ul_ch_mag;

    fprint_time( f_in, ts );
    fprintf( f_in, "QAM16\n" );
    fprintf( f_in, "%d\n", nb_re );
    for(int i = 0; i < nb_re; i++ )
      fprintf( f_in, "%hd %hd %hd %hd\n", rxF[i].r, rxF[i].i, ul_ch_mag_i16[2*i+0], ul_ch_mag_i16[2*i+1] );
    fflush( f_in );

    fprint_time( f_out, ts );
    fprintf( f_out, "QAM16\n" );
    fprintf( f_out, "%d\n", nb_re );
    for(int i = 0; i < nb_re; i++ )
      fprintf( f_out, "%hd %hd %hd %hd\n", ulsch_llr[4*i+0], ulsch_llr[4*i+1], ulsch_llr[4*i+2], ulsch_llr[4*i+3] );
    fflush( f_out );

    pthread_mutex_unlock( &capture_lock );
}

// Plugin Init / Shutdown

int32_t demapper_init( void )
{
    // initialize capture mutex
    pthread_mutex_init( &capture_lock, NULL );

    // open capture files
    f_in = fopen( filename_in, "w" );
    AssertFatal( f_in != NULL, "Cannot open file %s for writing\n", filename_in );

    f_out = fopen( filename_out, "w");
    AssertFatal( f_out != NULL, "Cannot open file %s for writing\n", filename_out );

    // print clock resolution
    struct timespec ts;
    clock_getres( TIMESTAMP_CLOCK_SOURCE, &ts );
    fprint_time( f_in, &ts );
    fprint_time( f_out, &ts );

    return 0;
}

int32_t demapper_shutdown( void )
{
    fclose( f_in );
    fclose( f_out );

    pthread_mutex_destroy( &capture_lock );

    return 0;
}

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
    nr_ulsch_qpsk_llr(rxdataF_comp,
                        ulsch_llr,
                        nb_re,
                        symbol);
    capture_qpsk(rxdataF_comp, ulsch_llr, nb_re , &ts);
      break;
    case 4:
    nr_ulsch_16qam_llr(rxdataF_comp,
                         ul_ch_mag,
                         ulsch_llr,
                         nb_re,
                         symbol);
    capture_qam16(rxdataF_comp, ul_ch_mag, ulsch_llr, nb_re, &ts);
      break;

#if 0 // disabled
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
#endif

    default:
      AssertFatal(1==0,"capture_demapper_compute_llr: invalid/unhandled Qm value, symbol = %d, Qm = %d\n",symbol, mod_order);
      return 0; // unhandled
  }
  return 1; // handled
}
// END marker-capture-full
