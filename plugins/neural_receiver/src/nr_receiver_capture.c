#define _GNU_SOURCE
#include "openair1/PHY/TOOLS/tools_defs.h"
#include "openair1/PHY/defs_gNB.h"
#include "openair1/PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/sse_intrin.h"
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <cuda_runtime_api.h>
#ifdef __aarch64__
#define USE_128BIT
#endif

#define OUTPUT_CAPTURE

#define TIMESTAMP_CLOCK_SOURCE CLOCK_MONOTONIC

void fprint_time( FILE* file, struct timespec *ts )
{
  fprintf( file, "%jd.%09ld\n", ts->tv_sec, ts->tv_nsec );
  return;
}

static char *filename_in = "receiver_in.txt";
static char *filename_out = "receiver_out.txt";
static FILE *f_in;
static FILE *f_out;
static pthread_mutex_t capture_lock = PTHREAD_MUTEX_INITIALIZER;

// Plugin Init / Shutdown

int32_t receiver_init( void )
{
    printf("Receiver TRT init\n");
    fflush(stdout);

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

#if 0
    printf("Initializing TRT receiver (TID %d)\n", (int) gettid());
    FILE* cf = fopen("receiver_trt.config", "r");
    if (cf) {
        char weights_buf[1024] = { 0 };
        fgets(weights_buf, sizeof(weights_buf), cf);
        if (weights_buf[0]) // remove newline
            weights_buf[strlen(weights_buf)-1] = 0;
        int normalized_input = 0;
        fscanf(cf, "%d\n", &normalized_input);
        trt_receiver_configure(weights_buf, normalized_input);
        fclose(cf);
    }

    context = trt_receiver_init(1);
    if (context) {
        printf("Initialized TRT receiver\n");
        return 0;
    }
    else
        return -1;
#endif
    return 0;
}

int32_t receiver_init_thread( void ) {
#if 0
    printf("Initializing TRT receiver context (TID %d)\n", (int) gettid());
    if (!context) {
        context = trt_receiver_init(1);
        fflush(stdout);
    }
    if (!context)
        return -1;
#endif
    return 0;
}

int32_t receiver_shutdown( void )
{
    fclose( f_in );
    fclose( f_out );

    pthread_mutex_destroy( &capture_lock );

    //trt_receiver_shutdown();
    return 0;
}

// receiver handler

int receiver_compute_llr(PHY_VARS_gNB *gNB,
                         int ulsch_id,
                         int slot,
                         frame_t frame,
                         NR_DL_FRAME_PARMS *frame_parms,
                         NR_gNB_PUSCH *pusch_vars,
                         nfapi_nr_pusch_pdu_t *rel15_ul,
                         c16_t **rxFs,
                         c16_t **ul_chs,
                         int16_t *llr,
                         int soffset,
                         int16_t const* lengths,
                         int start_symbol,
                         int num_symbols,
                         int output_shift,
                         uint32_t nvar) {
  struct timespec ts;
  clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts );

  if (ul_chs) // start with input capture
    pthread_mutex_lock( &capture_lock );

#ifdef OUTPUT_CAPTURE
  // output capture
  if (!ul_chs) {
    //int16_t* llr_decode = (int16_t*) rxFs[0];
    int16_t* llr_unscrambled = llr;

    fprint_time( f_out, &ts );
    fprintf( f_out, "QAM%d\n", (int) rel15_ul->qam_mod_order );
    fprintf( f_out, "%d %d\n", num_symbols, rel15_ul->rb_size );
    fprintf( f_out, "slot %d\n", slot );
    fprintf( f_out, "frame %lld\n", (long long) frame );
    fprintf( f_out, "rnti %d\n", (int) rel15_ul->rnti );

    fprintf( f_out, "dmrs_p %x\n", rel15_ul->ul_dmrs_symb_pos >> start_symbol );
    for(int s = 0; s < num_symbols; s++ ) {
      int symbol = start_symbol+s;
      int nb_re_pusch = pusch_vars->ul_valid_re_per_slot[symbol];

      int16_t* llr_ptr;
      //int16_t* llr_ptr = &llr_decode[0];
      //for(int i = 0; i < nb_re_pusch; i++ ) {
      //  for (int m = 0; m < rel15_ul->qam_mod_order; m++) {
      //    fprintf( f_out, "%hd ", llr_ptr[i * rel15_ul->qam_mod_order + m] );
      //  }
      //  fputc( '\n', f_out );
      //}

      fprintf( f_out, "sc_dmrs_data_ids %d %d %d\n", (int) rel15_ul->scid, (int) rel15_ul->ul_dmrs_scrambling_id, (int) rel15_ul->data_scrambling_id );

      llr_ptr = &llr_unscrambled[pusch_vars->llr_offset[symbol] * 1];
      for(int i = 0; i < nb_re_pusch; i++ ) {
        for (int m = 0; m < rel15_ul->qam_mod_order; m++) {
          fprintf( f_out, "%hd ", llr_ptr[i * rel15_ul->qam_mod_order + m] );
        }
        fputc( '\n', f_out );
      }
    }
    fflush( f_out );

    // capture end
    pthread_mutex_unlock( &capture_lock );
    return 0;
  }
#endif

  int nb_re_per_sym = NR_NB_SC_PER_RB * rel15_ul->rb_size;
  int nb_re = num_symbols * nb_re_per_sym;

  int start_re = (frame_parms->first_carrier_offset + (rel15_ul->rb_start + rel15_ul->bwp_start) * NR_NB_SC_PER_RB) % frame_parms->ofdm_symbol_size;
  int re_wrap_offset = frame_parms->ofdm_symbol_size;

  int32_t drms_symb_pos[12] = { 0 };
  uint32_t num_dmrs_symb_pos = 0;
  for(uint32_t dmrs_symb_mask = rel15_ul->ul_dmrs_symb_pos & ((1 << 12) - 1); dmrs_symb_mask; ) {
      drms_symb_pos[num_dmrs_symb_pos] = __builtin_ffs(dmrs_symb_mask) - 1;
      dmrs_symb_mask ^= 1 << drms_symb_pos[num_dmrs_symb_pos++];
  }

  fprint_time( f_in, &ts );
  fprintf( f_in, "QAM%d\n", (int) rel15_ul->qam_mod_order );
  fprintf( f_in, "%d %d\n", num_symbols, rel15_ul->rb_size );
  fprintf( f_in, "%d:%d\n", start_re, start_re + nb_re_per_sym );
  fprintf( f_in, "slot %d\n", slot );
  fprintf( f_in, "frame %lld\n", (long long) frame );
  fprintf( f_in, "rnti %d\n", (int) rel15_ul->rnti );
  fprintf( f_in, "sc_dmrs_data_ids %d %d %d\n", (int) rel15_ul->scid, (int) rel15_ul->ul_dmrs_scrambling_id, (int) rel15_ul->data_scrambling_id );
  
  fprintf( f_in, "dmrs_p %x\n", rel15_ul->ul_dmrs_symb_pos >> start_symbol );
  fprintf( f_in, "%d\n", nb_re );
  for(int s = 0; s < num_symbols; s++ ) {
    int symbol = start_symbol+s;
    int dmrs_symbol = symbol;
#if 1
    if (gNB->chest_time == 0)
      dmrs_symbol = (rel15_ul->ul_dmrs_symb_pos >> symbol) & 0x01 ? symbol : get_valid_dmrs_idx_for_channel_est(rel15_ul->ul_dmrs_symb_pos, symbol);
    else { // average of channel estimates stored in first symbol
      int end_symbol = rel15_ul->start_symbol_index + rel15_ul->nr_of_symbols;
      dmrs_symbol = get_next_dmrs_symbol_in_slot(rel15_ul->ul_dmrs_symb_pos, rel15_ul->start_symbol_index, end_symbol);
    }
#endif

    c16_t *rxF = (c16_t *)rxFs[0] + symbol * frame_parms->ofdm_symbol_size + soffset;
    c16_t *ul_ch_est = (c16_t *)pusch_vars->ul_ch_estimates[0 * 1 + 0] + dmrs_symbol * frame_parms->ofdm_symbol_size; // note: different addressing from inner_rx

    //AssertFatal( start_re + nb_re_per_sym <= frame_parms->ofdm_symbol_size, "Start offset %d overlaps %d with %d REs\n", start_re, nb_re_per_sym, frame_parms->ofdm_symbol_size );
    //rxF += start_re;

    for(int i = 0, k = start_re; i < nb_re_per_sym; k = (k+1 < re_wrap_offset ? k+1 : 0), ++i)
      fprintf( f_in, "%hd %hd %hd %hd\n", rxF[k].r, rxF[k].i, ul_ch_est[i].r, ul_ch_est[i].i );
  }
  fflush( f_in );

#if 0
  fprint_time( f_out, ts );
  fprintf( f_out, "QAM%d\n", (int) rel15_ul->qam_mod_order );
  fprintf( f_out, "%d\n", nb_re );
  for(int i = 0; i < nb_re; i++ )
    fprintf( f_out, "%hd %hd %hd %hd\n", ulsch_llr[4*i+0], ulsch_llr[4*i+1], ulsch_llr[4*i+2], ulsch_llr[4*i+3] );
  fflush( f_out );
#endif

#ifdef OUTPUT_CAPTURE
  return -1;
#else
  pthread_mutex_unlock( &capture_lock );
  return 0;
#endif
}

