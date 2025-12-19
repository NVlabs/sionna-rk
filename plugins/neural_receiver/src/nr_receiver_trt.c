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

//#define PRINT_TIMES
#define VISUALIZE_STATS

#define VISUALIZE_PLOT_HEIGHT 32
#define VISUALIZE_WINDOW_SLICES 80

#define TIMESTAMP_CLOCK_SOURCE CLOCK_MONOTONIC

void fprint_time( FILE* file, struct timespec *ts )
{
  fprintf( file, "%jd.%09ld\n", ts->tv_sec, ts->tv_nsec );
  return;
}

void trt_receiver_configure(char const* weight_file);

struct TRTContext;
struct TRTContext* trt_receiver_init(int make_stream);
void trt_receiver_shutdown();

void trt_receiver_decode(struct TRTContext* context_, cudaStream_t stream, int16_t const* in_active_ports, size_t num_tx,
                                          int16_t const* in_symbols, size_t num_subcarriers, size_t num_ofdm_symbols, size_t num_antenna,
                                          float norm_scale,
                                          int16_t const* in_h, size_t num_pilots,
                                          int32_t const* in_dmrs_ofdm_pos, size_t num_dmrs_symbols, // num_tx x num_dmrs_symbols
                                          int32_t const* in_dmrs_subcarrier_pos, size_t num_pilots_per_prb,
                                          int16_t* outputs, uint32_t num_bits_per_symbol);

// Plugin Init / Shutdown

static __thread struct TRTContext* context;

#define MAX_WORKER_THREADS 128
static unsigned int local_thread_count;
static __thread unsigned int local_thread_id;

static frame_t shared_log_frame;

#if defined(PRINT_TIMES) || defined(VISUALIZE_STATS)
struct TimeMeasurements {
    unsigned long long avg_ns;
    unsigned long long max_ns;
    size_t count;
};
static __thread struct TimeMeasurements inference_time;
static __thread struct TimeMeasurements input_time;
static __thread struct TimeMeasurements output_time;

static unsigned add_measurement(struct TimeMeasurements* time, unsigned long long time_ns, size_t max_samples) {
    if (++time->count > max_samples)
        time->count = max_samples;
    time->avg_ns = (unsigned long long)((long long)time->avg_ns + (long long)(time_ns - time->avg_ns) / (int) time->count);
    if (time_ns > time->max_ns)
        time->max_ns = time_ns;
    return time->count;
}
#endif

#ifdef VISUALIZE_STATS

struct LabeledStats {
    unsigned int data;
    unsigned int label;
};

#define STATS_TIME_WINDOW 30
#define STATS_TIME_RESOLUTION 10
static struct LabeledStats inference_count_stats[STATS_TIME_WINDOW * STATS_TIME_RESOLUTION][MAX_WORKER_THREADS];
static struct LabeledStats prb_count_stats[STATS_TIME_WINDOW * STATS_TIME_RESOLUTION][MAX_WORKER_THREADS];
static struct LabeledStats inference_time_stats[STATS_TIME_WINDOW * STATS_TIME_RESOLUTION][MAX_WORKER_THREADS];

static unsigned int get_time_slice(struct timespec* ts) {
    return (ts->tv_sec % STATS_TIME_WINDOW) * STATS_TIME_RESOLUTION + (ts->tv_nsec / 10000 * STATS_TIME_RESOLUTION / 100000);
}

static void accumulate_in_slice(unsigned int slice, struct LabeledStats (*all_stats)[MAX_WORKER_THREADS], unsigned int data, struct timespec* ts_label) {
    struct LabeledStats* stats = &all_stats[slice][local_thread_id];
    if (stats->label != ts_label->tv_sec) {
        stats->label = ts_label->tv_sec;
        stats->data = 0;
    }
    stats->data += data;
}

static size_t extract_slices_before(unsigned int* extracted_slices, struct LabeledStats (*stats)[MAX_WORKER_THREADS], size_t current_time_slice, struct timespec* ts_label, size_t extract_slices, size_t counting_slices) {
  size_t counting_window_count = 0;
  assert(extract_slices >= counting_slices);
  unsigned int start_time_slice = (current_time_slice - extract_slices + STATS_TIME_WINDOW * STATS_TIME_RESOLUTION) % (STATS_TIME_WINDOW * STATS_TIME_RESOLUTION);
  unsigned int ref_label = ts_label->tv_sec - extract_slices / STATS_TIME_RESOLUTION - 1;
  for (unsigned i = 0; i < extract_slices; ++i) {
    unsigned y = 0;
    unsigned int s = (start_time_slice + i) % (STATS_TIME_WINDOW * STATS_TIME_RESOLUTION);
    for (unsigned t = 0; t < MAX_WORKER_THREADS; ++t)
      if (stats[s][t].label - ref_label >= 0)
          y += stats[s][t].data;
    if (i >= extract_slices - counting_slices)
      counting_window_count += y;
    extracted_slices[i] = y;
  }
  return counting_window_count;
}

static unsigned max_of(unsigned int* data, size_t extract_slices, unsigned init) {
  unsigned max_val = init;
  for (unsigned i = 0; i < extract_slices; ++i) {
    if (data[i] > max_val)
      max_val = data[i];
  }
  return max_val;
}

static void tui_plot_data(char* bitmap, unsigned int* data, unsigned plot_max_val) {
  assert(VISUALIZE_WINDOW_SLICES % 2 == 0);
  assert(VISUALIZE_PLOT_HEIGHT % 4 == 0);

  #define TUI_PLOT_LINE_PREFIX 2
  #define TUI_PLOT_LINE_WIDTH (VISUALIZE_WINDOW_SLICES/2 * 3 + 1 + TUI_PLOT_LINE_PREFIX)
  #define TUI_PLOT_LINE_HEIGHT (VISUALIZE_PLOT_HEIGHT/4+1)
  //char bitmap[TUI_PLOT_LINE_WIDTH * TUI_PLOT_LINE_HEIGHT + 1] = { };

  for (unsigned c = 0; c <= VISUALIZE_PLOT_HEIGHT; c += 4) {
    bitmap[c/4 * TUI_PLOT_LINE_WIDTH] = '|';
    bitmap[c/4 * TUI_PLOT_LINE_WIDTH+1] = ' ';
    bitmap[(c/4+1) * TUI_PLOT_LINE_WIDTH - 1] = '\n';
  }
  for (unsigned i = 0; i < 1+VISUALIZE_WINDOW_SLICES/2; ++i)
    bitmap[VISUALIZE_PLOT_HEIGHT/4 * TUI_PLOT_LINE_WIDTH+1+i] = '_';

  for (unsigned i = 0; i < VISUALIZE_WINDOW_SLICES; i+=2) {
    unsigned y[2] = { data[i], data[i+1] };

    for (unsigned c = 0; c < VISUALIZE_PLOT_HEIGHT; c += 4) {
      unsigned symbol = 0;
      for (unsigned d = 0; d < 2; ++d) {
        unsigned val = (y[d] * VISUALIZE_PLOT_HEIGHT + (plot_max_val - 1)) / plot_max_val;
        symbol += (val > c + 3) << (0+3*d);
        symbol += (val > c + 2) << (1+3*d);
        symbol += (val > c + 1) << (2+3*d);
        symbol += (val > c + 0) << (6+d);
      }
      unsigned bo = (VISUALIZE_PLOT_HEIGHT/4 - 1 - c/4) * TUI_PLOT_LINE_WIDTH + TUI_PLOT_LINE_PREFIX + i/2 * 3;
      bitmap[bo+0] = 0xe0 | 0x02;
      bitmap[bo+1] = 0x80 | 0x20 | ((symbol >> 6) & 3);
      bitmap[bo+2] = 0x80 | (symbol & 0x3f);
    }
  }
}

#endif

int32_t receiver_init( void )
{
    printf("Receiver TRT init\n");
    fflush(stdout);

    printf("Initializing TRT receiver (TID %d)\n", (int) gettid());
    FILE* cf = fopen("plugins/neural_receiver/config/receiver_trt.config", "r");
    if (cf) {
        char weights_buf[1024] = { 0 };
        fgets(weights_buf, sizeof(weights_buf), cf);
        if (weights_buf[0]) // remove newline
            weights_buf[strlen(weights_buf)-1] = 0;
        //int normalized_input = 0;
        //fscanf(cf, "%d\n", &normalized_input);
        trt_receiver_configure(weights_buf);
        fclose(cf);
    }

    context = trt_receiver_init(1);
    if (context) {
        printf("Initialized TRT receiver\n");
        return 0;
    }
    else
        return -1;
    return 0;
}

int32_t receiver_init_thread( void ) {
    printf("Initializing TRT receiver context (TID %d)\n", (int) gettid());
    if (!context) {
        context = trt_receiver_init(1);
        local_thread_id = __atomic_add_fetch(&local_thread_count, 1, __ATOMIC_ACQ_REL);
        assert(local_thread_id < MAX_WORKER_THREADS);
        fflush(stdout);
    }
    if (!context)
        return -1;
    return 0;
}

int32_t receiver_shutdown( void )
{
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
  if (!context) {
      printf("Warning: uninitialized NRX thread context!\n");
      context = trt_receiver_init(1);
      fflush(stdout);
  }

  if (rel15_ul->qam_mod_order < 4)
    return 0;
  AssertFatal(rel15_ul->qam_mod_order == 4, "Only mod order <= QAM16 supported by NRX");

  struct timespec ts_begin, ts_cursor, ts_end;
  clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_begin );

#ifdef VISUALIZE_STATS
  unsigned current_stats_slice = get_time_slice(&ts_begin);
  accumulate_in_slice(current_stats_slice, inference_count_stats, 1, &ts_begin);
  accumulate_in_slice(current_stats_slice, prb_count_stats, rel15_ul->rb_size, &ts_begin);
#endif

  int num_ofdm_symbols = frame_parms->symbols_per_slot - 1; // Our OAI configs never use last slot
  //AssertFatal(num_ofdm_symbols == rel15_ul->nr_of_symbols, "Only full symbol blocks supported currently");
  if (num_ofdm_symbols != rel15_ul->nr_of_symbols || num_symbols != num_ofdm_symbols)
    printf("Only full symbol blocks tested currently, got (%d:%d) for #%d out of %d\n", start_symbol, start_symbol+num_symbols, rel15_ul->nr_of_symbols, num_ofdm_symbols);
  AssertFatal(start_symbol == 0, "Only 0-based symbol blocks supported currently");

  int nb_re_per_sym = NR_NB_SC_PER_RB * rel15_ul->rb_size;
  int nb_re = num_symbols * nb_re_per_sym;

  int start_re = (frame_parms->first_carrier_offset + (rel15_ul->rb_start + rel15_ul->bwp_start) * NR_NB_SC_PER_RB) % frame_parms->ofdm_symbol_size;
  int re_wrap_offset = frame_parms->ofdm_symbol_size;

  int32_t dmrs_symb_pos[12] = { 0 };
  uint32_t num_dmrs_symb_pos = 0;
  for(uint32_t dmrs_symb_mask = rel15_ul->ul_dmrs_symb_pos & ((1 << 12) - 1); dmrs_symb_mask; ) {
      dmrs_symb_pos[num_dmrs_symb_pos] = __builtin_ffs(dmrs_symb_mask) - 1;
      dmrs_symb_mask ^= 1 << dmrs_symb_pos[num_dmrs_symb_pos++];
  }

  c16_t* symbols = alloca(sizeof(int16_t) * 1 * 2 * nb_re_per_sym * num_ofdm_symbols);
  memset(symbols, 0, sizeof(int16_t) * 1 * 2 * nb_re_per_sym * num_ofdm_symbols);
  c16_t* h_hat = alloca(sizeof(int16_t) * 1 * 2 * nb_re_per_sym / 2 * num_dmrs_symb_pos);
  memset(h_hat, 0, sizeof(int16_t) * 1 * 2 * nb_re_per_sym / 2 * num_dmrs_symb_pos);

  float norm_scale = 0.0;
  int norm_scale_count = 0;

  // transpose
  for(int s = 0; s < num_symbols; s++ ) {
    int symbol = start_symbol+s;
    AssertFatal(symbol < num_ofdm_symbols, "Symbol %d out of bounds %d", symbol, num_ofdm_symbols);

    c16_t *rxF = (c16_t*) rxFs[0] + symbol * frame_parms->ofdm_symbol_size + soffset;

    int is_dmrs = (rel15_ul->ul_dmrs_symb_pos & (1 << symbol)) != 0;

    //AssertFatal(start_re + nb_re_per_sym <= frame_parms->ofdm_symbol_size, "Start offset %d overlaps %d with %d REs\n", start_re, nb_re_per_sym, frame_parms->ofdm_symbol_size);
    for(int i = 0, k = start_re; i < nb_re_per_sym; k = (k+1 < re_wrap_offset ? k+1 : 0), ++i) {
        c16_t rxs = rxF[k];
        symbols[num_ofdm_symbols * i + symbol] = rxs;

        float r = ldexpf((float) rxs.r, -8);
        float i = ldexpf((float) rxs.i, -8);
        float m = r*r + i*i;
        norm_scale += (m - norm_scale) / (float) ++norm_scale_count;
    }
  }
  norm_scale = 1.0f / sqrtf(norm_scale);

  // select
  for(int s = 0; s < num_dmrs_symb_pos; s++ ) {
    int dmrs_symbol = dmrs_symb_pos[s];

    c16_t *ul_ch_est = (c16_t *)pusch_vars->ul_ch_estimates[0 * 1 + 0] + dmrs_symbol * frame_parms->ofdm_symbol_size; // note: different addressing from inner_rx

    //for(int i = 0; i < nb_re_per_sym; i++)
    //    h_hat[num_dmrs_symb_pos * i + s] = ul_ch_est[i];
    for(int i = 0, k = 0; i < nb_re_per_sym; i+=2, ++k)
        h_hat[k + s * nb_re_per_sym/2] = ul_ch_est[i];
  }

  int32_t subcarrier_pos[] = { 0, 2, 4, 6, 8, 10 };
  int16_t port_mask[] = { 1 };

  int16_t* outputs = alloca(sizeof(int16_t) * 1 * nb_re_per_sym * num_ofdm_symbols * 4);

  clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_end );
  unsigned long long time_ns;
  unsigned message_count;
#ifdef PRINT_TIMES
  time_ns = ts_end.tv_nsec - ts_begin.tv_nsec + 1000000000ll * (ts_end.tv_sec - ts_begin.tv_sec);
  message_count = add_measurement(&input_time, time_ns, 500);
  if (message_count % 500 == 499) {
    time_ns = input_time.avg_ns;
    printf("Input processing runtime: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
    fflush(stdout);
    memset(&input_time, 0, sizeof(input_time));
  }
#endif

  trt_receiver_decode(context, 0, port_mask, 1 /*num_tx*/, (int16_t const*) symbols, nb_re_per_sym, num_ofdm_symbols, 1 /*num_antenna*/, norm_scale,
                      (int16_t const*) h_hat, num_dmrs_symb_pos * nb_re_per_sym/2, dmrs_symb_pos, num_dmrs_symb_pos, subcarrier_pos, 6 * num_dmrs_symb_pos,
                      outputs, 4 /* num_bits_per_symbol*/);

  clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_cursor );

  // transpose
  for(int s = 0; s < num_symbols; s++ ) {
    int symbol = start_symbol+s;
    int16_t *llrs_symbol = &llr[pusch_vars->llr_offset[symbol] * 1];

    int is_dmrs = (rel15_ul->ul_dmrs_symb_pos & (1 << symbol)) != 0;

    for(int i = is_dmrs, k = 0; i < nb_re_per_sym; i+=1+is_dmrs, ++k) {
      for(int j = 0; j < 4; j++) {
        int16_t v = outputs[(symbol + i * num_ofdm_symbols) * 4 + j];
        int16_t o = (v / 256);
        if (o > 255) o = 255;
        if (o < -255) o = -255;
        llrs_symbol[k * 4 + j] = o ? o : (v < 0 ? -1 : 1);
      }
    }
  }

    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_end );
#ifdef PRINT_TIMES
  time_ns = ts_end.tv_nsec - ts_cursor.tv_nsec + 1000000000ll * (ts_end.tv_sec - ts_cursor.tv_sec);
  message_count = add_measurement(&output_time, time_ns, 500);
  if (message_count % 500 == 499) {
    time_ns = output_time.avg_ns;
    printf("Output processing runtime: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
    fflush(stdout);
    memset(&output_time, 0, sizeof(output_time));
  }
#endif

  time_ns = ts_end.tv_nsec - ts_begin.tv_nsec + 1000000000ll * (ts_end.tv_sec - ts_begin.tv_sec);
#ifdef VISUALIZE_STATS
  accumulate_in_slice(current_stats_slice, inference_time_stats, time_ns / 1000, &ts_begin);
#endif
#ifdef PRINT_TIMES
  message_count = add_measurement(&inference_time, time_ns, 500);
  if (message_count % 500 == 499) {
    time_ns = inference_time.avg_ns;
    printf("NRX runtime: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
    fflush(stdout);
    memset(&inference_time, 0, sizeof(inference_time));
  }
#endif

#ifdef VISUALIZE_STATS
  // stats
  frame_t prev_frame = *(frame_t volatile*) &shared_log_frame; // may be changed concurrently, re-checked with atomic below
  if (/*(slot == 0) &&*/ (frame & 127) == 0 && (prev_frame != frame)) {
    if (__atomic_compare_exchange(&shared_log_frame, &prev_frame, &frame, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {

      static const unsigned int avg_slices = STATS_TIME_RESOLUTION;
      unsigned inference_count_data[VISUALIZE_WINDOW_SLICES] = { };

      double inference_count_avg = extract_slices_before(inference_count_data, inference_count_stats, current_stats_slice, &ts_begin, VISUALIZE_WINDOW_SLICES, avg_slices) / (double) avg_slices * STATS_TIME_RESOLUTION;
      unsigned max_inferences_bar = max_of(inference_count_data, VISUALIZE_WINDOW_SLICES, 1);
      char inferences_plot[TUI_PLOT_LINE_WIDTH * TUI_PLOT_LINE_HEIGHT + 1] = { };
      tui_plot_data(inferences_plot, inference_count_data, max_inferences_bar);

      unsigned plot_data[VISUALIZE_WINDOW_SLICES] = { };

      double inference_us_avg = extract_slices_before(plot_data, inference_time_stats, current_stats_slice, &ts_begin, VISUALIZE_WINDOW_SLICES, avg_slices) / (double) avg_slices * STATS_TIME_RESOLUTION;
      for (int i = 0; i < VISUALIZE_WINDOW_SLICES; ++i)
        if (plot_data[i])
          plot_data[i] /= inference_count_data[i];
      unsigned max_latencies_bar = max_of(plot_data, VISUALIZE_WINDOW_SLICES, 1);
      char inference_time_plot[TUI_PLOT_LINE_WIDTH * TUI_PLOT_LINE_HEIGHT + 1] = { };
      tui_plot_data(inference_time_plot, plot_data, max_latencies_bar);

      double prb_count_avg = extract_slices_before(plot_data, prb_count_stats, current_stats_slice, &ts_begin, VISUALIZE_WINDOW_SLICES, avg_slices) / (double) avg_slices * STATS_TIME_RESOLUTION;
      unsigned max_prbs_bar = max_of(plot_data, VISUALIZE_WINDOW_SLICES, 1);
      char prbs_plot[TUI_PLOT_LINE_WIDTH * TUI_PLOT_LINE_HEIGHT + 1] = { };
      tui_plot_data(prbs_plot, plot_data, max_prbs_bar);

      LOG_I(NR_PHY, "NRX:\n"
        "Inference count\n^ of max %u / s\n%s\n%9.2f infer / s @ %9.2f us / infer\n\n"
        "PRB count\n^ of max %u / s\n%s\n%9.2f PRBs / s @ %9.2f us / PRB\n\n"
        "Latency\n^ of max %u us\n%s\n\n",
        max_inferences_bar * STATS_TIME_RESOLUTION, inferences_plot,
        inference_count_avg,
        inference_us_avg / inference_count_avg,
        max_prbs_bar * STATS_TIME_RESOLUTION, prbs_plot,
        prb_count_avg,
        inference_us_avg / prb_count_avg,
        max_latencies_bar, inference_time_plot );

      //LOG_I(NR_PHY, "NRX: Inference count\n%s\n %9.2f PRBs / batch %9.2f PRBs / s\n\n", bitmap, 0.0, 0.0);
    }
  }
#endif

  return 1;
}

