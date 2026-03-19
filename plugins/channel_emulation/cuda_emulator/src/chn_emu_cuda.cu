/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include <cstring>
#include <cstdio>
#include <cmath>
#include <random>
#include <chrono>
#include <unistd.h>
#include <time.h>
#include <cassert>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#include "chn_emu_cuda.h"

// --- CUDA/cuBLAS Error Checking Macro ---
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#ifdef __aarch64__
#define USE_128BIT
#endif

#define TIMESTAMP_CLOCK_SOURCE CLOCK_MONOTONIC

//#define PRINT_TIMES

#ifdef PRINT_TIMES
struct TimeMeasurements {
    unsigned long long avg_ns;
    unsigned long long max_ns;
    size_t count;
};
static __thread struct TimeMeasurements input_copy_time = {};
static __thread struct TimeMeasurements output_copy_time = {};
static __thread struct TimeMeasurements compute_time = {};

static unsigned add_measurement(struct TimeMeasurements* time, unsigned long long time_ns, size_t max_samples) {
    size_t samples = ++time->count;
    if (samples > max_samples)
        samples = max_samples;
    time->avg_ns = (unsigned long long)((long long)time->avg_ns + (long long)(time_ns - time->avg_ns) / (int) samples);
    if (time_ns > time->max_ns)
        time->max_ns = time_ns;
    return time->count;
}
#endif

static int g_num_taps = 0; // Number of taps per CIR
static int g_num_symbols_per_slot = 0; // Number of OFDM symbols per slot
static float g_sigma_scaling = 0.0f; // Noise standard deviation scaling factor
static float g_sigma_max = 0.0f; // Maximum noise standard deviation
static int g_cir_entry_size_bytes = 0; // Size of each CIR entry in bytes

extern "C" void chn_emu_cuda_set_sigma_scaling(float val) { g_sigma_scaling = val; }
extern "C" void chn_emu_cuda_set_sigma_max(float val) { g_sigma_max = val; }

// Buffers used for channel emulation
struct ThreadContext {
    cudaStream_t stream = 0;

    // Channel inputs
    short2 *x = NULL;

    // Channel outputs
    cuComplex *y = NULL;
    cuComplex *host_y = NULL;

    // Storage for the channel impulse response on the device
    void *cir = NULL;

    // list of thread contexts for shutdown
    ThreadContext* next_initialized_context = nullptr;
};
static __thread ThreadContext* thread_context = { }; // note: for now, single-threaded

static ThreadContext* initialized_thread_contexts = nullptr;
static struct timespec ts_init_complete;

// Pool of noise samples to add to the signal
static cuComplex *noise_pool = NULL;
// Index of the noise sample to add to the signal
static unsigned int noise_idx = 0;

#define USE_SHARED_MEMORY
//#define ENABLE_DGX_OPTIMIZATIONS

#ifdef USE_SHARED_MEMORY

#ifndef ENABLE_DGX_OPTIMIZATIONS
#define cudaMallocStaging(pp, size, hostFlags) cudaHostAlloc(pp, size, hostFlags)
#define cudaFreeStaging cudaFreeHost
#else
#define cudaMallocStaging(pp, size, hostFlags) cudaMallocManaged(pp, size)
#define cudaFreeStaging cudaFree
#endif

#else

#define cudaMallocStaging(pp, size, hostFlags) cudaMalloc(pp, size)
#define cudaFreeStaging cudaFree

#endif

// START marker-plugin
int32_t chn_emu_cuda_init_thread(void)
{
    if (thread_context) return 0;
    thread_context = new ThreadContext();
    ThreadContext& context = *thread_context;

    printf("Initializing channel emulator context (TID %d)\n", (int) gettid());
    fflush(stdout);

    int highPriority = 0;
    if (cudaDeviceGetStreamPriorityRange(NULL, &highPriority))
        printf("CUDA stream priorities unsupported, %s:%d", __FILE__, __LINE__);
    CHECK_CUDA(cudaStreamCreateWithPriority(&context.stream, cudaStreamNonBlocking, highPriority));

    cudaStreamAttrValue attr = {};
    attr.syncPolicy = cudaSyncPolicyYield;
    cudaStreamSetAttribute(context.stream, cudaStreamAttributeSynchronizationPolicy, &attr);

    // Allocate pinned memory for the channel outputs on the host
    CHECK_CUDA(cudaMallocStaging(&context.y, MAX_SAMPLES_PER_SLOT * sizeof(cuComplex), cudaHostAllocMapped));
#ifndef USE_SHARED_MEMORY
    context.host_y = (cuComplex*) malloc(MAX_SAMPLES_PER_SLOT * sizeof(cuComplex));
#endif

    // Allocate memory for the channel inputs on the device
    // We allocate MAX_TAP_DELAY - 1 more samples than the number of samples per slot to
    // simulate inter-symbol interference through the channel time-domain convolution.
    CHECK_CUDA(cudaMallocStaging(&context.x, (MAX_SAMPLES_PER_SLOT + MAX_TAP_DELAY - 1) * sizeof(short2), cudaHostAllocMapped | cudaHostAllocWriteCombined));

    CHECK_CUDA(cudaMallocStaging(&context.cir, g_num_symbols_per_slot * g_cir_entry_size_bytes, cudaHostAllocMapped | cudaHostAllocWriteCombined));

    // keep track of active thread contexts for shutdown
    ThreadContext* self = &context;
    __atomic_exchange(&initialized_thread_contexts, &self, &self->next_initialized_context, __ATOMIC_ACQ_REL);

    return 0;
}

int32_t chn_emu_cuda_init(int num_taps, int num_symbols_per_slot, float sigma_scaling, float sigma_max) {

    assert(num_taps <= MAX_TAP_DELAY);

    g_num_taps = num_taps;
    g_num_symbols_per_slot = num_symbols_per_slot;
    g_sigma_scaling = sigma_scaling;
    g_sigma_max = sigma_max;

    // Each entry for a single CIR consists of:
    // - 1 float for the CIR norm
    // - num_taps*2 float for the channel coefficients
    // - num_taps int for the tap indices
    g_cir_entry_size_bytes = sizeof(float)
                             + num_taps * sizeof(float) * 2
                             + num_taps * sizeof(uint16_t);
    // Pad to 4-byte alignment for proper GPU memory access
    g_cir_entry_size_bytes = (g_cir_entry_size_bytes + 3) & ~3;

    CHECK_CUDA(cudaMalloc(&noise_pool, NOISE_POOL_SIZE * sizeof(cuComplex)));

    // Initialize the noise pool
    cuComplex* h_noise_pool = (cuComplex*) malloc(NOISE_POOL_SIZE * sizeof(cuComplex));
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0f, sqrtf(0.5f));
    for (int i = 0; i < NOISE_POOL_SIZE; i++) {
        h_noise_pool[i].x = distribution(generator);
        h_noise_pool[i].y = distribution(generator);
    }
    CHECK_CUDA(cudaMemcpy(noise_pool, h_noise_pool, NOISE_POOL_SIZE * sizeof(cuComplex), cudaMemcpyHostToDevice));
    free(h_noise_pool);
    h_noise_pool = NULL;

    int result = chn_emu_cuda_init_thread();

    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_init_complete );
    return result;
}

int32_t chn_emu_cuda_shutdown(void)
{
    cudaDeviceSynchronize();

    ThreadContext* active_context = nullptr;
    __atomic_exchange(&initialized_thread_contexts, &active_context, &active_context, __ATOMIC_ACQ_REL);
    while (active_context) {
        ThreadContext& context = *active_context;

        if (active_context->stream)
            cudaStreamDestroy(active_context->stream);

        // Free the shared and host memory
        CHECK_CUDA(cudaFreeStaging(context.x));
        CHECK_CUDA(cudaFreeStaging(context.y));
        CHECK_CUDA(cudaFreeStaging(context.cir));

        free(context.host_y);

        active_context = active_context->next_initialized_context;
    }
    thread_context = nullptr;

    // Free the device memory
    CHECK_CUDA(cudaFree(noise_pool));
    noise_pool = NULL;

    return 0;
}

//#define TAP_KERNEL_BLOCK 32
//#define UNROLL_TAPS 16
#define TAP_KERNEL_BLOCK 256
#define UNROLL_TAPS 1

 __global__ void tapped_delay_line_kernel(cuComplex* y,
                                          const short2* x,
                                          const void* cir,
                                          int samples_per_slot,
                                          int num_taps,
                                          int num_symbols_per_slot,
                                          int samples_first_symbol,
                                          int samples_other_symbols,
                                          float sigma_scaling,
                                          float sigma_max,
                                          int cir_entry_size_bytes) {

    // Calculate the global thread index 'n'
    int s = blockIdx.x * blockDim.x + threadIdx.x;

#if UNROLL_TAPS > 1
    __shared__ float res_local[TAP_KERNEL_BLOCK][UNROLL_TAPS+1];
    __shared__ float ims_local[TAP_KERNEL_BLOCK][UNROLL_TAPS+1];
    res_local[threadIdx.x][threadIdx.y] = 0.0f;
    ims_local[threadIdx.x][threadIdx.y] = 0.0f;
#endif

    const float *norm; // initialized in the loop below
    // Boundary check: ensure the thread is within the bounds of the output vector
    if (s < samples_per_slot) {
        cuComplex sum = make_cuComplex(0.0f, 0.0f);

        // Compute the indices offset to gather the CIR taps and the noise std
        // for the OFDM symbol
        int cir_index_offset = 0;
        int symbol_max_samples = samples_first_symbol;
        for (int i = 0; i < num_symbols_per_slot; i++) {
            if (s < symbol_max_samples) {
                break;
            } else {
                cir_index_offset += cir_entry_size_bytes;
                symbol_max_samples += samples_other_symbols;
            }
        }

        // Create pointers for norm, taps, and tap indices using cir_entry_size_bytes from cir
        const uint8_t* cir_base_ptr = reinterpret_cast<const uint8_t*>(cir) + cir_index_offset;

        norm = reinterpret_cast<const float*>(cir_base_ptr);
        const float* taps_coeff = reinterpret_cast<const float*>(cir_base_ptr + sizeof(float));
        const uint16_t* taps_indices = reinterpret_cast<const uint16_t*>(cir_base_ptr + sizeof(float) + num_taps * sizeof(float) * 2);

        for (int l = threadIdx.y; l < num_taps; l += UNROLL_TAPS) {

            if (taps_indices[l] >= MAX_TAP_DELAY) {
                // This should not happen.
                // We ignore tap delays that are greater than MAX_TAP_DELAY.
                return;
            }

            // Calculate the source index into the input signal 'x'
            int s_ind = s - taps_indices[l] + MAX_TAP_DELAY - 1;

            // Read h_l (complex tap coefficient as interleaved real/imag) and x_{n-t[l]}
            cuComplex h_val = make_cuComplex(taps_coeff[l * 2], taps_coeff[l * 2 + 1]);
            cuComplex x_val = make_cuComplex((float)x[s_ind].x, (float)x[s_ind].y);

            // Perform the complex multiplication: h_l * x_{n-t[l]}
            cuComplex product = cuCmulf(h_val, x_val);

            // Add the product to the running sum
            sum = cuCaddf(sum, product);
        }

#if UNROLL_TAPS > 1
	res_local[threadIdx.x][threadIdx.y] = sum.x;
	ims_local[threadIdx.x][threadIdx.y] = sum.y;
    }

    __syncthreads();

    for (int i = threadIdx.y; i < TAP_KERNEL_BLOCK; i += UNROLL_TAPS) {
        float im_sum = 0.0f, re_sum = 0.0f;
        if (threadIdx.x < UNROLL_TAPS) {
            re_sum = res_local[i][threadIdx.x];
            im_sum = ims_local[i][threadIdx.x];
        }
        re_sum += __shfl_xor_sync(0xffffffff, re_sum, 1); // xxyyzzww
        im_sum += __shfl_xor_sync(0xffffffff, im_sum, 1); // xxyyzzww
        re_sum += __shfl_xor_sync(0xffffffff, re_sum, 2); // xxxxyyyy
        im_sum += __shfl_xor_sync(0xffffffff, im_sum, 2); // xxxxyyyy
        re_sum += __shfl_xor_sync(0xffffffff, re_sum, 4); // xxxxxxxx
        im_sum += __shfl_xor_sync(0xffffffff, im_sum, 4); // xxxxxxxx
        re_sum += __shfl_xor_sync(0xffffffff, re_sum, 8); // xxxxxxxx...
        im_sum += __shfl_xor_sync(0xffffffff, im_sum, 8); // xxxxxxxx...
	if (threadIdx.x == 0) {
            res_local[i][0] = re_sum;
            ims_local[i][0] = im_sum;
	}
    }

    __syncthreads();

    if (s < samples_per_slot && threadIdx.y == 0) {
        cuComplex sum = make_cuComplex(res_local[threadIdx.x][0], ims_local[threadIdx.x][0]);
#endif
        // Write the final computed value to the output vector
        float noise_std = sigma_scaling/norm[0] > sigma_max ? sigma_max : sigma_scaling/norm[0];
        cuComplex noise_std_cpx = make_cuComplex(noise_std, 0.0f);
        y[s] = cuCaddf(cuCmulf(y[s], noise_std_cpx), sum);
    }
}

#ifdef USE_SHARED_MEMORY
#define hostMemcpy(dst, src, size, hint, stream) \
	memcpy(dst, src, size)
#else
#define hostMemcpy(dst, src, size, hint, stream) \
	CHECK_CUDA(cudaMemcpyAsync(dst, src, size, hint, stream))
#endif

void chn_emu_cuda_compute(void *data,
                          int samples_per_slot,
                          int samples_per_frame,
                          int samples_first_symbol,
                          int samples_other_symbols,
                          int data_offset,
                          const char *direction,
                          const void* cir)
{
    assert(samples_per_slot <= MAX_SAMPLES_PER_SLOT);

    struct timespec ts_begin;
#ifdef PRINT_TIMES
    struct timespec ts_cursor, ts_end;
    unsigned long long time_ns;
    size_t message_count;

#endif
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_begin );
#ifndef ENABLE_NANOBIND
    // time backoff until system runs stably (disabled for Python testing)
    if (ts_init_complete.tv_sec + 10 > ts_begin.tv_sec)
        return;
#endif

    short2 *data_as_short2 = (short2 *)data;

    if (!thread_context) chn_emu_cuda_init_thread(); // note: currently no way to initialize L1 threads calling in
    ThreadContext& context = *thread_context;
    cudaStream_t cuda_stream = context.stream;
    short2 *d_x = NULL;
    cuComplex *h_y = NULL;
    cuComplex *d_y = NULL;

    /*if (strcmp(direction, "rx") == 0)*/ {
        h_y = d_y = context.y;
        d_x = context.x;
#ifndef USE_SHARED_MEMORY
	h_y = context.host_y;
#endif
    }

    // Copy to the device

    // y is prefilled with noise, and the result of the convolution is added to it
    // Randomly select noise samples
    unsigned int noise_offset = __atomic_fetch_add(&noise_idx, 1, __ATOMIC_RELAXED);
    noise_offset = (noise_offset * 5237U) % (NOISE_POOL_SIZE - samples_per_slot);
    CHECK_CUDA(cudaMemcpyAsync(d_y, noise_pool + noise_offset, samples_per_slot * sizeof(cuComplex), cudaMemcpyDeviceToDevice, cuda_stream));

    hostMemcpy(context.cir, cir, g_num_symbols_per_slot * g_cir_entry_size_bytes, cudaMemcpyHostToDevice, cuda_stream);

    // For channel inputs, we assume that `data` is a circular buffer
    const int x_chunk_size = samples_per_slot + MAX_TAP_DELAY - 1;
    const int start_sample_x = (data_offset - MAX_TAP_DELAY + 1 + samples_per_frame) % samples_per_frame;
    const int end_sample_x = (data_offset + samples_per_slot - 1 + samples_per_frame) % samples_per_frame;
    if (start_sample_x <= end_sample_x) {
        hostMemcpy(d_x, data_as_short2 + start_sample_x, x_chunk_size * sizeof(short2), cudaMemcpyHostToDevice, cuda_stream);
    } else {
        const int first_part_size = samples_per_frame - start_sample_x;
        const int second_part_size = x_chunk_size - first_part_size;
        assert(second_part_size + first_part_size == x_chunk_size);
        hostMemcpy(d_x, data_as_short2 + start_sample_x, first_part_size * sizeof(short2), cudaMemcpyHostToDevice, cuda_stream);
        hostMemcpy(d_x + first_part_size, data_as_short2, second_part_size * sizeof(short2), cudaMemcpyHostToDevice, cuda_stream);
   }

#ifdef PRINT_TIMES
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_cursor );

    time_ns = ts_cursor.tv_nsec - ts_begin.tv_nsec + 1000000000ll * (ts_cursor.tv_sec - ts_begin.tv_sec);
    message_count = add_measurement(&input_copy_time, time_ns, 500);
    if (message_count % 500 == 0) {
      time_ns = input_copy_time.avg_ns;
      printf("Input copy time: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
      fflush(stdout);
    }
#endif

    // Computes the convolution using matrix-vector multiplication
    // Use 256 threads per block as a general-purpose choice
    const dim3 block_size = { TAP_KERNEL_BLOCK, UNROLL_TAPS, 1 };
    // Calculate grid size to cover all samples_per_slot output samples
    const int grid_size = (samples_per_slot + block_size.x - 1) / block_size.x;
    tapped_delay_line_kernel<<<grid_size, block_size, 0, cuda_stream>>>(d_y, d_x, context.cir,
        samples_per_slot, g_num_taps, g_num_symbols_per_slot, samples_first_symbol, samples_other_symbols,
        g_sigma_scaling, g_sigma_max, g_cir_entry_size_bytes);
    // Check for any errors launched by the kernel
    CHECK_CUDA(cudaGetLastError());

#ifndef USE_SHARED_MEMORY
    CHECK_CUDA(cudaMemcpyAsync(h_y, d_y, samples_per_slot * sizeof(cuComplex), cudaMemcpyDeviceToHost, cuda_stream));
#endif

    // Synchronize to ensure the cuBLAS operation is complete
    CHECK_CUDA(cudaStreamSynchronize(cuda_stream));

#ifdef PRINT_TIMES
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_cursor );
#endif

#ifdef PRINT_TIMES
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_end );

    time_ns = ts_end.tv_nsec - ts_cursor.tv_nsec + 1000000000ll * (ts_end.tv_sec - ts_cursor.tv_sec);
    message_count = add_measurement(&output_copy_time, time_ns, 500);
    if (message_count % 500 == 0) {
      time_ns = output_copy_time.avg_ns;
      printf("Output copy time: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
      fflush(stdout);
    }
#endif

    // Writes the channel output back to the circular buffer data
    //
    // The buffer wraps around if the end of the buffer is before the start of the buffer.
    // This is handled by copying in two parts.
    // NOTE: memcpy is not used as casting to int16_t is required.
    const int start_sample_y = (data_offset + samples_per_frame) % samples_per_frame;
    const int first_part_size = samples_per_frame - start_sample_y;
    if (first_part_size < samples_per_slot) {
      // Buffer wraps around - copy in two parts
      // Copy first part (from start_sample_y to end of frame)
      int i = start_sample_y;
      for (int j = 0; j < first_part_size; j++) {
        data_as_short2[i].x = (int16_t)(h_y[j].x);
        data_as_short2[i].y = (int16_t)(h_y[j].y);
        i++;
      }
      // Copy second part (from beginning of frame)
      i = 0;
      for (int j = first_part_size; j < samples_per_slot; j++) {
        data_as_short2[i].x = (int16_t)(h_y[j].x);
        data_as_short2[i].y = (int16_t)(h_y[j].y);
        i++;
      }
    }
    else {
      int i = start_sample_y;
      for (int j = 0; j < samples_per_slot; j++) {
        data_as_short2[i].x = (int16_t)(h_y[j].x);
        data_as_short2[i].y = (int16_t)(h_y[j].y);
        i++;
      }
    }

#ifdef PRINT_TIMES
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_end );

    time_ns = ts_end.tv_nsec - ts_begin.tv_nsec + 1000000000ll * (ts_end.tv_sec - ts_begin.tv_sec);
    message_count = add_measurement(&compute_time, time_ns, 500);
    if (message_count % 500 == 0) {
      time_ns = compute_time.avg_ns;
      printf("CUDA sync runtime: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
      fflush(stdout);
    }
#endif
}

#ifdef ENABLE_NANOBIND

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

namespace nb = nanobind;

NB_MODULE(chn_emu_cuda, m) {
    m.def("init", [](int num_taps, int num_symbols_per_slot, float sigma_scaling, float sigma_max) {
        return chn_emu_cuda_init(num_taps, num_symbols_per_slot, sigma_scaling, sigma_max);
    }, "Initialize the channel emulator");

    m.def("shutdown", []() {
        return chn_emu_cuda_shutdown();
    }, "Shutdown the channel emulator");

    m.def("compute", [](
        nb::ndarray<int16_t, nb::shape<-1, 2>, nb::device::cpu>& data,  // IQ samples as [N, 2]
        nb::ndarray<float, nb::shape<-1>, nb::device::cpu>& cir_norms,
        nb::ndarray<float, nb::shape<-1, 2>, nb::device::cpu>& cir_taps,  // CIR as [num_taps*num_symbols, 2] (real, imag)
        nb::ndarray<uint16_t, nb::shape<-1>, nb::device::cpu>& tap_indices,
        int samples_per_slot,
        int samples_per_frame,
        int samples_first_symbol,
        int samples_other_symbols,
        int data_offset
    ) {
        // Get dimensions
        int num_symbols = cir_norms.shape(0);
        int total_taps = tap_indices.shape(0);
        int num_taps_per_symbol = total_taps / num_symbols;

        // Calculate the size of each CIR entry (packed format)
        // Format: [norm (float)] [taps real/imag (float*2*num_taps)] [tap_indices (uint16_t*num_taps)]
        size_t cir_entry_size = sizeof(float)
                                + num_taps_per_symbol * sizeof(float) * 2
                                + num_taps_per_symbol * sizeof(uint16_t);
        // Pad to 4-byte alignment
        cir_entry_size = (cir_entry_size + 3) & ~3;

        // Allocate packed CIR data buffer
        std::vector<uint8_t> packed_cir(num_symbols * cir_entry_size);

        const float* taps_ptr = reinterpret_cast<const float*>(cir_taps.data());
        const uint16_t* indices_ptr = tap_indices.data();
        const float* norms_ptr = reinterpret_cast<const float*>(cir_norms.data());

        // Pack the CIR data for each symbol
        for (int sym = 0; sym < num_symbols; sym++) {
            uint8_t* entry_ptr = packed_cir.data() + sym * cir_entry_size;

            // Write norm (noise_std for this symbol)
            *reinterpret_cast<float*>(entry_ptr) = norms_ptr[sym];
            entry_ptr += sizeof(float);

            // Write taps (interleaved real/imag)
            memcpy(entry_ptr, taps_ptr + sym * num_taps_per_symbol * 2, num_taps_per_symbol * sizeof(float) * 2);
            entry_ptr += num_taps_per_symbol * sizeof(float) * 2;

            // Write tap indices
            memcpy(entry_ptr, indices_ptr + sym * num_taps_per_symbol, num_taps_per_symbol * sizeof(uint16_t));
        }

        // Call the CUDA compute function
        chn_emu_cuda_compute(
            data.data(),
            samples_per_slot,
            samples_per_frame,
            samples_first_symbol,
            samples_other_symbols,
            data_offset,
            "rx",  // direction
            packed_cir.data()
        );
    }, "Apply channel emulation to IQ samples");
}

#endif
