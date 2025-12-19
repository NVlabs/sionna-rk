/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>

#include "ldpc_tables_bg1.h"
#include "ldpc_tables_bg2.h"

#define TIMESTAMP_CLOCK_SOURCE CLOCK_MONOTONIC

// START marker-grid-blocks-util
inline __host__ __device__ uint32_t blocks_for(uint32_t elements, int block_size) {
    return int( uint32_t(elements + (block_size-1)) / uint32_t(block_size) );
}
// END marker-grid-blocks-util

#define USE_UNIFIED_MEMORY
//#define ENABLE_DGX_OPTIMIZATIONS
//#define USE_GRAPHS
//#define PRINT_TIMES

#ifdef USE_UNIFIED_MEMORY

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

static uint32_t const* bg_cn_degree[2][8] = {};
static uint32_t const* bg_vn_degree[2][8] = {};
static uint32_t const* bg_cn[2][8] = {};
static uint32_t const* bg_vn[2][8] = {};
static uint32_t bg_cn_size[2][8] = {};
static uint32_t bg_vn_size[2][8] = {};

static uint32_t const MAX_BG_ROWS = 46;
static uint32_t const MAX_BG_COLS = 68;
static uint32_t const MAX_Z = 384; // lifting set according to 38.212 Tab 5.3.2-1
static uint32_t const MAX_BLOCK_LENGTH = MAX_BG_COLS * MAX_Z;

// START marker-dtypes
static const int MAX_LLR_ACCUMULATOR_VALUE = 127;
typedef int8_t llr_accumulator_t;
static const int MAX_LLR_MSG_VALUE = 127;
typedef int8_t llr_msg_t;
// END marker-dtypes

// START marker-damping-factor
#define APPLY_DAMPING_INT(x) (x*3/4)
// END marker-damping-factor

// START marker-thread-context
struct ThreadContext {
    cudaStream_t stream = 0;

    // Device memory declarations - use raw pointers instead of device symbols
    int8_t* llr_in_buffer = nullptr;
    uint8_t* llr_bits_out_buffer = nullptr;
    uint32_t* syndrome_buffer = nullptr;
#ifndef USE_UNIFIED_MEMORY
    uint32_t* host_syndrome_buffer = nullptr;
#endif
    llr_msg_t* llr_msg_buffer = nullptr;
    llr_accumulator_t* llr_total_buffer = nullptr;

#ifdef USE_GRAPHS
    cudaGraphExec_t graphCtx = nullptr;
#endif

    // list of thread contexts for shutdown
    ThreadContext* next_initialized_context = nullptr;
};
static __thread ThreadContext thread_context = { };
// END marker-thread-context

static ThreadContext* initialized_thread_contexts = nullptr;

#ifdef PRINT_TIMES
struct TimeMeasurements {
    unsigned long long avg_ns;
    unsigned long long max_ns;
    size_t count;
};
static __thread struct TimeMeasurements input_copy_time = {};
static __thread struct TimeMeasurements output_copy_time = {};
static __thread struct TimeMeasurements decoding_time = {};

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

#define CHECK_CUDA(call) do { \
        cudaError_t err = (call); \
        if (err) \
            printf("CUDA error %d: %s; in %s|%d|\n", (int) err, cudaGetErrorString(err), __FILE__, __LINE__); \
    } while (false)

ThreadContext& ldpc_decoder_init_context(int make_stream);

struct BaseGraph {
    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t num_edges;
    uint32_t const* cn_degree;
    uint32_t const* vn_degree;
    uint32_t const* cn;
    uint32_t const* vn;
    uint32_t cn_stride;
    uint32_t vn_stride;
};

static BaseGraph get_basegraph(uint32_t BG, uint32_t Z) {
    BaseGraph bg;
    uint32_t num_nnz;
    // select base graph dimensions
    if (BG == 1) {
        bg.num_rows = 46;
        bg.num_cols = 68;
        num_nnz = 316; // num non-zero elements in BG
    }
    else {
        bg.num_rows = 42;
        bg.num_cols = 52;
        num_nnz = 197; // num non-zero elements in BG
    }

    // number of variable nodes
    // uint32_t num_vns = bg.num_cols * Z;
    // number of check nodes
    // uint32_t num_cns = bg.num_rows * Z;

    // number of edges/messages in the graph
    bg.num_edges = num_nnz * Z;

    // lifting set according to 38.212 Tab 5.3.2-1
    static uint32_t const s_val[8][8] = {{2, 4, 8, 16, 32, 64, 128, 256},
             {3, 6, 12, 24, 48, 96, 192, 384},
             {5, 10, 20, 40, 80, 160, 320},
             {7, 14, 28, 56, 112, 224},
             {9, 18, 36, 72, 144, 288},
             {11, 22, 44, 88, 176, 352},
             {13, 26, 52, 104, 208},
             {15, 30, 60, 120, 240}};

    // find lifting set index
    int ils = -1;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (Z == s_val[i][j]) {
                ils = i;
                break;
            }
        }
        if (ils != -1)
            break;
    }
    // this case should not happen
    assert(ils != -1 && "Lifting factor not found in lifting set");

    bg.cn_degree = bg_cn_degree[BG-1][ils];
    bg.vn_degree = bg_vn_degree[BG-1][ils];
    bg.cn = bg_cn[BG-1][ils];
    bg.vn = bg_vn[BG-1][ils];
    bg.cn_stride = bg_cn_size[BG-1][ils] / (sizeof(bg.cn[0]) * bg.num_rows);
    bg.vn_stride = bg_vn_size[BG-1][ils] / (sizeof(bg.vn[0]) * bg.num_cols);

    return bg;
 }

//#define NODE_KERNEL_BLOCK 512
//#define UNROLL_NODES 1
#define NODE_KERNEL_BLOCK 128
#define UNROLL_NODES 4

// START marker-cnp-kernel
__launch_bounds__(UNROLL_NODES*NODE_KERNEL_BLOCK, 3)
static __global__ void update_cn_kernel(llr_accumulator_t const* __restrict__ llr_total, llr_msg_t* __restrict__ llr_msg,
                                        uint32_t Z, uint32_t const* __restrict__ bg_cn, uint32_t const* __restrict__ bg_cn_degree, uint32_t max_degree, uint32_t num_rows,
                                        bool first_iter) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t i = tid % Z; // for i in range(Z)
    uint32_t idx_row = tid / Z; // for idx_row in range(num_rows)

    uint32_t cn_degree = bg_cn_degree[idx_row];

    // list of tuples (idx_col = idx_vn, s),
    // idx_row = idx_cn and msg_offset omitted,
    // msg spread out to idx_cn + idx_vn * num_cn
    uint32_t const* check_nodes = &bg_cn[idx_row * max_degree]; // len(cn) = cn_degree

    // search the "extrinsic" min of all incoming LLRs
    // this means we need to find the min and the second min of all incoming LLRs
    int min_1 = INT_MAX;
    int min_2 = INT_MAX;
    int idx_min = -1;
    int node_sign = 1;
    uint32_t msg_signs = 0; // bitset, 0 == positive; max degree is 19

#if UNROLL_NODES > 1
    __shared__ int mins1[UNROLL_NODES][NODE_KERNEL_BLOCK+1];
    __shared__ int mins2[UNROLL_NODES][NODE_KERNEL_BLOCK+1];
    __shared__ int idx_mins[UNROLL_NODES][NODE_KERNEL_BLOCK+1];
    __shared__ uint32_t signs[UNROLL_NODES][NODE_KERNEL_BLOCK+1];
    mins1[threadIdx.y][threadIdx.x] = INT_MAX;
    mins2[threadIdx.y][threadIdx.x] = INT_MAX;
    signs[threadIdx.y][threadIdx.x] = 0;
#endif

    __syncwarp();

    if (idx_row < num_rows) {
    for (uint32_t ii = threadIdx.y; ii < cn_degree; ii += UNROLL_NODES) {
        uint32_t cn = check_nodes[ii];

        // see packing layout above
        uint32_t idx_col = cn & 0xffffu; // note: little endian
        uint32_t s = cn >> 16;           // ...
        uint32_t msg_offset = idx_row + idx_col * num_rows;

        uint32_t msg_idx = msg_offset * Z + i;

        // total VN message
        int t = llr_total[idx_col*Z + (i+s)%Z];

        // make extrinsic by subtracting the previous msg
        if (!first_iter)
            t -= __ldg(&llr_msg[msg_idx]);

        // store sign for 2nd recursion
        // note: could be also used for syndrome-based check or early termination
        int sign = (t >= 0 ? 1 : -1);
        node_sign *= sign;
        msg_signs |= (t < 0) << ii; // for later sign calculation

        // find min and second min
        int t_abs = abs(t);
        if (t_abs < min_1) {
            min_2 = min_1;
            min_1 = t_abs;
            idx_min = msg_idx;
        } else if (t_abs < min_2)
            min_2 = t_abs;
    }
    }

#if UNROLL_NODES > 1
	mins1[threadIdx.y][threadIdx.x] = min_1;
	mins2[threadIdx.y][threadIdx.x] = min_2;
	idx_mins[threadIdx.y][threadIdx.x] = idx_min;
	signs[threadIdx.y][threadIdx.x] = msg_signs;

    __syncthreads();

    if (threadIdx.y == 0) {
        int min_1 = INT_MAX;
        int min_2 = INT_MAX;
        int idx_min = -1;
        uint32_t msg_signs = 0; // bitset, 0 == positive; max degree is 19
        for (int i = 0; i < UNROLL_NODES; ++i) {
            int t_abs = mins1[i][threadIdx.x];
            if (t_abs < min_1) {
                min_2 = min_1;
                min_1 = t_abs;
                idx_min = idx_mins[i][threadIdx.x];
            } else if (t_abs < min_2)
                min_2 = t_abs;
            min_2 = min(min_2, mins2[i][threadIdx.x]);
            msg_signs |= signs[i][threadIdx.x];
        }
        mins1[0][threadIdx.x] = min_1;
        mins2[0][threadIdx.x] = min_2;
        idx_mins[0][threadIdx.x] = idx_min;
        signs[0][threadIdx.x] = msg_signs;
    }

    __syncthreads();

    min_1 = mins1[0][threadIdx.x];
    min_2 = mins2[0][threadIdx.x];
    idx_min = idx_mins[0][threadIdx.x];
    msg_signs = signs[0][threadIdx.x];

    node_sign = (__popc(msg_signs) & 1) ? -1 : 1;
#endif

    // START marker-cnp-damping
    // apply damping factor
    min_1 = APPLY_DAMPING_INT(min_1); // min_1 * DAMPING_FACTOR, e.g. *3/4
    min_2 = APPLY_DAMPING_INT(min_2); // min_2 * DAMPING_FACTOR, e.g. *3/4
    // END marker-cnp-damping

    // clip msg magnitudes to MAX_LLR_VALUE
    min_1 = min(max(min_1, -MAX_LLR_MSG_VALUE), MAX_LLR_MSG_VALUE);
    min_2 = min(max(min_2, -MAX_LLR_MSG_VALUE), MAX_LLR_MSG_VALUE);
    // END marker-vnp-clipping

    __syncwarp();

    // apply min and second min to the outgoing LLR
    if (idx_row < num_rows) {
    for (uint32_t ii = threadIdx.y; ii < cn_degree; ii += UNROLL_NODES) {
         uint32_t cn = check_nodes[ii];

        // see packing layout above
        uint32_t idx_col = cn & 0xffffu; // note: little endian
        uint32_t msg_offset = idx_row + idx_col * num_rows;

        uint32_t msg_idx = msg_offset * Z + i;
        int min_val;
        if (msg_idx == idx_min)
            min_val = min_2;
        else
            min_val = min_1;

        int msg_sign = (msg_signs >> ii) & 0x1 ? -1 : 1;

        // and update outgoing msg including sign
        llr_msg[msg_idx] = llr_msg_t(min_val * node_sign * msg_sign);
    }
    }
}
// END marker-cnp-kernel

// START marker-vnp-kernel
__launch_bounds__(UNROLL_NODES*NODE_KERNEL_BLOCK, 3)
static __global__ void update_vn_kernel(llr_msg_t const* __restrict__ llr_msg, int8_t const* __restrict__ llr_ch, llr_accumulator_t* __restrict__ llr_total,
                                        uint32_t Z, uint32_t const* __restrict__ bg_vn, uint32_t const* __restrict__ bg_vn_degree, uint32_t max_degree, uint32_t num_cols, uint32_t num_rows) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t i = tid % Z; // for i in range(Z)
    uint32_t idx_col = tid / Z; // for idx_col in range(num_cols)

    uint32_t vn_degree = bg_vn_degree[idx_col];

    // list of tuples (idx_row = index_cn, s)
    // idx_col = idx_vn and msg_offset omitted,
    // msg spread out to idx_cn + idx_vn * num_cn
    uint32_t const* variable_nodes = &bg_vn[idx_col * max_degree]; // len(vn) = vn_degree

#if UNROLL_NODES > 1
    __shared__ int msg_sums[UNROLL_NODES][NODE_KERNEL_BLOCK+1];
    msg_sums[threadIdx.y][threadIdx.x] = 0;
#endif

    __syncwarp();

    int msg_sum = 0;
    // accumulate all incoming LLRs
    if (idx_col < num_cols) {
    for (uint32_t j = threadIdx.y; j < vn_degree; j += UNROLL_NODES) {
        uint32_t vn = variable_nodes[j];

        // see packing layout above
        uint32_t idx_row = vn & 0xffffu; // note: little endian
        uint32_t s = vn >> 16;           // ...
        uint32_t msg_offset = idx_row + idx_col * num_rows;

        // index of the msg in the LLR array
        // it is the idx_col-th variable node, and the j-th message from the idx_row-th check node
        uint32_t msg_idx = msg_offset * Z + (i-s+(Z<<8))%Z;

        // accumulate all incoming LLRs
        msg_sum += llr_msg[msg_idx];
    }
    }

    // add the channel LLRs
    __syncwarp();
    if (threadIdx.y == 0) {
        if (idx_col < num_cols)
        msg_sum += llr_ch[idx_col*Z + i];
    }

    msg_sum = min(max(msg_sum, -MAX_LLR_ACCUMULATOR_VALUE), MAX_LLR_ACCUMULATOR_VALUE);

#if UNROLL_NODES > 1
	msg_sums[threadIdx.y][threadIdx.x] = msg_sum;

    __syncthreads();

    if (threadIdx.y == 0) {
        int msg_sum = 0;
        for (int i = 0; i < UNROLL_NODES; ++i) {
            msg_sum += msg_sums[i][threadIdx.x];
        }
        msg_sums[0][threadIdx.x] = msg_sum;
    }

    __syncthreads();

    msg_sum = msg_sums[0][threadIdx.x];
#endif

    msg_sum = min(max(msg_sum, -MAX_LLR_ACCUMULATOR_VALUE), MAX_LLR_ACCUMULATOR_VALUE);

    __syncwarp();
    if (idx_col < num_cols)
    llr_total[idx_col*Z + i] = llr_accumulator_t(msg_sum);
}
// END marker-vnp-kernel

__launch_bounds__(512, 3)
static __global__ void compute_syndrome_kernel(llr_accumulator_t const* __restrict__ llr_total, uint32_t* __restrict__ syndrome,
                                               uint32_t Z, uint32_t const* __restrict__ bg_cn, uint32_t const* __restrict__ bg_cn_degree, uint32_t max_degree, uint32_t num_rows) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t i = tid % Z; // for i in range(Z)
    uint32_t idx_row = tid / Z; // for idx_row in range(num_rows)
    if (idx_row >= num_rows) return;

    uint32_t cn_degree = bg_cn_degree[idx_row];

    // list of tuples (idx_col = idx_vn, s),
    // idx_row = idx_cn and msg_offset omitted,
    // msg spread out to idx_cn + idx_vn * num_cn
    uint32_t const* check_nodes = &bg_cn[idx_row * max_degree]; // len(cn) = cn_degree

    __syncwarp();

    uint32_t sign = 0;
    for (uint32_t ii = 0; ii < cn_degree; ++ii) {
        uint32_t cn = check_nodes[ii];

        // see packing layout above
        uint32_t idx_col = cn & 0xffffu; // note: little endian
        uint32_t s = cn >> 16;           // ...

        uint32_t t = llr_total[idx_col*Z + (i+s)%Z] < 0;
        sign = sign ^ t;
    }

    sign = __any_sync(0xffffffff, sign);
    if (threadIdx.x % 32 == 0)
        syndrome[tid / 32] = sign;
}

static const uint32_t PACK_BITS_KERNEL_THREADS = 256;

// START marker-pack-bits
__launch_bounds__(PACK_BITS_KERNEL_THREADS, 6)
static __global__ void pack_bits_kernel(llr_accumulator_t const* __restrict__ llr_total, uint8_t* __restrict__ bits, uint32_t block_length) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t coop_byte = 0;
    // 1 bit per thread
    if (tid < block_length)
        coop_byte = (llr_total[tid] < 0) << (7 - (threadIdx.x & 7)); // note: highest to lowest bit

    // use fast lane shuffles to assemble one byte per group of 8 adjacent threads
    coop_byte += __shfl_xor_sync(0xffffffff, coop_byte, 1); // xxyyzzww
    coop_byte += __shfl_xor_sync(0xffffffff, coop_byte, 2); // xxxxyyyy
    coop_byte += __shfl_xor_sync(0xffffffff, coop_byte, 4); // xxxxxxxx

    // share bytes across thread group to allow one coalesced write by first N threads
    __shared__ uint32_t bit_block_shared[PACK_BITS_KERNEL_THREADS / 8];
    if ((threadIdx.x & 0x7) == 0)
        bit_block_shared[threadIdx.x / 8] = coop_byte;

    __syncthreads();

    // the first (PACK_BITS_KERNEL_THREADS / 8) threads pack 8 bits each
    if (threadIdx.x < PACK_BITS_KERNEL_THREADS / 8 && blockIdx.x * PACK_BITS_KERNEL_THREADS + threadIdx.x * 8 < block_length) {
        bits[blockIdx.x * PACK_BITS_KERNEL_THREADS / 8 + threadIdx.x] = bit_block_shared[threadIdx.x];
    }
}
// END marker-pack-bits

extern "C" uint32_t ldpc_decode(ThreadContext* context_, cudaStream_t stream, uint32_t BG, uint32_t Z,
                                int8_t const* llr_in, uint32_t block_length,
                                uint8_t* llr_bits, uint32_t num_iter,
                                uint32_t perform_syndrome_check) {
    auto& context = context_ ? *context_ : ldpc_decoder_init_context(0);
    if (context_ && stream == 0)
        stream = context_->stream;

    BaseGraph bg = get_basegraph(BG, Z);

    const uint32_t num_llrs = bg.num_cols * Z;
    const uint32_t num_cn = bg.num_rows * Z;
    const uint32_t num_out_bytes = blocks_for(block_length, 8);

#ifdef PRINT_TIMES
    struct timespec ts_begin, ts_cursor, ts_end;
    unsigned long long time_ns;
    size_t message_count;

    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_begin );
#endif

#ifdef USE_GRAPHS
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
#endif

    int8_t const *mapped_llr_in = context.llr_in_buffer;
    // START marker-copy-input
    // copy input data to device-visible memory
#ifdef USE_UNIFIED_MEMORY
    memcpy(const_cast<int8_t*>(mapped_llr_in), llr_in, num_llrs * sizeof(*llr_in));
#else
    CHECK_CUDA(cudaMemcpyAsync(const_cast<int8_t*>(mapped_llr_in), llr_in, num_llrs * sizeof(*llr_in), cudaMemcpyHostToDevice, stream));
#endif
    // END marker-copy-input

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

    int8_t const* llr_total = mapped_llr_in;

    for (uint32_t i = 0; i < num_iter; ++i) {
        dim3 threads(NODE_KERNEL_BLOCK, UNROLL_NODES);

        // check node update
        dim3 blocks_cn(blocks_for(bg.num_rows * Z, threads.x));
        // note: llr_msg not not read, only written to in first iteration; will be filled with outputs of this function
        update_cn_kernel<<<blocks_cn, threads, 0, stream>>>(
            llr_total, context.llr_msg_buffer,
            Z, bg.cn, bg.cn_degree, bg.cn_stride, bg.num_rows, i==0);

        // variable node update
        dim3 blocks_vn(blocks_for(bg.num_cols * Z, threads.x));
        // note: llr_total only written to
        update_vn_kernel<<<blocks_vn, threads, 0, stream>>>(
            context.llr_msg_buffer, mapped_llr_in, context.llr_total_buffer,
            Z, bg.vn, bg.vn_degree, bg.vn_stride, bg.num_cols, bg.num_rows);
        llr_total = context.llr_total_buffer;
    }

    uint8_t *mapped_llr_bits_out = context.llr_bits_out_buffer;

    // pack bits
    dim3 threads_pack(PACK_BITS_KERNEL_THREADS);
    dim3 blocks_pack(blocks_for(block_length, threads_pack.x));
    pack_bits_kernel<<<blocks_pack, threads_pack, 0, stream>>>(
        llr_total, mapped_llr_bits_out, block_length);
#ifndef USE_UNIFIED_MEMORY
    CHECK_CUDA(cudaMemcpyAsync(llr_bits, mapped_llr_bits_out, num_out_bytes, cudaMemcpyDeviceToHost, stream));
#endif

    // allow CPU access of output bits while computing syndrome
#if defined(USE_UNIFIED_MEMORY) && !defined(USE_GRAPHS)
    cudaStreamSynchronize(stream);
#endif

#ifdef PRINT_TIMES
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_cursor );
#endif

    // check syndrome if additional testing is requested
    if (perform_syndrome_check) {
        dim3 threads(512);
        dim3 blocks_cn(blocks_for(num_cn, threads.x));
        compute_syndrome_kernel<<<blocks_cn, threads, 0, stream>>>(
            context.llr_total_buffer, context.syndrome_buffer,
            Z, bg.cn, bg.cn_degree, bg.cn_stride, bg.num_rows);
#ifndef USE_UNIFIED_MEMORY
      CHECK_CUDA(cudaMemcpyAsync(context.host_syndrome_buffer, context.syndrome_buffer, num_cn * sizeof(*context.syndrome_buffer) / 32, cudaMemcpyDeviceToHost, stream));
#endif
    }

#ifdef USE_GRAPHS
    cudaGraph_t graphUpdate = {};
    CHECK_CUDA(cudaStreamEndCapture(stream, &graphUpdate));
    if (context.graphCtx) {
        cudaGraphNode_t errorNode;
        cudaGraphExecUpdateResult updateResult;
        CHECK_CUDA(cudaGraphExecUpdate(context.graphCtx, graphUpdate, &errorNode, &updateResult));
    }
    else
         CHECK_CUDA(cudaGraphInstantiate(&context.graphCtx, graphUpdate, 0));
    cudaGraphDestroy(graphUpdate);
    CHECK_CUDA(cudaGraphLaunch(context.graphCtx, stream));
#endif

#if !defined(USE_UNIFIED_MEMORY) || defined(USE_GRAPHS)
    // allow CPU access of output bits and syndrome
    cudaStreamSynchronize(stream);
#endif

#ifdef USE_UNIFIED_MEMORY
    // note: GPU synchronized before async syndrome check
    memcpy(llr_bits, mapped_llr_bits_out, num_out_bytes);
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

    if (perform_syndrome_check) {
      uint32_t* p_syndrome;
#ifdef USE_UNIFIED_MEMORY
    #ifndef USE_GRAPHS
      // allow reading syndrome
      cudaStreamSynchronize(stream);
    #endif
      p_syndrome = context.syndrome_buffer;
#else
      // note: already synchronized above
      p_syndrome = context.host_syndrome_buffer;
#endif

      // check any errors indicated by syndrome
      for (uint32_t i = 0; i < num_cn / 32; i++) {
          if (p_syndrome[i] != 0) {
              return num_iter+1;
          }
      }
    }

#ifdef PRINT_TIMES
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_end );

    time_ns = ts_end.tv_nsec - ts_begin.tv_nsec + 1000000000ll * (ts_end.tv_sec - ts_begin.tv_sec);
    message_count = add_measurement(&decoding_time, time_ns, 500);
    if (message_count % 500 == 0) {
      time_ns = decoding_time.avg_ns;
      printf("CUDA sync runtime: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
      fflush(stdout);
    }
#endif

    return num_iter-1; // note: now index of successful iteration
}

ThreadContext& ldpc_decoder_init_context(int make_stream) {
    auto& context = thread_context;
    if (context.llr_in_buffer) // lazy
        return context;

    printf("Initializing LDPC context (TID %d)\n", (int) gettid());

    if (make_stream) {
        int highPriority = 0;
        if (cudaDeviceGetStreamPriorityRange(NULL, &highPriority))
            printf("CUDA stream priorities unsupported, %s:%d", __FILE__, __LINE__);
        CHECK_CUDA(cudaStreamCreateWithPriority(&context.stream, cudaStreamNonBlocking, highPriority));

        cudaStreamAttrValue attr = {};
        attr.syncPolicy = cudaSyncPolicyYield;
        cudaStreamSetAttribute(context.stream, cudaStreamAttributeSynchronizationPolicy, &attr);
    }

    CHECK_CUDA(cudaMallocStaging(&context.llr_in_buffer, MAX_BG_COLS * MAX_Z * sizeof(int8_t), cudaHostAllocMapped | cudaHostAllocWriteCombined));
    CHECK_CUDA(cudaMallocStaging(&context.llr_bits_out_buffer, (MAX_BLOCK_LENGTH + 7) / 8 * sizeof(uint8_t), cudaHostAllocMapped));
    CHECK_CUDA(cudaMallocStaging(&context.syndrome_buffer, MAX_BG_ROWS * MAX_Z * sizeof(uint32_t) / 32, cudaHostAllocMapped));
#ifndef USE_UNIFIED_MEMORY
    context.host_syndrome_buffer = (uint8_t*) malloc(MAX_BG_ROWS * MAX_Z * sizeof(uint8_t));
#endif
    CHECK_CUDA(cudaMalloc(&context.llr_msg_buffer, MAX_BG_ROWS * MAX_BG_COLS * MAX_Z * sizeof(llr_msg_t)));
    CHECK_CUDA(cudaMalloc(&context.llr_total_buffer, MAX_BG_COLS * MAX_Z * sizeof(llr_accumulator_t)));

    // keep track of active thread contexts for shutdown
    ThreadContext* self = &context;
    __atomic_exchange(&initialized_thread_contexts, &self, &self->next_initialized_context, __ATOMIC_ACQ_REL);

    return context;
}

extern "C" ThreadContext* ldpc_decoder_init(int make_stream) {
    if (bg_cn[0][0])  // lazy, global
        return &ldpc_decoder_init_context(make_stream);

    printf("Initializing LDPC runtime %d\n", (int) gettid());

    const uint32_t* table_bg_cn_degree[2][8] = { { BG1_CN_DEGREE_TABLE() }, { BG2_CN_DEGREE_TABLE() } };
    const uint32_t* table_bg_vn_degree[2][8] = { { BG1_VN_DEGREE_TABLE() }, { BG2_VN_DEGREE_TABLE() } };
    const uint32_t table_bg_cn_degree_size[2][8] = { { BG1_CN_DEGREE_TABLE(sizeof) }, { BG2_CN_DEGREE_TABLE(sizeof) } };
    const uint32_t table_bg_vn_degree_size[2][8] = { { BG1_VN_DEGREE_TABLE(sizeof) }, { BG2_VN_DEGREE_TABLE(sizeof) } };
    const void* table_bg_cn[2][8] = { { BG1_CN_TABLE() }, { BG2_CN_TABLE() } };
    const void* table_bg_vn[2][8] = { { BG1_VN_TABLE() }, { BG2_VN_TABLE() } };
    const uint32_t table_bg_cn_size[2][8] = { { BG1_CN_TABLE(sizeof) }, { BG2_CN_TABLE(sizeof) } };
    const uint32_t table_bg_vn_size[2][8] = { { BG1_VN_TABLE(sizeof) }, { BG2_VN_TABLE(sizeof) } };

    for (int b = 0; b < 2; ++b) {
        for (int ils = 0; ils < 8; ++ils) {
            CHECK_CUDA(cudaMalloc(&bg_cn_degree[b][ils], table_bg_cn_degree_size[b][ils]));
            CHECK_CUDA(cudaMemcpy(const_cast<uint32_t*>(bg_cn_degree[b][ils]), table_bg_cn_degree[b][ils], table_bg_cn_degree_size[b][ils], cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&bg_vn_degree[b][ils], table_bg_vn_degree_size[b][ils]));
            CHECK_CUDA(cudaMemcpy(const_cast<uint32_t*>(bg_vn_degree[b][ils]), table_bg_vn_degree[b][ils], table_bg_vn_degree_size[b][ils], cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc(&bg_cn[b][ils], table_bg_cn_size[b][ils]));
            CHECK_CUDA(cudaMemcpy(const_cast<uint32_t*>(bg_cn[b][ils]), table_bg_cn[b][ils], table_bg_cn_size[b][ils], cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMalloc(&bg_vn[b][ils], table_bg_vn_size[b][ils]));
            CHECK_CUDA(cudaMemcpy(const_cast<uint32_t*>(bg_vn[b][ils]), table_bg_vn[b][ils], table_bg_vn_size[b][ils], cudaMemcpyHostToDevice));

            bg_cn_size[b][ils] = table_bg_cn_size[b][ils];
            bg_vn_size[b][ils] = table_bg_vn_size[b][ils];
        }
    }

    return &ldpc_decoder_init_context(make_stream);
}

extern "C" void ldpc_decoder_shutdown() {
    cudaDeviceSynchronize();

    ThreadContext* active_context = nullptr;
    __atomic_exchange(&initialized_thread_contexts, &active_context, &active_context, __ATOMIC_ACQ_REL);
    while (active_context) {
        cudaFreeStaging(active_context->llr_in_buffer);
        cudaFree(active_context->llr_msg_buffer);
        cudaFreeStaging(active_context->llr_bits_out_buffer);
        cudaFree(active_context->llr_total_buffer);
        cudaFreeStaging(active_context->syndrome_buffer);
#ifndef USE_UNIFIED_MEMORY
        free(active_context->host_syndrome_buffer);
#endif
        if (active_context->stream)
            cudaStreamDestroy(active_context->stream);

#ifdef USE_GRAPHS
        cudaGraphExecDestroy(active_context->graphCtx);
#endif

        active_context = active_context->next_initialized_context;
    }

    for (int b = 0; b < 2; ++b) {
        for (int ils = 0; ils < 8; ++ils) {
            cudaFree(&bg_cn_degree[b][ils]);
            cudaFree(&bg_vn_degree[b][ils]);
            cudaFree(&bg_cn[b][ils]);
            cudaFree(&bg_vn[b][ils]);
        }
    }
}

#ifdef ENABLE_NANOBIND

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

NB_MODULE(ldpc_decoder, m) {
    m.def("decode", [](uint32_t BG, uint32_t Z,
                       const nb::ndarray<int8_t, nb::shape<-1>, nb::device::cpu>& llrs,
                       uint32_t block_length, uint32_t num_iter) {
        auto* context = ldpc_decoder_init(1); // lazy

        size_t num_bytes = (block_length + 7) / 8 * 8;
        uint8_t *data = new uint8_t[num_bytes];
        memset(data, 0, num_bytes);
        nb::capsule owner(data, [](void *p) noexcept { delete[] (uint8_t*) p; });

        ldpc_decode(context, 0, BG, Z, llrs.data(),
                    block_length, data,
                    num_iter, true);

        return nb::ndarray<nb::numpy, uint8_t>(data, {num_bytes}, owner);
    });
}

#endif
