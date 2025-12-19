#include "NvInfer.h"
#include "data_processing.h"
#include <cuda_fp16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <alloca.h>
#include <string>

using namespace nvinfer1;

struct Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            printf("%s\n", msg);
    }
} logger;

std::vector<char> readModelFromFile(char const* filepath) {
    std::vector<char> bytes;
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        logger.log(Logger::Severity::kERROR, filepath);
        return bytes;
    }
    fseek(f, 0, SEEK_END);
    bytes.resize((size_t) ftell(f));
    fseek(f, 0, SEEK_SET);
    if (bytes.size() != fread(bytes.data(), 1, bytes.size(), f))
        logger.log(Logger::Severity::kWARNING, filepath);
    fclose(f);
    return bytes;
}

static IRuntime* runtime = nullptr;
static ICudaEngine* engine = nullptr;

static uint32_t const MAX_BATCH_SIZE = 1;
//static uint32_t const OPT_BLOCK_LEN = 96;
static uint32_t const MAX_BLOCK_LEN = 288;
static uint32_t const OPT_BLOCK_LEN = MAX_BLOCK_LEN;
static uint32_t const MAX_BITS_PER_SYMBOL = 4;

#define FORCE_HOST_COPY
#ifdef ENABLE_DGX_OPTIMIZATIONS
#define USE_GRAPHS
#endif
// #define PRINT_TIMES

struct GraphWithNodes {
    cudaGraph_t graph = nullptr;
    cudaGraphNode_t input_nodes[4] = { 0 }; // 2x convert and reshape
    cudaKernelNodeParams input_params[4] = { 0 }; // 2x convert and reshape
    cudaGraphNode_t output_nodes[4] = { 0 }; // 2x convert and reshape
    cudaKernelNodeParams output_params[3] = { 0 }; // 2x convert and reshape
};

struct TRTContext {
    cudaStream_t default_stream = 0;
    IExecutionContext* trt = nullptr;
    GraphWithNodes graph_opt, graph_max;
    cudaGraphExec_t record_opt = nullptr, record_max = nullptr;
    __half* input_buffer = nullptr;
    __half* input_h_buffer = nullptr;
    __half* batch_buffer = nullptr;
    __half* batch_h_buffer = nullptr;
    __half* output_buffer = nullptr;
    int16_t* decode_buffer = nullptr;

    int16_t* symbol_buffer= nullptr;
    int16_t* h_buffer = nullptr;
    int32_t* aux_buffer = nullptr;
    int16_t* llr_buffer = nullptr;
    //int16_t* ref_buffer = nullptr;
};
static __thread TRTContext thread_context = { };

#ifdef PRINT_TIMES
struct TimeMeasurements {
    unsigned long long total_ns;
    unsigned long long max_ns;
    unsigned count;
};
static __thread struct TimeMeasurements cuda_time;

static unsigned add_measurement(struct TimeMeasurements& time, unsigned long long time_ns) {
    time.total_ns += time_ns;
    if (time_ns > time.max_ns)
        time.max_ns = time_ns;
    return time.count++;
}
#endif

#define CHECK_CUDA(call) do { cudaError_t err = (call); \
    if (err) printf("CUDA error %d: %s; in %s|%d|\n", (int) err, cudaGetErrorString(err), __FILE__, __LINE__); } while (false);

static char const* trt_weight_file = "plugins/neural_receiver/models/nrx_oai.plan";

static size_t max_num_ofdm_symbols = 13;
static size_t max_num_antenna = 1;
static size_t max_num_tx = 1;
static size_t max_num_dmrs_symbols = 3;
static size_t max_num_pilots_per_prb = 6 * max_num_dmrs_symbols;

extern "C" void trt_receiver_configure(char const* weight_file) {
    trt_weight_file = weight_file;
}

TRTContext& trt_receiver_init_context(int make_stream);

void trt_receiver_run(TRTContext* context_, cudaStream_t stream,
                      size_t batch_size, __half const* in_active_ports, size_t num_tx,
                      __half const* in_symbols, size_t num_subcarriers, size_t num_ofdm_symbols, size_t num_antenna,
                      __half const* in_h, size_t num_pilots,
                      int32_t const* in_dmrs_ofdm_pos, size_t num_dmrs_symbols, // num_tx x num_dmrs_symbols
                      int32_t const* in_dmrs_subcarrier_pos, size_t num_pilots_per_prb,
                      __half* outputs, uint32_t num_bits_per_symbol) {
    auto& context = context_ ? *context_ : trt_receiver_init_context(0);
    if (context_ && stream == 0)
        stream = context_->default_stream;
    if (in_symbols) {
        context.trt->setTensorAddress("rx_slot", (void*) in_symbols);
        Dims symbol_dims;
        symbol_dims.nbDims = 5;
        symbol_dims.d[0] = 1;
        symbol_dims.d[1] = num_subcarriers;
        symbol_dims.d[2] = num_ofdm_symbols;
        symbol_dims.d[3] = num_antenna;
        symbol_dims.d[4] = 2;
        context.trt->setInputShape("rx_slot", symbol_dims);

        context.trt->setTensorAddress("h_hat", (void*) in_h);
        Dims h_dims = symbol_dims;
        h_dims.d[1] = num_pilots;
        h_dims.d[2] = num_tx;
        context.trt->setInputShape("h_hat", h_dims);

        context.trt->setTensorAddress("active_dmrs_ports", (void*) in_active_ports);
        context.trt->setInputShape("active_dmrs_ports", Dims2(1, num_tx));
        context.trt->setTensorAddress("dmrs_ofdm_pos", (void*) in_dmrs_ofdm_pos);
        context.trt->setInputShape("dmrs_ofdm_pos", Dims2(num_tx, num_dmrs_symbols));
        context.trt->setTensorAddress("dmrs_subcarrier_pos", (void*) in_dmrs_subcarrier_pos);
        context.trt->setInputShape("dmrs_subcarrier_pos", Dims2(num_tx, num_pilots_per_prb / num_dmrs_symbols));
    }
    if (outputs) {
        context.trt->setTensorAddress("output_1", outputs);
    }
    context.trt->enqueueV3(stream);
}

#define TIMESTAMP_CLOCK_SOURCE CLOCK_MONOTONIC

static inline void memcpy_0ext(void* target, const void* source, size_t source_size, size_t ext_size) {
    if (source)
        memcpy(target, source, source_size);
    size_t zero_offset = source_size;
    size_t aligned_offset = (zero_offset + 3) / 4 * 4;
    if (aligned_offset != zero_offset && aligned_offset < ext_size) {
        memset((char*) target + zero_offset, 0, aligned_offset - zero_offset);
        zero_offset = aligned_offset;
    }
    memset((char*) target + zero_offset, 0, ext_size - zero_offset);
}

extern "C" void trt_receiver_decode_block(TRTContext* context_, cudaStream_t stream,
                                          size_t batch_size, int16_t const* in_active_ports, size_t num_tx,
                                          int16_t const* in_symbols, size_t num_subcarriers, size_t num_ofdm_symbols, size_t num_antenna,
                                          float norm_scale,
                                          int16_t const* in_h, size_t num_pilots,
                                          int32_t const* in_dmrs_ofdm_pos, size_t num_dmrs_symbols, // num_tx x num_dmrs_symbols
                                          int32_t const* in_dmrs_subcarrier_pos, size_t num_pilots_per_prb,
                                          size_t num_batch_subcarriers,
                                          int16_t* outputs, uint32_t num_bits_per_symbol) {
    auto& context = *context_;

    uint32_t block_size = num_batch_subcarriers > OPT_BLOCK_LEN ? MAX_BLOCK_LEN : OPT_BLOCK_LEN;
    uint32_t pilot_block_size = block_size * max_num_pilots_per_prb / 12;
    GraphWithNodes& graph_cfg = block_size == OPT_BLOCK_LEN ? context.graph_opt : context.graph_max;
    cudaGraphExec_t& graphCtx = block_size == OPT_BLOCK_LEN ? context.record_opt : context.record_max;

    size_t num_batch_pilots = (num_subcarriers > 0) ? num_batch_subcarriers * num_pilots / num_subcarriers : num_batch_subcarriers * max_num_pilots_per_prb / 12;
    num_subcarriers = std::min(num_subcarriers, num_batch_subcarriers);
    num_pilots = std::min(num_pilots, num_batch_pilots);

#if defined(FORCE_HOST_COPY)
    struct timespec ts_begin, ts_end;
    unsigned long long time_ns;
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_begin );

    auto* mapped_symbols = context.symbol_buffer;
    auto* mapped_h = context.h_buffer;
    auto* mapped_port_mask = (int16_t*) context.aux_buffer;
    auto* mapped_ofdm_pos = (int32_t*) (mapped_port_mask + (batch_size * max_num_tx + 1) / 2 * 2);
    auto* mapped_subcarrier_pos = mapped_ofdm_pos + max_num_tx * max_num_dmrs_symbols;
    auto* mapped_llrs = context.llr_buffer;
    //auto* mapped_refs = context.ref_buffer;
    if (num_subcarriers > 0) {
        memcpy_0ext((void*) mapped_symbols, in_symbols
                  , sizeof(*in_symbols) * 2 * batch_size * num_subcarriers * num_ofdm_symbols * num_antenna
                  , sizeof(*in_symbols) * 2 * batch_size * num_batch_subcarriers * num_ofdm_symbols * num_antenna);
        memcpy_0ext((void*) mapped_h, in_h
                  , sizeof(*in_h) * 2 * batch_size * num_pilots * num_tx * num_antenna
                  , sizeof(*in_h) * 2 * batch_size * num_batch_pilots * num_tx * num_antenna);

        for (uint32_t i = 0; i < batch_size * num_tx; ++i)
            mapped_port_mask[i] = in_active_ports[i] ? 0x3c00 : 0; // half float 1.0
        memcpy_0ext(mapped_port_mask, nullptr
                  , (batch_size * num_tx) * sizeof(*mapped_port_mask)
                  , (batch_size * max_num_tx + 1) / 2 * 2 * sizeof(*mapped_port_mask));

        memcpy_0ext((void*) mapped_ofdm_pos, in_dmrs_ofdm_pos
                  , sizeof(*in_dmrs_ofdm_pos) * num_tx * num_dmrs_symbols
                  , sizeof(*in_dmrs_ofdm_pos) * max_num_tx * max_num_dmrs_symbols);
        memcpy_0ext((void*) mapped_subcarrier_pos, in_dmrs_subcarrier_pos
                    , sizeof(*in_dmrs_subcarrier_pos) * num_tx * num_pilots_per_prb / num_dmrs_symbols
                    , sizeof(*in_dmrs_subcarrier_pos) * max_num_tx * max_num_pilots_per_prb / max_num_dmrs_symbols);
    }

    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_end );
#ifdef PRINT_TIMES
    time_ns = ts_end.tv_nsec - ts_begin.tv_nsec + 1000000000ll * (ts_end.tv_sec - ts_begin.tv_sec);
    if (cuda_time.count % 500 == 499) {
      printf("CUDA input copy runtime: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
      fflush(stdout);
    }
#endif
#endif

#if !defined(FORCE_HOST_COPY)
    cudaMemcpyAsync((void*) mapped_symbols, in_symbols, sizeof(*in_symbols) * 2 * num_symbols, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync((void*) mapped_mags, in_mags, sizeof(*in_mags) * 2 * num_symbols, cudaMemcpyHostToDevice, stream);
#endif

    // graph capture
    if (!graph_cfg.graph) {
        bool recording = false;
#ifdef USE_GRAPHS
        // allow pre-allocation before recording
        if (num_subcarriers > 0) {
            CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
            recording = true;
        }
#endif

        {
            int16_symbols_to_float16(stream, mapped_symbols, batch_size * num_batch_subcarriers * num_ofdm_symbols * num_antenna, norm_scale,
                                     (uint16_t*) context.input_buffer, 1);
            int16_symbols_to_float16(stream, mapped_h, batch_size * num_batch_pilots * num_tx * num_antenna, norm_scale,
                                     (uint16_t*) context.input_h_buffer, 1);

            // note: only supports batch_size == 1 currently
            assert(batch_size == 1);
            reshape_and_pad_32bit(stream, (uint32_t*) context.input_buffer, batch_size, num_batch_subcarriers, num_ofdm_symbols, num_antenna,
                                  (uint32_t*) context.batch_buffer, batch_size, block_size, max_num_ofdm_symbols, max_num_antenna);
            reshape_and_pad_32bit(stream, (uint32_t*) context.input_h_buffer, batch_size*max_num_dmrs_symbols, num_batch_pilots/max_num_dmrs_symbols, num_tx, num_antenna,
                                  (uint32_t*) context.batch_h_buffer, batch_size*max_num_dmrs_symbols, pilot_block_size/max_num_dmrs_symbols, max_num_tx, max_num_antenna);
        }

        trt_receiver_run(&context, stream, batch_size, (__half*) mapped_port_mask, max_num_tx,
                         context.batch_buffer, block_size, max_num_ofdm_symbols, max_num_antenna,
                         context.batch_h_buffer, pilot_block_size,
                         mapped_ofdm_pos, max_num_dmrs_symbols, // num_tx x num_dmrs_symbols
                         mapped_subcarrier_pos, max_num_pilots_per_prb,
                         context.output_buffer, num_bits_per_symbol);

        {
            // note: only supports batch_size == 1 currently
            assert(batch_size == 1);
            assert(max_num_tx == num_tx);
            gather_transposed_llrs(stream, (int16_t const*) context.output_buffer, max_num_tx * block_size, max_num_ofdm_symbols,
                                   (int16_t*) context.decode_buffer, num_tx * num_batch_subcarriers, num_ofdm_symbols, num_bits_per_symbol);

            float16_llrs_to_int16(stream, (uint16_t const*) context.decode_buffer, batch_size * num_tx * num_batch_subcarriers * num_ofdm_symbols,
                                  mapped_llrs, num_bits_per_symbol);
        }

#ifdef USE_GRAPHS
        if (recording) {
            CHECK_CUDA(cudaStreamEndCapture(stream, &graph_cfg.graph));
            printf("Recorded CUDA graph with code (TID %d), stream %llX\n", (int) gettid(), (unsigned long long) stream);

            // collect all nodes
            size_t num_nodes = 0;
            CHECK_CUDA(cudaGraphGetNodes(graph_cfg.graph, nullptr, &num_nodes));
            auto all_nodes = (cudaGraphNode_t*) alloca(sizeof(cudaGraphNode_t) * num_nodes);
            CHECK_CUDA(cudaGraphGetNodes(graph_cfg.graph, all_nodes, &num_nodes));

            // collect input and output nodes
            for (unsigned int i = 0; i < 4; ++i) {
                graph_cfg.input_nodes[i] = all_nodes[i];

                cudaGraphNodeType node_t = cudaGraphNodeTypeCount;
                CHECK_CUDA(cudaGraphNodeGetType(graph_cfg.input_nodes[i], &node_t));
                assert(node_t == cudaGraphNodeTypeKernel);
                CHECK_CUDA(cudaGraphKernelNodeGetParams(graph_cfg.input_nodes[i], &graph_cfg.input_params[i]));
            }
            for (unsigned int i = 0; i < 2; ++i) {
                graph_cfg.output_nodes[i] = all_nodes[num_nodes-1-i];

                cudaGraphNodeType node_t = cudaGraphNodeTypeCount;
                CHECK_CUDA(cudaGraphNodeGetType(graph_cfg.output_nodes[i], &node_t));
                assert(node_t == cudaGraphNodeTypeKernel);
                CHECK_CUDA(cudaGraphKernelNodeGetParams(graph_cfg.output_nodes[i], &graph_cfg.output_params[i]));
            }
        }
#endif
    }

#ifdef USE_GRAPHS
    if (graph_cfg.graph && !graphCtx) {
        auto res = cudaGraphInstantiate(&graphCtx, graph_cfg.graph, 0);
        printf("Instantiated CUDA graph with code %d (TID %d)\n", (int) res, (int) gettid());
    }
    if (num_subcarriers > 0) {
        // batch sizes and norms
        *((unsigned*) graph_cfg.input_params[0].kernelParams[1]) = batch_size * num_batch_subcarriers * num_ofdm_symbols * num_antenna;
        *((float*) graph_cfg.input_params[0].kernelParams[2]) = norm_scale;
        *((unsigned*) graph_cfg.input_params[1].kernelParams[1]) = batch_size * num_batch_pilots * num_tx * num_antenna;
        *((float*) graph_cfg.input_params[1].kernelParams[2]) = norm_scale;
        // batch dimensions for reshape
        *((unsigned*) graph_cfg.input_params[2].kernelParams[1]) = batch_size;
        *((unsigned*) graph_cfg.input_params[2].kernelParams[2]) = num_batch_subcarriers;
        *((unsigned*) graph_cfg.input_params[2].kernelParams[3]) = num_ofdm_symbols;
        *((unsigned*) graph_cfg.input_params[2].kernelParams[4]) = num_antenna;
        *((unsigned*) graph_cfg.input_params[3].kernelParams[1]) = batch_size*max_num_dmrs_symbols;
        *((unsigned*) graph_cfg.input_params[3].kernelParams[2]) = num_batch_pilots/max_num_dmrs_symbols;
        *((unsigned*) graph_cfg.input_params[3].kernelParams[3]) = num_tx;
        *((unsigned*) graph_cfg.input_params[3].kernelParams[4]) = num_antenna;

        for (unsigned int i = 0; i < 4; ++i)
            cudaGraphExecKernelNodeSetParams(graphCtx, graph_cfg.input_nodes[i], &graph_cfg.input_params[i]);

        // output reshape and convert
        *((unsigned*) graph_cfg.output_params[1].kernelParams[4]) = num_tx * num_batch_subcarriers;
        *((unsigned*) graph_cfg.output_params[1].kernelParams[5]) = num_ofdm_symbols;
        *((unsigned*) graph_cfg.output_params[0].kernelParams[1]) = batch_size * num_tx * num_batch_subcarriers * num_ofdm_symbols * num_bits_per_symbol;

        for (unsigned int i = 0; i < 2; ++i)
            cudaGraphExecKernelNodeSetParams(graphCtx, graph_cfg.output_nodes[i], &graph_cfg.output_params[i]);

        cudaGraphLaunch(graphCtx, stream);
    }
#endif

#if !defined(FORCE_HOST_COPY)
    cudaMemcpyAsync(outputs, mapped_llrs, sizeof(*outputs) * batch_size * num_bits_per_symbol * num_tx * num_subcarriers * num_ofdm_symbols, cudaMemcpyDeviceToHost, stream);
#endif

#if defined(FORCE_HOST_COPY)
    CHECK_CUDA(cudaStreamSynchronize(stream));
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_begin );
    memcpy(outputs, mapped_llrs, sizeof(*outputs) * batch_size * num_tx * num_subcarriers * num_ofdm_symbols * num_bits_per_symbol);
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_end );
#ifdef PRINT_TIMES
    time_ns = ts_end.tv_nsec - ts_begin.tv_nsec + 1000000000ll * (ts_end.tv_sec - ts_begin.tv_sec);
    if (cuda_time.count % 500 == 499) {
      printf("CUDA output copy runtime: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
      fflush(stdout);
    }
#endif
#endif
}

extern "C" void trt_receiver_decode(TRTContext* context_, cudaStream_t stream, int16_t const* in_active_ports, size_t num_tx,
                                          int16_t const* in_symbols, size_t num_subcarriers, size_t num_ofdm_symbols, size_t num_antenna,
                                          float norm_scale,
                                          int16_t const* in_h, size_t num_pilots,
                                          int32_t const* in_dmrs_ofdm_pos, size_t num_dmrs_symbols, // num_tx x num_dmrs_symbols
                                          int32_t const* in_dmrs_subcarrier_pos, size_t num_pilots_per_prb,
                                          int16_t* outputs, uint32_t num_bits_per_symbol) {
    auto& context = context_ ? *context_ : trt_receiver_init_context(0);
    if (context_ && stream == 0)
        stream = context_->default_stream;

    struct timespec ts_begin, ts_end;
    unsigned long long time_ns;

    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_begin );

    for (size_t offset = 0; offset < num_subcarriers; offset += MAX_BLOCK_LEN) {
        uint32_t num_batch_subcarriers = (uint32_t) std::min(num_subcarriers - offset, (size_t) MAX_BLOCK_LEN);

        size_t symbol_offset = 2 * num_antenna * num_ofdm_symbols * offset;
        size_t pilot_offset = symbol_offset * num_pilots / num_subcarriers * num_tx / num_ofdm_symbols;
        size_t output_offset = num_tx * num_ofdm_symbols * num_bits_per_symbol * offset;

        trt_receiver_decode_block(&context, stream,
                                  1, in_active_ports, num_tx, // todo: convert port mask
                                  &in_symbols[symbol_offset], num_subcarriers, num_ofdm_symbols, num_antenna,
                                  norm_scale,
                                  &in_h[pilot_offset], num_pilots,
                                  in_dmrs_ofdm_pos, num_dmrs_symbols, // num_tx x num_dmrs_symbols
                                  in_dmrs_subcarrier_pos, num_pilots_per_prb,
                                  num_batch_subcarriers,
                                  &outputs[output_offset], num_bits_per_symbol);
    }

#if !(defined(FORCE_HOST_COPY))
    // synchronize if not already done for host copy
    cudaStreamSynchronize(stream);
#endif

    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_end );

#ifdef PRINT_TIMES
    time_ns = ts_end.tv_nsec - ts_begin.tv_nsec + 1000000000ll * (ts_end.tv_sec - ts_begin.tv_sec);
    unsigned message_count = add_measurement(cuda_time, time_ns);
    if (message_count % 500 == 499) {
      time_ns = cuda_time.total_ns / cuda_time.count;
      printf("CUDA sync runtime: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
      fflush(stdout);
      memset(&cuda_time, 0, sizeof(cuda_time));
    }
#endif
}

TRTContext& trt_receiver_init_context(int make_stream) {
    auto& context = thread_context;
    printf("Requesting TRT context %d\n", (int) gettid());
    if (context.trt) // lazy
        return context;

    printf("Initializing TRT context (TID %d)\n", (int) gettid());
    context.trt = engine->createExecutionContextWithoutDeviceMemory();
    size_t preallocSize = engine->getDeviceMemorySize();
    void* preallocMem;
    printf("Prealloc result %d for size %llu Kb\n", (int) cudaMalloc(&preallocMem, preallocSize), (unsigned long long) preallocSize / 1024);
    context.trt->setDeviceMemory(preallocMem);

    if (make_stream) {
        int highPriority = 0;
        if (cudaDeviceGetStreamPriorityRange(NULL, &highPriority))
            printf("CUDA stream priorities unsupported, %s:%d", __FILE__, __LINE__);
        CHECK_CUDA(cudaStreamCreateWithPriority(&context.default_stream, cudaStreamNonBlocking, highPriority));

        cudaStreamAttrValue attr = {};
        attr.syncPolicy = cudaSyncPolicyYield;
        cudaStreamSetAttribute(context.default_stream, cudaStreamAttributeSynchronizationPolicy, &attr);
    }

    cudaMalloc((void**) &context.input_buffer, sizeof(*context.input_buffer) * 2 * MAX_BATCH_SIZE * MAX_BLOCK_LEN * max_num_ofdm_symbols * max_num_antenna);
    cudaMalloc((void**) &context.input_h_buffer, sizeof(*context.input_h_buffer) * 2 * MAX_BATCH_SIZE * MAX_BLOCK_LEN/12 * max_num_pilots_per_prb * max_num_tx * max_num_antenna);
    cudaMalloc((void**) &context.batch_buffer, sizeof(*context.batch_buffer) * 2 * MAX_BATCH_SIZE * MAX_BLOCK_LEN * max_num_ofdm_symbols * max_num_antenna);
    cudaMalloc((void**) &context.batch_h_buffer, sizeof(*context.batch_h_buffer) * 2 * MAX_BATCH_SIZE * MAX_BLOCK_LEN/12 * max_num_pilots_per_prb * max_num_tx * max_num_antenna);
    cudaMalloc((void**) &context.output_buffer, sizeof(*context.output_buffer) * MAX_BITS_PER_SYMBOL * MAX_BATCH_SIZE * MAX_BLOCK_LEN * max_num_tx * max_num_ofdm_symbols);
    cudaMalloc((void**) &context.decode_buffer, sizeof(*context.decode_buffer) * MAX_BITS_PER_SYMBOL * MAX_BATCH_SIZE * MAX_BLOCK_LEN * max_num_tx * max_num_ofdm_symbols);

  #ifdef FORCE_HOST_COPY
    #define DEVICE_IO_ALLOC cudaHostAlloc // cudaMallocManaged(p, s, cudaMemAttachHost)
  #else
    #define DEVICE_IO_ALLOC(p, s, f) cudaMalloc(p, s)
  #endif
    DEVICE_IO_ALLOC((void**) &context.symbol_buffer, sizeof(*context.symbol_buffer) * 2 * MAX_BATCH_SIZE * MAX_BLOCK_LEN * max_num_ofdm_symbols * max_num_antenna, cudaHostAllocMapped | cudaHostAllocWriteCombined);
    DEVICE_IO_ALLOC((void**) &context.h_buffer, sizeof(*context.h_buffer) * 2 * MAX_BATCH_SIZE * MAX_BLOCK_LEN/12 * max_num_pilots_per_prb * max_num_ofdm_symbols * max_num_antenna, cudaHostAllocMapped | cudaHostAllocWriteCombined);
    DEVICE_IO_ALLOC((void**) &context.aux_buffer, sizeof(*context.aux_buffer) * max_num_tx * (MAX_BATCH_SIZE + max_num_dmrs_symbols + max_num_pilots_per_prb / max_num_dmrs_symbols), cudaHostAllocMapped | cudaHostAllocWriteCombined);
    DEVICE_IO_ALLOC((void**) &context.llr_buffer, sizeof(*context.llr_buffer) * MAX_BITS_PER_SYMBOL * MAX_BATCH_SIZE * MAX_BLOCK_LEN * max_num_tx * max_num_ofdm_symbols, cudaHostAllocMapped);
    //DEVICE_IO_ALLOC((void**) &context.ref_buffer, sizeof(*context.ref_buffer) * 2 * MAX_BLOCK_LEN * max_num_tx * max_num_ofdm_symbols * max_num_antenna, cudaHostAllocMapped);
  #undef DEVICE_IO_ALLOC

#ifdef USE_GRAPHS
    // record graphs for optimal and max block size
    for (int i = 0; i < 1+(MAX_BLOCK_LEN!=OPT_BLOCK_LEN); ++i) {
        int16_t in_symbols[MAX_BATCH_SIZE * MAX_BLOCK_LEN * 14 * 16] = {};
        int16_t in_h[MAX_BATCH_SIZE * MAX_BLOCK_LEN * 4 * 2 * 16] = {};
        int16_t outputs[MAX_BATCH_SIZE * MAX_BITS_PER_SYMBOL * MAX_BLOCK_LEN * 14 * 2] = {};
        int16_t in_active_ports[2 * 2] = { 1, 1 };
        int32_t in_dmrs_ofdm_pos[2 * 6] = { 0, 2, 4, 6, 8, 10 };
        int32_t in_dmrs_subcarrier_pos[2 * 7] = { 0, 2, 4, 6, 8, 10, 12 };
        size_t num_batch_symbols = i == 0 ? OPT_BLOCK_LEN : MAX_BLOCK_LEN;
        size_t num_bits_per_symbol = 4;

        // pre-allocate, then record
        for (int a = 0; a < 50; ++a) { // run the graph a couple of times to trigger re-optimizers
            trt_receiver_decode_block(&context, context.default_stream,
                                      1, in_active_ports, 1,
                                      in_symbols, a * num_batch_symbols, max_num_ofdm_symbols, max_num_antenna,
                                      1.0f,
                                      in_h, a * num_batch_symbols * max_num_pilots_per_prb / 12,
                                      in_dmrs_ofdm_pos, max_num_dmrs_symbols,
                                      in_dmrs_subcarrier_pos, max_num_pilots_per_prb / max_num_dmrs_symbols,
                                      num_batch_symbols,
                                      outputs, num_bits_per_symbol);
        }
    }
#endif
    return context;
}

extern "C" TRTContext* trt_receiver_init(int make_stream) {
    if (runtime)  // lazy, global
        return &trt_receiver_init_context(make_stream);

    printf("Initializing TRT runtime %d\n", (int) gettid());
    runtime = createInferRuntime(logger);
    
    // Check PLAN_POSTFIX environment variable to use alternative plan file
    // PLAN_POSTFIX defaults to empty, can be set to ".host" or other value
    std::string weight_file = trt_weight_file;
    const char* plan_postfix = std::getenv("PLAN_POSTFIX");
    if (plan_postfix && plan_postfix[0] != '\0') {
        // Insert postfix before .plan extension
        size_t pos = weight_file.rfind(".plan");
        if (pos != std::string::npos) {
            weight_file = weight_file.substr(0, pos) + plan_postfix + ".plan";
        }
    }
    
    printf("Loading TRT engine %s\n", weight_file.c_str());
    std::vector<char> modelData = readModelFromFile(weight_file.c_str());
    engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());

    return &trt_receiver_init_context(make_stream);
}

#ifdef ENABLE_NANOBIND

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace nanobind::detail {
    template <> struct dtype_traits<__half> {
        static constexpr dlpack::dtype value {
            (uint8_t) dlpack::dtype_code::Float, // type code
            16, // size in bits
            1   // lanes (simd), usually set to 1
        };
        static constexpr auto name = const_name("float16");
    };
}

NB_MODULE(trt_receiver, m) {
    m.def("run_nrx", [](const nb::ndarray<__half, nb::shape<1, -1, -1, -1, 2>, nb::device::cpu>& symbols,
                        const nb::ndarray<__half, nb::shape<1, -1, -1, -1, 2>, nb::device::cpu>& h,
                        const nb::ndarray<__half, nb::shape<1, -1>, nb::device::cpu>& active_ports,
                        const nb::ndarray<int32_t, nb::shape<-1, -1>, nb::device::cpu>& dmrs_pos,
                        const nb::ndarray<int32_t, nb::shape<-1, -1>, nb::device::cpu>& subcarrier_pos,
                        uint32_t bits_per_symbol) {
        auto* context = trt_receiver_init(1); // lazy

        __half *data = nullptr;
        size_t num_llrs = symbols.shape(0) * symbols.shape(1) * symbols.shape(2) * bits_per_symbol * active_ports.shape(1);
        cudaMallocManaged((void**) &data, std::max(sizeof(*data) * num_llrs, (size_t) 16));
        memset(data, 0, sizeof(*data) * num_llrs);
        nb::capsule owner(data, [](void *p) noexcept { cudaFree(p); });

        cudaHostRegister(symbols.data(), sizeof(*symbols.data()) * symbols.size(), cudaHostRegisterDefault);
        __half const *mappedData = nullptr;
        cudaHostGetDevicePointer((void**) &mappedData, (void*) symbols.data(), 0);

        cudaHostRegister(h.data(), sizeof(*h.data()) * h.size(), cudaHostRegisterDefault);
        __half const *mappedH = nullptr;
        cudaHostGetDevicePointer((void**) &mappedH, (void*) h.data(), 0);

        cudaHostRegister(active_ports.data(), sizeof(*active_ports.data()) * active_ports.size(), cudaHostRegisterDefault);
        __half const *mappedPorts = nullptr;
        cudaHostGetDevicePointer((void**) &mappedPorts, (void*) active_ports.data(), 0);

        cudaHostRegister(dmrs_pos.data(), sizeof(*dmrs_pos.data()) * dmrs_pos.size(), cudaHostRegisterDefault);
        int32_t const *mappedDMRSPos = nullptr;
        cudaHostGetDevicePointer((void**) &mappedDMRSPos, (void*) dmrs_pos.data(), 0);

        cudaHostRegister(subcarrier_pos.data(), sizeof(*subcarrier_pos.data()) * subcarrier_pos.size(), cudaHostRegisterDefault);
        int32_t const *mappedSubcarrierPos = nullptr;
        cudaHostGetDevicePointer((void**) &mappedSubcarrierPos, (void*) subcarrier_pos.data(), 0);

        trt_receiver_run(context, 0,
                         1, mappedPorts, active_ports.shape(1),
                         mappedData, symbols.shape(1), symbols.shape(2), symbols.shape(3),
                         mappedH, h.shape(1),
                         mappedDMRSPos, dmrs_pos.shape(1),
                         mappedSubcarrierPos, subcarrier_pos.shape(1) * dmrs_pos.shape(1),
                         data, bits_per_symbol);
        cudaStreamSynchronize(context->default_stream);

        cudaHostUnregister(symbols.data());
        cudaHostUnregister(h.data());
        cudaHostUnregister(active_ports.data());
        cudaHostUnregister(dmrs_pos.data());
        cudaHostUnregister(subcarrier_pos.data());

        return nb::ndarray<nb::numpy, __half>(data,
            {symbols.shape(0), bits_per_symbol, active_ports.shape(1), symbols.shape(1), symbols.shape(2)}, owner);
    });
    m.def("decode", [](const nb::ndarray<int16_t, nb::shape<1, -1, -1, -1, 2>, nb::device::cpu>& symbols,
                       const nb::ndarray<int16_t, nb::shape<1, -1, -1, -1, 2>, nb::device::cpu>& h,
                       float norm_scale,
                       const nb::ndarray<int16_t, nb::shape<1, -1>, nb::device::cpu>& active_ports,
                       const nb::ndarray<int32_t, nb::shape<-1, -1>, nb::device::cpu>& dmrs_pos,
                       const nb::ndarray<int32_t, nb::shape<-1, -1>, nb::device::cpu>& subcarrier_pos,
                       uint32_t bits_per_symbol) {
        auto* context = trt_receiver_init(1); // lazy

        int16_t *data = nullptr;
        size_t num_llrs = symbols.shape(0) * symbols.shape(1) * symbols.shape(2) * bits_per_symbol * active_ports.shape(1);
        cudaMallocManaged((void**) &data, std::max(sizeof(*data) * num_llrs, (size_t) 16));
        memset(data, 0, sizeof(*data) * num_llrs);
        nb::capsule owner(data, [](void *p) noexcept { cudaFree(p); });

        trt_receiver_decode(context, 0,
                            active_ports.data(), active_ports.shape(1),
                            symbols.data(), symbols.shape(1), symbols.shape(2), symbols.shape(3),
                            norm_scale,
                            h.data(), h.shape(1),
                            dmrs_pos.data(), dmrs_pos.shape(1),
                            subcarrier_pos.data(), subcarrier_pos.shape(1) * dmrs_pos.shape(1),
                            data, bits_per_symbol);

        return nb::ndarray<nb::numpy, int16_t>(data,
            {symbols.shape(0), active_ports.shape(1) * symbols.shape(1), symbols.shape(2), bits_per_symbol}, owner);
    });
}

#endif
