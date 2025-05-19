/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include "NvInfer.h"
#include "data_processing.h"
#include <cuda_fp16.h>
#include <vector>
#include <cstdio>
#include <unistd.h>
#include <time.h>

using namespace nvinfer1;

static IRuntime* runtime = nullptr;
static ICudaEngine* engine = nullptr;

static uint32_t const OPT_BLOCK_LEN = 64;
static uint32_t const MAX_BLOCK_LEN = 512;
static uint32_t const MAX_BITS_PER_SYMBOL = 16;

#define PERSISTENT_DEVICE_MEMORY
#define USE_UNIFIED_MEMORY
#define USE_GRAPHS
// #define PRINT_TIMES

struct TRTContext {
    cudaStream_t default_stream = 0;
    IExecutionContext* trt = nullptr;
    void* prealloc_memory = nullptr;
    cudaGraph_t graph_opt = nullptr, graph_max = nullptr;
    cudaGraphExec_t record_opt = nullptr, record_max = nullptr;
    __half* input_buffer = nullptr;
    __half* output_buffer = nullptr;
#ifdef PERSISTENT_DEVICE_MEMORY
    int16_t* symbol_buffer= nullptr;
    int16_t* magnitude_buffer = nullptr;
    int16_t* llr_buffer = nullptr;
#endif

    // list of thread contexts for shutdown
    TRTContext* next_initialized_context = nullptr;
};
static __thread TRTContext* thread_context = nullptr;
static TRTContext* initialized_thread_contexts = nullptr;

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
    if (err) printf("CUDA error %d: %s; in %s|%d|: %s\n", (int) err, cudaGetErrorString(err), __FILE__, __LINE__, #call); } while (false)

#define PRINT_INFO_VERBOSE(...) // printf(__VA_ARGS__)

struct Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            printf("TensorRT %s: %s\n", severity == Severity::kWARNING ? "WARNING" : "ERROR", msg);
    }
} logger;

static char const* trt_weight_file = "models/neural_demapper_qam16_2.plan";
static bool trt_normalized_inputs = true;

extern "C" void trt_demapper_configure(char const* weight_file, int normalized_inputs) {
    trt_weight_file = weight_file;
    trt_normalized_inputs = (bool) normalized_inputs;
}

TRTContext& trt_demapper_init_context(int make_stream);

void trt_demapper_run(TRTContext* context_, cudaStream_t stream, __half const* inputs, size_t numInputs, size_t numInputComponents, __half* outputs) {
    auto& context = context_ ? *context_ : trt_demapper_init_context(0);
    if (context_ && stream == 0)
        stream = context_->default_stream;

    if (inputs) {
        context.trt->setTensorAddress("y", (void*) inputs);
        context.trt->setInputShape("y", Dims2(numInputs, numInputComponents));
    }
    if (outputs) {
        context.trt->setTensorAddress("output_1", outputs);
    }
    context.trt->enqueueV3(stream);
}

#define TIMESTAMP_CLOCK_SOURCE CLOCK_MONOTONIC

extern "C" void trt_demapper_decode_block(TRTContext* context_, cudaStream_t stream, int16_t const* in_symbols, int16_t const* in_mags, size_t num_symbols,
                                          int16_t const *mapped_symbols, int16_t const *mapped_mags, size_t num_batch_symbols,
                                          int16_t* outputs, uint32_t num_bits_per_symbol, int16_t* mapped_outputs) {
    auto& context = *context_;

    uint32_t block_size = num_batch_symbols > OPT_BLOCK_LEN ? MAX_BLOCK_LEN : OPT_BLOCK_LEN;
    cudaGraph_t& graph = block_size == OPT_BLOCK_LEN ? context.graph_opt : context.graph_max;
    cudaGraphExec_t& graphCtx = block_size == OPT_BLOCK_LEN ? context.record_opt : context.record_max;

#if defined(PERSISTENT_DEVICE_MEMORY) && defined(USE_UNIFIED_MEMORY)
    struct timespec ts_begin, ts_end;
    unsigned long long time_ns;
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_begin );

    if (num_symbols > 0) {
        memcpy((void*) mapped_symbols, in_symbols, sizeof(*in_symbols) * 2 * num_symbols);
        memcpy((void*) mapped_mags, in_mags, sizeof(*in_mags) * 2 * num_symbols);
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

#if defined(PERSISTENT_DEVICE_MEMORY) && !defined(USE_UNIFIED_MEMORY)
    cudaMemcpyAsync((void*) mapped_symbols, in_symbols, sizeof(*in_symbols) * 2 * num_symbols, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync((void*) mapped_mags, in_mags, sizeof(*in_mags) * 2 * num_symbols, cudaMemcpyHostToDevice, stream);
#endif

    // graph capture
    if (!graph) {
        bool recording = false;
#ifdef USE_GRAPHS
        // allow pre-allocation before recording
        if (num_symbols > 0) {
            CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
            num_batch_symbols = block_size;
            recording = true;
        }
#endif

        size_t num_in_components;
        if (trt_normalized_inputs) {
            norm_int16_symbols_to_float16(stream, mapped_symbols, mapped_mags, num_batch_symbols,
                                          (uint16_t*) context.input_buffer, 1);
            num_in_components = 2;
        }
        else {
            int16_symbols_to_float16(stream, mapped_symbols, num_batch_symbols,
                                     (uint16_t*) context.input_buffer, 2);
            int16_symbols_to_float16(stream, mapped_mags, num_batch_symbols,
                                     (uint16_t*) context.input_buffer + 2, 2);
            num_in_components = 4;
        }

        trt_demapper_run(&context, stream, recording ? nullptr : context.input_buffer, block_size, num_in_components, recording ? nullptr : context.output_buffer);

        float16_llrs_to_int16(stream, (uint16_t const*) context.output_buffer, num_batch_symbols,
                              mapped_outputs, num_bits_per_symbol);

#ifdef USE_GRAPHS
        if (num_symbols > 0) {
            CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
            PRINT_INFO_VERBOSE("Recorded CUDA graph (TID %d), stream %llX\n", (int) gettid(), (unsigned long long) stream);
        }
#endif
    }

#ifdef USE_GRAPHS
    if (graph && !graphCtx) {
        CHECK_CUDA(cudaGraphInstantiate(&graphCtx, graph, 0));
        PRINT_INFO_VERBOSE("Instantiated CUDA graph (TID %d)\n", (int) gettid());
    }
    else if (num_symbols > 0) {
        cudaGraphLaunch(graphCtx, stream);
    }
#endif

#if defined(PERSISTENT_DEVICE_MEMORY) && !defined(USE_UNIFIED_MEMORY)
    cudaMemcpyAsync(outputs, mapped_outputs, sizeof(*outputs) * num_bits_per_symbol * num_symbols, cudaMemcpyDeviceToHost, stream);
#endif

#if defined(PERSISTENT_DEVICE_MEMORY) && defined(USE_UNIFIED_MEMORY)
    CHECK_CUDA(cudaStreamSynchronize(stream));
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_begin );
    memcpy(outputs, mapped_outputs, sizeof(*outputs) * num_bits_per_symbol * num_symbols);
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

extern "C" void trt_demapper_decode(TRTContext* context_, cudaStream_t stream, int16_t const* in_symbols, int16_t const* in_mags, size_t num_symbols,
                                    int16_t* outputs, uint32_t num_bits_per_symbol) {
    auto& context = context_ ? *context_ : trt_demapper_init_context(0);
    if (context_ && stream == 0)
        stream = context_->default_stream;

    struct timespec ts_begin, ts_end;
    unsigned long long time_ns;
#ifdef PERSISTENT_DEVICE_MEMORY
    int16_t const *mapped_symbols = context.symbol_buffer;
    int16_t const *mapped_mags = context.magnitude_buffer;
    int16_t *mapped_outputs = context.llr_buffer;
#else
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_begin );

    cudaHostRegister((void*) in_symbols, sizeof(*in_symbols) * 2 * num_symbols, cudaHostRegisterDefault);
    int16_t const *mapped_symbols = nullptr;
    cudaHostGetDevicePointer((void**) &mapped_symbols, (void*) in_symbols, 0);
    cudaHostRegister((void*) in_mags, sizeof(*in_mags) * 2 * num_symbols, cudaHostRegisterDefault);
    int16_t const *mapped_mags = nullptr;
    cudaHostGetDevicePointer((void**) &mapped_mags, (void*) in_mags, 0);

    cudaHostRegister(outputs, sizeof(*outputs) * num_bits_per_symbol * num_symbols, cudaHostRegisterDefault);
    int16_t *mapped_outputs = nullptr;
    cudaHostGetDevicePointer((void**) &mapped_outputs, (void*) outputs, 0);

    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_end );

#ifdef PRINT_TIMES
    time_ns = ts_end.tv_nsec - ts_begin.tv_nsec + 1000000000ll * (ts_end.tv_sec - ts_begin.tv_sec);
    if (cuda_time.count % 500 == 499) {
      printf("CUDA mapping runtime: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
      fflush(stdout);
    }
#endif
#endif

    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_begin );

    for (size_t offset = 0; offset < num_symbols; offset += MAX_BLOCK_LEN) {
        uint32_t num_batch_symbols = (uint32_t) std::min(num_symbols - offset, (size_t) MAX_BLOCK_LEN);

        size_t map_offset = offset;
#ifdef PERSISTENT_DEVICE_MEMORY
        map_offset = 0;
#endif

        trt_demapper_decode_block(&context, stream, &in_symbols[2 * offset], &in_mags[2 * offset], num_batch_symbols,
                                  &mapped_symbols[2 * map_offset], &mapped_mags[2 * map_offset], num_batch_symbols,
                                  &outputs[num_bits_per_symbol * offset], num_bits_per_symbol, &mapped_outputs[num_bits_per_symbol * map_offset]);
    }

#if !(defined(PERSISTENT_DEVICE_MEMORY) && defined(USE_UNIFIED_MEMORY))
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

#ifndef PERSISTENT_DEVICE_MEMORY
    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_begin );

    cudaHostUnregister((void*) in_symbols);
    cudaHostUnregister((void*) in_mags);
    cudaHostUnregister(outputs);

    clock_gettime( TIMESTAMP_CLOCK_SOURCE, &ts_end );

#ifdef PRINT_TIMES
    time_ns = ts_end.tv_nsec - ts_begin.tv_nsec + 1000000000ll * (ts_end.tv_sec - ts_begin.tv_sec);
    if (message_count % 500 == 499) {
      printf("CUDA unmap runtime: %llu us %llu ns\n", time_ns / 1000, time_ns - time_ns / 1000 * 1000);
      fflush(stdout);
    }
#endif
#endif
}

TRTContext& trt_demapper_init_context(int make_stream) {
    if (!thread_context)
        thread_context = new TRTContext();
    auto& context = *thread_context;
    if (context.trt) // lazy
        return context;

    printf("Initializing TRT context (TID %d)\n", (int) gettid());
#if NV_TENSORRT_MAJOR >= 10
    context.trt = engine->createExecutionContext(ExecutionContextAllocationStrategy::kSTATIC);
#else
    context.trt = engine->createExecutionContextWithoutDeviceMemory();
    size_t preallocSize = engine->getDeviceMemorySize();
    cudaError_t preallocStatus = cudaMalloc(&context.prealloc_memory, preallocSize);
    PRINT_INFO_VERBOSE("Prealloc result %d for size %llu Kb\n", (int) preallocStatus, (unsigned long long) preallocSize / 1024);
    if (0 == preallocStatus && context.prealloc_memory)
        context.trt->setDeviceMemory(context.prealloc_memory);
#endif

    if (make_stream)
        CHECK_CUDA(cudaStreamCreateWithFlags(&context.default_stream, cudaStreamNonBlocking));

    cudaMalloc((void**) &context.input_buffer, sizeof(*context.input_buffer) * 4 * MAX_BLOCK_LEN);
    cudaMalloc((void**) &context.output_buffer, sizeof(*context.output_buffer) * MAX_BITS_PER_SYMBOL * MAX_BLOCK_LEN);

#ifdef PERSISTENT_DEVICE_MEMORY
  #ifdef USE_UNIFIED_MEMORY
    #define DEVICE_IO_ALLOC cudaHostAlloc // cudaMallocManaged(p, s, cudaMemAttachHost)
  #else
    #define DEVICE_IO_ALLOC(p, s, f) cudaMalloc(p, s)
  #endif
    DEVICE_IO_ALLOC((void**) &context.symbol_buffer, sizeof(*context.symbol_buffer) * 2 * MAX_BLOCK_LEN, cudaHostAllocMapped | cudaHostAllocWriteCombined);
    DEVICE_IO_ALLOC((void**) &context.magnitude_buffer, sizeof(*context.magnitude_buffer) * 2 * MAX_BLOCK_LEN, cudaHostAllocMapped | cudaHostAllocWriteCombined);
    DEVICE_IO_ALLOC((void**) &context.llr_buffer, sizeof(*context.llr_buffer) * MAX_BITS_PER_SYMBOL * MAX_BLOCK_LEN, cudaHostAllocMapped);
  #undef DEVICE_IO_ALLOC
#endif

// START marker-record-graph
#ifdef USE_GRAPHS
    // record graphs for optimal and max block size
    for (int i = 0; i < 2; ++i) {
        int16_t in_symbols[2], in_mags[2], outputs[MAX_BITS_PER_SYMBOL];
        unsigned num_batch_symbols = i == 0 ? MAX_BLOCK_LEN : OPT_BLOCK_LEN;
        unsigned num_bits_per_symbol = 4;

        // pre-allocate, then record
        for (int a = 0; a < 2; ++a) {
            trt_demapper_decode_block(&context, context.default_stream, in_symbols, in_mags, a,
                                      context.symbol_buffer, context.magnitude_buffer, num_batch_symbols,
                                      outputs, num_bits_per_symbol, context.llr_buffer);
        }
    }
#endif
// END marker-record-graph

    // keep track of active thread contexts for shutdown
    TRTContext* self = &context;
    __atomic_exchange(&initialized_thread_contexts, &self, &self->next_initialized_context, __ATOMIC_ACQ_REL);

    return context;
}

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

extern "C" void trt_demapper_shutdown();

extern "C" TRTContext* trt_demapper_init(int make_stream) {
    if (runtime)  // lazy, global
        return &trt_demapper_init_context(make_stream);

    printf("Initializing TRT runtime %d\n", (int) gettid());
    runtime = createInferRuntime(logger);
    printf("Loading TRT engine %s (normalized inputs: %d)\n", trt_weight_file, trt_normalized_inputs);
    std::vector<char> modelData = readModelFromFile(trt_weight_file);
    engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());

#ifdef ENABLE_NANOBIND
    // automatic shutdown before TensorRT module is torn down (note: in initialization sequence after createInferRuntime!)
    static struct AutoTRTShutdown {
        ~AutoTRTShutdown() {
            trt_demapper_shutdown();
        }
    } trt_shutdown_guard;
#endif

    return &trt_demapper_init_context(make_stream);
}

extern "C" void trt_demapper_shutdown() {
    TRTContext* active_context = nullptr;
    __atomic_exchange(&initialized_thread_contexts, &active_context, &active_context, __ATOMIC_ACQ_REL);
    while (active_context) {
#ifdef USE_GRAPHS
        cudaGraphExecDestroy(active_context->record_opt);
        cudaGraphDestroy(active_context->graph_opt);
        cudaGraphExecDestroy(active_context->record_max);
        cudaGraphDestroy(active_context->graph_max);
#endif
#if NV_TENSORRT_MAJOR >= 10
        delete active_context->trt;
#else
        if(active_context->trt)
            active_context->trt->destroy();
#endif

        cudaFree(active_context->prealloc_memory);
        cudaFree(active_context->input_buffer);
        cudaFree(active_context->output_buffer);
#ifdef PERSISTENT_DEVICE_MEMORY
        cudaFree(active_context->symbol_buffer);
        cudaFree(active_context->magnitude_buffer);
        cudaFree(active_context->llr_buffer);
#endif
        if (active_context->default_stream)
            cudaStreamDestroy(active_context->default_stream);

        TRTContext* next_context = active_context->next_initialized_context;
        delete active_context;
        active_context = next_context;
    }

#if NV_TENSORRT_MAJOR >= 10
    delete engine;
    delete runtime;
#else
    if (engine)
        engine->destroy();
    if (runtime)
        runtime->destroy();
#endif
    engine = nullptr;
    runtime = nullptr;
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

NB_MODULE(trt_demapper, m) {
    m.def("run_qam", [](const nb::ndarray<__half, nb::shape<-1, -1>, nb::device::cpu>& symbols_and_magnitudes, uint32_t bits_per_symbol) {
        auto* context = trt_demapper_init(1); // lazy

        __half *data = nullptr;
        size_t num_llrs = symbols_and_magnitudes.shape(0) * bits_per_symbol;
        cudaMallocManaged((void**) &data, std::max(sizeof(*data) * num_llrs, (size_t) 16));
        memset(data, 0, sizeof(*data) * num_llrs);
        nb::capsule owner(data, [](void *p) noexcept { cudaFree(p); });

        cudaHostRegister(symbols_and_magnitudes.data(), sizeof(*symbols_and_magnitudes.data()) * symbols_and_magnitudes.size(), cudaHostRegisterDefault);
        __half const *mappedData = nullptr;
        cudaHostGetDevicePointer((void**) &mappedData, (void*) symbols_and_magnitudes.data(), 0);
        trt_demapper_run(context, 0, mappedData, symbols_and_magnitudes.shape(0), trt_normalized_inputs ? 2 : 4, data);
        cudaStreamSynchronize(context->default_stream);
        cudaHostUnregister(symbols_and_magnitudes.data());

        return nb::ndarray<nb::numpy, __half, nb::ndim<2>>(data, {symbols_and_magnitudes.shape(0), bits_per_symbol}, owner);
    });
    m.def("decode_qam", [](const nb::ndarray<int16_t, nb::shape<-1, 2>, nb::device::cpu>& symbols,
                           const nb::ndarray<int16_t, nb::shape<-1, 2>, nb::device::cpu>& magnitudes,
                           uint32_t bits_per_symbol) {
        auto* context = trt_demapper_init(1); // lazy

        size_t num_llrs = symbols.shape(0) * bits_per_symbol;
        int16_t *data = new int16_t[num_llrs];
        memset(data, 0, sizeof(*data) * num_llrs);
        nb::capsule owner(data, [](void *p) noexcept { delete[] (int16_t*) p; });

        trt_demapper_decode(context, 0, symbols.data(), magnitudes.data(), symbols.shape(0),
                            data, bits_per_symbol);

        return nb::ndarray<nb::numpy, int16_t, nb::ndim<2>>(data, {symbols.shape(0), bits_per_symbol}, owner);
    });
}

#endif
