#include "data_processing.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cassert>

inline __host__ __device__ size_t blocks_for(size_t elements, int block_size) {
    return int( size_t(elements + (block_size-1)) / size_t(block_size) );
}
inline __host__ __device__ uint32_t blocks_for(uint32_t elements, int block_size) {
    return int( uint32_t(elements + (block_size-1)) / uint32_t(block_size) );
}

inline __host__ __device__ bool is_po2(uint32_t n) {
    return (n & (n-1)) == 0 && n != 0;
}

__global__ void
//__launch_bounds__(512, 3)
int16_symbols_to_float16_kernel(
    const int16_t* __restrict__ symbols_i,
    uint32_t num_symbols,
    float scale,
    __half2* __restrict symbols_h,
    uint32_t output_int32_stride
    ) {
    uint32_t globalIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (globalIdx >= num_symbols)
        return;

    uint32_t symbolBits = reinterpret_cast<const uint32_t*>(symbols_i)[globalIdx];
    int16_t s_r = int16_t(uint16_t(symbolBits & 0xffff)); // note: little endian
    int16_t s_i = int16_t(uint16_t(symbolBits >> 16));    // ...

    float2 sf;
    sf.x = ldexpf(float(s_r), -8) * scale;
    sf.y = ldexpf(float(s_i), -8) * scale;
    symbols_h[globalIdx * output_int32_stride] = __float22half2_rn(sf);
}

void int16_symbols_to_float16(
    cudaStream_t stream,
    const int16_t* symbols_i,
    uint32_t num_symbols,
    float scale,
    uint16_t* symbols_h,
    uint32_t output_int32_stride
    ) {
    dim3 threads(256);
    dim3 blocks(blocks_for(num_symbols, threads.x));

    int16_symbols_to_float16_kernel<<<blocks, threads, 0, stream>>>(
        symbols_i,
        num_symbols,
        scale,
        reinterpret_cast<__half2*>(symbols_h),
        output_int32_stride
    );
}


__global__ void
//__launch_bounds__(512, 3)
reshape_and_pad_32bit_kernel(
    const uint32_t* __restrict__ inputs,
    uint32_t num_dim0,
    uint32_t num_dim1,
    uint32_t num_dim2,
    uint32_t num_dim3,
    uint32_t* __restrict outputs,
    uint32_t max_num_dim0,
    uint32_t max_num_dim1,
    uint32_t max_num_dim2,
    uint32_t max_num_dim3
    ) {
    uint32_t dim3Idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t dim2Idx = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t dim1Idx = threadIdx.z + blockDim.z * blockIdx.z;

    uint32_t dim0Idx = dim2Idx / max_num_dim2;
    dim2Idx -= dim0Idx * max_num_dim2;

    if (dim0Idx >= max_num_dim0 || dim1Idx >= max_num_dim1 /*|| dim2Idx >= max_num_dim2*/ || dim3Idx >= max_num_dim3)
        return;

    uint32_t globalIdx = ((dim0Idx * max_num_dim1 + dim1Idx) * max_num_dim2 + dim2Idx) * max_num_dim3 + dim3Idx;

    // clamp 0, 3 / circular 1, 2
#if 1
    if (dim0Idx >= num_dim0)
        dim0Idx = num_dim0-1;
    uint32_t reflect_count1 = max(num_dim1-1, 1);
    uint32_t odd_dim1_repetition = (dim1Idx / reflect_count1) & 0x1;
    dim1Idx %= reflect_count1;
    dim2Idx %= num_dim2;
    if (dim3Idx >= num_dim3)
        dim3Idx = num_dim3-1;
#endif
    //reflect 1
#if 1
    if (odd_dim1_repetition)
        dim1Idx = num_dim1 - 1 - dim1Idx;
#endif

    uint32_t val = 0;
    if (dim0Idx < num_dim0 && dim1Idx < num_dim1 && dim2Idx < num_dim2 && dim3Idx < num_dim3) {
        uint32_t inputIdx = ((dim0Idx * num_dim1 + dim1Idx) * num_dim2 + dim2Idx) * num_dim3 + dim3Idx;
        val = inputs[inputIdx];
    }

    outputs[globalIdx] = val;
}

void reshape_and_pad_32bit(
    cudaStream_t stream,
    const uint32_t* inputs,
    uint32_t num_dim0,
    uint32_t num_dim1,
    uint32_t num_dim2,
    uint32_t num_dim3,
    uint32_t* outputs,
    uint32_t max_num_dim0,
    uint32_t max_num_dim1,
    uint32_t max_num_dim2,
    uint32_t max_num_dim3
    ) {
    dim3 threads(1, 4, 64);
    dim3 blocks(blocks_for(max_num_dim3, threads.x), blocks_for(max_num_dim2*max_num_dim0, threads.y), blocks_for(max_num_dim1, threads.z));

    assert(num_dim0 > 0);
    assert(num_dim1 > 0);
    assert(num_dim2 > 0);
    assert(num_dim3 > 0);
    reshape_and_pad_32bit_kernel<<<blocks, threads, 0, stream>>>(
        inputs,
        num_dim0, num_dim1, num_dim2, num_dim3,
        outputs,
        max_num_dim0, max_num_dim1, max_num_dim2, max_num_dim3
    );
}

// START marker-quantize-llrs
__global__ void
float16_llrs_to_int16_kernel(
    __half const* __restrict llrs_h,
    uint32_t num_llrs,
    int16_t* __restrict__ llrs_i
    ) {
    uint32_t globalIdx = threadIdx.x + blockDim.x * blockIdx.x;

    float2 tuple = {};
    if (2 * globalIdx + 1 < num_llrs)
        tuple = __half22float2( reinterpret_cast<const __half2*>(llrs_h)[globalIdx] );
    else if (2 * globalIdx < num_llrs)
        tuple.x = llrs_h[2 * globalIdx];
    else
        return;

    int16_t s1 = int16_t(__float2int_rn(ldexpf(tuple.x, 8)));
    int16_t s2 = int16_t(__float2int_rn(ldexpf(tuple.y, 8)));

    if (2 * globalIdx + 1 < num_llrs)
        reinterpret_cast<uint32_t*>(llrs_i)[globalIdx] = (uint32_t(s2 & 0xffffu) << 16) + uint32_t(s1 & 0xffffu); // note: little endian
    else
        llrs_i[2 * globalIdx] = s1;
}

void float16_llrs_to_int16(
    cudaStream_t stream,
    uint16_t const* llrs_h,
    uint32_t num_symbols,
    int16_t* llrs_i,
    uint32_t num_bits
    ) {

    dim3 threads(256);
    dim3 blocks(blocks_for(blocks_for(num_symbols * num_bits, 2u), threads.x));

    float16_llrs_to_int16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half const*>(llrs_h),
        num_symbols * num_bits,
        llrs_i
    );
}
// END marker-quantize-llrs

// START marker-gather-llrs
__global__ void
gather_transposed_llrs_kernel(
    int16_t const* __restrict input,
    uint32_t max_num_symbols,
    uint32_t max_num_ofdm_symbols,
    int16_t* __restrict__ output,
    uint32_t num_symbols,
    uint32_t num_ofdm_symbols,
    uint32_t num_bits_per_symbol
    ) {
    uint32_t globalIdx = threadIdx.x + blockDim.x * blockIdx.x;

    if (globalIdx >= num_symbols * num_ofdm_symbols)
        return;

    uint32_t globalInIdx = globalIdx + globalIdx / num_ofdm_symbols * (max_num_ofdm_symbols - num_ofdm_symbols);

    int16_t llrs[8];
    #pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        if (i < num_bits_per_symbol)
            llrs[i] = input[globalInIdx + i * max_num_symbols * max_num_ofdm_symbols];
    }

    // todo: warp shuffle transpose for coalesced writes

    #pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        if (i < num_bits_per_symbol)
            output[globalIdx * num_bits_per_symbol + i] = llrs[i];
    }
}

void gather_transposed_llrs(
    cudaStream_t stream,
    int16_t const* inputs,
    uint32_t max_num_symbols,
    uint32_t max_num_ofdm_symbols,
    int16_t* outputs,
    uint32_t num_symbols,
    uint32_t num_ofdm_symbols,
    uint32_t num_bits
    ) {

    dim3 threads(256);
    dim3 blocks(blocks_for(num_symbols * num_ofdm_symbols, threads.x));

    gather_transposed_llrs_kernel<<<blocks, threads, 0, stream>>>(
        inputs,
        max_num_symbols, max_num_ofdm_symbols,
        outputs,
        num_symbols, num_ofdm_symbols, num_bits
    );
}
// END marker-gather-llrs

#ifdef ENABLE_NANOBIND

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

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

NB_MODULE(data_processing, m) {
    m.def("int16_symbols_to_float16", [](const nb::ndarray<int16_t, nb::shape<-1, 2>, nb::device::cpu>& symbols_i, float scale) {
        __half *data = nullptr;
        cudaMallocManaged((void**) &data, std::max(sizeof(*data) * symbols_i.size(), (size_t) 16));
        memset(data, 0, sizeof(*data) * symbols_i.size());
        nb::capsule owner(data, [](void *p) noexcept { cudaFree(p); });

        cudaHostRegister(symbols_i.data(), sizeof(*symbols_i.data()) * symbols_i.size(), cudaHostRegisterDefault);
        int16_t const *mappedData = nullptr;
        cudaHostGetDevicePointer((void**) &mappedData, (void*) symbols_i.data(), 0);
        int16_symbols_to_float16(0, mappedData, symbols_i.shape(0), scale, (uint16_t*) data, 1);
        cudaDeviceSynchronize();
        cudaHostUnregister(symbols_i.data());

        return nb::ndarray<nb::numpy, __half, nb::ndim<2>>(data, {symbols_i.shape(0), 2}, owner);
    });
    m.def("float16_llrs_to_int16", [](const nb::ndarray<__half, nb::shape<-1, -1>, nb::device::cpu>& llrs_h) {
        int16_t *data = new int16_t[llrs_h.size()];
        cudaMallocManaged((void**) &data, std::max(sizeof(*data) * llrs_h.size(), (size_t) 16));
        memset(data, 0, sizeof(*data) * llrs_h.size());
        nb::capsule owner(data, [](void *p) noexcept { cudaFree(p); });

        cudaHostRegister(llrs_h.data(), sizeof(*llrs_h.data()) * llrs_h.size(), cudaHostRegisterDefault);
        uint16_t const *mappedData = nullptr;
        cudaHostGetDevicePointer((void**) &mappedData, (void*) llrs_h.data(), 0);
        float16_llrs_to_int16(0, mappedData, llrs_h.shape(0), data, llrs_h.shape(1));
        cudaDeviceSynchronize();
        cudaHostUnregister(llrs_h.data());

        return nb::ndarray<nb::numpy, int16_t, nb::ndim<2>>(data, {llrs_h.shape(0), llrs_h.shape(1)}, owner);
    });
    m.def("reshape_and_pad_32bit", [](const nb::ndarray<uint32_t, nb::shape<-1, -1, -1, -1>, nb::device::cpu>& values,
                                      const std::array<size_t, 4>& out_shape) {
        size_t out_numel = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
        uint32_t *data = new uint32_t[out_numel];
        cudaMallocManaged((void**) &data, std::max(sizeof(*data) * out_numel, (size_t) 16));
        memset(data, 0, sizeof(*data) * out_numel);
        nb::capsule owner(data, [](void *p) noexcept { cudaFree(p); });

        cudaHostRegister(values.data(), sizeof(*values.data()) * values.size(), cudaHostRegisterDefault);
        uint32_t const *mappedData = nullptr;
        cudaHostGetDevicePointer((void**) &mappedData, (void*) values.data(), 0);
        reshape_and_pad_32bit(0, mappedData, values.shape(0), values.shape(1), values.shape(2), values.shape(3),
                              data, out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
        cudaDeviceSynchronize();
        cudaHostUnregister(values.data());

        return nb::ndarray<nb::numpy, uint32_t, nb::ndim<4>>(data, {out_shape[0], out_shape[1], out_shape[2], out_shape[3]}, owner);
    });
    m.def("gather_transposed_llrs", [](const nb::ndarray<int16_t, nb::shape<-1, -1, -1>, nb::device::cpu>& llrs,
                                       const std::array<size_t, 2>& out_shape) {
        int16_t *data = new int16_t[llrs.size()];
        cudaMallocManaged((void**) &data, std::max(sizeof(*data) * llrs.size(), (size_t) 16));
        memset(data, 0, sizeof(*data) * llrs.size());
        nb::capsule owner(data, [](void *p) noexcept { cudaFree(p); });

        cudaHostRegister(llrs.data(), sizeof(*llrs.data()) * llrs.size(), cudaHostRegisterDefault);
        int16_t const *mappedData = nullptr;
        cudaHostGetDevicePointer((void**) &mappedData, (void*) llrs.data(), 0);
        gather_transposed_llrs(0, mappedData, llrs.shape(1), llrs.shape(2), data, out_shape[0], out_shape[1], llrs.shape(0));
        cudaDeviceSynchronize();
        cudaHostUnregister(llrs.data());

        return nb::ndarray<nb::numpy, int16_t, nb::ndim<3>>(data, {out_shape[0], out_shape[1], llrs.shape(0)}, owner);
    });
}

#endif
