/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include "data_processing.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
    sf.x = ldexpf(float(s_r), -8);
    sf.y = ldexpf(float(s_i), -8);
    symbols_h[globalIdx * output_int32_stride] = __float22half2_rn(sf);
}

void int16_symbols_to_float16(
    cudaStream_t stream,
    const int16_t* symbols_i,
    uint32_t num_symbols,
    uint16_t* symbols_h,
    uint32_t output_int32_stride
    ) {
    dim3 threads(256);
    dim3 blocks(blocks_for(num_symbols, threads.x));

    int16_symbols_to_float16_kernel<<<blocks, threads, 0, stream>>>(
        symbols_i,
        num_symbols,
        reinterpret_cast<__half2*>(symbols_h),
        output_int32_stride
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

// START marker-normalize-symbols
__global__ void
norm_int16_symbols_to_float16_kernel(
    const int16_t* __restrict__ symbols_i,
    const int16_t* __restrict__ magnitudes_i,
    uint32_t num_symbols,
    __half2* __restrict symbols_h,
    uint32_t output_int32_stride
    ) {
    uint32_t globalIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (globalIdx >= num_symbols)
        return;

    uint32_t symbolBits = reinterpret_cast<const uint32_t*>(symbols_i)[globalIdx];
    int16_t s_r = int16_t(uint16_t(symbolBits & 0xffff)); // note: little endian
    int16_t s_i = int16_t(uint16_t(symbolBits >> 16));    // ...

    uint32_t magBits = reinterpret_cast<const uint32_t*>(magnitudes_i)[globalIdx];
    int16_t m_r = int16_t(uint16_t(magBits & 0xffff)); // note: little endian
    int16_t m_i = int16_t(uint16_t(magBits >> 16));    // ...

    float2 sf;
    sf.x = float(s_r) / float(m_r);
    sf.y = float(s_i) / float(m_i);
    symbols_h[globalIdx * output_int32_stride] = __float22half2_rn(sf);
}

void norm_int16_symbols_to_float16(
    cudaStream_t stream,
    const int16_t* symbols_i,
    const int16_t* magnitudes_i,
    uint32_t num_symbols,
    uint16_t* symbols_h,
    uint32_t output_int32_stride
    ) {
    dim3 threads(256);
    dim3 blocks(blocks_for(num_symbols, threads.x));

    norm_int16_symbols_to_float16_kernel<<<blocks, threads, 0, stream>>>(
        symbols_i,
        magnitudes_i,
        num_symbols,
        reinterpret_cast<__half2*>(symbols_h),
        output_int32_stride
    );
}
// END marker-normalize-symbols

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

NB_MODULE(data_processing, m) {
    m.def("int16_symbols_to_float16", [](const nb::ndarray<int16_t, nb::shape<-1, 2>, nb::device::cpu>& symbols_i) {
        __half *data = nullptr;
        cudaMallocManaged((void**) &data, std::max(sizeof(*data) * symbols_i.size(), (size_t) 16));
        memset(data, 0, sizeof(*data) * symbols_i.size());
        nb::capsule owner(data, [](void *p) noexcept { cudaFree(p); });

        cudaHostRegister(symbols_i.data(), sizeof(*symbols_i.data()) * symbols_i.size(), cudaHostRegisterDefault);
        int16_t const *mappedData = nullptr;
        cudaHostGetDevicePointer((void**) &mappedData, (void*) symbols_i.data(), 0);
        int16_symbols_to_float16(0, mappedData, symbols_i.shape(0), (uint16_t*) data, 1);
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
    m.def("norm_int16_symbols_to_float16", [](const nb::ndarray<int16_t, nb::shape<-1, 2>, nb::device::cpu>& symbols_i, const nb::ndarray<int16_t, nb::shape<-1, 2>, nb::device::cpu>& magnitudes_i) {
        __half *data = nullptr;
        cudaMallocManaged((void**) &data, std::max(sizeof(*data) * symbols_i.size(), (size_t) 16));
        memset(data, 0, sizeof(*data) * symbols_i.size());
        nb::capsule owner(data, [](void *p) noexcept { cudaFree(p); });

        assert(magnitudes_i.shape(0) == symbols_i.shape(0));
        cudaHostRegister(symbols_i.data(), sizeof(*symbols_i.data()) * symbols_i.size(), cudaHostRegisterDefault);
        cudaHostRegister(magnitudes_i.data(), sizeof(*magnitudes_i.data()) * magnitudes_i.size(), cudaHostRegisterDefault);
        int16_t const *mappedData = nullptr;
        int16_t const *mappedDataMag = nullptr;
        cudaHostGetDevicePointer((void**) &mappedData, (void*) symbols_i.data(), 0);
        cudaHostGetDevicePointer((void**) &mappedDataMag, (void*) magnitudes_i.data(), 0);
        norm_int16_symbols_to_float16(0, mappedData, mappedDataMag, symbols_i.shape(0), (uint16_t*) data, 1);
        cudaDeviceSynchronize();
        cudaHostUnregister(symbols_i.data());

        return nb::ndarray<nb::numpy, __half, nb::ndim<2>>(data, {symbols_i.shape(0), 2}, owner);
    });
}

#endif
