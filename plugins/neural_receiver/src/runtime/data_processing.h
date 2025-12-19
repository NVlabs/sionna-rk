#pragma once
#include <cstdint>

void int16_symbols_to_float16(
    cudaStream_t stream,
    const int16_t* symbols_i,
    uint32_t num_symbols,
    float scale,
    uint16_t* symbols_h,
    uint32_t output_int32_stride
    );

void float16_llrs_to_int16(
    cudaStream_t stream,
    uint16_t const* llrs_h,
    uint32_t num_symbols,
    int16_t* llrs_i,
    uint32_t num_bits // expect at least 2
    );

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
    );

void gather_transposed_llrs(
    cudaStream_t stream,
    int16_t const* inputs,
    uint32_t max_num_symbols,
    uint32_t max_num_ofdm_symbols,
    int16_t* outputs,
    uint32_t num_symbols,
    uint32_t num_ofdm_symbols,
    uint32_t num_bits
    );
