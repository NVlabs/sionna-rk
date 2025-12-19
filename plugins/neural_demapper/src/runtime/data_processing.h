/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#pragma once
#include <cstdint>

void int16_symbols_to_float16(
    cudaStream_t stream,
    const int16_t* symbols_i,
    uint32_t num_symbols,
    uint16_t* symbols_h,
    uint32_t output_int32_stride
    );

void norm_int16_symbols_to_float16(
    cudaStream_t stream,
    const int16_t* symbols_i,
    const int16_t* magnitudes_i,
    uint32_t num_symbols,
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
