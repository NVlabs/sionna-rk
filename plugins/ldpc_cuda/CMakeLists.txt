#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LDPC CUDA Plugin - Independent CMakeLists.txt
#
# Note: The main LDPC CUDA decoder is built via openair1/PHY/CODING/CMakeLists.txt
# which references files from this plugin. This CMakeLists.txt is for any
# additional plugin-specific build targets.
#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

set(LDPC_SRC_DIR ${PROJECT_SOURCE_DIR}/openair1/PHY/CODING)

find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
    message(FATAL_ERROR "no CUDA found")
endif()
SET(CUDA_VERBOSE_BUILD ON)
if (ENABLE_DGX_OPTIMIZATIONS)
    add_compile_definitions(ENABLE_DGX_OPTIMIZATIONS)
endif ()
cuda_add_library(ldpc_cuda MODULE
                ${LDPC_SRC_DIR}/nrLDPC_encoder/ldpc_encoder_optim8segmulti.c
                # The slot coding layer cannot be linked with
                # target_link_libraries like above
                # because of cuda_add_library
                # which already uses target_link_libraries
                ${LDPC_SRC_DIR}/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_decoder.c
                ${LDPC_SRC_DIR}/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_encoder.c
                ${LDPC_SRC_DIR}/nrLDPC_coding/nrLDPC_coding_segment/nr_rate_matching.c
                src/nr_ldpc_cuda.c
                src/runtime/ldpc_decoder.cu
)
target_compile_options(ldpc_cuda PRIVATE -Wno-error=unused-variable -Wno-error=unused-function -Wno-error=unused-result)

target_compile_definitions(ldpc_cuda PRIVATE LDPC_DECODER_NAME="CUDA LDPC decoder")

set_target_properties(ldpc_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ldpc_cuda PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_dependencies(ldpctest ldpc_cuda)
add_dependencies(nr-softmodem ldpc_cuda)
add_dependencies(nr-uesoftmodem ldpc_cuda)
add_dependencies(nr_ulsim ldpc_cuda)
add_dependencies(nr_ulschsim ldpc_cuda)
add_dependencies(nr_dlsim ldpc_cuda)
add_dependencies(nr_dlschsim ldpc_cuda)


