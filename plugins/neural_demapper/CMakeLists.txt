#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Support multiple CUDA architectures: Orin (sm_87), Thor (sm_110), DGX Spark (sm_121)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "87;110;121")
endif()
if (ENABLE_DGX_OPTIMIZATIONS)
    add_compile_definitions(ENABLE_DGX_OPTIMIZATIONS)
endif ()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(PLUGINS_SRC ${PLUGINS_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/src/nr_demapper_load.c PARENT_SCOPE)

# trt
if (ENABLE_CUDA)
  # find TensorRT package
  find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h PATHS ${tensorrt_ROOT} /usr/include /usr/local/TensorRT PATH_SUFFIXES include)
  find_library(TensorRT_LIBRARY NAMES nvinfer PATHS ${tensorrt_ROOT} /usr/lib /usr/local/TensorRT PATH_SUFFIXES lib)
  #find_library(TensorRT_PARSER_LIBRARY NAMES nvparsers PATHS ${tensorrt_ROOT} /usr/lib /usr/local/TensorRT PATH_SUFFIXES lib)
  find_library(TensorRT_ONNX_LIBRARY NAMES nvonnxparser PATHS ${tensorrt_ROOT} /usr/lib /usr/local/TensorRT PATH_SUFFIXES lib)
  if (NOT TensorRT_INCLUDE_DIR OR NOT TensorRT_LIBRARY OR NOT TensorRT_ONNX_LIBRARY)
    message(FATAL_ERROR "TensorRT installation missing ${TensorRT_INCLUDE_DIR} ${TensorRT_LIBRARY} ${TensorRT_PARSER_LIBRARY} ${TensorRT_ONNX_LIBRARY}")
  endif ()

  set(PHY_DEMAPPER_TRT_RUNTIME_SRC
    src/runtime/data_processing.cu
    src/runtime/trt_demapper.cpp
  )

  cuda_add_library( demapper_trt MODULE ${PHY_DEMAPPER_TRT_RUNTIME_SRC}
    src/nr_demapper_trt.c
  )
  set_target_properties(demapper_trt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_include_directories(demapper_trt PRIVATE ${TensorRT_INCLUDE_DIR})
  target_compile_options(demapper_trt PRIVATE -ftls-model=global-dynamic)
  target_link_options(demapper_trt PRIVATE -ftls-model=global-dynamic)
  target_compile_options(demapper_trt PRIVATE -Wno-deprecated-declarations -Wno-error=unused-variable -Wno-error=unused-function -Wno-error=unused-result)
  target_link_libraries(demapper_trt ${TensorRT_LIBRARY})

  add_dependencies(nr-uesoftmodem demapper_trt)
  add_dependencies(nr-softmodem demapper_trt)
  add_dependencies(demapper_trt generate_T)
endif ()
