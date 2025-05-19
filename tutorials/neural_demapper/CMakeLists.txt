#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# START marker-plugin-cmake
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# demapper libraries

# original
set(PHY_DEMAPPER_ORIG_SRC
  nr_demapper_orig.c
)

add_library(demapper_orig MODULE ${PHY_DEMAPPER_ORIG_SRC})
add_library(demapper MODULE ${PHY_DEMAPPER_ORIG_SRC})

add_dependencies(nr-softmodem demapper_orig demapper)
add_dependencies(nr-uesoftmodem demapper_orig demapper)
add_dependencies(demapper generate_T)
add_dependencies(demapper_orig generate_T)
# END marker-plugin-cmake

# START marker-plugin-cmake-capture
# capture
set(PHY_DEMAPPER_CAPTURE_SRC
  ${OPENAIR1_DIR}/PHY/NR_TRANSPORT/nr_ulsch_llr_computation.c
  nr_demapper_capture.c
)

add_library(demapper_capture MODULE ${PHY_DEMAPPER_CAPTURE_SRC})
target_link_libraries(demapper_capture PRIVATE pthread)

add_dependencies(nr-softmodem demapper_capture)
add_dependencies(nr-uesoftmodem demapper_capture)
add_dependencies(demapper_capture generate_T)
# END marker-plugin-cmake-capture

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
    runtime/data_processing.cu
    runtime/trt_demapper.cpp
  )

  cuda_add_library( demapper_trt MODULE ${PHY_DEMAPPER_TRT_RUNTIME_SRC}
    nr_demapper_trt.c
  )
  set_target_properties(demapper_trt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_include_directories(demapper_trt PRIVATE ${TensorRT_INCLUDE_DIR})
  target_compile_options(demapper_trt PRIVATE -ftls-model=global-dynamic -mtls-dialect=desc)
  target_link_options(demapper_trt PRIVATE -ftls-model=global-dynamic -mtls-dialect=desc)
  target_link_libraries(demapper_trt ${TensorRT_LIBRARY})

  add_dependencies(nr-uesoftmodem demapper_trt)
  add_dependencies(nr-softmodem demapper_trt)
  add_dependencies(demapper_trt generate_T)
endif ()
