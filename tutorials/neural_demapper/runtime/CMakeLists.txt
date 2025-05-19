#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required(VERSION 3.18)
project(python_extension_modules)

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

enable_language(CUDA)

nanobind_add_module(data_processing data_processing.cu)
target_compile_definitions(data_processing PRIVATE ENABLE_NANOBIND)
#target_include_directories(data_processing PRIVATE ${CUDA_INCLUDE_DIRS})
#target_link_libraries(data_processing PRIVATE cuda)

find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h PATHS ${tensorrt_ROOT} /usr/include /usr/local/TensorRT PATH_SUFFIXES include)
find_library(TensorRT_LIBRARY NAMES nvinfer PATHS ${tensorrt_ROOT} /usr/lib /usr/local/TensorRT PATH_SUFFIXES lib)
#find_library(TensorRT_PARSER_LIBRARY NAMES nvparsers PATHS ${tensorrt_ROOT} /usr/lib /usr/local/TensorRT PATH_SUFFIXES lib)
find_library(TensorRT_ONNX_LIBRARY NAMES nvonnxparser PATHS ${tensorrt_ROOT} /usr/lib /usr/local/TensorRT PATH_SUFFIXES lib)
if (NOT TensorRT_INCLUDE_DIR OR NOT TensorRT_LIBRARY OR NOT TensorRT_ONNX_LIBRARY)
  message(FATAL_ERROR "TensorRT installation missing ${TensorRT_INCLUDE_DIR} ${TensorRT_LIBRARY} ${TensorRT_PARSER_LIBRARY} ${TensorRT_ONNX_LIBRARY}")
endif ()

nanobind_add_module(trt_demapper trt_demapper.cpp data_processing.cu)
target_compile_definitions(trt_demapper PRIVATE ENABLE_NANOBIND)
target_compile_options(trt_demapper PRIVATE -Wno-deprecated-declarations)

target_include_directories(trt_demapper PRIVATE ${TensorRT_INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(trt_demapper PRIVATE ${TensorRT_LIBRARY})
