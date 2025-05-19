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

nanobind_add_module(ldpc_decoder ldpc_decoder.cu)
target_compile_definitions(ldpc_decoder PRIVATE ENABLE_NANOBIND)
#target_include_directories(ldpc_decoder PRIVATE ${CUDA_INCLUDE_DIRS})
#target_link_libraries(ldpc_decoder PRIVATE cuda)
