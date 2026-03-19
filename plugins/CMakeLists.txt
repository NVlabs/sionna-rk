#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sionna Research Kit - plugins Build Configuration
#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

option(ENABLE_CUDA "Enable CUDA package" ON)

if (ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  # Support multiple CUDA architectures: Orin (sm_87), Thor (sm_110), DGX Spark (sm_121)
  # Can be overridden via -DCMAKE_CUDA_ARCHITECTURES=<arch>; plugins may set CUDA_ARCHITECTURES per target
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "87;110;121")
  endif()
endif()

# Base plugin entry points (subdirectories add to this via PARENT_SCOPE)
set(PLUGINS_SRC
    common/src/plugins.c
)

# Add plugin subdirectories
if (ENABLE_CUDA)
  add_subdirectory(neural_demapper)
  add_subdirectory(neural_receiver)
  add_subdirectory(ldpc_cuda)
  add_subdirectory(channel_emulation)
endif ()
add_subdirectory(data_acquisition)

# Build plugins plugin library with all sources
add_library(plugins_plugins STATIC ${PLUGINS_SRC})
add_dependencies(plugins_plugins generate_T)
target_link_libraries(nr-uesoftmodem PUBLIC plugins_plugins)
target_link_libraries(nr-softmodem PUBLIC plugins_plugins)
