#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Data Acquisition Tutorial - CMakeLists.txt
#
# This tutorial demonstrates data capture and acquisition techniques.
#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(PLUGINS_SRC ${PLUGINS_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/src/nr_demapper_load.c PARENT_SCOPE)

# Use local header files
set(DEMAPPER_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# original/passthrough demapper
set(PHY_DEMAPPER_ORIG_SRC
  src/nr_demapper_orig.c
)

add_library(demapper_orig MODULE ${PHY_DEMAPPER_ORIG_SRC})
target_include_directories(demapper_orig PRIVATE ${DEMAPPER_HEADERS_DIR})

add_dependencies(nr-softmodem demapper_orig)
add_dependencies(nr-uesoftmodem demapper_orig)
add_dependencies(demapper_orig generate_T)

# capture demapper
set(PHY_DEMAPPER_CAPTURE_SRC
  ${OPENAIR1_DIR}/PHY/NR_TRANSPORT/nr_ulsch_llr_computation.c
  src/nr_demapper_capture.c
)

add_library(demapper_capture MODULE ${PHY_DEMAPPER_CAPTURE_SRC})
target_include_directories(demapper_capture PRIVATE ${DEMAPPER_HEADERS_DIR})
target_include_directories(demapper_capture PUBLIC ${OPENAIR1_DIR}/PHY/nr_phy_common/inc/)
target_link_libraries(demapper_capture PRIVATE pthread)

add_dependencies(nr-softmodem demapper_capture)
add_dependencies(nr-uesoftmodem demapper_capture)
add_dependencies(demapper_capture generate_T)
