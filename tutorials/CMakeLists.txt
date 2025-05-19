#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# build tutorials plugin entry points
set( PLUGINS_SRC
    plugins.c
    neural_demapper/nr_demapper_load.c
)

add_library( plugins STATIC ${PLUGINS_SRC})
add_dependencies(plugins generate_T)
target_link_libraries(nr-uesoftmodem PUBLIC plugins)
target_link_libraries(nr-softmodem PUBLIC plugins)

add_subdirectory(neural_demapper)
