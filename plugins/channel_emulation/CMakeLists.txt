#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
# Update both local and parent so final propagation includes channel_emulator.c
set(PLUGINS_SRC ${PLUGINS_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/common/channel_emulator.c)
set(PLUGINS_SRC ${PLUGINS_SRC} PARENT_SCOPE)

add_subdirectory(cuda_emulator/src)
add_subdirectory(file_loader/src)
add_subdirectory(zmq_loader/src)

# Propagate PLUGINS_SRC (including loader sources from subdirs) to plugins
set(PLUGINS_SRC ${PLUGINS_SRC} PARENT_SCOPE)
