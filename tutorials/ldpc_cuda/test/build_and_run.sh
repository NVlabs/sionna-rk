#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
set -e
cmake ../runtime -B build -G Ninja
ninja -C build
pytest -- run_decoder_test.py
