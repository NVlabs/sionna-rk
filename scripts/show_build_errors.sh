#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
date -r build.log || echo "build.log not found"
grep -iC 2 -e "\(^\|\s\)error\([[:space:]:]\|$\)" --color build.log || echo "done scanning build.log"
