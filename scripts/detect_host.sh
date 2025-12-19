#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# default values
family="Unknown"
model="Unknown"

if [ -f "/sys/devices/virtual/dmi/id/product_family" ]; then
    family=$(cat /sys/devices/virtual/dmi/id/product_family)
fi

if [ -f "/sys/devices/virtual/dmi/id/product_name" ]; then
    model=$(cat /sys/devices/virtual/dmi/id/product_name)
fi

# select the script according to platform
if [ "$model" == "NVIDIA Jetson AGX Orin Developer Kit" ]; then
    echo "AGX Orin"
    exit 0
fi

if [ "$model" == "NVIDIA Jetson AGX Thor Developer Kit" ]; then
    echo "AGX Thor"
    exit 0
fi

if [ "$model" == "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super" ]; then
    echo "Orin Nano Super"
    exit 0
fi

if [ "$family" == "DGX Spark" ]; then
    echo "DGX Spark"
    exit 0
fi

echo "Unknown"
exit 1