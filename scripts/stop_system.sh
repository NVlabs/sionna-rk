#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
set -e  # Stop script on any error

# supress outputs from pushd and popd
function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

# defaults
CONFIG_NAME=${1:-b200_arm64}
configs_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/../configs)

echo "Shutting down network"

pushd ${configs_dir}/$CONFIG_NAME

docker compose down

popd
