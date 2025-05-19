#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# supress outputs from pushd and popd
function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

default_dir=$(realpath $(dirname "${BASH_SOURCE[0]}"))
models_dir=${default_dir}/models

/usr/src/tensorrt/bin/trtexec --fp16 --onnx="${models_dir}/neural_demapper.2xfloat16.onnx" --saveEngine="${models_dir}/neural_demapper.2xfloat16.plan" --preview=+profileSharing0806 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --minShapes=y:1x2 --optShapes=y:64x2 --maxShapes=y:512x2