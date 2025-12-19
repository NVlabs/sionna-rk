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
models_dir=$(realpath ${default_dir}/../models)

# defaults
plan_file="${models_dir}/neural_demapper_qam16_2.plan"
onnx_file="${models_dir}/neural_demapper_qam16_2.onnx"

while [[ $# -gt 0 ]]; do
    case $1 in
        --plan)
            plan_file="$2"
            shift 2
            ;;
        --onnx)
            onnx_file="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--plan PATH] [--onnx PATH]"
            echo "  --plan PATH   Output plan file path (default: models/neural_demapper_qam16_2.plan)"
            echo "  --onnx PATH   Input ONNX file path (default: models/neural_demapper_qam16_2.onnx)"
            exit 1
            ;;
    esac
done

# normalize paths
plan_file=$(realpath -sm "$plan_file")
onnx_file=$(realpath -sm "$onnx_file")

echo "Using plan file: $plan_file"
echo "Using ONNX file: $onnx_file"

/usr/src/tensorrt/bin/trtexec --fp16 --onnx="${onnx_file}" --saveEngine="${plan_file}" --preview=+profileSharing0806 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --minShapes=y:1x2 --optShapes=y:64x2 --maxShapes=y:512x2
