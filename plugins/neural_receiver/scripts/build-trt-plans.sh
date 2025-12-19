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
models_dir=$(realpath ${default_dir}/../../../ext/neural_rx/onnx_models)

# defaults
plan_file="${default_dir}/../models/nrx_oai.plan"
onnx_file="${models_dir}/nrx_oai.onnx"

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
            echo "  --plan PATH   Output plan file path (default: plugins/neural_receiver/models/nrx_oai.plan)"
            echo "  --onnx PATH   Input ONNX file path (default: ../../../ext/neural_rx/onnx_models/nrx_oai.onnx)"
            exit 1
            ;;
    esac
done

# normalize paths
plan_file=$(realpath -sm "$plan_file")
onnx_file=$(realpath -sm "$onnx_file")

echo "Using plan file: $plan_file"
echo "Using ONNX file: $onnx_file"

# 24 PRBs
/usr/src/tensorrt/bin/trtexec --fp16 --onnx=${onnx_file} --saveEngine=${plan_file} --minShapes=rx_slot:1x288x13x1x2,h_hat:1x432x1x1x2 --optShapes=rx_slot:1x288x13x1x2,h_hat:1x432x1x1x2 --maxShapes=rx_slot:1x288x13x1x2,h_hat:1x432x1x1x2 --inputIOFormats=fp16:chw,fp16:chw,fp16:chw,int32:chw,int32:chw --outputIOFormats=fp16:chw

# 51 PRBs
# requires re-export of ONNX model via /ext/neural_rx/scripts/export_onnx.py and modify the number of PRBs in the config file n_size_bwp_eval=51
# Note that MAX_BLOCK_LEN is currently hardcoded in the receiver plugin.
#/usr/src/tensorrt/bin/trtexec --fp16 --onnx=${onnx_file} --saveEngine=${plan_file} --minShapes=rx_slot:1x612x13x1x2,h_hat:1x918x1x1x2 --optShapes=rx_slot:1x612x13x1x2,h_hat:1x918x1x1x2 --maxShapes=rx_slot:1x612x13x1x2,h_hat:1x918x1x1x2 --inputIOFormats=fp16:chw,fp16:chw,fp16:chw,int32:chw,int32:chw --outputIOFormats=fp16:chw
