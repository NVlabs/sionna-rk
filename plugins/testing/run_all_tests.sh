#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run all tutorial tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIALS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FRAMEWORK_DIR="$SCRIPT_DIR/framework"

# Parse arguments
VERBOSE=false
TUTORIAL=""
USE_HOST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --tutorial)
            TUTORIAL="$2"
            shift 2
            ;;
        --host)
            USE_HOST=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -v, --verbose     Verbose output"
            echo "  --tutorial NAME   Run tests for specific tutorial"
            echo "  --host            Use host-specific TensorRT plans (for unit tests)"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build arguments for Python script
PYTHON_ARGS=""
if [ "$VERBOSE" = true ]; then
    PYTHON_ARGS="$PYTHON_ARGS --verbose"
fi
if [ -n "$TUTORIAL" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --tutorial $TUTORIAL"
fi
if [ "$USE_HOST" = true ]; then
    PYTHON_ARGS="$PYTHON_ARGS --host"
fi

# Run the Python test runner
cd "$TUTORIALS_DIR"
export PYTHONPATH="$FRAMEWORK_DIR:$PYTHONPATH"
python3 "$FRAMEWORK_DIR/test_runner.py" --tutorials-dir "$TUTORIALS_DIR" $PYTHON_ARGS

