#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import pytest
import sys
import subprocess
import tempfile
import os
from pathlib import Path

@pytest.fixture(scope="session")
def compiled_modules():
    """
    Compiles the Neural Receiver extension modules in a temporary directory
    and adds them to sys.path.
    Returns a dictionary with imported modules.
    """
    # Source directory (where CMakeLists.txt is)
    src_dir = Path(__file__).parent.parent.parent / "src" / "runtime"
    src_dir = src_dir.resolve()

    # Repo root (needed for the hardcoded model path in trt_receiver.cpp)
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    repo_root = repo_root.resolve()

    # Get platform from env var
    srk_platform = os.getenv("SRK_PLATFORM")
    if srk_platform == "DGX Spark":
        cuda_arch = "-DCMAKE_CUDA_ARCHITECTURES=120"
    elif srk_platform == "AGX Thor":
        cuda_arch = "-DCMAKE_CUDA_ARCHITECTURES=110"
    elif srk_platform == "AGX Orin" or srk_platform == "Nano Super":
        cuda_arch = "-DCMAKE_CUDA_ARCHITECTURES=87"
    else:
        cuda_arch = "-DCMAKE_CUDA_ARCHITECTURES=native"

    # Create a temporary directory for the build
    with tempfile.TemporaryDirectory() as build_dir:
        build_path = Path(build_dir)

        print(f"\nBuilding Neural Receiver extensions in {build_path}...")

        # Configure with CMake
        cmd_config = [
            "cmake",
            str(src_dir),
            f"-DPython_EXECUTABLE={sys.executable}",
            f"{cuda_arch}",
            "-DCMAKE_BUILD_TYPE=Release"
        ]

        subprocess.check_call(cmd_config, cwd=build_path)

        # Build
        cmd_build = ["cmake", "--build", "."]
        subprocess.check_call(cmd_build, cwd=build_path)

        # Add build directory to sys.path so we can import them
        sys.path.insert(0, str(build_path))

        # Change to repo root so the hardcoded model path works
        original_cwd = os.getcwd()
        os.chdir(repo_root)

        try:
            import data_processing
            import trt_receiver
            yield {
                "data_processing": data_processing,
                "trt_receiver": trt_receiver
            }
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            # Remove from sys.path
            if str(build_path) in sys.path:
                sys.path.remove(str(build_path))
            # TemporaryDirectory handles cleanup of files

