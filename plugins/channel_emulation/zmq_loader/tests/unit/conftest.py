#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import sys
import subprocess
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def compiled_cir_zmq():
    """
    Compiles the CIR ZMQ extension in a temporary directory
    and adds it to sys.path.
    """
    # Source directory (where CMakeLists.txt is)
    src_dir = Path(__file__).parent.parent
    src_dir = src_dir.resolve()

    # Create a temporary directory for the build
    with tempfile.TemporaryDirectory() as build_dir:
        build_path = Path(build_dir)

        print(f"\nBuilding CIR ZMQ extension in {build_path}...")

        # Configure with CMake
        cmd_config = [
            "cmake",
            str(src_dir),
            f"-DPython_EXECUTABLE={sys.executable}"
        ]
        subprocess.check_call(cmd_config, cwd=build_path)

        # Build
        cmd_build = ["cmake", "--build", "."]
        subprocess.check_call(cmd_build, cwd=build_path)

        # Add build directory to sys.path so we can import it
        sys.path.insert(0, str(build_path))

        try:
            import cir_zmq_py
            yield cir_zmq_py
        finally:
            # Remove from sys.path
            if str(build_path) in sys.path:
                sys.path.remove(str(build_path))
