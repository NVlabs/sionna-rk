import pytest
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

@pytest.fixture(scope="session")
def compiled_decoder():
    """
    Compiles the CUDA LDPC decoder extension in a temporary directory
    and adds it to sys.path.
    """
    # Source directory (where CMakeLists.txt is)
    src_dir = Path(__file__).parent.parent.parent / "src" / "runtime"
    src_dir = src_dir.resolve()

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

        print(f"\nBuilding LDPC CUDA extension in {build_path}...")
        
        # Configure with CMake
        cmd_config = [
            "cmake",
            str(src_dir),
            f"-DPython_EXECUTABLE={sys.executable}",
            f"{cuda_arch}"
        ]

        subprocess.check_call(cmd_config, cwd=build_path)

        # Build
        cmd_build = ["cmake", "--build", "."]
        subprocess.check_call(cmd_build, cwd=build_path)

        # Find the compiled module (usually .so file)
        # Add build directory to sys.path so we can import it
        sys.path.insert(0, str(build_path))

        try:
            import ldpc_decoder
            yield ldpc_decoder
        finally:
            # Remove from sys.path
            if str(build_path) in sys.path:
                sys.path.remove(str(build_path))
            # TemporaryDirectory handles cleanup of files


