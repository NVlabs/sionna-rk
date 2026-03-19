#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TUTORIALS_DIR = PROJECT_ROOT / "tutorials"
OAI_DIR = PROJECT_ROOT / "ext" / "openairinterface5g"
REGISTRY_PATH = PROJECT_ROOT / "tutorials" / "tutorial_registry.yaml"
PATCHES_DIR = PROJECT_ROOT / "patches"
CMAKE_GENERATOR = TUTORIALS_DIR / "common" / "scripts" / "generate_cmake_config.py"

def load_registry():
    if not REGISTRY_PATH.exists():
        print(f"Error: Registry not found at {REGISTRY_PATH}")
        sys.exit(1)

    with open(REGISTRY_PATH, "r") as f:
        return yaml.safe_load(f)

def reset_oai():
    """Reset OAI submodule to clean state."""
    print(f"[INFO] Resetting OAI repository at {OAI_DIR}...")

    if not OAI_DIR.exists():
        print(f"[ERROR] OAI directory not found at {OAI_DIR}")
        sys.exit(1)

    try:
        # Reset tracked files
        subprocess.run(["git", "checkout", "."], cwd=OAI_DIR, check=True, capture_output=True)
        # Clean untracked files/directories
        subprocess.run(["git", "clean", "-fd"], cwd=OAI_DIR, check=True, capture_output=True)
        print("[INFO] OAI repository reset to clean state.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to reset OAI repository: {e}")
        sys.exit(1)

def apply_base_patches():
    """Apply the common base patches required for the system."""
    base_patches = ["openairinterface5g.patch", "tutorials.patch"]

    print("[INFO] Applying base patches...")
    for patch_name in base_patches:
        patch_path = PATCHES_DIR / patch_name
        if patch_path.exists():
            try:
                subprocess.run(["git", "apply", str(patch_path)], cwd=OAI_DIR, check=True)
                print(f"[INFO] Applied base patch: {patch_name}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to apply base patch {patch_name}: {e}")
                sys.exit(1)
        else:
            print(f"[WARN] Base patch {patch_name} not found at {patch_path}")

def apply_patch(tutorial_name):
    """Apply the patch script for the given tutorial."""
    patch_script = TUTORIALS_DIR / tutorial_name / "patch_tutorial.sh"

    if not patch_script.exists():
        print(f"[INFO] No patch script found for '{tutorial_name}'. Skipping tutorial-specific patch.")
        return

    print(f"[INFO] Applying patches for '{tutorial_name}'...")
    try:
        # Make executable just in case
        os.chmod(patch_script, 0o755)
        # Run the patch script, passing the OAI directory as argument (standard convention)
        subprocess.run([str(patch_script), str(OAI_DIR)], check=True)
        print(f"[SUCCESS] Patches applied for '{tutorial_name}'.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to apply patches for '{tutorial_name}': {e}")
        sys.exit(1)

def regenerate_cmake_config():
    """Regenerate the CMakeLists.txt configuration."""
    print("[INFO] Regenerating CMake configuration...")
    if CMAKE_GENERATOR.exists():
        try:
            subprocess.run([sys.executable, str(CMAKE_GENERATOR)], check=True)
            print("[INFO] CMake configuration regenerated.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to regenerate CMake config: {e}")
            sys.exit(1)
    else:
        print(f"[WARN] CMake config generator not found at {CMAKE_GENERATOR}")

def main():
    parser = argparse.ArgumentParser(description="Activate a Sionna-RK tutorial by patching OAI.")
    parser.add_argument("tutorial", help="Name of the tutorial to activate")

    args = parser.parse_args()

    registry = load_registry()
    valid_tutorials = [t['name'] for t in registry.get('tutorials', [])]

    if args.tutorial not in valid_tutorials:
        print(f"[ERROR] Invalid tutorial '{args.tutorial}'. Available tutorials:")
        for t in valid_tutorials:
            print(f"  - {t}")
        sys.exit(1)

    # 1. Reset OAI to a clean slate
    reset_oai()

    # 2. Apply base patches (restore common environment)
    apply_base_patches()

    # 3. Apply the specific tutorial patches
    apply_patch(args.tutorial)

    # 4. Regenerate CMake configuration
    regenerate_cmake_config()

    # 5. Record active state
    with open(PROJECT_ROOT / ".active_tutorial", "w") as f:
        f.write(args.tutorial)

    print("\n" + "="*60)
    print(f"Tutorial '{args.tutorial}' ACTIVATED successfully.")
    print("IMPORTANT: You must now rebuild the docker images for changes to take effect.")
    print(f"Run: make build-gnb")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
