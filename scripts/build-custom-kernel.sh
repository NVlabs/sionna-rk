#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

echo "This script requires elevated privileges. It will ask for password on the first call to sudo. This is required to install dependencies and the compiled kernel. Use --dry-run to see the operations instead."

usage() {
    echo "Usage: $0 [-h|--help] [--dry-run] [--verbose] [--clean] [--source <path>] <destination_path>"
    exit 1
}

execute() {
    # Always show the command
    if [ "$VERBOSE" == "1" ]; then
        echo "Executing: $@"
    fi

    if [ "$DRYRUN" == "1" ]; then
        # only print the command
        echo "[DRY-RUN] $@"
    else
        # actually execute the command
        eval "$@"
    fi
}

# default values
source_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/../)
dest_dir=$(realpath -sm "./ext/l4t")
VERBOSE=0
DRYRUN=0
CLEAN=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        --verbose) VERBOSE=1 ; shift ;;
        --dry-run) DRYRUN=1 ; shift ;;
        --clean) CLEAN=1 ; shift ;;
        --source) source_dir="$2" ; shift 2 ;;
        *) dest_dir="$1" ; shift ;;
    esac
done

# Check if both source and destination directories are provided
if [ -z "$source_dir" ] || [ -z "$dest_dir" ]; then
    usage
fi

# Convert src_dir and dest_dir to absolute paths
source_dir=$(realpath -sm "$source_dir")
dest_dir=$(realpath -sm "$dest_dir")

#echo "source_dir: $source_dir"
#echo "dest_dir: $dest_dir"
#echo "Verbose: $VERBOSE"
#echo "Dry-Run: $DRYRUN"
#echo "clean: $CLEAN"

# check if kernel config file is found under source_dir
defconfig_file="$source_dir/l4t/kernel/defconfig"
if [ ! -d "$(dirname defconfig_file)" ] || [ ! -f "$defconfig_file" ]; then
    echo "Kernel defconfig cannot be found at: ${defconfig_file}, please check your source directory or use --source."
    usage
fi

# If clean install, remove destination directory
if [ "$clean_dest" == "1" ] && [ -d "$dest_dir" ]; then
    echo "Removing directory $dest_dir ..."
    execute rm -rf "$dest_dir"
fi

# create destination directory if needed
if [ ! -d "$dest_dir" ]; then
    execute mkdir -p "$dest_dir"
else
    echo "Destination directory $dest_dir already exists. Use the --clean option or remove it before proceeding."
    usage
    exit 1
fi

# install software dependencies, needed to build the kernel
execute sudo apt update
execute sudo apt install git-core build-essential bc libssl-dev

# get into directory
execute pushd "$dest_dir"

# get l4t public sources
execute wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/sources/public_sources.tbz2

# extract source packages
execute tar xf public_sources.tbz2

# go into source directory
execute pushd  "$dest_dir/Linux_for_Tegra/source"

# expand required sources (kernel, OOT modules, display driver)
execute tar xf kernel_src.tbz2
execute tar xf kernel_oot_modules_src.tbz2
execute tar xf nvidia_kernel_display_driver_source.tbz2

# configure the kernel
execute cp "$source_dir/l4t/kernel/defconfig $dest_dir/Linux_for_Tegra/source/kernel/kernel-jammy-src/arch/arm64/configs/defconfig"

# build the kernel
execute make -C kernel

#create temporary root filesystem target directories
execute mkdir -p "$dest_dir/Linux_for_Tegra/rootfs/boot"
execute mkdir -p "$dest_dir/Linux_for_Tegra/rootfs/kernel"

# set the installation target for modules
execute export INSTALL_MOD_PATH="$dest_dir/Linux_for_Tegra/rootfs"

# install kernel in temporary rootfs
execute sudo -E make install -C kernel

# build modules (including OOT modules)
execute export KERNEL_HEADERS="$dest_dir/Linux_for_Tegra/source/kernel/kernel-jammy-src"
execute make modules

# install modules in temporary rootfs
execute sudo -E make modules_install

# go back
execute popd
execute popd
