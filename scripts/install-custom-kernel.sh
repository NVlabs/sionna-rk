#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

echo "This script requires elevated privileges. It will ask for password on the first call to sudo. This is required to install the compiled kernel and its modules, and to modify the boot sequence. Use --dry-run to see the operations instead."

function usage() {
    echo "Usage: $0 [-h|--help] [--dry-run] [--verbose] [--source <path>]"
    exit 1
}

# supress outputs from pushd and popd
function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

function execute() {
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

function backup_file() {
    backup_file="$1$2"
    if [ -f "$backup_file" ]; then
        echo "Backup file $backup_file already exists, skipping."
    elif [ ! -f "$1" ]; then
        echo "Source file $1 does not exist, skipping."
    else
        execute sudo cp "$1" "$backup_file"
    fi
}

# default values
source_dir=$(realpath -sm "./ext/l4t/")
dest_dir="/"
VERBOSE=0
DRYRUN=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        --verbose) VERBOSE=1 ; shift ;;
        --dry-run) DRYRUN=1 ; shift ;;
        --source) source_dir="$2" ; shift 2 ;;
        *) shift ;;
    esac
done

# Check if both source and destination directories are provided
if [ -z "$source_dir" ] || [ -z "$dest_dir" ]; then
    usage
fi

# Convert src_dir and dest_dir to absolute paths
source_dir=$(realpath -sm "$source_dir")
dest_dir=$(realpath -sm "$dest_dir")

# set variables
execute export INSTALL_MOD_PATH="$source_dir/Linux_for_Tegra/rootfs"
execute export KERNEL_HEADERS="$source_dir/Linux_for_Tegra/source/kernel/kernel-jammy-src"

# backup existing setup (kernel, initrd, and modules)
execute pushd $(realpath -s "$dest_dir/boot")
backup_file Image .original
backup_file initrd .original
backup_file initrd.img-5.15.136-tegra .original

if [ ! -d $(realpath -sm "$dest_dir/lib/modules/5.15.136-tegra.original") ]; then
    execute sudo mv $(realpath -s "$dest_dir/lib/modules/5.15.136-tegra") $(realpath -sm "$dest_dir/lib/modules/5.15.136-tegra.original")
fi

# new kernel
execute sudo cp "$source_dir/Linux_for_Tegra/rootfs/boot/Image" "Image.compiled"

# new modules
execute sudo cp -r "$source_dir/Linux_for_Tegra/rootfs/lib/modules/5.15.136-tegra" $(realpath -sm "$dest_dir/lib/modules/5.15.136-tegra.compiled")

# create symlinks
# symlink modules
execute pushd $(realpath -s "$dest_dir/lib/modules")
execute sudo rm -rf 5.15.136-tegra
execute sudo ln -s 5.15.136-tegra.compiled 5.15.136-tegra
execute popd
# back in /boot

# symlink Image
execute sudo rm -f Image
execute sudo ln -s Image.compiled Image

# build new initrd
# note: this only works with $dest_dir=/
execute sudo nv-update-initrd

# symlinks for initrd
execute sudo mv initrd initrd.compiled
execute sudo ln -s initrd.compiled initrd

# update extlinux boot config
# note: this only works with $dest_dir=/
execute sudo nv-update-extlinux generic

# return to original working directory
execute popd
