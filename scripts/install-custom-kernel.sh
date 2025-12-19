#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

echo "This script requires elevated privileges. It will ask for password on the first call to sudo. This is required to install the compiled kernel and its modules, and to modify the boot sequence. Use --dry-run to see the operations instead."

function usage() {
    echo "Usage: $0 [-h|--help] [--source <path>] [--dest <path>] [--dry-run] [--verbose] [--backup-postfix <postfix>] [--kernel-version <version>]"
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
        ret_val=$?
        if [ $ret_val -ne 0 ]; then
            echo "Command failed with exit code $ret_val"
            exit $ret_val
        fi
    fi
}

function backup_file() {
    backup_file="$1.$2"
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
BACKUP_POSTFIX="original"
KERNEL_VERSION=$(uname -r)
CI=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        --verbose) VERBOSE=1 ; shift ;;
        --dry-run) DRYRUN=1 ; shift ;;
        --backup-postfix) BACKUP_POSTFIX="$2" ; shift 2 ;;
        --source) source_dir="$2" ; shift 2 ;;
        --dest) dest_dir="$2" ; shift 2 ;;
        --kernel-version) KERNEL_VERSION="$2" ; shift 2 ;;
        --ci) CI=1 ; shift ;;
        *) shift ;;
    esac
done

# check if tegra platform
if [ "$CI" == "0" ] && [ ! -f "/etc/nv_tegra_release" ]; then
    echo "This script is only supported on Tegra platforms. It is not needed in DGX Spark. Exiting..."
    exit 0
fi

# source distro info
source /etc/lsb-release

SOURCE_PATH=""
# check if we are on jammy
if [ "$DISTRIB_CODENAME" == "jammy" ]; then
    SOURCE_PATH="kernel-jammy-src"
fi

# check if we are on noble
if [ "$DISTRIB_CODENAME" == "noble" ]; then
    SOURCE_PATH="kernel-noble"
fi

if [ -z "$SOURCE_PATH" ]; then
    echo "This script is only supported on Ubuntu 22.04 (Jammy Jellyfish) and Ubuntu 24.04 (Noble Narwhal)."
    usage
fi


# Check if both source and destination directories are provided
if [ -z "$source_dir" ] || [ -z "$dest_dir" ]; then
    usage
fi

# Convert src_dir and dest_dir to absolute paths
source_dir=$(realpath -sm "$source_dir")
dest_dir=$(realpath -sm "$dest_dir")


# set variables
execute export INSTALL_MOD_PATH="$source_dir/Linux_for_Tegra/rootfs"
execute export KERNEL_HEADERS="$source_dir/Linux_for_Tegra/source/kernel/${SOURCE_PATH}"

# confirm that the kernel version in source matches the one specified.

highest_kernel_compiled=eval "ls \"$source_dir/Linux_for_Tegra/rootfs/lib/modules/\" | sort -V | tail -n 1"

if [ "$CI" == "0" ] && [ ! -d "$source_dir/Linux_for_Tegra/rootfs/lib/modules/$KERNEL_VERSION" ]; then
    echo "Kernel version in source ($highest_kernel_compiled) does not match the one specified ($KERNEL_VERSION). Exiting..."
    exit 1
fi

if [ "$CI" == "1" ] && [ ! -d "$source_dir/Linux_for_Tegra/rootfs/lib/modules/$KERNEL_VERSION" ]; then
    echo "CI mode: use the kernel version compiled in the source directory, update the KERNEL_VERSION variable."
    KERNEL_VERSION=$highest_kernel_compiled
    echo "Using kernel version $KERNEL_VERSION"
fi

# backup existing setup (kernel, initrd, and modules)
echo "Backing up existing files"
execute pushd $(realpath -s "$dest_dir/boot")
backup_file Image $BACKUP_POSTFIX
backup_file initrd $BACKUP_POSTFIX

if [ -d $(realpath -sm "$dest_dir/lib/modules/$KERNEL_VERSION.$BACKUP_POSTFIX") ]; then
    echo "Backup directory '$dest_dir/lib/modules/$KERNEL_VERSION.$BACKUP_POSTFIX' already exists. Skipping."
else
    if [ -d $(realpath -sm "$dest_dir/lib/modules/$KERNEL_VERSION") ]; then
        execute sudo mv $(realpath -s "$dest_dir/lib/modules/$KERNEL_VERSION") $(realpath -sm "$dest_dir/lib/modules/$KERNEL_VERSION.$BACKUP_POSTFIX")
    else
        echo "Modules directory '$dest_dir/lib/modules/$KERNEL_VERSION' does not exist. Skipping backup."
    fi
fi

echo "Installing new files"
# new kernel
execute sudo cp "$source_dir/Linux_for_Tegra/rootfs/boot/Image" "Image"

# new modules
if [ -d $(realpath -sm "$dest_dir/lib/modules/$KERNEL_VERSION") ]; then
    echo "Removing existing directory $dest_dir/lib/modules/$KERNEL_VERSION"
    execute sudo rm -rf $(realpath -s "$dest_dir/lib/modules/$KERNEL_VERSION")
fi
execute sudo cp -r "$source_dir/Linux_for_Tegra/rootfs/lib/modules/$KERNEL_VERSION" $(realpath -sm "$dest_dir/lib/modules/$KERNEL_VERSION")

# back in /boot

echo "Updating initrd and extlinux boot config"
# note: this only works with $dest_dir=/
if [ $CI == "0" ] && [ "$dest_dir" == "/" ]; then
    # build new initrd
    execute sudo nv-update-initrd

    # update extlinux boot config
    execute sudo nv-update-extlinux generic
else
    echo "Skipping initrd and extlinux boot config update because in CI mode or '$dest_dir' is not /"
fi

# return to original working directory
execute popd
