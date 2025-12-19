#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

echo "This script requires elevated privileges. It will ask for password on the first call to sudo. This is required to install dependencies and the compiled kernel. Use --dry-run to see the operations instead."

# supress outputs from pushd and popd
function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

function usage() {
    echo "Usage: $0 [-h|--help] [--dry-run] [--verbose] [--clean] [--force] [--ci] [--l4t-version <version>] [--source <path>] <destination_path>"
    exit 1
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

# Usage: path_in_list "/my/dir" "/path/one" "/my/dir" "/another/path"
function path_in_list() {
    local target="$1"
    shift
    for dir in "$@"; do
        if [[ "$target" == "$dir" ]]; then
            return 0
        fi
    done
    return 1
}

function module_exists() {
    local module="$1"
    if [ -z "$module" ]; then
        echo "Usage: module_exists <module_name>"
        return 2
    fi
    if find /lib/modules/$(uname -r) -type f -name "$module.ko*" | grep -q .; then
        echo "Module '$module' found."
        return 0
    else
        echo "Module '$module' not found."
        return 1
    fi
}

# globals
L4T_VERSION_MAJOR=0
L4T_VERSION_MINOR=0
L4T_VERSION_PATCH=0

function parse_version() {
    local input="$1"
    # Extract the version number using regex
    local version="$(echo "$input" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
    # Split version into components
    local major="${version%%.*}"
    local rest="${version#*.}"
    local minor="${rest%%.*}"
    local patch="${rest#*.}"

    # Print or export as needed
    L4T_VERSION_MAJOR=$major
    L4T_VERSION_MINOR=$minor
    L4T_VERSION_PATCH=$patch
}

function configure_kernel() {
    # Arguments:
    # 1: Path to kernel source tree
    # 2: Path to file containing list of CONFIG options to set

    local OPTIONS_LIST="$1"
    local KERNEL_SRC="$2"

    pushd "$KERNEL_SRC" || exit 1

    # Start from the clean default config
    make mrproper
    make defconfig

    # Apply each option from the options list
    while IFS= read -r option; do
        # Use scripts/config to set each option (located under kernel/scripts)
        ./scripts/config --file .config --set-val "${option%=*}" "${option#*=}"
    done < "$OPTIONS_LIST"

    # Make sure dependencies are resolved and new options are saved correctly
    make olddefconfig

    # Optional: Save the new config as defconfig
    cp .config "arch/arm64/configs/defconfig"

    popd
    
    echo "Config updated successfully."
}

# default values
source_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/../)
dest_dir=$(realpath -sm "./ext/l4t")
VERBOSE=0
DRYRUN=0
CLEAN=0
CI=0
L4T_VERSION=""
FORCE=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        --verbose) VERBOSE=1 ; shift ;;
        --dry-run) DRYRUN=1 ; shift ;;
        --clean) CLEAN=1 ; shift ;;
        --force) FORCE=1 ; shift ;;
        --ci) CI=1 ; shift ;;
        --l4t-version) L4T_VERSION="$2" ; shift 2 ;;
        --source) source_dir="$2" ; shift 2 ;;
        *) dest_dir="$1" ; shift ;;
    esac
done

# check if tegra platform
if [ "$CI" == "0" ] && [ ! -f "/etc/nv_tegra_release" ]; then
    echo "This script is only supported on Tegra platforms. It is not needed in DGX Spark. Exiting..."
    exit 0
fi

# check if modules sctp and qmi_wwan exist
if [ "$CI" == "0" ] && [ "$FORCE" == "0" ] && module_exists "sctp" && module_exists "qmi_wwan"; then
    echo "Found modules 'sctp' and 'qmi_wwan'. No need to build the kernel. Exiting..."
    exit 0
fi

# parse version
if [ "$CI" == "1" ] || [ -n "$L4T_VERSION" ]; then
    if [ -z "$L4T_VERSION" ]; then
        echo "L4T version is not set. Please use --l4t-version to set the version."
        usage
    fi
    parse_version "$L4T_VERSION"
else
    # parse version from installed package
    parse_version "$(dpkg-query --show nvidia-l4t-core)"
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

# check if we are on aarch64

# Check if both source and destination directories are provided
if [ -z "$source_dir" ] || [ -z "$dest_dir" ]; then
    usage
fi

# Convert src_dir and dest_dir to absolute paths
source_dir=$(realpath -sm "$source_dir")
dest_dir=$(realpath -sm "$dest_dir")

# check if kernel config file is found under source_dir
config_options="$source_dir/l4t/kernel/config.options"
if [ ! -d "$(dirname config_options)" ] || [ ! -f "$config_options" ]; then
    echo "Kernel config options cannot be found at: ${config_options}, please check your source directory or use --source."
    usage
fi

# If clean install, remove destination directory
if [ "$CLEAN" == "1" ] && [ -d "$dest_dir" ]; then
    disallowed_paths=("/" "/boot" "/etc" "/home" "/lib" "/lib64" "/root" "/sbin" "/usr" "/var")
    if path_in_list "$dest_dir" "${disallowed_paths[@]}"; then
        echo "Destination directory is '$dest_dir'. Refusing to remove it."
        usage
    fi

    echo "Removing directory $dest_dir ..."
    execute sudo rm -rf "$dest_dir"
fi

# create destination directory if needed
if [ ! -d "$dest_dir" ] || [ "$DRYRUN" == "1" ]; then
    if [ "DRYRUN" == "1" ]; then
        echo "Assuming destination directory $dest_dir does not exist for the dry run."
    fi
    echo "Creating destination directory $dest_dir ..."
    execute mkdir -p "$dest_dir"
else
    echo "Destination directory $dest_dir already exists. Use the --clean option or remove it before proceeding."
    usage
    exit 1
fi

# install software dependencies, needed to build the kernel
echo "Installing software dependencies ..."
execute sudo apt update
execute sudo apt install -y git-core build-essential flex bison bc kmod libssl-dev

# get into directory
execute pushd "$dest_dir"

# check for 38.2.2
if [ $CI == "0" ] && [ "$L4T_VERSION_MAJOR" -eq 38 ] && [ "$L4T_VERSION_MINOR" -gt 2 ] || ( [ "$L4T_VERSION_MINOR" -eq 2 ] && [ "$L4T_VERSION_PATCH" -ge 2 ] ); then
    echo "Checking out kernel sources for L4T 38.2.2 or greater is currently not supported. Need to checkout BSP 38.2.1 and get sources another way."

    execute pushd "$dest_dir"

    execute wget https://developer.nvidia.com/downloads/embedded/L4T/r38_Release_v2.1/release/Jetson_Linux_R38.2.1_aarch64.tbz2
    execute tar xf Jetson_Linux_R38.2.1_aarch64.tbz2

    execute pushd "$dest_dir/Linux_for_Tegra/source"
    echo "Fetch sources from git..."

    execute ./source_sync.sh -k -t jetson_${L4T_VERSION_MAJOR}.${L4T_VERSION_MINOR}.${L4T_VERSION_PATCH}
else
    # get l4t public sources
    echo "Getting l4t public sources ..."
    execute wget "https://developer.nvidia.com/downloads/embedded/l4t/r${L4T_VERSION_MAJOR}_release_v${L4T_VERSION_MINOR}.${L4T_VERSION_PATCH}/sources/public_sources.tbz2"

    # check if source package is found
    if [ ! -f "public_sources.tbz2" ]; then
        echo "Could not download L4T public sources. Please check the L4T version or use --l4t-version."
        usage
    fi

    # extract source packages
    echo "Extracting source packages ..."
    execute tar xf public_sources.tbz2

    # go into source directory
    execute pushd  "$dest_dir/Linux_for_Tegra/source"

    # expand required sources (kernel, OOT modules, display driver)
    echo "Expanding sources ..."
    execute tar xf kernel_src.tbz2
    execute tar xf kernel_oot_modules_src.tbz2
    execute tar xf nvidia_kernel_display_driver_source.tbz2
    if [ $DISTRIB_CODENAME == "noble" ]; then
        # this is new in Thor
        execute tar xf nvidia_unified_gpu_display_driver_source.tbz2
    fi
fi

# configure the kernel
echo "Configuring the kernel ..."
execute configure_kernel "$config_options" "$dest_dir/Linux_for_Tegra/source/kernel/${SOURCE_PATH}"

# build the kernel
echo "Building the kernel ..."
execute make -C kernel

#create temporary root filesystem target directories
echo "Creating temporary root filesystem target directories ..."
execute mkdir -p "$dest_dir/Linux_for_Tegra/rootfs/boot"
execute mkdir -p "$dest_dir/Linux_for_Tegra/rootfs/kernel"

# set the installation target for modules
execute export INSTALL_MOD_PATH="$dest_dir/Linux_for_Tegra/rootfs"

# install kernel in temporary rootfs
echo "Installing kernel in temporary rootfs ..."
execute sudo -E make install -C kernel

# build modules (including OOT modules)
echo "Building modules (including OOT modules) ..."
execute export KERNEL_HEADERS="$dest_dir/Linux_for_Tegra/source/kernel/${SOURCE_PATH}"
if [ $DISTRIB_CODENAME == "noble" ]; then
    # this is new in Thor
    execute export kernel_name=noble 
fi

execute make modules

# install modules in temporary rootfs
echo "Installing modules in temporary rootfs ..."
execute sudo -E make modules_install

# go back
execute popd
execute popd

echo "Done."
