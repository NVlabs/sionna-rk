#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Function to print usage
function usage() {
    echo "Usage: $0 [-h|--help] --source <source_directory> --dest <destination_directory> [--patch-file <patch_file>] [--binary]"
    exit 1
}

binary_patch=0

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        --source) source_dir="$2"; shift ;;
        --dest) dest_dir="$2"; shift ;;
	    --patch-file) patch_file="$2"; shift ;;
        --binary) binary_patch=1 ;;
        *) usage ;;
    esac
    shift
done

# Convert source_dir and dest_dir to absolute paths
source_dir=$(realpath -sm "$source_dir")
dest_dir=$(realpath -sm "$dest_dir")

# Convert patch_file to absolute path if needed
if [ -n "$patch_file" ]; then
    patch_file=$(realpath -sm "$patch_file")
fi

# Check if both source and destination directories are provided
if [ -z "$source_dir" ] || [ -z "$dest_dir" ]; then
    usage
fi

# Check if source directory exists and is a git repository
if [ ! -d "$source_dir" ] || [ ! -d "$source_dir/.git" ]; then
    echo "Error: Source directory is not a valid git repository"
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$dest_dir"

# ensure we return to the same path at the end
pushd "$(pwd)"

# Change to the source directory
cd "$source_dir" || exit 1

# Create patch file if --patch-file option is provided
if [ -n "$patch_file" ]; then
    patch_dir_path=$(dirname "$patch_file")
    mkdir -p "$patch_dir_path"
    if [ "$binary_patch" == "1" ]; then
        git diff --binary > "$patch_file"
        echo "Binary patch file created: $patch_file"
    else
        git diff > "$patch_file"
        echo "Patch file created: $patch_file"
    fi
fi

# Read the output of git diff --names-only and copy each file
git diff --name-only | while read -r file; do
    if [ -f "$file" ]; then
        # Create the directory structure in the destination
        dest_file="$dest_dir/$file"
        dest_dir_path=$(dirname "$dest_file")
        mkdir -p "$dest_dir_path"

        # Copy the file
        cp -v "$file" "$dest_file"
    else
        echo "Warning: $file not found"
    fi
done

# return to original path
popd
