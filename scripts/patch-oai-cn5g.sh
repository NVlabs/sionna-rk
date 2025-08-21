#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

#!/bin/bash

# Function to print usage
function usage() {
    echo "Usage: $0 [-h|--help] (--source <source_directory> | --patch file) --dest <destination_directory>"
    exit 1
}

# supress outputs from pushd and popd
function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        --source) source_dir="$2"; shift ;;
        --patch) patch_file="$2"; shift ;;
        --dest) dest_dir="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

# Check if parameters are provided
if ([ -z "$source_dir" ] && [ -z "$patch_file" ]) || [ -z "$dest_dir" ]; then
    usage
fi
if [ ! -z "$source_dir" ] && [ ! -z "$patch_file" ]; then
    echo "specify either a source directory or a patch file, but not both."
    usage
fi

# Convert patchfile, source_dir and dest_dir to absolute paths, check if they exist.
if [ ! -z $patchfile ]; then
    patch_file=$(realpath -s "$patch_file")

    if [ ! -f "$patch_file" ]; then
        echo "Patch file '${patch_file}' does not exist."
        exit 1
    fi
fi

if [ ! -z $source_dir ]; then
    source_dir=$(realpath -s "$source_dir")

    if [ ! -d "$source_dir" ]; then
        echo "directory '${source_dir}' does not exist."
        exit 1
    fi
fi

dest_dir=$(realpath -sm "$dest_dir")

if [ ! -d ${dest_dir}/component/oai-upf/src ]; then
    echo "directory '${dest_dir}/component/oai-upf/src' does not exist."
    echo "Run '${dest_dir}/scripts/syncComponents.sh' and try again."
    exit 1
fi

echo "Applying changes from ${source_dir} into ${dest_dir}..."
if [ -f "$patch_file" ]; then
    pushd ${dest_dir}
    git apply < "$patch_file"
    popd
else
    rsync -av ${source_dir}/ ${dest_dir}/
fi
