#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Function to print usage
function usage() {
    echo "Usage: $0 [-h|--help] [--rk-dir <kit-rootdir>] --source <source_directory> --dest <destination-for-patches>"
    exit 1
}

# supress outputs from pushd and popd
function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

# exclude files from patch by removing them from the add list
function exclude_files() {
    local cfg=$1
    local -n excludes="$2"

    for line in "${excludes[@]}"; do
        c=$(echo "$line" | cut -d'|' -f1)    # first column is the config
        file=$(echo "$line" | cut -d'|' -f2)    # second column is the filename

        if [ "$cfg" == "$c" ]; then
            if [ -f "$file" ]; then
                echo "File '$file' excluded, unstaging."
                git restore --staged "$file"
                git reset --quiet "$file"
            else
                echo "File '$file' excluded but does not exists, skipping. Update exclusion list?"
            fi
        fi
    done
}

function create_patch() {
    local cfg_name=$1
    local src=$2
    local patch=$3
    local -n exclude_list="$4"

    echo "Source: $src"
    echo "Patch file: $patch"

    pushd $src

    git add -N "*"

    exclude_files $cfg exclude_list

    git diff --no-prefix > "$patch"

    popd
}

# defaults
default_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/../)
rk_dir=${rk_dir:-"$default_dir"}

# Parse command line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        --rk-dir) rk_dir="$2"; shift ;;
        --source) source_dir="$2"; shift ;;
 	    --dest) dest_dir="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

# Check if directories are provided
if [ -z "$rk_dir" ] || [ -z "$source_dir" ] || [ -z "$dest_dir" ]; then
    usage
fi

# Convert to absolute paths
rk_dir=$(realpath -sm "$rk_dir")
source_dir=$(realpath -sm "$source_dir")
dest_dir=$(realpath -sm "$dest_dir")

# check if config files exist
config_list_filename="${rk_dir}/patches/configs/config-list.txt"
config_mappings_filename="${rk_dir}/patches/configs/config-mappings.txt"

if [ ! -f "$config_list_filename" ] || [ ! -f "$config_mappings_filename" ]; then
    echo "Config Files missing. Check:\n ${config_list_filename}\n ${config_mappings_filename}"
    usage
    exit 1
fi

# Read configs and mappings from disk
config_list_tmp=$(cat "${rk_dir}/patches/configs/config-list.txt")
config_mappings_tmp=$(cat "${rk_dir}/patches/configs/config-mappings.txt")
config_excludes_tmp=$(cat "${rk_dir}/patches/configs/config-excludes.txt")

IFS=$'\n' readarray -t config_list <<< "$config_list_tmp"
IFS=$'\n' readarray -t config_mappings <<< "$config_mappings_tmp"
IFS=$'\n' readarray -t config_excludes <<< "$config_excludes_tmp"

# create destination directory if needed
if [ ! -d "$dest_dir" ]; then
    mkdir -p $(dirname "$dest_dir")
fi

# iterate over config dirs
for cfg in "${config_list[@]}"; do
    create_patch "$cfg" "${source_dir}/${cfg}" "${dest_dir}/config.${cfg}.patch" config_excludes
done

echo "Patch files created."
