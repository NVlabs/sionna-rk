#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# File and configurations
# Common files go into config-dir/common
# Compose files fo into config-dir/<config-name>
# config_list is a list of configuration directories to initialize.
# patch file names are also picked from the config_list

# functions
function usage() {
    echo "Usage: $0 [-h|--help] [--clean] [--no-patching] [--init-nested-repos] --rk-dir <kit-rootdir> --oai-dir <openairinterface5g_dir> --dest <config-dir>"
    exit 1
}

# supress outputs from pushd and popd
function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

function init_nested_repo() {
    local cfg=$1
    local cfg_dir=$2

    pushd $cfg_dir

    git init                                        # create nested repo
    git add *                                       # add files to repo
    git commit --no-gpg-sign -m "init config $cfg"  # commit initial changes to it

    popd
}

# defaults
default_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/../)
rk_dir=${rk_dir:-"$default_dir"}
oai_dir=${oai_dir:-$(realpath -sm "./ext/openairinterface5g")}
dest_dir=${dest_dir:-$(realpath -sm "${rk_dir}/configs")}
clean_dest=0
patching=1
init=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        --rk-dir) rk_dir="$2"; shift ;;
        --oai-dir) oai_dir="$2"; shift ;;
        --dest) dest_dir="$2"; shift ;;
        --clean) clean_dest=1 ;;
        --no-patching) patching=0 ;;
        --init-nested-repos) init=1 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

# Check if directories are provided
if [ -z "$rk_dir" ] || [ -z "$oai_dir" ] || [ -z "$dest_dir" ]; then
    usage
fi

# Convert dirs to absolute paths
rk_dir=$(realpath -sm "$rk_dir")
oai_dir=$(realpath -sm "$oai_dir")
dest_dir=$(realpath -sm "$dest_dir")

echo "Sionna RK dir: $rk_dir"
echo "OAI dir: $oai_dir"
echo "Configs: $dest_dir"

# If clean install, remove destination directory, and local config branches
if [ "$clean_dest" = "1" ] && [ -d "$dest_dir" ]; then
    echo "Removing directory $dest_dir ..."
    rm -rf "$dest_dir"
fi

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

IFS=$'\n' readarray -t config_list <<< "$config_list_tmp"
IFS=$'\n' readarray -t config_mappings <<< "$config_mappings_tmp"

# create destination directory if needed
if [ ! -d "$dest_dir" ]; then
    mkdir -p $(dirname "$dest_dir")
else
    echo "Destination directory $dest_dir already exists. Use the --clean option or remove it before proceeding."
    usage
    exit 1
fi

# check if the files needed from OAI are there
for line in "${config_mappings[@]}"; do
    file=$(echo "$line" | cut -d'|' -f3)    # third column is the OAI path
    if [ -f "${oai_dir}/${file}" ]; then
        echo "Found: $file"
    else
        echo "Not found: $file"
        echo "Ensure you have a valid OpenAirInterface source directory or run quickstart-oai.sh"
        echo "aborting."
        usage
        exit 1
    fi
done

# copy all files as described in the mappings, per config.
for cfg in "${config_list[@]}"; do
    # create dir
    echo "Creating config dir: $cfg"
    mkdir -p "${dest_dir}/${cfg}"

    # copy files
    for line in "${config_mappings[@]}"; do
        cfg_map=$(echo "$line" | cut -d'|' -f1)
        file_dest=$(echo "$line" | cut -d'|' -f2)
        file_src=$(echo "$line" | cut -d'|' -f3)

        # copy files for the selected cfg only
        if [ "$cfg_map" = "$cfg" ]; then
            cp "${oai_dir}/${file_src}" "${dest_dir}/${cfg}/${file_dest}"
        fi
    done

    # patch dir
    if [ "$patching" = "1" ]; then
        echo "Patching: ${cfg}"
        pushd "${dest_dir}/${cfg}"
        patch -p0 < "${rk_dir}/patches/configs/config.${cfg}.patch"
        popd
    fi

    if [ "$init" = "1" ]; then
        echo "Initializing nested git repo for: ${cfg}"
        init_nested_repo "$cfg" "${dest_dir}/${cfg}"
    fi
done

echo "Generate config files done."
