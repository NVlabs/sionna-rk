#!/bin/bash

# functions
function usage() {
    echo "Usage: $0 [-h|--help] [--force-platform <platform>] [--verbose] [--dry-run]"
    echo "Platforms: agx-orin, agx-thor, orin-nano, dgx-spark"
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

# default values
source_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/../)
platform="unknown"
family="unknown"
model="unknown"
VERBOSE=0
DRYRUN=0

if [ -f "/sys/devices/virtual/dmi/id/product_family" ]; then
    family=$(cat /sys/devices/virtual/dmi/id/product_family)
fi

if [ -f "/sys/devices/virtual/dmi/id/product_name" ]; then
    model=$(cat /sys/devices/virtual/dmi/id/product_name)
fi

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        --force-platform) platform="$2" ; shift 2 ;;
        --verbose) VERBOSE=1 ; shift ;;
        --dry-run) DRYRUN=1 ; shift ;;
        *) shift ;; # ignore other arguments
    esac
done

echo "Platform: $platform"
echo "Model: $model"
echo "Family: $family"

# select the script according to platform
if [ "$platform" == "agx-orin" ] || ([ "$model" == "NVIDIA Jetson AGX Orin Developer Kit" ] && [ "$platform" == "unknown" ]); then
    echo "Configuring system as 'Jetson AGX Orin Developer Kit'"
    execute "${source_dir}/scripts/configure-system.agx-orin.sh"
    exit 0
fi

if [ "$platform" == "agx-thor" ] || ([ "$model" == "NVIDIA Jetson AGX Thor Developer Kit" ] && [ "$platform" == "unknown" ]); then
    echo "Configuring system as 'Jetson AGX Thor Developer Kit'"
    execute "${source_dir}/scripts/configure-system.agx-thor.sh"
    exit 0
fi

if [ "$platform" == "orin-nano" ] || ([ "$model" == "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super" ] && [ "$platform" == "unknown" ]); then
    echo "Configuring system as 'Jetson Orin Nano Super Developer Kit'"
    execute "${source_dir}/scripts/configure-system.orin-nano.sh"
    exit 0
fi

if [ "$platform" == "dgx-spark" ] || ([ "$family" == "DGX Spark" ] && [ "$platform" == "unknown" ]); then
    echo "Configuring system as 'DGX Spark'"
    execute "${source_dir}/scripts/configure-system.dgx-spark.sh"
    exit 0
fi

echo "Unknown platform detected. Try forcing the platform with --force-platform <platform>. Exiting."
usage
exit 1