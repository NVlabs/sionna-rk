#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# supress outputs from pushd and popd
function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

OAI_DEST=$1

pushd ${OAI_DEST}

# see: https://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux/20983251#20983251
BOLD=$(tput bold)
NORMAL=$(tput sgr0)
RED=$(tput setaf 1)
TITLE="${RED}"
SUBTITLE="${BOLD}"

BASE=$(pwd)

print_info() {
	echo "${TITLE}$1${NORMAL}"
	cd "$2" && pushd "$_" > /dev/null
	git --no-pager log --oneline --no-abbrev-commit -1
	echo "${SUBTITLE}submodules:${NORMAL}"
	git submodule status
	echo -e "\n"
	popd > /dev/null
}

# start from the openairinterface5g directory
print_info "openairinterface5g" "$BASE"

popd
