#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# suppress outputs from pushd and popd
function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

OAI5G_DEST=$1

pushd ${OAI5G_DEST}

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

# start from the -fed directory
print_info "oai-cn5g-fed" "$BASE"
print_info "oai-cn5g-amf" "$BASE/component/oai-amf"
print_info "oai-cn5g-ausf" "$BASE/component/oai-ausf"
print_info "oai-cn5g-nef" "$BASE/component/oai-nef"
print_info "oai-cn5g-nrf" "$BASE/component/oai-nrf"
print_info "oai-cn5g-nssf" "$BASE/component/oai-nssf"
print_info "oai-cn5g-pcf" "$BASE/component/oai-pcf"
print_info "oai-cn5g-smf" "$BASE/component/oai-smf"
print_info "oai-cn5g-udm" "$BASE/component/oai-udm"
print_info "oai-cn5g-udr" "$BASE/component/oai-udr"
print_info "oai-cn5g-upf" "$BASE/component/oai-upf"
print_info "oai-cn5g-upf-vpp" "$BASE/component/oai-upf-vpp"

popd
