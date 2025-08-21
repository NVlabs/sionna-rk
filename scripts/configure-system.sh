#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

echo "This script requires elevated privileges."
echo "It will ask for password on the first call to sudo."

# install git, docker and a few build dependencies
sudo apt update
sudo apt dist-upgrade -y
sudo apt install -y apt-utils coreutils git-core git cmake build-essential bc libssl-dev python3 python3-pip ninja-build ca-certificates curl pandoc

# Add Docker's official GPG key:
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# install docker and its plugins
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container

# add user to docker group
sudo usermod -aG docker $USER  # Log out and back in after this

# newer versions of docker require an extra daemon configuration
if [ -f /etc/systemd/system/docker.service.d/override.conf ]; then
    echo "Docker override '/etc/systemd/system/docker.service.d/override.conf' already exists, ensure the following is added, then reload the daemon:"
    echo "[Service]"
    echo "Environment=\"DOCKER_INSECURE_NO_IPTABLES_RAW=1\""
else
  sudo mkdir -p /etc/systemd/system/docker.service.d
  sudo tee /etc/systemd/system/docker.service.d/override.conf <<EOF
[Service]
Environment="DOCKER_INSECURE_NO_IPTABLES_RAW=1"
EOF
  echo "Reloading docker daemon..."
  sudo systemctl daemon-reload
  sudo systemctl restart docker
fi

# install tensorrt
sudo apt install -y cuda-toolkit nvidia-l4t-dla-compiler tensorrt

# add trtexec alias
touch ~/.bash_aliases
echo 'alias trtexec=/usr/src/tensorrt/bin/trtexec' >> ~/.bash_aliases

# install python utility
sudo python3 -m pip install -U jetson-stats

# install tensorflow for sionna-no-rt
python3 -m pip install --user --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60 tensorflow==2.15.0+nv24.05

# install python requirements for tutorials
base_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/../)
python3 -m pip install --user -r "${base_dir}/requirements.txt"

# install USRP drivers
${base_dir}/scripts/install-usrp.sh

# Query the current power mode
sudo nvpmodel -q

# Set power mode to remove limits
sudo nvpmodel -m 0

# Change power mode permanently
sudo sed -i 's|< PM_CONFIG DEFAULT=2 >|< PM_CONFIG DEFAULT=0 >|' /etc/nvpmodel.conf

# Needs to be done for each core group individually
# Change governor of cores 0-4
sudo cpufreq-set -c 0 -g performance

# Change governor of cores 5-8
sudo cpufreq-set -c 5 -g performance

# Change governor of cores 9-12
sudo cpufreq-set -c 9 -g performance

# set governor to performance, persist reboots
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils

echo "User $(whoami) has been added to the 'docker' group. Log out and back in to take effect."
