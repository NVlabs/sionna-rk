#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

echo "This script requires elevated privileges."
echo "It will ask for password on the first call to sudo."

# install git, docker, jetpack  and a few build dependencies
sudo apt update
sudo apt dist-upgrade -y
sudo apt install -y apt-utils coreutils git-core git cmake build-essential bc libssl-dev python3 python3-pip ninja-build ca-certificates curl pandoc nvidia-jetpack

if [ "$(which nvcc)" == "" ]; then
  # finish CUDA installation, add path resolutions
  echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc
  echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
  source ~/.bashrc
fi

if [ ! -f /etc/apt/keyrings/docker.asc ]; then
  # Add Docker's official GPG key:
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc

  # Add the repository to Apt sources:
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
else
  echo "Docker's official GPG key already exists. Skipping..."
fi

# install docker and its plugins
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit

# Thor needs an updated nvidia-container toolkit
if [ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
  echo "Installing nvidia-container toolkit repo..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  # uncomment if experimental packages are needed
  #sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update

  sudo apt install -y libnvidia-container-tools libnvidia-container1 nvidia-container-toolkit nvidia-container-toolkit-base

  # use instead if a specific version is needed.
  # export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.1-1
  # sudo apt-get install -y nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
  # docker daemon will be restarted in the next step
else
  echo "NVIDIA container toolkit keyring already exists. Skipping..."
fi

if id -nG "$USER" | grep -qw "docker"; then
  echo "User $USER is already in the 'docker' group. Skipping..."
else
  # add user to docker group
  sudo usermod -aG docker $USER  # Log out and back in after this
  echo "User $USER has been added to the 'docker' group. Log out and back in to take effect."
fi

# we need to configure the docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
# docker daemon will be restarted in the next step

# newer versions of docker require an extra daemon configuration
append_docker_setting=0
sudo mkdir -p /etc/systemd/system/docker.service.d
if [ -f /etc/systemd/system/docker.service.d/override.conf ]; then
  # file exists, check if we need to append
  if grep -q "Environment=\"DOCKER_INSECURE_NO_IPTABLES_RAW=1\"" /etc/systemd/system/docker.service.d/override.conf; then
    echo "Docker override '/etc/systemd/system/docker.service.d/override.conf' already exists and contains the required setting. Skipping..."
  else
    append_docker_setting=1
  fi
else
  append_docker_setting=1
fi

if [ "$append_docker_setting" == "1" ]; then
  sudo tee /etc/systemd/system/docker.service.d/override.conf <<EOF
[Service]
Environment="DOCKER_INSECURE_NO_IPTABLES_RAW=1"
EOF
fi

echo "Reloading docker daemon..."
sudo systemctl daemon-reload
sudo systemctl restart docker

# install tensorrt
sudo apt install -y cuda-toolkit tensorrt

# add trtexec alias
touch ~/.bash_aliases
if grep -q "alias trtexec=/usr/src/tensorrt/bin/trtexec" ~/.bash_aliases; then
  echo "trtexec alias already exists. Skipping..."
else
  echo 'alias trtexec=/usr/src/tensorrt/bin/trtexec' >> ~/.bash_aliases
fi

# install python requirements for tutorials
# TODO: cannot install requirements because tensorrt needs to come from OS.
# therefore, requirements.txt does not work inside an environment as is.

base_dir=$(realpath $(dirname "${BASH_SOURCE[0]}")/../)

#pushd $base_dir
#python3 -m venv env
#source "${base_dir}/env/bin/activate"
#python -m pip install -r "${base_dir}/requirements.txt"
#deactivate

# install USRP drivers
${base_dir}/scripts/install-usrp.sh


# Query the current power mode
sudo nvpmodel -q

# Set power mode to remove limits
sudo nvpmodel -m 0

# Change power mode permanently
if grep -q "< PM_CONFIG DEFAULT=0 >" /etc/nvpmodel.conf; then
  echo "Power mode already set to max-performance. Skipping..."
else
  sudo sed -i 's|< PM_CONFIG DEFAULT=1 >|< PM_CONFIG DEFAULT=0 >|' /etc/nvpmodel.conf
fi

# Needs to be done for each core group individually
# Change governor of core pairs
sudo cpufreq-set -c 0 -g performance
sudo cpufreq-set -c 2 -g performance
sudo cpufreq-set -c 4 -g performance
sudo cpufreq-set -c 6 -g performance
sudo cpufreq-set -c 8 -g performance
sudo cpufreq-set -c 10 -g performance
sudo cpufreq-set -c 12 -g performance

# set governor to performance, persist reboots
if [ -f /etc/default/cpufrequtils ] && grep -q "GOVERNOR=\"performance\"" /etc/default/cpufrequtils; then
  echo "Governor already set to performance. Skipping..."
else
  echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
fi

# configure SRK_PLATFORM variable. Used to decide platform flags
if [ -z "$SRK_PLATFORM" ]; then
  export SRK_PLATFORM=$( ${base_dir}/scripts/detect_host.sh )
  echo "export SRK_PLATFORM=\"$SRK_PLATFORM\"" >> ~/.profile
  echo "SRK_PLATFORM set to $SRK_PLATFORM"
else
  echo "SRK_PLATFORM set to $SRK_PLATFORM, not changed."
fi

if [ -z "$SRK_THREAD_POOL" ]; then
  export SRK_THREAD_POOL="9,10,11,12,13"
  echo "export SRK_THREAD_POOL=\"$SRK_THREAD_POOL\"" >> ~/.profile
  echo "SRK_THREAD_POOL set to $SRK_THREAD_POOL"
else
  echo "SRK_THREAD_POOL set to $SRK_THREAD_POOL, not changed."
fi

if [ -z "$SRK_UE_THREAD_POOL" ]; then
  export SRK_UE_THREAD_POOL="4,5"
  echo "export SRK_UE_THREAD_POOL=\"$SRK_UE_THREAD_POOL\"" >> ~/.profile
  echo "SRK_UE_THREAD_POOL set to $SRK_UE_THREAD_POOL"
else
  echo "SRK_UE_THREAD_POOL set to $SRK_UE_THREAD_POOL, not changed."
fi
