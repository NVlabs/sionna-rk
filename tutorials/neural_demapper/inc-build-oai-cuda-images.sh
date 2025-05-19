#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
DOCKER_CUSTOM_IMAGE_TAG="${DOCKER_CUSTOM_IMAGE_TAG:-latest}"

mkdir -p ./override/lib
docker run -v.:/incremental-source -v ./override/lib:/incremental-build --rm -it ran-build:${DOCKER_CUSTOM_IMAGE_TAG} bash -c "rsync -vruc /incremental-source/ ./ && cd cmake_targets/ran_build/build && touch incremental_tag && ninja ${@} && find . -type f -newer incremental_tag | rsync -av --files-from=- ./ /incremental-build/"
docker build --target oai-gnb --tag oai-gnb:${DOCKER_CUSTOM_IMAGE_TAG} --file docker/Dockerfile.gNB.cuda.aarch64 --build-arg DOCKER_CUSTOM_IMAGE_TAG=${DOCKER_CUSTOM_IMAGE_TAG} .
