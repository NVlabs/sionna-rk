#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

ROOT_SOURCE=$(dirname "${BASH_SOURCE[0]}")/../tutorials/common/openairinterface5g
OAI_DEST=$1

echo "Applying changes from ${ROOT_SOURCE} into ${OAI_DEST}..."
rsync -av ${ROOT_SOURCE}/ ${OAI_DEST}/
echo "Applying changes from tutorials into ${OAI_DEST}/tutorials..."
rsync -av ${ROOT_SOURCE}/../../../tutorials/ --exclude 'common/openairinterface5g/' ${OAI_DEST}/tutorials/
