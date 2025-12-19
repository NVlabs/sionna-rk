##
## SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
## SPDX-License-Identifier: Apache-2.0
##

GPU=
ifdef gpus
    GPU=--gpus=$(gpus)
endif
export GPU

.PHONY: doc prepare-system sionna-rk build-gnb

prepare-system:
	./scripts/configure-system.sh
	./scripts/build-custom-kernel.sh
	./scripts/install-custom-kernel.sh
	echo "Reboot to load the new kernel and continue the installation."

sionna-rk:
	./scripts/quickstart-oai.sh
	./scripts/generate-configs.sh
	./plugins/common/build_all_plugins.sh --host
	./plugins/common/build_all_plugins.sh --container

build-gnb:
	./scripts/build-oai-images.sh --debug ext/openairinterface5g

doc: FORCE
	cd doc && ./build_docs.sh

test:
	./plugins/common/build_all_plugins.sh --host
	./plugins/common/build_all_plugins.sh --container
	./plugins/testing/run_all_tests.sh --host

FORCE:
