##
## SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
## SPDX-License-Identifier: Apache-2.0
##

GPU=
ifdef gpus
    GPU=--gpus=$(gpus)
endif
export GPU

.PHONY: doc prepare-system sionna-rk

prepare-system:
	./scripts/configure-system.sh
	./scripts/build-custom-kernel.sh
	./scripts/install-custom-kernel.sh
	echo "Reboot to load the new kernel and continue the installation."

sionna-rk:
	./scripts/quickstart-cn5g.sh
	./scripts/quickstart-oai.sh
	./scripts/generate-configs.sh
	./tutorials/neural_demapper/build-trt-plans.sh

doc: FORCE
	cd doc && ./build_docs.sh

FORCE:
