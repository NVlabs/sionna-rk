/*
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/

#ifndef __CHANNEL_EMULATOR_H__
#define __CHANNEL_EMULATOR_H__

#include "plugins/channel_emulation/cuda_emulator/src/chn_emu_extern.h"
#include "plugins/channel_emulation/file_loader/src/cir_file_extern.h"
#include "plugins/channel_emulation/zmq_loader/src/cir_zmq_extern.h"

void init_channel_emulator_libs(const NR_DL_FRAME_PARMS *fp);
void init_channel_emulator_worker_thread(void);
void free_channel_emulator_libs(void);

int is_channel_emulation_enabled(void);

// global function to fetch CIR data from an abstracted source (ZMQ or file).
const void *channel_emulator_cir_read_and_apply(void);

#endif // __CHANNEL_EMULATOR_H__