/*
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/
#include "plugins.h"
#include <stddef.h>

// START marker-plugins
// global entry points for the tutorial plugins
// any global structures defined here.

demapper_interface_t demapper_interface = {0};


void init_plugins()
{
    // insert your plugin init here.

    load_demapper_lib( NULL, &demapper_interface);
}

void free_plugins()
{
    // insert your plugin release/free here.

    free_demapper_lib(&demapper_interface);
}

void worker_thread_plugin_init()
{
    // insert your plugin per thread initialization here.

    if (demapper_interface.init_thread)
        demapper_interface.init_thread();
}
// END marker-plugins
