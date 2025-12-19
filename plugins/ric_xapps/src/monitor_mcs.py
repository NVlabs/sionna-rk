#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Simple FlexRIC xApp - Monitor MCS and BLER every 2 seconds
Subscribes to MAC statistics and prints MCS and BLER values
"""

# [DOC_START]
import xapp_sdk as ric
import time
import signal
import sys

# Global variables to keep references alive and handle state
mac_handlers = []
mac_callbacks = []
last_print_time = 0

def signal_handler(sig, frame):
    print("\nCleaning up...")
    for handler in mac_handlers:
        try:
            ric.rm_mac_sm(handler)
        except Exception as e:
            print(f"Error removing handler: {e}")
    print("Cleanup complete")
    sys.exit(0)

# Define a callback class to handle MAC statistics
class MCSMonitorCallback(ric.mac_cb):
    def __init__(self):
        # Initialize the base class if necessary (SWIG usually requires this)
        ric.mac_cb.__init__(self)

    def handle(self, ind):
        global last_print_time
        current_time = time.time()

        # Print every 2 seconds
        if (current_time - last_print_time) < 2.0:
            return

        last_print_time = current_time
        print(f"\n[{time.strftime('%H:%M:%S')}]", flush=True)

        if hasattr(ind, 'ue_stats') and len(ind.ue_stats) > 0:
            for ue in ind.ue_stats:
                rnti = ue.rnti if hasattr(ue, 'rnti') else 0
                print(f"  RNTI: {rnti} (0x{rnti:04x})", flush=True)

                if hasattr(ue, 'dl_mcs1'):
                    bler = ue.dl_bler if hasattr(ue, 'dl_bler') else 0.0
                    print(f"    DL MCS: {ue.dl_mcs1}, BLER: {bler:.3f}", flush=True)

                if hasattr(ue, 'ul_mcs1'):
                    bler = ue.ul_bler if hasattr(ue, 'ul_bler') else 0.0
                    print(f"    UL MCS: {ue.ul_mcs1}, BLER: {bler:.3f}", flush=True)
        else:
            print("  No active UEs", flush=True)

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize RIC
ric.init()

    # Get connected nodes
nodes = ric.conn_e2_nodes()
print(f"Connected to {len(nodes)} E2 node(s)")

    if len(nodes) == 0:
        print("No E2 nodes found. Waiting...")
        # Just wait instead of exiting, maybe nodes come up later?
        # Usually conn_e2_nodes returns what's currently there.
        # But let's stick to the pattern.

for node in nodes:
        print(f"Subscribing to node: Global E2 Node ID: {node.id}")
    cb = MCSMonitorCallback()
        mac_callbacks.append(cb) # CRITICAL: Keep callback object alive

        # Subscribe to MAC stats
        # Using Interval_ms_1 as in original code, though we only print every 5s
    handler = ric.report_mac_sm(node.id, ric.Interval_ms_1, cb)
    mac_handlers.append(handler)

print("Monitoring... Press Ctrl+C to stop")

    # Keep main thread alive
while True:
        time.sleep(1)
# [DOC_END]

if __name__ == "__main__":
    main()
