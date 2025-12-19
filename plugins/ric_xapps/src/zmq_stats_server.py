#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
FlexRIC xApp with ZeroMQ Publisher - Real-time UE Statistics
Subscribes to FlexRIC MAC statistics and publishes to ZeroMQ clients at configurable intervals.
"""

import zmq
import json
import time
import logging
import signal
import sys
import os
from typing import Dict, Any, List
import xapp_sdk as ric

CHECK_RNTI = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for callback (to avoid SWIG issues with object references)
latest_ue_stats = None
last_indication_time = 0
callback_count = 0
# RNTI to IMSI/UE number mapping
rnti_to_imsi_map = {}  # {rnti: {'imsi': str, 'ue_number': int}}
# PRB tracking for interval - track per-slot PRB usage from each callback
prb_tracking = {}  # {rnti: {'dl_prbs': [prbs], 'ul_prbs': [prbs]}}


class FlexRICStatsPublisher:
    def __init__(self, port: int = 5555, interval: float = 0.5, rnti_imsi_map_file: str = None):
        """
        Initialize the FlexRIC xApp with ZeroMQ publisher.

        Args:
            port: Port to bind the ZeroMQ publisher to
            interval: Interval in seconds to publish data (default: 0.5s)
            rnti_imsi_map_file: Path to JSON file with RNTI to IMSI/UE number mapping
        """
        self.port = port
        self.interval = interval

        # Auto-detect mapping file location (Docker vs host)
        if rnti_imsi_map_file is None:
            # Try Docker path first, then local path
            docker_path = "/xapp/rnti_imsi_map.json"
            local_path = "rnti_imsi_map.json"
            if os.path.exists(docker_path):
                self.rnti_imsi_map_file = docker_path
            else:
                self.rnti_imsi_map_file = local_path
        else:
            self.rnti_imsi_map_file = rnti_imsi_map_file

        # ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)  # Publisher socket

        # FlexRIC variables
        self.mac_handlers = []
        self.mac_callbacks = []  # CRITICAL: Keep callback objects alive!
        self.nodes = []

        # Stats tracking
        self.last_publish_time = 0
        self.publish_count = 0

        # Control variables
        self.running = False
        self.ric_initialized = False

    def _load_rnti_imsi_mapping(self):
        """Load RNTI to IMSI/UE number mapping from JSON file."""
        global rnti_to_imsi_map
        try:
            with open(self.rnti_imsi_map_file, 'r') as f:
                mapping = json.load(f)
                # Convert string RNTIs to integers
                rnti_to_imsi_map = {int(rnti): info for rnti, info in mapping.items()}
        except FileNotFoundError:
            rnti_to_imsi_map = {}
        except Exception as e:
            logger.error(f"Error loading RNTI mapping: {e}")
            rnti_to_imsi_map = {}

    def _publish_stats(self, stats_data: Dict[str, Any]):
        """Publish stats to ZeroMQ subscribers."""
        try:
            # Load IMSI mapping before each transmission
            self._load_rnti_imsi_mapping()

            # Filter out UEs that don't have IMSI mapping
            filtered_ue_stats = []
            for ue_idx,ue in enumerate(stats_data.get('UE_stats', [])):
                rnti = ue.get('rnti')
                if CHECK_RNTI:
                    if rnti in rnti_to_imsi_map:
                        # Add IMSI info to the UE data
                        imsi_info = rnti_to_imsi_map[rnti]
                        ue['ue_id'] = f"UE {imsi_info['ue_number']}"
                        ue['imsi'] = imsi_info['imsi']
                        filtered_ue_stats.append(ue)
                else:
                    ue['ue_id'] = f"UE {ue_idx}"
                    filtered_ue_stats.append(ue)
            # Update stats_data with filtered list
            stats_data['UE_stats'] = filtered_ue_stats

            topic = "ue_stats"
            self.socket.send_multipart([
                topic.encode('utf-8'),
                json.dumps(stats_data).encode('utf-8')
            ])

            self.publish_count += 1
            num_ues = len(stats_data.get('UE_stats', []))

            # Log summary - every 20th message show detailed info
            if self.publish_count % 20 == 0:
                logger.info(f"Published #{self.publish_count}: {num_ues} UEs at TS {stats_data['timestamp']}")
                for ue in stats_data.get('UE_stats', []):
                    imsi_str = f"IMSI:{ue['imsi']} " if 'imsi' in ue else ""
                    logger.info(f"  {ue['ue_id']} {imsi_str}RNTI:{ue['rnti']} MCS↓:{ue['mcs_down']}/↑:{ue['mcs_up']} "
                              f"BLER↓:{ue['bler_down']:.3f}/↑:{ue['bler_up']:.3f} "
                              f"PRBs(max)↓:{ue['num_prbs_down']}/↑:{ue['num_prbs_up']} "
                              f"PRBs(total)↓:{ue['aggr_prbs_down']}/↑:{ue['aggr_prbs_up']}")
            else:
                logger.info(f"Published #{self.publish_count}: {num_ues} UEs at TS {stats_data['timestamp']}")

        except Exception as e:
            logger.error(f"Error publishing stats: {e}")

    def _init_flexric(self):
        """Initialize FlexRIC connection and subscribe to MAC stats."""
        try:
            logger.info("Connecting to nearRT-RIC...")
            ric.init()

            # Get E2 nodes
            self.nodes = ric.conn_e2_nodes()
            logger.info(f"✓ Connected to RIC")
            logger.info(f"✓ Found {len(self.nodes)} E2 node(s)")

            if len(self.nodes) == 0:
                logger.error("✗ No E2 nodes connected. Please start your gNB.")
                return False

            for i, node in enumerate(self.nodes):
                logger.info(f"Node {i + 1}: PLMN {node.id.plmn.mcc}.{node.id.plmn.mnc}")

            logger.info("Subscribing to MAC statistics...")

            # Subscribe to MAC with callback
            # Use Interval_ms_1 for 1ms reporting period (fastest)
            for i, node in enumerate(self.nodes):
                mac_cb = MACStatsCallback()
                self.mac_callbacks.append(mac_cb)  # CRITICAL: Keep callback alive!
                handler = ric.report_mac_sm(node.id, ric.Interval_ms_1, mac_cb)
                self.mac_handlers.append(handler)

            logger.info("✓ Subscribed to MAC service model")
            self.ric_initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize FlexRIC: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start_publisher(self):
        """Start the FlexRIC xApp and ZeroMQ publisher."""
        global latest_ue_stats

        try:
            # Bind ZeroMQ socket
            self.socket.bind(f"tcp://*:{self.port}")
            logger.info(f"ZeroMQ Publisher started on port {self.port}")

            # Initialize FlexRIC
            if not self._init_flexric():
                logger.error("Failed to initialize FlexRIC - exiting")
                self._cleanup()
                sys.exit(1)

            logger.info(f"Publishing UE stats every {self.interval} seconds")
            logger.info("Press Ctrl+C to stop")
            logger.info("=" * 70)

            self.running = True
            self.last_publish_time = time.time()

            # Main loop - publish at regular intervals
            while self.running:
                try:
                    current_time = time.time()

                    # Check if it's time to publish
                    if (current_time - self.last_publish_time) >= self.interval:
                        if latest_ue_stats is not None:
                            self._publish_stats(latest_ue_stats)
                        else:
                            # No data yet, publish empty stats
                            empty_stats = {
                                "msg_type": "ue_stats_report",
                                "timestamp": int(time.time() * 1000),
                                "UE_stats": []
                            }
                            self._publish_stats(empty_stats)

                        self.last_publish_time = current_time

                    # Sleep briefly to avoid busy-waiting
                    time.sleep(0.01)

                except zmq.error.ZMQError as e:
                    logger.error(f"ZMQ Error: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\nPublisher shutdown requested (Ctrl+C)")
        finally:
            self._cleanup()

    def stop_publisher(self):
        """Stop the publisher."""
        self.running = False

    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.running = False

        # Stop FlexRIC subscriptions
        if self.ric_initialized:
            for handler in self.mac_handlers:
                try:
                    ric.rm_mac_sm(handler)
                    logger.info("✓ Unsubscribed from MAC stats")
                except Exception as e:
                    logger.warning(f"Error removing MAC handler: {e}")

        # Close ZeroMQ
        try:
            self.socket.close()
            self.context.term()
            logger.info("✓ ZeroMQ closed")
        except Exception as e:
            logger.warning(f"Error closing ZeroMQ: {e}")

        logger.info(f"Publisher stopped. Total messages published: {self.publish_count}")


class MACStatsCallback(ric.mac_cb):
    """Callback for MAC indications - converts and stores UE stats."""

    def __init__(self):
        ric.mac_cb.__init__(self)

    def handle(self, ind):
        """Handle MAC indication from FlexRIC."""
        global latest_ue_stats, last_indication_time, callback_count, rnti_to_imsi_map, prb_tracking

        # Increment counter
        callback_count += 1

        # Track PRB usage per slot for all callbacks in the interval
        if hasattr(ind, 'ue_stats') and len(ind.ue_stats) > 0:
            for ue_idx in range(len(ind.ue_stats)):
                ue = ind.ue_stats[ue_idx]

                if hasattr(ue, 'rnti'):
                    rnti = ue.rnti

                    # Initialize tracking for this RNTI if needed
                    if rnti not in prb_tracking:
                        prb_tracking[rnti] = {'dl_prbs': [], 'ul_prbs': []}

                    # Get per-slot PRB values (fixed with OAI patch)
                    # These contain actual current_rbs values from UE->mac_stats.{dl,ul}.current_rbs
                    dl_prbs = ue.dl_sched_rb if hasattr(ue, 'dl_sched_rb') else 0
                    ul_prbs = ue.ul_sched_rb if hasattr(ue, 'ul_sched_rb') else 0

                    # Track PRB values from each callback
                    prb_tracking[rnti]['dl_prbs'].append(dl_prbs)
                    prb_tracking[rnti]['ul_prbs'].append(ul_prbs)

        # Only process full report every 500th callback (roughly 0.5s at 1ms interval)
        if callback_count % 500 != 0:
            return

        # Build list
        ue_stats_list = []

        # Check if we have UE statistics
        if hasattr(ind, 'ue_stats') and len(ind.ue_stats) > 0:
            num_ues = len(ind.ue_stats)

            for ue_idx in range(num_ues):
                ue = ind.ue_stats[ue_idx]

                # Build dict with defaults
                ue_data = {
                    "ue_id": f"UE_{ue_idx + 1:03d}",
                    "rnti": 0,
                    "mcs_up": 0,
                    "mcs_down": 0,
                    "bler_up": 0.0,
                    "bler_down": 0.0
                }

                # RNTI
                if hasattr(ue, 'rnti'):
                    ue_data["rnti"] = ue.rnti
                    rnti = ue.rnti

                    # Check if we have IMSI mapping for this RNTI
                    if rnti in rnti_to_imsi_map:
                        imsi_info = rnti_to_imsi_map[rnti]
                        ue_data["ue_id"] = f"UE {imsi_info['ue_number']}"
                        ue_data["imsi"] = imsi_info['imsi']

                # MCS values (codeword 1)
                if hasattr(ue, 'ul_mcs1'):
                    ue_data["mcs_up"] = ue.ul_mcs1

                if hasattr(ue, 'dl_mcs1'):
                    ue_data["mcs_down"] = ue.dl_mcs1

                # BLER values
                if hasattr(ue, 'ul_bler'):
                    ue_data["bler_up"] = ue.ul_bler

                if hasattr(ue, 'dl_bler'):
                    ue_data["bler_down"] = ue.dl_bler

                # Add aggregated PRB counts (cumulative total since UE connection)
                ue_data["aggr_prbs_down"] = ue.dl_aggr_prb if hasattr(ue, 'dl_aggr_prb') else 0
                ue_data["aggr_prbs_up"] = ue.ul_aggr_prb if hasattr(ue, 'ul_aggr_prb') else 0

                # Calculate max PRBs from samples collected during interval
                rnti = ue_data["rnti"]
                if rnti in prb_tracking:
                    dl_samples = prb_tracking[rnti]['dl_prbs']
                    ul_samples = prb_tracking[rnti]['ul_prbs']

                    # Calculate total and max PRBs observed in this interval
                    ue_data["delta_prbs_down"] = sum(dl_samples)
                    ue_data["delta_prbs_up"] = sum(ul_samples)
                    ue_data["num_prbs_down"] = max(dl_samples) if dl_samples else 0
                    ue_data["num_prbs_up"] = max(ul_samples) if ul_samples else 0

                    # Clear samples for next interval
                    prb_tracking[rnti]['dl_prbs'] = []
                    prb_tracking[rnti]['ul_prbs'] = []
                else:
                    # No tracking data yet for this UE
                    ue_data["delta_prbs_down"] = 0
                    ue_data["delta_prbs_up"] = 0
                    ue_data["num_prbs_down"] = 0
                    ue_data["num_prbs_up"] = 0

                ue_stats_list.append(ue_data)

        # Create stats message
        latest_ue_stats = {
            "msg_type": "ue_stats_report",
            "timestamp": int(time.time() * 1000),
            "UE_stats": ue_stats_list
        }
        last_indication_time = time.time()


def main():
    """Main function to run the FlexRIC xApp with ZeroMQ publisher."""
    import argparse

    parser = argparse.ArgumentParser(
        description='FlexRIC xApp with ZeroMQ Publisher - Real-time UE Statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (0.5s interval, port 5555)
  %(prog)s

  # Custom publish interval (1 second)
  %(prog)s --interval 1.0

  # Custom port and faster interval (100ms)
  %(prog)s --port 5556 --interval 0.1
        """
    )
    parser.add_argument('--port', type=int, default=5555,
                       help='Port to bind ZeroMQ publisher to (default: 5555)')
    parser.add_argument('--interval', type=float, default=0.5,
                       help='Publish interval in seconds (default: 0.5)')

    args = parser.parse_args()

    # Validate arguments
    if args.interval <= 0:
        logger.error("Interval must be positive")
        sys.exit(1)

    if args.port < 1024 or args.port > 65535:
        logger.error("Port must be between 1024 and 65535")
        sys.exit(1)

    # Create and start publisher
    publisher = FlexRICStatsPublisher(
        port=args.port,
        interval=args.interval
    )

    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\n⚠️  Interrupted by user (Ctrl+C)")
        publisher.stop_publisher()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the xApp and publisher
    publisher.start_publisher()


if __name__ == "__main__":
    main()
