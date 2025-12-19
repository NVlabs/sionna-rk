#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
ZeroMQ Client for receiving UE statistics from a server.
Subscribes to UE stats data published by the server at regular intervals.
"""

import zmq
import json
import time
import logging
from typing import Dict, Any, Optional
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UEStatsSubscriber:
    def __init__(self, server_host: str = "localhost", server_port: int = 5555):
        """
        Initialize the ZeroMQ UE stats subscriber.

        Args:
            server_host: Server hostname or IP address
            server_port: Server port number
        """
        self.server_host = server_host
        self.server_port = server_port
        self.client_id = socket.gethostname()

        # ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = None

        # Control variables
        self.running = False
        self.message_count = 0


    def connect(self) -> bool:
        """
        Connect to the UE stats publisher.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.socket = self.context.socket(zmq.SUB)  # Subscriber socket

            # Subscribe to "ue_stats" topic
            self.socket.setsockopt(zmq.SUBSCRIBE, b"ue_stats")

            # Set socket options
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout

            server_url = f"tcp://{self.server_host}:{self.server_port}"
            self.socket.connect(server_url)

            logger.info(f"Connected to UE stats publisher at {server_url}")
            logger.info("Subscribed to 'ue_stats' topic")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to publisher: {e}")
            return False

    def _process_ue_stats(self, ue_stats: Dict[str, Any]):
        """Process received UE statistics."""
        try:
            self.message_count += 1
            timestamp = ue_stats.get("timestamp", 0)
            num_ues = len(ue_stats.get("UE_stats", []))

            logger.info(f"Received UE stats #{self.message_count}: {num_ues} UEs at timestamp {timestamp}")
            self._print_ue_details(ue_stats)

        except Exception as e:
            logger.error(f"Error processing UE stats: {e}")

    def _print_ue_details(self, ue_stats: Dict[str, Any]):
        """Print UE details."""
        print(f"\n--- UE Stats #{self.message_count} (TS: {ue_stats.get('timestamp', 'N/A')}) ---")

        for ue in ue_stats.get("UE_stats", []):
            rnti = ue.get('rnti', 0)
            imsi = ue.get('imsi', 'N/A')
            imsi_str = f"IMSI {imsi}, " if imsi != 'N/A' else ""

            print(f"{ue.get('ue_id')}, {imsi_str}RNTI {rnti} (0x{rnti:04x}): "
                  f"MCS ↑{ue.get('mcs_up')}/↓{ue.get('mcs_down')}, "
                  f"BLER ↑{ue.get('bler_up', 0):.3f}/↓{ue.get('bler_down', 0):.3f}, "
                  f"PRBs ↑{ue.get('num_prbs_up', 0)}/↓{ue.get('num_prbs_down', 0)}")

        print("---")

    def start_subscriber(self, max_messages: Optional[int] = None):
        """
        Start subscribing to UE stats data.

        Args:
            max_messages: Maximum number of messages to receive (None for unlimited)
        """
        try:
            logger.info("Starting UE stats subscriber...")
            logger.info(f"Waiting for UE stats data (max_messages: {max_messages or 'unlimited'})")
            logger.info("Press Ctrl+C to stop")

            self.running = True
            received_count = 0

            while self.running:
                try:
                    # Receive multipart message
                    message_parts = self.socket.recv_multipart(zmq.NOBLOCK)

                    if len(message_parts) >= 2:
                        data = json.loads(message_parts[1].decode('utf-8'))
                        self._process_ue_stats(data)
                        received_count += 1

                        if max_messages and received_count >= max_messages:
                            logger.info(f"Reached maximum message limit: {max_messages}")
                            break

                except zmq.error.Again:
                    # No message available, continue listening
                    time.sleep(0.1)
                    continue
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON message: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    time.sleep(1)
                    continue

        except KeyboardInterrupt:
            logger.info("Subscriber shutdown requested")
        finally:
            self._cleanup()

    def stop_subscriber(self):
        """Stop the subscriber."""
        self.running = False
        logger.info("Stopping subscriber...")

    def get_statistics(self) -> Dict[str, Any]:
        """Get subscriber statistics."""
        return {
            "messages_received": self.message_count,
            "client_id": self.client_id,
            "connected_to": f"{self.server_host}:{self.server_port}"
        }

    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up subscriber resources...")
        self.running = False
        if self.socket:
            self.socket.close()
        self.context.term()
        logger.info(f"Subscriber stopped. Total messages received: {self.message_count}")


def main():
    """Main function to run the UE stats subscriber."""
    import argparse

    parser = argparse.ArgumentParser(description='ZeroMQ UE Statistics Subscriber')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Publisher hostname or IP (default: localhost)')
    parser.add_argument('--port', type=int, default=5555,
                       help='Publisher port (default: 5555)')
    parser.add_argument('--max-messages', type=int, default=None,
                       help='Maximum number of messages to receive (default: unlimited)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')

    args = parser.parse_args()

    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Create subscriber
    subscriber = UEStatsSubscriber(
        server_host=args.host,
        server_port=args.port
    )

    try:
        # Connect to publisher
        if not subscriber.connect():
            logger.error("Failed to connect to publisher")
            return


        # Start subscribing
        subscriber.start_subscriber(max_messages=args.max_messages)

        # Print final statistics
        stats = subscriber.get_statistics()
        logger.info(f"Final statistics: {stats}")

    except KeyboardInterrupt:
        logger.info("Subscription interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
