"""
WebSocket Streaming Test

Tests the real-time streaming of:
- Training metrics (loss, throughput, step)
- Worker logs
- Job state updates

This test script demonstrates how dashboards/monitoring tools
can connect to the WebSocket endpoint and receive real-time updates.
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import websockets
except ImportError:
    print("⚠️  websockets library not installed. Run: pip install websockets")
    sys.exit(1)


class WebSocketStreamMonitor:
    """
    Test client that connects to the WebSocket endpoint and monitors
    real-time metrics, logs, and state updates.
    """

    def __init__(self, coordinator_url: str = "ws://localhost:8000"):
        self.coordinator_url = coordinator_url
        self.is_running = False

        # Counters for different message types
        self.metrics_count = 0
        self.log_count = 0
        self.state_update_count = 0

    async def monitor_job(self, job_id: str, duration: int = 60):
        """
        Connect to WebSocket and monitor a job for the specified duration.

        Args:
            job_id: Job ID to monitor
            duration: How long to monitor (seconds)
        """
        ws_url = f"{self.coordinator_url}/ws/jobs/{job_id}/stream"

        print(f"🔌 Connecting to WebSocket: {ws_url}")
        print(f"⏱️  Will monitor for {duration} seconds")
        print("=" * 70)

        try:
            async with websockets.connect(ws_url) as websocket:
                self.is_running = True

                # Send periodic pings
                ping_task = asyncio.create_task(self._send_pings(websocket))

                # Set timeout for monitoring
                start_time = datetime.now()

                while self.is_running:
                    # Check if duration exceeded
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed >= duration:
                        print(f"\n⏹️  Monitoring duration reached ({duration}s)")
                        break

                    try:
                        # Wait for message with timeout
                        message_json = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=1.0
                        )

                        # Parse and handle message
                        message = json.loads(message_json)
                        self._handle_message(message)

                    except asyncio.TimeoutError:
                        # No message received, continue waiting
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print("\n❌ WebSocket connection closed by server")
                        break

                # Cancel ping task
                ping_task.cancel()

        except websockets.exceptions.WebSocketException as e:
            print(f"\n❌ WebSocket error: {e}")
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
        finally:
            self.is_running = False
            self._print_summary()

    async def _send_pings(self, websocket):
        """Send periodic pings to keep connection alive"""
        while self.is_running:
            try:
                await websocket.send(json.dumps({"command": "ping"}))
                await asyncio.sleep(30)
            except Exception:
                break

    def _handle_message(self, message: dict):
        """
        Handle incoming WebSocket message and display it.

        Args:
            message: Parsed JSON message
        """
        msg_type = message.get("type")
        timestamp = message.get("timestamp", "N/A")

        if msg_type == "connection":
            # Connection acknowledgment
            print(f"✅ Connected to job {message.get('job_id')}")
            print(f"   Status: {message.get('status')}")
            print(f"   Time: {timestamp}")
            print()

        elif msg_type == "metrics":
            # Training metrics
            self.metrics_count += 1
            data = message.get("data", {})

            print(f"📊 METRICS #{self.metrics_count}")
            print(f"   Step: {data.get('step')}")
            print(f"   Loss: {data.get('loss', 0):.4f}")
            print(f"   Throughput: {data.get('throughput', 0):.1f} samples/s")
            print(f"   Worker: {data.get('worker_id')}")
            print(f"   Rank: {data.get('rank')}")
            print(f"   Time: {timestamp}")
            print()

        elif msg_type == "log":
            # Worker log entry
            self.log_count += 1
            data = message.get("data", {})

            level = data.get("level", "INFO")
            log_message = data.get("message", "")
            worker_id = data.get("worker_id", "unknown")

            # Color code by log level
            level_emoji = {
                "DEBUG": "🐛",
                "INFO": "ℹ️",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "CRITICAL": "🔥"
            }.get(level, "📝")

            print(f"{level_emoji} LOG [{level}]")
            print(f"   Worker: {worker_id}")
            print(f"   Message: {log_message}")
            print(f"   Time: {timestamp}")
            print()

        elif msg_type == "state_update":
            # Job state change
            self.state_update_count += 1
            data = message.get("data", {})

            old_state = data.get("old_state")
            new_state = data.get("new_state")

            print(f"🔄 STATE UPDATE #{self.state_update_count}")
            print(f"   Transition: {old_state} → {new_state}")
            print(f"   Time: {timestamp}")
            print()

        elif msg_type == "pong":
            # Response to ping (keep-alive)
            print("💓 Ping response received")

        else:
            # Unknown message type
            print(f"❓ Unknown message type: {msg_type}")
            print(f"   Data: {message}")
            print()

    def _print_summary(self):
        """Print monitoring summary"""
        print("=" * 70)
        print("📈 MONITORING SUMMARY")
        print(f"   Metrics received: {self.metrics_count}")
        print(f"   Logs received: {self.log_count}")
        print(f"   State updates received: {self.state_update_count}")
        print("=" * 70)


async def main():
    """
    Main test function.

    Usage:
        python test_websocket_streaming.py <job_id> [duration]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor a DistroML job via WebSocket streaming"
    )
    parser.add_argument(
        "job_id",
        help="Job ID to monitor (from coordinator when job is submitted)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="How long to monitor in seconds (default: 60)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="ws://localhost:8000",
        help="Coordinator WebSocket URL (default: ws://localhost:8000)"
    )

    args = parser.parse_args()

    monitor = WebSocketStreamMonitor(coordinator_url=args.url)

    print("🚀 WebSocket Streaming Monitor")
    print(f"Job ID: {args.job_id}")
    print()

    try:
        await monitor.monitor_job(args.job_id, duration=args.duration)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")


if __name__ == "__main__":
    asyncio.run(main())
