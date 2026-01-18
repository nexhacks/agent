#!/usr/bin/env python3
"""
Test script for the new SSE video stream processor.
Run this to verify video processing works without needing the frontend.

Usage:
    python test_stream.py path/to/video.mp4
"""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env.local"))

from video_stream_processor import VideoStreamProcessor


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_stream.py <video_path>")
        print("Example: python test_stream.py uploads/test.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing video: {video_path}")
    print(f"Video ID: {video_id}")
    print("=" * 60)
    print()

    # Create processor and start
    processor = VideoStreamProcessor(video_id, video_path)
    processor.start()

    # Stream events to console
    try:
        for event in processor.events():
            offset = f"[{event.offset_sec:6.1f}s]"
            kind = f"[{event.kind:12}]"

            # Color coding
            if event.kind == "action":
                color = "\033[32m"  # Green
            elif event.kind == "transcript":
                color = "\033[36m"  # Cyan
            elif event.kind == "status":
                color = "\033[33m"  # Yellow
            else:
                color = ""
            reset = "\033[0m"

            print(f"{color}{offset} {kind} {event.text}{reset}")
    except KeyboardInterrupt:
        print("\n\nStopping...")
        processor.stop()

    print()
    print("=" * 60)
    print("Processing complete!")


if __name__ == "__main__":
    main()
