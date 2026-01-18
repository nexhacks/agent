"""
Publish a video file (with audio) to a LiveKit room as if it were a live camera feed.
The realtime agent (main.py) will process the video frames and audio through the OpenAI Realtime API.

Usage:
    python publish_video.py path/to/video.mp4

This simulates a live video stream, so the agent processes it in real-time.
To process faster than real-time, set VIDEO_SPEED_MULTIPLIER (e.g., 2.0 for 2x speed).
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import uuid

import cv2
from dotenv import load_dotenv
from livekit import rtc
from livekit.api import LiveKitAPI
from livekit.protocol.agent_dispatch import CreateAgentDispatchRequest

try:
    from livekit import api
except ImportError:
    api = None


# Audio settings for OpenAI Realtime API
AUDIO_SAMPLE_RATE = 24000  # 24kHz for OpenAI Realtime
AUDIO_CHANNELS = 1  # Mono
AUDIO_FRAME_DURATION_MS = 20  # 20ms frames
AUDIO_SAMPLES_PER_FRAME = int(AUDIO_SAMPLE_RATE * AUDIO_FRAME_DURATION_MS / 1000)


def build_token(room_name: str, identity: str) -> str:
    token = os.getenv("LIVEKIT_TOKEN")
    if token:
        return token

    if api is None:
        raise RuntimeError("Set LIVEKIT_TOKEN or install livekit-server-sdk for token creation.")

    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]

    access_token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(
                room=room_name,
                room_join=True,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )
    return access_token.to_jwt()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Ensure it is set in .env.local or exported.")
    return value


def extract_audio_to_pcm(video_path: str, output_path: str) -> bool:
    """Extract audio from video to raw PCM format using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # 16-bit signed little-endian
        "-ar", str(AUDIO_SAMPLE_RATE),  # Sample rate
        "-ac", str(AUDIO_CHANNELS),  # Channels
        "-f", "s16le",  # Raw PCM format
        output_path,
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("ffmpeg not found - audio will not be streamed")
        return False


async def publish_video_file(video_path: str, loop_video: bool = False):
    env_path = os.path.join(os.path.dirname(__file__), ".env.local")
    load_dotenv(env_path)

    url = require_env("LIVEKIT_URL")
    agent_name = os.getenv("LIVEKIT_AGENT_NAME", "assistant")
    room_name = os.getenv("LIVEKIT_ROOM", f"video-{uuid.uuid4().hex[:8]}")
    identity = os.getenv("LIVEKIT_IDENTITY", f"video-publisher-{uuid.uuid4().hex[:8]}")
    speed_multiplier = float(os.getenv("VIDEO_SPEED_MULTIPLIER", "1.0"))

    token = build_token(room_name, identity)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize to reasonable dimensions for the API (max 1280x720)
    max_width, max_height = 1280, 720
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        width = int(width * scale)
        height = int(height * scale)

    print(f"Video: {video_path}")
    print(f"  Duration: {duration:.1f}s, FPS: {fps:.1f}, Size: {width}x{height}")
    print(f"  Speed: {speed_multiplier}x (effective FPS: {fps * speed_multiplier:.1f})")
    print(f"  Room: {room_name}")

    # Extract audio
    audio_data = None
    audio_temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
            audio_temp_path = f.name

        if extract_audio_to_pcm(video_path, audio_temp_path):
            with open(audio_temp_path, "rb") as f:
                audio_data = f.read()
            audio_samples = len(audio_data) // 2
            audio_duration = audio_samples / AUDIO_SAMPLE_RATE
            print(f"  Audio: {audio_duration:.1f}s extracted")
        else:
            print("  Audio: none (video may have no audio track)")
    finally:
        if audio_temp_path and os.path.exists(audio_temp_path):
            try:
                os.remove(audio_temp_path)
            except Exception:
                pass

    # Connect to room
    room = rtc.Room()

    @room.on("participant_connected")
    def _on_participant_connected(participant):
        print(f"Participant connected: {participant.identity}")

    await room.connect(url, token, rtc.RoomOptions(auto_subscribe=True))
    print("Connected to LiveKit.")

    # Create video source and track
    video_source = rtc.VideoSource(width, height)
    video_track = rtc.LocalVideoTrack.create_video_track("video-file", video_source)
    await room.local_participant.publish_track(
        video_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
    )
    print(f"Published video track {width}x{height}")

    # Create audio source and track if we have audio
    audio_source = None
    if audio_data:
        audio_source = rtc.AudioSource(AUDIO_SAMPLE_RATE, AUDIO_CHANNELS)
        audio_track = rtc.LocalAudioTrack.create_audio_track("audio-file", audio_source)
        await room.local_participant.publish_track(
            audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        print(f"Published audio track ({AUDIO_SAMPLE_RATE}Hz)")

    # Dispatch agent
    print(f"Requesting agent '{agent_name}' for room '{room_name}'...")
    api_client = LiveKitAPI(url=url)
    try:
        for attempt in range(1, 6):
            try:
                await api_client.agent_dispatch.create_dispatch(
                    CreateAgentDispatchRequest(agent_name=agent_name, room=room_name)
                )
                print("Agent dispatch created.")
                break
            except Exception as exc:
                if attempt == 5:
                    print(f"Warning: Agent dispatch failed: {exc}")
                    print("Make sure main.py agent is running!")
                else:
                    print(f"Agent dispatch attempt {attempt}/5 failed: {exc}")
                    await asyncio.sleep(1.0)
    finally:
        await api_client.aclose()

    # Allow agent to connect
    await asyncio.sleep(2.0)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def handle_stop():
        print("\nStopping...")
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, handle_stop)
        loop.add_signal_handler(signal.SIGTERM, handle_stop)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda *_: handle_stop())
        signal.signal(signal.SIGTERM, lambda *_: handle_stop())

    # Calculate frame interval (accounting for speed multiplier)
    frame_interval = 1.0 / (fps * speed_multiplier)
    frame_count = 0
    start_time = loop.time()

    # Audio streaming settings
    audio_frame_bytes = AUDIO_SAMPLES_PER_FRAME * 2  # 2 bytes per sample
    audio_frame_duration = AUDIO_FRAME_DURATION_MS / 1000.0
    audio_position = 0

    print("\nStreaming video+audio to agent... (Ctrl+C to stop)")

    # Audio streaming task
    async def stream_audio():
        nonlocal audio_position
        if not audio_source or not audio_data:
            return

        while not stop_event.is_set():
            current_time = (loop.time() - start_time) * speed_multiplier
            expected_audio_pos = int(current_time * AUDIO_SAMPLE_RATE * 2)

            # Stream audio frames to catch up to current video time
            while audio_position < expected_audio_pos and audio_position < len(audio_data):
                end_pos = min(audio_position + audio_frame_bytes, len(audio_data))
                chunk = audio_data[audio_position:end_pos]

                # Pad if needed
                if len(chunk) < audio_frame_bytes:
                    chunk = chunk + b'\x00' * (audio_frame_bytes - len(chunk))

                # Create audio frame
                audio_frame = rtc.AudioFrame(
                    data=chunk,
                    samples_per_channel=AUDIO_SAMPLES_PER_FRAME,
                    sample_rate=AUDIO_SAMPLE_RATE,
                    num_channels=AUDIO_CHANNELS,
                )
                await audio_source.capture_frame(audio_frame)
                audio_position += audio_frame_bytes

            await asyncio.sleep(audio_frame_duration / speed_multiplier)

    # Start audio streaming task
    audio_task = None
    if audio_source:
        audio_task = asyncio.create_task(stream_audio())

    try:
        while not stop_event.is_set():
            ok, frame_bgr = await asyncio.to_thread(cap.read)

            if not ok:
                if loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    audio_position = 0
                    start_time = loop.time()
                    print("Video looped.")
                    continue
                else:
                    print("Video playback complete.")
                    # Keep connection alive briefly so agent can finish processing
                    await asyncio.sleep(5.0)
                    break

            # Resize if needed
            if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                frame_bgr = cv2.resize(frame_bgr, (width, height))

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Create and publish frame
            video_frame = rtc.VideoFrame(
                width,
                height,
                rtc.VideoBufferType.RGB24,
                frame_rgb.tobytes(),
            )
            video_source.capture_frame(video_frame)

            frame_count += 1

            # Progress update every 100 frames
            if frame_count % 100 == 0:
                elapsed = loop.time() - start_time
                video_time = frame_count / fps
                print(f"  Progress: {video_time:.1f}s / {duration:.1f}s ({100 * video_time / duration:.1f}%)")

            # Pace the video playback
            expected_time = start_time + (frame_count * frame_interval)
            sleep_time = expected_time - loop.time()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    finally:
        if audio_task:
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass
        cap.release()
        await room.disconnect()
        print("Disconnected.")


def main():
    parser = argparse.ArgumentParser(
        description="Publish a video file (with audio) to LiveKit for realtime agent processing."
    )
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--loop", action="store_true", help="Loop the video continuously")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    asyncio.run(publish_video_file(args.video_path, loop_video=args.loop))


if __name__ == "__main__":
    main()
