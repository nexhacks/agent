"""
Realtime video processor that streams uploaded videos to LiveKit
for processing by the Realtime API agent.

This replaces the frame-by-frame gpt-4o-mini approach with true
realtime streaming through the OpenAI Realtime API.

Streams both video AND audio to give the model complete context.
"""

import asyncio
import os
import subprocess
import tempfile
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv
from livekit import rtc
from livekit.api import LiveKitAPI
from livekit.protocol.agent_dispatch import CreateAgentDispatchRequest

try:
    from livekit import api
except ImportError:
    api = None

from video_store import insert_event, open_video_db, update_video_status


# Audio settings for OpenAI Realtime API
AUDIO_SAMPLE_RATE = 24000  # 24kHz for OpenAI Realtime
AUDIO_CHANNELS = 1  # Mono
AUDIO_FRAME_DURATION_MS = 20  # 20ms frames
AUDIO_SAMPLES_PER_FRAME = int(AUDIO_SAMPLE_RATE * AUDIO_FRAME_DURATION_MS / 1000)


def _build_token(room_name: str, identity: str) -> str:
    if api is None:
        raise RuntimeError("Install livekit-server-sdk for token creation.")

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


def _extract_audio_to_pcm(video_path: str, output_path: str) -> bool:
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
        result = subprocess.run(
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


class RealtimeVideoProcessor:
    """Streams a video file (with audio) to LiveKit for realtime agent processing."""

    def __init__(self, video_id: str, video_path: str):
        self.video_id = video_id
        self.video_path = video_path
        self.room_name = f"video-{video_id}"
        self.identity = f"video-streamer-{video_id}"
        self._stop_event = asyncio.Event()
        self._room: rtc.Room | None = None
        self._task: asyncio.Task | None = None
        # Pause/resume state for coordinated responses
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Start unpaused
        self._video_offset = 0.0  # Current video position in seconds
        self._pause_start_offset = 0.0  # Video offset when paused

    async def start(self):
        """Start streaming the video to LiveKit."""
        self._task = asyncio.create_task(self._run())

    def stop(self):
        """Stop the video stream."""
        self._stop_event.set()

    async def wait(self):
        """Wait for processing to complete."""
        if self._task:
            await self._task

    async def _run(self):
        env_path = os.path.join(os.path.dirname(__file__), ".env.local")
        load_dotenv(env_path)

        url = os.environ.get("LIVEKIT_URL")
        if not url:
            print(f"[{self.video_id}] Missing LIVEKIT_URL")
            return

        agent_name = os.getenv("LIVEKIT_AGENT_NAME", "assistant")
        speed_multiplier = float(os.getenv("VIDEO_SPEED_MULTIPLIER", "1.0"))

        token = _build_token(self.room_name, self.identity)

        # Open video file
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[{self.video_id}] Failed to open video: {self.video_path}")
            update_video_status(open_video_db(), self.video_id, "error")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize to reasonable dimensions
        max_width, max_height = 1280, 720
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            width = int(width * scale)
            height = int(height * scale)

        print(f"[{self.video_id}] Streaming video: {duration:.1f}s @ {fps:.1f}fps, {width}x{height}")

        # Extract audio to temporary PCM file
        audio_data = None
        audio_temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
                audio_temp_path = f.name

            if _extract_audio_to_pcm(self.video_path, audio_temp_path):
                with open(audio_temp_path, "rb") as f:
                    audio_data = f.read()
                audio_samples = len(audio_data) // 2  # 16-bit = 2 bytes per sample
                audio_duration = audio_samples / AUDIO_SAMPLE_RATE
                print(f"[{self.video_id}] Audio extracted: {audio_duration:.1f}s")
            else:
                print(f"[{self.video_id}] No audio extracted (video may have no audio track)")
        finally:
            if audio_temp_path and os.path.exists(audio_temp_path):
                try:
                    os.remove(audio_temp_path)
                except Exception:
                    pass

        # Connect to room
        self._room = rtc.Room()
        conn = open_video_db()

        # Listen for transcriptions from the agent
        @self._room.on("transcription_received")
        def _on_transcription(segments, participant, publication):
            if not segments:
                return
            # Only capture agent transcriptions
            if participant and participant.identity == self.identity:
                return

            for seg in segments:
                if not seg.text or not seg.text.strip():
                    continue
                # Calculate approximate offset based on elapsed time
                offset = (asyncio.get_event_loop().time() - start_time) / speed_multiplier
                insert_event(
                    conn,
                    video_id=self.video_id,
                    offset_sec=offset,
                    kind="action_realtime",
                    text=seg.text.strip(),
                )
                print(f"[{self.video_id}] Action @ {offset:.1f}s: {seg.text.strip()[:50]}...")

        try:
            await self._room.connect(url, token, rtc.RoomOptions(auto_subscribe=True))
            print(f"[{self.video_id}] Connected to LiveKit room: {self.room_name}")

            # Register RPC handlers for pause/resume coordination
            @self._room.local_participant.register_rpc_method("pause_stream")
            async def pause_stream(data: rtc.RpcInvocationData) -> str:
                if not self._paused:
                    self._paused = True
                    self._pause_event.clear()
                    self._pause_start_offset = self._video_offset
                    print(f"[{self.video_id}] Stream PAUSED at {self._video_offset:.1f}s")
                return f"paused at {self._video_offset:.1f}s"

            @self._room.local_participant.register_rpc_method("resume_stream")
            async def resume_stream(data: rtc.RpcInvocationData) -> str:
                if self._paused:
                    self._paused = False
                    self._pause_event.set()
                    print(f"[{self.video_id}] Stream RESUMED, catching up from {self._pause_start_offset:.1f}s")
                return f"resumed from {self._pause_start_offset:.1f}s"

            @self._room.local_participant.register_rpc_method("get_video_offset")
            async def get_video_offset(data: rtc.RpcInvocationData) -> str:
                return f"{self._video_offset:.2f}"

            # Create video source and track
            video_source = rtc.VideoSource(width, height)
            video_track = rtc.LocalVideoTrack.create_video_track("video", video_source)
            await self._room.local_participant.publish_track(
                video_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
            )
            print(f"[{self.video_id}] Published video track")

            # Create audio source and track if we have audio
            audio_source = None
            if audio_data:
                audio_source = rtc.AudioSource(AUDIO_SAMPLE_RATE, AUDIO_CHANNELS)
                audio_track = rtc.LocalAudioTrack.create_audio_track("audio", audio_source)
                await self._room.local_participant.publish_track(
                    audio_track,
                    rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
                )
                print(f"[{self.video_id}] Published audio track ({AUDIO_SAMPLE_RATE}Hz)")

            # Dispatch agent
            api_client = LiveKitAPI(url=url)
            try:
                await api_client.agent_dispatch.create_dispatch(
                    CreateAgentDispatchRequest(agent_name=agent_name, room=self.room_name)
                )
                print(f"[{self.video_id}] Agent dispatched")
            except Exception as exc:
                print(f"[{self.video_id}] Agent dispatch warning: {exc}")
            finally:
                await api_client.aclose()

            # Wait for agent to connect
            await asyncio.sleep(2.0)

            # Prepare audio streaming
            audio_frame_bytes = AUDIO_SAMPLES_PER_FRAME * 2  # 2 bytes per sample
            audio_frame_duration = AUDIO_FRAME_DURATION_MS / 1000.0
            audio_position = 0

            # Stream video and audio
            frame_interval = 1.0 / (fps * speed_multiplier)
            frame_count = 0
            start_time = asyncio.get_event_loop().time()

            update_video_status(conn, self.video_id, "streaming")

            # Audio streaming task
            async def stream_audio():
                nonlocal audio_position
                if not audio_source or not audio_data:
                    return

                while not self._stop_event.is_set():
                    current_time = (asyncio.get_event_loop().time() - start_time) * speed_multiplier
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
                # Catch-up speed when resuming from pause (10x normal)
                catchup_multiplier = 10.0

                while not self._stop_event.is_set():
                    # Wait if paused
                    if self._paused:
                        await self._pause_event.wait()
                        # After resume, we need to catch up
                        # Adjust start_time to account for pause duration
                        continue

                    ok, frame_bgr = await asyncio.to_thread(cap.read)

                    if not ok:
                        print(f"[{self.video_id}] Video complete")
                        await asyncio.sleep(3.0)  # Allow agent to finish
                        break

                    # Update video offset
                    frame_count += 1
                    self._video_offset = frame_count / fps

                    # Resize if needed
                    if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                        frame_bgr = cv2.resize(frame_bgr, (width, height))

                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    # Publish frame
                    video_frame = rtc.VideoFrame(
                        width, height,
                        rtc.VideoBufferType.RGB24,
                        frame_rgb.tobytes(),
                    )
                    video_source.capture_frame(video_frame)

                    # Determine playback speed: normal or catch-up
                    elapsed_real_time = asyncio.get_event_loop().time() - start_time
                    expected_video_time = elapsed_real_time * speed_multiplier

                    # If we're behind (just resumed from pause), catch up fast
                    if self._video_offset < expected_video_time - 0.5:
                        # Catch-up mode: minimal delay
                        current_speed = speed_multiplier * catchup_multiplier
                    else:
                        # Normal mode
                        current_speed = speed_multiplier

                    # Pace the playback
                    frame_interval_adjusted = 1.0 / (fps * current_speed)
                    expected_time = start_time + (frame_count * (1.0 / (fps * speed_multiplier)))
                    sleep_time = expected_time - asyncio.get_event_loop().time()

                    # In catch-up mode, use minimal sleep
                    if current_speed > speed_multiplier:
                        sleep_time = min(sleep_time, 0.001)

                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            finally:
                if audio_task:
                    audio_task.cancel()
                    try:
                        await audio_task
                    except asyncio.CancelledError:
                        pass

            update_video_status(conn, self.video_id, "done")

        except Exception as exc:
            print(f"[{self.video_id}] Error: {exc}")
            import traceback
            traceback.print_exc()
            update_video_status(conn, self.video_id, "error")
        finally:
            cap.release()
            conn.close()
            if self._room:
                await self._room.disconnect()


# Track active processors
_active_processors: dict[str, RealtimeVideoProcessor] = {}


async def start_realtime_processing(video_id: str, video_path: str) -> RealtimeVideoProcessor:
    """Start realtime processing for a video."""
    processor = RealtimeVideoProcessor(video_id, video_path)
    _active_processors[video_id] = processor
    await processor.start()
    return processor


def get_processor(video_id: str) -> RealtimeVideoProcessor | None:
    """Get an active processor by video ID."""
    return _active_processors.get(video_id)
