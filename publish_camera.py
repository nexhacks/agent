import asyncio
import os
import signal
import uuid
import sys
import re
import shutil
import subprocess
from collections import deque
from datetime import datetime

import cv2
from dotenv import load_dotenv
import numpy as np
from event_log import get_event_logger
from livekit import rtc
from livekit.api import LiveKitAPI
from livekit.protocol.agent_dispatch import CreateAgentDispatchRequest
import sounddevice as sd

try:
    from livekit import api
except ImportError:  # pragma: no cover - optional dependency fallback
    api = None


def build_token() -> str:
    token = os.getenv("LIVEKIT_TOKEN")
    if token:
        return token

    if api is None:
        raise RuntimeError("Set LIVEKIT_TOKEN or install livekit-server-sdk for token creation.")

    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    room = os.getenv("LIVEKIT_ROOM", "demo")
    identity = os.getenv("LIVEKIT_IDENTITY", f"opencv-{uuid.uuid4().hex[:8]}")

    access_token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(
                room=room,
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

def _open_camera(index: int, backend: int | None) -> cv2.VideoCapture:
    if backend is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def _probe_camera(index: int, backend: int | None) -> tuple[bool, tuple[int, int] | None]:
    cap = _open_camera(index, backend)
    if not cap.isOpened():
        cap.release()
        return False, None
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False, None
    height, width, _ = frame.shape
    return True, (width, height)

def _list_avfoundation_devices() -> list[tuple[int, str]]:
    if sys.platform != "darwin":
        return []
    if shutil.which("ffmpeg") is None:
        return []
    cmd = ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return []
    text = (result.stderr or "") + (result.stdout or "")
    devices: list[tuple[int, str]] = []
    in_video = False
    for line in text.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            in_video = False
            continue
        if not in_video:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match:
            devices.append((int(match.group(1)), match.group(2).strip()))
    return devices


def _choose_camera_index() -> int:
    env_index = os.getenv("CAMERA_INDEX")
    if env_index is not None:
        return int(env_index)

    if sys.platform == "darwin" and os.getenv("DISABLE_CONTINUITY_CAMERA", "1") == "1":
        devices = _list_avfoundation_devices()
        if devices:
            print("Available cameras (avfoundation):")
            for idx, name in devices:
                print(f"  [{idx}] {name}")
            for idx, name in devices:
                lowered = name.lower()
                if "facetime" in lowered or "built-in" in lowered or "internal" in lowered:
                    print(f"Selected built-in camera: [{idx}] {name}")
                    return idx
            for idx, name in devices:
                lowered = name.lower()
                if "obs" in lowered or "virtual" in lowered or "continuity" in lowered or "iphone" in lowered:
                    continue
                print(f"Selected camera: [{idx}] {name}")
                return idx
            if sys.stdin.isatty():
                while True:
                    choice = input("Select camera index: ").strip()
                    if choice.isdigit():
                        return int(choice)
                    print("Please enter a numeric camera index.")
            raise RuntimeError(
                "Unable to auto-select a non-continuity camera. Set CAMERA_INDEX to a built-in webcam."
            )

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else None
    candidates = []
    max_index = 2 if sys.platform == "darwin" else 5
    for idx in range(max_index):
        ok, size = _probe_camera(idx, backend)
        if ok:
            candidates.append((idx, size))

    if not candidates:
        return 0

    if len(candidates) == 1 or not sys.stdin.isatty():
        return candidates[0][0]

    print("Available cameras:")
    for idx, size in candidates:
        label = f"{idx}"
        if size:
            label += f" ({size[0]}x{size[1]})"
        print(f"  [{idx}] {label}")

    while True:
        choice = input("Select camera index: ").strip()
        if choice.isdigit():
            return int(choice)
        print("Please enter a numeric camera index.")


async def main():
    base_dir = os.path.dirname(__file__)
    load_dotenv(os.path.join(base_dir, ".env"))
    load_dotenv(os.path.join(base_dir, ".env.local"), override=True)
    url = require_env("LIVEKIT_URL")
    token = build_token()
    agent_name = os.getenv("LIVEKIT_AGENT_NAME", "assistant")
    event_logger = get_event_logger()

    room = rtc.Room()
    @room.on("participant_connected")
    def _on_participant_connected(participant):
        print(f"Participant connected: {participant.identity}")

    @room.on("track_published")
    def _on_track_published(publication, participant):
        print(f"Track published: {publication.kind} by {participant.identity}")

    @room.on("local_track_published")
    def _on_local_track_published(publication, track):
        print(f"Local track published: {publication.kind} source={publication.source}")

    @room.on("transcription_received")
    def _on_transcription(segments, participant, publication):
        if os.getenv("SHOW_AGENT_TRANSCRIPT", "0") != "1":
            return
        # Print only remote (agent) transcriptions.
        if participant and participant.identity == room.local_participant.identity:
            return
        if not segments:
            return
        text = " ".join(seg.text for seg in segments if seg.text)
        if not text:
            return
        final = all(seg.final for seg in segments)
        prefix = "Agent (final)" if final else "Agent (partial)"
        print(f"{prefix}: {text}")

    await room.connect(url, token, rtc.RoomOptions(auto_subscribe=True))
    print("Connected to LiveKit.")

    audio_ready = asyncio.Event()
    video_ready = asyncio.Event()

    async def dispatch_when_ready() -> None:
        await audio_ready.wait()
        await video_ready.wait()
        print(f"Requesting agent '{agent_name}' for room '{room.name}'...")
        api_client = LiveKitAPI(url=url)
        try:
            for attempt in range(1, 6):
                try:
                    await api_client.agent_dispatch.create_dispatch(
                        CreateAgentDispatchRequest(agent_name=agent_name, room=room.name)
                    )
                    print("Agent dispatch created.")
                    break
                except Exception as exc:
                    if attempt == 5:
                        raise
                    print(f"Agent dispatch failed (attempt {attempt}/5): {exc}")
                    await asyncio.sleep(1.0)
        finally:
            await api_client.aclose()

    dispatch_task = asyncio.create_task(dispatch_when_ready())

    # Open the camera (prefer macOS AVFoundation backend).
    cam_index = _choose_camera_index()
    backend = None
    if os.getenv("CAMERA_BACKEND") == "avfoundation" or sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    cap = _open_camera(cam_index, backend)
    if backend is not None and not cap.isOpened():
        cap.release()
        cap = _open_camera(cam_index, None)
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open camera device {cam_index}. "
            "Check macOS camera permissions for your terminal and ensure no other app is using it."
        )
    target_width = int(os.getenv("CAMERA_WIDTH", "640"))
    target_height = int(os.getenv("CAMERA_HEIGHT", "360"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    print(f"Camera opened: index={cam_index} backend={'avfoundation' if backend else 'default'}")

    video_source = None
    video_track = None

    # Open default microphone and publish audio from a custom capture loop.
    sample_rate = 48000
    channels = 1
    frame_samples = 480  # 10ms at 48kHz
    audio_source = rtc.AudioSource(sample_rate, channels)
    audio_track = rtc.LocalAudioTrack.create_audio_track("mic", audio_source)
    await room.local_participant.publish_track(
        audio_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    print("Published microphone track.")

    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[rtc.AudioFrame] = asyncio.Queue(maxsize=50)
    transcribe_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=50)
    transcribe_enabled = os.getenv("MIC_TRANSCRIBE", "1") == "1"
    transcript_stdout = os.getenv("MIC_TRANSCRIPT_STDOUT", "0") == "1"
    transcribe_interval = float(os.getenv("MIC_TRANSCRIBE_INTERVAL", "2.0"))
    transcribe_window = float(os.getenv("MIC_TRANSCRIBE_WINDOW", "6.0"))
    transcribe_model = os.getenv("MIC_TRANSCRIBE_MODEL", "tiny")
    transcribe_rate = 16000
    downsample = sample_rate // transcribe_rate
    if sample_rate % transcribe_rate != 0:
        raise RuntimeError("Sample rate must be divisible by 16000 for transcription.")
    green = "\033[32m"
    reset = "\033[0m"

    def _audio_callback(indata, frames, time_info, status):
        if frames == 0:
            return
        if indata.ndim > 1:
            samples = indata[:, 0]
        else:
            samples = indata
        # Chunk into 10ms frames and enqueue for async capture.
        num_frames = frames // frame_samples
        for i in range(num_frames):
            start = i * frame_samples
            end = start + frame_samples
            chunk = samples[start:end]
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                samples_per_channel=frame_samples,
                sample_rate=sample_rate,
                num_channels=channels,
            )
            if not audio_queue.full() and not loop.is_closed():
                loop.call_soon_threadsafe(audio_queue.put_nowait, frame)
        if transcribe_enabled and not transcribe_queue.full() and not loop.is_closed():
            if downsample > 1:
                chunk = samples[::downsample]
            else:
                chunk = samples
            loop.call_soon_threadsafe(transcribe_queue.put_nowait, chunk.copy())

    input_stream = sd.InputStream(
        callback=_audio_callback,
        dtype="int16",
        channels=channels,
        samplerate=sample_rate,
        blocksize=frame_samples,
    )
    input_stream.start()
    audio_ready.set()

    async def pump_audio():
        await video_ready.wait()
        while True:
            frame = await audio_queue.get()
            await audio_source.capture_frame(frame)

    audio_task = asyncio.create_task(pump_audio())

    async def mic_transcribe_loop():
        if not transcribe_enabled:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("Mic transcript disabled: faster-whisper not installed.")
            return

        model = WhisperModel(transcribe_model, device="cpu", compute_type="int8")
        buffer = deque(maxlen=int(transcribe_window * transcribe_rate))
        last_text = ""
        last_run = 0.0
        while True:
            chunk = await transcribe_queue.get()
            buffer.extend(chunk)
            now = loop.time()
            if now - last_run < transcribe_interval:
                continue
            if len(buffer) < transcribe_rate:
                continue
            audio = np.array(buffer, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _info = await asyncio.to_thread(
                model.transcribe,
                audio,
                language="en",
                task="transcribe",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                without_timestamps=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if text and text != last_text:
                stamp = datetime.now().strftime("%H:%M:%S")
                if transcript_stdout:
                    print(f"{green}Transcript [{stamp}]: {text}{reset}")
                if event_logger:
                    event_logger.log_event(
                        kind="transcript",
                        text=text,
                        ts=datetime.now(),
                        source="mic",
                    )
                last_text = text
            last_run = now

    mic_task = asyncio.create_task(mic_transcribe_loop())

    async def pump_video():
        nonlocal video_source, video_track
        deadline = asyncio.get_running_loop().time() + 5.0
        width = height = None
        while True:
            ok, frame_bgr = await asyncio.to_thread(cap.read)
            if not ok:
                if asyncio.get_running_loop().time() > deadline and video_source is None:
                    raise RuntimeError(
                        f"Failed to read from camera device {cam_index}. "
                        "Try CAMERA_INDEX=1 or grant camera permissions in System Settings."
                    )
                await asyncio.sleep(0.05)
                continue
            if video_source is None:
                height, width = target_height, target_width
                video_source = rtc.VideoSource(width, height)
                video_track = rtc.LocalVideoTrack.create_video_track("camera", video_source)
                await room.local_participant.publish_track(
                    video_track,
                    rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
                )
                print(f"Published camera track {width}x{height}.")
                video_ready.set()
            if frame_bgr.shape[1] != target_width or frame_bgr.shape[0] != target_height:
                frame_bgr = cv2.resize(frame_bgr, (target_width, target_height))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            video_frame = rtc.VideoFrame(
                target_width,
                target_height,
                rtc.VideoBufferType.RGB24,
                frame_rgb.tobytes(),
            )
            video_source.capture_frame(video_frame)
            if os.getenv("SHOW_PREVIEW", "0") == "1":
                try:
                    cv2.imshow("LiveKit Camera Preview (press q to quit)", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise asyncio.CancelledError
                except cv2.error as exc:
                    print(f"Preview disabled (OpenCV GUI error): {exc}")
                    os.environ["SHOW_PREVIEW"] = "0"

    video_task = asyncio.create_task(pump_video())

    stop_event = asyncio.Event()

    def handle_stop():
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, handle_stop)
        loop.add_signal_handler(signal.SIGTERM, handle_stop)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda *_: handle_stop())
        signal.signal(signal.SIGTERM, lambda *_: handle_stop())

    try:
        await stop_event.wait()
    finally:
        dispatch_task.cancel()
        video_task.cancel()
        audio_task.cancel()
        mic_task.cancel()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            input_stream.stop()
            input_stream.close()
        except Exception:
            pass
        if event_logger:
            event_logger.close()
        await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
