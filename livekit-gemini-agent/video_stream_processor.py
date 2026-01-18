"""
Video processor that analyzes frames using GPT-4o-mini Vision API
and streams audio to Deepgram's live API for real-time transcription.

Transcripts and visual analysis happen in parallel for live updates.
"""

import base64
import json
import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Generator

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from video_store import insert_event, open_video_db, update_video_status

# Load environment
_env_path = os.path.join(os.path.dirname(__file__), ".env.local")
load_dotenv(_env_path)

# Configuration
FRAME_INTERVAL = float(os.getenv("STREAM_FRAME_INTERVAL", "3"))  # Analyze every N seconds
IMAGE_MAX_SIZE = int(os.getenv("IMAGE_MAX_SIZE", "512"))
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "70"))
MODEL_NAME = os.getenv("VIDEO_ACTION_MODEL", "gpt-4o-mini")

# Audio settings for extraction
AUDIO_SAMPLE_RATE = 16000

# System prompt for video analysis - event-focused, not frame-descriptive
SYSTEM_PROMPT = """You are a body cam footage event logger. Your job is to describe what is happening in each moment.

!!! HIGHEST PRIORITY - ALERTS !!!
If you see ANY of these, START your response with the alert in ALL CAPS:

WEAPONS:
- GUN DRAWN: "⚠️ GUN DRAWN! Officer/Subject draws firearm..."
- TASER DRAWN: "⚠️ TASER DRAWN! Officer deploys taser..."
- TASER FIRED: "⚠️ TASER FIRED! Taser discharged at subject..."
- SHOTS FIRED: "⚠️ SHOTS FIRED! Gunfire detected..."
- KNIFE/WEAPON: "⚠️ WEAPON! Subject brandishes knife/weapon..."
- GUN VISIBLE: "⚠️ GUN VISIBLE! Firearm seen on subject's person..."
- GUN POINTED: "⚠️ GUN POINTED! Weapon aimed at..."

CAMERA STATUS:
- CAMERA BLOCKED: "⚠️ CAMERA BLOCKED! View obstructed by hand/object/darkness..."
- CAMERA OBSCURED: "⚠️ CAMERA OBSCURED! Partial obstruction, limited visibility..."

PERSON DOWN:
- PERSON ON FLOOR: "⚠️ PERSON ON FLOOR! Individual lying on ground/floor..."
- PERSON DOWN: "⚠️ PERSON DOWN! Subject fallen/taken down..."
- PERSON PRONE: "⚠️ PERSON PRONE! Individual face-down on ground..."

ALSO CALL OUT:
- AGGRESSIVE ACTIONS: Lunging, striking, charging, fighting, resisting
- PHYSICAL ALTERCATION: Any physical contact between officer and subject

RULES:
- ALWAYS describe what people are doing: standing, walking, talking, gesturing, looking around
- Log positions: "Officer stands by driver door", "Subject seated in vehicle"
- Log interactions: conversations, handoffs, pointing, approaching, backing away
- NEVER start with "In this frame" - just state what's happening
- Use active voice: "Officer speaks with driver" not "The officer is speaking"
- Be concise: 1-2 sentences
- Use third person: 'the officer', 'the subject', 'the individual'

GOOD: "⚠️ TASER DRAWN! Officer draws taser, subject backs away with hands up."
GOOD: "⚠️ GUN DRAWN! Officer unholsters service weapon, takes cover position."
GOOD: "Officer stands at driver window speaking with occupant."
BAD: "No new activity" (always describe what you see, even if routine)

You are logging a continuous record of events - describe what's visible even during calm moments."""

# Separate prompt for scene descriptions - used at start and periodically
SCENE_PROMPT = """Describe the scene and people in plain text, no markdown or bullet points.

Include: Location type, then each person with their role, clothing colors/types, build, hair, and position.

Example format:
"Interior residence, stairway with yellow walls. Officer in dark blue uniform and tactical vest, medium build, hair in bun, standing on stairs. Male subject in light blue shirt and dark pants, slim build, short hair, at top of stairs facing officer."

Write as a single flowing paragraph. Be specific about clothing colors for identification."""

# How often to do full scene descriptions (in seconds)
SCENE_DESCRIPTION_INTERVAL = 30.0


@dataclass
class VideoEvent:
    """A single event from video processing."""
    offset_sec: float
    kind: str  # 'action', 'transcript', 'status'
    text: str
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "offset_sec": self.offset_sec,
            "kind": self.kind,
            "text": self.text,
            "timestamp": self.timestamp or datetime.now().isoformat(),
        }

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        data = json.dumps(self.to_dict())
        return f"data: {data}\n\n"


class VideoStreamProcessor:
    """Processes video and yields events for SSE streaming."""

    def __init__(self, video_id: str, video_path: str):
        self.video_id = video_id
        self.video_path = video_path
        self._client = OpenAI()
        self._event_queue: queue.Queue[VideoEvent | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._processing_thread: threading.Thread | None = None
        # Thread-safe transcript storage for real-time updates
        self._transcripts: list[tuple[float, str]] = []
        self._transcripts_lock = threading.Lock()

    def start(self) -> None:
        """Start processing in a background thread."""
        self._processing_thread = threading.Thread(
            target=self._process_video,
            daemon=True,
        )
        self._processing_thread.start()

    def stop(self) -> None:
        """Stop processing."""
        self._stop_event.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)

    def events(self) -> Generator[VideoEvent, None, None]:
        """Yield events as they become available (for SSE streaming)."""
        while True:
            try:
                event = self._event_queue.get(timeout=1.0)
                if event is None:  # Sentinel for end of stream
                    break
                yield event
            except queue.Empty:
                # Check if processing is done
                if self._stop_event.is_set():
                    break
                continue

    def _emit(self, offset: float, kind: str, text: str) -> None:
        """Emit an event to the queue and save to database."""
        event = VideoEvent(
            offset_sec=offset,
            kind=kind,
            text=text,
            timestamp=datetime.now().isoformat(),
        )
        self._event_queue.put(event)

        # Also save to database
        try:
            conn = open_video_db()
            insert_event(
                conn,
                video_id=self.video_id,
                offset_sec=offset,
                kind=kind,
                text=text,
            )
            conn.close()
        except Exception as e:
            print(f"[{self.video_id}] Failed to save event: {e}")

    def _add_transcript(self, offset: float, text: str) -> None:
        """Add a transcript to the thread-safe list."""
        with self._transcripts_lock:
            self._transcripts.append((offset, text))

    def _get_transcripts_in_window(self, current_time: float, window: float = 5.0) -> str:
        """Get transcripts within a time window of the current frame."""
        with self._transcripts_lock:
            relevant = []
            for t, text in self._transcripts:
                if current_time - window <= t <= current_time + 1.0:
                    relevant.append(text)
            return " ".join(relevant) if relevant else ""

    def _encode_frame(self, frame_bgr: np.ndarray) -> str:
        """Encode frame to base64 data URL."""
        h, w = frame_bgr.shape[:2]
        if w > IMAGE_MAX_SIZE or h > IMAGE_MAX_SIZE:
            scale = min(IMAGE_MAX_SIZE / w, IMAGE_MAX_SIZE / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY])
        if not ok:
            return ""
        data = base64.b64encode(buf).decode("ascii")
        return f"data:image/jpeg;base64,{data}"

    def _analyze_frame_with_audio(self, image_url: str, audio_context: str, visual_context: str = "") -> str:
        """Analyze a frame with audio transcript context."""
        prompt_parts = []

        if visual_context:
            prompt_parts.append(f"Previous event logged: {visual_context}")

        if audio_context:
            prompt_parts.append(f"Speech detected: \"{audio_context}\"")

        prompt_parts.append(
            "Describe what is happening. What are people doing? Any movement, gestures, or interaction? "
            "If truly nothing has changed from the previous log (same positions, no movement), say: No new activity"
        )

        prompt = "\n\n".join(prompt_parts)

        try:
            response = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=200,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[{self.video_id}] Frame analysis error: {e}")
            return ""

    def _analyze_scene(self, image_url: str) -> str:
        """Analyze a frame for scene and person descriptions."""
        try:
            response = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a scene and person description specialist for body cam footage review."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                            {"type": "text", "text": SCENE_PROMPT},
                        ],
                    },
                ],
                max_tokens=300,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[{self.video_id}] Scene analysis error: {e}")
            return ""

    async def _transcribe_with_deepgram(self, wav_path: str) -> None:
        """Stream audio to Deepgram for segmented transcription."""
        import asyncio
        try:
            import websockets
        except ImportError:
            print(f"[{self.video_id}] websockets not installed")
            return

        deepgram_key = os.getenv("DEEPGRAM_API_KEY", "")
        if not deepgram_key:
            print(f"[{self.video_id}] No DEEPGRAM_API_KEY, skipping transcription")
            return

        # Deepgram URL - use nova-2 with utterances for better segmentation
        url = (
            f"wss://api.deepgram.com/v1/listen?"
            f"model=nova-2&"
            f"punctuate=true&"
            f"utterances=true&"
            f"utt_split=0.8&"
            f"encoding=linear16&"
            f"sample_rate={AUDIO_SAMPLE_RATE}&"
            f"channels=1&"
            f"language=en"
        )

        headers = {"Authorization": f"Token {deepgram_key}"}

        print(f"[{self.video_id}] Connecting to Deepgram...")

        try:
            async with websockets.connect(url, additional_headers=headers) as ws:
                print(f"[{self.video_id}] Connected to Deepgram")

                async def send_audio():
                    """Send audio chunks to Deepgram."""
                    chunk_size = int(AUDIO_SAMPLE_RATE * 0.5) * 2  # 500ms chunks

                    with open(wav_path, "rb") as f:
                        f.seek(44)  # Skip WAV header
                        chunk_count = 0
                        while not self._stop_event.is_set():
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            await ws.send(chunk)
                            chunk_count += 1
                            await asyncio.sleep(0.02)  # Fast streaming

                    await ws.send(json.dumps({"type": "CloseStream"}))
                    print(f"[{self.video_id}] Sent {chunk_count} audio chunks")

                async def receive_transcripts():
                    """Receive and emit transcripts."""
                    count = 0
                    try:
                        async for message in ws:
                            if self._stop_event.is_set():
                                break

                            data = json.loads(message)

                            if data.get("type") == "Results":
                                channel = data.get("channel", {})
                                alternatives = channel.get("alternatives", [])

                                if alternatives:
                                    transcript = alternatives[0].get("transcript", "").strip()
                                    if transcript:
                                        count += 1
                                        start = data.get("start", 0)

                                        # Get speaker if available
                                        words = alternatives[0].get("words", [])
                                        if words and "speaker" in words[0]:
                                            speaker = words[0]["speaker"]
                                            text = f"Speaker {speaker + 1}: {transcript}"
                                        else:
                                            text = transcript

                                        print(f"[{self.video_id}] TRANSCRIPT #{count} @ {start:.1f}s: {text[:50]}...")
                                        self._add_transcript(start, text)
                                        self._emit(start, "transcript", text)

                    except websockets.exceptions.ConnectionClosed:
                        pass

                    print(f"[{self.video_id}] Deepgram: {count} transcripts received")

                await asyncio.gather(send_audio(), receive_transcripts())

        except Exception as e:
            print(f"[{self.video_id}] Deepgram error: {e}")
            import traceback
            traceback.print_exc()

    def _extract_audio(self) -> str | None:
        """Extract audio from video to a WAV file."""
        wav_path = f"{self.video_path}.wav"

        try:
            cmd = [
                "ffmpeg", "-y", "-i", self.video_path,
                "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
                wav_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return wav_path
        except Exception as e:
            print(f"[{self.video_id}] Audio extraction failed: {e}")
            return None

    def _process_video(self) -> None:
        """Main processing loop with parallel audio streaming and frame analysis."""
        conn = open_video_db()
        wav_path = None

        try:
            # Open video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self._emit(0, "status", "Error: Failed to open video")
                update_video_status(conn, self.video_id, "error")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            print(f"[{self.video_id}] Processing video: {duration:.1f}s @ {fps:.1f}fps")
            self._emit(0, "status", f"Processing video ({duration:.1f}s)")
            update_video_status(conn, self.video_id, "processing")

            # Extract audio first
            print(f"[{self.video_id}] Extracting audio...")
            self._emit(0, "status", "Extracting audio...")
            wav_path = self._extract_audio()

            # Transcribe audio using Deepgram for segmented transcripts
            transcribe_thread = None
            if wav_path:
                print(f"[{self.video_id}] Starting Deepgram transcription...")
                self._emit(0, "status", "Transcribing audio with Deepgram...")

                # Run async Deepgram transcription in a thread with its own event loop
                def run_deepgram_transcription():
                    import asyncio
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self._transcribe_with_deepgram(wav_path))
                        loop.close()
                    except Exception as e:
                        print(f"[{self.video_id}] Deepgram thread error: {e}")
                        import traceback
                        traceback.print_exc()

                transcribe_thread = threading.Thread(target=run_deepgram_transcription, daemon=True)
                transcribe_thread.start()
                print(f"[{self.video_id}] Deepgram transcription thread started")

            # Process frames at regular intervals (parallel with audio streaming)
            frame_interval_frames = int(fps * FRAME_INTERVAL)
            frame_count = 0
            last_context = ""
            first_frame_analyzed = False
            last_activity_time = 0.0  # Track when we last had real activity
            no_activity_start = None  # Track start of "no activity" period
            last_scene_time = -999.0  # Track when we last did a scene description

            self._emit(0, "status", "Analyzing video frames...")

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                frame_count += 1
                current_time = frame_count / fps

                # Analyze the FIRST frame immediately (at ~0.5s) to capture opening scene
                should_analyze = False
                is_first_frame = False
                if not first_frame_analyzed and frame_count >= int(fps * 0.5):
                    should_analyze = True
                    is_first_frame = True
                    first_frame_analyzed = True
                elif frame_count % frame_interval_frames == 0:
                    should_analyze = True

                if not should_analyze:
                    continue

                # Encode and analyze frame
                image_url = self._encode_frame(frame)
                if not image_url:
                    continue

                # Check if we should do a scene description (first frame or periodic)
                should_describe_scene = is_first_frame or (current_time - last_scene_time >= SCENE_DESCRIPTION_INTERVAL)

                if should_describe_scene:
                    scene_desc = self._analyze_scene(image_url)
                    if scene_desc:
                        self._emit(current_time, "scene", scene_desc)
                        print(f"[{self.video_id}] SCENE @ {current_time:.1f}s: {scene_desc[:80]}...")
                        last_scene_time = current_time

                # Get audio context for this time window (from live transcripts)
                audio_context = self._get_transcripts_in_window(current_time)

                description = self._analyze_frame_with_audio(image_url, audio_context, last_context)

                if description:
                    # Check if it's a "no activity" response
                    is_no_activity = self._is_no_activity_response(description)

                    if is_no_activity:
                        # Start or continue tracking no-activity period
                        if no_activity_start is None:
                            no_activity_start = last_activity_time if last_activity_time > 0 else current_time - FRAME_INTERVAL
                        print(f"[{self.video_id}] @ {current_time:.1f}s: (no activity)")
                    elif self._is_valid_response(description):
                        # Real activity - emit any pending no-activity period first
                        if no_activity_start is not None:
                            self._emit_no_activity_range(no_activity_start, current_time)
                            no_activity_start = None

                        # Emit the actual activity
                        self._emit(current_time, "action", description)
                        print(f"[{self.video_id}] @ {current_time:.1f}s: {description[:60]}...")
                        last_context = description
                        last_activity_time = current_time
                    else:
                        # Filtered but not "no activity" - log for debugging
                        print(f"[{self.video_id}] @ {current_time:.1f}s: (filtered) {description[:40]}...")

                # Progress update every 30 seconds
                if int(current_time) % 30 == 0 and int(current_time) > 0:
                    progress = (current_time / duration) * 100
                    self._emit(current_time, "status", f"Processing: {progress:.0f}% complete")

            # Emit any remaining no-activity period at the end
            if no_activity_start is not None:
                self._emit_no_activity_range(no_activity_start, duration)

            cap.release()

            # Wait for Deepgram transcription to finish
            if transcribe_thread and transcribe_thread.is_alive():
                print(f"[{self.video_id}] Waiting for Deepgram transcription to complete...")
                transcribe_thread.join(timeout=60.0)  # 60 second timeout for transcription

            self._emit(duration, "status", "Processing complete")
            update_video_status(conn, self.video_id, "done")
            print(f"[{self.video_id}] Processing complete")

        except Exception as e:
            print(f"[{self.video_id}] Processing error: {e}")
            import traceback
            traceback.print_exc()
            self._emit(0, "status", f"Error: {str(e)}")
            update_video_status(conn, self.video_id, "error")

        finally:
            conn.close()
            # Clean up audio file
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
            self._event_queue.put(None)  # Signal end of stream
            self._stop_event.set()

    def _is_no_activity_response(self, text: str) -> bool:
        """Check if response indicates no new activity - balanced matching."""
        text_lower = text.lower().strip()

        # Too short to be meaningful
        if len(text_lower) < 15:
            return True

        # Check if the response STARTS with a no-activity phrase
        # This catches "No new activity." but not "Officer stands still, no new activity since last update."
        no_activity_starts = [
            "no new activity",
            "no activity",
            "no change",
            "nothing new",
            "scene unchanged",
            "no movement",
        ]
        for phrase in no_activity_starts:
            if text_lower.startswith(phrase):
                return True

        return False

    def _emit_no_activity_range(self, start_time: float, end_time: float) -> None:
        """Emit a no-activity event showing the time range."""
        if end_time - start_time < FRAME_INTERVAL:
            return  # Skip very short periods

        # Format times as MM:SS
        def fmt(sec: float) -> str:
            m = int(sec // 60)
            s = int(sec % 60)
            return f"{m}:{s:02d}"

        text = f"{fmt(start_time)}-{fmt(end_time)} No new activity"
        self._emit(start_time, "no_activity", text)
        print(f"[{self.video_id}] No activity period: {fmt(start_time)}-{fmt(end_time)}")

    def _is_valid_response(self, text: str) -> bool:
        """Filter out invalid responses - minimal filtering."""
        text_lower = text.lower().strip()
        # Skip empty or too short
        if len(text.strip()) < 5:
            return False
        # Skip frame-descriptive prose that starts with certain patterns
        if text_lower.startswith("in this frame") or text_lower.startswith("the frame shows"):
            return False
        if text_lower.startswith("the image shows") or text_lower.startswith("this image"):
            return False
        return True


# Active processors for SSE streaming
_active_processors: dict[str, VideoStreamProcessor] = {}


def start_stream_processing(video_id: str, video_path: str) -> VideoStreamProcessor:
    """Start processing a video and return the processor for SSE streaming."""
    processor = VideoStreamProcessor(video_id, video_path)
    _active_processors[video_id] = processor
    processor.start()
    return processor


def get_stream_processor(video_id: str) -> VideoStreamProcessor | None:
    """Get an active processor."""
    return _active_processors.get(video_id)


def stop_stream_processing(video_id: str) -> None:
    """Stop processing a video."""
    processor = _active_processors.pop(video_id, None)
    if processor:
        processor.stop()
