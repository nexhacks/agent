"""
Video processor that analyzes frames using GPT-4o-mini Vision API
with parallel audio processing for:
- Gunshot/taser detection using YAMNet ML model (with heuristic fallback)
- Speech transcription using OpenAI Whisper and/or Deepgram

All processing happens in parallel for real-time SSE streaming:
- Video frames analyzed every 3 seconds
- Audio gunshot detection runs concurrently
- Transcription runs concurrently (OpenAI Whisper + Deepgram)

Configuration via environment variables:
- TRANSCRIPTION_PROVIDER: 'openai', 'deepgram', 'both', or 'none' (default: 'both')
- DEEPGRAM_API_KEY: Required for Deepgram transcription
- OPENAI_API_KEY: Required for both vision analysis and Whisper transcription
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

from video_store import insert_event, insert_screenshot, open_video_db, update_video_status

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

# Gunshot/impulsive sound detection settings (fallback if YAMNet unavailable)
GUNSHOT_CHUNK_MS = 100  # Analyze in 100ms windows
GUNSHOT_AMPLITUDE_THRESHOLD = 0.80  # Raised to filter false positives (0.71-0.74) while catching real shots (0.86-0.93)
GUNSHOT_FREQ_MIN = 2500  # Hz - raised to filter out false positives at 2490Hz
GUNSHOT_FREQ_MAX = 2600  # Hz - real gunshots observed at 2515-2558 Hz
TASER_FREQ_THRESHOLD = 3500  # Hz - tasers have high frequency (observed: 3919Hz)
GUNSHOT_COOLDOWN = 0.3  # Seconds between alerts (gunshots can be rapid)

# Impulsiveness detection - gunshots have very fast attack, yelling builds slowly
ATTACK_TIME_THRESHOLD_MS = 10  # Gunshots reach peak within 10ms (yelling takes 50-100ms+)
SOUND_DURATION_MAX_MS = 150  # Gunshots are very short (<150ms), yelling is sustained
CREST_FACTOR_THRESHOLD = 3.0  # Peak-to-RMS ratio (gunshots ~4-6, yelling ~2-3)

# YAMNet ML-based detection settings
YAMNET_CONFIDENCE_THRESHOLD = 0.15  # Lowered to catch more potential gunshots
YAMNET_WINDOW_SEC = 0.96  # YAMNet expects ~1 second windows
USE_YAMNET = True  # Enable YAMNet ML detection (falls back to heuristics if unavailable)
YAMNET_DEBUG = True  # Print top predictions for debugging

# YAMNet class indices for relevant sounds (from AudioSet ontology)
# See: https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv
YAMNET_GUNSHOT_CLASSES = {
    # Gunfire related
    427: "Gunshot, gunfire",
    428: "Machine gun",
    429: "Fusillade",
    430: "Artillery fire",
    # Explosion related
    426: "Explosion",
    494: "Bang",  # Generic bang sound
    # Cap gun / toy gun (sometimes detected)
    431: "Cap gun",
}

# Minimum activity update interval - force emit even during quiet periods
MIN_ACTIVITY_INTERVAL = 10.0  # At least one activity update every 10 seconds

# Transcription settings
TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER", "both")  # 'openai', 'deepgram', 'both', or 'none'
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
TRANSCRIPTION_CHUNK_SEC = 30  # Process audio in 30-second chunks for transcription

# Screenshot capture settings for police reports
SCREENSHOT_PERIODIC_INTERVAL = 30.0  # Capture periodic screenshot every 30 seconds
CRITICAL_KEYWORDS = [
    "GUN DRAWN", "TASER DRAWN", "TASER FIRED", "SHOTS FIRED",
    "WEAPON!", "GUN VISIBLE", "GUN POINTED",
    "CAMERA BLOCKED", "CAMERA OBSCURED",
    "PERSON ON FLOOR", "PERSON DOWN", "PERSON PRONE",
    "AUDIO: SHOTS FIRED", "AUDIO: TASER FIRED", "AUDIO: EXPLOSION",
]

# Critical presence tracking (avoid repeating the same warning every frame)
CRITICAL_PRESENCE_CLEAR_SEC = float(os.getenv("CRITICAL_PRESENCE_CLEAR_SEC", "6"))
CRITICAL_CATEGORY_KEYWORDS = {
    "gun": ["GUN DRAWN", "GUN VISIBLE", "GUN POINTED", "WEAPON!"],
    "taser": ["TASER DRAWN", "TASER FIRED"],
    "camera": ["CAMERA BLOCKED", "CAMERA OBSCURED"],
    "person_down": ["PERSON ON FLOOR", "PERSON DOWN", "PERSON PRONE"],
    # Shots fired is treated as a discrete event, not persistent presence
}

# System prompt for video analysis - describe what's happening
SYSTEM_PROMPT = """You are analyzing FIRST-PERSON body camera footage worn by a police officer.

PERSPECTIVE AWARENESS:
- This is POV footage from a camera on the OFFICER'S chest/shoulder
- The officer wearing the camera is NOT visible except for their hands/arms/weapon
- Camera movement = officer movement (walking, turning, taking cover)
- The officer's voice is typically the closest/clearest audio
- Everyone else visible in frame are SUBJECTS, CIVILIANS, or OTHER OFFICERS

!!! CRITICAL ALERTS - ALWAYS START WITH THESE !!!
If you see ANY of these, START with the alert:
- GUN DRAWN/VISIBLE/POINTED: "⚠️ GUN DRAWN! Officer draws weapon..." or "⚠️ GUN VISIBLE! Subject has weapon..."
- TASER DRAWN/FIRED: "⚠️ TASER FIRED! ..."
- SHOTS FIRED: "⚠️ SHOTS FIRED! ..."
- WEAPON VISIBLE: "⚠️ WEAPON! ..."
- PERSON DOWN/PRONE: "⚠️ PERSON DOWN! ..."
- CAMERA BLOCKED: "⚠️ CAMERA BLOCKED! ..."

DESCRIBE BOTH:
1. OFFICER (camera wearer) actions - inferred from:
   - Hands/arms visible in frame (reaching, pointing, holding weapon)
   - Camera movement (approaching, retreating, turning)
   - Voice/commands given (if audio provided)

2. SUBJECT(S) actions - what others are doing:
   - Their movements, positions, compliance
   - Responses to officer commands
   - Demeanor and behavior

STYLE:
- 2-4 sentences: first describe OFFICER action, then SUBJECT response
- Active voice: "Officer approaches vehicle. Subject exits with hands raised."
- Be specific about positions, movements, and compliance level
- Distinguish officer's weapon vs subject's weapon clearly"""

# Separate prompt for scene descriptions - used at start and periodically
SCENE_PROMPT = """Describe the scene from this FIRST-PERSON body camera footage. No markdown or bullet points.

REMEMBER: This is POV from the officer's body camera. The camera-wearing officer is NOT visible (except their hands/weapon).

Structure your description:
1. LOCATION: Type of location, lighting, key features
2. OFFICER (camera wearer): Only describe what's visible - hands, weapon drawn/holstered, any equipment visible
3. SUBJECTS/OTHERS: Each person visible with role (subject, civilian, backup officer), clothing colors, build, position relative to camera

Example format:
"Nighttime traffic stop on residential street, patrol vehicle lights illuminating scene. Officer's right hand visible resting on holstered weapon. Male subject (driver) standing outside white sedan, Hispanic male, early 30s, wearing gray hoodie and jeans, hands visible at sides, approximately 8 feet from camera. Female passenger remaining seated in vehicle, blonde hair, blue top visible through window."

Write as a flowing paragraph. Be specific about clothing colors and positions for identification."""

# Scene description settings
ENABLE_SCENE_DESCRIPTIONS = os.getenv("ENABLE_SCENE_DESCRIPTIONS", "1") == "1"
SCENE_DESCRIPTION_INTERVAL = 30.0  # How often to do full scene descriptions (in seconds)


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
        # Buffer for recent transcripts (for context during frame analysis)
        self._recent_transcripts: list[tuple[float, str]] = []  # (offset_sec, text)
        self._transcript_lock = threading.Lock()
        # Track active critical presence (gun/camera blocked/etc)
        self._active_critical: dict[str, float] = {}

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

        # If this is a transcript, add to recent buffer for frame context
        if kind == "transcript":
            self._add_transcript_to_buffer(offset, text)

    def _refresh_active_critical(self, current_time: float) -> None:
        if not hasattr(self, "_active_critical"):
            self._active_critical = {}
            return
        to_clear = [
            category
            for category, last_seen in self._active_critical.items()
            if current_time - last_seen >= CRITICAL_PRESENCE_CLEAR_SEC
        ]
        for category in to_clear:
            self._active_critical.pop(category, None)

    def _extract_critical_categories(self, text: str) -> set[str]:
        text_upper = text.upper()
        categories = set()
        for category, keywords in CRITICAL_CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_upper:
                    categories.add(category)
                    break
        return categories

    def _strip_active_critical_prefixes(self, text: str, categories: set[str]) -> str:
        cleaned = text
        for category in categories:
            keywords = CRITICAL_CATEGORY_KEYWORDS.get(category, [])
            for keyword in keywords:
                pattern = r"^\\s*⚠️\\s*" + re.escape(keyword) + r"!\\s*"
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _add_transcript_to_buffer(self, offset_sec: float, text: str) -> None:
        """Add a transcript segment to the recent buffer."""
        with self._transcript_lock:
            # Remove provider prefix for cleaner context
            clean_text = text
            for prefix in ["[OpenAI] ", "[Deepgram] "]:
                if clean_text.startswith(prefix):
                    clean_text = clean_text[len(prefix):]
                    break
            self._recent_transcripts.append((offset_sec, clean_text))
            # Keep only last 30 seconds of transcripts
            cutoff = offset_sec - 30.0
            self._recent_transcripts = [
                (t, txt) for t, txt in self._recent_transcripts if t >= cutoff
            ]

    def _get_recent_transcript_context(self, current_time: float, lookback_sec: float = 10.0) -> str:
        """Get transcript text from the last N seconds for context."""
        with self._transcript_lock:
            cutoff = current_time - lookback_sec
            recent = [txt for t, txt in self._recent_transcripts if t >= cutoff and t <= current_time]
            if not recent:
                return ""
            return " ".join(recent)


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

    def _is_critical_event(self, text: str) -> bool:
        """Check if text contains critical keywords that warrant a screenshot."""
        text_upper = text.upper()
        for keyword in CRITICAL_KEYWORDS:
            if keyword in text_upper:
                return True
        return False

    def _capture_screenshot(self, frame_bgr: np.ndarray, offset_sec: float, trigger_type: str) -> None:
        """Capture and save a screenshot to the database."""
        try:
            # Encode frame at higher quality for screenshots
            h, w = frame_bgr.shape[:2]
            max_size = 800  # Higher resolution for report screenshots
            if w > max_size or h > max_size:
                scale = min(max_size / w, max_size / h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                print(f"[{self.video_id}] Failed to encode screenshot")
                return

            image_data = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"

            conn = open_video_db()
            screenshot_id = insert_screenshot(
                conn,
                video_id=self.video_id,
                offset_sec=offset_sec,
                trigger_type=trigger_type,
                image_data=image_data,
            )
            conn.close()
            print(f"[{self.video_id}] SCREENSHOT #{screenshot_id} @ {offset_sec:.1f}s ({trigger_type})")
        except Exception as e:
            print(f"[{self.video_id}] Failed to capture screenshot: {e}")

    def _analyze_frame_with_audio(
        self,
        image_url: str,
        visual_context: str = "",
        audio_context: str = "",
        full_description: bool = False
    ) -> str:
        """
        Analyze a frame focusing on CHANGES from previous state.

        Args:
            image_url: Base64 encoded frame
            visual_context: Previous visual description
            audio_context: Recent transcript/dialogue
            full_description: If True, give full scene description (for periodic updates)
        """
        prompt_parts = []

        if full_description:
            # FULL DESCRIPTION MODE - used periodically or for first frame
            prompt_parts.append(
                "Provide a FULL scene description (4-5 sentences). Remember this is FIRST-PERSON body cam:\n"
                "- OFFICER (camera wearer): What are their hands doing? Weapon drawn/holstered? Movement?\n"
                "- SUBJECTS: Position, actions, demeanor of each person visible\n"
                "- ENVIRONMENT: Location type, lighting, obstacles, vehicles\n"
                "- INTERACTION: How are officer and subjects engaging?"
            )
            if audio_context:
                prompt_parts.append(f'AUDIO CONTEXT (officer or subject speaking): "{audio_context}"')
                prompt_parts.append("Who is speaking? How are others responding to the dialogue/commands?")
            max_tokens = 400
        else:
            # INCREMENTAL MODE - describe current state, focusing on what's different
            if visual_context:
                prompt_parts.append(f"PREVIOUS: {visual_context}")
                prompt_parts.append(
                    "Describe what is happening NOW. Focus on changes in:\n"
                    "- Officer's position/actions (hands, movement, weapon)\n"
                    "- Subject's response/compliance/movement"
                )
            else:
                prompt_parts.append(
                    "Describe what is happening. Remember: camera = officer's POV.\n"
                    "What is the OFFICER doing (hands visible, movement)?\n"
                    "What are SUBJECTS doing (position, compliance)?"
                )

            if audio_context:
                prompt_parts.append(f'AUDIO (spoken by officer or subject): "{audio_context}"')
                prompt_parts.append(
                    "Identify who is speaking (officer giving command? subject responding?).\n"
                    "Describe physical reactions to what was said."
                )
                # Always describe something when there's audio - it's context worth noting
                max_tokens = 250
            else:
                prompt_parts.append(
                    "If the scene is essentially static with no movement from officer or subjects, say: NO_CHANGE"
                )
                max_tokens = 180

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
                max_tokens=max_tokens,
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

    def _load_yamnet(self):
        """Load YAMNet model (cached after first load)."""
        if not hasattr(self, '_yamnet_model'):
            try:
                print(f"[{self.video_id}] Importing tensorflow_hub...")
                import tensorflow_hub as hub
                print(f"[{self.video_id}] Loading YAMNet model from TensorFlow Hub...")
                print(f"[{self.video_id}] (This may take a minute on first run as the model downloads)")
                self._yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
                print(f"[{self.video_id}] YAMNet model loaded successfully!")
            except ImportError as e:
                print(f"[{self.video_id}] tensorflow_hub not installed: {e}")
                print(f"[{self.video_id}] Run: pip install tensorflow tensorflow-hub")
                self._yamnet_model = None
            except Exception as e:
                print(f"[{self.video_id}] Failed to load YAMNet: {e}")
                import traceback
                traceback.print_exc()
                self._yamnet_model = None
        return self._yamnet_model

    def _detect_gunshots(self, wav_path: str) -> None:
        """Analyze audio for gunshots using YAMNet ML model with heuristic fallback."""
        import wave

        print(f"[{self.video_id}] Starting gunshot detection on: {wav_path}")

        try:
            with wave.open(wav_path, "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                duration = n_frames / sample_rate
                raw_data = wf.readframes(n_frames)

            print(f"[{self.video_id}] Audio: {duration:.1f}s, {sample_rate}Hz, {n_channels}ch, {sample_width*8}bit")

            # Convert to numpy array
            if sample_width == 2:
                audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 1:
                audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            else:
                print(f"[{self.video_id}] Unsupported audio format: {sample_width} bytes")
                return

            # Convert stereo to mono
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            # Audio stats
            peak = np.max(np.abs(audio))
            rms = np.sqrt(np.mean(audio**2))
            print(f"[{self.video_id}] Audio stats: peak={peak:.3f}, rms={rms:.3f}, samples={len(audio)}")

            # Try YAMNet first
            if USE_YAMNET:
                print(f"[{self.video_id}] Attempting YAMNet ML detection...")
                yamnet_model = self._load_yamnet()
                if yamnet_model is not None:
                    self._detect_gunshots_yamnet(audio, sample_rate, yamnet_model)
                    return
                else:
                    print(f"[{self.video_id}] YAMNet model failed to load, falling back to heuristics")

            # Fallback to heuristic detection
            print(f"[{self.video_id}] Using heuristic gunshot detection")
            self._detect_gunshots_heuristic(audio, sample_rate)

        except Exception as e:
            print(f"[{self.video_id}] Gunshot detection error: {e}")
            import traceback
            traceback.print_exc()

    def _detect_gunshots_yamnet(self, audio: np.ndarray, sample_rate: int, model) -> None:
        """Detect gunshots using YAMNet ML model."""
        # Load class names for debugging
        class_names = self._get_yamnet_class_names()

        # YAMNet expects 16kHz audio
        if sample_rate != 16000:
            from scipy import signal
            print(f"[{self.video_id}] Resampling audio from {sample_rate}Hz to 16000Hz...")
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples)
            sample_rate = 16000

        # Process in sliding windows
        window_samples = int(sample_rate * YAMNET_WINDOW_SEC)
        hop_samples = window_samples // 2  # 50% overlap
        last_alert_time = -GUNSHOT_COOLDOWN
        alert_count = 0
        loud_window_count = 0

        audio_duration = len(audio) / sample_rate
        print(f"[{self.video_id}] YAMNet: Analyzing {audio_duration:.1f}s of audio...")
        print(f"[{self.video_id}] YAMNet: Window={YAMNET_WINDOW_SEC}s, Hop={hop_samples/sample_rate:.2f}s, Threshold={YAMNET_CONFIDENCE_THRESHOLD}")

        for i in range(0, len(audio) - window_samples, hop_samples):
            if self._stop_event.is_set():
                break

            chunk = audio[i:i + window_samples]
            current_time = i / sample_rate
            peak_amplitude = np.max(np.abs(chunk))

            # Skip very quiet sections
            if peak_amplitude < 0.1:
                continue

            loud_window_count += 1

            # Check cooldown
            if current_time - last_alert_time < GUNSHOT_COOLDOWN:
                continue

            # Run YAMNet inference
            try:
                scores, embeddings, spectrogram = model(chunk)
                scores = scores.numpy()

                # Average scores across all frames in this window
                mean_scores = np.mean(scores, axis=0)

                # Debug: print top 5 predictions for loud sounds (peak > 0.5)
                if YAMNET_DEBUG and peak_amplitude > 0.5:
                    top_indices = np.argsort(mean_scores)[-5:][::-1]
                    top_preds = [(class_names.get(idx, f"class_{idx}"), mean_scores[idx]) for idx in top_indices]
                    print(f"[{self.video_id}] YAMNet @ {current_time:.1f}s (peak={peak_amplitude:.2f}): "
                          f"{', '.join([f'{name}:{conf:.2f}' for name, conf in top_preds])}")

                    # Also show gunshot-related class scores
                    gunshot_scores = [(YAMNET_GUNSHOT_CLASSES[idx], mean_scores[idx])
                                     for idx in YAMNET_GUNSHOT_CLASSES.keys()]
                    print(f"[{self.video_id}]   Gunshot classes: {', '.join([f'{name}:{conf:.3f}' for name, conf in gunshot_scores])}")

                # Check for gunshot-related classes
                for class_idx, class_name in YAMNET_GUNSHOT_CLASSES.items():
                    confidence = mean_scores[class_idx]

                    if confidence >= YAMNET_CONFIDENCE_THRESHOLD:
                        # Determine sound type
                        if class_idx in [427, 428, 429, 430, 431]:  # Gunshot classes
                            sound_type = "SHOTS FIRED"
                            description = f"ML detected: {class_name} (confidence={confidence:.2f})"
                        elif class_idx == 426:  # Explosion
                            sound_type = "EXPLOSION"
                            description = f"ML detected: {class_name} (confidence={confidence:.2f})"
                        elif class_idx == 494:  # Bang
                            sound_type = "SHOTS FIRED"
                            description = f"ML detected: {class_name} (confidence={confidence:.2f})"
                        else:
                            continue  # Skip non-critical detections

                        alert_count += 1
                        last_alert_time = current_time
                        alert_text = f"⚠️ AUDIO: {sound_type}! {description}"
                        self._emit(current_time, "action", alert_text)
                        print(f"[{self.video_id}] *** ALERT @ {current_time:.1f}s: {sound_type} "
                              f"(YAMNet: {class_name}, conf={confidence:.2f}) ***")
                        break  # One alert per window

            except Exception as e:
                print(f"[{self.video_id}] YAMNet inference error at {current_time:.1f}s: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"[{self.video_id}] YAMNet complete: analyzed {loud_window_count} loud windows, {alert_count} alerts")

    def _get_yamnet_class_names(self) -> dict:
        """Get YAMNet class names from the model or use a cached version."""
        if hasattr(self, '_yamnet_class_names'):
            return self._yamnet_class_names

        # Common YAMNet class names (subset of the 521 classes)
        # Full list: https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv
        self._yamnet_class_names = {
            0: "Speech",
            1: "Child speech",
            2: "Conversation",
            3: "Narration",
            4: "Babbling",
            5: "Speech synthesizer",
            6: "Shout",
            7: "Bellow",
            8: "Whoop",
            9: "Yell",
            10: "Children shouting",
            11: "Screaming",
            12: "Whispering",
            13: "Laughter",
            137: "Music",
            288: "Vehicle",
            420: "Thump",
            421: "Thunk",
            426: "Explosion",
            427: "Gunshot, gunfire",
            428: "Machine gun",
            429: "Fusillade",
            430: "Artillery fire",
            431: "Cap gun",
            494: "Bang",
            495: "Slap",
            496: "Whack",
            500: "Finger snapping",
            506: "Silence",
            520: "Inside, small room",
        }
        return self._yamnet_class_names

    def _detect_gunshots_heuristic(self, audio: np.ndarray, sample_rate: int) -> None:
        """Fallback heuristic-based gunshot detection."""
        chunk_samples = int(sample_rate * GUNSHOT_CHUNK_MS / 1000)
        attack_samples = int(sample_rate * ATTACK_TIME_THRESHOLD_MS / 1000)
        duration_samples = int(sample_rate * SOUND_DURATION_MAX_MS / 1000)
        last_alert_time = -GUNSHOT_COOLDOWN
        alert_count = 0

        print(f"[{self.video_id}] Analyzing audio with heuristic detection...")

        for i in range(0, len(audio) - chunk_samples, chunk_samples // 2):
            if self._stop_event.is_set():
                break

            chunk = audio[i:i + chunk_samples]
            current_time = i / sample_rate

            abs_chunk = np.abs(chunk)
            peak_amplitude = np.max(abs_chunk)
            if peak_amplitude < 0.45:
                continue

            if current_time - last_alert_time < GUNSHOT_COOLDOWN:
                continue

            # Spectral analysis
            spectrum = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1 / sample_rate)
            spectrum_sum = np.sum(spectrum) + 1e-10
            spectral_centroid = np.sum(freqs * spectrum) / spectrum_sum

            # Crest factor
            rms = np.sqrt(np.mean(chunk ** 2)) + 1e-10
            crest_factor = peak_amplitude / rms

            # Attack time
            peak_idx = np.argmax(abs_chunk)
            threshold_20pct = peak_amplitude * 0.2
            attack_start = peak_idx
            for j in range(peak_idx, max(0, peak_idx - attack_samples * 2), -1):
                if abs_chunk[j] < threshold_20pct:
                    attack_start = j
                    break
            attack_time_ms = ((peak_idx - attack_start) / sample_rate) * 1000

            # Duration
            threshold_30pct = peak_amplitude * 0.3
            duration_end = min(len(chunk), peak_idx + duration_samples)
            for j in range(peak_idx, duration_end):
                if abs_chunk[j] < threshold_30pct:
                    duration_end = j
                    break
            sound_duration_ms = ((duration_end - attack_start) / sample_rate) * 1000

            sound_type = None

            # TASER detection
            if spectral_centroid > TASER_FREQ_THRESHOLD and peak_amplitude > 0.50:
                sound_type = "TASER FIRED"
                description = f"Taser discharge (peak={peak_amplitude:.2f}, freq={spectral_centroid:.0f}Hz)"

            # GUNSHOT detection
            elif peak_amplitude >= GUNSHOT_AMPLITUDE_THRESHOLD:
                is_freq_ok = GUNSHOT_FREQ_MIN <= spectral_centroid <= GUNSHOT_FREQ_MAX
                is_impulsive = attack_time_ms <= ATTACK_TIME_THRESHOLD_MS
                is_short = sound_duration_ms <= SOUND_DURATION_MAX_MS
                is_crest_ok = crest_factor >= CREST_FACTOR_THRESHOLD

                impulsive_checks_passed = sum([is_impulsive, is_short, is_crest_ok])
                if is_freq_ok and impulsive_checks_passed >= 2:
                    sound_type = "SHOTS FIRED"
                    description = f"Gunshot (peak={peak_amplitude:.2f}, freq={spectral_centroid:.0f}Hz)"

            if sound_type:
                alert_count += 1
                last_alert_time = current_time
                alert_text = f"⚠️ AUDIO: {sound_type}! {description}"
                self._emit(current_time, "action", alert_text)
                print(f"[{self.video_id}] AUDIO @ {current_time:.1f}s: {sound_type}")

        print(f"[{self.video_id}] Heuristic gunshot detection complete: {alert_count} alerts")

    def _transcribe_with_openai(self, wav_path: str) -> None:
        """Transcribe audio using OpenAI Whisper API."""
        print(f"[{self.video_id}] Starting OpenAI Whisper transcription...")

        try:
            # Read the audio file
            with open(wav_path, "rb") as audio_file:
                # Use OpenAI Whisper API with timestamps
                response = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            # Process segments with timestamps
            if hasattr(response, 'segments') and response.segments:
                for segment in response.segments:
                    # Segments are objects with attributes, not dicts
                    start_time = getattr(segment, 'start', 0)
                    text = getattr(segment, 'text', '').strip()

                    if text and len(text) > 2:
                        self._emit(start_time, "transcript", f"[OpenAI] {text}")
                        print(f"[{self.video_id}] TRANSCRIPT (OpenAI) @ {start_time:.1f}s: {text[:60]}...")
            elif hasattr(response, 'text') and response.text:
                # Fallback if no segments available
                self._emit(0, "transcript", f"[OpenAI] {response.text}")
                print(f"[{self.video_id}] TRANSCRIPT (OpenAI): {response.text[:60]}...")

            print(f"[{self.video_id}] OpenAI Whisper transcription complete")

        except Exception as e:
            print(f"[{self.video_id}] OpenAI transcription error: {e}")
            import traceback
            traceback.print_exc()

    def _transcribe_with_deepgram(self, wav_path: str) -> None:
        """Transcribe audio using Deepgram REST API directly."""
        if not DEEPGRAM_API_KEY:
            print(f"[{self.video_id}] Deepgram API key not configured, skipping Deepgram transcription")
            return

        print(f"[{self.video_id}] Starting Deepgram transcription...")

        try:
            import requests

            # Read the audio file
            with open(wav_path, "rb") as audio_file:
                audio_data = audio_file.read()

            # Deepgram REST API endpoint with query parameters
            url = "https://api.deepgram.com/v1/listen"
            params = {
                "model": "nova-2",
                "smart_format": "true",
                "punctuate": "true",
                "paragraphs": "true",  # Get sentence-level timestamps
                "utterances": "true",
                "utt_split": "0.2",  # Split on 0.2s silence (high sensitivity)
            }

            headers = {
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            }

            # Make the request - send raw audio data
            print(f"[{self.video_id}] Sending {len(audio_data)} bytes to Deepgram...")
            response = requests.post(url, params=params, headers=headers, data=audio_data, timeout=120)

            if response.status_code != 200:
                print(f"[{self.video_id}] Deepgram API error: {response.status_code} - {response.text}")
                return

            data = response.json()
            print(f"[{self.video_id}] Deepgram response received")

            # Process results
            results = data.get("results", {})
            channels = results.get("channels", [])

            # First try to get sentences from paragraphs (most granular)
            sentences_emitted = False
            if channels:
                for channel in channels:
                    alternatives = channel.get("alternatives", [])
                    if alternatives:
                        alt = alternatives[0]
                        paragraphs_data = alt.get("paragraphs", {})
                        paragraphs_list = paragraphs_data.get("paragraphs", [])

                        if paragraphs_list:
                            print(f"[{self.video_id}] Deepgram: Found {len(paragraphs_list)} paragraphs with sentences")
                            for para in paragraphs_list:
                                sentences = para.get("sentences", [])
                                for sentence in sentences:
                                    start_time = sentence.get("start", 0)
                                    text = sentence.get("text", "").strip()

                                    if text and len(text) > 2:
                                        self._emit(start_time, "transcript", f"[Deepgram] {text}")
                                        print(f"[{self.video_id}] TRANSCRIPT (Deepgram) @ {start_time:.1f}s: {text[:60]}...")
                                        sentences_emitted = True

            # Fall back to utterances if no sentences
            if not sentences_emitted:
                utterances = results.get("utterances", [])
                if utterances:
                    print(f"[{self.video_id}] Deepgram: Found {len(utterances)} utterances")
                    for utterance in utterances:
                        start_time = utterance.get("start", 0)
                        text = utterance.get("transcript", "").strip()

                        if text and len(text) > 2:
                            self._emit(start_time, "transcript", f"[Deepgram] {text}")
                            print(f"[{self.video_id}] TRANSCRIPT (Deepgram) @ {start_time:.1f}s: {text[:60]}...")
                elif channels:
                    # Final fallback to full transcript
                    print(f"[{self.video_id}] Deepgram: Using full transcript fallback")
                    for channel in channels:
                        alternatives = channel.get("alternatives", [])
                        if alternatives:
                            transcript = alternatives[0].get("transcript", "")
                            if transcript:
                                self._emit(0, "transcript", f"[Deepgram] {transcript}")
                                print(f"[{self.video_id}] TRANSCRIPT (Deepgram) full: {transcript[:60]}...")

            print(f"[{self.video_id}] Deepgram transcription complete")

        except Exception as e:
            print(f"[{self.video_id}] Deepgram transcription error: {e}")
            import traceback
            traceback.print_exc()

    def _run_transcription(self, wav_path: str) -> None:
        """Run OpenAI transcription only (Deepgram runs separately before video analysis)."""
        if TRANSCRIPTION_PROVIDER == "none":
            print(f"[{self.video_id}] Transcription disabled")
            return

        print(f"[{self.video_id}] Running OpenAI transcription...")

        if TRANSCRIPTION_PROVIDER in ("openai", "both"):
            self._transcribe_with_openai(wav_path)

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
            print(f"[{self.video_id}] Scene descriptions: {'enabled' if ENABLE_SCENE_DESCRIPTIONS else 'disabled'}")
            self._emit(0, "status", f"Processing video ({duration:.1f}s)")
            update_video_status(conn, self.video_id, "processing")

            # Extract audio first
            print(f"[{self.video_id}] Extracting audio...")
            self._emit(0, "status", "Extracting audio...")
            wav_path = self._extract_audio()

            # Run Deepgram transcription FIRST (before video analysis)
            # This pre-loads transcripts so they're ready when video plays
            if wav_path and TRANSCRIPTION_PROVIDER in ("deepgram", "both"):
                print(f"[{self.video_id}] Pre-loading Deepgram transcription...")
                self._emit(0, "status", "Loading Deepgram transcription...")
                self._transcribe_with_deepgram(wav_path)
                print(f"[{self.video_id}] Deepgram transcription pre-loaded")

            # Start gunshot detection in parallel
            gunshot_thread = None
            if wav_path:
                print(f"[{self.video_id}] Starting audio gunshot detection...")
                gunshot_thread = threading.Thread(
                    target=self._detect_gunshots,
                    args=(wav_path,),
                    daemon=True,
                )
                gunshot_thread.start()
                print(f"[{self.video_id}] Gunshot detection thread started")

            # Start OpenAI transcription in parallel (runs alongside video analysis)
            transcription_thread = None
            if wav_path and TRANSCRIPTION_PROVIDER in ("openai", "both"):
                print(f"[{self.video_id}] Starting OpenAI transcription...")
                transcription_thread = threading.Thread(
                    target=self._run_transcription,
                    args=(wav_path,),
                    daemon=True,
                )
                transcription_thread.start()
                print(f"[{self.video_id}] OpenAI transcription thread started")

            # Process frames at regular intervals (parallel with audio streaming)
            frame_interval_frames = int(fps * FRAME_INTERVAL)
            frame_count = 0
            last_context = ""
            first_frame_analyzed = False
            last_activity_time = 0.0  # Track when we last had real activity
            last_emitted_time = 0.0  # Track when we last emitted ANY activity event (for 10s min interval)
            no_activity_start = None  # Track start of "no activity" period
            last_scene_time = -999.0  # Track when we last did a scene description
            last_screenshot_time = -999.0  # Track when we last took a periodic screenshot

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
                if ENABLE_SCENE_DESCRIPTIONS:
                    should_describe_scene = is_first_frame or (current_time - last_scene_time >= SCENE_DESCRIPTION_INTERVAL)

                    if should_describe_scene:
                        scene_desc = self._analyze_scene(image_url)
                        if scene_desc:
                            self._emit(current_time, "scene", scene_desc)
                            print(f"[{self.video_id}] SCENE @ {current_time:.1f}s: {scene_desc[:80]}...")
                            last_scene_time = current_time
                            # Capture screenshot for scene description
                            self._capture_screenshot(frame, current_time, "scene")
                            last_screenshot_time = current_time

                # Get recent transcript context (last 5 seconds for tighter relevance)
                audio_context = self._get_recent_transcript_context(current_time, lookback_sec=5.0)

                # Determine if we need a FULL description vs change detection
                # Full description: first frame OR every 30 seconds
                need_full_description = is_first_frame or (int(current_time) % 30 == 0 and int(current_time) > 0)

                description = self._analyze_frame_with_audio(
                    image_url,
                    visual_context=last_context,
                    audio_context=audio_context,
                    full_description=need_full_description
                )

                if audio_context:
                    print(f"[{self.video_id}] @ {current_time:.1f}s: Audio: \"{audio_context[:40]}...\"")

                # Handle NO_CHANGE response - skip emitting
                if description and description.strip().upper() == "NO_CHANGE":
                    print(f"[{self.video_id}] @ {current_time:.1f}s: (no change)")
                    continue

                # Track critical presence so we don't repeat the same warning each frame
                self._refresh_active_critical(current_time)
                if description:
                    categories = self._extract_critical_categories(description)
                    if categories:
                        newly_active = {c for c in categories if c not in self._active_critical}
                        for category in categories:
                            self._active_critical[category] = current_time
                        if newly_active:
                            already_active = categories - newly_active
                            if already_active:
                                description = self._strip_active_critical_prefixes(description, already_active)
                        else:
                            description = self._strip_active_critical_prefixes(description, categories)
                        if not description:
                            print(f"[{self.video_id}] @ {current_time:.1f}s: (critical already active, skipping)")
                            continue

                # Check if description is too similar to previous (AI didn't say NO_CHANGE but repeated itself)
                # BUT always emit critical events even if similar
                if description and last_context and not self._is_critical_event(description):
                    if self._is_similar_to_previous(description, last_context):
                        print(f"[{self.video_id}] @ {current_time:.1f}s: (similar to previous, skipping)")
                        continue

                if description:
                    # Check if it's a "no activity" response
                    is_no_activity = self._is_no_activity_response(description)

                    # Check if we need to force an update (10 second minimum interval)
                    time_since_last_emit = current_time - last_emitted_time
                    force_emit = time_since_last_emit >= MIN_ACTIVITY_INTERVAL

                    if is_no_activity and not force_emit:
                        # Start or continue tracking no-activity period
                        if no_activity_start is None:
                            no_activity_start = last_activity_time if last_activity_time > 0 else current_time - FRAME_INTERVAL
                        print(f"[{self.video_id}] @ {current_time:.1f}s: (no activity)")
                    elif is_no_activity and force_emit:
                        # Force emit after 10 seconds - emit the no-activity range so far, then continue
                        if no_activity_start is not None:
                            self._emit_no_activity_range(no_activity_start, current_time)
                            no_activity_start = current_time  # Start a new no-activity period
                        else:
                            # No period tracked yet, emit current state
                            self._emit(current_time, "action", "Scene continues unchanged.")
                        last_emitted_time = current_time
                        print(f"[{self.video_id}] @ {current_time:.1f}s: (forced update after {time_since_last_emit:.0f}s)")
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
                        last_emitted_time = current_time

                        # Capture screenshot for critical events
                        if self._is_critical_event(description):
                            self._capture_screenshot(frame, current_time, "critical")
                            last_screenshot_time = current_time
                        # Capture periodic screenshot every SCREENSHOT_PERIODIC_INTERVAL seconds
                        elif current_time - last_screenshot_time >= SCREENSHOT_PERIODIC_INTERVAL:
                            self._capture_screenshot(frame, current_time, "periodic")
                            last_screenshot_time = current_time
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

            # Wait for gunshot detection to finish
            if gunshot_thread and gunshot_thread.is_alive():
                print(f"[{self.video_id}] Waiting for gunshot detection to complete...")
                gunshot_thread.join(timeout=30.0)

            # Wait for transcription to finish
            if transcription_thread and transcription_thread.is_alive():
                print(f"[{self.video_id}] Waiting for transcription to complete...")
                transcription_thread.join(timeout=60.0)  # Transcription may take longer

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
        """Check if response is ONLY a 'no activity' stub with no real description."""
        text_lower = text.lower().strip()

        # Too short to be meaningful (less than 10 chars)
        if len(text_lower) < 10:
            return True

        # Check for no-change indicators
        no_change_phrases = [
            "no_change", "no change", "no new activity", "no activity",
            "nothing new", "scene unchanged", "no movement",
            "no significant change", "nothing has changed"
        ]

        for phrase in no_change_phrases:
            if text_lower == phrase or text_lower == phrase + ".":
                return True

        return False

    def _is_similar_to_previous(self, new_text: str, prev_text: str, threshold: float = 0.7) -> bool:
        """Check if new description is too similar to previous (avoid repetition)."""
        if not prev_text or not new_text:
            return False

        # Normalize texts
        new_words = set(new_text.lower().split())
        prev_words = set(prev_text.lower().split())

        # Remove common words that don't indicate real similarity
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "and", "of", "with"}
        new_words -= stop_words
        prev_words -= stop_words

        if not new_words or not prev_words:
            return False

        # Jaccard similarity
        intersection = len(new_words & prev_words)
        union = len(new_words | prev_words)

        similarity = intersection / union if union > 0 else 0

        return similarity > threshold

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


class AudioStreamProcessor(VideoStreamProcessor):
    """Audio-only processor that emits gunshot/taser alerts via SSE."""

    def start(self) -> None:
        self._processing_thread = threading.Thread(
            target=self._process_audio_only,
            daemon=True,
        )
        self._processing_thread.start()

    def _emit(self, offset: float, kind: str, text: str) -> None:
        event = VideoEvent(
            offset_sec=offset,
            kind=kind,
            text=text,
            timestamp=datetime.now().isoformat(),
        )
        self._event_queue.put(event)

    def _process_audio_only(self) -> None:
        wav_path = None
        try:
            self._emit(0, "status", "Extracting audio for gunshot detection...")
            wav_path = self._extract_audio()
            if not wav_path:
                self._emit(0, "status", "Error: Audio extraction failed")
                return
            self._emit(0, "status", "Running gunshot detection...")
            self._detect_gunshots(wav_path)
            self._emit(0, "status", "Gunshot detection complete")
        except Exception as exc:
            self._emit(0, "status", f"Error: {exc}")
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
            self._event_queue.put(None)
            self._stop_event.set()


_active_audio_processors: dict[str, AudioStreamProcessor] = {}


def start_audio_stream_processing(video_id: str, video_path: str) -> AudioStreamProcessor:
    """Start audio-only gunshot detection and return the processor for SSE streaming."""
    processor = AudioStreamProcessor(video_id, video_path)
    _active_audio_processors[video_id] = processor
    processor.start()
    return processor


def get_audio_stream_processor(video_id: str) -> AudioStreamProcessor | None:
    """Get an active audio-only processor."""
    return _active_audio_processors.get(video_id)


def stop_audio_stream_processing(video_id: str) -> None:
    """Stop audio-only processing."""
    processor = _active_audio_processors.pop(video_id, None)
    if processor:
        processor.stop()
