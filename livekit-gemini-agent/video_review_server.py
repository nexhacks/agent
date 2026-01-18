import argparse
import asyncio
import base64
import cgi
import json
import mimetypes
import os
import re
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from dotenv import load_dotenv

from livekit.agents.llm import ChatContext, ImageContent
from livekit.plugins import openai

from video_store import insert_event, insert_video, list_events, list_videos, open_video_db, update_video_status


UPLOAD_DIR = os.getenv("VIDEO_UPLOAD_DIR", "uploads")
SHORT_INTERVAL = float(os.getenv("VIDEO_SHORT_INTERVAL", "10"))
LONG_INTERVAL = float(os.getenv("VIDEO_LONG_INTERVAL", "25"))
MODEL_NAME = os.getenv("VIDEO_ACTION_MODEL", "gpt-4o-mini")
PROCESS_REALTIME = os.getenv("VIDEO_PROCESS_REALTIME", "1") == "1"
TRANSCRIBE_MODEL = os.getenv("VIDEO_TRANSCRIBE_MODEL", "tiny")
PROVIDER = os.getenv("VIDEO_LLM_PROVIDER", "openai").lower()

# Use the OpenAI Realtime API instead of frame-by-frame processing
USE_REALTIME_API = os.getenv("USE_REALTIME_API", "1") == "1"

# Rate limiting: tokens per minute budget (leave headroom under 200k limit)
TPM_BUDGET = int(os.getenv("VIDEO_TPM_BUDGET", "150000"))
# Estimated tokens per image (base64 JPEG ~1024x1024)
TOKENS_PER_IMAGE = int(os.getenv("VIDEO_TOKENS_PER_IMAGE", "1500"))


class RateLimiter:
    """Simple token bucket rate limiter for OpenAI TPM limits."""

    def __init__(self, tokens_per_minute: int):
        self._tpm = tokens_per_minute
        self._tokens_used = 0
        self._window_start = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            # Reset window every minute
            if now - self._window_start >= 60.0:
                self._tokens_used = 0
                self._window_start = now

            # If we'd exceed budget, wait until window resets
            if self._tokens_used + tokens > self._tpm:
                wait_time = 60.0 - (now - self._window_start) + 1.0
                print(f"Rate limit: waiting {wait_time:.1f}s before next API call")
                await asyncio.sleep(wait_time)
                self._tokens_used = 0
                self._window_start = asyncio.get_running_loop().time()

            self._tokens_used += tokens


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def _get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(TPM_BUDGET)
    return _rate_limiter


SYSTEM_PROMPT = (
    "You are a body-cam footage analyst. Output only detailed observations and analysis of visible actions, "
    "environment, and notable changes. Use English only. Use third-person language only; refer to 'the subject' "
    "or 'the scene'. Never use first-person or second-person pronouns. Never address the user, ask questions, "
    "greet, or offer help. Do not mention training, prompts, or the system."
)


def _safe_text(text: str) -> str:
    text = text.strip()
    if not text:
        return "No significant change."
    if not text.isascii():
        return "No significant change."
    if re.search(r"\b(i|we|you|you're|youre|i'm|im|i've|ive|our|your)\b", text, re.IGNORECASE):
        return "No significant change."
    if "?" in text:
        return "No significant change."
    return text


def _encode_frame_bgr(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    data = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


async def _describe_frame(image_data_url: str, prompt: str, max_tokens: int) -> str:
    # Proactive rate limiting - acquire tokens before making the call
    # Estimate: image tokens + prompt tokens + max output tokens
    estimated_tokens = TOKENS_PER_IMAGE + 200 + max_tokens
    await _get_rate_limiter().acquire(estimated_tokens)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content=[SYSTEM_PROMPT])
    chat_ctx.add_message(role="user", content=[ImageContent(image=image_data_url), prompt])
    if PROVIDER == "gemini":
        raise RuntimeError(
            "Gemini provider not installed. Install the LiveKit Google plugin to use it."
        )

    def _rate_limit_delay(message: str, attempt: int) -> float:
        match = re.search(r"try again in (\d+)\s*ms", message, re.IGNORECASE)
        if match:
            return max(float(match.group(1)) / 1000.0, 0.5)
        # Exponential backoff with longer waits
        return min(2.0 ** (attempt + 1), 30.0)

    def _is_rate_limit(exc: Exception) -> bool:
        message = str(exc).lower()
        return "rate limit" in message or "429" in message

    last_exc: Exception | None = None
    for attempt in range(5):
        llm = None
        try:
            llm = openai.LLM(model=MODEL_NAME, temperature=0.2)
            stream = llm.chat(
                chat_ctx=chat_ctx,
                extra_kwargs={"max_completion_tokens": max_tokens},
            )
            text = ""
            async for piece in stream.to_str_iterable():
                text += piece
            await llm.aclose()
            return _safe_text(text)
        except Exception as exc:
            last_exc = exc
            if llm is not None:
                try:
                    await llm.aclose()
                except Exception:
                    pass
            if _is_rate_limit(exc) and attempt < 4:
                delay = _rate_limit_delay(str(exc), attempt)
                print(f"Rate limit hit, waiting {delay:.1f}s (attempt {attempt + 1}/5)")
                await asyncio.sleep(delay)
                continue
            raise

    if last_exc:
        raise last_exc
    return "No significant change."


def _extract_audio(video_path: str, wav_path: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "aresample=async=1:first_pts=0",
        wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _transcribe_audio(video_id: str, wav_path: str) -> None:
    from faster_whisper import WhisperModel

    conn = open_video_db()
    model = WhisperModel(TRANSCRIBE_MODEL, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        wav_path,
        language="en",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
    )
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        insert_event(conn, video_id=video_id, offset_sec=seg.start, kind="transcript", text=text)
    conn.close()


def _video_duration(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps <= 0 or total <= 0:
        return 0.0
    return total / fps


def _ffprobe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _process_actions(video_id: str, video_path: str) -> None:
    conn = open_video_db()
    cap = cv2.VideoCapture(video_path)

    async def _run() -> None:
        short_prompt = (
            "One short English sentence (6-12 words) describing the main visible action or change. "
            "Third-person only; begin with 'The subject' or 'The scene'. "
            "Never use first-person or second-person pronouns. "
            "No questions, no advice, no compliments, no greetings."
        )
        long_prompt = (
            "Two or three English sentences (40-70 words total) with detailed scene description, "
            "environment, and recent actions. Third-person only; begin each sentence with "
            "'The subject' or 'The scene'. Never use first-person or second-person pronouns. "
            "No questions, no advice, no compliments, no greetings. "
            "If no clear change, say: No significant change."
        )
        short_next = 0.0 if SHORT_INTERVAL > 0 else None
        long_next = 0.0 if LONG_INTERVAL > 0 else None
        start_wall = asyncio.get_running_loop().time()
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0
        frame_index = 0
        epsilon = 0.001

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if frame_time <= 0:
                frame_time = frame_index / fps
            if PROCESS_REALTIME:
                wait = start_wall + frame_time - asyncio.get_running_loop().time()
                if wait > 0:
                    await asyncio.sleep(wait)

            if short_next is not None and frame_time + epsilon >= short_next:
                data_url = _encode_frame_bgr(frame)
                if data_url:
                    try:
                        text = await _describe_frame(data_url, short_prompt, max_tokens=80)
                        insert_event(
                            conn,
                            video_id=video_id,
                            offset_sec=frame_time,
                            kind="action_short",
                            text=text,
                        )
                    except Exception as exc:
                        print(f"Action short failed at {frame_time:.2f}s: {exc}")
                short_next += SHORT_INTERVAL
                while short_next is not None and short_next <= frame_time:
                    short_next += SHORT_INTERVAL

            if long_next is not None and frame_time + epsilon >= long_next:
                data_url = _encode_frame_bgr(frame)
                if data_url:
                    try:
                        text = await _describe_frame(data_url, long_prompt, max_tokens=200)
                        insert_event(
                            conn,
                            video_id=video_id,
                            offset_sec=frame_time,
                            kind="action_long",
                            text=text,
                        )
                    except Exception as exc:
                        print(f"Action long failed at {frame_time:.2f}s: {exc}")
                long_next += LONG_INTERVAL
                while long_next is not None and long_next <= frame_time:
                    long_next += LONG_INTERVAL

    asyncio.run(_run())
    cap.release()
    conn.close()


def _process_video_realtime(video_id: str, video_path: str) -> None:
    """Process video using the OpenAI Realtime API via LiveKit streaming."""
    from realtime_video_processor import RealtimeVideoProcessor

    async def _run():
        processor = RealtimeVideoProcessor(video_id, video_path)
        await processor.start()
        await processor.wait()

    asyncio.run(_run())


def _process_video_legacy(video_id: str, video_path: str) -> None:
    """Process video using frame-by-frame gpt-4o-mini API calls."""
    conn = open_video_db()
    wav_path = f"{video_path}.wav"
    try:
        _extract_audio(video_path, wav_path)
        threads = [
            threading.Thread(target=_transcribe_audio, args=(video_id, wav_path), daemon=True),
            threading.Thread(target=_process_actions, args=(video_id, video_path), daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        update_video_status(conn, video_id, "done")
    except Exception as exc:
        print(f"Legacy processing error: {exc}")
        update_video_status(conn, video_id, "error")
    finally:
        conn.close()
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


def process_video(video_id: str, video_path: str) -> None:
    """Process video - uses Realtime API by default, falls back to legacy."""
    if USE_REALTIME_API:
        print(f"[{video_id}] Using Realtime API processing")
        try:
            _process_video_realtime(video_id, video_path)
            return
        except Exception as exc:
            print(f"[{video_id}] Realtime API failed, falling back to legacy: {exc}")

    # Legacy frame-by-frame processing
    print(f"[{video_id}] Using legacy frame-by-frame processing")
    conn = open_video_db()
    wav_path = f"{video_path}.wav"
    try:
        _extract_audio(video_path, wav_path)
        threads = [
            threading.Thread(target=_transcribe_audio, args=(video_id, wav_path), daemon=True),
            threading.Thread(target=_process_actions, args=(video_id, video_path), daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        update_video_status(conn, video_id, "done")
    except Exception:
        update_video_status(conn, video_id, "error")
    finally:
        conn.close()
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


class VideoHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: str) -> None:
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                try:
                    with open(path, "rb") as f:
                        f.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk = f.read(min(1024 * 512, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 512)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(os.path.join("templates", "video_review.html"))
            return
        if parsed.path == "/api/videos":
            conn = open_video_db()
            payload = {"videos": list_videos(conn)}
            conn.close()
            self._send_json(payload)
            return
        if parsed.path == "/api/events":
            params = parse_qs(parsed.query)
            video_id = params.get("video_id", [""])[0]
            since_id = int(params.get("since_id", ["0"])[0])
            conn = open_video_db()
            payload = {"events": list_events(conn, video_id, since_id)}
            conn.close()
            self._send_json(payload)
            return
        if parsed.path.startswith("/video/"):
            video_id = parsed.path.split("/video/")[1]
            path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
            if not os.path.exists(path):
                self.send_error(404)
                return
            self._send_file(path)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/upload":
            self.send_error(404)
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")},
        )
        if "file" not in form:
            self._send_json({"error": "missing file"}, status=400)
            return
        file_item = form["file"]
        if not file_item.filename:
            self._send_json({"error": "empty filename"}, status=400)
            return
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        video_id = uuid.uuid4().hex[:12]
        dst_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
        with open(dst_path, "wb") as f:
            f.write(file_item.file.read())
        cap = cv2.VideoCapture(dst_path)
        duration = _video_duration(cap)
        cap.release()
        if duration <= 0:
            duration = _ffprobe_duration(dst_path)
        conn = open_video_db()
        insert_video(conn, video_id, os.path.basename(file_item.filename), duration)
        conn.close()
        thread = threading.Thread(target=process_video, args=(video_id, dst_path), daemon=True)
        thread.start()
        self._send_json({"video_id": video_id})


def main() -> None:
    parser = argparse.ArgumentParser(description="Local video review server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    env_path = os.path.join(os.path.dirname(__file__), ".env.local")
    load_dotenv(env_path)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    open_video_db().close()

    server = ThreadingHTTPServer((args.host, args.port), VideoHandler)
    print(f"Video review server running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
