import asyncio
import os
import re
from datetime import datetime

from dotenv import load_dotenv

from event_log import get_event_logger, open_event_db
from video_store import open_video_db, insert_event as insert_video_event
from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io, llm
from livekit.agents.voice import io as voice_io
from livekit.plugins import (
    openai,
    silero,
)
from openai.types.beta.realtime.session import TurnDetection


class VideoEventLogger:
    """Logger that writes events to the video database with offset timing."""

    def __init__(self, video_id: str, start_time: float):
        self._video_id = video_id
        self._start_time = start_time
        self._conn = open_video_db()

    def log_event(
        self,
        *,
        kind: str,
        text: str,
        ts: datetime | float | str | None = None,
        source: str | None = None,
    ) -> None:
        if not text:
            return
        # Calculate offset from session start
        if isinstance(ts, datetime):
            offset = ts.timestamp() - self._start_time
        elif isinstance(ts, (int, float)):
            offset = ts - self._start_time
        else:
            offset = datetime.now().timestamp() - self._start_time

        offset = max(0.0, offset)  # Ensure non-negative

        insert_video_event(
            self._conn,
            video_id=self._video_id,
            offset_sec=offset,
            kind=kind,
            text=text,
        )

    def close(self) -> None:
        self._conn.close()

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env.local")
load_dotenv(ENV_PATH)
AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "assistant")


class VideoAssistant(Agent):
    def __init__(self, video_processing_mode: bool = False) -> None:
        realtime_model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
        realtime_voice = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
        realtime_modalities = ["text"]

        # For video processing: disable automatic responses from turn detection
        # Audio is still received for context, but responses only happen on explicit generate_reply() calls
        if video_processing_mode:
            turn_detection_config = TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=500,
                create_response=False,  # Don't auto-respond to audio turns
                interrupt_response=False,  # Don't interrupt on audio
            )
        else:
            turn_detection_config = None  # Use default turn detection

        super().__init__(
            instructions=(
                "You are a body-cam footage analyst. "
                "Output only detailed observations and analysis of visible actions, environment, and notable changes. "
                "Use English only. "
                "Use third-person language only; refer to 'the subject' or 'the scene'. "
                "Never use first-person ('I', 'we') or second-person ('you', 'your'). "
                "Never address the user, ask questions, greet, or offer help. "
                "Do not mention training, prompts, or the system."
            ),
            llm=openai.realtime.RealtimeModel(
                model=realtime_model,
                voice=realtime_voice,
                modalities=realtime_modalities,
                turn_detection=turn_detection_config,
            ),
        )
        self._realtime_model = realtime_model
        self._realtime_voice = realtime_voice
        self._realtime_modalities = realtime_modalities
        self._video_processing_mode = video_processing_mode


class ConsoleTextOutput(voice_io.TextOutput):
    def __init__(
        self,
        *,
        label: str,
        next_in_chain: voice_io.TextOutput | None = None,
        logger: "EventLogger | None" = None,
        start_time: float | None = None,
    ) -> None:
        super().__init__(label=label, next_in_chain=next_in_chain)
        self._buffer = ""
        self._first_seen_at: datetime | None = None
        self._logger = logger
        self._start_time = start_time  # For calculating video offset
        self._forbidden_re = re.compile(
            r"\b(i|i'm|im|ive|i've|id|i'd|me|my|we|we're|were|weve|we've|our|ours|us|you|your|you're|youre|you've|youve)\b",
            re.IGNORECASE,
        )
        self._red = "\033[31m"
        self._reset = "\033[0m"

    def _normalize_text(self, text: str) -> str:
        replacements = {
            "\u2019": "'",
            "\u2018": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2014": "-",
            "\u2013": "-",
            "\u2026": "...",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text

    def _get_offset(self) -> str:
        """Return video offset in seconds, or wall-clock time if no start_time."""
        if self._start_time is not None:
            stamp = self._first_seen_at or datetime.now()
            offset_sec = stamp.timestamp() - self._start_time
            return f"{max(0, offset_sec):.1f}s"
        else:
            stamp = self._first_seen_at or datetime.now()
            return stamp.strftime("%H:%M:%S")

    def _is_allowed(self, text: str) -> bool:
        return text.isascii() and not self._forbidden_re.search(text) and "?" not in text

    async def capture_text(self, text: str) -> None:
        if not text:
            return
        if not self._buffer:
            self._first_seen_at = datetime.now()
        self._buffer += text
        if self.next_in_chain:
            await self.next_in_chain.capture_text(text)

    def flush(self) -> None:
        if self._buffer:
            cleaned = self._normalize_text(self._buffer).strip()
            if self._is_allowed(cleaned):
                text = cleaned
            else:
                text = "No significant change."
            print(f"{self._red}Action [{self._get_offset()}]: {text}{self._reset}")
            if os.getenv("REALTIME_DEBUG", "0") == "1":
                print(f"Realtime response: {text}")
            if self._logger:
                self._logger.log_event(
                    kind="action",
                    text=text,
                    ts=self._first_seen_at,
                    source="agent",
                )
            self._buffer = ""
            self._first_seen_at = None
        if self.next_in_chain:
            self.next_in_chain.flush()


server = AgentServer(port=0)


@server.rtc_session(agent_name=AGENT_NAME)
async def my_agent(ctx: agents.JobContext):
    print(f"Agent session started in room: {ctx.room.name} (agent={AGENT_NAME})")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Set it in .env.local.")

    # Detect if this is a video processing room (room name: "video-{video_id}")
    room_name = ctx.room.name
    video_id = None
    session_start_time = datetime.now().timestamp()

    # For video processing rooms, enable audio input so the model can hear the video's audio
    is_video_room = room_name.startswith("video-")

    # Session uses VAD for all cases; turn detection is configured at the model level
    session = AgentSession(vad=silero.VAD.load())

    if is_video_room:
        video_id = room_name[6:]  # Extract video_id from "video-{video_id}"
        event_logger = VideoEventLogger(video_id, session_start_time)
        print(f"Video processing mode: video_id={video_id} (audio+video input, no auto-response)")
    else:
        event_logger = get_event_logger()

    # For video rooms: use video_processing_mode=True to disable automatic responses
    # Audio is received for context but model only responds on explicit generate_reply() calls
    agent = VideoAssistant(video_processing_mode=is_video_room)
    if not isinstance(agent.llm, openai.realtime.RealtimeModel):
        raise RuntimeError("Expected OpenAI RealtimeModel; check plugin installation and config.")
    print(
        "OpenAI Realtime enabled "
        f"(model={agent._realtime_model}, modalities={agent._realtime_modalities})."
    )

    await session.start(
        room=ctx.room,
        agent=agent,
        room_options=room_io.RoomOptions(
            video_input=True,
            text_output=True,
            audio_input=is_video_room,  # Enable audio for video processing
            text_input=False,
            audio_output=False,
        ),
    )
    session.output.transcription = ConsoleTextOutput(
        label="console",
        next_in_chain=session.output.transcription,
        logger=event_logger,
        start_time=session_start_time if is_video_room else None,
    )
    loop = asyncio.get_running_loop()
    cooldown_until = 0.0
    consecutive_errors = 0

    # Colors for console output
    green = "\033[32m"
    reset_color = "\033[0m"

    # Log user transcripts (audio from video) for video rooms
    @session.on("user_input_transcribed")
    def _on_transcript(transcript):
        if not is_video_room:
            return
        if not transcript.transcript or not transcript.transcript.strip():
            return
        # Calculate video offset
        offset_sec = datetime.now().timestamp() - session_start_time
        offset_str = f"{max(0, offset_sec):.1f}s"
        text = transcript.transcript.strip()
        final_marker = " [FINAL]" if transcript.is_final else ""
        print(f"{green}Transcript [{offset_str}]{final_marker}: {text}{reset_color}")
        # Log final transcripts to the database
        if transcript.is_final and event_logger:
            event_logger.log_event(
                kind="transcript",
                text=text,
                ts=datetime.now(),
                source="audio",
            )

    async def reset_chat_ctx() -> None:
        if session.llm and hasattr(session.llm, "update_chat_ctx"):
            try:
                await session.llm.update_chat_ctx(llm.ChatContext.empty())
            except Exception as exc:
                print(f"Chat context reset failed: {exc}")

    @session.on("error")
    def _on_error(ev):
        nonlocal cooldown_until, consecutive_errors
        print(f"Agent error: {ev}")
        err_str = str(ev).lower()
        if "tokens" in err_str or "rate" in err_str or "429" in err_str:
            # Longer cooldown (30s) to let the rate limit window reset
            cooldown_until = max(cooldown_until, loop.time() + 30.0)
            consecutive_errors += 1
            asyncio.create_task(reset_chat_ctx())

    stop_event = asyncio.Event()

    @session.on("close")
    def _on_close(_):
        stop_event.set()

    async def tail_mic_transcripts() -> None:
        if os.getenv("SHOW_MIC_TRANSCRIPT", "1") != "1":
            return
        conn = open_event_db()
        green = "\033[32m"
        reset = "\033[0m"
        last_id = 0
        try:
            while not stop_event.is_set():
                rows = conn.execute(
                    "SELECT id, ts, text FROM events WHERE kind = ? AND id > ? ORDER BY id",
                    ("transcript", last_id),
                ).fetchall()
                for row in rows:
                    last_id = row[0]
                    ts = row[1]
                    text = row[2]
                    if not text:
                        continue
                    time_part = ts[-8:] if isinstance(ts, str) and len(ts) >= 8 else ts
                    print(f"{green}Transcript [{time_part}]: {text}{reset}")
                await asyncio.sleep(0.5)
        finally:
            conn.close()

    # Longer intervals to avoid rate limits (realtime API has its own limits)
    short_interval = float(os.getenv("SCENE_SHORT_INTERVAL", "8"))
    long_interval = float(os.getenv("SCENE_LONG_INTERVAL", "20"))
    llm_lock = asyncio.Lock()

    async def run_reply(instructions: str, max_tokens: int) -> None:
        nonlocal consecutive_errors
        async with llm_lock:
            # Apply adaptive backoff based on consecutive errors
            if consecutive_errors > 0:
                backoff = min(2.0 ** consecutive_errors, 60.0)
                print(f"Adaptive backoff: waiting {backoff:.1f}s due to {consecutive_errors} recent errors")
                await asyncio.sleep(backoff)

            if loop.time() < cooldown_until:
                wait_time = cooldown_until - loop.time()
                print(f"Cooldown active: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

            if session.llm and hasattr(session.llm, "update_options"):
                try:
                    session.llm.update_options(max_response_output_tokens=max_tokens)
                except Exception:
                    pass
            if os.getenv("REALTIME_DEBUG", "0") == "1":
                print(f"Realtime request: {instructions}")
            try:
                # Interruption handling is configured at the model level (TurnDetection)
                await session.generate_reply(instructions=instructions)
                consecutive_errors = 0  # Reset on success
            except Exception as exc:
                consecutive_errors += 1
                if "rate" in str(exc).lower() or "429" in str(exc):
                    print(f"Rate limit in run_reply: {exc}")
                raise
            await reset_chat_ctx()

    async def short_loop():
        while not stop_event.is_set():
            try:
                await run_reply(
                    instructions=(
                        "One short English sentence (6-12 words) describing the main visible action or change. "
                        "Third-person only; begin with 'The subject' or 'The scene'. "
                        "Never use first-person or second-person pronouns. "
                        "No questions, no advice, no compliments, no greetings."
                    ),
                    max_tokens=60,
                )
            except Exception as exc:
                print(f"Short loop stopped: {exc}")
                break
            await asyncio.sleep(short_interval)

    async def long_loop():
        while not stop_event.is_set():
            try:
                await run_reply(
                    instructions=(
                        "Two or three English sentences (40-70 words total) with detailed scene description, environment, and recent actions. "
                        "Third-person only; begin each sentence with 'The subject' or 'The scene'. "
                        "Never use first-person or second-person pronouns. "
                        "No questions, no advice, no compliments, no greetings. "
                        "If no clear change, say: No significant change."
                    ),
                    max_tokens=220,
                )
            except Exception as exc:
                print(f"Long loop stopped: {exc}")
                break
            await asyncio.sleep(long_interval)

    tasks = []
    tasks.append(asyncio.create_task(tail_mic_transcripts()))
    if short_interval > 0:
        tasks.append(asyncio.create_task(short_loop()))
    if long_interval > 0:
        tasks.append(asyncio.create_task(long_loop()))
    await stop_event.wait()
    for task in tasks:
        task.cancel()
    if event_logger:
        event_logger.close()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Set it in .env.local.")
    _startup_agent = VideoAssistant()
    if not isinstance(_startup_agent.llm, openai.realtime.RealtimeModel):
        raise RuntimeError("Expected OpenAI RealtimeModel; check plugin installation and config.")
    print(
        "OpenAI Realtime configured "
        f"(model={_startup_agent._realtime_model}, modalities={_startup_agent._realtime_modalities})."
    )
    agents.cli.run_app(server)
