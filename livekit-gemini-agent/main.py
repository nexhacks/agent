import asyncio
import json
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
    google,
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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Default instructions for video analysis
DEFAULT_INSTRUCTIONS = (
    "You are an objective body cam footage analyst. "
    "Focus on describing: what people are doing and saying; "
    "key interactions, actions, and events; any notable behavior or dialogue. "
    "Be specific about what you observe - actions taken, words spoken, gestures made. "
    "Use English only. Use third-person language ('the officer', 'the individual', 'the person'). "
    "Never use first-person or second-person pronouns. "
    "Do not describe scenery, environment, or background details unless directly relevant. "
    "Do not address the user, ask questions, or offer help."
)


def create_realtime_model(provider: str, video_processing_mode: bool = False):
    """Create a realtime model based on the configured provider.

    Args:
        provider: 'openai' or 'gemini'
        video_processing_mode: If True, disable auto-response for video processing

    Returns:
        Configured RealtimeModel instance
    """
    if provider == "gemini":
        # Must use a native audio model that supports Live API (bidiGenerateContent)
        model = os.getenv("GOOGLE_REALTIME_MODEL", "gemini-live-2.5-flash-preview-native-audio-09-2025")
        voice = os.getenv("GOOGLE_REALTIME_VOICE", "Puck")

        # Build kwargs for RealtimeModel - keep it minimal for compatibility
        kwargs = {
            "model": model,
            "voice": voice,
            "instructions": DEFAULT_INSTRUCTIONS,
        }

        return google.realtime.RealtimeModel(**kwargs), model, voice, ["AUDIO"]

    else:  # OpenAI (default)
        model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
        voice = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
        modalities = ["text"]

        # OpenAI turn detection config for video processing
        if video_processing_mode:
            turn_detection_config = TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=500,
                create_response=False,
                interrupt_response=False,
            )
        else:
            turn_detection_config = None

        return openai.realtime.RealtimeModel(
            model=model,
            voice=voice,
            modalities=modalities,
            turn_detection=turn_detection_config,
        ), model, voice, modalities


class VideoAssistant(Agent):
    def __init__(self, video_processing_mode: bool = False, provider: str | None = None) -> None:
        self._provider = provider or LLM_PROVIDER

        realtime_model, model_name, voice, modalities = create_realtime_model(
            self._provider, video_processing_mode
        )

        super().__init__(
            instructions=DEFAULT_INSTRUCTIONS,
            llm=realtime_model,
        )
        self._realtime_model = model_name
        self._realtime_voice = voice
        self._realtime_modalities = modalities
        self._video_processing_mode = video_processing_mode


class ConsoleTextOutput(voice_io.TextOutput):
    def __init__(
        self,
        *,
        label: str,
        next_in_chain: voice_io.TextOutput | None = None,
        logger: "EventLogger | None" = None,
        start_time: float | None = None,
        room: rtc.Room | None = None,
    ) -> None:
        super().__init__(label=label, next_in_chain=next_in_chain)
        self._buffer = ""
        self._first_seen_at: datetime | None = None
        self._logger = logger
        self._start_time = start_time  # For calculating video offset
        self._room = room  # For publishing to frontend
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
            if not self._is_allowed(cleaned):
                # Skip output for disallowed text (first/second person, questions, etc.)
                self._buffer = ""
                self._first_seen_at = None
                if self.next_in_chain:
                    self.next_in_chain.flush()
                return
            text = cleaned
            offset = self._get_offset()
            print(f"{self._red}Action [{offset}]: {text}{self._reset}")
            if os.getenv("REALTIME_DEBUG", "0") == "1":
                print(f"Realtime response: {text}")
            if self._logger:
                self._logger.log_event(
                    kind="action",
                    text=text,
                    ts=self._first_seen_at,
                    source="agent",
                )
            # Publish to frontend via data stream
            if self._room and self._room.local_participant:
                try:
                    message = json.dumps({
                        "type": "activity",
                        "offset": offset,
                        "text": text,
                        "timestamp": datetime.now().isoformat(),
                    })
                    asyncio.create_task(
                        self._room.local_participant.publish_data(
                            message.encode("utf-8"),
                            topic="lk.activity",
                        )
                    )
                except Exception as e:
                    print(f"Failed to publish activity: {e}")
            self._buffer = ""
            self._first_seen_at = None
        if self.next_in_chain:
            self.next_in_chain.flush()


server = AgentServer(port=0)


@server.rtc_session(agent_name=AGENT_NAME)
async def my_agent(ctx: agents.JobContext):
    print(f"Agent session started in room: {ctx.room.name} (agent={AGENT_NAME}, provider={LLM_PROVIDER})")

    # Validate API key based on provider
    if LLM_PROVIDER == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("Missing GOOGLE_API_KEY. Set it in .env.local.")
    else:
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

    # Validate model type based on provider
    if LLM_PROVIDER == "gemini":
        if not isinstance(agent.llm, google.realtime.RealtimeModel):
            raise RuntimeError("Expected Google RealtimeModel; check plugin installation and config.")
        print(
            f"Gemini Realtime enabled "
            f"(model={agent._realtime_model}, modalities={agent._realtime_modalities})."
        )
    else:
        if not isinstance(agent.llm, openai.realtime.RealtimeModel):
            raise RuntimeError("Expected OpenAI RealtimeModel; check plugin installation and config.")
        print(
            f"OpenAI Realtime enabled "
            f"(model={agent._realtime_model}, modalities={agent._realtime_modalities})."
        )

    await session.start(
        room=ctx.room,
        agent=agent,
        room_options=room_io.RoomOptions(
            video_input=True,
            audio_input=is_video_room,  # Enable audio for video processing
            text_input=is_video_room,   # Enable text input for Q&A after video ends
            text_output=True,
            audio_output=False,
            close_on_disconnect=False,  # Keep session alive after video streamer leaves
        ),
    )
    session.output.transcription = ConsoleTextOutput(
        label="console",
        next_in_chain=session.output.transcription,
        logger=event_logger,
        start_time=session_start_time if is_video_room else None,
        room=ctx.room,  # For publishing to frontend
    )
    loop = asyncio.get_running_loop()
    cooldown_until = 0.0
    consecutive_errors = 0

    # Colors for console output
    green = "\033[32m"
    reset_color = "\033[0m"
    yellow = "\033[33m"

    # Debug: Listen to raw server events if REALTIME_DEBUG is enabled
    if os.getenv("REALTIME_DEBUG", "0") == "1" and session.llm:
        if LLM_PROVIDER == "openai":
            @session.llm.on("openai_server_event_received")
            def _on_raw_event(event):
                event_type = event.get("type", "unknown")
                # Only log error-related events and response.done with non-completed status
                if event_type == "error":
                    print(f"{yellow}[RAW EVENT] error: {event}{reset_color}")
                elif event_type == "response.done":
                    response = event.get("response", {})
                    status = response.get("status")
                    if status != "completed":
                        print(f"{yellow}[RAW EVENT] response.done (status={status}):{reset_color}")
                        print(f"{yellow}  status_details: {response.get('status_details')}{reset_color}")

    # Log user transcripts (audio from video) for video rooms and publish to frontend
    @session.on("user_input_transcribed")
    def _on_transcript(transcript):
        if not is_video_room:
            return
        if not transcript.transcript or not transcript.transcript.strip():
            return
        text = transcript.transcript.strip()
        ts = datetime.now()

        # Log final transcripts to the database
        if transcript.is_final and event_logger:
            event_logger.log_event(
                kind="transcript",
                text=text,
                ts=ts,
                source="audio",
            )

        # Publish all transcripts to frontend (both interim and final)
        try:
            offset_sec = ts.timestamp() - session_start_time
            message = json.dumps({
                "type": "transcript",
                "text": text,
                "is_final": transcript.is_final,
                "offset": f"{max(0, offset_sec):.1f}s",
                "timestamp": ts.isoformat(),
            })
            asyncio.create_task(
                ctx.room.local_participant.publish_data(
                    message.encode("utf-8"),
                    topic="lk.activity",
                )
            )
        except Exception as e:
            print(f"Failed to publish transcript: {e}")

    async def reset_chat_ctx() -> None:
        if session.llm and hasattr(session.llm, "update_chat_ctx"):
            try:
                await session.llm.update_chat_ctx(llm.ChatContext.empty())
            except Exception as exc:
                print(f"Chat context reset failed: {exc}")

    @session.on("error")
    def _on_error(ev):
        nonlocal cooldown_until, consecutive_errors
        # Extract detailed error information - be defensive about attribute access
        print(f"{yellow}=== Agent Error Details ==={reset_color}")
        print(f"{yellow}Raw error: {ev}{reset_color}")
        print(f"{yellow}Error type: {type(ev)}{reset_color}")

        # Print all available attributes
        for attr in ['type', 'timestamp', 'recoverable', 'error', 'message']:
            if hasattr(ev, attr):
                print(f"{yellow}{attr}: {getattr(ev, attr)}{reset_color}")

        # If there's an inner error object, extract its details
        inner_error = getattr(ev, 'error', None)
        if inner_error:
            # If the error has a body attribute (APIError), print it
            if hasattr(inner_error, 'body') and inner_error.body:
                body = inner_error.body
                print(f"{yellow}Error body: {body}{reset_color}")
                # Try to extract specific fields from the body
                for field in ['type', 'code', 'message', 'param']:
                    if hasattr(body, field):
                        print(f"{yellow}  - {field}: {getattr(body, field)}{reset_color}")

        print(f"{yellow}========================={reset_color}")

        err_str = str(ev).lower()
        if "tokens" in err_str or "rate" in err_str or "429" in err_str or "requests" in err_str:
            # Longer cooldown (30s) to let the rate limit window reset
            cooldown_until = max(cooldown_until, loop.time() + 30.0)
            consecutive_errors += 1
            asyncio.create_task(reset_chat_ctx())

    stop_event = asyncio.Event()
    video_ended_event = asyncio.Event()  # Signals video processing is done

    @session.on("close")
    def _on_close(_):
        stop_event.set()

    # Register RPC handler for video completion signal from video processor
    @ctx.room.local_participant.register_rpc_method("video_complete")
    async def on_video_complete(data: rtc.RpcInvocationData) -> str:
        print(f"\n{yellow}{'='*50}{reset_color}")
        print(f"{yellow}VIDEO PLAYBACK COMPLETE{reset_color}")
        print(f"{yellow}Switching to text chat mode for Q&A...{reset_color}")
        print(f"{yellow}{'='*50}{reset_color}\n")
        video_ended_event.set()
        return "ack"

    # Fallback: detect when video streamer disconnects
    @ctx.room.on("participant_disconnected")
    def _on_participant_left(participant: rtc.RemoteParticipant):
        if is_video_room and participant.identity.startswith("video-streamer-"):
            if not video_ended_event.is_set():
                print(f"\n{yellow}Video streamer disconnected. Switching to chat mode.{reset_color}\n")
                video_ended_event.set()

    async def tail_mic_transcripts() -> None:
        # Disable transcript tailing for video rooms (too noisy) or if explicitly disabled
        if is_video_room or os.getenv("SHOW_MIC_TRANSCRIPT", "0") != "1":
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

    # Helper to find the video streamer participant for RPC calls
    def _find_streamer() -> rtc.RemoteParticipant | None:
        if not is_video_room:
            return None
        for p in ctx.room.remote_participants.values():
            if p.identity.startswith("video-streamer-"):
                return p
        return None

    async def _pause_stream() -> bool:
        """Pause the video stream before generating a reply."""
        streamer = _find_streamer()
        if not streamer:
            return False
        try:
            await ctx.room.local_participant.perform_rpc(
                destination_identity=streamer.identity,
                method="pause_stream",
                payload="",
            )
            await asyncio.sleep(0.1)  # Brief pause for stream to stop
            return True
        except Exception as e:
            print(f"Failed to pause stream: {e}")
            return False

    async def _resume_stream() -> None:
        """Resume the video stream after generating a reply."""
        streamer = _find_streamer()
        if not streamer:
            return
        try:
            await ctx.room.local_participant.perform_rpc(
                destination_identity=streamer.identity,
                method="resume_stream",
                payload="",
            )
        except Exception as e:
            print(f"Failed to resume stream: {e}")

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

            # Pause stream before generating reply (for video rooms)
            stream_was_paused = await _pause_stream()

            if os.getenv("REALTIME_DEBUG", "0") == "1":
                print(f"Realtime request: {instructions}")
            try:
                # Generate reply while stream is paused, disable interruptions
                speech_handle = session.generate_reply(
                    instructions=instructions,
                )
                # Wait for the response to fully complete
                await speech_handle
                consecutive_errors = 0  # Reset on success
            except Exception as exc:
                consecutive_errors += 1
                if "rate" in str(exc).lower() or "429" in str(exc):
                    print(f"Rate limit in run_reply: {exc}")
                raise
            finally:
                # Always resume stream after reply is complete
                if stream_was_paused:
                    await _resume_stream()

            await reset_chat_ctx()

    async def short_loop():
        while not stop_event.is_set() and not video_ended_event.is_set():
            try:
                await run_reply(
                    instructions=(
                        "One short sentence describing the most notable action or statement "
                        "that just occurred. Be specific about what was said or done. "
                        "If nothing notable, say: No significant activity."
                    ),
                    max_tokens=60,
                )
            except Exception as exc:
                print(f"Short loop stopped: {exc}")
                break
            await asyncio.sleep(short_interval)

    async def long_loop():
        while not stop_event.is_set() and not video_ended_event.is_set():
            try:
                await run_reply(
                    instructions=(
                        "Summarize the recent activity: what actions were taken? "
                        "What was said? Who interacted with whom and how? "
                        "Note any notable events, statements, or behaviors. "
                        "Use third-person ('the officer approached...', 'the individual stated...'). "
                        "If no activity occurred, say: No significant activity detected."
                    ),
                    max_tokens=220,
                )
            except Exception as exc:
                print(f"Long loop stopped: {exc}")
                break
            await asyncio.sleep(long_interval)

    async def interactive_chat_loop():
        """Wait for video to end, then enable interactive text chat."""
        await video_ended_event.wait()

        # Update agent instructions for Q&A mode
        qa_instructions = (
            "The video analysis is complete. You are now in Q&A mode. "
            "Answer questions about what you observed in the video footage. "
            "Be specific and reference the events, actions, statements, and interactions you witnessed. "
            "Use third-person language ('the officer', 'the individual'). "
            "Provide detailed, factual responses based on your observations."
        )

        if session.llm and hasattr(session.llm, 'update_options'):
            try:
                session.llm.update_options(instructions=qa_instructions)
            except Exception as e:
                print(f"Could not update instructions: {e}")

        print(f"{green}Q&A mode active. Send text messages to ask about the video.{reset_color}")

        # Keep session alive for text interaction - the agent will auto-respond to text input
        await stop_event.wait()

    tasks = []
    tasks.append(asyncio.create_task(tail_mic_transcripts()))
    if is_video_room:
        if short_interval > 0:
            tasks.append(asyncio.create_task(short_loop()))
        if long_interval > 0:
            tasks.append(asyncio.create_task(long_loop()))
        tasks.append(asyncio.create_task(interactive_chat_loop()))
    await stop_event.wait()
    for task in tasks:
        task.cancel()
    if event_logger:
        event_logger.close()


if __name__ == "__main__":
    # Validate API key based on provider
    if LLM_PROVIDER == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("Missing GOOGLE_API_KEY. Set it in .env.local.")
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY. Set it in .env.local.")

    _startup_agent = VideoAssistant()

    # Validate model type and print config
    if LLM_PROVIDER == "gemini":
        if not isinstance(_startup_agent.llm, google.realtime.RealtimeModel):
            raise RuntimeError("Expected Google RealtimeModel; check plugin installation and config.")
        print(
            f"Gemini Realtime configured "
            f"(model={_startup_agent._realtime_model}, modalities={_startup_agent._realtime_modalities})."
        )
    else:
        if not isinstance(_startup_agent.llm, openai.realtime.RealtimeModel):
            raise RuntimeError("Expected OpenAI RealtimeModel; check plugin installation and config.")
        print(
            f"OpenAI Realtime configured "
            f"(model={_startup_agent._realtime_model}, modalities={_startup_agent._realtime_modalities})."
        )

    agents.cli.run_app(server)
