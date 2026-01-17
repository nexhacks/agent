import asyncio
import os
import re
from datetime import datetime

from dotenv import load_dotenv

from event_log import get_event_logger, open_event_db
from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io, llm
from livekit.agents.voice import io as voice_io
from livekit.plugins import (
    openai,
    silero,
)

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env.local")
load_dotenv(ENV_PATH)
AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "assistant")


class VideoAssistant(Agent):
    def __init__(self) -> None:
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
                model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime"),
                voice=os.getenv("OPENAI_REALTIME_VOICE", "alloy"),
                modalities=["text"],
            ),
        )


class ConsoleTextOutput(voice_io.TextOutput):
    def __init__(
        self,
        *,
        label: str,
        next_in_chain: voice_io.TextOutput | None = None,
        logger: "EventLogger | None" = None,
    ) -> None:
        super().__init__(label=label, next_in_chain=next_in_chain)
        self._buffer = ""
        self._first_seen_at: datetime | None = None
        self._logger = logger
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

    def _timestamp(self) -> str:
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
            print(f"{self._red}Action [{self._timestamp()}]: {text}{self._reset}")
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
    session = AgentSession(vad=silero.VAD.load())
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Set it in .env.local.")
    event_logger = get_event_logger()

    await session.start(
        room=ctx.room,
        agent=VideoAssistant(),
        room_options=room_io.RoomOptions(
            video_input=True,
            text_output=True,
            audio_input=False,
            text_input=False,
            audio_output=False,
        ),
    )
    session.output.transcription = ConsoleTextOutput(
        label="console",
        next_in_chain=session.output.transcription,
        logger=event_logger,
    )
    loop = asyncio.get_running_loop()
    cooldown_until = 0.0

    async def reset_chat_ctx() -> None:
        if session.llm and hasattr(session.llm, "update_chat_ctx"):
            try:
                await session.llm.update_chat_ctx(llm.ChatContext.empty())
            except Exception as exc:
                print(f"Chat context reset failed: {exc}")

    @session.on("error")
    def _on_error(ev):
        nonlocal cooldown_until
        print(f"Agent error: {ev}")
        if "tokens" in str(ev).lower():
            cooldown_until = max(cooldown_until, loop.time() + 5.0)
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

    short_interval = float(os.getenv("SCENE_SHORT_INTERVAL", "3"))
    long_interval = float(os.getenv("SCENE_LONG_INTERVAL", "7"))
    llm_lock = asyncio.Lock()

    async def run_reply(instructions: str, max_tokens: int) -> None:
        async with llm_lock:
            if loop.time() < cooldown_until:
                await asyncio.sleep(cooldown_until - loop.time())
            if session.llm and hasattr(session.llm, "update_options"):
                try:
                    session.llm.update_options(max_response_output_tokens=max_tokens)
                except Exception:
                    pass
            await session.generate_reply(instructions=instructions)
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
    agents.cli.run_app(server)
