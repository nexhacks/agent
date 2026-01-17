# LiveKit Body-Cam Analyst Agent

This folder contains a LiveKit agent that watches a video track in real time and emits
short, third-person scene observations. It is built on LiveKit Agents with the OpenAI
Realtime model, plus a local publisher that streams your webcam + microphone into a room.

## What the agent does

- `main.py` starts an `AgentServer` and registers an agent named by `LIVEKIT_AGENT_NAME`.
- The `VideoAssistant` model is an OpenAI realtime LLM configured for text-only output.
- A `ConsoleTextOutput` filter enforces strict language rules and logs output to SQLite.
- Two loops run in the session:
  - **Short loop** every `SCENE_SHORT_INTERVAL` seconds (default `3`) with a 6–12 word
    action summary.
  - **Long loop** every `SCENE_LONG_INTERVAL` seconds (default `7`) with richer context.
- If mic transcription is enabled in the publisher, the agent can tail those transcripts
  and print them in green from the shared event log.

Supporting scripts:

- `publish_camera.py`: publishes webcam + mic to LiveKit, then dispatches the agent.
- `record_and_transcribe.py`: quick mic recording + Vosk/Whisper sanity test.
- `event_log.py`: SQLite-backed event logger (`events.db` by default).
- `video_review_server.py`: offline (non-LiveKit) local video analysis + web UI.

## Local test setup

### 1) Prereqs

- Python 3.11+
- LiveKit server (Cloud or local) with API key/secret
- OpenAI API key
- Webcam + microphone permissions for your terminal

Optional:
- `ffmpeg` (macOS device listing / camera selection helper)

### 2) Install dependencies

From the agent folder:

```bash
cd /Users/tanujsiripurapu/Code/nexhacks/agent
uv sync
```

If you are not using `uv`, install with pip:

```bash
python -m pip install .
```

### 3) Configure `.env.local` (or `.env`)

Create `/Users/tanujsiripurapu/Code/nexhacks/agent/.env.local` (or `.env`):

```bash
OPENAI_API_KEY=sk-...
LIVEKIT_URL=wss://your-livekit-host
LIVEKIT_API_KEY=lk_api_key
LIVEKIT_API_SECRET=lk_api_secret
LIVEKIT_ROOM=demo
LIVEKIT_AGENT_NAME=assistant
```

Notes:
- You can also set `LIVEKIT_TOKEN` to bypass token creation in `publish_camera.py`.
- The LiveKit API key/secret are still needed for agent dispatch.

### 4) Run the agent server

```bash
cd /Users/tanujsiripurapu/Code/nexhacks/agent
uv run python main.py start
```

You should see: `Agent session started in room: ...` once dispatched.

If you want the LiveKit Agents CLI help:

```bash
uv run python main.py --help
```

### 5) Publish camera + mic and dispatch the agent

In a second terminal:

```bash
cd /Users/tanujsiripurapu/Code/nexhacks/agent
uv run python publish_camera.py
```

This will:
- Connect to LiveKit
- Publish the microphone and camera tracks
- Create an agent dispatch for the room
- Stream frames to the agent for analysis

If successful, the agent terminal prints red `Action [...]` lines at intervals.

## Useful environment options

### Agent (`main.py`)
- `SCENE_SHORT_INTERVAL` / `SCENE_LONG_INTERVAL`: tweak response cadence.
- `OPENAI_REALTIME_MODEL`, `OPENAI_REALTIME_VOICE`: override realtime model config.
  - Example: `OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview` or
    `OPENAI_REALTIME_MODEL=gpt-4o-mini-realtime-preview`
- `SHOW_MIC_TRANSCRIPT=1`: tail mic transcripts from the event DB.
- `EVENT_LOG_ENABLED=1`, `EVENT_LOG_PATH=events.db`: logging controls.

### Publisher (`publish_camera.py`)
- `CAMERA_INDEX`: force a specific camera device index.
- `CAMERA_WIDTH`, `CAMERA_HEIGHT`: capture resolution (default 640x360).
- `SHOW_PREVIEW=1`: OpenCV preview window.
- `MIC_TRANSCRIBE=1`: enable local Whisper transcripts (default on).
- `MIC_TRANSCRIBE_MODEL=tiny`: Whisper model size.
- `MIC_TRANSCRIPT_STDOUT=1`: print mic transcripts to stdout.
- `MIC_TRANSCRIBE_INTERVAL`, `MIC_TRANSCRIBE_WINDOW`: tuning knobs.

## Troubleshooting

- **Camera not opening**: grant camera permission to your terminal app and set
  `CAMERA_INDEX`. On macOS, try `CAMERA_BACKEND=avfoundation`.
- **No agent output**: confirm the agent terminal is running and that the publisher
  successfully dispatched the agent (it retries up to 5 times).
- **OpenAI Realtime 401**: the API key is missing/invalid or the model is not
  allowed for your key. Verify `OPENAI_API_KEY` is loaded and set
  `OPENAI_REALTIME_MODEL` to a model your account can access.
- **Token issues**: either set `LIVEKIT_TOKEN` directly or ensure
  `LIVEKIT_API_KEY/SECRET` are correct.
- **Mic transcript errors**: install `faster-whisper` (already in dependencies) or
  set `MIC_TRANSCRIBE=0` to disable.

## Deploy to LiveKit Cloud Agents

LiveKit Cloud Agents runs this worker remotely so you can dispatch it from any
publisher. High-level flow:

1) build a container from this folder
2) register/deploy it to LiveKit Cloud
3) set secrets (OpenAI key, model overrides, agent name)
4) connect remotely using `publish_camera.py` or your own app

### 1) Install the LiveKit CLI

```bash
brew install livekit
```

### 2) Authenticate and select your Cloud project

```bash
lk cloud auth
lk project set-default "<your-project-name>"
```

### 3) Build container locally (optional smoke check)

```bash
cd /Users/tanujsiripurapu/Code/nexhacks/agent
docker build -t livekit-bodycam-agent .
```

### 4) Create and deploy the agent in LiveKit Cloud

Run from the `agent/` directory:

```bash
lk agent create
# later updates:
lk agent deploy
```

This generates a `livekit.toml` and builds/deploys the image to your project.

### 5) Configure Cloud secrets / env vars

Set these in the LiveKit Cloud console (or `lk agent secrets set`):

- `OPENAI_API_KEY`
- `LIVEKIT_AGENT_NAME` (must match what you dispatch; default is `assistant`)
- `OPENAI_REALTIME_MODEL` (optional, set to a model your key can access)

LiveKit Cloud injects `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`
automatically for the agent worker.

### 6) Connect remotely

From your laptop (or any machine), point `publish_camera.py` at your Cloud URL
and dispatch the same agent name:

```bash
LIVEKIT_URL=wss://<your-subdomain>.livekit.cloud \
LIVEKIT_AGENT_NAME=assistant \
uv run python publish_camera.py
```

You should see the agent logs in the Cloud console and red `Action [...]` lines
locally from the publisher.

## Minimal smoke test (no LiveKit)

```bash
cd /Users/tanujsiripurapu/Code/nexhacks/agent
uv run python record_and_transcribe.py --seconds 3
```

This verifies microphone capture and local transcription.

