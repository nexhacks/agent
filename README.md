# Video Processing Server (LiveKit Gemini Agent)

This repository hosts a local video review server that ingests body-cam style
video, analyzes it with LLM vision + audio pipelines, and streams events to a
simple web UI via Server-Sent Events (SSE). The core implementation lives in
`livekit-gemini-agent/`.

## What It Does

- Accepts video uploads and stores them under `uploads/`.
- Extracts audio and runs gunshot/taser detection.
- Analyzes frames with an LLM for event-style logging (alerts, actions).
- Streams events live to the browser via SSE.
- Persists videos and events in a local SQLite database.

## Processing Modes

The server supports three processing paths:

1. **SSE stream processor (default, recommended)**  
   Uses `video_stream_processor.py` to analyze frames every N seconds, emit
   SSE events, and optionally generate scene descriptions. Audio gunshot
   detection runs in parallel.

2. **LiveKit Realtime API (experimental)**  
   Uses `realtime_video_processor.py` to stream video + audio into a LiveKit
   room for real-time agent processing.

3. **Legacy frame-by-frame**  
   Uses `video_review_server.py` to sample frames on fixed intervals and call
   the vision model directly.

Toggle these with `USE_STREAM_PROCESSOR` and `USE_REALTIME_API`.

## Quick Start

From `livekit-gemini-agent/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[server]"
python video_review_server.py --host 127.0.0.1 --port 8000
```

Open: `http://127.0.0.1:8000`

## API Endpoints

- `GET /`  
  Serves the `templates/video_review.html` UI.
- `POST /upload`  
  Upload a video file (multipart form `file`). Returns `video_id` and
  `stream_url` when SSE streaming is enabled.
- `POST /audio/upload`  
  Upload a video for audio-only gunshot detection. Returns `video_id` and
  `stream_url`.
- `GET /api/videos`  
  List processed videos.
- `GET /api/events?video_id=...&since_id=...`  
  Fetch stored events for a video.
- `GET /api/stream/:video_id`  
  SSE stream of live video+audio events.
- `GET /api/audio/stream/:video_id`  
  SSE stream for audio-only gunshot detection.
- `GET /video/:video_id`  
  Stream the original MP4 file with Range support.

## Configuration

The server reads a `.env.local` file from `livekit-gemini-agent/`.

Common settings:

- `VIDEO_UPLOAD_DIR` (default `uploads`)
- `VIDEO_DB_PATH` (default `video_events.db`)
- `VIDEO_ACTION_MODEL` (default `gpt-4o-mini`)
- `STREAM_FRAME_INTERVAL` (default `3`)
- `IMAGE_MAX_SIZE` (default `512`)
- `IMAGE_QUALITY` (default `70`)
- `USE_STREAM_PROCESSOR` (default `1`)
- `USE_REALTIME_API` (default `1`)
- `ENABLE_SCENE_DESCRIPTIONS` (default `1`)
- `SCENE_DESCRIPTION_INTERVAL` (default `30`)

Realtime (LiveKit) settings:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `LIVEKIT_AGENT_NAME` (default `assistant`)
- `VIDEO_SPEED_MULTIPLIER` (default `1.0`)
- `VIDEO_FRAME_SAMPLE_INTERVAL` (default `5.0`)

Legacy frame-by-frame tuning:

- `VIDEO_SHORT_INTERVAL` (default `10`)
- `VIDEO_LONG_INTERVAL` (default `25`)
- `VIDEO_PROCESS_REALTIME` (default `1`)
- `VIDEO_TPM_BUDGET` (default `150000`)
- `VIDEO_TOKENS_PER_IMAGE` (default `500`)
- `IMAGE_MAX_WIDTH`, `IMAGE_MAX_HEIGHT`, `IMAGE_QUALITY`

Audio gunshot detection (optional):

- `USE_YAMNET` (default `True` in code)
- `YAMNET_CONFIDENCE_THRESHOLD`
- `GUNSHOT_AMPLITUDE_THRESHOLD`
- `GUNSHOT_FREQ_MIN`, `GUNSHOT_FREQ_MAX`

## Data Storage

The server uses SQLite (`video_events.db`) with two tables:

- `videos` — uploaded files and status
- `video_events` — timestamped events (alerts, actions, transcripts)

## Dependencies

- Python 3.12+
- `ffmpeg` (audio extraction)
- `opencv-python`, `numpy`, `python-dotenv`
- LiveKit and OpenAI dependencies from `pyproject.toml`
- Optional: `tensorflow` + `tensorflow-hub` for YAMNet gunshot detection

## Example Upload

```bash
curl -F "file=@/path/to/video.mp4" http://127.0.0.1:8000/upload
```

Then connect to the returned `stream_url` for live events.
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

