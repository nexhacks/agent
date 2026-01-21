# Clearance - Tamper-Evident Body Cam Video Analysis System

A real-time body camera video analysis and verification system that combines AI-powered threat detection with blockchain-backed evidence integrity. The system processes body cam footage to detect critical incidents (gunshots, weapons, persons down), streams analysis events in real-time, and creates immutable cryptographic receipts anchored on the Solana blockchain.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Features Overview](#features-overview)
- [Technology Stack](#technology-stack)
- [Web Frontend (Next.js)](#web-frontend-nextjs)
- [Python Backend](#python-backend)
- [Machine Learning Models](#machine-learning-models)
- [External Services & APIs](#external-services--apis)
- [Event Tier Classification](#event-tier-classification)
- [Database Schemas](#database-schemas)
- [Environment Configuration](#environment-configuration)
- [Workflow Diagrams](#workflow-diagrams)
- [API Reference](#api-reference)
- [Setup & Deployment](#setup--deployment)
- [Security & Compliance](#security--compliance)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLEARANCE SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐         ┌──────────────────────────────────────┐   │
│  │   Web Frontend      │         │        Python Backend                │   │
│  │   (Next.js)         │  HTTP   │        (Port 8000)                   │   │
│  │                     │◄───────►│                                      │   │
│  │  • Upload Page      │  SSE    │  ┌─────────────────────────────────┐ │   │
│  │  • Receipt Page     │         │  │    Video Stream Processor       │ │   │
│  │  • Verify Page      │         │  │    (SSE Streaming)              │ │   │
│  │  • Events Page      │         │  │                                 │ │   │
│  └─────────────────────┘         │  │  • Frame Extraction (3s)        │ │   │
│           │                      │  │  • GPT-4o-mini Vision           │ │   │
│           │                      │  │  • Parallel Audio Analysis      │ │   │
│           ▼                      │  └─────────────────────────────────┘ │   │
│  ┌─────────────────────┐         │                                      │   │
│  │   External APIs     │         │  ┌─────────────────────────────────┐ │   │
│  │                     │         │  │    Audio Gunshot Detector       │ │   │
│  │  • Solana Devnet    │         │  │    (YAMNet ML Model)            │ │   │
│  │  • IPFS (Pinata)    │         │  │                                 │ │   │
│  │  • Supabase         │         │  │  • 521 AudioSet Classes         │ │   │
│  │  • Vercel Blob      │         │  │  • Gunshot/Explosion Detection  │ │   │
│  │  • Overshoot SDK    │         │  │  • Heuristic Fallback           │ │   │
│  │  • LiveKit          │         │  └─────────────────────────────────┘ │   │
│  │  • VAPI (Dispatch)  │         │                                      │   │
│  └─────────────────────┘         │  ┌─────────────────────────────────┐ │   │
│                                  │  │    SQLite Database              │ │   │
│                                  │  │    (video_events.db)            │ │   │
│                                  │  └─────────────────────────────────┘ │   │
│                                  └──────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features Overview

### 1. Real-Time Video Analysis
- **Frame-by-Frame Processing**: Analyzes video frames every 3 seconds using GPT-4o-mini Vision
- **Action Detection**: Identifies critical events (weapons, threats, physical altercations)
- **Scene Descriptions**: Periodic environmental and person descriptions every 30 seconds
- **Server-Sent Events (SSE)**: Real-time streaming of analysis events to frontend

### 2. Audio Gunshot Detection
- **YAMNet ML Model**: TensorFlow-based sound classification with 521 AudioSet classes
- **Gunshot Classes Detected**: Gunshot/gunfire, machine gun, fusillade, artillery fire, cap gun, explosion, bang
- **Heuristic Fallback**: Spectral analysis for systems without TensorFlow
- **Taser Detection**: High-frequency sound detection for taser deployment

### 3. Blockchain Evidence Integrity
- **SHA-256 File Hashing**: Cryptographic fingerprint of video files
- **IPFS Storage**: Distributed content-addressed storage via Pinata
- **Solana Memo Transactions**: Immutable on-chain record of evidence metadata
- **Tamper-Proof Receipts**: Verifiable receipts with blockchain anchoring

### 4. Event Classification & Display
- **Color-Coded Event Tiers**: Critical (red), Warning (orange), Audio (blue), Action (red), Scene (purple), Transcript (green)
- **Time-Synced Playback**: Events appear as video reaches their timestamp
- **Critical Alert Animations**: Pulsing red background for SHOTS FIRED, PERSON DOWN

### 5. Emergency Dispatch (VAPI Integration)
- **Automated Phone Calls**: Trigger EMS and backup dispatch on critical events
- **Phone Number Configuration**: EMS, Backup, and Supervisor numbers
- **Cooldown Protection**: Rate limiting to prevent duplicate calls

---

## Technology Stack

### Frontend (web-demo)
| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 16.1.3 | React framework with App Router |
| React | 19.2.3 | UI library |
| TailwindCSS | 4.x | Utility-first CSS |
| @solana/web3.js | 1.98.4 | Solana blockchain interaction |
| @supabase/supabase-js | 2.x | PostgreSQL database client |
| livekit-client | 2.6.0 | WebRTC video streaming |
| @overshoot/sdk | 0.1.0-alpha.2 | Real-time video vision AI |
| @vercel/blob | 0.x | File storage |

### Backend (livekit-gemini-agent)
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.12+ | Runtime |
| OpenCV | 4.12+ | Video frame extraction |
| TensorFlow | 2.15+ | YAMNet ML model |
| TensorFlow Hub | 0.16+ | Pretrained model loading |
| livekit-agents | 1.3+ | LiveKit SDK |
| NumPy | 1.26+ | Numerical operations |
| SciPy | 1.11+ | Signal processing (FFT) |
| FFmpeg | - | Audio extraction |

---

## Web Frontend (Next.js)

### Directory Structure
```
web-demo/
├── app/
│   ├── api/
│   │   ├── publish/route.ts       # Receipt creation endpoint
│   │   ├── receipt/[id]/route.ts  # Fetch receipt details
│   │   ├── verify/[id]/route.ts   # Blockchain verification
│   │   ├── blob/upload/route.ts   # Vercel Blob upload
│   │   └── livekit-sandbox/       # LiveKit token generation
│   ├── upload/page.tsx            # Main video upload & analysis UI
│   ├── events/page.tsx            # Event history display
│   ├── receipt/[id]/page.tsx      # Receipt display page
│   ├── verify/[id]/page.tsx       # Verification page
│   ├── globals.css                # Global styles & animations
│   └── layout.tsx                 # Root layout
├── lib/
│   ├── solana.ts                  # Solana connection & memo functions
│   ├── pinata.ts                  # IPFS pinning utilities
│   ├── supabase.ts                # Supabase client setup
│   ├── hashing.ts                 # SHA-256 computation
│   └── memo.ts                    # Receipt memo serialization
└── .env.local                     # Environment variables
```

### Key Pages

#### `/upload` - Video Upload & Analysis
The main interface for body cam video analysis:
- **Video Upload**: Drag-and-drop or file picker for video upload
- **SSE Event Stream**: Real-time connection to backend for analysis events
- **Dual Audio Processing**: Parallel audio gunshot detection stream
- **Time-Synced Display**: Events appear as video playback reaches their timestamps
- **Event Filtering**: Filter by tier (Critical, Warning, Audio, Action, Scene, Transcript)
- **Transcript Panel**: Dedicated area for speech transcription

#### `/receipt/[id]` - Receipt Display
Shows the cryptographic receipt for a video:
- SHA-256 file hash
- IPFS CID (Content Identifier)
- Solana transaction signature
- File metadata (size, filename, timestamp)
- Links to IPFS gateway and Solana explorer

#### `/verify/[id]` - Blockchain Verification
Verifies receipt integrity against blockchain:
- Fetches receipt from database
- Queries Solana blockchain for memo transaction
- Compares on-chain data with database records
- Shows match/mismatch status for each field

### Utility Libraries

#### `lib/solana.ts`
```typescript
// Connection to Solana devnet
getSolanaConnection(): Connection

// Load server keypair from environment
getServerKeypair(): Keypair

// Write memo to blockchain
anchorMemo(memo: string): Promise<string> // Returns tx signature

// Read memo from existing transaction
fetchMemoForTransaction(signature: string): Promise<string>
```

#### `lib/pinata.ts`
```typescript
// Upload file to IPFS
pinFileToIPFS(file: File): Promise<string> // Returns IPFS CID
```

#### `lib/memo.ts`
```typescript
// Receipt memo format
interface ReceiptMemo {
  v: number;           // Version (1)
  app: string;         // "clearance"
  receiptId: string;   // UUID
  hashAlgo: string;    // "sha256"
  sha256: string;      // File hash
  cid: string;         // IPFS CID
  bytes: number;       // File size
}
```

---

## Python Backend

### Directory Structure
```
livekit-gemini-agent/
├── video_review_server.py      # HTTP server (port 8000)
├── video_stream_processor.py   # SSE streaming & GPT-4o-mini analysis
├── realtime_video_processor.py # Experimental LiveKit Realtime API
├── video_store.py              # SQLite database operations
├── main.py                     # Legacy entry point
├── templates/
│   └── video_review.html       # Standalone review interface
├── uploads/                    # Video file storage
└── video_events.db             # SQLite database
```

### Processing Modes

#### 1. SSE Stream Processor (Recommended)
```python
class VideoStreamProcessor:
    """Real-time video analysis with Server-Sent Events streaming"""

    Features:
    - Frame extraction every 3 seconds
    - GPT-4o-mini Vision API for frame analysis
    - Parallel audio gunshot detection
    - Event queue for SSE streaming
    - Image compression (512x512 @ 60% JPEG)
    - Rate limiting (150k TPM budget)
```

#### 2. Audio Gunshot Detector
```python
class AudioStreamProcessor:
    """Specialized audio-only gunshot detection"""

    Methods:
    - YAMNet ML detection (TensorFlow)
    - Heuristic fallback (spectral analysis)
    - Taser sound detection

    Gunshot Indicators:
    - Frequency: 2500-2600 Hz
    - Crest factor: >= 3.0
    - Attack time: < 10ms
    - Duration: < 150ms
```

#### 3. Realtime API (Experimental)
```python
class RealtimeVideoProcessor:
    """LiveKit + OpenAI Realtime API streaming"""

    Status: Experimental (connection timeout issues)
    - Streams video at 5 FPS max
    - Audio in PCM format (24kHz)
    - RPC control methods
```

### System Prompts

#### Main Analysis Prompt
```
You are an AI analyzing police body camera footage. Focus ONLY on critical events:

CRITICAL ALERTS (use these exact phrases):
- "GUN DRAWN" / "GUN POINTED" / "SHOTS FIRED"
- "TASER DRAWN" / "TASER FIRED"
- "WEAPON!" (knives, blunt objects)
- "CAMERA BLOCKED" / "CAMERA OBSCURED"
- "PERSON DOWN" / "PERSON PRONE"
- "AGGRESSIVE ACTIONS" / "PHYSICAL ALTERCATION"

Rules:
- Maximum 15 words per observation
- Third-person language only
- Never use officer names
- Report positions and movements
```

#### Scene Description Prompt (Every 30s)
```
Describe the scene in detail:
- Location type (interior/exterior, building type)
- Each person: role, clothing colors, build, hair, position
- Write as flowing paragraph (2-3 sentences)
```

### HTTP Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves video_review.html template |
| GET | `/api/videos` | List all processed videos |
| GET | `/api/events?video_id=X&since_id=Y` | Get events for video |
| GET | `/api/stream/{video_id}` | SSE stream for video events |
| GET | `/api/audio/stream/{video_id}` | SSE stream for audio detection |
| GET | `/video/{video_id}` | Stream MP4 with range support |
| POST | `/upload` | Upload video, returns SSE stream |
| POST | `/audio/upload` | Audio-only gunshot detection |

### Rate Limiting
```python
class RateLimiter:
    """Token bucket rate limiter for OpenAI API"""

    Configuration:
    - TPM Budget: 150,000 tokens/minute
    - Tokens per image: ~500 (after compression)
    - Exponential backoff on 429 errors
    - Automatic retry with delays
```

---

## Machine Learning Models

### 1. YAMNet (Audio Classification)

| Property | Value |
|----------|-------|
| Source | TensorFlow Hub |
| Model Size | ~3.5MB |
| Input Format | 16kHz mono audio |
| Classes | 521 AudioSet categories |
| Window Size | 0.96 seconds |
| Overlap | 50% |

**Gunshot-Related Class IDs:**
```python
GUNSHOT_CLASS_IDS = [
    427,  # Gunshot, gunfire
    428,  # Machine gun
    429,  # Fusillade
    430,  # Artillery fire
    431,  # Cap gun
    426,  # Explosion
    494,  # Bang
]
```

**Detection Configuration:**
```python
YAMNET_CONFIDENCE_THRESHOLD = 0.15  # Minimum confidence
GUNSHOT_COOLDOWN = 0.3              # Seconds between alerts
```

### 2. GPT-4o-mini (Vision)

| Property | Value |
|----------|-------|
| Provider | OpenAI |
| Use Case | Frame-by-frame video analysis |
| Token Usage | ~500 tokens per 512x512 image |
| Temperature | 0.2 (deterministic) |
| Max Tokens | 80-200 per response |

**Image Preprocessing:**
```python
MAX_WIDTH = 512
MAX_HEIGHT = 512
JPEG_QUALITY = 60
```

### 3. Heuristic Gunshot Detection (Fallback)

For systems without TensorFlow, spectral analysis is used:

```python
# Detection Parameters
GUNSHOT_FREQ_MIN = 2500      # Hz
GUNSHOT_FREQ_MAX = 2600      # Hz
AMPLITUDE_THRESHOLD = 0.80
CREST_FACTOR_THRESHOLD = 3.0
ATTACK_TIME_THRESHOLD = 10   # ms
SOUND_DURATION_MAX = 150     # ms

# Taser Detection
TASER_FREQ_THRESHOLD = 3500  # Hz
TASER_AMPLITUDE_THRESHOLD = 0.50
```

---

## External Services & APIs

### OpenAI
- **Vision Model**: `gpt-4o-mini` for frame analysis
- **Realtime Model**: `gpt-realtime` (experimental)
- **Rate Limits**: 150k TPM budget enforced
- **Authentication**: Bearer token via `OPENAI_API_KEY`

### LiveKit
- **Purpose**: WebRTC video/audio streaming
- **URL**: `wss://clearance-z4l9q6tv.livekit.cloud`
- **Features**: Room creation, token generation, agent dispatch
- **Authentication**: API key + secret

### Solana (Devnet)
- **Purpose**: Immutable evidence anchoring
- **RPC**: `https://api.devnet.solana.com`
- **Program**: Memo Program (`MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr`)
- **Authentication**: Ed25519 keypair

### IPFS (Pinata)
- **Purpose**: Distributed file storage
- **Endpoint**: `https://api.pinata.cloud/pinning/pinFileToIPFS`
- **Gateway**: `https://gateway.pinata.cloud/ipfs/{CID}`
- **Authentication**: JWT token

### Supabase
- **Purpose**: PostgreSQL database
- **Tables**: `receipts`, `events`
- **Authentication**: Service role key

### Vercel Blob
- **Purpose**: Video file storage
- **Features**: Public URLs, streaming support
- **Authentication**: Read/write token

### VAPI (Voice AI)
- **Purpose**: Automated phone dispatch
- **Features**: Outbound calls to EMS/backup
- **Endpoint**: `https://api.vapi.ai/call/phone`
- **Authentication**: Private API key

### Overshoot SDK
- **Purpose**: Real-time video vision analysis
- **Features**: Alternative to GPT-4o-mini Vision
- **Authentication**: Public API key

---

## Event Tier Classification

Events are classified into tiers for visual distinction:

| Tier | Color | Keywords/Triggers |
|------|-------|-------------------|
| **Critical** | Red (#ff0000) | SHOTS FIRED, PERSON ON FLOOR, PERSON DOWN, PERSON PRONE, CAMERA BLOCKED |
| **Warning** | Orange (#f59e0b) | GUN DRAWN, GUN VISIBLE, GUN POINTED, TASER DRAWN, TASER FIRED, WEAPON!, CAMERA OBSCURED |
| **Audio** | Blue (#60a5fa) | Events prefixed with "AUDIO:" |
| **Action** | Light Red | General action events |
| **Scene** | Purple (#c792ea) | Scene descriptions |
| **Transcript** | Green (#6ee7a8) | Speech transcriptions |
| **No Activity** | Gray (muted) | Placeholder during quiet periods |

### CSS Animations

```css
/* Critical alert pulsing animation */
@keyframes alertPulse {
  0%, 100% {
    background: linear-gradient(90deg, rgba(255,0,0,0.2), rgba(255,0,0,0.02), transparent);
  }
  50% {
    background: linear-gradient(90deg, rgba(255,0,0,0.35), rgba(255,0,0,0.1), transparent);
  }
}

/* New event fade-in */
@keyframes fadeIn {
  from {
    opacity: 0;
    background: rgba(74, 163, 255, 0.3);
  }
  to {
    opacity: 1;
    background: #101622;
  }
}
```

---

## Database Schemas

### SQLite (Python Backend)

```sql
-- Videos table
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,  -- 'processing', 'streaming', 'done', 'error'
    duration_sec REAL
);

-- Events table
CREATE TABLE video_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    offset_sec REAL NOT NULL,
    kind TEXT NOT NULL,  -- 'action', 'scene', 'action_realtime', 'audio', 'no_activity'
    text TEXT NOT NULL
);

-- Indexes
CREATE INDEX idx_events_video_id ON video_events(video_id);
CREATE INDEX idx_events_offset ON video_events(offset_sec);
```

### Supabase (PostgreSQL)

```sql
-- Receipts table
CREATE TABLE receipts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sha256_hex TEXT NOT NULL,
    ipfs_cid TEXT NOT NULL,
    solana_tx_sig TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    filename TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Events table
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event TEXT NOT NULL,
    tag TEXT,
    occurred_at TIMESTAMP NOT NULL,
    room TEXT,
    camera TEXT,
    transcript TEXT,
    source TEXT,
    solana_tx_sig TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Environment Configuration

### Web Frontend (.env.local)

```bash
# Database
DATABASE_URL="file:./prisma/dev.db"

# IPFS Storage (Pinata)
PINATA_JWT="your-pinata-jwt-token"

# Solana Blockchain
SOLANA_KEYPAIR_JSON=[...ed25519-keypair-bytes...]
SOLANA_RPC_URL=https://api.devnet.solana.com

# LiveKit Video Streaming
LIVEKIT_URL=wss://clearance-z4l9q6tv.livekit.cloud
LIVEKIT_SANDBOX_ID=clearance-1op0ce

# Vercel Blob Storage
BLOB_READ_WRITE_TOKEN="vercel_blob_rw_..."

# Supabase Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# Overshoot Vision AI
NEXT_PUBLIC_OVERSHOOT_API_KEY=ovs_...

# VAPI Phone Dispatch
VAPI_PRIVATE_KEY=your-vapi-private-key
VAPI_PHONE_NUMBER_ID=your-phone-number-id
DISPATCH_EMS_PHONE=+14436367028
DISPATCH_BACKUP_PHONE=+16696390502
DISPATCH_SUPERVISOR_PHONE=+14083103927
VAPI_ASSISTANT_ID=your-assistant-id
```

### Python Backend (.env)

```bash
# OpenAI API
OPENAI_API_KEY=sk-proj-...

# Processing Configuration
VIDEO_ACTION_MODEL=gpt-4o-mini
USE_STREAM_PROCESSOR=1
STREAM_FRAME_INTERVAL=3
ENABLE_SCENE_DESCRIPTIONS=1
SCENE_DESCRIPTION_INTERVAL=30

# Image Compression
IMAGE_MAX_WIDTH=512
IMAGE_MAX_HEIGHT=512
IMAGE_QUALITY=60

# Rate Limiting
VIDEO_TPM_BUDGET=150000
VIDEO_TOKENS_PER_IMAGE=500

# YAMNet Configuration
USE_YAMNET=1
YAMNET_CONFIDENCE_THRESHOLD=0.15
YAMNET_WINDOW_SEC=0.96
YAMNET_DEBUG=1

# Heuristic Detection Fallback
GUNSHOT_AMPLITUDE_THRESHOLD=0.80
GUNSHOT_FREQ_MIN=2500
GUNSHOT_FREQ_MAX=2600
TASER_FREQ_THRESHOLD=3500
GUNSHOT_COOLDOWN=0.3

# LiveKit (for Realtime API mode)
LIVEKIT_URL=wss://...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...

# Storage
VIDEO_UPLOAD_DIR=uploads
VIDEO_DB_PATH=video_events.db
```

---

## Workflow Diagrams

### Video Upload & Analysis Flow

```
┌──────────┐    ┌──────────────┐    ┌─────────────────┐
│  User    │───►│  Upload Page │───►│  Python Backend │
│  Browser │    │  (Next.js)   │    │   (Port 8000)   │
└──────────┘    └──────────────┘    └─────────────────┘
                       │                     │
                       │ SSE Connection      │ Video Processing
                       │◄────────────────────┤
                       │                     │
                       │                     ▼
                       │            ┌─────────────────┐
                       │            │  Frame Extract  │
                       │            │  (Every 3 sec)  │
                       │            └────────┬────────┘
                       │                     │
                       │                     ▼
                       │            ┌─────────────────┐
                       │            │  GPT-4o-mini    │
                       │            │  Vision API     │
                       │            └────────┬────────┘
                       │                     │
                       │  Events (SSE)       │
                       │◄────────────────────┤
                       ▼                     │
              ┌─────────────────┐            │
              │  Event Display  │            │
              │  (Time-synced)  │            │
              └─────────────────┘            │
                                            │
              Parallel Audio Analysis        │
              ┌─────────────────┐            │
              │  YAMNet ML      │◄───────────┘
              │  Gunshot Detect │  (Audio extracted)
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  AUDIO: Events  │
              │  (Blue tier)    │
              └─────────────────┘
```

### Receipt Creation & Verification Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RECEIPT CREATION                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Video Upload ──► 2. SHA-256 Hash ──► 3. IPFS Pin (Pinata)       │
│                             │                    │                   │
│                             ▼                    ▼                   │
│                      ┌─────────────┐      ┌─────────────┐           │
│                      │  hash: a3f..│      │  CID: Qm... │           │
│                      └─────────────┘      └─────────────┘           │
│                             │                    │                   │
│                             └────────┬───────────┘                   │
│                                      │                               │
│                                      ▼                               │
│                      4. Create Receipt Memo (JSON)                   │
│                                      │                               │
│                                      ▼                               │
│                      5. Sign & Submit to Solana                      │
│                                      │                               │
│                                      ▼                               │
│                      ┌───────────────────────────┐                   │
│                      │  Solana Memo Transaction  │                   │
│                      │  sig: 4x7k...            │                   │
│                      └───────────────────────────┘                   │
│                                      │                               │
│                                      ▼                               │
│                      6. Store in Supabase                            │
│                                      │                               │
│                                      ▼                               │
│                      7. Return Receipt ID to User                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        VERIFICATION                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. User enters Receipt ID                                          │
│                 │                                                    │
│                 ▼                                                    │
│  2. Fetch from Supabase ──────────────► Database Record             │
│                 │                        • sha256_hex               │
│                 │                        • ipfs_cid                 │
│                 ▼                        • solana_tx_sig            │
│  3. Query Solana RPC ─────────────────► Blockchain Memo             │
│                 │                        • sha256 (from memo)       │
│                 │                        • cid (from memo)          │
│                 ▼                                                    │
│  4. Compare Values                                                   │
│                 │                                                    │
│                 ▼                                                    │
│     ┌───────────────────┬───────────────────┐                       │
│     │ ✓ VERIFIED        │ ✗ TAMPERED        │                       │
│     │ All fields match  │ Mismatch detected │                       │
│     └───────────────────┴───────────────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## API Reference

### Web Frontend APIs

#### POST `/api/publish`
Create a new evidence receipt.

**Request:**
```typescript
FormData {
  file: File  // Video file
}
```

**Response:**
```json
{
  "receiptId": "uuid",
  "sha256": "hex-string",
  "cid": "Qm...",
  "solanaTxSig": "base58-string",
  "bytes": 123456,
  "filename": "video.mp4"
}
```

#### GET `/api/receipt/[id]`
Retrieve receipt details.

**Response:**
```json
{
  "id": "uuid",
  "sha256_hex": "hex-string",
  "ipfs_cid": "Qm...",
  "solana_tx_sig": "base58-string",
  "bytes": 123456,
  "filename": "video.mp4",
  "created_at": "ISO-8601"
}
```

#### GET `/api/verify/[id]`
Verify receipt against blockchain.

**Response:**
```json
{
  "verified": true,
  "database": {
    "sha256": "...",
    "cid": "...",
    "bytes": 123456
  },
  "blockchain": {
    "sha256": "...",
    "cid": "...",
    "bytes": 123456
  }
}
```

### Python Backend APIs

#### POST `/upload`
Upload video for analysis.

**Request:**
```
Content-Type: multipart/form-data
file: video file
```

**Response:** Server-Sent Events stream
```
event: event
data: {"id": 1, "video_id": "abc", "offset_sec": 3.0, "kind": "action", "text": "Officer approaches vehicle"}

event: status
data: {"status": "processing", "progress": 0.5}

event: done
data: {"video_id": "abc", "total_events": 42}
```

#### GET `/api/stream/{video_id}`
SSE stream for video events.

**Response:** Server-Sent Events
```
data: {"id": 1, "offset_sec": 3.0, "kind": "action", "text": "..."}
```

#### GET `/api/audio/stream/{video_id}`
SSE stream for audio detection.

**Response:** Server-Sent Events
```
data: {"id": 1, "offset_sec": 5.2, "kind": "audio", "text": "AUDIO: Gunshot detected (confidence: 0.87)"}
```

---

## Setup & Deployment

### Prerequisites
- Node.js 18+
- Python 3.12+
- FFmpeg (for audio extraction)
- TensorFlow (optional, for YAMNet)

### Web Frontend Setup

```bash
cd web-demo

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local with your credentials

# Run development server
npm run dev
# Runs on http://localhost:3000
```

### Python Backend Setup

```bash
cd livekit-gemini-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run server
python video_review_server.py
# Runs on http://localhost:8000
```

### Production Deployment

**Frontend (Vercel):**
```bash
vercel deploy --prod
```

**Backend (Docker):**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "video_review_server.py"]
```

---

## Security & Compliance

### Cryptographic Integrity
- **SHA-256**: FIPS 180-4 compliant file hashing
- **Ed25519**: Solana transaction signing
- **Immutable Storage**: IPFS + Solana blockchain

### Privacy Considerations
- No PII stored by default
- Video content remains on user-controlled storage
- Only cryptographic hashes stored on-chain
- Service role keys protected server-side

### Rate Limiting
- OpenAI TPM budget enforcement
- Exponential backoff on API errors
- Cooldown protection for dispatch calls

### CORS Configuration
- Controlled cross-origin access
- Preflight request handling
- Credential support where needed

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Support

For issues and feature requests, please open a GitHub issue.
