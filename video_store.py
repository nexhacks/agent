import os
import sqlite3
from datetime import datetime


DEFAULT_DB_PATH = os.getenv("VIDEO_DB_PATH", "video_events.db")


def open_video_db(path: str | None = None) -> sqlite3.Connection:
    db_path = path or DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            duration_sec REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS video_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            offset_sec REAL NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_video_events ON video_events(video_id, offset_sec)"
    )
    conn.commit()
    return conn


def insert_video(conn: sqlite3.Connection, video_id: str, filename: str, duration_sec: float | None) -> None:
    conn.execute(
        "INSERT INTO videos (id, filename, created_at, status, duration_sec) VALUES (?, ?, ?, ?, ?)",
        (video_id, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "processing", duration_sec),
    )
    conn.commit()


def update_video_status(conn: sqlite3.Connection, video_id: str, status: str) -> None:
    conn.execute("UPDATE videos SET status = ? WHERE id = ?", (status, video_id))
    conn.commit()


def insert_event(
    conn: sqlite3.Connection,
    *,
    video_id: str,
    offset_sec: float,
    kind: str,
    text: str,
) -> None:
    conn.execute(
        "INSERT INTO video_events (video_id, offset_sec, kind, text) VALUES (?, ?, ?, ?)",
        (video_id, float(offset_sec), kind, text),
    )
    conn.commit()


def list_videos(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, filename, created_at, status, duration_sec FROM videos ORDER BY created_at DESC"
    ).fetchall()
    return [
        {
            "id": r[0],
            "filename": r[1],
            "created_at": r[2],
            "status": r[3],
            "duration_sec": r[4],
        }
        for r in rows
    ]


def list_events(conn: sqlite3.Connection, video_id: str, since_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, offset_sec, kind, text FROM video_events WHERE video_id = ? AND id > ? ORDER BY id",
        (video_id, since_id),
    ).fetchall()
    return [
        {"id": r[0], "offset_sec": r[1], "kind": r[2], "text": r[3]} for r in rows
    ]
