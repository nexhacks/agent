import os
import sqlite3
import threading
import time
from datetime import datetime


DEFAULT_DB_PATH = os.getenv("EVENT_LOG_PATH", "events.db")


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ts_unix REAL NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            source TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_kind_ts ON events(kind, ts_unix)"
    )
    conn.commit()
    return conn


def get_event_log_path() -> str:
    return os.getenv("EVENT_LOG_PATH", DEFAULT_DB_PATH)


def get_event_logger() -> "EventLogger | None":
    if os.getenv("EVENT_LOG_ENABLED", "1") != "1":
        return None
    return EventLogger(get_event_log_path())


def open_event_db(path: str | None = None) -> sqlite3.Connection:
    return _connect(path or get_event_log_path())


class EventLogger:
    def __init__(self, path: str) -> None:
        self._path = path
        self._conn = _connect(path)
        self._lock = threading.Lock()

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
        if isinstance(ts, datetime):
            ts_unix = ts.timestamp()
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(ts, (int, float)):
            ts_unix = float(ts)
            ts_str = datetime.fromtimestamp(ts_unix).strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(ts, str):
            ts_unix = time.time()
            ts_str = ts
        else:
            now = datetime.now()
            ts_unix = now.timestamp()
            ts_str = now.strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            self._conn.execute(
                "INSERT INTO events (ts, ts_unix, kind, text, source) VALUES (?, ?, ?, ?, ?)",
                (ts_str, ts_unix, kind, text, source),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
