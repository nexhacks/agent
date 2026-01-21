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
    # Screenshots table for police reports
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS video_screenshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            offset_sec REAL NOT NULL,
            trigger_type TEXT NOT NULL,
            image_data TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_video_screenshots ON video_screenshots(video_id, offset_sec)"
    )
    # Reports table for storing generated police reports
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            report_data TEXT NOT NULL,
            format TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_reports_video ON reports(video_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at DESC)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS twelvelabs_videos (
            video_id TEXT PRIMARY KEY,
            tl_index_id TEXT,
            tl_task_id TEXT,
            tl_video_id TEXT,
            content_hash TEXT,
            status TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # Add content_hash column if upgrading existing DB
    columns = {row[1] for row in conn.execute("PRAGMA table_info(twelvelabs_videos)").fetchall()}
    if "content_hash" not in columns:
        conn.execute("ALTER TABLE twelvelabs_videos ADD COLUMN content_hash TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_twelvelabs_status ON twelvelabs_videos(status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_twelvelabs_hash ON twelvelabs_videos(content_hash)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS report_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT,
            video_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_report_notes_video ON report_notes(video_id, created_at DESC)"
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


def get_all_events(conn: sqlite3.Connection, video_id: str) -> list[dict]:
    """Get all events for a video (not paginated), ordered by offset time."""
    rows = conn.execute(
        "SELECT id, offset_sec, kind, text FROM video_events WHERE video_id = ? ORDER BY offset_sec",
        (video_id,),
    ).fetchall()
    return [
        {"id": r[0], "offset_sec": r[1], "kind": r[2], "text": r[3]} for r in rows
    ]


def get_video_metadata(conn: sqlite3.Connection, video_id: str) -> dict | None:
    """Get video record by ID."""
    row = conn.execute(
        "SELECT id, filename, created_at, status, duration_sec FROM videos WHERE id = ?",
        (video_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "filename": row[1],
        "created_at": row[2],
        "status": row[3],
        "duration_sec": row[4],
    }


def insert_screenshot(
    conn: sqlite3.Connection,
    *,
    video_id: str,
    offset_sec: float,
    trigger_type: str,
    image_data: str,
) -> int:
    """Insert a screenshot and return its ID."""
    cursor = conn.execute(
        "INSERT INTO video_screenshots (video_id, offset_sec, trigger_type, image_data, created_at) VALUES (?, ?, ?, ?, ?)",
        (video_id, float(offset_sec), trigger_type, image_data, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    return cursor.lastrowid


def get_screenshots_for_video(conn: sqlite3.Connection, video_id: str) -> list[dict]:
    """Get all screenshots for a video, ordered by offset time."""
    rows = conn.execute(
        "SELECT id, offset_sec, trigger_type, image_data, created_at FROM video_screenshots WHERE video_id = ? ORDER BY offset_sec",
        (video_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "offset_sec": r[1],
            "trigger_type": r[2],
            "image_data": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


def insert_report(
    conn: sqlite3.Connection,
    *,
    report_id: str,
    video_id: str,
    report_data: str,
    format_type: str,
) -> None:
    """Insert or update a generated report."""
    conn.execute(
        "INSERT OR REPLACE INTO reports (id, video_id, report_data, format, created_at) VALUES (?, ?, ?, ?, ?)",
        (report_id, video_id, report_data, format_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()


def get_cached_report(
    conn: sqlite3.Connection,
    video_id: str,
    report_type: str = "standard",
) -> dict | None:
    """Get a cached report for a video by type (standard, traffic_stop, ois, general).

    Returns the most recent report matching the criteria.
    """
    # Map report types to format patterns
    if report_type == "standard":
        # Standard reports have IDs starting with RPT-
        pattern = "RPT-%"
    elif report_type in ("traffic_stop", "ois", "officer_involved", "general", "auto"):
        # Formal reports have specific ID patterns based on their type
        if report_type == "traffic_stop":
            pattern = "SBI122-%"
        elif report_type in ("ois", "officer_involved"):
            pattern = "OIS-%"
        else:
            # For general or auto, look for any formal report
            pattern = None
    else:
        pattern = None

    if pattern:
        row = conn.execute(
            """
            SELECT r.id, r.video_id, r.report_data, r.format, r.created_at, v.filename, v.duration_sec
            FROM reports r
            LEFT JOIN videos v ON r.video_id = v.id
            WHERE r.video_id = ? AND r.id LIKE ?
            ORDER BY r.created_at DESC
            LIMIT 1
            """,
            (video_id, pattern),
        ).fetchone()
    else:
        # Get any formal report (not starting with RPT-)
        row = conn.execute(
            """
            SELECT r.id, r.video_id, r.report_data, r.format, r.created_at, v.filename, v.duration_sec
            FROM reports r
            LEFT JOIN videos v ON r.video_id = v.id
            WHERE r.video_id = ? AND r.id NOT LIKE 'RPT-%'
            ORDER BY r.created_at DESC
            LIMIT 1
            """,
            (video_id,),
        ).fetchone()

    if not row:
        return None

    return {
        "id": row[0],
        "video_id": row[1],
        "report_data": row[2],
        "format": row[3],
        "created_at": row[4],
        "filename": row[5],
        "duration_sec": row[6],
    }


def list_reports(conn: sqlite3.Connection) -> list[dict]:
    """List all reports with video metadata, ordered by creation date."""
    rows = conn.execute(
        """
        SELECT r.id, r.video_id, r.format, r.created_at, v.filename, v.duration_sec
        FROM reports r
        LEFT JOIN videos v ON r.video_id = v.id
        ORDER BY r.created_at DESC
        """
    ).fetchall()
    return [
        {
            "id": r[0],
            "video_id": r[1],
            "format": r[2],
            "created_at": r[3],
            "filename": r[4],
            "duration_sec": r[5],
        }
        for r in rows
    ]


def get_report(conn: sqlite3.Connection, report_id: str) -> dict | None:
    """Get a report by ID."""
    row = conn.execute(
        """
        SELECT r.id, r.video_id, r.report_data, r.format, r.created_at, v.filename, v.duration_sec
        FROM reports r
        LEFT JOIN videos v ON r.video_id = v.id
        WHERE r.id = ?
        """,
        (report_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "video_id": row[1],
        "report_data": row[2],
        "format": row[3],
        "created_at": row[4],
        "filename": row[5],
        "duration_sec": row[6],
    }


def get_reports_for_video(conn: sqlite3.Connection, video_id: str) -> list[dict]:
    """Get all reports for a specific video."""
    rows = conn.execute(
        "SELECT id, format, created_at FROM reports WHERE video_id = ? ORDER BY created_at DESC",
        (video_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "format": r[1],
            "created_at": r[2],
        }
        for r in rows
    ]


def get_video_thumbnail(conn: sqlite3.Connection, video_id: str) -> str | None:
    """Get the first screenshot (thumbnail) for a video."""
    row = conn.execute(
        "SELECT image_data FROM video_screenshots WHERE video_id = ? ORDER BY offset_sec LIMIT 1",
        (video_id,),
    ).fetchone()
    return row[0] if row else None


def get_report_json_data(conn: sqlite3.Connection, report_id: str) -> dict | None:
    """Get report data parsed as JSON (only for JSON format reports)."""
    row = conn.execute(
        """
        SELECT r.id, r.video_id, r.report_data, r.format, r.created_at, v.filename, v.duration_sec
        FROM reports r
        LEFT JOIN videos v ON r.video_id = v.id
        WHERE r.id = ?
        """,
        (report_id,),
    ).fetchone()
    if not row:
        return None

    import json
    report_data = row[2]

    # If stored as JSON, parse it
    if row[3] == "json":
        try:
            return json.loads(report_data)
        except json.JSONDecodeError:
            return None

    # For HTML reports, we can't return structured data
    return None


def upsert_twelvelabs_video(
    conn: sqlite3.Connection,
    *,
    video_id: str,
    tl_index_id: str | None,
    tl_task_id: str | None,
    tl_video_id: str | None,
    content_hash: str | None,
    status: str,
    error: str | None = None,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO twelvelabs_videos (video_id, tl_index_id, tl_task_id, tl_video_id, content_hash, status, error, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            tl_index_id=excluded.tl_index_id,
            tl_task_id=excluded.tl_task_id,
            tl_video_id=excluded.tl_video_id,
            content_hash=COALESCE(excluded.content_hash, twelvelabs_videos.content_hash),
            status=excluded.status,
            error=excluded.error,
            updated_at=excluded.updated_at
        """,
        (video_id, tl_index_id, tl_task_id, tl_video_id, content_hash, status, error, now, now),
    )
    conn.commit()


def get_twelvelabs_video(conn: sqlite3.Connection, video_id: str) -> dict | None:
    row = conn.execute(
        """
        SELECT video_id, tl_index_id, tl_task_id, tl_video_id, content_hash, status, error, created_at, updated_at
        FROM twelvelabs_videos
        WHERE video_id = ?
        """,
        (video_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "video_id": row[0],
        "tl_index_id": row[1],
        "tl_task_id": row[2],
        "tl_video_id": row[3],
        "content_hash": row[4],
        "status": row[5],
        "error": row[6],
        "created_at": row[7],
        "updated_at": row[8],
    }


def get_twelvelabs_video_by_hash(conn: sqlite3.Connection, content_hash: str) -> dict | None:
    row = conn.execute(
        """
        SELECT video_id, tl_index_id, tl_task_id, tl_video_id, content_hash, status, error, created_at, updated_at
        FROM twelvelabs_videos
        WHERE content_hash = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (content_hash,),
    ).fetchone()
    if not row:
        return None
    return {
        "video_id": row[0],
        "tl_index_id": row[1],
        "tl_task_id": row[2],
        "tl_video_id": row[3],
        "content_hash": row[4],
        "status": row[5],
        "error": row[6],
        "created_at": row[7],
        "updated_at": row[8],
    }


def update_twelvelabs_status(
    conn: sqlite3.Connection,
    *,
    video_id: str,
    status: str,
    tl_video_id: str | None = None,
    tl_task_id: str | None = None,
    error: str | None = None,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        UPDATE twelvelabs_videos
        SET status = ?, tl_video_id = COALESCE(?, tl_video_id),
            tl_task_id = COALESCE(?, tl_task_id),
            error = ?, updated_at = ?
        WHERE video_id = ?
        """,
        (status, tl_video_id, tl_task_id, error, now, video_id),
    )
    conn.commit()


def insert_report_note(
    conn: sqlite3.Connection,
    *,
    report_id: str | None,
    video_id: str,
    question: str,
    answer: str,
    source: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO report_notes (report_id, video_id, question, answer, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (report_id, video_id, question, answer, source, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    return cursor.lastrowid


def list_report_notes(conn: sqlite3.Connection, *, video_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, report_id, question, answer, source, created_at
        FROM report_notes
        WHERE video_id = ?
        ORDER BY created_at DESC
        """,
        (video_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "report_id": r[1],
            "question": r[2],
            "answer": r[3],
            "source": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]
