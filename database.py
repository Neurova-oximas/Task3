"""
database.py
-----------
SQLite3 prediction logger.
Handles all DB operations — app.py only calls log_prediction() and get_history().

Schema
------
predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT,       -- ISO-8601 datetime string
    input_type      TEXT,       -- "text" | "image"
    input_content   TEXT,       -- raw text OR generated image caption
    predicted_class TEXT,       -- top predicted label
    confidence      REAL,       -- top confidence in [0.0, 1.0]
    all_scores      TEXT        -- JSON blob of {class: score} for ALL classes
)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ── DB file lives next to this module ─────────────────────────────────────────
DB_PATH = Path(__file__).parent / "predictions.db"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Open a connection with row_factory so rows behave like dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _create_table_if_needed(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            input_type      TEXT    NOT NULL,
            input_content   TEXT    NOT NULL,
            predicted_class TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            all_scores      TEXT    NOT NULL
        )
    """)
    conn.commit()


# ── Public API ─────────────────────────────────────────────────────────────────

def log_prediction(
    input_type: str,
    input_content: str,
    predicted_class: str,
    confidence: float,
    all_scores: Dict[str, float],
) -> None:
    """
    Insert one prediction row into the database.

    Parameters
    ----------
    input_type      : "text" or "image"
    input_content   : raw user text OR the BLIP-generated caption for images
    predicted_class : top predicted label string
    confidence      : float in [0.0, 1.0]
    all_scores      : dict of {class_name: score} for all classes
    """
    timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
    all_scores_json = json.dumps(all_scores)

    with _get_connection() as conn:
        _create_table_if_needed(conn)
        conn.execute(
            """
            INSERT INTO predictions
                (timestamp, input_type, input_content, predicted_class, confidence, all_scores)
            VALUES
                (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, input_type, input_content, predicted_class, confidence, all_scores_json),
        )
        conn.commit()


def get_history(limit: int = 10) -> List[dict]:
    """
    Fetch the most recent predictions, newest first.

    Parameters
    ----------
    limit : how many rows to return (default 10)

    Returns
    -------
    List of dicts with keys:
        id, timestamp, input_type, input_content,
        predicted_class, confidence, all_scores (already parsed back to dict)
    """
    with _get_connection() as conn:
        _create_table_if_needed(conn)
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    result = []
    for row in rows:
        entry = dict(row)
        entry["all_scores"] = json.loads(entry["all_scores"])
        result.append(entry)
    return result


def get_total_count() -> int:
    """Return total number of predictions logged so far."""
    with _get_connection() as conn:
        _create_table_if_needed(conn)
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]