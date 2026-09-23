"""SQLite: sessions, metrics, the error corpus and its review schedule.

One user, one file, on his own machine. Postgres here would be theatre --
a daemon, a connection pool and a migration tool to serve queries that
take microseconds against a few thousand rows. The cost of being wrong
about that is a `pg_dump` away, and the cost of being wrong the other way
is a project that needs `docker compose` before it can record a sentence.

Two properties the rest of the system leans on:

**Ingesting the same recording twice changes nothing.** The session id is
the hash of the audio, so a retried upload, a double-clicked button or a
re-run of the importer lands on the same row. Without it, a duplicate
session inflates every metric on the dashboard and the error corpus
starts claiming he makes the same mistake twice as often as he does.

**Nothing is deleted on the way in.** Metrics are stored beside the
transcript that produced them, with the model and decoding that produced
*that*. When the transcriber changes -- and it will, that is the point of
the Protocol -- the old numbers stay readable and stay labelled with the
instrument that measured them.

The API is synchronous. SQLite on a local file answers in microseconds,
and wrapping it in a thread pool would add more machinery than it saves.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.asr.base import Transcript, dump_transcript, transcript_from_payload
from app.review.scheduler import ReviewState

SCHEMA_VERSION = 1

#: Numbered and applied in order, the way a schema that will outlive its
#: first design has to be. `user_version` is SQLite's own counter, so the
#: version travels inside the database file rather than beside it.
MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE session (
        id             TEXT PRIMARY KEY,
        kind           TEXT NOT NULL,
        prompt_id      TEXT,
        recorded_at    TEXT NOT NULL,
        audio_sha256   TEXT NOT NULL UNIQUE,
        audio_path     TEXT,
        duration       REAL NOT NULL DEFAULT 0
    );

    CREATE TABLE transcript (
        session_id  TEXT PRIMARY KEY REFERENCES session(id) ON DELETE CASCADE,
        text        TEXT NOT NULL,
        model       TEXT NOT NULL,
        decoding    TEXT NOT NULL,
        payload     TEXT NOT NULL
    );

    CREATE TABLE metric (
        session_id  TEXT NOT NULL REFERENCES session(id) ON DELETE CASCADE,
        name        TEXT NOT NULL,
        value       REAL,
        PRIMARY KEY (session_id, name)
    );

    CREATE TABLE session_detail (
        session_id  TEXT PRIMARY KEY REFERENCES session(id) ON DELETE CASCADE,
        payload     TEXT NOT NULL
    );

    CREATE TABLE error (
        id           TEXT PRIMARY KEY,
        session_id   TEXT NOT NULL REFERENCES session(id) ON DELETE CASCADE,
        category     TEXT NOT NULL,
        original     TEXT NOT NULL,
        correction   TEXT NOT NULL,
        explanation  TEXT NOT NULL DEFAULT '',
        detector     TEXT NOT NULL,
        created_at   TEXT NOT NULL
    );

    CREATE INDEX error_by_category ON error(category);

    CREATE TABLE review (
        error_id         TEXT PRIMARY KEY REFERENCES error(id) ON DELETE CASCADE,
        due_at           TEXT NOT NULL,
        interval_days    REAL NOT NULL,
        ease             REAL NOT NULL,
        repetitions      INTEGER NOT NULL DEFAULT 0,
        lapses           INTEGER NOT NULL DEFAULT 0,
        last_reviewed_at TEXT
    );

    CREATE INDEX review_by_due ON review(due_at);

    CREATE TABLE placement (
        id           TEXT PRIMARY KEY,
        exam_version TEXT NOT NULL,
        taken_at     TEXT NOT NULL,
        payload      TEXT NOT NULL
    );
    """,
)


def now() -> str:
    """UTC, ISO 8601, second precision. Sorts as a string, which is the
    only property a timestamp column in SQLite needs."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def session_id_for(audio: bytes) -> str:
    """The recording's own hash.

    Content-addressed on purpose: a session that is re-uploaded, retried
    or re-imported is the same session, and the database says so without
    anyone having to remember to check.
    """
    return hashlib.sha256(audio).hexdigest()[:32]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # `check_same_thread=False` because the web layer answers some
    # requests from a thread pool. There is one user and every write is a
    # single statement, so SQLite's own locking is the whole concurrency
    # story; a connection pool would be machinery for a contention that
    # cannot happen.
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    migrate(connection)
    return connection


def migrate(connection: sqlite3.Connection) -> int:
    applied = connection.execute("PRAGMA user_version").fetchone()[0]
    for version, statements in enumerate(MIGRATIONS[applied:], start=applied + 1):
        connection.executescript(statements)
        connection.execute(f"PRAGMA user_version = {version}")
    connection.commit()
    return len(MIGRATIONS) - applied


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    connection.commit()


@dataclass(frozen=True)
class Session:
    id: str
    kind: str
    """`practice`, `placement` or `conversation`. The dashboard charts
    practice sessions and the placement exam on the same axes but never
    in the same series: they are different tasks and mixing them would
    make the baseline meaningless."""

    recorded_at: str
    duration: float
    prompt_id: str | None = None
    audio_path: str | None = None


@dataclass(frozen=True)
class StoredError:
    id: str
    session_id: str
    category: str
    original: str
    correction: str
    explanation: str
    detector: str
    created_at: str


def error_id_for(session_id: str, category: str, original: str) -> str:
    """Stable within a session, distinct across sessions.

    The same mistake made on Tuesday and again on Friday is two rows --
    that repetition is the signal the whole corpus exists to capture. The
    same mistake found twice while re-processing Tuesday is one row.
    """
    raw = f"{session_id}|{category}|{original.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def save_session(
    connection: sqlite3.Connection,
    *,
    session: Session,
    audio_sha256: str,
    transcript: Transcript,
    metrics: dict[str, Any],
) -> bool:
    """Store a session. Returns False if it was already there.

    The whole call is one transaction and the first insert is the guard:
    if the session exists, nothing else runs, so a retried upload cannot
    half-write a second copy of the metrics.
    """
    scalars = {
        name: value
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }

    with transaction(connection):
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO session
                (id, kind, prompt_id, recorded_at, audio_sha256, audio_path,
                 duration)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.kind,
                session.prompt_id,
                session.recorded_at,
                audio_sha256,
                session.audio_path,
                session.duration,
            ),
        )
        if cursor.rowcount == 0:
            return False

        connection.execute(
            """
            INSERT INTO transcript (session_id, text, model, decoding, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session.id,
                transcript.text,
                transcript.model,
                transcript.decoding,
                json.dumps(dump_transcript(transcript)),
            ),
        )
        connection.execute(
            "INSERT INTO session_detail (session_id, payload) VALUES (?, ?)",
            (session.id, json.dumps(metrics, default=list)),
        )
        connection.executemany(
            "INSERT INTO metric (session_id, name, value) VALUES (?, ?, ?)",
            [(session.id, name, float(value)) for name, value in scalars.items()],
        )
    return True


def get_session(connection: sqlite3.Connection, session_id: str) -> Session | None:
    row = connection.execute(
        "SELECT * FROM session WHERE id = ?", (session_id,)
    ).fetchone()
    return None if row is None else _as_session(row)


def _as_session(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        kind=row["kind"],
        recorded_at=row["recorded_at"],
        duration=row["duration"],
        prompt_id=row["prompt_id"],
        audio_path=row["audio_path"],
    )


def sessions(
    connection: sqlite3.Connection, kind: str | None = None, limit: int = 100
) -> list[Session]:
    if kind is None:
        rows = connection.execute(
            "SELECT * FROM session ORDER BY recorded_at DESC LIMIT ?", (limit,)
        )
    else:
        rows = connection.execute(
            "SELECT * FROM session WHERE kind = ? "
            "ORDER BY recorded_at DESC LIMIT ?",
            (kind, limit),
        )
    return [_as_session(row) for row in rows]


def get_transcript(
    connection: sqlite3.Connection, session_id: str
) -> Transcript | None:
    row = connection.execute(
        "SELECT payload FROM transcript WHERE session_id = ?", (session_id,)
    ).fetchone()
    # Read back through the same function that reads the fixtures, so the
    # stored shape cannot drift into meaning two different things.
    return (
        None if row is None else transcript_from_payload(json.loads(row["payload"]))
    )


def session_detail(
    connection: sqlite3.Connection, session_id: str
) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT payload FROM session_detail WHERE session_id = ?", (session_id,)
    ).fetchone()
    return None if row is None else json.loads(row["payload"])


def metric_series(
    connection: sqlite3.Connection, name: str, kind: str = "practice"
) -> list[tuple[str, float]]:
    """One metric over time, oldest first. The shape a chart wants."""
    rows = connection.execute(
        """
        SELECT session.recorded_at AS at, metric.value AS value
        FROM metric
        JOIN session ON session.id = metric.session_id
        WHERE metric.name = ? AND session.kind = ? AND metric.value IS NOT NULL
        ORDER BY session.recorded_at
        """,
        (name, kind),
    )
    return [(row["at"], row["value"]) for row in rows]


def add_errors(
    connection: sqlite3.Connection, errors: Iterable[StoredError]
) -> int:
    """Append to the corpus, skipping anything already recorded.

    `INSERT OR IGNORE` on a deterministic id is what makes re-assessing a
    session safe. Re-running the assessor with a better prompt should
    improve the corpus, not duplicate it.
    """
    rows = [
        (
            error.id,
            error.session_id,
            error.category,
            error.original,
            error.correction,
            error.explanation,
            error.detector,
            error.created_at,
        )
        for error in errors
    ]
    with transaction(connection):
        cursor = connection.executemany(
            """
            INSERT OR IGNORE INTO error
                (id, session_id, category, original, correction, explanation,
                 detector, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


def errors_for(
    connection: sqlite3.Connection, session_id: str
) -> list[StoredError]:
    rows = connection.execute(
        "SELECT * FROM error WHERE session_id = ? ORDER BY created_at",
        (session_id,),
    )
    return [_as_error(row) for row in rows]


def _as_error(row: sqlite3.Row) -> StoredError:
    return StoredError(
        id=row["id"],
        session_id=row["session_id"],
        category=row["category"],
        original=row["original"],
        correction=row["correction"],
        explanation=row["explanation"],
        detector=row["detector"],
        created_at=row["created_at"],
    )


def all_errors(
    connection: sqlite3.Connection, category: str | None = None, limit: int = 500
) -> list[StoredError]:
    if category is None:
        rows = connection.execute(
            "SELECT * FROM error ORDER BY created_at DESC LIMIT ?", (limit,)
        )
    else:
        rows = connection.execute(
            "SELECT * FROM error WHERE category = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (category, limit),
        )
    return [_as_error(row) for row in rows]


def category_counts(connection: sqlite3.Connection) -> list[tuple[str, int]]:
    """How often each kind of mistake happens. The answer to "what should
    I practise", which is the question no generic curriculum can ask."""
    rows = connection.execute(
        "SELECT category, COUNT(*) AS n FROM error "
        "GROUP BY category ORDER BY n DESC"
    )
    return [(row["category"], row["n"]) for row in rows]


def save_placement(
    connection: sqlite3.Connection,
    *,
    placement_id: str,
    exam_version: str,
    taken_at: str,
    payload: dict[str, Any],
) -> bool:
    with transaction(connection):
        cursor = connection.execute(
            "INSERT OR IGNORE INTO placement "
            "(id, exam_version, taken_at, payload) VALUES (?, ?, ?, ?)",
            (placement_id, exam_version, taken_at, json.dumps(payload)),
        )
        return cursor.rowcount == 1


def placements(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every sitting, oldest first. Comparing two of them is the entire
    point of fixing the exam rather than generating it."""
    rows = connection.execute(
        "SELECT * FROM placement ORDER BY taken_at"
    )
    return [
        {
            "id": row["id"],
            "exam_version": row["exam_version"],
            "taken_at": row["taken_at"],
            **json.loads(row["payload"]),
        }
        for row in rows
    ]


# --- review schedule -----------------------------------------------------
#
# The schedule lives beside the corpus rather than in its own store: every
# query the review queue needs is a join between an error and its next
# date, and splitting them would mean keeping two files consistent for no
# gain at this size.


def save_review(connection: sqlite3.Connection, state: ReviewState) -> None:
    """Write the next date for one error. Last write wins.

    An upsert rather than an insert because a review is an update to a
    single card, and a schedule that accumulated rows would eventually
    disagree with itself about when a card is due.
    """
    with transaction(connection):
        connection.execute(
            """
            INSERT INTO review
                (error_id, due_at, interval_days, ease, repetitions, lapses,
                 last_reviewed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(error_id) DO UPDATE SET
                due_at = excluded.due_at,
                interval_days = excluded.interval_days,
                ease = excluded.ease,
                repetitions = excluded.repetitions,
                lapses = excluded.lapses,
                last_reviewed_at = excluded.last_reviewed_at
            """,
            (
                state.error_id,
                state.due_at,
                state.interval_days,
                state.ease,
                state.repetitions,
                state.lapses,
                state.last_reviewed_at,
            ),
        )


def get_review(
    connection: sqlite3.Connection, error_id: str
) -> ReviewState | None:
    row = connection.execute(
        "SELECT * FROM review WHERE error_id = ?", (error_id,)
    ).fetchone()
    return None if row is None else _as_review(row)


def _as_review(row: sqlite3.Row) -> ReviewState:
    return ReviewState(
        error_id=row["error_id"],
        due_at=row["due_at"],
        interval_days=row["interval_days"],
        ease=row["ease"],
        repetitions=row["repetitions"],
        lapses=row["lapses"],
        last_reviewed_at=row["last_reviewed_at"],
    )


def due_reviews(
    connection: sqlite3.Connection, at: str, limit: int = 20
) -> list[tuple[StoredError, ReviewState]]:
    """The queue, oldest due first.

    Ordered by due date and not by category, so a bad week with
    prepositions cannot crowd everything else out of the queue for a
    fortnight.
    """
    rows = connection.execute(
        """
        SELECT error.*, review.due_at AS r_due, review.interval_days AS r_interval,
               review.ease AS r_ease, review.repetitions AS r_reps,
               review.lapses AS r_lapses,
               review.last_reviewed_at AS r_last
        FROM review
        JOIN error ON error.id = review.error_id
        WHERE review.due_at <= ?
        ORDER BY review.due_at
        LIMIT ?
        """,
        (at, limit),
    )
    return [
        (
            _as_error(row),
            ReviewState(
                error_id=row["id"],
                due_at=row["r_due"],
                interval_days=row["r_interval"],
                ease=row["r_ease"],
                repetitions=row["r_reps"],
                lapses=row["r_lapses"],
                last_reviewed_at=row["r_last"],
            ),
        )
        for row in rows
    ]


def mastery(connection: sqlite3.Connection) -> dict[str, int]:
    """The curve the dashboard draws: how much of the corpus has stuck.

    `mature` is an interval past three weeks, which is the point at which
    a card stops being something he is currently learning. `unscheduled`
    exists because an error with no review row is a bug in the pipeline,
    and a number nobody can see is a bug nobody fixes.
    """
    row = connection.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM error) AS total,
          (SELECT COUNT(*) FROM review WHERE repetitions = 0) AS learning,
          (SELECT COUNT(*) FROM review
             WHERE repetitions > 0 AND interval_days < 21) AS young,
          (SELECT COUNT(*) FROM review WHERE interval_days >= 21) AS mature,
          (SELECT COUNT(*) FROM review WHERE lapses > 0) AS lapsed
        """
    ).fetchone()
    counted = row["learning"] + row["young"] + row["mature"]
    return {
        "total": row["total"],
        "learning": row["learning"],
        "young": row["young"],
        "mature": row["mature"],
        "lapsed": row["lapsed"],
        "unscheduled": row["total"] - counted,
    }


def get_error(
    connection: sqlite3.Connection, error_id: str
) -> StoredError | None:
    row = connection.execute(
        "SELECT * FROM error WHERE id = ?", (error_id,)
    ).fetchone()
    return None if row is None else _as_error(row)
