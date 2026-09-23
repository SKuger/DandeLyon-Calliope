"""What happens to a recording: transcribe, measure, assess, schedule.

The order is not arbitrary. The session is stored as soon as it has been
transcribed and measured, *before* assessment runs, because the metrics
and the audio are the part that has to survive a bad day: an API outage,
a model that times out, a rule that crashes. Feedback is the thing worth
losing.

The whole path is idempotent, which matters more here than it looks. The
browser retries uploads, the button gets double-clicked, and a recording
re-imported from disk is the same recording. A duplicate session would
put a second point on every chart and tell him he makes a mistake twice
as often as he does -- which would then send him to practise it.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.asr.base import Transcriber, Transcript
from app.assess.base import Assessor, Finding
from app.history import Recurrence, build_index, recurrence
from app.metrics.fluency import FluencyMetrics, measure
from app.review.scheduler import Scheduler
from app.store import (
    Session,
    StoredError,
    add_errors,
    errors_for,
    get_session,
    get_transcript,
    save_review,
    save_session,
    session_id_for,
)

logger = logging.getLogger(__name__)

#: How many earlier utterances are handed to the assessor as context.
#: Three, because the model is being asked to recognise a pattern, not to
#: re-read his month.
CONTEXT_UTTERANCES = 3


@dataclass(frozen=True)
class SessionResult:
    session_id: str
    transcript: Transcript
    metrics: FluencyMetrics
    findings: tuple[Finding, ...]
    recurrences: tuple[Recurrence, ...]
    already_processed: bool = False
    """True when this audio had been ingested before. The caller gets the
    stored result rather than a second copy of it."""


class MergingAssessor:
    """Runs several assessors and keeps the union of what they find.

    The offline rules go first and the model is only credited with what
    they missed, so a model that rediscovers "responsible of" does not
    create a second row in the corpus for the same mistake. That would
    make the repetition count -- the number the practice schedule is
    built on -- depend on how many assessors happened to be configured.
    """

    name = "merged"

    def __init__(self, assessors: Sequence[Assessor]) -> None:
        self._assessors = tuple(assessors)

    async def assess(
        self, transcript: Transcript, context: Sequence[str] = ()
    ) -> list[Finding]:
        found: list[Finding] = []
        seen: set[tuple[str, str]] = set()
        for assessor in self._assessors:
            for finding in await assessor.assess(transcript, context):
                key = (finding.category, finding.original.strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                found.append(finding)
        return found


async def process(
    connection: sqlite3.Connection,
    *,
    audio: Path,
    transcriber: Transcriber,
    assessor: Assessor,
    scheduler: Scheduler,
    kind: str = "practice",
    prompt_id: str | None = None,
    at: datetime | None = None,
) -> SessionResult:
    """One recording, from file to scheduled practice."""
    moment = at or datetime.now(UTC)
    digest = session_id_for(audio.read_bytes())

    stored = get_session(connection, digest)
    if stored is not None:
        return _replay(connection, stored)

    transcript = await transcriber.transcribe(audio)
    metrics = measure(transcript)

    fresh = save_session(
        connection,
        session=Session(
            id=digest,
            kind=kind,
            recorded_at=moment.replace(microsecond=0).isoformat(),
            duration=transcript.duration,
            prompt_id=prompt_id,
            audio_path=str(audio),
        ),
        audio_sha256=digest,
        transcript=transcript,
        metrics=metrics.as_dict(),
    )
    if not fresh:
        # Two uploads of the same file racing each other. The other one
        # won; read back what it wrote rather than writing it again.
        return _replay(connection, get_session(connection, digest))

    findings = await assessor.assess(
        transcript, context=_context(connection, transcript, digest)
    )

    errors = _persist(connection, digest, findings, moment)
    for error in errors:
        save_review(connection, scheduler.first(error.id, moment))

    return SessionResult(
        session_id=digest,
        transcript=transcript,
        metrics=metrics,
        findings=tuple(findings),
        recurrences=tuple(
            recurrence(connection, f.category, f.original, at=moment)
            for f in findings
        ),
    )


def _context(
    connection: sqlite3.Connection, transcript: Transcript, session_id: str
) -> list[str]:
    """A few things he has said before that resemble this session.

    Best effort: retrieval failing is not a reason to lose an assessment,
    so this returns nothing rather than raising.
    """
    try:
        index = build_index(connection)
        hits = index.search(
            transcript.text,
            limit=CONTEXT_UTTERANCES,
            exclude_session=session_id,
        )
        return [utterance.text for utterance, _ in hits]
    except Exception as exc:  # noqa: BLE001 - context is optional
        logger.warning("could not build history context: %s", exc)
        return []


def _persist(
    connection: sqlite3.Connection,
    session_id: str,
    findings: Sequence[Finding],
    moment: datetime,
) -> list[StoredError]:
    from app.store import error_id_for

    stamp = moment.replace(microsecond=0).isoformat()
    errors = [
        StoredError(
            id=error_id_for(session_id, finding.category, finding.original),
            session_id=session_id,
            category=finding.category,
            original=finding.original,
            correction=finding.correction,
            explanation=finding.explanation,
            detector=finding.detector,
            created_at=stamp,
        )
        for finding in findings
    ]
    add_errors(connection, errors)
    # Read back rather than trusting the list: a re-assessment of an
    # existing session inserts nothing, and the review schedule must be
    # written against what is actually in the corpus.
    return errors_for(connection, session_id)


def _replay(connection: sqlite3.Connection, session: Session) -> SessionResult:
    transcript = get_transcript(connection, session.id)
    stored = errors_for(connection, session.id)

    # Recomputed from the stored transcript rather than read back from
    # the metrics table. The stored numbers are the historical record and
    # keep their own definitions; this is a view, and a view should show
    # what today's code makes of the same words.

    return SessionResult(
        session_id=session.id,
        transcript=transcript or Transcript(text="", duration=session.duration),
        metrics=measure(transcript) if transcript else measure(
            Transcript(text="", duration=session.duration)
        ),
        findings=tuple(
            Finding(
                category=error.category,
                original=error.original,
                correction=error.correction,
                explanation=error.explanation,
                detector=error.detector,
            )
            for error in stored
        ),
        recurrences=(),
        already_processed=True,
    )


def store_recording(directory: Path, name: str, payload: bytes) -> Path:
    """Write an upload to disk under a content-addressed name.

    Named by hash so that saving the same recording twice writes the same
    file, and so that the filename carries no date, no prompt and nothing
    about what he said. The recordings never leave the machine; the
    filenames should not describe them either.
    """
    directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(name).suffix or ".webm"
    path = directory / f"{session_id_for(payload)}{suffix}"
    if not path.exists():
        path.write_bytes(payload)
    return path
