"""Storage, tested at the point where the same thing arrives twice.

A duplicated session is not a cosmetic bug here. Every chart in the
project is a series over sessions, and the error corpus counts
repetitions to decide what to practise -- so a session stored twice tells
the user he makes a mistake twice as often as he does, and the system
then spends his time on it.
"""

import sqlite3

import pytest

from app.asr.base import Transcript, Word
from app.metrics.fluency import measure
from app.store import (
    Session,
    StoredError,
    add_errors,
    all_errors,
    category_counts,
    connect,
    error_id_for,
    errors_for,
    get_session,
    get_transcript,
    metric_series,
    migrate,
    now,
    placements,
    save_placement,
    save_session,
    session_detail,
    session_id_for,
    sessions,
)

AUDIO = b"RIFF....fake wav bytes"


@pytest.fixture
def db(tmp_path) -> sqlite3.Connection:
    connection = connect(tmp_path / "calliope.db")
    yield connection
    connection.close()


def transcript_of(text: str = "yesterday I go to the office") -> Transcript:
    words = tuple(
        Word(text=token, start=i * 0.4, end=i * 0.4 + 0.3, confidence=0.9)
        for i, token in enumerate(text.split())
    )
    return Transcript(
        text=text,
        words=words,
        duration=len(words) * 0.4,
        model="faster-whisper:small.en",
        decoding="faithful",
    )


def store(db, audio: bytes = AUDIO, kind: str = "practice", **overrides) -> bool:
    transcript = overrides.pop("transcript", transcript_of())
    session = Session(
        id=session_id_for(audio),
        kind=kind,
        recorded_at=overrides.pop("recorded_at", now()),
        duration=transcript.duration,
        prompt_id=overrides.pop("prompt_id", "describe-your-week"),
    )
    return save_session(
        db,
        session=session,
        audio_sha256=session_id_for(audio),
        transcript=transcript,
        metrics=overrides.pop("metrics", measure(transcript).as_dict()),
    )


# --- migrations ----------------------------------------------------------


def test_opening_an_existing_database_applies_nothing(tmp_path):
    path = tmp_path / "calliope.db"
    connect(path).close()

    second = connect(path)

    # Reopening has to be free. A migration that reruns is a migration
    # that eventually drops something.
    assert migrate(second) == 0


# --- idempotent ingest ---------------------------------------------------


def test_the_same_recording_is_stored_once(db):
    assert store(db) is True
    assert store(db) is False

    assert len(sessions(db)) == 1


def test_a_duplicate_upload_does_not_duplicate_the_metrics(db):
    store(db)
    store(db)

    series = metric_series(db, "words_per_minute")

    # This is the failure the content-addressed id exists to prevent: the
    # chart would show two points for one recording and the average would
    # weight that day twice.
    assert len(series) == 1


def test_a_different_recording_is_a_different_session(db):
    store(db, audio=b"one")
    store(db, audio=b"two")

    assert len(sessions(db)) == 2


def test_a_session_survives_a_round_trip(db):
    store(db)
    stored = get_session(db, session_id_for(AUDIO))

    assert stored.kind == "practice"
    assert stored.prompt_id == "describe-your-week"


def test_the_transcript_comes_back_with_its_instrument(db):
    store(db)

    transcript = get_transcript(db, session_id_for(AUDIO))

    # Which model and which decoding produced it. Without those, a number
    # from March cannot honestly be compared with one from September.
    assert transcript.model == "faster-whisper:small.en"
    assert transcript.decoding == "faithful"
    assert transcript.words[0].confidence == 0.9


# --- what becomes a metric ----------------------------------------------


def test_unmeasurable_metrics_are_not_charted_but_are_kept(db):
    silence = Transcript(text="", duration=0.0)
    store(db, transcript=silence, metrics=measure(silence).as_dict())

    assert metric_series(db, "words_per_minute") == []
    # Still readable in full, so a gap in a chart can be explained rather
    # than guessed at.
    assert session_detail(db, session_id_for(AUDIO))["words_per_minute"] is None


def test_a_series_is_ordered_oldest_first(db):
    store(db, audio=b"b", recorded_at="2026-02-01T09:00:00+00:00")
    store(db, audio=b"a", recorded_at="2026-01-01T09:00:00+00:00")

    at = [when for when, _ in metric_series(db, "words_per_minute")]

    assert at == sorted(at)


def test_the_placement_exam_is_not_mixed_into_the_practice_series(db):
    store(db, audio=b"practice", kind="practice")
    store(db, audio=b"exam", kind="placement")

    # Same axes on the dashboard, different series. They are different
    # tasks, and averaging them would make the baseline meaningless.
    assert len(metric_series(db, "words_per_minute", kind="practice")) == 1
    assert len(metric_series(db, "words_per_minute", kind="placement")) == 1


# --- the error corpus ----------------------------------------------------


def error(session_id: str, category="verb_tense", original="yesterday I go"):
    return StoredError(
        id=error_id_for(session_id, category, original),
        session_id=session_id,
        category=category,
        original=original,
        correction="yesterday I went",
        explanation="Past time reference takes the past simple.",
        detector="rules",
        created_at=now(),
    )


def test_reassessing_a_session_does_not_duplicate_its_errors(db):
    store(db)
    session_id = session_id_for(AUDIO)

    assert add_errors(db, [error(session_id)]) == 1
    assert add_errors(db, [error(session_id)]) == 0

    # Re-running the assessor with a better prompt should improve the
    # corpus, not inflate it.
    assert len(errors_for(db, session_id)) == 1


def test_the_same_mistake_in_two_sessions_is_two_rows(db):
    store(db, audio=b"monday")
    store(db, audio=b"friday")
    add_errors(db, [error(session_id_for(b"monday"))])
    add_errors(db, [error(session_id_for(b"friday"))])

    # The repetition is the signal. Deduplicating across sessions would
    # throw away the only evidence that an error is worth practising.
    assert len(all_errors(db)) == 2
    assert category_counts(db) == [("verb_tense", 2)]


def test_an_error_cannot_outlive_its_session(db):
    store(db)
    session_id = session_id_for(AUDIO)
    add_errors(db, [error(session_id)])

    db.execute("DELETE FROM session WHERE id = ?", (session_id,))
    db.commit()

    # Foreign keys are on. An orphaned error would keep being scheduled
    # for review with no audio to play back.
    assert all_errors(db) == []


def test_errors_can_be_filtered_by_category(db):
    store(db)
    session_id = session_id_for(AUDIO)
    add_errors(
        db,
        [
            error(session_id, "verb_tense", "yesterday I go"),
            error(session_id, "preposition", "responsible of"),
        ],
    )

    assert len(all_errors(db, category="preposition")) == 1


# --- the placement exam --------------------------------------------------


def test_a_placement_sitting_is_stored_once(db):
    payload = {"bands": {"speaking": "B1"}}

    assert save_placement(
        db, placement_id="v1-2026-01", exam_version="v1",
        taken_at=now(), payload=payload
    ) is True
    assert save_placement(
        db, placement_id="v1-2026-01", exam_version="v1",
        taken_at=now(), payload=payload
    ) is False


def test_sittings_come_back_in_order_for_comparison(db):
    save_placement(db, placement_id="b", exam_version="v1",
                   taken_at="2026-06-01T09:00:00+00:00", payload={"bands": {}})
    save_placement(db, placement_id="a", exam_version="v1",
                   taken_at="2026-01-01T09:00:00+00:00", payload={"bands": {}})

    # Comparing two sittings of the same fixed exam is the only reason to
    # fix the exam in the first place.
    assert [row["id"] for row in placements(db)] == ["a", "b"]
