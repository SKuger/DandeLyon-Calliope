"""The path from a recording to a scheduled exercise.

Two properties are worth more than the rest: the same upload twice
changes nothing, and an assessor that fails costs the feedback but never
the session. Both are tested by breaking things on purpose.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.asr.base import Transcript, TranscriptionError, Word
from app.assess.base import Finding
from app.assess.rules import RuleBasedAssessor
from app.pipeline import MergingAssessor, process, store_recording
from app.review.scheduler import SM2Scheduler
from app.store import all_errors, connect, get_review, sessions

MONDAY = datetime(2026, 3, 2, 20, 0, tzinfo=UTC)
SPOKEN = "I am responsible of the deployment pipeline and I am developer"


@pytest.fixture
def db(tmp_path):
    connection = connect(tmp_path / "calliope.db")
    yield connection
    connection.close()


@pytest.fixture
def audio(tmp_path) -> Path:
    path = tmp_path / "monday.wav"
    path.write_bytes(b"RIFF....monday")
    return path


class FakeTranscriber:
    def __init__(self, text: str = SPOKEN, confidence: float | None = 0.9):
        self.text = text
        self.confidence = confidence
        self.calls = 0

    async def transcribe(self, audio: Path) -> Transcript:
        self.calls += 1
        words = tuple(
            Word(token, i * 0.4, i * 0.4 + 0.3, self.confidence)
            for i, token in enumerate(self.text.split())
        )
        return Transcript(
            text=self.text,
            words=words,
            duration=len(words) * 0.4,
            model="fake",
            decoding="faithful",
        )


class BrokenTranscriber:
    async def transcribe(self, audio: Path) -> Transcript:
        raise TranscriptionError("no model installed")


class BrokenAssessor:
    async def assess(self, transcript, context=()):
        raise RuntimeError("the API is down")


async def run(db, audio, transcriber=None, assessor=None, **kwargs):
    return await process(
        db,
        audio=audio,
        transcriber=transcriber or FakeTranscriber(),
        assessor=assessor or RuleBasedAssessor(),
        scheduler=SM2Scheduler(),
        at=kwargs.pop("at", MONDAY),
        **kwargs,
    )


# --- the happy path, once ------------------------------------------------


async def test_a_recording_becomes_a_session_with_errors_and_a_schedule(db, audio):
    result = await run(db, audio)

    assert len(sessions(db)) == 1
    assert {finding.category for finding in result.findings} == {
        "preposition", "article"
    }
    for error in all_errors(db):
        assert get_review(db, error.id) is not None


async def test_the_first_review_is_due_the_same_evening(db, audio):
    await run(db, audio)

    (error,) = [e for e in all_errors(db) if e.category == "article"]

    # While he still remembers saying it.
    assert get_review(db, error.id).due_at == "2026-03-02T20:00:00+00:00"


async def test_a_session_knows_how_often_each_mistake_has_come_back(db, audio):
    result = await run(db, audio)

    assert all(item.occurrences == 1 for item in result.recurrences)


# --- the same recording twice -------------------------------------------


async def test_uploading_the_same_recording_twice_changes_nothing(db, audio):
    transcriber = FakeTranscriber()

    first = await run(db, audio, transcriber)
    second = await run(db, audio, transcriber)

    assert first.session_id == second.session_id
    assert second.already_processed is True
    assert len(sessions(db)) == 1
    assert len(all_errors(db)) == len(first.findings)
    # The expensive half is skipped too: a replay reads the database, it
    # does not run Whisper again.
    assert transcriber.calls == 1


async def test_a_replay_still_returns_what_was_found(db, audio):
    first = await run(db, audio)
    second = await run(db, audio)

    assert {f.original for f in second.findings} == {
        f.original for f in first.findings
    }


async def test_a_different_recording_of_the_same_sentence_is_a_new_session(
    db, audio, tmp_path
):
    other = tmp_path / "tuesday.wav"
    other.write_bytes(b"RIFF....tuesday")

    await run(db, audio)
    await run(db, other)

    # Same words, different audio: two occasions, and the repetition is
    # the signal the corpus exists to record.
    assert len(sessions(db)) == 2
    assert len(all_errors(db)) == 4


# --- what happens when something breaks ---------------------------------


async def test_an_assessor_that_fails_does_not_cost_the_session(db, audio):
    with pytest.raises(RuntimeError):
        await run(db, audio, assessor=BrokenAssessor())

    # The session was stored before assessment ran. The metrics and the
    # audio survive the day the API is down; the feedback is what is worth
    # losing.
    assert len(sessions(db)) == 1


async def test_a_transcriber_that_cannot_run_stores_nothing(db, audio):
    with pytest.raises(TranscriptionError):
        await run(db, audio, transcriber=BrokenTranscriber())

    # Nothing to store and nothing invented. An empty session would put a
    # zero on every chart.
    assert sessions(db) == []


async def test_silence_is_stored_without_findings(db, audio):
    result = await run(db, audio, transcriber=FakeTranscriber(text=""))

    assert len(sessions(db)) == 1
    assert result.findings == ()


async def test_a_low_confidence_transcript_produces_no_corrections(db, audio):
    result = await run(db, audio, FakeTranscriber(confidence=0.2))

    # The words the recogniser was unsure of are exactly the ones a
    # correction must not be built on.
    assert result.findings == ()
    assert len(sessions(db)) == 1


# --- merging two assessors ----------------------------------------------


class Canned:
    def __init__(self, findings):
        self._findings = findings
        self.contexts = []

    async def assess(self, transcript, context=()):
        self.contexts.append(list(context))
        return list(self._findings)


async def test_the_second_assessor_is_only_credited_with_what_it_adds():
    duplicate = Finding("preposition", "responsible of", "responsible for", "")
    extra = Finding("false_friend", "sensible", "sensitive", "")
    merged = MergingAssessor([Canned([duplicate]), Canned([duplicate, extra])])

    findings = await merged.assess(Transcript(text=SPOKEN, duration=5))

    # Otherwise the repetition count -- which is what the practice
    # schedule is built on -- would depend on how many assessors happened
    # to be configured.
    assert len(findings) == 2


async def test_the_assessor_is_given_earlier_utterances_as_context(db, tmp_path):
    first = tmp_path / "one.wav"
    first.write_bytes(b"RIFF....one")
    second = tmp_path / "two.wav"
    second.write_bytes(b"RIFF....two")
    listener = Canned([])

    await run(db, first, FakeTranscriber("the payment queue duplicated charges"),
              assessor=listener)
    await run(db, second,
              FakeTranscriber("the payment queue is duplicating charges again"),
              assessor=listener)

    assert listener.contexts[-1] == ["the payment queue duplicated charges"]


# --- saving the upload ---------------------------------------------------


def test_a_recording_is_saved_under_its_own_hash(tmp_path):
    first = store_recording(tmp_path, "whatever.webm", b"audio bytes")
    second = store_recording(tmp_path, "different-name.webm", b"audio bytes")

    # One file. The name carries no date, no prompt and nothing about what
    # he said -- the recordings never leave the machine and the filenames
    # should not describe them either.
    assert first == second
    assert len(list(tmp_path.iterdir())) == 1


def test_the_extension_of_the_upload_is_kept(tmp_path):
    assert store_recording(tmp_path, "a.ogg", b"x").suffix == ".ogg"


def test_an_upload_with_no_extension_is_assumed_to_be_what_browsers_send(tmp_path):
    assert store_recording(tmp_path, "blob", b"x").suffix == ".webm"
