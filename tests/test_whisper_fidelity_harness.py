"""The harness around the experiment: resumability and re-synthesis.

Neither of these is about Whisper. They are about the experiment being
repeatable in six months, which is the only reason to write it down at
all. The matrix takes half an hour, so a run that cannot resume is a run
nobody repeats; and audio regenerated from an edited sentence, into the
same filename, would be scored against a sentence nobody said.
"""

import json
from pathlib import Path

import pytest

from app.asr.base import DecodeOptions, Transcript, Word
from experiments.whisper_fidelity.run import Key, existing, row_for, run, summarize
from experiments.whisper_fidelity.score import Sample
from experiments.whisper_fidelity.synthesize import Voice, plan

SAMPLES = (
    Sample(
        id="tense",
        category="verb_tense",
        spoken="Yesterday I go to the office.",
        error="yesterday i go",
        repaired=("yesterday i went",),
    ),
    Sample(
        id="control",
        category="control",
        spoken="I reviewed two pull requests.",
    ),
)

CONFIGS = (DecodeOptions(name="faithful"),)


def as_transcript(text: str, confidence: float = 0.9) -> Transcript:
    words = tuple(
        Word(text=token, start=i * 0.4, end=i * 0.4 + 0.3, confidence=confidence)
        for i, token in enumerate(text.split())
    )
    return Transcript(
        text=text,
        words=words,
        duration=len(words) * 0.4,
        model="stub",
        decoding="faithful",
    )


class StubTranscriber:
    """Returns a canned transcript and counts how often it was asked."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.calls: list[str] = []

    async def transcribe(self, audio: Path) -> Transcript:
        self.calls.append(audio.stem)
        return as_transcript(self.replies[audio.stem])


@pytest.fixture
def audio(tmp_path) -> Path:
    root = tmp_path / "recordings"
    voice = root / "native-us"
    voice.mkdir(parents=True)
    for sample in SAMPLES:
        (voice / f"{sample.id}.wav").write_bytes(b"RIFF")
    return root


REPLIES = {
    "tense": "Yesterday I went to the office.",
    "control": "I reviewed two pull requests.",
}


# --- resumability --------------------------------------------------------


async def test_a_finished_row_is_not_recomputed(audio, tmp_path):
    rows = tmp_path / "rows.jsonl"
    stub = StubTranscriber(REPLIES)

    first = await run(audio, SAMPLES, ("base.en",), CONFIGS, rows,
                      build=lambda *_: stub)
    second = await run(audio, SAMPLES, ("base.en",), CONFIGS, rows,
                       build=lambda *_: stub)

    assert (first, second) == (2, 0)
    # The real cost is thirty seconds of decoding per row, and the point
    # of the ledger is that it is paid once.
    assert len(stub.calls) == 2
    assert len(rows.read_text(encoding="utf-8").strip().splitlines()) == 2


async def test_adding_a_model_only_runs_the_new_one(audio, tmp_path):
    rows = tmp_path / "rows.jsonl"
    stub = StubTranscriber(REPLIES)

    await run(audio, SAMPLES, ("base.en",), CONFIGS, rows,
              build=lambda *_: stub)
    added = await run(audio, SAMPLES, ("base.en", "small.en"), CONFIGS, rows,
                      build=lambda *_: stub)

    assert added == 2


async def test_a_missing_recording_is_skipped_rather_than_fatal(audio, tmp_path):
    (audio / "native-us" / "control.wav").unlink()

    written = await run(
        audio, SAMPLES, ("base.en",), CONFIGS, tmp_path / "rows.jsonl",
        build=lambda *_: StubTranscriber(REPLIES),
    )

    # A voice with a gap in it still contributes its other rows. Half a
    # matrix is a result; a crash on row four is not.
    assert written == 1


async def test_no_audio_at_all_says_which_command_to_run(tmp_path):
    empty = tmp_path / "recordings"
    empty.mkdir()

    with pytest.raises(SystemExit) as raised:
        await run(empty, SAMPLES, ("base.en",), CONFIGS, tmp_path / "r.jsonl")

    assert "synthesize" in str(raised.value)


def test_a_ledger_that_does_not_exist_yet_is_empty(tmp_path):
    assert existing(tmp_path / "nothing.jsonl") == set()


# --- what a row records --------------------------------------------------


def test_a_repaired_row_records_the_confidence_of_the_invented_words():
    transcript = as_transcript("Yesterday I went to the office.", confidence=0.4)

    row = row_for(
        Key("native-us", "base.en", "faithful", "tense"),
        SAMPLES[0],
        transcript,
        seconds=1.2,
    )

    assert row["outcome"] == "repaired"
    assert row["span"] == "yesterday i went"
    assert row["span_confidence"] == [0.4, 0.4, 0.4]


def test_a_row_that_is_neither_keeps_the_transcript_for_a_human_to_read():
    row = row_for(
        Key("es-mx-male", "base.en", "faithful", "tense"),
        SAMPLES[0],
        as_transcript("Just a day ago told the office."),
        seconds=1.0,
    )

    assert row["outcome"] == "other"
    assert row["span"] is None
    assert row["heard"] == "Just a day ago told the office."
    assert row["wer"] > 0


def test_a_row_records_what_produced_it():
    row = row_for(
        Key("native-us", "medium.en", "whisper-defaults", "tense"),
        SAMPLES[0],
        as_transcript("Yesterday I go to the office."),
        seconds=5.0,
    )

    assert (row["model"], row["config"]) == ("medium.en", "whisper-defaults")
    assert row["outcome"] == "survived"


# --- the summary ---------------------------------------------------------


def test_the_summary_quotes_every_unscoreable_transcript(tmp_path):
    rows = [
        row_for(
            Key("es-mx-male", "base.en", "faithful", "tense"),
            SAMPLES[0],
            as_transcript("Just a day ago told the office."),
            seconds=1.0,
        )
    ]

    # The aggregate is not interpretable without these, so the generator
    # is not allowed to drop them.
    assert "Just a day ago told the office." in summarize(rows)


def test_the_summary_separates_controls_from_planted_errors(tmp_path):
    rows = [
        row_for(Key("native-us", "base.en", "faithful", "tense"), SAMPLES[0],
                as_transcript("Yesterday I go to the office."), 1.0),
        row_for(Key("native-us", "base.en", "faithful", "control"), SAMPLES[1],
                as_transcript("I review two pull requests."), 1.0),
    ]

    text = summarize(rows)

    assert "1 planted errors, 1 controls" in text
    assert "did the transcriber invent an error?" in text


# --- re-synthesis --------------------------------------------------------


def test_audio_already_made_from_the_same_sentence_is_left_alone(tmp_path):
    voice = Voice("native-us", "Microsoft Zira Desktop")
    out = tmp_path / "native-us"
    out.mkdir()
    pending = plan(SAMPLES, voice, out)
    for item in pending:
        Path(item["path"]).write_bytes(b"RIFF")
    (out / "manifest.json").write_text(
        json.dumps({item["id"]: item["fingerprint"] for item in pending}),
        encoding="utf-8",
    )

    assert plan(SAMPLES, voice, out) == []


def test_an_edited_sentence_invalidates_its_audio(tmp_path):
    voice = Voice("native-us", "Microsoft Zira Desktop")
    out = tmp_path / "native-us"
    out.mkdir()
    for item in plan(SAMPLES, voice, out):
        Path(item["path"]).write_bytes(b"RIFF")
    (out / "manifest.json").write_text(
        json.dumps({"tense": "stale", "control": "stale"}), encoding="utf-8"
    )

    # Otherwise the file says "yesterday I go" and the fixture says
    # something else, and the experiment scores a sentence nobody said.
    assert {item["id"] for item in plan(SAMPLES, voice, out)} == {
        "tense",
        "control",
    }


def test_changing_the_speaking_rate_is_a_different_recording(tmp_path):
    out = tmp_path / "native-us"
    out.mkdir()
    slow = plan(SAMPLES, Voice("v", "Microsoft Zira Desktop", rate=0), out)
    fast = plan(SAMPLES, Voice("v", "Microsoft Zira Desktop", rate=3), out)

    assert slow[0]["fingerprint"] != fast[0]["fingerprint"]
