"""The transcription boundary, tested at the places it gives up.

The interesting assertions here are the refusals. A transcriber that
returns something plausible when it does not know is the single worst
failure this project can have: the invented sentence goes into the error
corpus, and the user practises a mistake he never made.
"""

import json
from pathlib import Path

import pytest

from app.asr.base import (
    WHISPER_DEFAULTS,
    DecodeOptions,
    ScriptedTranscriber,
    Transcript,
    TranscriptionError,
    Word,
    dump_transcript,
    load_transcript,
)
from app.asr.whisper import FasterWhisperTranscriber


def write_fixture(directory: Path, stem: str, **overrides) -> Path:
    payload = {
        "text": "yesterday I go to the office",
        "duration": 3.5,
        "words": [
            {"text": "yesterday", "start": 0.0, "end": 0.6, "confidence": 0.95},
            {"text": "I", "start": 0.6, "end": 0.7, "confidence": 0.99},
            {"text": "go", "start": 0.7, "end": 1.0, "confidence": 0.41},
            {"text": "to", "start": 1.0, "end": 1.1, "confidence": 0.98},
            {"text": "the", "start": 1.1, "end": 1.2, "confidence": 0.97},
            {"text": "office", "start": 1.2, "end": 1.8, "confidence": 0.96},
        ],
    }
    payload.update(overrides)
    path = directory / f"{stem}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- the offline default -------------------------------------------------


async def test_unconfigured_transcriber_says_what_is_missing():
    with pytest.raises(TranscriptionError) as raised:
        await ScriptedTranscriber().transcribe(Path("session.wav"))

    # The message is the whole value of this branch: someone who cloned
    # the repo and pressed record needs to know why nothing happened.
    assert "requirements-asr" in str(raised.value)


async def test_an_unknown_recording_raises_instead_of_inventing_text(tmp_path):
    transcriber = ScriptedTranscriber(tmp_path)

    with pytest.raises(TranscriptionError):
        await transcriber.transcribe(tmp_path / "monday.wav")


async def test_a_scripted_transcript_is_marked_as_scripted(tmp_path):
    write_fixture(tmp_path, "monday", model="faster-whisper:medium.en")

    transcript = await ScriptedTranscriber(tmp_path).transcribe(
        tmp_path / "monday.wav"
    )

    # Provenance is overwritten, not trusted. A replayed fixture must
    # never be filed as if a model had produced it, or six months of
    # measurements end up comparing two different instruments.
    assert transcript.model == "scripted"
    assert transcript.decoding == "fixture"
    assert transcript.token_count == 6


# --- serialisation -------------------------------------------------------


def test_duration_falls_back_to_the_last_word_when_absent(tmp_path):
    path = write_fixture(tmp_path, "monday", duration=0)

    assert load_transcript(path).duration == pytest.approx(1.8)


def test_a_transcript_survives_a_round_trip(tmp_path):
    original = load_transcript(write_fixture(tmp_path, "monday"))
    path = tmp_path / "again.json"
    path.write_text(json.dumps(dump_transcript(original)), encoding="utf-8")

    assert load_transcript(path) == original


def test_a_missing_confidence_stays_missing(tmp_path):
    path = write_fixture(
        tmp_path,
        "monday",
        words=[{"text": "hello", "start": 0.0, "end": 0.4}],
    )

    # None and 0.0 mean opposite things: "the backend does not report
    # confidence" versus "the model was certain this was not a word".
    assert load_transcript(path).words[0].confidence is None


def test_an_empty_transcript_is_a_valid_transcript():
    silence = Transcript(text="", duration=4.0)

    assert silence.token_count == 0
    assert silence.words == ()


def test_word_duration_never_goes_negative():
    assert Word("go", start=1.0, end=0.9).duration == 0.0


# --- decoding settings ---------------------------------------------------


def test_the_default_decoding_refuses_the_settings_that_repair_grammar():
    faithful = DecodeOptions()

    # This is the project's premise as an assertion. If someone "fixes"
    # these defaults to Whisper's, the tutor starts grading a transcript
    # that has been quietly corrected, and this test is the alarm.
    assert faithful.condition_on_previous_text is False
    assert faithful.temperature == (0.0,)
    assert faithful.beam_size == 1
    assert faithful.vad_filter is False


def test_the_control_condition_is_kept_separate():
    assert WHISPER_DEFAULTS.condition_on_previous_text is True
    assert len(WHISPER_DEFAULTS.temperature) > 1


def test_decode_options_only_pass_a_prompt_when_there_is_one():
    assert "initial_prompt" not in DecodeOptions().as_kwargs()
    assert DecodeOptions(initial_prompt="Um, so, like,").as_kwargs()[
        "initial_prompt"
    ] == "Um, so, like,"


# --- the real backend, without the weights -------------------------------


async def test_whisper_reports_a_missing_recording_before_loading_a_model():
    # No weights are installed in CI, so this asserts the check happens
    # first. A FileNotFoundError from inside CTranslate2 half a minute
    # later would be the same bug with a worse error message.
    with pytest.raises(TranscriptionError) as raised:
        await FasterWhisperTranscriber().transcribe(Path("nope.wav"))

    assert "nope.wav" in str(raised.value)


def test_whisper_names_the_model_it_used():
    assert FasterWhisperTranscriber("medium.en").name == (
        "faster-whisper:medium.en"
    )
