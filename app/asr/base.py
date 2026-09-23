"""The speech recognition boundary.

Two reasons this is a `Protocol` and not a direct call into
faster-whisper:

- The repository has to run, and its suite has to pass, on a machine with
  no model weights. A dependency that downloads half a gigabyte before
  anything works is a dependency nobody tries.
- The whole project rests on a claim about *how* a transcript was
  produced. Whisper with a beam search and a conditioning prompt is a
  different instrument from Whisper decoding greedily at temperature
  zero, and this project exists because the difference shows up in the
  data (see `docs/whisper-grammar-fidelity.md`). So a transcript carries
  the model and the decoding settings that produced it, and anything
  that stores a measurement stores those too.

A transcript that cannot say where it came from is not evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol


class TranscriptionError(Exception):
    """Transcription could not be produced, and the caller has to know.

    Raised instead of returning an empty or invented transcript. Made-up
    text would flow straight into the metrics and the error corpus, where
    it is indistinguishable from something the user actually said.
    """


@dataclass(frozen=True)
class Word:
    """One token with its place in time and how sure the model was.

    `confidence` is the probability the decoder assigned to this token,
    already exponentiated out of log space, or `None` when the backend
    does not report one. It is the cheapest signal available for the
    question this project cares about: a word Whisper *inserted* to
    repair a sentence tends to be less certain than the words it heard.
    """

    text: str
    start: float
    end: float
    confidence: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class Transcript:
    text: str
    words: tuple[Word, ...] = ()
    language: str = "en"

    duration: float = 0.0
    """Length of the audio, not of the speech. Words per minute needs the
    former; it is the denominator of half the metrics."""

    model: str = "unknown"
    decoding: str = "unknown"
    """Provenance. See the module docstring: without these two, a number
    measured today cannot be compared with a number measured in June."""

    @property
    def token_count(self) -> int:
        return len(self.words)

    def with_provenance(self, model: str, decoding: str) -> Transcript:
        return replace(self, model=model, decoding=decoding)


@dataclass(frozen=True)
class DecodeOptions:
    """Decoding settings, named so a result can be filed under them.

    The defaults here are not faster-whisper's. They are the settings
    measured to be the least likely to silently repair the speaker's
    grammar: greedy, no temperature fallback, no conditioning on
    previously decoded text.

    `condition_on_previous_text` is the dangerous one. It feeds the last
    segment back in as a prompt, which is how Whisper keeps a long
    transcript coherent -- and coherent is exactly what a learner's
    speech is not.
    """

    name: str = "faithful"
    beam_size: int = 1
    temperature: tuple[float, ...] = (0.0,)
    condition_on_previous_text: bool = False
    initial_prompt: str | None = None
    word_timestamps: bool = True

    vad_filter: bool = False
    """Voice-activity filtering drops audio it judges to be silence. It
    also drops the hesitations and restarts this project measures, so it
    is off by default."""

    extra: dict[str, Any] = field(default_factory=dict)

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "beam_size": self.beam_size,
            "temperature": list(self.temperature),
            "condition_on_previous_text": self.condition_on_previous_text,
            "word_timestamps": self.word_timestamps,
            "vad_filter": self.vad_filter,
        }
        if self.initial_prompt is not None:
            kwargs["initial_prompt"] = self.initial_prompt
        kwargs.update(self.extra)
        return kwargs


#: What Whisper is normally called with: a beam search, a temperature
#: ladder to fall back on, and the previous text as context. Kept as a
#: named constant because it is the control condition in the experiment,
#: not because anything in the app should use it.
WHISPER_DEFAULTS = DecodeOptions(
    name="whisper-defaults",
    beam_size=5,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    condition_on_previous_text=True,
)


class Transcriber(Protocol):
    async def transcribe(self, audio: Path) -> Transcript: ...


class ScriptedTranscriber:
    """The default: transcripts from a fixture directory, no model.

    This is not a test double. It is what runs when no weights are
    installed, which is what makes `docker compose up` and the suite work
    on a machine that has never downloaded Whisper. Each fixture is a
    JSON file named after the audio it stands for.

    An unknown recording raises instead of guessing. A transcriber that
    invents text when it does not recognise a file would put invented
    sentences into the error corpus, and the user would end up practising
    mistakes he never made.
    """

    name = "scripted"

    def __init__(self, fixtures: Path | None = None) -> None:
        self._fixtures = fixtures

    async def transcribe(self, audio: Path) -> Transcript:
        if self._fixtures is None:
            raise TranscriptionError(
                "No transcriber is configured. Install the ASR extra "
                "(pip install -r requirements-asr.txt) or point "
                "TRANSCRIPT_FIXTURES at a fixture directory."
            )

        path = self._fixtures / f"{audio.stem}.json"
        if not path.is_file():
            raise TranscriptionError(
                f"No scripted transcript for {audio.name}. This build has "
                f"no speech recognition installed; it can only replay the "
                f"sample recordings in {self._fixtures}."
            )
        return load_transcript(path).with_provenance(
            model=self.name, decoding="fixture"
        )


def load_transcript(path: Path) -> Transcript:
    return transcript_from_payload(json.loads(path.read_text(encoding="utf-8")))


def transcript_from_payload(raw: dict[str, Any]) -> Transcript:
    words = tuple(
        Word(
            text=word["text"],
            start=float(word["start"]),
            end=float(word["end"]),
            confidence=(
                None
                if word.get("confidence") is None
                else float(word["confidence"])
            ),
        )
        for word in raw.get("words", ())
    )
    duration = float(raw.get("duration") or (words[-1].end if words else 0.0))
    return Transcript(
        text=raw["text"],
        words=words,
        language=raw.get("language", "en"),
        duration=duration,
        model=raw.get("model", "fixture"),
        decoding=raw.get("decoding", "fixture"),
    )


def dump_transcript(transcript: Transcript) -> dict[str, Any]:
    return {
        "text": transcript.text,
        "language": transcript.language,
        "duration": round(transcript.duration, 3),
        "model": transcript.model,
        "decoding": transcript.decoding,
        "words": [
            {
                "text": word.text,
                "start": round(word.start, 3),
                "end": round(word.end, 3),
                "confidence": (
                    None
                    if word.confidence is None
                    else round(word.confidence, 4)
                ),
            }
            for word in transcript.words
        ],
    }
