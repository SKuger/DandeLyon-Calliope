"""Scoring one transcript against one planted error.

The whole experiment reduces to a single question per sample: is the
mistake still in the transcript? Answering it by eye does not scale past
a dozen samples and cannot be re-run in six months, so it is answered
here, deterministically, and the cases the rules cannot settle are
reported verbatim instead of being guessed at.

Three outcomes for a planted error:

- `survived`  -- the mistake is in the transcript. Usable.
- `repaired`  -- Whisper wrote the grammatical form. The evidence is gone.
- `other`     -- neither. The transcriber heard a different sentence, so
                 this sample says nothing about fidelity and everything
                 about word error rate.

Two for a control sentence, which is already correct:

- `clean`     -- transcribed as spoken.
- `altered`   -- the transcriber changed it. That matters as much as a
                 repair: an invented error becomes a lesson about a
                 mistake the user never made.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.asr.base import Transcript


class Outcome(StrEnum):
    SURVIVED = "survived"
    REPAIRED = "repaired"
    OTHER = "other"
    CLEAN = "clean"
    ALTERED = "altered"


@dataclass(frozen=True)
class Sample:
    id: str
    category: str
    spoken: str
    error: str | None = None
    repaired: tuple[str, ...] = ()
    note: str | None = None

    @property
    def is_control(self) -> bool:
        return self.error is None


def load_samples(path: Path) -> tuple[Sample, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        Sample(
            id=entry["id"],
            category=entry["category"],
            spoken=entry["spoken"],
            error=entry.get("error"),
            repaired=tuple(entry.get("repaired", ())),
            note=entry.get("note"),
        )
        for entry in raw["samples"]
    )


# Numbers are the one normalisation that is not cosmetic. The speaker
# says "twenty eight"; Whisper writes "28", and a substring match on
# "i have twenty eight years" would score a surviving error as a repair.
_NUMBERS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = (
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
)


def spell_number(token: str) -> str | None:
    """Digits to words for 0-99, which is ages, months, hours and counts.

    Larger numbers are left alone: a year reads as "twenty twenty one" or
    "two thousand twenty one" depending on the speaker, and guessing
    between them would introduce the error it is meant to remove.
    """
    if not token.isdigit():
        return None
    value = int(token)
    if value < 20:
        return _NUMBERS[value]
    if value < 100:
        tens, units = divmod(value, 10)
        return _TENS[tens] if units == 0 else f"{_TENS[tens]} {_NUMBERS[units]}"
    return None


_PUNCTUATION = re.compile(r"[^a-z0-9' ]+")
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'"})


def normalize(text: str) -> str:
    """Lowercase, de-punctuate, split hyphens, spell out small numbers.

    Hyphens become spaces so "front-end", "front end" and "frontend" do
    not count as three different transcripts of the same word. The last
    one still differs, and it shows up as an `other`, which is the honest
    outcome: we genuinely do not know what was said.
    """
    text = text.translate(_APOSTROPHES).lower().replace("-", " ")
    text = _PUNCTUATION.sub(" ", text)
    tokens = []
    for token in text.split():
        spelled = spell_number(token)
        tokens.extend(spelled.split() if spelled else [token])
    return " ".join(tokens)


def classify(sample: Sample, hypothesis: str) -> Outcome:
    """Score one transcript. Order of checks is deliberate.

    Survival is tested first. If a transcript somehow contains both the
    mistake and its repair, the mistake is visible to the assessor, and
    visible is the only property that matters downstream.
    """
    heard = normalize(hypothesis)

    if sample.is_control:
        return Outcome.CLEAN if normalize(sample.spoken) == heard else Outcome.ALTERED

    if normalize(sample.error or "") in heard:
        return Outcome.SURVIVED
    if any(normalize(form) in heard for form in sample.repaired):
        return Outcome.REPAIRED
    return Outcome.OTHER


def find_span(transcript: Transcript, span: str) -> tuple[int, int] | None:
    """Locate a normalised phrase in the transcript's word list.

    Returns the half-open index range, or None. Needed because the
    per-word confidences are the second method the brief asks about, and
    a confidence is only interesting at the place where the text changed.
    """
    wanted = normalize(span).split()
    if not wanted:
        return None

    words = [normalize(word.text) for word in transcript.words]
    # A single spoken word can normalise to several tokens ("28"), so the
    # haystack is matched on a flattened token stream and mapped back.
    flat: list[tuple[str, int]] = [
        (token, index)
        for index, word in enumerate(words)
        for token in word.split()
    ]
    for start in range(len(flat) - len(wanted) + 1):
        if [token for token, _ in flat[start : start + len(wanted)]] == wanted:
            first = flat[start][1]
            last = flat[start + len(wanted) - 1][1]
            return first, last + 1
    return None


def span_confidence(
    transcript: Transcript, span: str
) -> tuple[float, ...] | None:
    """Per-word confidence over a phrase, or None if it is not there."""
    located = find_span(transcript, span)
    if located is None:
        return None
    start, end = located
    scores = [
        word.confidence
        for word in transcript.words[start:end]
        if word.confidence is not None
    ]
    return tuple(scores) if scores else None


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein over words, normalised by reference length.

    Reported alongside the survival rate because the two failure modes
    are different problems. A repaired error means the transcript is too
    good; a high word error rate means it is too bad to grade at all, and
    on accented speech that turns out to be the binding constraint.
    """
    source = normalize(reference).split()
    target = normalize(hypothesis).split()
    if not source:
        return 0.0 if not target else 1.0

    previous = list(range(len(target) + 1))
    for i, want in enumerate(source, start=1):
        current = [i]
        for j, got in enumerate(target, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (want != got),
                )
            )
        previous = current
    return previous[-1] / len(source)
