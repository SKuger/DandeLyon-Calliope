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

Normalisation, span lookup and word error rate live in `app.text`: the
assessor needs the same three operations, and two copies would drift
until the experiment measured something the product does not do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.text import normalize, span_confidence, word_error_rate

__all__ = [
    "Outcome",
    "Sample",
    "classify",
    "load_samples",
    "normalize",
    "span_confidence",
    "word_error_rate",
]


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


def classify(sample: Sample, hypothesis: str) -> Outcome:
    """Score one transcript. The order of the checks is deliberate.

    Survival is tested first. If a transcript somehow contains both the
    mistake and its repair, the mistake is visible to the assessor, and
    visible is the only property anything downstream depends on.
    """
    heard = normalize(hypothesis)

    if sample.is_control:
        return (
            Outcome.CLEAN if normalize(sample.spoken) == heard else Outcome.ALTERED
        )

    if normalize(sample.error or "") in heard:
        return Outcome.SURVIVED
    if any(normalize(form) in heard for form in sample.repaired):
        return Outcome.REPAIRED
    return Outcome.OTHER
