"""The assessor boundary: what is wrong with a sentence, never a score.

The split this module enforces is the one the whole project rests on.
Numbers come from arithmetic over the transcript (`app.metrics`); a model
is only ever asked for the thing arithmetic cannot do -- naming the
mistake and writing the corrected sentence. Nothing here returns a rating,
a level or a percentage, and nothing downstream will accept one.

Two guards sit between any assessor and the error corpus, and they are
the reason this is a module rather than a function:

**A finding must quote the transcript.** A model asked to correct English
will happily correct a sentence the user did not say -- paraphrasing it
first, then fixing the paraphrase. That finding looks perfect and teaches
nothing, because the user cannot recognise the sentence. Anything whose
`original` is not literally in the transcript is dropped.

**A finding must be about words the transcriber actually heard.** This is
the product consequence of `docs/whisper-grammar-fidelity.md`: on accented
speech, a large share of what looks like a grammar mistake is the ASR
mishearing a word. Correcting those trains him out of mistakes he never
made, which is worse than missing a real one. So a finding whose words
carry low recogniser confidence is dropped, and the threshold is a
measured number rather than a guess.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.asr.base import Transcript
from app.text import normalize, span_confidence

logger = logging.getLogger(__name__)

#: Below this mean per-word confidence, a correction is more likely to be
#: about the transcriber than about the speaker.
#:
#: Measured rather than guessed. On the accented recordings
#: (`experiments/whisper_fidelity/results/confidence.md`) a word Whisper
#: actually heard has a median confidence of 0.94, and a word it invented
#: has 0.47. A floor here drops 62% of the invented words and takes 13% of
#: the real ones with them -- a deliberate trade in that direction,
#: because a correction of something he did not say costs more than a
#: missed mistake he will make again next week.
MIN_SPAN_CONFIDENCE = 0.55


@dataclass(frozen=True)
class Finding:
    """One mistake, in the user's own words.

    `original` is the phrase as it appears in the transcript, not a
    paraphrase. It is what gets played back to him, what the exercise is
    built from, and what the corpus counts repetitions of -- all three
    break if it is a reconstruction.
    """

    category: str
    original: str
    correction: str
    explanation: str
    detector: str = "rules"
    confidence: float | None = None
    """Mean recogniser confidence over `original`, or None when the
    backend reports none. Stored so a later run can re-examine what was
    let through."""


class Assessor(Protocol):
    async def assess(
        self, transcript: Transcript, context: Sequence[str] = ()
    ) -> list[Finding]: ...


def quoted_from(transcript: Transcript, finding: Finding) -> bool:
    return normalize(finding.original) in normalize(transcript.text)


def admissible(
    transcript: Transcript,
    findings: Sequence[Finding],
    min_confidence: float = MIN_SPAN_CONFIDENCE,
) -> list[Finding]:
    """Drop findings that are about something other than what he said.

    Both rejections are logged rather than silently dropped: a detector
    that is routinely filtered out is a detector that needs fixing, and
    that is invisible if the filtering leaves no trace.
    """
    kept: list[Finding] = []
    for finding in findings:
        if not quoted_from(transcript, finding):
            logger.info(
                "dropped %s finding: %r is not in the transcript",
                finding.detector,
                finding.original,
            )
            continue

        scores = span_confidence(transcript, finding.original)
        mean = None if scores is None else sum(scores) / len(scores)
        if mean is not None and mean < min_confidence:
            logger.info(
                "dropped %s finding on %r: confidence %.2f below %.2f",
                finding.detector,
                finding.original,
                mean,
                min_confidence,
            )
            continue

        kept.append(
            finding if mean is None else Finding(**{**finding.__dict__,
                                                   "confidence": round(mean, 4)})
        )
    return kept


class NullAssessor:
    """Finds nothing, on purpose.

    Used when assessment is switched off entirely. It exists so that
    "no assessor configured" is a supported state with a name, rather
    than a `None` that every caller has to remember to check.
    """

    name = "none"

    async def assess(
        self, transcript: Transcript, context: Sequence[str] = ()
    ) -> list[Finding]:
        return []
