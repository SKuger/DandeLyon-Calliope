"""Exercises made out of his own mistakes, and graded without a model.

There is no course in this repository. An exercise is one error from the
corpus, handed back to him with an instruction to say it properly out
loud -- so the material is whatever he actually got wrong, and it stops
being generated the day he stops getting it wrong.

Grading is deterministic, and that is a design decision rather than a
shortcut. The grade feeds the review schedule, so a model in this path
would mean the interval between repetitions moves with the model's mood.
The judgement it has to make is narrow enough for rules: he either said
the structure correctly, said it wrong again, or avoided it.

Avoidance gets its own grade. An intermediate speaker who cannot say
"I have been working here for three months" says "I work here" instead
and never appears to make a mistake. Scoring that as a pass would let the
corpus quietly forget the structure he is hiding from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.asr.base import Transcript
from app.assess.base import Finding
from app.metrics.fluency import measure
from app.store import StoredError
from app.text import normalize

#: Grades, on SM-2's 0-5 scale.
REPEATED = 1
"""He made the same mistake again, word for word."""

SAME_CATEGORY = 2
"""Different words, same kind of mistake."""

AVOIDED = 3
"""No mistake, and no sign of the structure either. A pass he did not
earn: barely enough to push the card back, not enough to grow the
interval the way a real answer does."""

HESITANT = 4
"""Correct, with a restart or a long pause on the way."""

FLUENT = 5

#: A pause this long inside a short answer is him assembling the sentence
#: rather than recalling it. Longer than the utterance boundary in the
#: metrics on purpose: here it has to be unambiguous.
HESITATION_SECONDS = 1.2


@dataclass(frozen=True)
class Exercise:
    error_id: str
    kind: str
    prompt: str
    original: str
    target: str
    category: str
    explanation: str


def exercise_for(error: StoredError, repetitions: int = 0) -> Exercise:
    """One error, turned into something to say out loud.

    The second form appears only after he has already repaired the
    sentence twice. Repeating the same repair forever teaches the
    sentence, not the structure -- the point is to use it somewhere he has
    not used it before.
    """
    if repetitions >= 2:
        return Exercise(
            error_id=error.id,
            kind="reuse",
            prompt=(
                f'Say a new sentence about your work that uses '
                f'"{error.correction}".'
            ),
            original=error.original,
            target=error.correction,
            category=error.category,
            explanation=error.explanation,
        )
    return Exercise(
        error_id=error.id,
        kind="repair",
        prompt=(
            f'You said: "{error.original}". Say the whole thing again, '
            f'out loud and correctly.'
        ),
        original=error.original,
        target=error.correction,
        category=error.category,
        explanation=error.explanation,
    )


def grade_attempt(
    exercise: Exercise,
    transcript: Transcript,
    findings: Sequence[Finding] = (),
) -> int:
    """Score one spoken attempt, 0-5, with no model in the loop."""
    said = normalize(transcript.text)
    if not said.strip():
        return 0

    if normalize(exercise.original) in said:
        return REPEATED

    if any(finding.category == exercise.category for finding in findings):
        return SAME_CATEGORY

    if normalize(exercise.target) not in said:
        return AVOIDED

    metrics = measure(transcript)
    longest = (
        metrics.pauses.longest_seconds
        if metrics.pauses and metrics.pauses.longest_seconds is not None
        else 0.0
    )
    if metrics.self_corrections or longest >= HESITATION_SECONDS:
        return HESITANT
    return FLUENT
