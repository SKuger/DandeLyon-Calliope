"""When to ask him about a mistake again.

SM-2, behind a Protocol. The obvious alternative is FSRS, and it is
better -- but its advantage comes from parameters fitted to millions of
reviews, and this deck is one person's few hundred errors. Shipping FSRS
here would mean shipping its default weights, which is a guess with
seventeen more moving parts than SM-2 and no data to fit them with. SM-2
is twenty lines, deterministic, and replaceable the day there is a review
history worth fitting anything to. That day is the point of the Protocol.

The scheduler knows nothing about English. It takes a grade and returns
the next date, so the interesting decision -- what counts as knowing it --
lives with the thing that can actually judge an attempt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol

#: SM-2's ease factor floor. Below this the interval stops growing in any
#: useful way and the card comes back every day forever, which is how a
#: review queue becomes something you stop opening.
MIN_EASE = 1.3
INITIAL_EASE = 2.5

#: A grade below this is a failure and restarts the interval.
PASS_GRADE = 3


@dataclass(frozen=True)
class ReviewState:
    error_id: str
    due_at: str
    interval_days: float
    ease: float = INITIAL_EASE
    repetitions: int = 0
    lapses: int = 0
    """How many times he got it right and then lost it again. Not used by
    the schedule: it is the number that says an error needs a different
    kind of practice, not an earlier repeat."""

    last_reviewed_at: str | None = None


class Scheduler(Protocol):
    def first(self, error_id: str, at: datetime) -> ReviewState: ...
    def next(
        self, state: ReviewState, grade: int, at: datetime
    ) -> ReviewState: ...


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).replace(microsecond=0).isoformat()


class SM2Scheduler:
    """SuperMemo 2, with the first two intervals pinned.

    The original algorithm's first interval is one day and its second is
    six. Both are kept: they are the part of SM-2 that was tuned against
    real forgetting curves, and changing them to something rounder would
    throw away the only empirical content in it.
    """

    name = "sm2"

    def __init__(self, first_interval: float = 1.0, second_interval: float = 6.0):
        self._first = first_interval
        self._second = second_interval

    def first(self, error_id: str, at: datetime) -> ReviewState:
        # Due immediately. An error found this evening is worth one attempt
        # this evening, while he still remembers saying it.
        return ReviewState(
            error_id=error_id, due_at=_iso(at), interval_days=0.0
        )

    def next(self, state: ReviewState, grade: int, at: datetime) -> ReviewState:
        grade = max(0, min(5, int(grade)))
        ease = max(
            MIN_EASE,
            state.ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)),
        )

        if grade < PASS_GRADE:
            # Back to the start of the ladder, but the ease penalty stays.
            # An error he keeps losing should come back sooner *and* grow
            # more slowly than one he has never missed.
            return ReviewState(
                error_id=state.error_id,
                due_at=_iso(at + timedelta(days=self._first)),
                interval_days=self._first,
                ease=ease,
                repetitions=0,
                lapses=state.lapses + (1 if state.repetitions else 0),
                last_reviewed_at=_iso(at),
            )

        repetitions = state.repetitions + 1
        if repetitions == 1:
            interval = self._first
        elif repetitions == 2:
            interval = self._second
        else:
            interval = round(state.interval_days * ease, 2)

        return replace(
            state,
            due_at=_iso(at + timedelta(days=interval)),
            interval_days=interval,
            ease=ease,
            repetitions=repetitions,
            last_reviewed_at=_iso(at),
        )
