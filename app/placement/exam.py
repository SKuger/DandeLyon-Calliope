"""A placement exam that is fixed, versioned, and therefore comparable.

This is the part almost every language app gets wrong, and it is worth
being precise about why. An exam generated fresh each time cannot support
the only claim that matters after six months -- that he got better --
because the second exam is not the first one. The difference between two
sittings is then the difference between two exams plus the difference in
the speaker, and nothing separates the two terms.

So the exam is a file. The same items, in the same order, with the same
answers, in September as in March. It carries a version and a
fingerprint, the fingerprint goes into every stored result, and comparing
two sittings with different fingerprints is refused rather than silently
misreported.

**What the bands are and are not.** The CEFR band per skill is a coarse
estimate, calibrated by judgement against published descriptors and not
against a cohort of test-takers -- there is one test-taker. Treating it
as a certified level would be dishonest. What *is* exact is that the same
answers always produce the same band, so the movement between two
sittings is a measurement even when the absolute level is an opinion.

**Four bands, not one number.** His profile is uneven on purpose: strong
reading and writing, weak speaking. A single score would average that
into a number that describes nobody and would go up when the strong half
improved.

A section he did not attempt has no band. Not zero -- zero is a result,
and a section nobody answered is the absence of one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.assess.base import Finding
from app.metrics.fluency import FluencyMetrics

EXAM_DIR = Path(__file__).resolve().parent / "fixtures"

#: Percentage thresholds for the written sections. Coarse by design: the
#: point is repeatability, and a finer scale would imply a precision the
#: instrument does not have.
BANDS = ((80, "C1"), (60, "B2"), (40, "B1"), (0, "A2"))

#: Low to high. The order is what lets two sittings be subtracted.
BAND_ORDER = ("A2", "B1", "B2", "C1")


@dataclass(frozen=True)
class Item:
    id: str
    prompt: str
    options: tuple[str, ...] = ()
    answer: int | None = None
    script: str | None = None
    """For listening items: what the voice reads aloud. Stored as text
    rather than as an audio file so the exam stays one reviewable
    artifact, and so the recordings never end up in git."""

    seconds: int | None = None
    targets: tuple[str, ...] = ()
    """Structures the item is trying to elicit. Not scored directly --
    used to check the exam covers the ground it claims to."""


@dataclass(frozen=True)
class Section:
    id: str
    skill: str
    kind: str
    instructions: str
    items: tuple[Item, ...]


@dataclass(frozen=True)
class Exam:
    version: str
    sections: tuple[Section, ...]
    fingerprint: str

    def section(self, section_id: str) -> Section:
        for section in self.sections:
            if section.id == section_id:
                return section
        raise KeyError(section_id)

    @property
    def skills(self) -> tuple[str, ...]:
        return tuple(section.skill for section in self.sections)


@dataclass(frozen=True)
class SkillResult:
    skill: str
    band: str | None
    raw: float
    max: float
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def percentage(self) -> float | None:
        if self.max == 0:
            return None
        return round(100 * self.raw / self.max, 1)


def band_for(percentage: float | None) -> str | None:
    if percentage is None:
        return None
    for floor, band in BANDS:
        if percentage >= floor:
            return band
    return None


def load_exam(path: Path) -> Exam:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sections = tuple(
        Section(
            id=section["id"],
            skill=section["skill"],
            kind=section["kind"],
            instructions=section.get("instructions", ""),
            items=tuple(
                Item(
                    id=item["id"],
                    prompt=item["prompt"],
                    options=tuple(item.get("options", ())),
                    answer=item.get("answer"),
                    script=item.get("script"),
                    seconds=item.get("seconds"),
                    targets=tuple(item.get("targets", ())),
                )
                for item in section["items"]
            ),
        )
        for section in raw["sections"]
    )
    return Exam(
        version=raw["version"],
        sections=sections,
        fingerprint=fingerprint_of(raw),
    )


def fingerprint_of(raw: Mapping[str, Any]) -> str:
    """A hash of the exam's content, not of its file.

    Stored with every sitting. Two sittings of the same version with
    different fingerprints means someone edited an item without bumping
    the version, and the comparison between them is not a measurement of
    anything. Better to find that out from a mismatch than from a
    surprising improvement.
    """
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def score_choices(
    section: Section, answers: Mapping[str, int]
) -> SkillResult:
    """Score a multiple-choice section.

    Unanswered items are left out of the denominator rather than marked
    wrong. Half an exam is a partial measurement; scoring the unanswered
    half as failure turns it into a wrong one.
    """
    answered = [item for item in section.items if item.id in answers]
    correct = [item for item in answered if answers[item.id] == item.answer]
    result = SkillResult(
        skill=section.skill,
        band=None,
        raw=len(correct),
        max=len(answered),
        detail={
            "answered": len(answered),
            "of": len(section.items),
            "missed": [item.id for item in answered if item not in correct],
        },
    )
    return SkillResult(
        skill=result.skill,
        band=band_for(result.percentage),
        raw=result.raw,
        max=result.max,
        detail=result.detail,
    )


#: Thresholds for spoken production, one row per band, read top down. Each
#: is a floor that all of its conditions must clear. Calibrated by
#: judgement -- see the module docstring -- and kept here as data so that
#: recalibrating is an edit to a table rather than a rewrite.
SPEAKING_BANDS: tuple[tuple[str, dict[str, float]], ...] = (
    (
        "C1",
        {"words_per_minute": 130, "mean_utterance_words": 12,
         "lexical_diversity": 0.62, "max_errors_per_100": 2.0,
         "max_crutches_per_minute": 0.2},
    ),
    (
        "B2",
        {"words_per_minute": 105, "mean_utterance_words": 9,
         "lexical_diversity": 0.55, "max_errors_per_100": 4.0,
         "max_crutches_per_minute": 0.6},
    ),
    (
        "B1",
        {"words_per_minute": 75, "mean_utterance_words": 6,
         "lexical_diversity": 0.45, "max_errors_per_100": 8.0,
         "max_crutches_per_minute": 1.5},
    ),
)


def score_speaking(
    metrics: FluencyMetrics,
    findings: Sequence[Finding],
    skill: str = "speaking",
) -> SkillResult:
    """Turn the deterministic metrics into a band.

    Every input here is arithmetic from the transcript. No model is asked
    to rate anything, which is what makes a sitting in September
    comparable with one in March rather than an opinion recorded twice.

    An all-or-nothing rule per band, rather than a weighted average: an
    average lets a fast talker buy a band with speed while still making a
    mistake every other sentence, and speed is the easiest of these to
    fake.
    """
    if metrics.words == 0:
        return SkillResult(skill=skill, band=None, raw=0, max=0,
                           detail={"reason": "nothing was said"})

    errors_per_100 = round(100 * len(findings) / metrics.words, 2)
    measured = {
        "words_per_minute": metrics.words_per_minute,
        "mean_utterance_words": metrics.mean_utterance_words,
        "lexical_diversity": metrics.lexical_diversity,
        "errors_per_100_words": errors_per_100,
        "crutches_per_minute": metrics.spanish_crutches_per_minute,
    }

    band = "A2"
    for candidate, floors in SPEAKING_BANDS:
        if _clears(floors, metrics, errors_per_100):
            band = candidate
            break

    return SkillResult(
        skill=skill,
        band=band,
        raw=BAND_ORDER.index(band) + 1,
        max=len(BAND_ORDER),
        detail=measured,
    )


def _clears(
    floors: dict[str, float], metrics: FluencyMetrics, errors_per_100: float
) -> bool:
    def at_least(name: str) -> bool:
        value = getattr(metrics, name)
        # An unmeasurable input cannot clear a floor. Treating None as
        # passing would hand out a band for a recording with no timings.
        return value is not None and value >= floors[name]

    crutches = metrics.spanish_crutches_per_minute
    return (
        at_least("words_per_minute")
        and at_least("mean_utterance_words")
        and at_least("lexical_diversity")
        and errors_per_100 <= floors["max_errors_per_100"]
        and (crutches is not None and crutches <= floors["max_crutches_per_minute"])
    )


def compare(earlier: Mapping[str, Any], later: Mapping[str, Any]) -> dict[str, Any]:
    """Two sittings, side by side, or a refusal.

    The refusal is the feature. Comparing sittings of two different exams
    produces a number that looks exactly like progress.
    """
    if earlier.get("fingerprint") != later.get("fingerprint"):
        return {
            "comparable": False,
            "reason": (
                "These sittings used different exams "
                f"({earlier.get('exam_version')}/{earlier.get('fingerprint')} "
                f"vs {later.get('exam_version')}/{later.get('fingerprint')}). "
                "The difference between them is not a measurement."
            ),
        }

    moves = {}
    for skill, before in earlier.get("bands", {}).items():
        after = later.get("bands", {}).get(skill)
        if before is None or after is None:
            moves[skill] = {"from": before, "to": after, "change": None}
            continue
        moves[skill] = {
            "from": before,
            "to": after,
            "change": BAND_ORDER.index(after) - BAND_ORDER.index(before),
        }
    return {"comparable": True, "skills": moves}
