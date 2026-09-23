"""The placement exam: repeatable first, accurate second.

The absolute band this produces is an opinion. That two sittings six
months apart produce comparable bands is not, and it is the only property
the project actually needs -- so most of these tests are about the exam
not moving, and about the comparison refusing to run when it has.
"""

import json

import pytest

from app.asr.base import Transcript, Word
from app.assess.base import Finding
from app.metrics.fluency import measure
from app.placement.exam import (
    EXAM_DIR,
    band_for,
    compare,
    fingerprint_of,
    load_exam,
    score_choices,
    score_speaking,
)

EXAM_V1 = EXAM_DIR / "exam_v1.json"


@pytest.fixture
def exam():
    return load_exam(EXAM_V1)


# --- the exam is a fixed artifact ---------------------------------------


def test_the_fingerprint_is_stable_across_loads():
    assert load_exam(EXAM_V1).fingerprint == load_exam(EXAM_V1).fingerprint


def test_editing_a_single_item_changes_the_fingerprint(tmp_path):
    raw = json.loads(EXAM_V1.read_text(encoding="utf-8"))
    before = fingerprint_of(raw)
    raw["sections"][0]["items"][0]["answer"] = 2

    # This is what makes "we fixed the exam" a checkable claim rather than
    # an intention.
    assert fingerprint_of(raw) != before


def test_reordering_the_file_does_not_change_the_fingerprint():
    raw = json.loads(EXAM_V1.read_text(encoding="utf-8"))
    reordered = {"sections": raw["sections"], "version": raw["version"],
                 "about": raw["about"]}

    # Key order in JSON is not content. A fingerprint that moved when
    # someone reformatted the file would be ignored within a week.
    assert fingerprint_of(reordered) == fingerprint_of(raw)


def test_every_item_has_a_unique_id(exam):
    ids = [item.id for section in exam.sections for item in section.items]

    assert len(set(ids)) == len(ids)


def test_every_answer_points_at_an_option_that_exists(exam):
    for section in exam.sections:
        if section.kind not in ("choice", "listening"):
            continue
        for item in section.items:
            assert item.answer is not None, item.id
            assert 0 <= item.answer < len(item.options), item.id
            assert len(item.options) >= 3, item.id


def test_the_correct_answer_is_not_always_in_the_same_place(exam):
    positions = {
        item.answer
        for section in exam.sections
        if section.kind in ("choice", "listening")
        for item in section.items
    }

    # A test where the answer is always B measures pattern spotting.
    assert len(positions) >= 3


def test_every_listening_item_carries_its_script(exam):
    for item in exam.section("listening").items:
        assert item.script, item.id


def test_every_speaking_prompt_is_timed(exam):
    for item in exam.section("speaking").items:
        assert item.seconds and item.seconds >= 60, item.id


def test_the_exam_covers_four_skills_separately(exam):
    # One number would average an uneven profile into something that
    # describes nobody, and it would rise when the strong half improved.
    assert set(exam.skills) == {
        "grammar", "lexical_range", "listening", "speaking"
    }


# --- scoring choices -----------------------------------------------------


def test_the_same_answers_give_the_same_band_every_time(exam):
    answers = {item.id: 0 for item in exam.section("grammar").items}

    first = score_choices(exam.section("grammar"), answers)
    second = score_choices(exam.section("grammar"), answers)

    assert first == second


def test_a_perfect_section_is_the_top_band(exam):
    section = exam.section("grammar")
    answers = {item.id: item.answer for item in section.items}

    assert score_choices(section, answers).band == "C1"


def test_unanswered_items_are_not_counted_as_wrong(exam):
    section = exam.section("grammar")
    first_two = section.items[:2]
    answers = {item.id: item.answer for item in first_two}

    result = score_choices(section, answers)

    # Half an exam is a partial measurement. Marking the unanswered half
    # as failure turns it into a wrong one.
    assert (result.raw, result.max) == (2, 2)
    assert result.detail["of"] == len(section.items)


def test_a_section_nobody_touched_has_no_band(exam):
    result = score_choices(exam.section("listening"), {})

    # Not zero. Zero is a result; this is the absence of one.
    assert result.band is None
    assert result.percentage is None


@pytest.mark.parametrize(
    ("percentage", "band"),
    [(100, "C1"), (80, "C1"), (79.9, "B2"), (60, "B2"), (41, "B1"), (0, "A2")],
)
def test_band_thresholds(percentage, band):
    assert band_for(percentage) == band


# --- scoring speech ------------------------------------------------------


def said(text: str, seconds: float, *, gap: float = 0.1) -> Transcript:
    tokens = text.split()
    words = []
    clock = 0.0
    step = max((seconds - 0.3) / max(len(tokens), 1), 0.05)
    for token in tokens:
        words.append(Word(text=token, start=clock, end=clock + step * 0.7))
        clock += step
    return Transcript(text=text, words=tuple(words), duration=seconds)


FLUENT_ANSWER = (
    "Yesterday I spent most of the morning tracing a problem in the "
    "payment queue, because the retry logic had started duplicating "
    "charges whenever the provider timed out. I reproduced it locally, "
    "wrote a failing test, and shipped a fix before lunch. The afternoon "
    "went into reviewing two pull requests from my team and preparing "
    "the notes for tomorrow's architecture discussion, which I expect "
    "will be difficult because we still disagree about the database."
)


def test_a_strong_answer_with_no_errors_reaches_a_high_band():
    metrics = measure(said(FLUENT_ANSWER, seconds=35))

    result = score_speaking(metrics, findings=[])

    assert result.band in ("B2", "C1")


def test_speed_alone_does_not_buy_a_band():
    metrics = measure(said(FLUENT_ANSWER, seconds=25))
    errors = [
        Finding("verb_tense", "I go", "I went", "") for _ in range(6)
    ]

    # An average over the criteria would let a fast talker buy a band
    # with the one thing on this list that is easiest to fake.
    assert score_speaking(metrics, errors).band == "A2"


def test_silence_has_no_band():
    result = score_speaking(measure(Transcript(text="", duration=90)), [])

    assert result.band is None
    assert "nothing was said" in result.detail["reason"]


def test_a_recording_without_timings_cannot_clear_a_floor():
    metrics = measure(Transcript(text=FLUENT_ANSWER, duration=0))

    # Words per minute is None here. Treating an unmeasurable input as
    # passing would hand out a band for a recording nobody could measure.
    assert score_speaking(metrics, []).band == "A2"


def test_the_detail_shows_which_numbers_produced_the_band():
    metrics = measure(said(FLUENT_ANSWER, seconds=35))

    detail = score_speaking(metrics, []).detail

    assert set(detail) == {
        "words_per_minute", "mean_utterance_words", "lexical_diversity",
        "errors_per_100_words", "crutches_per_minute",
    }


# --- comparing two sittings ---------------------------------------------


def sitting(fingerprint: str, **bands):
    return {
        "exam_version": "v1",
        "fingerprint": fingerprint,
        "bands": bands,
    }


def test_two_sittings_of_the_same_exam_can_be_subtracted():
    result = compare(
        sitting("abc", speaking="B1", grammar="B2"),
        sitting("abc", speaking="B2", grammar="B2"),
    )

    assert result["comparable"] is True
    assert result["skills"]["speaking"]["change"] == 1
    assert result["skills"]["grammar"]["change"] == 0


def test_two_sittings_of_different_exams_are_refused():
    result = compare(
        sitting("abc", speaking="B1"), sitting("xyz", speaking="B2")
    )

    # The refusal is the feature. A comparison across two different exams
    # produces a number that looks exactly like progress.
    assert result["comparable"] is False
    assert "not a measurement" in result["reason"]


def test_a_skill_missing_from_one_sitting_has_no_change():
    result = compare(
        sitting("abc", speaking="B1", listening="B1"),
        sitting("abc", speaking="B2"),
    )

    assert result["skills"]["listening"]["change"] is None
