"""The review loop: when a mistake comes back, and what counts as knowing it.

The grading tests are the important ones. The grade decides the interval,
so a grader that is generous makes the whole schedule generous -- and the
most generous mistake available is to treat avoidance as success, which
is exactly what an intermediate speaker's strategy looks like from the
outside.
"""

from datetime import UTC, datetime, timedelta

import pytest

from app.asr.base import Transcript, Word
from app.assess.base import Finding
from app.review.practice import (
    AVOIDED,
    FLUENT,
    HESITANT,
    REPEATED,
    SAME_CATEGORY,
    exercise_for,
    grade_attempt,
)
from app.review.scheduler import MIN_EASE, SM2Scheduler
from app.store import (
    Session,
    StoredError,
    add_errors,
    connect,
    due_reviews,
    error_id_for,
    get_review,
    mastery,
    now,
    save_review,
    save_session,
    session_id_for,
)

MONDAY = datetime(2026, 3, 2, 20, 0, tzinfo=UTC)


@pytest.fixture
def scheduler() -> SM2Scheduler:
    return SM2Scheduler()


def days_between(state, start: datetime) -> float:
    return (datetime.fromisoformat(state.due_at) - start).total_seconds() / 86400


# --- the schedule --------------------------------------------------------


def test_a_new_error_is_due_immediately(scheduler):
    state = scheduler.first("e1", MONDAY)

    # Found this evening, worth one attempt this evening, while he still
    # remembers saying it.
    assert state.due_at == "2026-03-02T20:00:00+00:00"
    assert state.repetitions == 0


def test_the_first_two_intervals_are_sm2s_own(scheduler):
    first = scheduler.next(scheduler.first("e1", MONDAY), 5, MONDAY)
    second = scheduler.next(first, 5, MONDAY + timedelta(days=1))

    # One day and six days are the only empirical content in SM-2. Rounder
    # numbers would be prettier and would throw that away.
    assert days_between(first, MONDAY) == 1
    assert days_between(second, MONDAY + timedelta(days=1)) == 6


def test_the_interval_grows_by_the_ease_after_that(scheduler):
    state = scheduler.first("e1", MONDAY)
    for step in range(3):
        state = scheduler.next(state, 5, MONDAY + timedelta(days=step))

    assert state.interval_days == pytest.approx(6 * state.ease, rel=0.01)


def test_failing_sends_it_back_to_the_start(scheduler):
    state = scheduler.first("e1", MONDAY)
    for step in range(3):
        state = scheduler.next(state, 5, MONDAY + timedelta(days=step))
    failed = scheduler.next(state, 1, MONDAY + timedelta(days=30))

    assert failed.repetitions == 0
    assert failed.interval_days == 1


def test_failing_also_costs_ease_so_it_grows_back_more_slowly(scheduler):
    state = scheduler.next(scheduler.first("e1", MONDAY), 5, MONDAY)
    failed = scheduler.next(state, 1, MONDAY + timedelta(days=1))

    # Without this an error he keeps losing would return to the same long
    # intervals as one he has never missed.
    assert failed.ease < state.ease


def test_ease_has_a_floor(scheduler):
    state = scheduler.first("e1", MONDAY)
    for step in range(20):
        state = scheduler.next(state, 0, MONDAY + timedelta(days=step))

    # Below the floor a card returns every day forever, which is how a
    # review queue becomes something you stop opening.
    assert state.ease == MIN_EASE


def test_a_lapse_is_only_counted_on_something_he_had_learned(scheduler):
    new = scheduler.next(scheduler.first("e1", MONDAY), 1, MONDAY)
    learned = scheduler.next(scheduler.first("e2", MONDAY), 5, MONDAY)
    lost = scheduler.next(learned, 1, MONDAY + timedelta(days=1))

    assert new.lapses == 0
    assert lost.lapses == 1


@pytest.mark.parametrize("grade", [-3, 9])
def test_a_grade_outside_the_scale_is_clamped(scheduler, grade):
    state = scheduler.next(scheduler.first("e1", MONDAY), grade, MONDAY)

    assert MIN_EASE <= state.ease <= 3.0


# --- grading an attempt --------------------------------------------------


def spoken(text: str, *, gap: float = 0.1, corrections: bool = False) -> Transcript:
    words = []
    clock = 0.0
    for token in text.split():
        words.append(Word(text=token, start=clock, end=clock + 0.3))
        clock += 0.3 + gap
    return Transcript(text=text, words=tuple(words), duration=clock)


ERROR = StoredError(
    id="e1",
    session_id="s1",
    category="preposition",
    original="responsible of",
    correction="responsible for",
    explanation="'Responsible for', always.",
    detector="rules",
    created_at=now(),
)


def test_saying_it_right_without_hesitating_is_the_top_grade():
    exercise = exercise_for(ERROR)

    assert grade_attempt(
        exercise, spoken("I am responsible for the deployment pipeline")
    ) == FLUENT


def test_saying_it_right_after_a_restart_is_worth_less():
    exercise = exercise_for(ERROR)

    attempt = spoken("I am I am responsible for the pipeline")

    assert grade_attempt(exercise, attempt) == HESITANT


def test_a_long_pause_before_getting_it_right_is_worth_less():
    words = (
        Word("I", 0.0, 0.3), Word("am", 0.3, 0.6),
        Word("responsible", 2.2, 2.9), Word("for", 2.9, 3.1),
        Word("it", 3.1, 3.3),
    )
    attempt = Transcript(
        text="I am responsible for it", words=words, duration=3.3
    )

    assert grade_attempt(exercise_for(ERROR), attempt) == HESITANT


def test_making_the_same_mistake_word_for_word_is_the_lowest_pass_grade():
    assert grade_attempt(
        exercise_for(ERROR), spoken("I am responsible of the pipeline")
    ) == REPEATED


def test_a_different_wording_of_the_same_mistake_scores_just_above_it():
    findings = [Finding("preposition", "in charge of", "in charge of", "")]

    assert grade_attempt(
        exercise_for(ERROR), spoken("I am in charge of it"), findings
    ) == SAME_CATEGORY


def test_avoiding_the_structure_is_not_a_success():
    # He said nothing wrong and nothing right: the answer routes around
    # the structure. Scoring this as a pass is how a corpus quietly
    # forgets the thing he is hiding from.
    assert grade_attempt(
        exercise_for(ERROR), spoken("I take care of the deployment")
    ) == AVOIDED


def test_silence_is_not_avoidance():
    assert grade_attempt(exercise_for(ERROR), Transcript(text="", duration=4)) == 0


def test_the_exercise_changes_once_he_can_repair_it():
    repair = exercise_for(ERROR, repetitions=0)
    reuse = exercise_for(ERROR, repetitions=2)

    # Repeating the same repair teaches the sentence, not the structure.
    assert repair.kind == "repair"
    assert reuse.kind == "reuse"
    assert ERROR.correction in reuse.prompt


def test_an_exercise_carries_the_explanation_it_was_built_from():
    assert exercise_for(ERROR).explanation == ERROR.explanation


# --- persistence ---------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    connection = connect(tmp_path / "calliope.db")
    session = Session(
        id=session_id_for(b"audio"),
        kind="practice",
        recorded_at=now(),
        duration=10.0,
    )
    save_session(
        connection,
        session=session,
        audio_sha256=session.id,
        transcript=Transcript(text="I am responsible of it", duration=10.0),
        metrics={},
    )
    add_errors(
        connection,
        [
            StoredError(
                id=error_id_for(session.id, "preposition", "responsible of"),
                session_id=session.id,
                category="preposition",
                original="responsible of",
                correction="responsible for",
                explanation="",
                detector="rules",
                created_at=now(),
            )
        ],
    )
    yield connection
    connection.close()


def error_id(db) -> str:
    return db.execute("SELECT id FROM error").fetchone()["id"]


def test_reviewing_twice_leaves_one_schedule(db, scheduler):
    identifier = error_id(db)
    state = scheduler.first(identifier, MONDAY)
    save_review(db, state)
    save_review(db, scheduler.next(state, 5, MONDAY))

    # A schedule that accumulated rows would eventually disagree with
    # itself about when a card is due.
    assert db.execute("SELECT COUNT(*) AS n FROM review").fetchone()["n"] == 1
    assert get_review(db, identifier).repetitions == 1


def test_the_queue_holds_only_what_is_due(db, scheduler):
    identifier = error_id(db)
    save_review(db, scheduler.next(scheduler.first(identifier, MONDAY), 5, MONDAY))

    assert due_reviews(db, at=MONDAY.isoformat()) == []
    assert len(due_reviews(db, at=(MONDAY + timedelta(days=2)).isoformat())) == 1


def test_the_queue_returns_the_error_alongside_its_schedule(db, scheduler):
    identifier = error_id(db)
    save_review(db, scheduler.first(identifier, MONDAY))

    (error, state), = due_reviews(db, at=MONDAY.isoformat())

    assert error.correction == "responsible for"
    assert state.error_id == identifier


def test_an_error_with_no_schedule_is_visible_as_a_gap(db):
    # An unscheduled error is a bug in the pipeline, and a number nobody
    # can see is a bug nobody fixes.
    assert mastery(db)["unscheduled"] == 1


def test_mastery_separates_what_stuck_from_what_is_still_being_learned(
    db, scheduler
):
    identifier = error_id(db)
    state = scheduler.first(identifier, MONDAY)
    for step in range(5):
        state = scheduler.next(state, 5, MONDAY + timedelta(days=step))
    save_review(db, state)

    counts = mastery(db)

    assert counts["mature"] == 1
    assert counts["unscheduled"] == 0
