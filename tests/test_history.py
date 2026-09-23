"""Recurrence and retrieval, which answer two different questions.

"You have said this three times in two weeks" has to be exactly right,
because it is the sentence that sends him to practise something. It is
counted, not scored. Retrieval is the other half, it only ever feeds the
assessor's context, and its failure mode is the one tested hardest here:
a sentence must not be offered as evidence about itself.
"""

from datetime import UTC, datetime, timedelta

import pytest

from app.asr.base import Transcript, Word
from app.history import (
    RECENT_DAYS,
    Utterance,
    UtteranceIndex,
    build_index,
    habits,
    recurrence,
    utterances_of,
)
from app.metrics.fluency import UTTERANCE_GAP
from app.store import (
    Session,
    StoredError,
    add_errors,
    connect,
    error_id_for,
    save_session,
)

NOW = datetime(2026, 3, 20, 20, 0, tzinfo=UTC)


@pytest.fixture
def db(tmp_path):
    connection = connect(tmp_path / "calliope.db")
    yield connection
    connection.close()


def record(db, session_id: str, at: datetime, text: str, words=()) -> None:
    save_session(
        db,
        session=Session(
            id=session_id, kind="practice", recorded_at=at.isoformat(),
            duration=10.0,
        ),
        audio_sha256=session_id,
        transcript=Transcript(text=text, words=tuple(words), duration=10.0),
        metrics={},
    )


def log_error(db, session_id: str, at: datetime, phrase: str,
              category: str = "preposition") -> None:
    add_errors(
        db,
        [
            StoredError(
                id=error_id_for(session_id, category, phrase),
                session_id=session_id,
                category=category,
                original=phrase,
                correction="responsible for",
                explanation="",
                detector="rules",
                created_at=at.isoformat(),
            )
        ],
    )


# --- counting, not scoring ----------------------------------------------


def test_the_same_mistake_across_sessions_is_counted(db):
    for day, session in enumerate(["a", "b", "c"]):
        at = NOW - timedelta(days=day)
        record(db, session, at, "I am responsible of it")
        log_error(db, session, at, "responsible of")

    found = recurrence(db, "preposition", "responsible of", at=NOW)

    assert found.occurrences == 3
    assert found.is_a_habit


def test_punctuation_and_case_do_not_split_a_habit_in_two(db):
    for day, (session, phrase) in enumerate(
        [("a", "Responsible of,"), ("b", "responsible of"),
         ("c", "RESPONSIBLE OF")]
    ):
        at = NOW - timedelta(days=day)
        record(db, session, at, phrase)
        log_error(db, session, at, phrase)

    assert recurrence(db, "preposition", "responsible of", at=NOW).occurrences == 3


def test_an_old_mistake_falls_out_of_the_window(db):
    old = NOW - timedelta(days=RECENT_DAYS + 5)
    record(db, "old", old, "I am responsible of it")
    log_error(db, "old", old, "responsible of")

    # A habit he fixed two months ago should stop being quoted at him.
    assert recurrence(db, "preposition", "responsible of", at=NOW).occurrences == 0


def test_the_same_phrase_wrong_for_two_reasons_is_two_habits(db):
    record(db, "a", NOW, "I am responsible of it")
    log_error(db, "a", NOW, "responsible of", category="preposition")
    log_error(db, "a", NOW, "responsible of", category="word_choice")

    assert recurrence(db, "preposition", "responsible of", at=NOW).occurrences == 1
    assert recurrence(db, "word_choice", "responsible of", at=NOW).occurrences == 1


def test_twice_is_not_yet_a_habit(db):
    for day, session in enumerate(["a", "b"]):
        at = NOW - timedelta(days=day)
        record(db, session, at, "text")
        log_error(db, session, at, "responsible of")

    assert not recurrence(db, "preposition", "responsible of", at=NOW).is_a_habit


def test_habits_come_back_worst_first(db):
    for day, session in enumerate(["a", "b", "c", "d"]):
        at = NOW - timedelta(days=day)
        record(db, session, at, "text")
        log_error(db, session, at, "responsible of")
        if day < 3:
            log_error(db, session, at, "depend of")

    ranked = habits(db, at=NOW)

    assert [item.phrase for item in ranked] == ["responsible of", "depend of"]


def test_a_quiet_fortnight_produces_no_habits(db):
    assert habits(db, at=NOW) == []


# --- splitting sessions into utterances ---------------------------------


def test_a_session_is_split_on_silence_not_stored_whole():
    words = (
        Word("I", 0.0, 0.3), Word("deployed", 0.4, 0.9),
        Word("then", 0.9 + UTTERANCE_GAP + 0.1, 2.0),
        Word("it", 2.0, 2.2), Word("broke", 2.2, 2.6),
    )
    transcript = Transcript(text="I deployed then it broke", words=words,
                            duration=2.6)

    # Retrieval over whole sessions returns two minutes of speech to
    # illustrate one phrase.
    assert [u.text for u in utterances_of(transcript, "s1", NOW.isoformat())] == [
        "I deployed",
        "then it broke",
    ]


def test_a_transcript_without_timings_is_one_utterance():
    transcript = Transcript(text="I deployed it", duration=5.0)

    assert len(utterances_of(transcript, "s1", NOW.isoformat())) == 1


def test_an_empty_transcript_contributes_nothing():
    assert utterances_of(Transcript(text="  ", duration=5.0), "s", "t") == []


# --- retrieval -----------------------------------------------------------


def index_of(*pairs: tuple[str, str]) -> UtteranceIndex:
    index = UtteranceIndex()
    index.add(
        [Utterance(session, NOW.isoformat(), text) for session, text in pairs]
    )
    return index


def test_a_sentence_is_not_evidence_about_itself():
    said_today = "the payment queue started duplicating charges again"
    index = index_of(
        ("today", said_today),
        ("old", "the payment queue was duplicating charges last month"),
    )

    without_guard = index.search(said_today)
    with_guard = index.search(said_today, exclude_session="today")

    # Without the guard the best match for today's sentence is today's
    # sentence, and the model is handed proof of a habit on the first
    # occasion he ever said it.
    assert without_guard[0][0].session_id == "today"
    assert [utterance.session_id for utterance, _ in with_guard] == ["old"]


def test_an_unrelated_question_returns_nothing():
    index = index_of(("a", "the deployment pipeline runs on Thursday"))

    # A model handed three irrelevant sentences will use them.
    assert index.search("my brother lives in Medellin") == []


def test_a_query_of_nothing_but_function_words_matches_nothing():
    index = index_of(("a", "I think that it is in the office and I am there"))

    # Without IDF and a stopword list, "and then it was in the" would be a
    # strong match for every sentence in the corpus.
    assert index.search("and then it was in the") == []


def test_a_real_overlap_is_found():
    index = index_of(
        ("a", "the payment queue started duplicating charges last night"),
        ("b", "I ate lunch with my sister"),
    )

    (utterance, score), = index.search("the payment queue duplicated charges")

    assert utterance.session_id == "a"
    assert score > 0.2


def test_an_empty_index_answers_nothing():
    assert UtteranceIndex().search("anything") == []


def test_an_empty_query_answers_nothing():
    assert index_of(("a", "the payment queue")).search("") == []


def test_the_index_is_built_from_what_is_stored(db):
    record(
        db, "s1", NOW, "I am responsible of the pipeline",
        words=[Word(token, i * 0.4, i * 0.4 + 0.3)
               for i, token in enumerate("I am responsible of the pipeline".split())],
    )

    index = build_index(db)

    assert len(index) == 1
