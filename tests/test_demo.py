"""The sample week, which is the first thing anyone sees.

It runs on every container start, so the test that matters is that
running it twice is the same as running it once. The rest is making sure
the demo is real arithmetic over invented input rather than a mock-up:
the same pipeline, the same metrics, the same scheduler.
"""


import pytest

from app.demo import SAMPLES, seed, transcript_of
from app.metrics.fluency import measure
from app.store import all_errors, connect, metric_series, sessions


@pytest.fixture
def db(tmp_path):
    connection = connect(tmp_path / "calliope.db")
    yield connection
    connection.close()


async def test_the_week_lands_as_seven_sessions(db, tmp_path):
    added = await seed(db, tmp_path / "audio")

    assert added == 7
    assert len(sessions(db)) == 7


async def test_seeding_twice_adds_nothing(db, tmp_path):
    await seed(db, tmp_path / "audio")
    again = await seed(db, tmp_path / "audio")

    # It runs on every container start.
    assert again == 0
    assert len(sessions(db)) == 7


async def test_the_week_draws_a_line(db, tmp_path):
    await seed(db, tmp_path / "audio")

    series = metric_series(db, "words_per_minute")

    assert len(series) == 7
    assert [when for when, _ in series] == sorted(when for when, _ in series)


async def test_the_sample_mistakes_reach_the_corpus(db, tmp_path):
    await seed(db, tmp_path / "audio")

    categories = {error.category for error in all_errors(db)}

    assert {"preposition", "verb_tense", "article"} <= categories


def test_timings_are_generated_the_same_way_every_time():
    entry = {"text": "I am responsible of it", "pace": 0.4, "pauses": [2]}

    assert transcript_of(entry) == transcript_of(entry)


def test_a_pause_shows_up_in_the_measurement():
    entry = {"text": "I deployed it and then it broke", "pace": 0.3,
             "pauses": [3]}

    metrics = measure(transcript_of(entry))

    assert metrics.pauses.count == 1
    assert metrics.utterances == 2


def test_the_sample_file_says_what_it_is():
    import json

    about = " ".join(json.loads(SAMPLES.read_text(encoding="utf-8"))["about"])

    # Published transcripts have to be written for the purpose, and the
    # file has to say so where anyone reading it will see it.
    assert "NOT recordings of the user" in about
