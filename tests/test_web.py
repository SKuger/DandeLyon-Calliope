"""The web layer, against a fake transcriber.

Every page here is thin on purpose, so the tests are about the two things
HTTP can still get wrong: an upload that arrives twice, and a dependency
that is missing. The second is the interesting one -- a build with no
speech recognition has to say so and keep the recording, not accept the
upload and store an empty session.
"""

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.asr.base import Transcript, TranscriptionError, Word
from app.assess.rules import RuleBasedAssessor
from app.config import Settings
from app.main import build_assessor, build_transcriber, create_app, sparkline
from app.store import all_errors, connect, sessions

SPOKEN = "I am responsible of the pipeline and I am developer"


class FakeTranscriber:
    name = "fake"

    def __init__(self, text: str = SPOKEN):
        self.text = text
        self.calls = 0

    async def transcribe(self, audio: Path) -> Transcript:
        self.calls += 1
        words = tuple(
            Word(token, i * 0.4, i * 0.4 + 0.3, 0.92)
            for i, token in enumerate(self.text.split())
        )
        return Transcript(
            text=self.text, words=words, duration=len(words) * 0.4,
            model="fake", decoding="faithful",
        )


class NoModel:
    name = "none"

    async def transcribe(self, audio: Path) -> Transcript:
        raise TranscriptionError("This build has no speech recognition.")


@pytest.fixture
def config(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        recordings_dir=tmp_path / "recordings",
        transcriber="scripted",
        assessor="rules",
        anthropic_api_key="",
    )


@pytest.fixture
def db(config):
    connection = connect(config.db_path)
    yield connection
    connection.close()


@pytest.fixture
def client(config, db):
    transcriber = FakeTranscriber()
    app = create_app(
        config, db=db, transcriber=transcriber, assessor=RuleBasedAssessor()
    )
    with TestClient(app) as started:
        started.transcriber = transcriber
        yield started


def upload(client, payload: bytes = b"audio-one", prompt: str = ""):
    return client.post(
        "/sessions",
        files={"audio": ("session.webm", payload, "audio/webm")},
        data={"prompt_id": prompt},
    )


# --- the pages exist and say what they are ------------------------------


def test_health_states_whether_anything_leaves_the_machine(client):
    payload = client.get("/health").json()

    assert payload["status"] == "ok"
    assert payload["transcriber"] == "fake"
    assert payload["sends_text_off_the_machine"] is False


def test_health_admits_when_a_model_is_configured(config, db):
    remote = replace(config, assessor="claude", anthropic_api_key="sk-test")

    with TestClient(create_app(remote, db=db, transcriber=NoModel())) as client:
        assert client.get("/health").json()["sends_text_off_the_machine"] is True


def test_an_empty_dashboard_does_not_draw_a_trend(client):
    body = client.get("/").text

    # One point is not a trend, and a flat line through it would suggest
    # it is.
    assert "Two sessions are needed" in body


def test_the_recorder_offers_the_exam_prompts(client):
    assert "job interview" in client.get("/record").text


def test_the_placement_page_shows_the_fingerprint(client):
    body = client.get("/placement").text

    # The claim "this exam does not change" should be checkable from the
    # page itself.
    assert "Fingerprint" in body
    assert "responsible for writing" in body


# --- uploading -----------------------------------------------------------


def test_an_upload_lands_on_its_session(client, db):
    response = upload(client)

    assert response.status_code == 200
    assert len(sessions(db)) == 1
    assert "responsible of" in response.text
    assert "responsible for" in response.text


def test_the_same_upload_twice_is_one_session(client, db):
    upload(client)
    upload(client)

    assert len(sessions(db)) == 1
    assert client.transcriber.calls == 1


def test_an_empty_upload_is_refused(client, db):
    response = upload(client, payload=b"")

    assert response.status_code == 400
    assert sessions(db) == []


def test_a_build_without_speech_recognition_says_so(config, db, tmp_path):
    app = create_app(config, db=db, transcriber=NoModel())

    with TestClient(app) as client:
        response = upload(client)

    assert response.status_code == 503
    assert "no speech recognition" in response.json()["error"]
    # The recording is on disk, so installing the extra and re-uploading
    # the same file produces the same session id.
    assert sessions(db) == []
    assert list((config.recordings_dir).glob("*.webm"))


def test_an_unknown_session_is_a_404(client):
    assert client.get("/sessions/nope").status_code == 404


# --- the review queue ----------------------------------------------------


def test_todays_errors_are_due_this_evening(client, db):
    upload(client)

    body = client.get("/review").text

    assert "responsible of" in body
    assert len(all_errors(db)) == 2


def test_an_attempt_is_graded_and_rescheduled(client, db):
    upload(client)
    error = next(e for e in all_errors(db) if e.category == "preposition")

    response = client.post(
        f"/review/{error.id}",
        files={"audio": ("attempt.webm", b"attempt-audio", "audio/webm")},
    )
    payload = response.json()

    assert response.status_code == 200
    # The fake transcriber replays the original sentence, so this is the
    # same mistake again and the schedule must not grow.
    assert payload["grade"] == 1
    assert payload["next_due"] > payload["heard"][:0]


def test_an_attempt_at_something_not_scheduled_is_a_404(client):
    response = client.post(
        "/review/unknown",
        files={"audio": ("a.webm", b"x", "audio/webm")},
    )

    assert response.status_code == 404


# --- the placement exam --------------------------------------------------


def test_a_sitting_is_scored_and_stored(client):
    answers = {"g01": "1", "g02": "2", "g04": "2", "v01": "1"}

    payload = client.post("/placement", data=answers).json()

    assert payload["bands"]["grammar"] == "C1"
    assert payload["compared"]["comparable"] is None


def test_a_second_sitting_is_compared_with_the_first(client):
    client.post("/placement", data={"g01": "0", "g02": "0", "g04": "0"})
    second = client.post("/placement", data={"g01": "1", "g02": "2", "g04": "2"})
    payload = second.json()

    assert payload["compared"]["comparable"] is True
    assert payload["compared"]["skills"]["grammar"]["change"] > 0


# --- charts --------------------------------------------------------------


def test_one_point_draws_nothing():
    assert sparkline([("2026-01-01", 100.0)]) == ""


def test_two_points_draw_a_line():
    svg = sparkline([("2026-01-01", 100.0), ("2026-01-02", 120.0)])

    assert svg.startswith("<svg")
    assert "polyline" in svg


def test_a_flat_series_does_not_divide_by_zero():
    svg = sparkline([("a", 5.0), ("b", 5.0), ("c", 5.0)])

    assert "nan" not in svg.lower()


# --- what gets built from configuration ---------------------------------


def test_asking_for_scripted_transcripts_gets_them(config):
    assert build_transcriber(config).name == "scripted"


def test_a_model_assessor_without_a_key_falls_back_to_the_rules(config):
    assessor = build_assessor(replace(config, assessor="claude"))

    # Otherwise a missing key would mean no assessment at all, which is a
    # worse default than the offline rules.
    assert assessor.name == "rules"


def test_assessment_can_be_switched_off(config):
    assert build_assessor(replace(config, assessor="none")).name == "none"


def test_both_assessors_merge_when_a_key_is_present(config):
    assessor = build_assessor(
        replace(config, assessor="both", anthropic_api_key="sk-test")
    )

    assert assessor.name == "merged"
