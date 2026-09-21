"""The web layer: a few pages, served by FastAPI, with no build step.

The frontend is HTML rendered on the server and one page of vanilla
JavaScript for the microphone. React here would add a toolchain, a second
language and a deployment story to a single-user tool whose whole job is
to record audio and draw four lines. The value of this project is not in
the interface, and an interface with a build step is one more thing that
can stop working between him and a recording.

The charts are inline SVG generated in Python. Same reason: a charting
library is 300KB to draw a line through twenty points.

Everything slow happens in the pipeline, not here. A route's job is to
take the upload, hand it over, and redirect.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from fastapi import APIRouter, FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.asr.base import ScriptedTranscriber, TranscriptionError
from app.assess.base import NullAssessor
from app.assess.rules import RuleBasedAssessor
from app.config import Settings, settings
from app.history import habits
from app.metrics.fluency import TRACKED_TENSES, metrics_from
from app.pipeline import MergingAssessor, process, store_recording
from app.placement.exam import (
    EXAM_DIR,
    compare,
    load_exam,
    score_choices,
    score_speaking,
)
from app.review.practice import exercise_for, grade_attempt
from app.review.scheduler import SM2Scheduler
from app.store import (
    category_counts,
    connect,
    due_reviews,
    errors_for,
    get_error,
    get_review,
    get_session,
    get_transcript,
    mastery,
    metric_series,
    now,
    placements,
    save_placement,
    save_review,
    session_detail,
    sessions,
)

logger = logging.getLogger(__name__)

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

#: What the dashboard charts. Each of these is arithmetic over a
#: transcript, which is what makes a line through them mean anything.
CHARTED = (
    ("words_per_minute", "Words per minute"),
    ("mean_utterance_words", "Words per utterance"),
    ("lexical_diversity", "Lexical diversity"),
    ("fillers_per_minute", "Fillers per minute"),
    ("spanish_crutches_per_minute", "Spanish per minute"),
    ("self_corrections_per_minute", "Restarts per minute"),
)


def build_transcriber(config: Settings):
    """Whisper if it is installed and wanted, fixtures otherwise.

    The fallback is silent in the logs and loud at the point of use: a
    recording it cannot transcribe raises with an explanation instead of
    returning invented text.
    """
    fixtures = Path(config.transcript_fixtures) if config.transcript_fixtures else None
    if config.transcriber != "whisper":
        return ScriptedTranscriber(fixtures)
    try:
        import faster_whisper  # noqa: F401

        from app.asr.whisper import FasterWhisperTranscriber

        return FasterWhisperTranscriber(config.whisper_model)
    except ImportError:
        logger.warning(
            "faster-whisper is not installed; falling back to scripted "
            "transcripts. pip install -r requirements-asr.txt"
        )
        return ScriptedTranscriber(fixtures)


def build_assessor(config: Settings):
    rules = RuleBasedAssessor()
    if config.assessor == "none":
        return NullAssessor()
    if config.assessor == "rules" or not config.anthropic_api_key:
        return rules

    from app.assess.claude import ClaudeAssessor

    model = ClaudeAssessor(config.anthropic_api_key, config.model)
    return model if config.assessor == "claude" else MergingAssessor([rules, model])


def create_app(config: Settings | None = None, **parts) -> FastAPI:
    """Build the app, with every collaborator replaceable.

    A factory rather than a module-level singleton, because the tests
    need a different database and a transcriber that does not need half a
    gigabyte of weights -- and a web layer that can only be exercised
    against the real Whisper is a web layer that does not get exercised.
    """
    chosen = config or settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = chosen
        # Only close what this app opened. A connection handed in belongs
        # to whoever handed it in, and closing it out from under them is
        # the kind of ownership bug that only shows up in someone else's
        # test.
        borrowed = parts.get("db")
        app.state.db = borrowed or connect(chosen.db_path)
        app.state.transcriber = parts.get("transcriber") or build_transcriber(
            chosen
        )
        app.state.assessor = parts.get("assessor") or build_assessor(chosen)
        app.state.scheduler = parts.get("scheduler") or SM2Scheduler()
        app.state.exam = load_exam(
            EXAM_DIR / f"exam_{chosen.exam_version}.json"
        )
        try:
            yield
        finally:
            if borrowed is None:
                app.state.db.close()

    built = FastAPI(title="DandeLyon Calliope", lifespan=lifespan)
    built.include_router(router)
    return built


router = APIRouter()


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    config: Settings = request.app.state.settings
    return JSONResponse(
        {
            "status": "ok",
            "transcriber": getattr(
                request.app.state.transcriber, "name", "unknown"
            ),
            "assessor": getattr(request.app.state.assessor, "name", "unknown"),
            # Stated rather than assumed: "nothing leaves the machine" is
            # a claim the user should be able to check.
            "sends_text_off_the_machine": config.uses_a_model,
        }
    )


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    db = request.app.state.db
    charts = [
        (label, sparkline(metric_series(db, name)), metric_series(db, name))
        for name, label in CHARTED
    ]
    recent = sessions(db, limit=10)
    detail = session_detail(db, recent[0].id) if recent else None

    return TEMPLATES.TemplateResponse(
        request,
        "dashboard.html",
        {
            "charts": charts,
            "sessions": recent,
            "habits": habits(db),
            "categories": category_counts(db),
            "mastery": mastery(db),
            "due": len(due_reviews(db, at=now(), limit=100)),
            "avoided": _avoided(detail),
            "placements": placements(db),
            "settings": request.app.state.settings,
        },
    )


def _avoided(detail: dict | None) -> list[str]:
    if not detail:
        return list(TRACKED_TENSES)
    return list(detail.get("tenses_absent") or ())


@router.get("/record", response_class=HTMLResponse)
async def record(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "record.html",
        {"prompts": request.app.state.exam.section("speaking").items},
    )


@router.post("/sessions")
async def upload(
    request: Request, audio: UploadFile, prompt_id: str = Form(default="")
):
    payload = await audio.read()
    if not payload:
        return JSONResponse({"error": "empty recording"}, status_code=400)

    config: Settings = request.app.state.settings
    path = store_recording(
        config.recordings_dir, audio.filename or "blob.webm", payload
    )

    try:
        result = await process(
            request.app.state.db,
            audio=path,
            transcriber=request.app.state.transcriber,
            assessor=request.app.state.assessor,
            scheduler=request.app.state.scheduler,
            prompt_id=prompt_id or None,
        )
    except TranscriptionError as exc:
        # The recording is already on disk, so nothing is lost: install
        # the ASR extra and re-upload the same file to get the same id.
        return JSONResponse({"error": str(exc)}, status_code=503)

    return RedirectResponse(f"/sessions/{result.session_id}", status_code=303)


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_page(request: Request, session_id: str):
    db = request.app.state.db
    stored = get_session(db, session_id)
    if stored is None:
        return HTMLResponse("No such session.", status_code=404)

    return TEMPLATES.TemplateResponse(
        request,
        "session.html",
        {
            "session": stored,
            "transcript": get_transcript(db, session_id),
            "metrics": session_detail(db, session_id) or {},
            "errors": errors_for(db, session_id),
        },
    )


@router.get("/review", response_class=HTMLResponse)
async def review(request: Request):
    db = request.app.state.db
    queue = due_reviews(db, at=now(), limit=20)
    exercises = [
        exercise_for(error, state.repetitions) for error, state in queue
    ]
    return TEMPLATES.TemplateResponse(
        request,
        "review.html",
        {"exercises": exercises, "remaining": len(queue)},
    )


@router.post("/review/{error_id}")
async def review_attempt(request: Request, error_id: str, audio: UploadFile):
    db = request.app.state.db
    state = get_review(db, error_id)
    if state is None:
        return JSONResponse({"error": "not scheduled"}, status_code=404)

    error = get_error(db, error_id)
    if error is None:
        return JSONResponse({"error": "no such error"}, status_code=404)

    config: Settings = request.app.state.settings
    payload = await audio.read()
    path = store_recording(
        config.recordings_dir / "review", audio.filename or "blob.webm", payload
    )

    try:
        transcript = await request.app.state.transcriber.transcribe(path)
    except TranscriptionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    exercise = exercise_for(error, state.repetitions)
    findings = await request.app.state.assessor.assess(transcript)
    grade = grade_attempt(exercise, transcript, findings)

    save_review(
        db,
        request.app.state.scheduler.next(state, grade, datetime.now(UTC)),
    )
    return JSONResponse(
        {
            "grade": grade,
            "heard": transcript.text,
            "target": exercise.target,
            "explanation": exercise.explanation,
            "next_due": get_review(db, error_id).due_at,
        }
    )


@router.get("/placement", response_class=HTMLResponse)
async def placement(request: Request):
    exam = request.app.state.exam
    return TEMPLATES.TemplateResponse(
        request,
        "placement.html",
        {
            "exam": exam,
            "sittings": placements(request.app.state.db),
        },
    )


@router.post("/placement")
async def placement_submit(request: Request):
    """Score the written sections and store the sitting.

    Spoken production is not scored here. It is recorded through the same
    upload path as everything else, so that the baseline recording is
    measured by exactly the same code as a Tuesday evening -- which is the
    only way the two are comparable six months from now.
    """
    exam = request.app.state.exam
    db = request.app.state.db
    form = await request.form()

    answers = {
        key: int(value)
        for key, value in form.items()
        if key.startswith(("g", "v", "l")) and str(value).isdigit()
    }
    results = {
        section.skill: score_choices(section, answers)
        for section in exam.sections
        if section.kind in ("choice", "listening")
    }

    spoken = sessions(db, kind="placement", limit=1)
    if spoken:
        # Read back rather than re-measured: the baseline has to stay
        # exactly as it was taken, or there is no baseline.
        detail = session_detail(db, spoken[0].id)
        if detail:
            results["speaking"] = score_speaking(
                metrics_from(detail), errors_for(db, spoken[0].id)
            )

    taken_at = now()
    payload = {
        "fingerprint": exam.fingerprint,
        "bands": {skill: result.band for skill, result in results.items()},
        "detail": {
            skill: {"raw": result.raw, "max": result.max, **result.detail}
            for skill, result in results.items()
        },
    }
    # A sitting is identified by the exam, the day and the answers. Two
    # submissions of the same answers on the same day are one sitting --
    # the submit button gets pressed twice -- while the same answers six
    # months later are a second sitting, which is the comparison this
    # whole section exists for.
    signature = sha256(
        f"{exam.fingerprint}|{sorted(answers.items())}".encode()
    ).hexdigest()[:8]

    save_placement(
        db,
        placement_id=f"{exam.version}-{taken_at[:10]}-{signature}",
        exam_version=exam.version,
        taken_at=taken_at,
        payload=payload,
    )

    history = placements(db)
    against = (
        compare(history[0], history[-1]) if len(history) > 1 else {"comparable": None}
    )
    return JSONResponse({"bands": payload["bands"], "compared": against})


def sparkline(
    series: list[tuple[str, float]], width: int = 260, height: int = 48
) -> str:
    """A line through the points, as inline SVG.

    Returns an empty string for fewer than two points. One measurement is
    not a trend, and drawing a flat line through it would suggest it is.
    """
    if len(series) < 2:
        return ""
    values = [value for _, value in series]
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    step = width / (len(values) - 1)
    points = " ".join(
        f"{index * step:.1f},{height - (value - low) / span * (height - 6) - 3:.1f}"
        for index, value in enumerate(values)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="spark" '
        f'preserveAspectRatio="none" role="img">'
        f'<polyline points="{points}" fill="none" stroke="currentColor" '
        f'stroke-width="2" /></svg>'
    )


app = create_app()
