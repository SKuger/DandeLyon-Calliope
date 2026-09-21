"""Seed the database with the sample week, so the dashboard has a shape.

`docker compose up` on an empty database shows six empty cards and a
sentence saying two sessions are needed before a line means anything.
That is correct behaviour and a bad first impression, so this loads the
seven invented sessions in `samples/week.json`.

Two things it is careful about. It runs them through the same pipeline as
a real recording -- same metrics, same assessor, same scheduler -- so the
demo dashboard is real arithmetic over fake input rather than a mock-up.
And it is idempotent, because it runs on every container start: the fake
audio for a session is derived from its text, so the content-addressed
session id is stable and a second run inserts nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.asr.base import Transcript, Word
from app.assess.rules import RuleBasedAssessor
from app.config import settings
from app.pipeline import process
from app.review.scheduler import SM2Scheduler
from app.store import connect, sessions

SAMPLES = Path(__file__).resolve().parent.parent / "samples" / "week.json"


def transcript_of(entry: dict) -> Transcript:
    """Deterministic word timings from a pace and a list of pause points.

    Generated rather than measured, which is the honest version of a
    sample: the timings are made up, and they are made up the same way
    every time, so the demo numbers do not wander between runs.
    """
    tokens = entry["text"].split()
    pace = float(entry["pace"])
    pauses = set(entry.get("pauses", ()))

    words: list[Word] = []
    clock = 0.0
    for index, token in enumerate(tokens):
        if index in pauses:
            clock += 1.1
        words.append(Word(token, clock, clock + pace * 0.75, 0.93))
        clock += pace

    return Transcript(
        text=entry["text"],
        words=tuple(words),
        duration=round(clock, 2),
        model="sample",
        decoding="sample",
    )


class SampleTranscriber:
    """Replays a prepared transcript. Keyed by the audio's stem, like the
    scripted transcriber, so the pipeline cannot tell the difference."""

    name = "sample"

    def __init__(self, by_stem: dict[str, Transcript]) -> None:
        self._by_stem = by_stem

    async def transcribe(self, audio: Path) -> Transcript:
        return self._by_stem[audio.stem]


async def seed(connection, directory: Path, samples: Path = SAMPLES) -> int:
    entries = json.loads(samples.read_text(encoding="utf-8"))["sessions"]
    directory.mkdir(parents=True, exist_ok=True)

    prepared: dict[str, Transcript] = {}
    plan = []
    for entry in entries:
        transcript = transcript_of(entry)
        # The "audio" is the text. That makes the session id a function of
        # the sample file, so re-running this changes nothing.
        payload = entry["text"].encode("utf-8")
        path = directory / f"sample-{entry['day']}.wav"
        path.write_bytes(payload)
        prepared[path.stem] = transcript
        plan.append((entry, path))

    transcriber = SampleTranscriber(prepared)
    assessor = RuleBasedAssessor()
    scheduler = SM2Scheduler()
    # Backdated so the week reads left to right, ending today.
    today = datetime.now(UTC).replace(hour=20, minute=0, second=0, microsecond=0)

    added = 0
    for entry, path in plan:
        result = await process(
            connection,
            audio=path,
            transcriber=transcriber,
            assessor=assessor,
            scheduler=scheduler,
            prompt_id=entry.get("prompt_id"),
            at=today - timedelta(days=6 - entry["day"]),
        )
        added += 0 if result.already_processed else 1
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Seed even if the database already has sessions.")
    args = parser.parse_args(argv)

    connection = connect(settings.db_path)
    try:
        if sessions(connection, limit=1) and not args.force:
            print("Already has sessions; leaving it alone.")
            return 0
        added = asyncio.run(
            seed(connection, settings.recordings_dir / "samples")
        )
        print(f"Seeded {added} sample sessions into {settings.db_path}")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
