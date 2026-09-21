"""Run the matrix: every voice, every model size, every decoding setting.

Output is an append-only JSONL of rows plus a regenerated summary, and a
row already present is not recomputed. That is not a convenience: the
full matrix is a few hundred transcriptions and takes half an hour on
CPU, so a run that cannot resume is a run nobody repeats, and an
experiment nobody repeats is an anecdote.

Every row carries the transcript verbatim. The aggregate numbers are
what the document quotes, but the `other` rows are only interpretable by
reading what the transcriber actually heard, and a summary that throws
those away cannot be argued with.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.asr.base import WHISPER_DEFAULTS, DecodeOptions, Transcript
from app.asr.whisper import FasterWhisperTranscriber
from experiments.whisper_fidelity.score import (
    Outcome,
    Sample,
    classify,
    load_samples,
    span_confidence,
    word_error_rate,
)
from experiments.whisper_fidelity.synthesize import DEFAULT_OUT, FIXTURE

RESULTS = Path(__file__).resolve().parent / "results"

# Deliberately disfluent and ungrammatical, and deliberately sharing no
# content words with the corpus. Priming the decoder with the actual test
# sentences would let it copy the error out of the prompt, which would
# measure nothing except that Whisper can read.
LEARNER_PROMPT = (
    "Uh, so, last week I go for the doctor, and, um, my brother have "
    "thirty years, he is teacher in a school, no?"
)

CONFIGS: tuple[DecodeOptions, ...] = (
    WHISPER_DEFAULTS,
    DecodeOptions(name="faithful"),
    DecodeOptions(name="faithful-prompted", initial_prompt=LEARNER_PROMPT),
)

MODELS = ("base.en", "small.en", "medium.en")


@dataclass(frozen=True)
class Key:
    voice: str
    model: str
    config: str
    sample: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.voice, self.model, self.config, self.sample)


def row_for(
    key: Key, sample: Sample, transcript: Transcript, seconds: float
) -> dict:
    outcome = classify(sample, transcript.text)

    # Confidence is only interesting where the text changed, so it is
    # read at the error site: the planted span when it survived, the
    # grammatical span when it was repaired.
    if outcome is Outcome.SURVIVED:
        span = sample.error
    elif outcome is Outcome.REPAIRED:
        span = next(
            (
                form
                for form in sample.repaired
                if span_confidence(transcript, form) is not None
            ),
            None,
        )
    else:
        span = None

    scores = span_confidence(transcript, span) if span else None
    confidences = [
        word.confidence
        for word in transcript.words
        if word.confidence is not None
    ]

    return {
        "voice": key.voice,
        "model": key.model,
        "config": key.config,
        "sample": key.sample,
        "category": sample.category,
        "outcome": str(outcome),
        "spoken": sample.spoken,
        "heard": transcript.text,
        "wer": round(word_error_rate(sample.spoken, transcript.text), 4),
        "span": span,
        "span_confidence": (
            None if scores is None else [round(s, 4) for s in scores]
        ),
        "mean_confidence": (
            None
            if not confidences
            else round(sum(confidences) / len(confidences), 4)
        ),
        "audio_seconds": round(transcript.duration, 2),
        "decode_seconds": round(seconds, 2),
    }


def existing(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.is_file():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done.add((row["voice"], row["model"], row["config"], row["sample"]))
    return done


def build_transcriber(model_size: str, config: DecodeOptions):
    return FasterWhisperTranscriber(model_size, decode=config)


async def run(
    audio_root: Path,
    samples: tuple[Sample, ...],
    models: tuple[str, ...],
    configs: tuple[DecodeOptions, ...],
    rows_path: Path,
    voices: list[str] | None = None,
    build=build_transcriber,
) -> int:
    voice_dirs = sorted(
        directory
        for directory in audio_root.iterdir()
        if directory.is_dir() and (not voices or directory.name in voices)
    )
    if not voice_dirs:
        raise SystemExit(
            f"No audio under {audio_root}. Run "
            f"`python -m experiments.whisper_fidelity.synthesize` first, or "
            f"put real recordings there as <voice>/<sample id>.wav."
        )

    done = existing(rows_path)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    for model_size in models:
        for config in configs:
            # One model instance per (model, config): loading weights is
            # seconds and the decoder is stateless between calls.
            transcriber = build(model_size, config)
            for voice_dir in voice_dirs:
                for sample in samples:
                    key = Key(
                        voice_dir.name, model_size, config.name, sample.id
                    )
                    if key.as_tuple() in done:
                        continue
                    audio = voice_dir / f"{sample.id}.wav"
                    if not audio.is_file():
                        continue

                    started = time.monotonic()
                    transcript = await transcriber.transcribe(audio)
                    row = row_for(
                        key, sample, transcript, time.monotonic() - started
                    )
                    with rows_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                    done.add(key.as_tuple())
                    written += 1
                    print(
                        f"{key.voice:15} {key.model:10} {key.config:18} "
                        f"{key.sample:28} {row['outcome']}"
                    )
    return written


def rows_of(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(rows: list[dict]) -> str:
    """The tables the document quotes, regenerated from the rows."""
    planted = [row for row in rows if row["category"] != "control"]
    controls = [row for row in rows if row["category"] == "control"]

    lines = [
        "# Whisper grammar fidelity: results",
        "",
        "Generated by `python -m experiments.whisper_fidelity.run "
        "--summarize`. Do not edit by hand.",
        "",
        f"{len(rows)} transcriptions: {len(planted)} planted errors, "
        f"{len(controls)} controls.",
        "",
        "## Planted errors: did the mistake survive?",
        "",
        "`survived` means the transcript still contains the mistake, so a "
        "tutor can see it. `repaired` means Whisper wrote the grammatical "
        "form. `other` means it heard a different sentence altogether.",
        "",
        "| voice | model | decoding | survived | repaired | other | mean WER |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    def group(rows: list[dict], *fields: str) -> dict[tuple, list[dict]]:
        grouped: dict[tuple, list[dict]] = {}
        for row in rows:
            grouped.setdefault(tuple(row[f] for f in fields), []).append(row)
        return dict(sorted(grouped.items()))

    for (voice, model, config), group_rows in group(
        planted, "voice", "model", "config"
    ).items():
        counts = Counter(row["outcome"] for row in group_rows)
        total = len(group_rows)
        wer = sum(row["wer"] for row in group_rows) / total
        lines.append(
            f"| {voice} | {model} | {config} | "
            f"{counts['survived']}/{total} | {counts['repaired']}/{total} | "
            f"{counts['other']}/{total} | {wer:.2f} |"
        )

    lines += [
        "",
        "## Controls: did the transcriber invent an error?",
        "",
        "These sentences are already correct. `altered` is a transcript "
        "that differs from what was said, which is how a tutor ends up "
        "correcting a mistake the user never made.",
        "",
        "| voice | model | decoding | clean | altered | mean WER |",
        "|---|---|---|---:|---:|---:|",
    ]
    for (voice, model, config), group_rows in group(
        controls, "voice", "model", "config"
    ).items():
        counts = Counter(row["outcome"] for row in group_rows)
        total = len(group_rows)
        wer = sum(row["wer"] for row in group_rows) / total
        lines.append(
            f"| {voice} | {model} | {config} | {counts['clean']}/{total} | "
            f"{counts['altered']}/{total} | {wer:.2f} |"
        )

    lines += [
        "",
        "## Per-word confidence at the error site",
        "",
        "The brief's second method: a word Whisper invented to repair a "
        "sentence should be less certain than the words it heard. Lowest "
        "confidence in the span, averaged over the rows of each outcome.",
        "",
        "| model | outcome | rows | mean lowest confidence | mean transcript |",
        "|---|---|---:|---:|---:|",
    ]
    for (model, outcome), group_rows in group(
        [row for row in planted if row["span_confidence"]],
        "model",
        "outcome",
    ).items():
        lowest = [min(row["span_confidence"]) for row in group_rows]
        overall = [
            row["mean_confidence"]
            for row in group_rows
            if row["mean_confidence"] is not None
        ]
        lines.append(
            f"| {model} | {outcome} | {len(group_rows)} | "
            f"{sum(lowest) / len(lowest):.3f} | "
            f"{(sum(overall) / len(overall)) if overall else float('nan'):.3f} |"
        )

    lines += [
        "",
        "## By error category, best configuration",
        "",
        "Accented voices only (`es-mx-*`), `faithful` decoding, since that "
        "is the condition the product actually runs in.",
        "",
        "| category | survived | repaired | other |",
        "|---|---:|---:|---:|",
    ]
    accented = [
        row
        for row in planted
        if row["voice"].startswith("es-mx") and row["config"] == "faithful"
    ]
    for (category,), group_rows in group(accented, "category").items():
        counts = Counter(row["outcome"] for row in group_rows)
        total = len(group_rows)
        lines.append(
            f"| {category} | {counts['survived']}/{total} | "
            f"{counts['repaired']}/{total} | {counts['other']}/{total} |"
        )

    lines += [
        "",
        "## Every transcript that was neither survived nor repaired",
        "",
        "Reported in full because the aggregate cannot be interpreted "
        "without them.",
        "",
    ]
    for row in planted:
        if row["outcome"] != "other":
            continue
        lines.append(
            f"- `{row['voice']}` / `{row['model']}` / `{row['config']}` / "
            f"`{row['sample']}`  \n"
            f"  said: {row['spoken']}  \n"
            f"  heard: {row['heard']}"
        )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--voice", action="append")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Regenerate the summary from the rows already collected.",
    )
    args = parser.parse_args(argv)

    rows_path = args.out / "rows.jsonl"
    if not args.summarize:
        written = asyncio.run(
            run(
                audio_root=args.audio,
                samples=load_samples(args.fixture),
                models=tuple(args.model or MODELS),
                configs=CONFIGS,
                rows_path=rows_path,
                voices=args.voice,
            )
        )
        print(f"\n{written} new transcriptions")

    summary = args.out / "summary.md"
    summary.write_text(summarize(rows_of(rows_path)), encoding="utf-8")
    print(f"summary -> {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
