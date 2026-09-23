"""Spoken conversation, and the two seconds it has to fit into.

Above roughly two seconds a spoken exchange stops feeling like talking
and starts feeling like filling in a form, so latency is the requirement
here rather than a quality. That makes the interesting object not the
dialogue but the stopwatch: every turn is timed per leg -- recognition,
reply, speech -- and a turn that misses the budget says which leg spent
the time.

**What is not here.** The three legs are not streamed. Each waits for the
one before it to finish, which is the simplest thing that works and also
the slowest. The measurements in `docs/latency.md` are of this path, and
they say plainly which leg would have to be streamed first. Publishing a
number for a pipeline that does not exist yet would be worse than
publishing a slow one that does.

**End of turn.** Deciding that someone has stopped speaking is its own
problem: cut too early and you interrupt a man thinking of a word in his
second language, which is precisely the user. The rule here is
deliberately patient and the threshold is a constant with an argument
attached.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.asr.base import Transcriber, Transcript, Word
from app.metrics.fluency import UTTERANCE_GAP

#: The whole budget for one turn, measured from the moment he stops
#: speaking to the moment the reply starts playing.
TURN_BUDGET_MS = 2000

#: How it is meant to be spent. Recognition gets the most because it is
#: the only leg that cannot start before he stops talking.
LEG_BUDGET_MS = {"asr": 900, "llm": 700, "tts": 400}

#: Silence that ends a turn. Longer than the utterance boundary used for
#: the metrics, and deliberately so: at 0.7s this would cut him off in
#: the middle of reaching for a word, which is the single most
#: discouraging thing a speaking tutor can do. The cost of being patient
#: is half a second of dead air; the cost of being eager is that he stops
#: using it.
END_OF_TURN_SILENCE = 1.4

#: Unless the turn is obviously finished. A trailing question or a full
#: clause followed by a shorter silence is a turn.
SETTLED_SILENCE = 0.9


class Replier(Protocol):
    async def reply(
        self, history: Sequence[tuple[str, str]], message: str
    ) -> str: ...


class Voice(Protocol):
    async def speak(self, text: str) -> bytes: ...


@dataclass(frozen=True)
class Timing:
    asr_ms: int
    llm_ms: int
    tts_ms: int

    @property
    def total_ms(self) -> int:
        return self.asr_ms + self.llm_ms + self.tts_ms

    @property
    def within_budget(self) -> bool:
        return self.total_ms <= TURN_BUDGET_MS

    @property
    def worst_leg(self) -> str:
        """Which leg is furthest over its share. The number a README
        should quote, because "it is slow" does not tell anyone what to
        fix."""
        overruns = {
            leg: getattr(self, f"{leg}_ms") / budget
            for leg, budget in LEG_BUDGET_MS.items()
        }
        return max(overruns, key=overruns.get)

    def as_dict(self) -> dict[str, int | bool | str]:
        return {
            "asr_ms": self.asr_ms,
            "llm_ms": self.llm_ms,
            "tts_ms": self.tts_ms,
            "total_ms": self.total_ms,
            "within_budget": self.within_budget,
            "worst_leg": self.worst_leg,
        }


@dataclass(frozen=True)
class Turn:
    heard: Transcript
    said: str
    audio: bytes
    timing: Timing


def is_end_of_turn(words: Sequence[Word], silence: float) -> bool:
    """Has he finished, or is he thinking?

    Two rules. Any silence past `END_OF_TURN_SILENCE` ends the turn,
    whatever the sentence looks like -- eventually the machine has to say
    something. Below that, a shorter silence only ends the turn if what
    came before sounds finished, which here means it did not stop on a
    word that obviously expects more.
    """
    if not words:
        return silence >= END_OF_TURN_SILENCE
    if silence >= END_OF_TURN_SILENCE:
        return True
    if silence < SETTLED_SILENCE:
        return False

    last = words[-1].text.strip().strip(".,?!").lower()
    # Stopping on one of these is someone mid-sentence looking for the
    # next word, not someone waiting for an answer.
    dangling = {
        "and", "but", "or", "so", "because", "that", "which", "to", "of",
        "for", "with", "the", "a", "an", "my", "is", "was", "in", "on",
        "at", "like", "um", "uh", "er",
    }
    return last not in dangling


class ScriptedReplier:
    """The offline default: short, deterministic, and always a question.

    Not a tutor. It exists so the conversation loop runs with no key and
    so the latency of the other two legs can be measured without a
    network call in the middle of the number. Corrections do not happen
    here -- they happen after the session, from the transcript, where
    they can be checked.
    """

    name = "scripted"

    OPENERS = (
        "Tell me more about that.",
        "What happened next?",
        "Why do you think that was?",
        "How would you do it differently?",
        "And how did that turn out?",
    )

    async def reply(
        self, history: Sequence[tuple[str, str]], message: str
    ) -> str:
        if not message.strip():
            return "I did not catch that. Could you say it again?"
        turns = sum(1 for role, _ in history if role == "user")
        return self.OPENERS[turns % len(self.OPENERS)]


class SilentVoice:
    """A valid WAV of the right length, with nothing in it.

    The offline default. It produces real audio a browser will play, so
    the loop and its timings are exercised end to end without a speech
    model -- and it is obviously silent, so nobody mistakes it for
    working text to speech.
    """

    name = "silent"

    SAMPLE_RATE = 16000
    #: Roughly conversational pace, so a turn that would be too long to
    #: listen to is too long here too.
    WORDS_PER_SECOND = 2.6

    async def speak(self, text: str) -> bytes:
        seconds = max(len(text.split()) / self.WORDS_PER_SECOND, 0.3)
        return wav_of_silence(seconds, self.SAMPLE_RATE)


def wav_of_silence(seconds: float, sample_rate: int = 16000) -> bytes:
    """A minimal 16-bit mono WAV. Written by hand to avoid a dependency
    for forty-four bytes of header."""
    frames = int(seconds * sample_rate)
    data = b"\x00\x00" * frames
    header = (
        b"RIFF"
        + (36 + len(data)).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + (sample_rate * 2).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + len(data).to_bytes(4, "little")
    )
    return header + data


async def take_turn(
    audio: Path,
    *,
    transcriber: Transcriber,
    replier: Replier,
    voice: Voice,
    history: Sequence[tuple[str, str]] = (),
) -> Turn:
    """One exchange, timed leg by leg.

    The clock is read between the legs rather than around the whole call,
    because a total is not actionable: the fix for a slow recogniser is a
    smaller model and the fix for a slow reply is a shorter prompt.
    """
    started = time.perf_counter()
    heard = await transcriber.transcribe(audio)
    after_asr = time.perf_counter()

    said = await replier.reply(list(history), heard.text)
    after_llm = time.perf_counter()

    spoken = await voice.speak(said)
    after_tts = time.perf_counter()

    return Turn(
        heard=heard,
        said=said,
        audio=spoken,
        timing=Timing(
            asr_ms=round((after_asr - started) * 1000),
            llm_ms=round((after_llm - after_asr) * 1000),
            tts_ms=round((after_tts - after_llm) * 1000),
        ),
    )


def trailing_silence(words: Sequence[Word], now_seconds: float) -> float:
    """How long since the last word ended. Zero if nothing was said."""
    if not words:
        return 0.0
    return max(0.0, now_seconds - words[-1].end)


def looks_like_a_pause(words: Sequence[Word]) -> bool:
    """Whether the speaker has already paused inside this turn.

    Used to widen patience: someone who has hesitated twice already is
    likely to hesitate again, and cutting him off on the third is how a
    tool gets abandoned.
    """
    return any(
        later.start - earlier.end >= UTTERANCE_GAP
        for earlier, later in zip(words, words[1:], strict=False)
    )
