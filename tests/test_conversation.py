"""The conversation loop, and the clock it has to beat.

The end-of-turn tests are the ones that matter. Cutting a speaker off
mid-hesitation is the single most discouraging thing a speaking tutor can
do, and the user of this one is a man reaching for words in his second
language -- so the rule is patient by design, and these tests pin that
rather than the other way round.
"""

import asyncio
from pathlib import Path

import pytest

from app.asr.base import Transcript, Word
from app.conversation.turn import (
    END_OF_TURN_SILENCE,
    LEG_BUDGET_MS,
    TURN_BUDGET_MS,
    ScriptedReplier,
    SilentVoice,
    Timing,
    is_end_of_turn,
    looks_like_a_pause,
    take_turn,
    trailing_silence,
    wav_of_silence,
)


def words(*tokens: str, gap: float = 0.1) -> tuple[Word, ...]:
    built = []
    clock = 0.0
    for token in tokens:
        built.append(Word(token, clock, clock + 0.3))
        clock += 0.3 + gap
    return tuple(built)


# --- knowing when he has finished ---------------------------------------


def test_a_long_silence_always_ends_the_turn():
    # Whatever the sentence looks like. Eventually the machine has to say
    # something.
    assert is_end_of_turn(words("and"), END_OF_TURN_SILENCE + 0.1) is True


def test_a_short_silence_never_ends_the_turn():
    assert is_end_of_turn(words("I", "deployed", "it"), 0.4) is False


def test_stopping_on_a_conjunction_is_not_the_end_of_a_turn():
    # He is looking for the next word, not waiting for an answer.
    assert is_end_of_turn(words("I", "deployed", "it", "because"), 1.0) is False


def test_a_finished_clause_ends_the_turn_sooner():
    assert is_end_of_turn(words("I", "deployed", "it"), 1.0) is True


def test_punctuation_does_not_change_the_decision():
    assert is_end_of_turn(words("it."), 1.0) == is_end_of_turn(words("it"), 1.0)


def test_silence_with_nothing_said_still_ends_eventually():
    assert is_end_of_turn((), END_OF_TURN_SILENCE + 0.1) is True
    assert is_end_of_turn((), 0.5) is False


def test_the_patience_threshold_is_longer_than_a_thinking_pause():
    from app.metrics.fluency import UTTERANCE_GAP

    # If these were the same number, the tutor would interrupt every time
    # he paused to think, which is most sentences.
    assert END_OF_TURN_SILENCE > UTTERANCE_GAP * 1.5


def test_trailing_silence_is_measured_from_the_last_word():
    assert trailing_silence(words("I", "deployed"), now_seconds=2.0) == (
        pytest.approx(1.3)
    )


def test_trailing_silence_of_an_empty_turn_is_zero():
    assert trailing_silence((), now_seconds=5.0) == 0.0


def test_a_turn_with_a_hesitation_in_it_is_recognisable():
    hesitant = words("I", "deployed", gap=1.0)

    assert looks_like_a_pause(hesitant) is True
    assert looks_like_a_pause(words("I", "deployed")) is False


# --- the budget ----------------------------------------------------------


def test_a_fast_turn_is_within_budget():
    timing = Timing(asr_ms=600, llm_ms=500, tts_ms=300)

    assert timing.total_ms == 1400
    assert timing.within_budget is True


def test_a_slow_turn_names_the_leg_that_spent_the_time():
    timing = Timing(asr_ms=2400, llm_ms=500, tts_ms=300)

    # "It is slow" does not tell anyone what to fix. The fix for a slow
    # recogniser is a smaller model; for a slow reply, a shorter prompt.
    assert timing.within_budget is False
    assert timing.worst_leg == "asr"


def test_the_legs_add_up_to_the_budget():
    assert sum(LEG_BUDGET_MS.values()) == TURN_BUDGET_MS


def test_a_timing_serialises_with_its_verdict():
    payload = Timing(asr_ms=100, llm_ms=100, tts_ms=100).as_dict()

    assert payload["within_budget"] is True
    assert payload["total_ms"] == 300


# --- the loop ------------------------------------------------------------


class InstantTranscriber:
    async def transcribe(self, audio: Path) -> Transcript:
        return Transcript(
            text="yesterday I go to the office",
            words=words(*"yesterday I go to the office".split()),
            duration=2.0,
        )


class SlowReplier:
    async def reply(self, history, message):
        await asyncio.sleep(0.05)
        return "Tell me more."


async def test_a_turn_comes_back_with_audio_and_a_measurement(tmp_path):
    audio = tmp_path / "turn.webm"
    audio.write_bytes(b"x")

    turn = await take_turn(
        audio,
        transcriber=InstantTranscriber(),
        replier=ScriptedReplier(),
        voice=SilentVoice(),
    )

    assert turn.heard.text.startswith("yesterday")
    assert turn.said in ScriptedReplier.OPENERS
    assert turn.audio.startswith(b"RIFF")
    assert turn.timing.total_ms >= 0


async def test_each_leg_is_timed_separately(tmp_path):
    audio = tmp_path / "turn.webm"
    audio.write_bytes(b"x")

    turn = await take_turn(
        audio,
        transcriber=InstantTranscriber(),
        replier=SlowReplier(),
        voice=SilentVoice(),
    )

    # The reply slept for 50ms and nothing else did. A single total would
    # have hidden which leg that was.
    assert turn.timing.llm_ms >= 45
    assert turn.timing.asr_ms < 40


async def test_the_scripted_replier_moves_the_conversation_on():
    replier = ScriptedReplier()

    first = await replier.reply([], "I deployed it yesterday")
    second = await replier.reply(
        [("user", "I deployed it"), ("assistant", first)], "it broke"
    )

    assert first != second


async def test_silence_gets_an_answer_rather_than_an_error():
    assert "again" in await ScriptedReplier().reply([], "   ")


async def test_the_offline_voice_produces_audio_a_browser_can_play():
    audio = await SilentVoice().speak("Tell me more about that.")

    assert audio[:4] == b"RIFF"
    assert audio[8:12] == b"WAVE"
    assert len(audio) > 44


async def test_a_longer_reply_takes_longer_to_say():
    short = await SilentVoice().speak("Yes.")
    long = await SilentVoice().speak(" ".join(["word"] * 40))

    assert len(long) > len(short)


def test_an_empty_utterance_still_makes_a_valid_wav():
    assert len(wav_of_silence(0.0)) == 44
