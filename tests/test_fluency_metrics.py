"""The deterministic metrics, and the inputs that break naive versions.

Three groups of assertions here matter more than the arithmetic:

- `None` where nothing was measured, never 0.0. A flat line at zero on a
  six-month chart is indistinguishable from a real result.
- Length invariance for lexical diversity. The obvious implementation
  makes a longer answer look like a smaller vocabulary, which would tell
  the user to talk less.
- The known limits of the tense heuristic, pinned. It is pattern
  matching; the tests say where it is wrong so nobody discovers it by
  trusting it.
"""

import pytest

from app.asr.base import Transcript, Word
from app.metrics.fluency import (
    MATTR_WINDOW,
    UTTERANCE_GAP,
    measure,
    moving_average_ttr,
    split_utterances,
    tenses_in,
    tokens_of,
)


def spoken(text: str, *, gap: float = 0.1, word_seconds: float = 0.3,
           duration: float | None = None) -> Transcript:
    """A transcript with evenly spaced word timings."""
    words = []
    clock = 0.0
    for token in text.split():
        words.append(Word(text=token, start=clock, end=clock + word_seconds))
        clock += word_seconds + gap
    return Transcript(
        text=text,
        words=tuple(words),
        duration=duration if duration is not None else max(clock, 0.0),
    )


def with_gaps(*pairs: tuple[str, float]) -> Transcript:
    """A transcript where each word carries the gap that precedes it."""
    words = []
    clock = 0.0
    for token, gap in pairs:
        clock += gap
        words.append(Word(text=token, start=clock, end=clock + 0.3))
        clock += 0.3
    return Transcript(
        text=" ".join(token for token, _ in pairs),
        words=tuple(words),
        duration=clock,
    )


# --- nothing measurable --------------------------------------------------


def test_silence_does_not_divide_by_zero():
    metrics = measure(Transcript(text="", duration=0.0))

    assert metrics.words == 0
    assert metrics.words_per_minute is None
    assert metrics.pauses is None
    assert metrics.lexical_diversity is None


def test_a_transcript_with_no_duration_reports_no_rate():
    # It happened in practice: a backend that returns text without
    # timings. Words per minute with a zero denominator is either a crash
    # or a lie, and the third option is to say so.
    metrics = measure(Transcript(text="I work with Python", duration=0.0))

    assert metrics.words == 4
    assert metrics.words_per_minute is None
    assert metrics.fillers_per_minute is None


def test_a_text_only_transcript_still_yields_vocabulary_and_tenses():
    metrics = measure(Transcript(text="I have finished the migration", duration=30))

    # Word timings are what pauses need, not what vocabulary needs. A
    # backend without them should lose one metric, not all of them.
    assert metrics.pauses is None
    assert metrics.words_per_minute == 10.0
    assert "present_perfect" in metrics.tenses_used


def test_a_single_word_has_no_pause_distribution():
    assert measure(spoken("yes")).pauses is None


# --- utterances ----------------------------------------------------------


def test_utterances_split_on_silence_not_on_punctuation():
    # The punctuation in a Whisper transcript is the same language model
    # this project distrusts, guessing at sentence boundaries. The silence
    # is in the audio.
    transcript = with_gaps(
        ("I", 0.0), ("deployed", 0.1), ("it.", 0.1),
        ("Then", UTTERANCE_GAP + 0.1), ("it", 0.1), ("broke.", 0.1),
    )

    assert [len(group) for group in split_utterances(transcript.words)] == [3, 3]


def test_a_pause_shorter_than_the_boundary_keeps_one_utterance():
    transcript = with_gaps(("I", 0.0), ("deployed", UTTERANCE_GAP - 0.2))

    assert len(split_utterances(transcript.words)) == 1


def test_mean_utterance_length_counts_words_not_characters():
    transcript = with_gaps(
        ("one", 0.0), ("two", 0.1),
        ("three", UTTERANCE_GAP + 0.1), ("four", 0.1), ("five", 0.1), ("six", 0.1),
    )

    assert measure(transcript).mean_utterance_words == 3.0


# --- pauses --------------------------------------------------------------


def test_pauses_inside_a_clause_are_counted_separately():
    transcript = with_gaps(
        ("I", 0.0), ("need", 0.4), ("the", 0.1), ("logs", 0.1),
        ("because", UTTERANCE_GAP + 0.5), ("it", 0.1), ("failed", 0.1),
    )

    profile = measure(transcript).pauses

    # A speaker pausing inside a clause is assembling it word by word; one
    # pausing between clauses is planning the next thought. Collapsing
    # them into one number hides the difference that matters.
    assert profile.count == 2
    assert profile.within_utterance == 1
    assert profile.longest_seconds == pytest.approx(UTTERANCE_GAP + 0.5)


def test_fluent_speech_reports_zero_pauses_rather_than_none():
    # Zero is a real measurement here, unlike the None cases above.
    profile = measure(spoken("I deployed the new version", gap=0.05)).pauses

    assert profile.count == 0
    assert profile.mean_seconds is None


# --- lexical diversity ---------------------------------------------------


def test_lexical_diversity_does_not_fall_just_because_he_talked_longer():
    vocabulary = [f"word{n}" for n in range(MATTR_WINDOW * 2)]
    short = vocabulary[:MATTR_WINDOW]
    long = vocabulary + vocabulary  # same vocabulary, four times the words

    # The plain ratio would drop from 1.0 to 0.5 here and the user would
    # read it as losing vocabulary for speaking for longer.
    assert moving_average_ttr(short) == 1.0
    assert moving_average_ttr(long) == 1.0


def test_repetition_does_lower_diversity():
    assert moving_average_ttr(["the"] * 200) == pytest.approx(1 / MATTR_WINDOW)


def test_a_sample_shorter_than_the_window_uses_the_plain_ratio():
    assert moving_average_ttr(["a", "b", "b"]) == pytest.approx(0.6667, abs=1e-4)


# --- hesitation ----------------------------------------------------------


def test_fillers_are_counted_per_minute():
    transcript = Transcript(text="uh I um think er yes", duration=60)

    assert measure(transcript).fillers_per_minute == 3.0


def test_like_is_not_counted_as_a_filler():
    transcript = Transcript(text="I like Python and I like Go", duration=60)

    # "like" is a filler in "it was, like, broken" and a verb here, and
    # telling them apart needs a parser. It is reported as a discourse
    # marker so that the filler count stays trustworthy.
    metrics = measure(transcript)

    assert metrics.fillers_per_minute == 0.0
    assert metrics.discourse_markers_per_minute == 2.0


def test_spanish_leaking_through_is_its_own_number():
    transcript = Transcript(
        text="the queue was, este, how do you say, backed up", duration=60
    )

    # Running out of English mid-sentence is a different event from
    # hesitating in English, and it is the one he wants to watch fall.
    assert measure(transcript).spanish_crutches_per_minute == 2.0


def test_a_repeated_word_counts_as_one_self_correction():
    assert measure(spoken("I I deployed the service")).self_corrections == 1


def test_a_repeated_filler_is_hesitation_not_a_restart():
    assert measure(spoken("uh uh I deployed it")).self_corrections == 0


def test_an_explicit_repair_marker_counts():
    assert measure(spoken("I deployed sorry I reverted it")).self_corrections == 1


def test_a_word_repeated_across_an_utterance_boundary_is_not_a_restart():
    transcript = with_gaps(
        ("yes", 0.0), ("yes", UTTERANCE_GAP + 0.2),
    )

    # Two separate answers, not a stutter. Counting it would inflate the
    # metric for anyone who speaks in short turns.
    assert measure(transcript).self_corrections == 0


# --- tenses --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I deployed it yesterday", "past_simple"),
        ("I went to the office", "past_simple"),
        ("I am working on the API", "present_continuous"),
        ("I have finished the migration", "present_perfect"),
        ("I have never seen that error", "present_perfect"),
        ("I had already deployed it", "past_perfect"),
        ("I was testing the queue", "past_continuous"),
        ("I will fix it tomorrow", "future_will"),
        ("I am going to refactor this", "future_going_to"),
        ("I would use Redis for that", "conditional"),
        ("I can review the pull request", "modal"),
    ],
)
def test_each_tracked_tense_is_recognised(text, expected):
    assert expected in tenses_in(tokens_of(text))


def test_had_without_a_participle_is_not_the_past_perfect():
    assert tenses_in(tokens_of("I had lunch with the team")) == {"past_simple"}


def test_avoided_tenses_are_reported_as_absent():
    metrics = measure(Transcript(text="I work with Python every day", duration=30))

    # Avoidance is how an intermediate speaker hides a gap, so the tenses
    # he never reached for are as informative as the ones he used.
    assert "present_simple" in metrics.tenses_used
    assert "present_perfect" in metrics.tenses_absent
    assert "past_perfect" in metrics.tenses_absent


def test_the_heuristic_over_reports_the_present_simple():
    # Pinned rather than hidden. A bare verb and a noun look the same
    # without parsing, so a sentence with no auxiliary anywhere is read
    # as the present simple. It inflates one bar on the chart and never
    # affects the others.
    assert tenses_in(tokens_of("more information about the incident")) == {
        "present_simple"
    }


def test_no_words_means_no_tenses():
    assert tenses_in([]) == set()


# --- the whole thing -----------------------------------------------------


def test_metrics_serialise_for_storage():
    payload = measure(spoken("I deployed the service this morning")).as_dict()

    assert payload["words"] == 6
    assert payload["pauses"]["count"] == 0
    assert isinstance(payload["tenses_used"], tuple)
