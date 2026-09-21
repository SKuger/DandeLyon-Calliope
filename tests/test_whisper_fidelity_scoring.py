"""The scorer, and the fixture it scores against.

The scorer is the measuring instrument of the experiment the whole
project depends on, so it is tested harder than the experiment itself.
An instrument that quietly counts a surviving error as a repair would
make the finding say the opposite of the truth.
"""

from pathlib import Path

import pytest

from app.asr.base import Transcript, Word
from app.text import (
    find_span,
    normalize,
    span_confidence,
    spell_number,
    word_error_rate,
)
from experiments.whisper_fidelity.score import (
    Outcome,
    Sample,
    classify,
    load_samples,
)

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "whisper_fidelity"
    / "fixtures"
    / "planted_errors.json"
)


# --- normalisation -------------------------------------------------------


def test_case_and_punctuation_do_not_change_a_transcript():
    assert normalize("Yesterday, I go!") == normalize("yesterday i go")


def test_hyphenation_does_not_change_a_transcript():
    assert normalize("front-end") == "front end"


def test_digits_are_spelled_out_so_a_survival_is_not_read_as_a_repair():
    # The speaker says "twenty eight"; Whisper writes "28". Without this,
    # "i have twenty eight years" would not be found in its own
    # transcript and a surviving error would be scored as repaired.
    assert normalize("I have 28 years") == "i have twenty eight years"


@pytest.mark.parametrize(
    ("digits", "spelled"),
    [("0", "zero"), ("9", "nine"), ("19", "nineteen"), ("40", "forty"),
     ("28", "twenty eight")],
)
def test_small_numbers_are_spelled(digits, spelled):
    assert spell_number(digits) == spelled


@pytest.mark.parametrize("token", ["2021", "100", "nine", ""])
def test_anything_ambiguous_is_left_alone(token):
    # "2021" is read aloud two different ways. Picking one would inject
    # the very error this function exists to remove.
    assert spell_number(token) is None


def test_apostrophes_survive_normalisation():
    assert normalize("I don’t have") == "i don't have"


# --- classification ------------------------------------------------------


PLANTED = Sample(
    id="t",
    category="verb_tense",
    spoken="Yesterday I go to the office.",
    error="yesterday i go",
    repaired=("yesterday i went",),
)

CONTROL = Sample(
    id="c",
    category="control",
    spoken="This morning I reviewed two pull requests.",
)


def test_a_surviving_error_is_scored_as_survived():
    assert classify(PLANTED, "Yesterday I go to the office.") is Outcome.SURVIVED


def test_a_repaired_error_is_scored_as_repaired():
    assert classify(PLANTED, "Yesterday I went to the office.") is Outcome.REPAIRED


def test_a_different_sentence_is_neither():
    # This is the outcome the accented recordings mostly produce, and
    # collapsing it into "repaired" would overstate the problem.
    assert classify(PLANTED, "Just a day ago to the office.") is Outcome.OTHER


def test_a_transcript_holding_both_forms_counts_as_survived():
    both = "Yesterday I go, sorry, yesterday I went to the office."

    # The mistake is visible to the assessor, which is the only property
    # anything downstream depends on.
    assert classify(PLANTED, both) is Outcome.SURVIVED


def test_silence_is_not_a_repair():
    assert classify(PLANTED, "") is Outcome.OTHER


def test_a_control_transcribed_as_spoken_is_clean():
    assert classify(CONTROL, "This morning, I reviewed 2 pull requests!") is (
        Outcome.CLEAN
    )


def test_a_control_the_transcriber_changed_is_altered():
    # The transcriber invented "review". A tutor built on this would
    # correct a mistake the user never made.
    assert classify(CONTROL, "This morning I review two pull requests.") is (
        Outcome.ALTERED
    )


# --- word error rate -----------------------------------------------------


def test_an_exact_transcript_has_no_word_errors():
    assert word_error_rate("I go to the office", "I go to the office.") == 0.0


def test_one_substitution_in_five_words():
    assert word_error_rate("I go to the office", "I got to the office") == (
        pytest.approx(0.2)
    )


def test_an_empty_transcript_loses_every_word():
    assert word_error_rate("I go to the office", "") == 1.0


def test_nothing_said_and_nothing_heard_is_not_an_error():
    assert word_error_rate("", "") == 0.0


# --- confidence over a span ----------------------------------------------


def transcript_of(*pairs: tuple[str, float]) -> Transcript:
    words = tuple(
        Word(text=text, start=index * 0.5, end=index * 0.5 + 0.4,
             confidence=confidence)
        for index, (text, confidence) in enumerate(pairs)
    )
    return Transcript(
        text=" ".join(text for text, _ in pairs),
        words=words,
        duration=len(pairs) * 0.5,
    )


def test_a_span_is_located_across_a_spelled_out_number():
    # "28" is one word in the transcript and two tokens after
    # normalisation. The mapping back to word indices is the part that
    # breaks if this is written naively.
    transcript = transcript_of(("I", 0.9), ("have", 0.9), ("28", 0.7),
                               ("years", 0.8))

    assert find_span(transcript, "have twenty eight years") == (1, 4)


def test_confidence_is_read_from_the_words_of_the_span():
    transcript = transcript_of(("yesterday", 0.95), ("I", 0.99), ("went", 0.31))

    assert span_confidence(transcript, "I went") == (0.99, 0.31)


def test_a_span_that_is_not_there_has_no_confidence():
    transcript = transcript_of(("yesterday", 0.95), ("I", 0.99), ("went", 0.31))

    assert span_confidence(transcript, "I go") is None


def test_a_backend_without_confidences_reports_none_rather_than_zero():
    transcript = Transcript(
        text="I went",
        words=(Word("I", 0.0, 0.1), Word("went", 0.1, 0.4)),
    )

    assert span_confidence(transcript, "I went") is None


# --- the fixture itself --------------------------------------------------


def test_every_planted_error_is_present_in_its_own_sentence():
    for sample in load_samples(FIXTURE):
        if sample.is_control:
            continue
        assert normalize(sample.error) in normalize(sample.spoken), sample.id


def test_no_planted_error_overlaps_its_own_repair():
    # If the error string were a substring of a repaired form, survival
    # would always win and the experiment could never observe a repair.
    for sample in load_samples(FIXTURE):
        for form in sample.repaired:
            assert normalize(sample.error) not in normalize(form), sample.id


def test_every_planted_sample_offers_at_least_one_repaired_form():
    for sample in load_samples(FIXTURE):
        if not sample.is_control:
            assert sample.repaired, sample.id


def test_controls_are_grammatical_and_carry_no_planted_error():
    controls = [s for s in load_samples(FIXTURE) if s.is_control]

    assert len(controls) >= 4
    assert all(not sample.repaired for sample in controls)


def test_sample_ids_are_unique():
    samples = load_samples(FIXTURE)

    assert len({sample.id for sample in samples}) == len(samples)


# --- reading a normalised phrase back ------------------------------------


def test_a_quoted_phrase_gets_its_pronoun_back():
    from app.text import as_spoken

    assert as_spoken("yesterday i go") == "yesterday I go"
    assert as_spoken("i have twenty eight years") == "I have twenty eight years"


def test_nothing_else_about_the_quotation_is_invented():
    from app.text import as_spoken

    # No capitals, no punctuation, no rewriting. It is a quotation.
    assert as_spoken("responsible of") == "responsible of"
    assert as_spoken("i think it is fine") == "I think it is fine"
