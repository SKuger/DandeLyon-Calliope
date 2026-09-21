"""The assessor, tested on what it must refuse to say.

Two refusals carry the design. A correction of a sentence he did not say
is worse than a missed mistake, because he cannot recognise it and
practises something that never happened -- so a finding has to quote the
transcript, and a finding over words the recogniser was unsure of is
thrown away. The second one is the product consequence of
`docs/whisper-grammar-fidelity.md`, and it is the reason that experiment
came before any of this.
"""

import pytest

from app.asr.base import Transcript, Word
from app.assess.base import Finding, NullAssessor, admissible
from app.assess.claude import ClaudeAssessor, parse
from app.assess.rules import RuleBasedAssessor


def heard(text: str, confidence: float | None = 0.95) -> Transcript:
    words = tuple(
        Word(text=token, start=i * 0.4, end=i * 0.4 + 0.3, confidence=confidence)
        for i, token in enumerate(text.split())
    )
    return Transcript(text=text, words=words, duration=len(words) * 0.4)


def unsure(text: str, phrase: str, confidence: float = 0.2) -> Transcript:
    """A transcript where one phrase was barely recognised."""
    low = set(phrase.lower().split())
    words = tuple(
        Word(
            text=token,
            start=i * 0.4,
            end=i * 0.4 + 0.3,
            confidence=confidence if token.lower().strip(".,") in low else 0.95,
        )
        for i, token in enumerate(text.split())
    )
    return Transcript(text=text, words=words, duration=len(words) * 0.4)


# --- the two filters -----------------------------------------------------


def test_a_correction_of_a_sentence_he_did_not_say_is_dropped():
    transcript = heard("I am responsible of the deployment")
    invented = Finding(
        category="verb_tense",
        original="I was responsible of the deployment",
        correction="I was responsible for the deployment",
        explanation="",
    )

    # The classic model failure: paraphrase the sentence, then correct the
    # paraphrase. It reads perfectly and teaches nothing.
    assert admissible(transcript, [invented]) == []


def test_a_correction_over_words_the_recogniser_doubted_is_dropped():
    transcript = unsure("I am responsible of the deployment", "responsible of")
    finding = Finding(
        category="preposition",
        original="responsible of",
        correction="responsible for",
        explanation="",
    )

    # On accented speech a large share of apparent grammar mistakes are
    # the transcriber mishearing a word. Correcting those trains him out
    # of mistakes he never made.
    assert admissible(transcript, [finding]) == []


def test_a_confident_finding_records_the_confidence_it_passed_on():
    transcript = heard("I am responsible of the deployment")
    finding = Finding("preposition", "responsible of", "responsible for", "")

    (kept,) = admissible(transcript, [finding])

    assert kept.confidence == pytest.approx(0.95)


def test_a_backend_without_confidences_does_not_block_everything():
    transcript = heard("I am responsible of the deployment", confidence=None)
    finding = Finding("preposition", "responsible of", "responsible for", "")

    # No confidence is not low confidence. Treating None as zero would
    # make the assessor silent on any backend that does not report one.
    (kept,) = admissible(transcript, [finding])
    assert kept.confidence is None


def test_nothing_configured_finds_nothing_rather_than_crashing():
    import asyncio

    assert asyncio.run(NullAssessor().assess(heard("anything"))) == []


# --- the offline rules ---------------------------------------------------


@pytest.mark.parametrize(
    ("said", "correction"),
    [
        ("Yesterday I go to the office", "yesterday i went"),
        ("I am responsible of the pipeline", "responsible for"),
        ("We depend of a third party API", "depend on"),
        ("I have twenty eight years", "i am twenty eight years old"),
        ("I need more informations", "information"),
        ("I am developer", "i am a developer"),
        ("My colleague work on the API", "my colleague works"),
        ("I have to assist to the review", "attend"),
        ("I am thinking to change my job", "thinking about changing"),
        ("Since three months I am learning Go", "for three months"),
        ("I don't have no experience with Kafka", "don't have any"),
        ("It is a problem very complex", "a very complex problem"),
        ("Do you know where is the staging server", "where the staging is"),
        ("This is more easy than the other", "easier"),
    ],
)
async def test_the_rules_catch_the_planted_error(said, correction):
    findings = await RuleBasedAssessor().assess(heard(said))

    assert any(correction in finding.correction for finding in findings), (
        [f.correction for f in findings]
    )


@pytest.mark.parametrize(
    "said",
    [
        "The developers work on the API",
        "My colleague works on the front end",
        "I have been here for three months",
        "It is a complex problem in the queue consumer",
        "I am a developer and I deployed it yesterday",
        "This morning I reviewed two pull requests",
        "The database was under heavy load",
        "I lead a team of six engineers",
    ],
)
async def test_the_rules_stay_quiet_on_correct_english(said):
    # A tutor that corrects correct sentences is worse than no tutor: he
    # stops trusting it, and then the real corrections go unread too.
    assert await RuleBasedAssessor().assess(heard(said)) == []


async def test_the_rules_cannot_see_a_false_friend():
    # "Sensible" is a real English word in a grammatical sentence that
    # means the opposite of what he intended. Pinned as a miss rather than
    # left to be discovered: this is the whole reason the Assessor
    # Protocol has a second implementation.
    assert await RuleBasedAssessor().assess(
        heard("The client is very sensible about the response times")
    ) == []


async def test_a_finding_carries_which_rule_produced_it():
    (finding,) = await RuleBasedAssessor().assess(heard("I am developer"))

    assert finding.detector == "rules:missing-article-profession"
    assert finding.explanation


async def test_the_same_mistake_twice_in_one_session_is_reported_once():
    findings = await RuleBasedAssessor().assess(
        heard("I am responsible of the pipeline and responsible of the rota")
    )

    assert len(findings) == 1


# --- parsing a model's answer -------------------------------------------


def test_prose_around_the_json_is_tolerated():
    raw = """Here is what I found:
    [{"category": "preposition", "original": "responsible of",
      "correction": "responsible for", "explanation": "Use for."}]
    Hope that helps."""

    (finding,) = parse(raw)

    assert finding.correction == "responsible for"


def test_a_malformed_entry_does_not_cost_the_others():
    raw = """[
      "not an object",
      {"original": "responsible of", "correction": "responsible for"},
      {"original": "", "correction": "nothing"}
    ]"""

    # One bad entry in an array of five is Tuesday, not an exception to
    # propagate.
    assert len(parse(raw)) == 1


def test_an_invented_category_lands_somewhere_known():
    raw = '[{"category": "vibes", "original": "a", "correction": "b"}]'

    # Otherwise the model quietly adds columns to the dashboard.
    assert parse(raw)[0].category == "word_choice"


def test_no_json_at_all_is_an_empty_result():
    assert parse("I could not find any mistakes.") == []
    assert parse("") == []


def test_broken_json_is_an_empty_result():
    assert parse('[{"original": "a", ') == []


# --- the model backend, without a key ------------------------------------


class Exploding:
    class messages:  # noqa: N801 - mirrors the SDK's shape
        @staticmethod
        async def create(**_):
            raise RuntimeError("503 from the API")


async def test_a_failing_model_costs_the_findings_not_the_session():
    assessor = ClaudeAssessor(api_key="", client=Exploding())

    # The metrics, the audio and the transcript are already stored and
    # are worth keeping on the day the API is down.
    assert await assessor.assess(heard("I am developer")) == []


class Slow:
    class messages:  # noqa: N801
        @staticmethod
        async def create(**_):
            import asyncio

            await asyncio.sleep(10)


async def test_a_hung_model_is_abandoned():
    assessor = ClaudeAssessor(api_key="", client=Slow(), timeout=0.05)

    assert await assessor.assess(heard("I am developer")) == []


async def test_silence_is_not_sent_to_the_model():
    class Counting:
        calls = 0

        class messages:  # noqa: N801
            @staticmethod
            async def create(**_):
                Counting.calls += 1

    await ClaudeAssessor(api_key="", client=Counting()).assess(
        Transcript(text="   ", duration=3.0)
    )

    assert Counting.calls == 0
