"""Numbers computed from a transcript, never asked of a model.

This is the load-bearing decision of the whole project. A language model
asked to rate fluency out of ten will return a different ten tomorrow,
for the same audio, and six months of those numbers is a mood diary, not
a measurement. Everything here is arithmetic over the words and their
timestamps: identical input gives identical output, today and in June.

What the model is for is the part arithmetic cannot do -- saying *why* a
sentence is wrong. That lives behind the assessor boundary and never
produces a score.

Three rules the module holds to:

- **Unmeasurable is `None`, never zero.** A recording with no word
  timings has no pause distribution. Reporting 0.0 would draw a
  confident flat line through a chart of something that was never
  measured.
- **Nothing that depends on how long he talked.** Raw type/token ratio
  falls as a monologue gets longer, so a user who speaks for three
  minutes instead of one looks like he is losing vocabulary. The
  windowed variant is comparable across lengths, which is the only
  property that matters here.
- **Heuristics say so.** The tense inventory is pattern matching, not
  parsing. Where it is wrong it is wrong in a documented direction, and
  the tests pin the cases that are known to fool it.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from app.asr.base import Transcript, Word

#: A gap this long ends an utterance. Below it, silence is breathing and
#: thinking; above it, the speaker finished. 0.7s is the conventional
#: boundary in the L2 fluency literature and, more usefully here, it is
#: the value the self-correction detector was calibrated against.
UTTERANCE_GAP = 0.7

#: A pause worth counting. Shorter gaps are articulation, not hesitation.
PAUSE_FLOOR = 0.25

#: Window for the moving-average type/token ratio. Fifty words is long
#: enough to see vocabulary and short enough that a two-minute answer and
#: a twenty-second one are on the same scale.
MATTR_WINDOW = 50

#: Sounds with no lexical content. Unambiguous: none of these is also a
#: real word in a sentence about work.
FILLERS = frozenset(
    "uh um uhm umm er erm ah eh mm mmm hmm ahem".split()
)

#: Discourse markers counted separately, because they are not errors.
#: "like" is a filler in "it was, like, broken" and a verb in "I like
#: Python", and telling those apart needs a parser. Kept out of the
#: filler count so the filler count stays trustworthy.
DISCOURSE_MARKERS = (
    ("you", "know"),
    ("i", "mean"),
    ("like",),
    ("sort", "of"),
    ("kind", "of"),
)

#: Spanish leaking through. These are the tell that he ran out of English
#: mid-sentence, which is a different event from hesitating in English and
#: is the one he most wants to see go down.
SPANISH_CRUTCHES = (
    ("este",),
    ("o", "sea"),
    ("digamos",),
    ("como", "que"),
    ("how", "do", "you", "say"),
    ("how", "you", "say"),
    ("what", "is", "the", "word"),
    ("i", "don't", "know", "how", "to", "say"),
)

#: Explicit repair markers. "no sorry" and "I mean" are the audible sound
#: of a sentence being restarted.
REPAIR_MARKERS = (
    ("sorry",),
    ("i", "mean"),
    ("no", "no"),
    ("wait",),
    ("perdon",),
)

_IRREGULAR_PAST = frozenset(
    """was were had did went said made took came saw got gave knew thought
    found told became left felt brought began kept held wrote stood heard
    let meant set met ran paid sat spoke lay led grew lost fell sent built
    understood drew broke spent cut rose drove bought wore chose""".split()
)

_PARTICIPLES = frozenset(
    """been had done gone said made taken come seen got given known thought
    found told become left felt brought begun kept held written stood heard
    let meant set met run paid sat spoken lain led grown lost fallen sent
    built understood drawn broken spent cut risen driven bought worn
    chosen""".split()
)

_BE_PRESENT = frozenset("am is are i'm you're we're they're he's she's it's".split())
_BE_PAST = frozenset("was were".split())
_HAVE_PRESENT = frozenset("have has i've you've we've they've he's she's".split())
_MODALS = frozenset("can could may might must should would will shall".split())

#: The tenses the placement exam and the dashboard track. A tense missing
#: from a session is as informative as one used badly: avoidance is how
#: an intermediate speaker hides a gap.
TRACKED_TENSES = (
    "present_simple",
    "present_continuous",
    "present_perfect",
    "past_simple",
    "past_continuous",
    "past_perfect",
    "future_will",
    "future_going_to",
    "conditional",
    "modal",
)

_WORD = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class PauseProfile:
    count: int
    total_seconds: float
    mean_seconds: float | None
    median_seconds: float | None
    longest_seconds: float | None
    within_utterance: int
    """Pauses that did not end an utterance. A speaker who pauses *inside*
    a clause is assembling it word by word; one who pauses between clauses
    is planning the next thought. The second is what fluent people do."""


@dataclass(frozen=True)
class FluencyMetrics:
    words: int
    speaking_seconds: float

    words_per_minute: float | None
    utterances: int
    mean_utterance_words: float | None

    lexical_diversity: float | None
    """Moving-average type/token ratio over `MATTR_WINDOW` words, or the
    plain ratio when the sample is shorter than one window. `None` when
    there are no words at all."""

    distinct_words: int

    fillers_per_minute: float | None
    discourse_markers_per_minute: float | None
    spanish_crutches_per_minute: float | None
    self_corrections: int
    self_corrections_per_minute: float | None

    tenses_used: tuple[str, ...]
    tenses_absent: tuple[str, ...]

    pauses: PauseProfile | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokens_of(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


def split_utterances(words: tuple[Word, ...]) -> list[list[Word]]:
    """Group words into utterances on silence.

    Punctuation would be the obvious signal and is the wrong one: the
    punctuation in a Whisper transcript is the model's guess at sentence
    boundaries, produced by the same language model that this project
    spends its time distrusting. Silence is in the audio.
    """
    if not words:
        return []
    groups: list[list[Word]] = [[words[0]]]
    for previous, word in pairwise(words):
        if word.start - previous.end >= UTTERANCE_GAP:
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def pause_profile(words: tuple[Word, ...]) -> PauseProfile | None:
    if len(words) < 2:
        return None
    gaps = [
        (word.start - previous.end, word.start - previous.end >= UTTERANCE_GAP)
        for previous, word in pairwise(words)
    ]
    pauses = [(gap, ends) for gap, ends in gaps if gap >= PAUSE_FLOOR]
    if not pauses:
        return PauseProfile(0, 0.0, None, None, None, 0)
    lengths = [gap for gap, _ in pauses]
    return PauseProfile(
        count=len(pauses),
        total_seconds=round(sum(lengths), 3),
        mean_seconds=round(statistics.fmean(lengths), 3),
        median_seconds=round(statistics.median(lengths), 3),
        longest_seconds=round(max(lengths), 3),
        within_utterance=sum(1 for _, ends in pauses if not ends),
    )


def moving_average_ttr(tokens: list[str], window: int = MATTR_WINDOW) -> float | None:
    """Type/token ratio that does not punish talking for longer.

    The plain ratio is a function of length: every additional word is more
    likely to be a repeat, so the number falls whether or not the speaker's
    vocabulary did. Averaging the ratio over a sliding window removes the
    length term, which is what makes two sessions comparable.
    """
    if not tokens:
        return None
    if len(tokens) <= window:
        return round(len(set(tokens)) / len(tokens), 4)
    ratios = [
        len(set(tokens[start : start + window])) / window
        for start in range(len(tokens) - window + 1)
    ]
    return round(statistics.fmean(ratios), 4)


def count_phrases(tokens: list[str], phrases) -> int:
    total = 0
    for phrase in phrases:
        size = len(phrase)
        total += sum(
            1
            for start in range(len(tokens) - size + 1)
            if tuple(tokens[start : start + size]) == phrase
        )
    return total


def count_self_corrections(utterances: list[list[Word]]) -> int:
    """Restarts: a word repeated, or an explicit repair marker.

    Deliberately narrow. A real disfluency annotation scheme needs the
    audio and a human; this catches the three patterns that are
    unambiguous in text -- a repeated word, a repeated two-word run
    ("I am, I am..."), and someone saying "sorry" or "I mean"
    mid-sentence -- and undercounts everything else. An undercount that
    moves consistently is still a usable trend line; a guess that moves
    with the model's mood is not.
    """
    total = 0
    for utterance in utterances:
        tokens = tokens_of(" ".join(word.text for word in utterance))
        total += sum(
            1
            for first, second in pairwise(tokens)
            if first == second and first not in FILLERS
        )
        # The repeated run is the commonest restart in his recordings and
        # the unigram check walks straight past it: "I am I am" has no two
        # adjacent identical words.
        total += sum(
            1
            for index in range(len(tokens) - 3)
            if tokens[index : index + 2] == tokens[index + 2 : index + 4]
            and tokens[index] != tokens[index + 1]
        )
        total += count_phrases(tokens, REPAIR_MARKERS)
    return total


def tenses_in(tokens: list[str]) -> set[str]:
    """Which tenses appear, by pattern. Not a parser.

    Enough to answer the question the dashboard asks -- which structures
    does he never reach for -- and not enough to grade a sentence. It
    over-reports `present_simple`, because a bare verb and a noun look
    alike without parsing, and it misses tenses split by an adverb
    ("I have never seen"), which is handled by looking a little ahead.
    """
    found: set[str] = set()
    for index, token in enumerate(tokens):
        window = tokens[index + 1 : index + 4]

        if token in ("will", "'ll"):
            found.add("future_will")
        if token in ("would", "'d"):
            found.add("conditional")
        if token in _MODALS - {"will", "would"}:
            found.add("modal")

        if token in _BE_PRESENT:
            if any(word == "going" for word in window[:1]):
                found.add("future_going_to")
            elif any(word.endswith("ing") for word in window):
                found.add("present_continuous")
            else:
                found.add("present_simple")

        if token in _BE_PAST:
            if any(word.endswith("ing") for word in window):
                found.add("past_continuous")
            else:
                found.add("past_simple")

        if token in _HAVE_PRESENT and any(
            word in _PARTICIPLES or word.endswith("ed") for word in window
        ):
            found.add("present_perfect")

        if token == "had":
            # "I had worked" is one tense and "I had lunch" is another,
            # and the only thing separating them without a parser is
            # whether a participle follows.
            following = any(
                word in _PARTICIPLES or word.endswith("ed") for word in window
            )
            found.add("past_perfect" if following else "past_simple")

        if token in _IRREGULAR_PAST - _BE_PAST - {"had"}:
            found.add("past_simple")
        elif token.endswith("ed") and len(token) > 3:
            found.add("past_simple")

    if not found & {"present_simple", "past_simple"} and tokens:
        # A bare verb with no auxiliary anywhere is the default reading.
        found.add("present_simple")
    return found


def measure(transcript: Transcript) -> FluencyMetrics:
    tokens = tokens_of(transcript.text)
    utterances = split_utterances(transcript.words)
    seconds = transcript.duration
    minutes = seconds / 60 if seconds > 0 else None

    def per_minute(count: int) -> float | None:
        return None if minutes is None else round(count / minutes, 2)

    fillers = sum(1 for token in tokens if token in FILLERS)
    used = tenses_in(tokens) if tokens else set()
    corrections = count_self_corrections(utterances)

    return FluencyMetrics(
        words=len(tokens),
        speaking_seconds=round(seconds, 2),
        words_per_minute=None if minutes is None else round(len(tokens) / minutes, 1),
        utterances=len(utterances),
        mean_utterance_words=(
            None
            if not utterances
            else round(
                statistics.fmean(
                    len(tokens_of(" ".join(w.text for w in utterance)))
                    for utterance in utterances
                ),
                2,
            )
        ),
        lexical_diversity=moving_average_ttr(tokens),
        distinct_words=len(set(tokens)),
        fillers_per_minute=per_minute(fillers),
        discourse_markers_per_minute=per_minute(
            count_phrases(tokens, DISCOURSE_MARKERS)
        ),
        spanish_crutches_per_minute=per_minute(
            count_phrases(tokens, SPANISH_CRUTCHES)
        ),
        self_corrections=corrections,
        self_corrections_per_minute=per_minute(corrections),
        tenses_used=tuple(sorted(used)),
        tenses_absent=tuple(
            tense for tense in TRACKED_TENSES if tense not in used
        ),
        pauses=pause_profile(transcript.words),
    )


def metrics_from(payload: dict[str, Any]) -> FluencyMetrics:
    """Rebuild a measurement that was stored as JSON.

    Needed because the placement exam scores a recording that was
    measured on a different day. Reading the stored numbers, rather than
    re-measuring, keeps the baseline exactly as it was taken -- which is
    the entire point of having a baseline.
    """
    pauses = payload.get("pauses")
    return FluencyMetrics(
        **{
            **payload,
            "pauses": PauseProfile(**pauses) if pauses else None,
            "tenses_used": tuple(payload.get("tenses_used") or ()),
            "tenses_absent": tuple(payload.get("tenses_absent") or ()),
        }
    )
