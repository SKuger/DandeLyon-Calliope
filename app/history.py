"""What he has said before, and what he keeps getting wrong.

Two different questions that get conflated, so they are two functions
here.

**"You have made this mistake three times in two weeks"** is a `GROUP BY`.
It is the headline feature of the whole project, it has to be exactly
right, and a vector search would answer it approximately for no reason.
The corpus stores the category and the phrase; counting them is SQL.

**"What else has he said that sounds like this?"** is retrieval, and it
is only ever used to give the assessor context -- never to make a claim
to the user. The index is the one from the sibling project (TF-IDF, IDF
weighting, a score floor below which it returns nothing), with two
changes that come from the corpus being one person's speech rather than a
support knowledge base:

- The utterance under assessment is excluded from its own results.
  Otherwise the strongest match for "I am responsible of the pipeline" is
  always itself, and the model is handed proof that he says this
  constantly on the first time he ever said it.
- The floor is a floor on usefulness, not on truth. A weak match here
  does not produce a wrong answer; it produces a distracting example. It
  still returns nothing rather than its best three, for the same reason
  the sibling project does: a model handed three irrelevant paragraphs
  will use them.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.asr.base import Transcript
from app.metrics.fluency import UTTERANCE_GAP
from app.text import normalize

#: Below this cosine score, an utterance is not similar enough to be worth
#: showing the model. Provisional, and calibrated against the sentences in
#: the test suite rather than against a year of his speech, because there
#: is not a year of his speech yet: two utterances about the same incident
#: score 0.45-0.52 there, unrelated ones score 0.0-0.1. Worth revisiting
#: once the corpus is real, and the tests are where that argument will
#: happen.
MIN_SCORE = 0.2

#: How far back "recently" reaches. Two weeks is roughly ten practice
#: sessions at his rate, which is enough for a pattern and short enough
#: that a habit he has already fixed does not keep being quoted at him.
RECENT_DAYS = 14

_STOPWORDS = frozenset(
    """a an and are as at be been but by can do does for from had has have
    i if in into is it its me my no not of on or our so than that the their
    them then there these they this to us was we were what when where which
    who will with would you your""".split()
)

_TOKEN = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Utterance:
    session_id: str
    recorded_at: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.recorded_at[:10]}#{self.session_id[:8]}"


@dataclass(frozen=True)
class Recurrence:
    category: str
    phrase: str
    occurrences: int
    first_seen: str
    last_seen: str
    within_days: int

    @property
    def is_a_habit(self) -> bool:
        """Three or more. Twice is a coincidence worth noting; three times
        inside a fortnight is the thing worth building an exercise from."""
        return self.occurrences >= 3


def tokenize(text: str) -> list[str]:
    words = [w for w in _TOKEN.findall(normalize(text)) if w not in _STOPWORDS]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]


def recurrence(
    connection: sqlite3.Connection,
    category: str,
    phrase: str,
    within_days: int = RECENT_DAYS,
    at: datetime | None = None,
) -> Recurrence:
    """Count how often this exact mistake has come back.

    Matched on the normalised phrase, so "responsible of" and
    "Responsible of," are the same mistake, and on the category, so a
    phrase that is wrong for two different reasons is counted twice --
    once per reason, which is how it will be practised.
    """
    since = ((at or datetime.now(UTC)) - timedelta(days=within_days)).isoformat()
    wanted = normalize(phrase)

    rows = connection.execute(
        "SELECT original, created_at FROM error "
        "WHERE category = ? AND created_at >= ? ORDER BY created_at",
        (category, since),
    ).fetchall()
    matched = [row for row in rows if normalize(row["original"]) == wanted]

    return Recurrence(
        category=category,
        phrase=phrase,
        occurrences=len(matched),
        first_seen=matched[0]["created_at"] if matched else "",
        last_seen=matched[-1]["created_at"] if matched else "",
        within_days=within_days,
    )


def habits(
    connection: sqlite3.Connection,
    within_days: int = RECENT_DAYS,
    at: datetime | None = None,
    minimum: int = 3,
) -> list[Recurrence]:
    """Every mistake repeated at least `minimum` times lately, worst first.

    This is what decides where his practice time goes, so it is exact
    counting rather than a similarity score. An approximate answer to
    "what do I keep getting wrong" would send him to practise the wrong
    thing, politely.
    """
    since = ((at or datetime.now(UTC)) - timedelta(days=within_days)).isoformat()
    rows = connection.execute(
        "SELECT category, original, created_at FROM error "
        "WHERE created_at >= ? ORDER BY created_at",
        (since,),
    ).fetchall()

    grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(
            (row["category"], normalize(row["original"])), []
        ).append(row)

    found = [
        Recurrence(
            category=category,
            phrase=group[0]["original"],
            occurrences=len(group),
            first_seen=group[0]["created_at"],
            last_seen=group[-1]["created_at"],
            within_days=within_days,
        )
        for (category, _), group in grouped.items()
        if len(group) >= minimum
    ]
    return sorted(found, key=lambda item: item.occurrences, reverse=True)


def utterances_of(transcript: Transcript, session_id: str, at: str) -> list[Utterance]:
    """Split a stored transcript into utterances on silence.

    The same boundary the metrics use, so an utterance in the index is the
    same object as an utterance in the numbers. Retrieval over whole
    sessions would return two minutes of speech to illustrate one phrase.
    """
    if not transcript.words:
        text = transcript.text.strip()
        return [Utterance(session_id, at, text)] if text else []

    groups: list[list[str]] = [[transcript.words[0].text]]
    for previous, word in zip(
        transcript.words, transcript.words[1:], strict=False
    ):
        if word.start - previous.end >= UTTERANCE_GAP:
            groups.append([word.text])
        else:
            groups[-1].append(word.text)
    return [Utterance(session_id, at, " ".join(group)) for group in groups]


class UtteranceIndex:
    """TF-IDF over his own speech. No model, no key, no network.

    Rebuilt from SQLite on demand rather than kept warm: a few hundred
    utterances is milliseconds, and an index that has to be invalidated
    is a second source of truth about what he said.
    """

    def __init__(self) -> None:
        self._utterances: list[Utterance] = []
        self._counts: list[Counter[str]] = []
        self._vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._utterances)

    def add(self, utterances: list[Utterance]) -> None:
        if not utterances:
            return
        for utterance in utterances:
            self._utterances.append(utterance)
            self._counts.append(Counter(tokenize(utterance.text)))
        self._reindex()

    def _reindex(self) -> None:
        total = len(self._counts)
        document_frequency: Counter[str] = Counter()
        for counts in self._counts:
            document_frequency.update(counts.keys())
        self._idf = {
            term: math.log((total + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self._vectors = [_unit(self._weigh(counts)) for counts in self._counts]

    def _weigh(self, counts: Counter[str]) -> dict[str, float]:
        return {
            term: (1.0 + math.log(count)) * self._idf.get(term, 1.0)
            for term, count in counts.items()
        }

    def search(
        self,
        query: str,
        limit: int = 3,
        min_score: float = MIN_SCORE,
        exclude_session: str | None = None,
    ) -> list[tuple[Utterance, float]]:
        if not self._utterances:
            return []
        vector = _unit(self._weigh(Counter(tokenize(query))))
        if not vector:
            return []

        scored = [
            (utterance, _dot(vector, other))
            for utterance, other in zip(
                self._utterances, self._vectors, strict=True
            )
            # Without this the best match for a sentence is always itself,
            # and the model is told he says this constantly on the first
            # day he said it once.
            if utterance.session_id != exclude_session
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [pair for pair in scored[:limit] if pair[1] >= min_score]


def _unit(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0.0:
        return {}
    return {term: value / norm for term, value in vector.items()}


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(b) < len(a):
        a, b = b, a
    return sum(value * b.get(term, 0.0) for term, value in a.items())


def build_index(
    connection: sqlite3.Connection, limit: int = 200
) -> UtteranceIndex:
    """Index the last `limit` sessions' utterances."""
    from app.store import get_transcript

    index = UtteranceIndex()
    rows = connection.execute(
        "SELECT id, recorded_at FROM session ORDER BY recorded_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    collected: list[Utterance] = []
    for row in rows:
        transcript = get_transcript(connection, row["id"])
        if transcript is not None:
            collected.extend(
                utterances_of(transcript, row["id"], row["recorded_at"])
            )
    index.add(collected)
    return index
