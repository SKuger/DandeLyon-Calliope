"""Comparing what was said with what was heard.

Written for the Whisper experiment, moved here when the assessor turned
out to need exactly the same three operations: put two transcripts on
comparable footing, find a phrase inside one, and say how far apart they
are. Keeping two copies would mean the experiment eventually measures
something the product does not do.

The only normalisation here that is not cosmetic is spelling out small
numbers. The speaker says "twenty eight", Whisper writes "28", and a
substring match between the two fails on a sentence where nothing
actually differs.
"""

from __future__ import annotations

import re

from app.asr.base import Transcript

_NUMBERS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

_PUNCTUATION = re.compile(r"[^a-z0-9' ]+")
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'"})


def spell_number(token: str) -> str | None:
    """Digits to words for 0-99: ages, months, hours, counts.

    Larger numbers are left alone. A year is read aloud as "twenty twenty
    one" or "two thousand twenty one" depending on the speaker, and
    guessing between them would introduce the mismatch this is meant to
    remove.
    """
    if not token.isdigit():
        return None
    value = int(token)
    if value < 20:
        return _NUMBERS[value]
    if value < 100:
        tens, units = divmod(value, 10)
        return _TENS[tens] if units == 0 else f"{_TENS[tens]} {_NUMBERS[units]}"
    return None


def normalize(text: str) -> str:
    """Lowercase, de-punctuate, split hyphens, spell out small numbers.

    Hyphens become spaces so "front-end" and "front end" are not two
    different transcripts of the same words. "frontend" still differs,
    and that is the honest answer: we do not know which was said.
    """
    text = text.translate(_APOSTROPHES).lower().replace("-", " ")
    text = _PUNCTUATION.sub(" ", text)
    tokens: list[str] = []
    for token in text.split():
        spelled = spell_number(token)
        tokens.extend(spelled.split() if spelled else [token])
    return " ".join(tokens)


def find_span(transcript: Transcript, span: str) -> tuple[int, int] | None:
    """Locate a normalised phrase in the transcript's word list.

    Returns the half-open index range over `transcript.words`, or None. A
    single spoken word can normalise to several tokens ("28" -> "twenty
    eight"), so the search runs over a flattened token stream and maps
    back to word indices.
    """
    wanted = normalize(span).split()
    if not wanted:
        return None

    flat: list[tuple[str, int]] = [
        (token, index)
        for index, word in enumerate(transcript.words)
        for token in normalize(word.text).split()
    ]
    for start in range(len(flat) - len(wanted) + 1):
        if [token for token, _ in flat[start : start + len(wanted)]] == wanted:
            return flat[start][1], flat[start + len(wanted) - 1][1] + 1
    return None


def span_confidence(transcript: Transcript, span: str) -> tuple[float, ...] | None:
    """Per-word confidence over a phrase, or None if it is not there.

    Used twice, for opposite purposes: in the experiment, to ask whether
    an invented word is less certain than a heard one; in the assessor,
    to refuse to correct a phrase the transcriber was not sure it heard.
    """
    located = find_span(transcript, span)
    if located is None:
        return None
    start, end = located
    scores = [
        word.confidence
        for word in transcript.words[start:end]
        if word.confidence is not None
    ]
    return tuple(scores) if scores else None


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein over words, divided by the length of the reference."""
    source = normalize(reference).split()
    target = normalize(hypothesis).split()
    if not source:
        return 0.0 if not target else 1.0

    previous = list(range(len(target) + 1))
    for i, want in enumerate(source, start=1):
        current = [i]
        for j, got in enumerate(target, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (want != got),
                )
            )
        previous = current
    return previous[-1] / len(source)


_STANDALONE_I = re.compile(r"\bi\b")


def as_spoken(phrase: str) -> str:
    """Make a normalised phrase readable again, for display only.

    The assessor works on normalised text, so a finding quotes the
    transcript in lower case with the punctuation gone -- which is what
    the two guards need and not what anyone wants to read on a card that
    says "you said this". This restores the one thing that actually looks
    wrong, the lower-case "I", and nothing else: guessing at capitals and
    punctuation would be rewriting the quotation.
    """
    return _STANDALONE_I.sub("I", phrase)
