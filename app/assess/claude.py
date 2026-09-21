"""The model half of the assessor, for the mistakes patterns cannot see.

The rules in `app.assess.rules` catch structures -- a missing article, a
preposition translated word for word. They cannot catch "the client is
very sensible about the response times", because that sentence is
grammatical English and simply means something else. Telling those apart
needs something that understands the sentence.

Three constraints on this one, all of them load-bearing:

- **It never returns a number.** Not a level, not a score out of ten.
  Those come from arithmetic (`app.metrics`), because a model's rating of
  the same audio moves between runs and six months of that measures the
  model, not the speaker.
- **It quotes or it is discarded.** A model asked to correct English will
  paraphrase the sentence and then correct the paraphrase. The shared
  filter in `app.assess.base` drops anything whose `original` is not
  literally in the transcript.
- **A failure costs the findings, never the session.** Every exception is
  caught and turned into an empty list, because the metrics, the audio
  and the transcript are worth keeping even on the day the API is down.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Sequence

from app.asr.base import Transcript
from app.assess.base import MIN_SPAN_CONFIDENCE, Finding, admissible

logger = logging.getLogger(__name__)

#: Longer than a chat reply should need and short enough that a hung API
#: does not hold a recording hostage. The session is already stored by the
#: time this runs; what times out is the feedback, not the data.
TIMEOUT_SECONDS = 25.0

CATEGORIES = (
    "verb_tense", "preposition", "word_order", "article", "agreement",
    "countability", "false_friend", "negation", "calque", "comparative",
    "word_choice", "register",
)

SYSTEM = f"""\
You are reviewing a verbatim transcript of a Colombian Spanish speaker
talking in English. The transcript is automatic and unedited: it contains
his real mistakes, and it may also contain words the recogniser got wrong.

Report only mistakes a teacher would correct, and only where you can
quote his exact words from the transcript.

Rules:
- Quote `original` exactly as it appears in the transcript. Never
  paraphrase it, never fix its spelling, never merge two phrases.
- If a phrase looks more like a transcription error than something a
  learner would say, leave it out. A correction of a word he never said
  is worse than a missed mistake.
- Do not rate, score or level him. No numbers.
- Do not report style preferences, filler words, or anything you would
  not correct in a colleague's email.
- `explanation` is one sentence, in English, aimed at a fluent reader.
  Where a Spanish structure explains the mistake, say which one.

Answer with a JSON array and nothing else. Each element:
{{"category": one of {list(CATEGORIES)},
  "original": "...", "correction": "...", "explanation": "..."}}

An empty array is a valid and expected answer.\
"""

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


class ClaudeAssessor:
    """Anthropic behind the `Assessor` Protocol."""

    name = "claude"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        timeout: float = TIMEOUT_SECONDS,
        min_confidence: float = MIN_SPAN_CONFIDENCE,
        client=None,
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._min_confidence = min_confidence
        self._client = client
        self._api_key = api_key

    def _connect(self):
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def assess(
        self, transcript: Transcript, context: Sequence[str] = ()
    ) -> list[Finding]:
        if not transcript.text.strip():
            return []

        try:
            async with asyncio.timeout(self._timeout):
                raw = await self._ask(transcript, context)
        except TimeoutError:
            logger.warning("assessor timed out after %ss", self._timeout)
            return []
        except Exception as exc:  # noqa: BLE001 - never lose the session
            logger.warning("assessor failed: %s", exc)
            return []

        return admissible(transcript, parse(raw), self._min_confidence)

    async def _ask(self, transcript: Transcript, context: Sequence[str]) -> str:
        prompt = [f"Transcript:\n{transcript.text}"]
        if context:
            # Earlier sessions, so the model can recognise a habit rather
            # than a one-off. It is told explicitly not to correct them.
            earlier = "\n".join(f"- {line}" for line in context)
            prompt.append(
                "Earlier things he has said, for context only. Do not "
                f"report mistakes from these:\n{earlier}"
            )

        response = await self._connect().messages.create(
            model=self._model,
            max_tokens=2000,
            temperature=0,
            system=SYSTEM,
            messages=[{"role": "user", "content": "\n\n".join(prompt)}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )


def parse(raw: str) -> list[Finding]:
    """Read findings out of a model's answer, dropping what is malformed.

    A model that answers with prose around its JSON, or invents a
    category, is not an exception to handle upstream -- it is Tuesday.
    The unusable elements are skipped and the rest are kept, because one
    bad entry should not cost the other four.
    """
    match = _JSON_ARRAY.search(raw or "")
    if match is None:
        logger.info("assessor returned no JSON array")
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.info("assessor returned invalid JSON: %s", exc)
        return []
    if not isinstance(payload, list):
        return []

    findings = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        original = str(entry.get("original", "")).strip()
        correction = str(entry.get("correction", "")).strip()
        if not original or not correction:
            continue
        category = str(entry.get("category", "")).strip()
        findings.append(
            Finding(
                # An unknown category becomes `word_choice` rather than a
                # new column in the dashboard nobody chose to add.
                category=category if category in CATEGORIES else "word_choice",
                original=original,
                correction=correction,
                explanation=str(entry.get("explanation", "")).strip(),
                detector="claude",
            )
        )
    return findings
