"""The conversational partner, when a model is configured.

It is a partner, not a tutor. It never corrects anything, and the prompt
says so twice, because a model asked to talk with a learner will start
teaching within three turns and the correction that arrives mid-sentence
is the one that makes him stop talking. Corrections happen afterwards,
from the transcript, where they can be checked against what he actually
said.

Short replies are a latency requirement rather than a style: a turn has
about two seconds, and the reply has to be spoken as well as generated.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 6.0

#: Enough for two sentences. A longer reply costs generation time and
#: then costs it again in speech, and it turns a conversation into a
#: podcast.
MAX_TOKENS = 120

SYSTEM = """\
You are having a spoken conversation in English with a Colombian
software developer who is practising. He is intermediate: he will make
mistakes.

Do not correct him. Not his grammar, not his word choice, not his
pronunciation, and not by repeating his sentence back correctly. That
happens elsewhere, after the conversation. Correcting him here is what
makes him stop talking.

Reply in one or two short sentences and end with something that gives
him room to say more. Keep the vocabulary ordinary. Never ask two
questions at once.

You are talking, so write what you would say out loud: no lists, no
markdown, no emoji.\
"""

FALLBACK = "Sorry, I lost that. Could you say it again?"


class ClaudeReplier:
    """Anthropic behind the `Replier` Protocol."""

    name = "claude"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        timeout: float = TIMEOUT_SECONDS,
        client=None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = client

    def _connect(self):
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def reply(
        self, history: Sequence[tuple[str, str]], message: str
    ) -> str:
        if not message.strip():
            return "I did not catch that. Could you say it again?"

        messages = [
            {"role": "assistant" if role == "assistant" else "user",
             "content": text}
            for role, text in history
            if text.strip()
        ]
        messages.append({"role": "user", "content": message})

        try:
            async with asyncio.timeout(self._timeout):
                response = await self._connect().messages.create(
                    model=self._model,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM,
                    messages=messages,
                )
        except TimeoutError:
            logger.warning("replier timed out after %ss", self._timeout)
            return FALLBACK
        except Exception as exc:  # noqa: BLE001 - dead air is worse
            logger.warning("replier failed: %s", exc)
            return FALLBACK

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        # Silence in a conversation is the user's problem, not ours.
        return text or FALLBACK
