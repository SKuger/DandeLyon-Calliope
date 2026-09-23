"""faster-whisper behind the `Transcriber` Protocol.

Imported lazily and installed from a separate requirements file, because
the contract of this repository is that it runs with no weights. Nothing
above this module knows which backend produced a transcript -- it only
reads `Transcript.model` and `Transcript.decoding` to find out.

The model is loaded once and reused. Loading it costs seconds and a few
hundred megabytes of RAM; doing that per request would make the latency
budget in Phase 3 unreachable before the first token is even decoded.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.asr.base import (
    DecodeOptions,
    Transcript,
    TranscriptionError,
    Word,
)

logger = logging.getLogger(__name__)

#: int8 on CPU is roughly four times faster than float32 and, on this
#: task, indistinguishable in the transcript. It is the difference
#: between a usable local tool and one that is not.
DEFAULT_COMPUTE_TYPE = "int8"


class FasterWhisperTranscriber:
    """Local Whisper. No network once the weights are cached, no audio
    leaving the machine -- which is the actual reason this is local
    rather than an API call.
    """

    def __init__(
        self,
        model_size: str = "small.en",
        decode: DecodeOptions | None = None,
        device: str = "cpu",
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        download_root: str | None = None,
        language: str = "en",
    ) -> None:
        self.model_size = model_size
        self.decode = decode or DecodeOptions()
        self.language = language
        self._device = device
        self._compute_type = compute_type
        self._download_root = download_root
        self._model = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return f"faster-whisper:{self.model_size}"

    def _load(self):
        from faster_whisper import WhisperModel

        logger.info(
            "loading %s on %s (%s)",
            self.model_size,
            self._device,
            self._compute_type,
        )
        return WhisperModel(
            self.model_size,
            device=self._device,
            compute_type=self._compute_type,
            download_root=self._download_root,
        )

    async def transcribe(self, audio: Path) -> Transcript:
        if not audio.is_file():
            raise TranscriptionError(f"No such recording: {audio}")

        # One load, even under concurrent requests: two threads racing to
        # build the same model would double the memory for no gain.
        async with self._lock:
            if self._model is None:
                try:
                    self._model = await asyncio.to_thread(self._load)
                except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                    raise TranscriptionError(
                        f"Could not load {self.model_size}: {exc}"
                    ) from exc

        return await asyncio.to_thread(self._transcribe, audio)

    def _transcribe(self, audio: Path) -> Transcript:
        try:
            segments, info = self._model.transcribe(
                str(audio), language=self.language, **self.decode.as_kwargs()
            )
            collected = list(segments)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            raise TranscriptionError(f"Whisper failed on {audio.name}: {exc}") from exc

        words: list[Word] = []
        for segment in collected:
            for word in segment.words or ():
                words.append(
                    Word(
                        text=word.word.strip(),
                        start=float(word.start),
                        end=float(word.end),
                        confidence=(
                            None
                            if word.probability is None
                            else float(word.probability)
                        ),
                    )
                )

        text = " ".join(segment.text.strip() for segment in collected).strip()

        # An empty transcript is a legitimate result -- the user held the
        # button and said nothing. It is not an error, and the metrics
        # downstream have to survive it.
        return Transcript(
            text=text,
            words=tuple(words),
            language=info.language or self.language,
            duration=float(info.duration),
            model=self.name,
            decoding=self.decode.name,
        )
