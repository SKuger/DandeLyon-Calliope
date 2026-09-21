"""Configuration, read once from the environment.

Every value has a default that works with nothing installed and nothing
configured. That is the contract: clone, run, speak -- and only then
decide whether to download a model or add a key.

The defaults are also the privacy policy. Transcription is local, the
assessor is the offline rule set, and nothing leaves the machine until
someone sets `ANTHROPIC_API_KEY` on purpose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    recordings_dir: Path = Path(os.getenv("RECORDINGS_DIR", "recordings"))

    transcriber: str = os.getenv("TRANSCRIBER", "whisper")
    """`whisper` (local, needs the ASR extra) or `scripted` (fixtures).
    It falls back to `scripted` on its own when faster-whisper is not
    installed, so this only has to be set to force the fallback."""

    whisper_model: str = os.getenv("WHISPER_MODEL", "small.en")
    """small.en is the floor for accented speech. The fidelity experiment
    found base.en unusable on it -- see
    docs/whisper-grammar-fidelity.md -- and medium.en is three times
    slower for a couple of points of word error rate."""

    transcript_fixtures: str = os.getenv("TRANSCRIPT_FIXTURES", "")

    assessor: str = os.getenv("ASSESSOR", "rules")
    """`rules`, `claude`, `both` or `none`. `both` runs the offline rules
    first and adds whatever the model finds that they did not."""

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = os.getenv("MODEL", "claude-sonnet-5")

    exam_version: str = os.getenv("EXAM_VERSION", "v1")

    replier: str = os.getenv("REPLIER", "claude")
    """Who answers in conversation mode: `claude` when a key is set, or
    `scripted` to force the offline partner."""

    voice: str = os.getenv("VOICE", "silent")
    """`silent`, `piper` or `sapi`. Silent by default so the turn loop
    works with nothing downloaded; a real voice is one variable away."""

    piper_model: str = os.getenv("PIPER_MODEL", "")
    sapi_voice: str = os.getenv("SAPI_VOICE", "Microsoft Zira Desktop")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "calliope.db"

    @property
    def uses_a_model(self) -> bool:
        """Whether anything in this configuration sends text off the
        machine. Surfaced on the dashboard, because "local by default" is
        a claim the user should be able to check rather than trust.

        Conversation counts. It was left out of this in the first draft,
        which would have shown "nothing leaves this machine" on a
        configuration that sends every spoken turn to an API.
        """
        if not self.anthropic_api_key:
            return False
        return self.assessor in ("claude", "both") or self.replier == "claude"


settings = Settings()
