"""Real voices, both optional, both local.

Piper is the one the project is built around: a small neural TTS that
runs on CPU and needs no network, which is the same reason Whisper is
local. It is a subprocess rather than a library binding because piper
ships as a binary, and shelling out to it is twenty lines against a
dependency that would have to be pinned, built and kept working.

Windows SAPI is here for one reason: it is already installed. It costs
nothing to offer and it means the conversation mode has a real voice on
the machine this is used on before anyone downloads anything.

Neither is the default. `SilentVoice` is, because the contract is that
the whole thing runs with nothing installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

#: A voice that takes longer than this has already lost the turn, so it
#: is abandoned rather than waited for.
SPEAK_TIMEOUT_SECONDS = 8.0


class VoiceError(Exception):
    """Speech could not be produced. The caller falls back to text."""


class PiperVoice:
    """Local neural TTS through the `piper` binary.

    The model path is required rather than discovered. Guessing at a
    voice file and silently producing a different accent than yesterday
    would be a quiet way to make two recordings incomparable.
    """

    name = "piper"

    def __init__(
        self,
        model: str | Path,
        binary: str = "piper",
        timeout: float = SPEAK_TIMEOUT_SECONDS,
    ) -> None:
        self.model = Path(model)
        self._binary = binary
        self._timeout = timeout

    async def speak(self, text: str) -> bytes:
        found = shutil.which(self._binary)
        if found is None:
            raise VoiceError(
                f"{self._binary} is not on PATH. Install piper or leave "
                f"PIPER_MODEL unset to use the silent voice."
            )
        if not self.model.is_file():
            raise VoiceError(f"No piper voice at {self.model}")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "turn.wav"
            process = await asyncio.create_subprocess_exec(
                found,
                "--model", str(self.model),
                "--output_file", str(output),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, errors = await asyncio.wait_for(
                    process.communicate(text.encode("utf-8")), self._timeout
                )
            except TimeoutError as exc:
                process.kill()
                raise VoiceError(
                    f"piper did not answer within {self._timeout}s"
                ) from exc

            if process.returncode != 0 or not output.is_file():
                raise VoiceError(
                    f"piper failed: {errors.decode(errors='replace').strip()}"
                )
            return output.read_bytes()


class SapiVoice:
    """Windows' built-in speech, for the machine this runs on.

    Synchronous under the hood and pushed onto a thread, because SAPI has
    no async interface and blocking the event loop during a conversation
    turn would stall every other request on the same process.
    """

    name = "sapi"

    def __init__(
        self,
        voice: str = "Microsoft Zira Desktop",
        rate: int = 0,
        timeout: float = SPEAK_TIMEOUT_SECONDS,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self._timeout = timeout

    async def speak(self, text: str) -> bytes:
        return await asyncio.to_thread(self._speak, text)

    def _speak(self, text: str) -> bytes:
        # PowerShell 7 first: 5.1's System.Speech only reaches the SAPI5
        # "Desktop" voices and fails on the others with a message about
        # them not being installed.
        host = shutil.which("pwsh") or shutil.which("powershell")
        if host is None:
            raise VoiceError("No PowerShell found; SAPI is Windows only.")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "turn.wav"
            # A script file rather than -Command: with -Command, trailing
            # arguments are appended to the command text instead of
            # becoming $args, and the JSON path lands in the middle of a
            # statement as a parse error.
            script = Path(directory) / "speak.ps1"
            script.write_text(
                "$ErrorActionPreference = 'Stop'\n"
                "Add-Type -AssemblyName System.Speech\n"
                "$plan = Get-Content -Raw -LiteralPath $args[0] | ConvertFrom-Json\n"
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
                "$s.SelectVoice($plan.voice)\n"
                "$s.Rate = [int]$plan.rate\n"
                "$s.SetOutputToWaveFile($plan.path)\n"
                "$s.Speak($plan.text)\n"
                "$s.SetOutputToNull()\n"
                "$s.Dispose()\n",
                encoding="utf-8",
            )
            plan = Path(directory) / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "voice": self.voice,
                        "rate": self.rate,
                        "path": str(output),
                        "text": text,
                    }
                ),
                encoding="utf-8",
            )
            done = subprocess.run(
                [host, "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", str(script), str(plan)],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
            if done.returncode != 0 or not output.is_file():
                raise VoiceError(f"SAPI failed: {done.stderr.strip()}")
            return output.read_bytes()


class FallbackVoice:
    """Try a real voice; fall back to silence rather than losing the turn.

    A conversation that stops because the speech synthesiser is missing a
    file is worse than one that answers in text. The failure is logged
    once per turn, not swallowed.
    """

    name = "fallback"

    def __init__(self, primary, secondary) -> None:
        self._primary = primary
        self._secondary = secondary

    async def speak(self, text: str) -> bytes:
        try:
            return await self._primary.speak(text)
        except VoiceError as exc:
            logger.warning("voice %s failed: %s", self._primary.name, exc)
            return await self._secondary.speak(text)
