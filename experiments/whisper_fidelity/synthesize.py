"""Audio for the experiment, without waiting for a recording session.

The honest input to this experiment is the user's own voice, and the
runner accepts exactly that: a directory of `<sample id>.wav` files. But
a finding that arrives a week from now, after someone has found time to
record eighteen sentences twice, is a finding that arrives after the
architecture was already chosen. So the first pass is synthesised.

The trick that makes synthesis worth anything here is the voice. A
native en-US voice reading a planted error is the *easy* case: the audio
is clean, the acoustic evidence is unambiguous, and Whisper has little
reason to guess. The interesting condition is a Spanish (es-MX) voice
reading English, which produces the phoneme substitutions and the vowel
inventory of a Spanish speaker. That is the acoustic situation the real
user is in, approximated by a machine.

It is an approximation and it overshoots: a Spanish TTS engine mapping
English orthography through Spanish phonology is a heavier accent than a
B2 speaker has. Treated as an upper bound on accent stress, and the whole
matrix is re-runnable against real recordings, which is the point of
taking audio from a directory instead of generating it inline.

Windows SAPI is used because it is already installed on the machine this
runs on and needs no download. Any TTS would do; nothing downstream knows
which one produced the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from experiments.whisper_fidelity.score import Sample, load_samples

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "planted_errors.json"

#: Audio is gitignored. It is synthetic here, but the same directory
#: holds real recordings when the matrix is re-run against them, and a
#: rule that only applies to some of the audio is a rule that fails once.
DEFAULT_OUT = Path("recordings") / "whisper_fidelity"


@dataclass(frozen=True)
class Voice:
    label: str
    sapi_name: str
    rate: int = 0
    """SAPI rate, -10 to 10. Faster speech means less acoustic evidence
    per phoneme, which is the other way to make the decoder guess."""

    note: str = ""


VOICES: tuple[Voice, ...] = (
    Voice(
        "native-us",
        "Microsoft Zira Desktop",
        note="Clean en-US. The control: if an error does not survive here, "
        "it was the language model and not the accent.",
    ),
    Voice(
        "native-us-fast",
        "Microsoft Zira Desktop",
        rate=3,
        note="Same voice, faster. Isolates speech rate from accent.",
    ),
    Voice(
        "es-mx-female",
        "Microsoft Sabina Desktop",
        note="Spanish phonology over English orthography. The condition "
        "the real user is in, overshot.",
    ),
    Voice(
        "es-mx-male",
        "Microsoft Raul",
        note="Second accented voice, so a result is not one voice's quirk.",
    ),
)

def host() -> str:
    """Prefer PowerShell 7 over the 5.1 that ships with Windows.

    Not a preference: 5.1's System.Speech only reaches the SAPI5
    "Desktop" voices, so the es-MX male voice is invisible to it and
    `SelectVoice` fails with "no matching voice installed" for a voice
    the same machine happily lists.
    """
    found = shutil.which("pwsh") or shutil.which("powershell")
    if found is None:
        raise RuntimeError(
            "No PowerShell found. Synthesis uses Windows SAPI; on another "
            "platform, put your own recordings in the output directory as "
            "<sample id>.wav and skip this step."
        )
    return found


def installed_voices(powershell: str) -> set[str]:
    """Which voices this host can actually select.

    Asked rather than assumed, because a machine with fewer voices should
    still produce a partial result. An experiment that only runs on the
    machine it was written on is not reproducible, which was the whole
    complaint about the placement tests.
    """
    done = subprocess.run(
        [powershell, "-NoProfile", "-Command",
         "Add-Type -AssemblyName System.Speech; "
         "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
         ".GetInstalledVoices() | "
         "ForEach-Object { $_.VoiceInfo.Name }"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in done.stdout.splitlines() if line.strip()}


# One PowerShell launch per voice: starting the runtime costs more than
# synthesising eighteen sentences does.
_SCRIPT = """
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$plan = Get-Content -Raw -LiteralPath $args[0] | ConvertFrom-Json
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.SelectVoice($plan.voice)
$speaker.Rate = [int]$plan.rate
foreach ($item in $plan.items) {
  $speaker.SetOutputToWaveFile($item.path)
  $speaker.Speak($item.text)
}
$speaker.SetOutputToNull()
$speaker.Dispose()
"""


def _fingerprint(sample: Sample, voice: Voice) -> str:
    """What the audio was made from, so a re-run is idempotent.

    Re-synthesising is slow and re-synthesising into the same filename
    with different text is worse: the transcript would be scored against
    a sentence nobody said. The fingerprint covers the text and the voice
    settings, which are the only inputs.
    """
    raw = f"{sample.spoken}|{voice.sapi_name}|{voice.rate}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def plan(
    samples: tuple[Sample, ...], voice: Voice, out_dir: Path
) -> list[dict[str, str]]:
    """The sentences that still need synthesising for this voice."""
    manifest_path = out_dir / "manifest.json"
    known: dict[str, str] = {}
    if manifest_path.is_file():
        known = json.loads(manifest_path.read_text(encoding="utf-8"))

    pending = []
    for sample in samples:
        target = out_dir / f"{sample.id}.wav"
        fingerprint = _fingerprint(sample, voice)
        if target.is_file() and known.get(sample.id) == fingerprint:
            continue
        pending.append(
            {"path": str(target.resolve()), "text": sample.spoken,
             "id": sample.id, "fingerprint": fingerprint}
        )
    return pending


def synthesize(
    voice: Voice,
    samples: tuple[Sample, ...],
    root: Path,
    powershell: str | None = None,
) -> int:
    out_dir = root / voice.label
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = plan(samples, voice, out_dir)
    if not pending:
        return 0

    powershell = powershell or host()
    script = out_dir / "_speak.ps1"
    script.write_text(_SCRIPT, encoding="utf-8")
    plan_file = out_dir / "_plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "voice": voice.sapi_name,
                "rate": voice.rate,
                "items": [
                    {"path": item["path"], "text": item["text"]}
                    for item in pending
                ],
            }
        ),
        encoding="utf-8",
    )

    done = subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(script), str(plan_file)],
        check=False,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        # The manifest is not updated, so the next run retries. Surface
        # what PowerShell said: the usual cause is a voice name that is
        # installed but not reachable through System.Speech, and that is
        # unguessable from an exit code.
        raise RuntimeError(
            f"Synthesis failed for {voice.label} ({voice.sapi_name}):\n"
            f"{done.stderr.strip() or done.stdout.strip()}"
        )

    manifest_path = out_dir / "manifest.json"
    manifest: dict[str, str] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({item["id"]: item["fingerprint"] for item in pending})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    plan_file.unlink()
    script.unlink()
    return len(pending)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument(
        "--voice",
        action="append",
        choices=[voice.label for voice in VOICES],
        help="Repeatable. Default: every voice.",
    )
    args = parser.parse_args(argv)

    samples = load_samples(args.fixture)
    wanted = [v for v in VOICES if not args.voice or v.label in args.voice]
    powershell = host()
    available = installed_voices(powershell)

    for voice in wanted:
        if voice.sapi_name not in available:
            print(f"{voice.label:16} skipped: {voice.sapi_name} not installed")
            continue
        written = synthesize(voice, samples, args.out, powershell)
        print(
            f"{voice.label:16} {written:3d} new / {len(samples)} samples"
            f"  -> {args.out / voice.label}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
