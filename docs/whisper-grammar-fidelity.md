# Does Whisper correct his grammar?

This project only makes sense if a transcript still contains the mistakes
the speaker made. Whisper is trained to produce plausible text, not to
document what was said, so the worry is that it quietly repairs
*"yesterday I go to the office"* into *"yesterday I went"* and a tutor
built on top tells him he speaks perfectly.

This was the first thing built, before any of the application, because
the answer changes the design and a wrong answer found in month two is a
month wasted.

**The short version.** The worry is real but small, and it is not the
problem. Across 504 transcriptions of sentences carrying a planted error,
Whisper repaired the grammar in **6% of cases on clean audio and 18% on
heavily accented audio**. What it does instead, on accented audio, is
mishear the sentence outright: 41% of those transcripts are of neither
the mistake nor its correction.

And it rewrites *correct* sentences far more often than it repairs
incorrect ones. Of 72 transcriptions of sentences that were already
grammatical, **94% came back altered on accented audio**. The design
consequence is the opposite of the one expected: the system had to be
protected against **inventing** mistakes, not against losing them.

## How it was measured

Eighteen sentences a backend developer would plausibly say
([`fixtures/planted_errors.json`](../experiments/whisper_fidelity/fixtures/planted_errors.json)):

- **Fourteen carry exactly one planted error** — wrong tense, Spanish
  word order, false friend, wrong preposition, missing article,
  uncountable plural, double negative, third-person agreement. One error
  each, because a transcript that repairs one of two and keeps the other
  cannot be scored without judgement.
- **Four are already correct.** These are the controls, and they turned
  out to matter more than the fourteen.

Each sentence was spoken by four voices, transcribed by three model sizes
under three decoding configurations: **648 transcriptions**, all of them
in [`results/rows.jsonl`](../experiments/whisper_fidelity/results/rows.jsonl)
with the transcript verbatim.

Each transcript is scored `survived` (the mistake is still there),
`repaired` (Whisper wrote the grammatical form) or `other` (it heard a
different sentence). Controls score `clean` or `altered`.

### The voices, and what they are worth

| Voice | What it is | What it stands for |
|---|---|---|
| `native-us` | Microsoft Zira, en-US | The easy case: clean acoustics |
| `native-us-fast` | Zira at +3 rate | Speech rate without accent |
| `es-mx-female` | Microsoft Sabina reading English | Spanish phonology over English |
| `es-mx-male` | Microsoft Raul reading English | The same, second voice |

The accented voices are synthetic: a Spanish TTS engine reading English
orthography through Spanish phonology. That is a **heavier** accent than
a B2 speaker has, so the accented columns are an upper bound on accent
stress rather than a portrait of the user. The harness takes audio from a
directory, so the entire matrix re-runs unchanged against real
recordings — and it should, because that is the number that would settle
it.

### The decoding configurations

| Name | Settings | Why |
|---|---|---|
| `whisper-defaults` | beam 5, temperature ladder, conditions on previous text | The control: what Whisper is normally called with |
| `faithful` | greedy, temperature 0, no conditioning | The brief's first hypothesis: these settings push toward correction |
| `faithful-prompted` | greedy + an ungrammatical, disfluent `initial_prompt` | Pushing harder: prime the decoder with learner English |

The prompt shares no content words with the test sentences. Priming with
the actual sentences would have measured whether Whisper can read.

## Result 1: on clean audio, Whisper leaves the mistakes alone

| Voice | Model | Survived | Repaired | Other |
|---|---|---:|---:|---:|
| `native-us` | `small.en` | 14/14 | 0/14 | 0/14 |
| `native-us` | `medium.en` | 14/14 | 0/14 | 0/14 |
| `native-us` | `base.en` | 13/14 | 1/14 | 0/14 |
| `native-us-fast` | `small.en` | 12/14 | 2/14 | 0/14 |
| `native-us-fast` | `medium.en` | 13/14 | 1/14 | 0/14 |

Every decoding configuration gives the same answer, within one sentence.
"Yesterday I go to the office" comes back verbatim from `base.en`,
`small.en` and `medium.en`, greedy or beam, conditioned or not.

**The premise of the project is mostly wrong on clean audio, and knowing
that is worth the day it took.** If the user had a native accent, the
naive pipeline would have worked.

## Result 2: on accented audio, it mishears rather than repairs

| Voice | Model | Survived | Repaired | Other | Mean WER |
|---|---|---:|---:|---:|---:|
| `es-mx-female` | `base.en` | 4/14 | 1/14 | **9/14** | 0.35 |
| `es-mx-female` | `small.en` | 7/14 | 3/14 | 4/14 | 0.21 |
| `es-mx-female` | `medium.en` | 8/14 | 1/14 | 5/14 | 0.18 |
| `es-mx-male` | `base.en` | 2/14 | 2/14 | **10/14** | 0.42 |
| `es-mx-male` | `small.en` | 5/14 | 5/14 | 4/14 | 0.25 |
| `es-mx-male` | `medium.en` | 6/14 | 3/14 | 5/14 | 0.23 |

(`faithful` decoding. The other two configurations are within a sentence
or two of these; the full table is in
[`results/summary.md`](../experiments/whisper_fidelity/results/summary.md).)

The dominant outcome on `base.en` is neither survival nor repair. It is a
different sentence:

> said: *Yesterday I go to the office and I speak with my manager about the project.*
> heard: *Just a day you go to the office and speak with my manager about the project.*

Cases like this are easy to mistake for grammar repair when read quickly.
`I go` → `I got` looks like a tense correction and is a vowel the model
did not catch. Scoring them as `repaired` would have overstated the
finding by a factor of three, which is why the scorer has a third
outcome.

## Result 3: the controls, which is the real problem

The four control sentences are already correct. A transcriber that
changes them hands the tutor a mistake the user never made.

| Voice | `base.en` | `small.en` | `medium.en` |
|---|---|---|---|
| `native-us` | 4/4 clean | 4/4 clean | 4/4 clean |
| `native-us-fast` | 4/4 clean | 4/4 clean | 4/4 clean |
| `es-mx-female` | 0/4 clean | 1/4 clean | 0/4 clean |
| `es-mx-male` | 0/4 clean | 0/4 clean | 0/4 clean |

**On accented speech, one of those 24 transcriptions came back as
spoken.** Word error rate on the controls runs 0.23–0.54. Over all three
decoding configurations it is 94% altered.

So the risk this project was built to avoid — losing his mistakes — is
real at 18% on accented audio. The risk nobody mentioned — inventing
mistakes he did not make — runs at 94%. A tutor that corrects
sentences he said correctly is worse than useless: he stops trusting it,
and then the real corrections go unread too.

This inverted the design. Everything downstream is built to be
conservative about correcting rather than thorough about catching.

## Result 4: decoding settings barely matter; model size does

The brief's first hypothesis was that temperature and the conditioning
prompt drive the repairs. They do not, at this scale:

| Decoding | Repaired (accented) | Repaired (clean) | Controls altered (clean) |
|---|---:|---:|---:|
| `whisper-defaults` | 17.9% | 6.0% | 0% |
| `faithful` | 17.9% | 7.1% | 0% |
| `faithful-prompted` | 17.9% | 6.0% | **20.8%** |

Across 84 accented transcriptions each, the three configurations produce
*the same* repair rate to one decimal place.

Model size moves the numbers far more: `base.en` → `medium.en` doubles
survival on accented audio and halves word error rate.

One clear finding about the prompt, in the other direction: priming with
ungrammatical learner English **damaged clean transcripts**. Mean WER on
`native-us` with `base.en` went from 0.03 to 0.28, and one clean control
in five came back altered where none had before. The prompt sets a register, and the register
leaks into audio that never needed it.

The `faithful` defaults are kept anyway — they cost nothing, they remove
a mechanism that could bite on longer recordings, and
`condition_on_previous_text` is a real risk over a two-minute monologue
in a way it is not over one sentence. But the honest summary is that
**this dial was not the problem.**

## Result 5: confidence knows when it is guessing

The brief's second method: a word Whisper invented should be less certain
than one it heard. Measured directly
([`results/confidence.md`](../experiments/whisper_fidelity/results/confidence.md)) by
transcribing everything with `small.en` and labelling each transcribed
word as heard (its token is in the sentence that was spoken) or invented:

| Voice | Words | Invented | Median confidence, heard | Median, invented |
|---|---:|---:|---:|---:|
| `es-mx-female` | 196 | 37 | 0.949 | 0.494 |
| `es-mx-male` | 191 | 45 | 0.938 | 0.444 |
| `native-us` | 205 | 5 | 0.988 | 0.617 |

Only 7–11% of invented words score above the median of heard words. That
is a usable signal, and it is the one the product runs on:

| Confidence floor | Invented words dropped | Real words dropped |
|---:|---:|---:|
| 0.40 | 46% | 7% |
| **0.55** | **62%** | **13%** |
| 0.70 | 78% | 20% |

0.55 is the floor in
[`app/assess/base.py`](../app/assess/base.py): no correction is issued on
a span whose mean word confidence falls below it. The trade is deliberate
— a missed mistake will be made again next week, and an invented one gets
practised.

**One thing confidence does not do.** It does not separate `repaired`
from `survived`. On the accented subset the mean span confidence for
repairs and for survivals is the same to three decimal places on
`small.en` (0.805 vs 0.807), and on `medium.en` it is *higher* for
repairs (0.901 vs 0.742). When Whisper rewrites a sentence into fluent
English it is confident about it. Confidence catches ASR failure; it does
not catch fluency.

## What was not tried

- **A second, phonetic model** (wav2vec2 without a language model) and
  disagreement between the two. This is the brief's third idea and it is
  the most promising thing left: the failure mode here is acoustic, which
  is exactly where an acoustic model is stronger. Not done because
  `small.en` with a confidence floor turned out to be enough to build on,
  and a second model doubles the latency budget that is
  [already over](latency.md).
- **Phoneme-level comparison.** Follows from the above.
- **The user's actual voice.** The synthetic accent overshoots. Every
  number in the accented columns should be read as a worst case, and the
  harness exists so this can be redone properly in an afternoon.

## Reproducing it

```bash
pip install -r requirements-asr.txt
python -m experiments.whisper_fidelity.synthesize
python -m experiments.whisper_fidelity.run
python -m experiments.whisper_fidelity.confidence
```

The runner keeps an append-only ledger and skips work already done, so it
resumes. Around 30 minutes on a CPU for the full matrix.

To run it against real recordings instead, put them in
`recordings/whisper_fidelity/<voice>/<sample id>.wav` and skip the
synthesis step. Nothing else changes.
