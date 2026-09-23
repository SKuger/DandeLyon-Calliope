# Latency: where the two seconds go

Above roughly two seconds a spoken exchange stops feeling like a
conversation and starts feeling like filling in a form. That is the
requirement in the brief, so this is the measurement of whether the
current pipeline meets it.

It does not. This document says by how much, which leg is responsible,
and what would have to change.

**Machine.** AMD Ryzen 5 5600X, 6 cores / 12 threads, 16GB, Windows 11.
CPU only, no GPU. faster-whisper 1.2.1 with `compute_type=int8`.

**What was timed.** The recognition numbers are the decode times already
recorded by the fidelity experiment (`experiments/whisper_fidelity/results/rows.jsonl`),
which is 636 transcriptions of clips averaging 3.7 seconds. The speech
numbers were measured directly, twelve utterances of the length the
scripted partner produces. The reply leg was **not measured**: it needs
an API key, and this machine does not have one. Its budget is stated
below and left empty rather than filled with a plausible number.

## The budget

| Leg | Budget | Why it gets that much |
|---|---:|---|
| Recognition | 900ms | The only leg that cannot start before he stops talking. |
| Reply | 700ms | A short answer from a fast model, first token to last. |
| Speech | 400ms | Enough to start playing; the rest can stream. |
| **Total** | **2000ms** | Past this it stops feeling like talking. |

## Recognition, measured

Median decode time for a 3.7-second clip, and the real-time factor
(decode seconds per audio second):

| Model | Median | p90 | Real-time factor | Fits the 900ms budget? |
|---|---:|---:|---:|---|
| `base.en` | 0.72s | 0.81s | 0.19 | Yes, comfortably |
| `small.en` | 2.38s | 2.42s | 0.63 | No, by 2.6× |
| `medium.en` | 6.88s | 7.09s | 1.82 | No, by 7.6× — slower than real time |

**This is the uncomfortable result of the whole project.** The fidelity
experiment ([`whisper-grammar-fidelity.md`](whisper-grammar-fidelity.md))
found `base.en` unusable on accented speech: it does not repair his
grammar so much as fail to hear him, and half its transcripts are of a
different sentence. The model that is fast enough is not accurate enough,
and the model that is accurate enough is not fast enough — on this CPU,
without streaming.

Three ways out, in the order they should be tried:

1. **Stream the recognition.** Decode is roughly linear in audio length,
   so decoding each second as it arrives leaves only the tail to decode
   when he stops. A 3.7-second turn at `small.en` would owe about 0.4s at
   the end instead of 2.4s, which fits. This is the single change worth
   making, and it is the one this repository has not made.
2. **A GPU.** Order-of-magnitude, removes the problem, and moves the
   project off "runs on his laptop".
3. **Two models.** `base.en` live for the conversation, `small.en`
   afterwards for the transcript the assessment is built from. Attractive
   until you notice the conversation would then be replying to a sentence
   it misheard.

## Speech, measured

| Voice | Median | p90 | Fits the 400ms budget? |
|---|---:|---:|---|
| `SilentVoice` (default) | 0.0ms | 0.1ms | Trivially — it synthesises nothing |
| `SapiVoice` (Windows) | 749ms | 780ms | No, by 1.9× |

Almost all of the SAPI figure is process startup: it shells out to
PowerShell, which loads .NET, which loads the speech stack, once per
utterance. Piper as a resident process would replace that with model
load once and a few tens of milliseconds per turn. The subprocess-per-
utterance design is the thing to fix, not the voice.

Measuring this is also how the Windows voice was found to be broken:
every call had been failing with a PowerShell parse error, because
`-Command` appends trailing arguments to the command text instead of
binding them to `$args`. It had no unit test, because a unit test of a
subprocess wrapper is a test of the mock.

## Reply

Not measured. The budget is 700ms to the last token of a two-sentence
answer, which is why `MAX_TOKENS` is 120 in
[`app/conversation/claude.py`](../app/conversation/claude.py) and why the
prompt asks for one or two sentences: generation time is roughly linear
in output length, and a chatty tutor is a slow one twice over — once
generating and once speaking.

## Where that leaves it

| Configuration | Recognition | Reply | Speech | Total |
|---|---:|---:|---:|---:|
| `base.en` + silent voice | 0.72s | — | 0.00s | **0.72s** |
| `small.en` + silent voice | 2.38s | — | 0.00s | **2.38s** |
| `small.en` + SAPI | 2.38s | — | 0.75s | **3.13s** |

The honest summary: conversation mode works and is measured, and at the
model size the transcripts are actually usable at, it is about 50% over
budget before a model has said a word. Every turn reports its own timing
in the UI and in the JSON response, including which leg is furthest over
its share, so this does not have to be remeasured by hand to know whether
it got better.

## Reproducing this

```bash
# Recognition: comes out of the fidelity matrix
python -m experiments.whisper_fidelity.run --model small.en
python - <<'EOF'
import json, statistics
rows = [json.loads(line) for line in
        open("experiments/whisper_fidelity/results/rows.jsonl")]
times = [r["decode_seconds"] for r in rows if r["model"] == "small.en"]
print(statistics.median(times))
EOF
```

Speech and the whole turn are timed on every request; the numbers are in
the response body of `POST /talk` and on the page.
