# DandeLyon Calliope

[![CI](https://github.com/SKuger/DandeLyon-Calliope/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SKuger/DandeLyon-Calliope/actions/workflows/ci.yml)

**A spoken English trainer for one person, built around a measurement
rather than a chatbot.**

It records him talking, transcribes it locally, computes the same
arithmetic over every transcript, keeps the mistakes it finds in a
personal corpus, and schedules them for practice. The numbers on the
dashboard are calculated by code. No model rates anything.

```
audio ──► Transcriber ──► deterministic metrics ──┐
            (local)            (arithmetic)        ├──► SQLite
              │                                    │      │
              └──► Assessor ───────────────────────┘      │
                   (rules, or a model)                    ▼
                                                   error corpus
                                                          │
                                                          ▼
                                             exercises + spaced repetition
```

It runs with no API key, no model weights and no network: `docker compose
up` gives a working dashboard, a working review queue and an honest error
message the moment you ask it to transcribe something. Install one extra
requirements file and the same code does real speech recognition.

## The measurement it is built around

Whisper is trained to produce plausible text, not to document what was
said. If it repairs *"yesterday I go to the office"* into *"yesterday I
went"*, the evidence a tutor needs is gone and the whole project is a
chatbot that tells its user he speaks perfectly.

So nothing was built until that was measured. Eighteen sentences — fourteen
carrying one planted error each, four already correct — spoken by four
voices, transcribed by three model sizes under three decoding
configurations: 648 transcriptions, written up in
[`docs/whisper-grammar-fidelity.md`](docs/whisper-grammar-fidelity.md)
with the transcripts kept verbatim.

**The worry was real, small, and not the problem.**

| | Clean audio | Heavily accented audio |
|---|---:|---:|
| Planted error repaired by Whisper | 6% | 18% |
| Planted error survived | 94% | 41% |
| Heard as a different sentence | 0% | 41% |
| **Correct sentence altered** | **7%** | **94%** |

On clean audio Whisper leaves the mistakes alone at every model size and
every decoding setting. On accented audio it does not repair the
grammar so much as fail to hear the sentence — and it rewrites sentences
that were *already correct* nineteen times out of twenty.

That inverted the design. The danger is not that the system misses his
mistakes; it is that it invents them, and a tutor that corrects sentences
he said correctly is worse than no tutor, because he stops reading the
corrections that were real.

Two smaller findings from the same run, both of which changed code:

- **Decoding settings do not matter here.** The three configurations
  produce the same repair rate to one decimal place. Model size moves it;
  temperature and conditioning do not. Priming the decoder with
  ungrammatical learner English — the most aggressive version of the idea
  — made things *worse*, damaging clean transcripts.
- **Confidence knows when it is guessing.** A word Whisper invented has a
  median confidence of 0.47; a word it heard, 0.94. That is the signal the
  assessor's guard runs on, and the threshold is that measurement rather
  than a round number.

## The seven problems this solves

**1. A number that means something six months from now.**
Everything charted is computed from the words and their timestamps
([`app/metrics/fluency.py`](app/metrics/fluency.py)): words per minute,
words per utterance, lexical diversity, fillers, Spanish crutches,
restarts, pause distribution, which tenses he used and which he never
reached for. A model asked to rate fluency out of ten returns a different
ten tomorrow for the same audio, and six months of those is a mood diary.

Two decisions inside that are less obvious than the rule:

- Lexical diversity is a *moving-average* type/token ratio. The plain
  ratio falls as a monologue gets longer, so speaking for three minutes
  instead of one would read as losing vocabulary — the chart would be
  telling him to talk less.
- Unmeasurable is `None`, never `0.0`. A recording with no word timings
  has no pause distribution, and a confident flat line at zero is
  indistinguishable from a real result.

**2. An exam that is still the same exam in March.**
This is where placement tests usually fail: they regenerate, so the
"improvement" after six months is the difference between two exams plus
the difference in the speaker, with nothing separating the terms. Here
the exam is a file with a version and a content fingerprint
([`app/placement/`](app/placement)), every sitting stores that
fingerprint, and comparing two sittings that used different exams is
**refused** rather than reported as progress.

Four bands, one per skill, never one number: his profile is uneven by
design and an average would rise whenever the strong half improved. The
absolute band is an opinion — calibrated by judgement against the CEFR
descriptors, because there is one test-taker. That the same answers
always produce the same band is not.

**3. The same recording counted twice.**
The session id is the hash of the audio
([`app/store.py`](app/store.py)), so a retried upload, a double-clicked
button or a re-import lands on the same row and the transcriber is never
run twice. Not hygiene: every chart is a series over sessions and the
corpus counts repetitions to decide what to practise, so a duplicate
tells him he makes a mistake twice as often as he does — and then the
system spends his evenings on it.

One level down the rule inverts. Error ids are stable *within* a session,
so re-assessing Tuesday with a better prompt improves the corpus instead
of duplicating it; the same mistake made again on Friday is a second row,
because that repetition is the only evidence an error is worth practising.

**4. A correction of something he never said.**
Two filters sit between any assessor and the corpus
([`app/assess/base.py`](app/assess/base.py)):

- **It has to quote the transcript.** Asked to correct English, a model
  paraphrases the sentence and then corrects the paraphrase. The result
  reads perfectly and teaches nothing, because he cannot recognise it as
  his. Anything whose quoted phrase is not literally in the transcript is
  dropped.
- **It has to be about words the recogniser heard.** The finding above,
  turned into code: a correction over a span whose mean word confidence
  is below 0.55 is discarded. That drops 62% of invented words and takes
  13% of real ones with them, which is the trade worth making — a missed
  mistake will be made again next week, an invented one gets practised.

**5. A tutor with no curriculum.**
There is no course in this repository and there is not going to be one.
An exercise is one row of the error corpus handed back with an
instruction to say it properly out loud, so the material is whatever he
actually got wrong, and it stops being generated the day he stops getting
it wrong.

The default assessor is offline rules
([`app/assess/rules.py`](app/assess/rules.py)) — twenty patterns that a
Spanish speaker's English produces and an English speaker's does not.
They catch thirteen of the fourteen planted errors and stay silent on
every control sentence. The fourteenth is why the `Assessor` Protocol has
a second implementation: *"the client is very sensible about the response
times"* is grammatical English that means the opposite of what he
intended, and no pattern over text can know that. There is a test
asserting that miss rather than a comment hoping nobody notices it.

**6. Avoidance, which looks exactly like success.**
An intermediate speaker who cannot say *"I have been working here for
three months"* says *"I work here"* instead and never appears to make a
mistake. So a spoken attempt is graded on three outcomes, not two
([`app/review/practice.py`](app/review/practice.py)): he repaired it, he
repeated the mistake, or he routed around the structure — and the third
gets a passing grade too low to grow the interval. The grading is
deterministic because the grade sets the review interval, and a model in
that path would make the spacing move with the model's mood.

Scheduling is SM-2 behind a Protocol. FSRS is better and its advantage
comes from parameters fitted to millions of reviews; this deck is one
person's few hundred errors, so shipping FSRS would mean shipping its
default weights — a guess with seventeen more moving parts and no data to
fit them with.

**7. "You have made this mistake three times in two weeks."**
That sentence sends him to spend an evening on something, so it is
counted rather than retrieved: a `GROUP BY` over the corpus, matched on
the normalised phrase and the category
([`app/history.py`](app/history.py)). Retrieval exists too, over his own
past utterances, and it never makes a claim to the user — it hands the
assessor a few things he has said before so feedback can recognise a
habit. TF-IDF with a score floor, below which it returns nothing, plus
one rule that only matters when the corpus is the user's own speech: an
utterance is excluded from its own results. Without it, the best match
for today's sentence is today's sentence, and the model is handed proof
of a habit on the first occasion he ever said it.

## Conversation, and the latency it misses

Above roughly two seconds a spoken exchange stops feeling like talking
and starts feeling like filling in a form. Every turn is timed per leg —
recognition, reply, speech — and reports which leg is furthest over its
share, because "it is slow" does not tell anyone what to fix.

Measured on a Ryzen 5 5600X, CPU only
([`docs/latency.md`](docs/latency.md)):

| Leg | Budget | Measured |
|---|---:|---:|
| Recognition, `base.en` | 900ms | 720ms |
| Recognition, `small.en` | 900ms | 2380ms |
| Reply | 700ms | not measured — no key on this machine |
| Speech, Windows SAPI | 400ms | 749ms |

**It misses.** `base.en` fits the budget and is the model the fidelity
experiment found unusable on accented speech; `small.en` is usable and is
2.6× over. The model that is fast enough is not accurate enough and the
model that is accurate enough is not fast enough, on this CPU, without
streaming — and the legs here are not streamed. Streaming the recogniser
is the one change that would fix it, and it is the change this repository
has not made.

The conversational partner never corrects anything, and its prompt says
so twice. A model asked to talk with a learner starts teaching within
three turns, and the correction that lands mid-sentence is the one that
makes him stop talking. Corrections happen afterwards, from the
transcript, where they can be checked.

## Privacy

The recordings are a real person's voice talking about his life.

- Speech recognition is local and the audio never leaves the machine.
  `recordings/`, `data/` and the database are gitignored — they were in
  the first commit, before there was anything to ignore.
- Recordings are stored under the hash of their bytes, so a filename
  carries no date, no prompt and nothing about what he said.
- A cloud assessor is opt-in, off by default, and `/health` reports
  whether the current configuration sends any text off the machine —
  including via conversation mode. "Local by default" should be checkable
  rather than trusted.
- Every transcript published in this repository was written for the
  purpose. The sample week in [`samples/week.json`](samples/week.json)
  says so in the file, and there is a test asserting that it says so.

## Run it

```bash
cp .env.example .env
docker compose up --build
```

Open http://localhost:8000. It starts seeded with seven invented practice
sessions so the dashboard has a shape; they run through the real
pipeline, so every number on it is real arithmetic over fake input.

That build has no speech recognition in it — 200MB instead of 2GB, and no
model weights downloaded at build time. Press record and it answers with
a 503 that explains itself and keeps your audio. For the real thing:

```bash
pip install -r requirements-dev.txt -r requirements-asr.txt
uvicorn app.main:app --reload
```

The first recording downloads `small.en` (~250MB) and takes a few seconds
longer than the rest. `small.en` is the floor for accented speech;
`base.en` is measurably not good enough and `medium.en` is three times
slower for a couple of points of word error rate.

## Tests

```bash
pytest
```

285 tests, no network, no keys, no model weights — CI runs the same
command. They are written against the failure modes:

- A transcriber that does not know a recording **raises** instead of
  returning plausible text, and there is a test for the error message.
- A correction quoting a sentence that is not in the transcript is
  dropped. So is one over words the recogniser was unsure of.
- The same upload twice produces one session, one set of errors, one
  review schedule, and one call to the transcriber.
- An assessor that raises loses the feedback and keeps the session; a
  transcriber that fails stores nothing at all rather than an empty
  session that would put a zero on every chart.
- Lexical diversity does not fall when the same vocabulary is spoken for
  four times as long.
- The tense heuristic's known blind spots are pinned as tests, including
  the one where it over-reports the present simple.
- The offline rules' inability to see a false friend is asserted, because
  it is the reason the assessor boundary has two implementations.
- Comparing two placement sittings that used different exams is refused.

CI runs a second job that installs the speech recognition extra alongside
the web stack, because faster-whisper brings CTranslate2, tokenizers and
ONNX Runtime, and a version conflict only appears when someone installs
both into one environment — which is exactly what the instructions above
tell you to do.

## Configuration

| Variable | Effect when unset |
|---|---|
| `ANTHROPIC_API_KEY` | the offline rule-based assessor runs instead of a model |
| `ASSESSOR` | `rules`; also `claude`, `both`, `none` |
| `TRANSCRIBER` | `whisper`, falling back to replaying fixtures when faster-whisper is absent |
| `WHISPER_MODEL` | `small.en` |
| `REPLIER` | `claude` when a key is set, the scripted partner otherwise |
| `VOICE` | `silent`; also `piper` (needs `PIPER_MODEL`) or `sapi` |
| `DATA_DIR` / `RECORDINGS_DIR` | `data/` and `recordings/`, both gitignored |
| `EXAM_VERSION` | `v1` |

## Branches

`main` is what would be in production. `develop` is where changes land
and are tested. Work happens on `develop`; releasing is a pull request
into `main`, which is protected and needs both CI jobs green. If it is on
`main`, it survived `develop` first.

## Layout

```
app/
  main.py          FastAPI: pages, uploads, the review and talk endpoints
  config.py        environment, read once, with working defaults
  pipeline.py      transcribe, measure, store, assess, schedule -- in that order
  store.py         SQLite: sessions, metrics, the corpus, the schedule
  text.py          normalisation, span lookup, word error rate
  history.py       recurrence counting, and retrieval over his own speech
  demo.py          the sample week, seeded idempotently on start
  asr/             Transcriber Protocol, fixture default, faster-whisper
  metrics/         the deterministic measurements
  assess/          Assessor Protocol, the two guards, offline rules, Claude
  review/          SM-2 scheduling, exercises built from the corpus, grading
  placement/       the fixed exam and its scoring
  conversation/    the timed turn, voices, the conversational partner
  templates/       six pages of server-rendered HTML
experiments/
  whisper_fidelity/  the experiment that came before the application
docs/
  whisper-grammar-fidelity.md   what Whisper does to his grammar, with numbers
  latency.md                    where the two seconds go, including the misses
samples/week.json  invented sessions, labelled as invented
tests/             285 tests
```

## What is not done

- **Streaming recognition.** It is the one change that would bring
  conversation mode inside its latency budget. See
  [`docs/latency.md`](docs/latency.md).
- **The experiment against his real voice.** The accented columns use a
  Spanish TTS engine reading English, which overshoots a B2 speaker's
  accent. The harness takes audio from a directory, so redoing it
  properly is an afternoon, and the numbers should be read as a worst
  case until then.
- **A second, phonetic model.** Disagreement between Whisper and an ASR
  without a strong language model is the most promising thing left, and
  it doubles a latency budget that is already over.
- **A real week of use.** The sample week in this repository is invented
  and says so. The week that matters is his, and it has not happened yet.

MIT licensed.
