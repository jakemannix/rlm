# Sleep-time memory consolidation

*Can you take the "memories" an AI assistant has accumulated about a person, and
train them into a small model's weights so it just **behaves** accordingly —
without pasting those memories into every prompt?*

That is the whole question this part of the project is trying to answer.

## The problem, in plain terms

Today's AI assistants start every conversation as a blank slate. To make one
"remember" that you prefer terse answers, or that it should double-check a CLI
flag before claiming it exists, you either re-explain it every time or stuff a
growing list of notes into the prompt. That list isn't free — it costs context
window, it's brittle, and the model often ignores it.

Humans don't work that way. We have experiences during the day and, during
sleep, consolidate the useful ones into how we *are* — we don't re-read a diary
every morning to remember we dislike cilantro. This project asks whether an
assistant can do the same: take the durable lessons from its history with a user
and **fold them into its weights overnight**, so the behavior is just there the
next day, with nothing in the prompt.

If that works, it's a path to assistants that genuinely adapt to a person over
time, cheaply and permanently, instead of carrying an ever-larger sticky-note
pile.

## How it works

The pipeline turns a person's real chat history into training signal and then
measures whether training actually changed behavior:

```
chat history ──► extract "memories" ──► write apply-examples ──► fine-tune ──► test on UNSEEN memories
(real convos)     (durable, reusable    (a realistic situation   (small LoRA   (did behavior change on
                   behavioral lessons)    + the ideal response)    adapter)      memories it never trained on?)
```

1. **Extract memories.** From a user's conversation archive, a strong model
   pulls out *durable behavioral lessons* — not topic summaries, but reusable
   dispositions like "verify a flag with `--help` before asserting it exists" or
   "don't pretend to have read a file you can't see; ask for it."
2. **Write apply-examples.** For each memory, generate one realistic *new*
   situation where that memory should change the assistant's behavior, plus the
   ideal response that visibly applies it. This `(memory, scenario, response)`
   triple is one training example.
3. **Fine-tune.** Train a small open model on these examples with a LoRA adapter
   (a lightweight, cheap-to-train weight patch — the "consolidation").
4. **Test transfer.** Here's the key: evaluate on a **held-out set of memories
   the model was never trained on**. If the trained model applies *those* better
   than the untrained one, it didn't just memorize — it learned a general
   "apply your memory" disposition. That's the thing worth having.

The code lives in `rlm/sleep/`; the two entry-point scripts are
`scripts/sleep/gen_breadth_data.py` (steps 1–2) and
`scripts/sleep/run_uptake_experiment.py` (steps 3–4).

## What we've found so far

**It works, and it gets better with model size — but you have to train on *many
distinct* memories, not a few memories repeated.**

An early version trained on only ~50 memories (with several examples each) on a
single small model, and **failed**: training didn't change behavior. The fix was
*breadth* — train on **~555 distinct memories**, one example apiece — plus
**bigger models**. With that recipe, evaluated on 71 held-out memories the model
never saw:

| model | size | behavior gain on held-out memories | preferred over the untrained model |
|-------|------|------------------------------------|-------------------------------------|
| Qwen3-4B | 4B | +4–6 points | 59–68% of the time |
| Olmo-3-7B | 7B | +8–9 points | 67–71% |
| Olmo-3.1-32B | 32B | +11–13 points | **73–75%** |

The effect **grows monotonically with model size**, and the bigger model also
applies memories *more selectively* (it doesn't start firing them when it
shouldn't). Figure: `docs/figures/breadth_size_sweep.png`.

### Why you can trust it (and what's still soft)

We deliberately tried to break this result before believing it:

- **Two independent judges** (from different model families) grade the behavior;
  both agree on the trend, and they agree at the level of individual examples.
- **A blind, order-randomized preference test** (trained vs. untrained responses)
  is the headline metric, and it's statistically significant at every size.
- **Negative controls** (situations where *no* memory should fire) confirm the
  trained model isn't just becoming generically pushier — over-application stays
  near zero.
- It's **not a length trick**: the trained model's answers are actually *shorter*.
- A red-team pass pressure-tested the remaining confounds (architecture, judge
  bias, leakage) and none overturned it.

Honest caveats: the **absolute** rate is still low (even the best model applies
the memory in only ~1 in 4 cases — there's a lot of headroom); the strongest
clean evidence is the preference test (the raw "did it apply it" rate is noisier
at the two smaller models); and "held-out" memories still come from the same
person's history, so they share a flavor with the training ones. The full,
calibrated write-up — numbers, statistics, and caveats — is in
**`docs/memory_experiment_status.md`** (see the "Phase E" section).

## Where to look

- **`docs/memory_experiment_status.md`** — the detailed experiment log and the
  calibrated conclusion. Start here for the real numbers.
- **`docs/sleep_consolidation.md`** — the mechanism/design of the consolidation
  loop (gate → judge → train → eval).
- **`docs/figures/breadth_size_sweep.png`**, **`docs/data/breadth_size_sweep_2026-06-13.csv`**
  — the headline result, plotted and tabulated.
- **`rlm/sleep/`** — the library (extraction, dataset assembly, LoRA training,
  judging, evaluation). **`scripts/sleep/`** — the runnable pipeline.

## Open questions / what's next

1. **Cross architecture × size.** Confirm the scaling is about *size*, not the
   particular model family, by adding strong **dense** models in the ~14–32B
   range from a second family (e.g. Qwen3-32B, Gemma-3-27B).
2. **A human spot-check** to confirm the gain is real outside of LLM judges.
3. **Raise the absolute ceiling** — more examples per memory, more data, or a
   richer training signal — so the behavior is *useful*, not just *detectable*.
4. **The real prize**: a true time-split — train on memories through day N, then
   test whether the assistant behaves per a day-N memory on a day-N+1 task it was
   never told about. That's the actual "consolidation works" claim.
