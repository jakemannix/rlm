"""Gold-labeling: a thorough LLM-as-a-Judge pass over distilled memories.

The nightly loop's judge (``judge.py``) optimizes for cheap online
curation: one reflection, one YES/NO verification. This module is the
offline, expensive version for building a *labeled dataset* of memory
candidates from rich traces (e.g. ``cc_traces`` episodes):

1. **Reflect** — reuse :class:`~rlm.sleep.judge.ReflectionJudge`
   (self-consistency sampling + calibrated verification, disk-cached).
2. **Rubric** — each verified example is scored 1–5 on five dimensions
   (correctness, reusability, grounding, specificity, self-containment)
   by ``rubric_samples`` independent judge calls; scores are averaged.
3. **Refute** — ``refuter_votes`` adversarial judges each try to find a
   concrete flaw; a majority must vote keep. Sampled (not greedy): the
   value of multiple refuters is diversity of attack angles.
4. **Label** — ``gold`` / ``silver`` / ``reject`` from the aggregate.

Everything that informed a label is kept on the :class:`GoldLabel` so
downstream filtering can re-threshold without re-spending judge compute.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rlm.clients.base_lm import BaseLM
from rlm.sleep.config import JudgeConfig
from rlm.sleep.judge import ReflectionJudge, extract_json, parse_verify_verdict
from rlm.sleep.types import Episode, TrainingExample

RUBRIC_DIMENSIONS = ("correctness", "reusability", "grounding", "specificity", "self_contained")

RUBRIC_PROMPT = """\
You are scoring one candidate training example that was distilled from an \
agent episode. Score strictly — 5s should be rare.

EPISODE EVIDENCE (truncated):
{transcript}

CANDIDATE:
LESSON: {lesson}
PROMPT: {prompt}
RESPONSE: {response}

Score each dimension from 1 (bad) to 5 (excellent):
- correctness: the response is factually and technically right (1 = wrong or harmful)
- reusability: the lesson generalizes beyond this one session (1 = session-specific trivia)
- grounding: the episode evidence actually shows the mistake or insight the lesson claims (1 = invented)
- specificity: concrete and actionable (1 = a platitude like "be careful")
- self_contained: the prompt and response make sense without reading the episode (1 = references unseen context)

Calibration anchor: a lesson "When a pip install fails with a compiler \
error, check that the Python version matches the package's requires-python" \
with a prompt asking about that failure and a response giving the exact \
check would score correctness 5, reusability 4, grounding 5 (if the episode \
shows that failure), specificity 4, self_contained 5.

Respond with ONLY a JSON object:
{{"scores": {{"correctness": n, "reusability": n, "grounding": n, "specificity": n, "self_contained": n}},
  "fatal_flaw": "<one short sentence if any single problem disqualifies this example, else an empty string>"}}
"""

REFUTE_PROMPT = """\
You are the last gate before a training example is written into a model's \
weights. Try to refute it: find a concrete reason it teaches something \
wrong, useless, or confusing. If your strongest objection is a real flaw, \
reject it. If you cannot find a real flaw, keep it — rejecting good \
examples also hurts the model.

Two reference reviews:

LESSON: Always be careful when running commands.
PROMPT: How should I run commands?
RESPONSE: Carefully, thinking about what could go wrong.
OBJECTION: Content-free platitude — nothing here changes any future action.
VERDICT: NO

LESSON: git push --force-with-lease refuses to clobber commits you haven't seen.
PROMPT: I need to force-push my rebased branch but a teammate may have pushed too.
RESPONSE: Use `git push --force-with-lease` — it fails if the remote moved since your last fetch, unlike `--force`.
OBJECTION: None that holds; the distinction is real and the advice is safe.
VERDICT: YES

Now review this candidate:

LESSON: {lesson}
PROMPT: {prompt}
RESPONSE: {response}

State your strongest objection in one sentence, then give your verdict on \
the last line in exactly this form:
VERDICT: YES
or
VERDICT: NO
"""


@dataclass
class GoldConfig:
    """Knobs for the offline gold-labeling pass."""

    rubric_samples: int = 3
    refuter_votes: int = 3
    gold_min_score: float = 4.0
    silver_min_score: float = 3.0
    max_transcript_chars: int = 4000


@dataclass
class GoldLabel:
    """One candidate memory with everything that informed its label."""

    episode_id: str
    example: TrainingExample
    label: str = "reject"  # gold | silver | reject
    verified: bool = False
    scores: dict[str, float] = field(default_factory=dict)
    mean_score: float = 0.0
    fatal_flaws: list[str] = field(default_factory=list)
    n_rubric_failures: int = 0
    refute_keep_votes: int = 0
    refute_votes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def score_rubric(
    judge_lm: BaseLM, episode: Episode, example: TrainingExample, config: GoldConfig
) -> tuple[dict[str, float], list[str], int]:
    """Run ``rubric_samples`` scorings; return (mean scores, fatal flaws, parse failures)."""
    prompt = RUBRIC_PROMPT.format(
        transcript=episode.transcript(max_chars=config.max_transcript_chars),
        lesson=example.lesson,
        prompt=example.prompt,
        response=example.response,
    )
    per_dim: dict[str, list[float]] = {d: [] for d in RUBRIC_DIMENSIONS}
    flaws: list[str] = []
    failures = 0
    for _ in range(config.rubric_samples):
        try:
            parsed = extract_json(judge_lm.completion(prompt))
            raw = parsed.get("scores") or {}
            sample = {d: float(raw[d]) for d in RUBRIC_DIMENSIONS}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            failures += 1
            continue
        for dim, value in sample.items():
            per_dim[dim].append(min(5.0, max(1.0, value)))
        flaw = str(parsed.get("fatal_flaw") or "").strip()
        if flaw:
            flaws.append(flaw)
    means = {d: round(statistics.mean(v), 2) for d, v in per_dim.items() if v}
    return means, flaws, failures


def refute(judge_lm: BaseLM, example: TrainingExample, config: GoldConfig) -> tuple[int, int]:
    """Adversarial keep/reject votes; returns (keep_votes, total_votes)."""
    prompt = REFUTE_PROMPT.format(
        lesson=example.lesson, prompt=example.prompt, response=example.response
    )
    keep = sum(
        parse_verify_verdict(judge_lm.completion(prompt)) for _ in range(config.refuter_votes)
    )
    return keep, config.refuter_votes


def label_example(
    judge_lm: BaseLM, episode: Episode, example: TrainingExample, config: GoldConfig
) -> GoldLabel:
    """Rubric + refute one verified example and assign its label."""
    result = GoldLabel(episode_id=episode.episode_id, example=example, verified=example.verified)
    if not example.verified:
        return result  # reject without spending rubric/refuter compute

    result.scores, result.fatal_flaws, result.n_rubric_failures = score_rubric(
        judge_lm, episode, example, config
    )
    if len(result.scores) < len(RUBRIC_DIMENSIONS):
        return result  # every sample failed to parse at least one dimension
    result.mean_score = round(statistics.mean(result.scores.values()), 3)

    # A fatal flaw claimed by a majority of rubric samples disqualifies.
    parsed_samples = config.rubric_samples - result.n_rubric_failures
    if parsed_samples and len(result.fatal_flaws) * 2 > parsed_samples:
        return result

    result.refute_keep_votes, result.refute_votes = refute(judge_lm, example, config)
    if result.refute_keep_votes * 2 <= result.refute_votes:
        return result

    if result.mean_score >= config.gold_min_score:
        result.label = "gold"
    elif result.mean_score >= config.silver_min_score:
        result.label = "silver"
    return result


def label_episodes(
    episodes: list[Episode],
    judge_lm: BaseLM,
    judge_config: JudgeConfig | None = None,
    gold_config: GoldConfig | None = None,
    cache_path: str | Path | None = None,
) -> list[GoldLabel]:
    """The full pass: reflect -> verify -> rubric -> refute over many episodes."""
    judge_config = judge_config or JudgeConfig()
    gold_config = gold_config or GoldConfig()
    reflector = ReflectionJudge(judge_lm, judge_config, cache_path=cache_path)
    labels: list[GoldLabel] = []
    for episode in episodes:
        output = reflector.reflect(episode)
        for example in output.examples:
            labels.append(label_example(judge_lm, episode, example, gold_config))
    return labels


def summarize(labels: list[GoldLabel]) -> dict:
    """Aggregate stats for a labeling run (printed and persisted by the CLI)."""
    by_label = {
        name: sum(1 for label in labels if label.label == name)
        for name in ("gold", "silver", "reject")
    }
    scored = [label for label in labels if label.scores]
    return {
        "n_candidates": len(labels),
        **{f"n_{k}": v for k, v in by_label.items()},
        "n_unverified": sum(1 for label in labels if not label.verified),
        "n_refuted": sum(
            1
            for label in labels
            if label.refute_votes and label.refute_keep_votes * 2 <= label.refute_votes
        ),
        "mean_score_overall": round(statistics.mean(label.mean_score for label in scored), 3)
        if scored
        else None,
        "dimension_means": {
            dim: round(
                statistics.mean(label.scores[dim] for label in scored if dim in label.scores), 2
            )
            for dim in RUBRIC_DIMENSIONS
        }
        if scored
        else {},
    }
