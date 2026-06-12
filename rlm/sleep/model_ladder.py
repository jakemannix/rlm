"""Model-quality ladder: where is the elbow for memory curation?

Sweeps a ladder of models (local SLMs through frontier, via OpenRouter)
on two targeted tasks against the frontier-curated gold set:

1. **classify** — single-call keep/reject + tier judgment on matched
   (frontier-approved) and derangement-mismatched triplets. Single-call
   by design: it measures the *model's* curation judgment, isolated from
   any multi-call gate-stack design.
2. **generate** — evidence context -> distilled memory, graded against
   the frontier reference by a fixed referee model.

All calls are disk-cached per (model, prompt), so reruns and ladder
extensions only pay for new cells.
"""

from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from rlm.clients.base_lm import BaseLM
from rlm.sleep.gold import _CallCache
from rlm.sleep.judge import extract_json

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

CLASSIFY_PROMPT = """\
You are curating training examples for a personal agent's long-term memory. \
Decide whether this candidate is worth training on.

CANDIDATE MEMORY (the durable lesson):
{memory}

SITUATION (a user message where the memory might apply):
{scenario}

PROPOSED RESPONSE (what the example would teach the agent to do):
{response}

EVIDENCE EXCERPT (where the memory came from):
{evidence}

Judge: (a) is the memory itself durable and worth keeping, AND (b) does the \
proposed response correctly apply THIS memory to THIS situation? A good \
memory paired with a response that ignores or contradicts it must be \
rejected.

Respond with ONLY a JSON object:
{{"keep": true or false,
  "tier": "gold" or "silver" or "reject",
  "reason": "<one short sentence>"}}
"""

GENERATE_PROMPT = """\
Below is evidence from one or more of a user's past conversations with an \
AI assistant.

{evidence}

Distill the single most durable, reusable memory a future agent working \
with this user should retain from this evidence. State it as one concrete, \
actionable note (1-3 sentences). Respond with ONLY the memory text.
"""

GRADE_PROMPT = """\
You are grading a model-generated memory against a reference memory that \
expert curation produced from the same evidence.

REFERENCE MEMORY:
{reference}

GENERATED MEMORY:
{generated}

Score the generated memory:
- coverage (1-5): captures the reference's core insight (5 = same insight; \
1 = misses it entirely)
- specificity (1-5): concrete and actionable vs vague platitude
- groundedness (1-5): plausibly derivable from the same evidence, no invention

Respond with ONLY a JSON object:
{{"coverage": n, "specificity": n, "groundedness": n,
  "equivalent": true or false,
  "comment": "<one short sentence>"}}
"""


@dataclass
class LadderItem:
    """One memory candidate prepared for ladder tasks."""

    memory_id: str
    tier: str
    memory_type: str
    memory: str
    evidence: str
    scenario: str
    response: str
    mismatched_scenario: str
    mismatched_response: str


def build_items(
    prep: list[dict], authored: dict[str, dict], evidence_chars: int
) -> list[LadderItem]:
    """Matched + derangement-mismatched triplets from prep/authored artifacts."""
    usable = [m for m in prep if m["memory_id"] in authored]
    items: list[LadderItem] = []
    n = len(usable)
    for i, memory in enumerate(usable):
        probe = authored[memory["memory_id"]]["probes"][0]
        other = authored[usable[(i + 1) % n]["memory_id"]]["probes"][0]
        evidence = "\n\n".join(e["context"] for e in memory.get("evidence", [])[:2])
        items.append(
            LadderItem(
                memory_id=memory["memory_id"],
                tier=memory["tier"],
                memory_type=memory["memory_type"],
                memory=memory["memory"],
                evidence=evidence[:evidence_chars],
                scenario=probe["scenario"],
                response=probe["exemplar_response"],
                mismatched_scenario=other["scenario"],
                mismatched_response=other["exemplar_response"],
            )
        )
    return items


@dataclass
class ModelResult:
    model: str
    classify_rows: list[dict] = field(default_factory=list)
    generate_rows: list[dict] = field(default_factory=list)

    def classify_metrics(self) -> dict:
        rows = self.classify_rows
        matched = [r for r in rows if r["condition"] == "matched" and not r.get("parse_error")]
        mismatched = [
            r for r in rows if r["condition"] == "mismatched" and not r.get("parse_error")
        ]
        parse_errors = sum(1 for r in rows if r.get("parse_error"))

        def keep_rate(subset):
            return sum(1 for r in subset if r["keep"]) / len(subset) if subset else None

        matched_keep = keep_rate(matched)
        mismatched_keep = keep_rate(mismatched)
        gold_matched = [r for r in matched if r["frontier_tier"] == "gold"]
        tier_match = (
            sum(1 for r in gold_matched if r.get("tier") == "gold") / len(gold_matched)
            if gold_matched
            else None
        )
        balanced = (
            ((matched_keep or 0) + (1 - (mismatched_keep or 1))) / 2
            if matched and mismatched
            else None
        )
        return {
            "n": len(rows),
            "parse_errors": parse_errors,
            "matched_keep": matched_keep,
            "mismatched_keep": mismatched_keep,
            "balanced_accuracy": balanced,
            "gold_tier_agreement": tier_match,
        }

    def generate_metrics(self) -> dict:
        rows = [r for r in self.generate_rows if not r.get("parse_error")]
        if not rows:
            return {"n": 0, "parse_errors": len(self.generate_rows)}
        return {
            "n": len(rows),
            "parse_errors": len(self.generate_rows) - len(rows),
            "coverage": round(statistics.mean(r["coverage"] for r in rows), 2),
            "specificity": round(statistics.mean(r["specificity"] for r in rows), 2),
            "groundedness": round(statistics.mean(r["groundedness"] for r in rows), 2),
            "equivalent_rate": round(sum(1 for r in rows if r["equivalent"]) / len(rows), 3),
        }


def _completion(lm: BaseLM, cache: _CallCache, kind: str, prompt: str) -> str:
    return cache.completions(lm, kind, prompt, 1)[0]


def classify_item(lm: BaseLM, cache: _CallCache, item: LadderItem, condition: str) -> dict:
    scenario = item.scenario if condition == "matched" else item.mismatched_scenario
    response = item.response if condition == "matched" else item.mismatched_response
    prompt = CLASSIFY_PROMPT.format(
        memory=item.memory, scenario=scenario, response=response, evidence=item.evidence
    )
    row = {
        "memory_id": item.memory_id,
        "condition": condition,
        "frontier_tier": item.tier,
        "memory_type": item.memory_type,
    }
    try:
        parsed = extract_json(_completion(lm, cache, "classify", prompt))
        row |= {
            "keep": bool(parsed.get("keep")),
            "tier": str(parsed.get("tier", "")),
            "reason": str(parsed.get("reason", ""))[:300],
        }
    except (ValueError, json.JSONDecodeError) as exc:
        row |= {"keep": False, "parse_error": str(exc)[:120]}
    return row


def generate_and_grade(lm: BaseLM, referee: BaseLM, cache: _CallCache, item: LadderItem) -> dict:
    generated = _completion(
        lm, cache, "generate", GENERATE_PROMPT.format(evidence=item.evidence)
    ).strip()
    row = {
        "memory_id": item.memory_id,
        "frontier_tier": item.tier,
        "memory_type": item.memory_type,
        "generated": generated[:1500],
    }
    try:
        parsed = extract_json(
            _completion(
                referee,
                cache,
                "grade",
                GRADE_PROMPT.format(reference=item.memory, generated=generated[:2000]),
            )
        )
        row |= {
            "coverage": float(parsed["coverage"]),
            "specificity": float(parsed["specificity"]),
            "groundedness": float(parsed["groundedness"]),
            "equivalent": bool(parsed.get("equivalent")),
        }
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        row |= {"parse_error": str(exc)[:120]}
    return row


def run_model(
    lm: BaseLM,
    items: list[LadderItem],
    cache: _CallCache,
    tasks: tuple[str, ...] = ("classify", "generate"),
    referee: BaseLM | None = None,
    workers: int = 6,
) -> ModelResult:
    result = ModelResult(model=lm.model_name)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if "classify" in tasks:
            jobs = [(item, cond) for item in items for cond in ("matched", "mismatched")]
            result.classify_rows = list(
                pool.map(lambda job: classify_item(lm, cache, job[0], job[1]), jobs)
            )
        if "generate" in tasks and referee is not None:
            result.generate_rows = list(
                pool.map(lambda item: generate_and_grade(lm, referee, cache, item), items)
            )
    return result
