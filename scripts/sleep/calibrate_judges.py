"""Calibrate the local judge stack against frontier-curated gold memories.

For each frontier memory candidate we build a (lesson, prompt, response)
triplet from its first authored probe and run the full local gate stack
(verify -> rubric -> refute). Two conditions:

- **matched**: the memory with its own probe scenario + exemplar response.
  Frontier curation says these are good; the local-gate keep rate is the
  false-reject calibration (broken down by tier and memory_type).
- **mismatched**: each memory paired with a *different* memory's probe
  (a derangement). The gates should reject these; the reject rate is the
  discrimination check.

Output: ``calibration_report.json`` — the data needed before trusting a
small judge to curate at scale (or fine-tuning one against this gold).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rlm.sleep.config import JudgeConfig
from rlm.sleep.gold import GoldConfig, _CallCache, label_example
from rlm.sleep.judge import ReflectionJudge
from rlm.sleep.types import Episode, Step, TrainingExample


def episode_for(memory: dict) -> Episode:
    """A minimal evidence episode so the rubric's grounding dimension has substance."""
    steps = [Step(role="user", content=e["context"][:3000]) for e in memory.get("evidence", [])[:2]]
    return Episode(
        episode_id=memory["memory_id"],
        source="frontier_audit",
        task="Calibration: judge this candidate memory example.",
        steps=steps or [Step(role="user", content=memory["memory"])],
        outcome="unknown",
    )


def gate_decision(judge, judge_lm, episode, lesson, prompt, response, gold_config, cache) -> dict:
    example = TrainingExample(
        prompt=prompt,
        response=response,
        lesson=lesson,
        source_episode_id=episode.episode_id,
    )
    example.verified = judge.verify(example)
    label = label_example(judge_lm, episode, example, gold_config, cache=cache)
    return {
        "verified": example.verified,
        "label": label.label,
        "mean_score": label.mean_score,
        "refute_keep": f"{label.refute_keep_votes}/{label.refute_votes}",
        "kept": label.label != "reject",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", default="runs/sleep/frontier_memories/prep.json")
    parser.add_argument("--authored", default="runs/sleep/frontier_memories/authored.json")
    parser.add_argument("--policy-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="runs/sleep/judge_calibration")
    args = parser.parse_args()

    memories = json.loads(Path(args.prep).read_text())
    authored = {a["memory_id"]: a for a in json.loads(Path(args.authored).read_text())}
    items = [m for m in memories if m["memory_id"] in authored]
    if args.limit:
        items = items[: args.limit]

    from rlm.sleep.local_judge import LocalHFJudge

    judge_lm = LocalHFJudge.from_policy(
        args.policy_model, device=args.device, torch_dtype=args.torch_dtype
    )
    judge = ReflectionJudge(judge_lm, JudgeConfig())
    gold_config = GoldConfig()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = _CallCache(out / "gold_cache.json")

    rows = []
    n = len(items)
    for i, memory in enumerate(items):
        mid = memory["memory_id"]
        probe = authored[mid]["probes"][0]
        episode = episode_for(memory)
        matched = gate_decision(
            judge,
            judge_lm,
            episode,
            memory["memory"],
            probe["scenario"],
            probe["exemplar_response"],
            gold_config,
            cache,
        )
        other = authored[items[(i + 1) % n]["memory_id"]]["probes"][0]
        mismatched = gate_decision(
            judge,
            judge_lm,
            episode,
            memory["memory"],
            other["scenario"],
            other["exemplar_response"],
            gold_config,
            cache,
        )
        rows.append(
            {
                "memory_id": mid,
                "tier": memory["tier"],
                "memory_type": memory["memory_type"],
                "matched": matched,
                "mismatched": mismatched,
            }
        )
        print(
            f"[{i + 1}/{n}] {mid} matched={matched['label']} mismatched={mismatched['label']}",
            flush=True,
        )

    def rate(subset, cond) -> str:
        if not subset:
            return "n/a"
        kept = sum(1 for r in subset if r[cond]["kept"])
        return f"{kept}/{len(subset)} ({100 * kept / len(subset):.0f}%)"

    report = {
        "n": len(rows),
        "matched_keep_rate": rate(rows, "matched"),
        "mismatched_keep_rate": rate(rows, "mismatched"),
        "matched_keep_by_tier": {
            t: rate([r for r in rows if r["tier"] == t], "matched") for t in ("gold", "silver")
        },
        "matched_keep_by_type": {},
        "rows": rows,
    }
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["memory_type"]].append(r)
    report["matched_keep_by_type"] = {t: rate(v, "matched") for t, v in sorted(by_type.items())}
    (out / "calibration_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
