"""Build a frontier-labeled trainset for fine-tuning a memory-worthiness judge.

The judge learns: given a candidate memory + its evidence, keep (gold/silver)
or reject. Two sources, both with frontier ground truth:

  - the 71 frontier-curated memories (their tier is the label) — clean positives;
  - candidates *generated* from a sample of the user's claude.ai-export episodes
    by a cheap proposer, then labeled keep/reject+tier by a frontier model (Opus).
    Generating from real episodes yields a natural keep/reject mix — the rejects
    the curated set lacks. (The audit's heuristic-candidates file is NOT used: it
    is a coarse coverage map with only ~6 distinct candidate strings across 985
    rows.)

    OPENROUTER_API_KEY=... python scripts/sleep/build_judge_trainset.py \
        --n-episodes 280 --labeler-model anthropic/claude-opus-4.8
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rlm.sleep.claude_export import load_claude_export
from rlm.sleep.gold import _CallCache
from rlm.sleep.judge_worthiness import WORTHINESS_PROMPT, parse_worthiness, worthiness_messages
from rlm.sleep.model_ladder import GENERATE_PROMPT, OPENROUTER_BASE_URL, CachedJudge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", default="runs/sleep/frontier_memories/prep.json")
    parser.add_argument("--export", default="personal_chat_archive")
    parser.add_argument("--n-episodes", type=int, default=280)
    parser.add_argument("--propose-model", default="qwen/qwen3.6-35b-a3b")
    parser.add_argument("--labeler-model", default="anthropic/claude-opus-4.8")
    parser.add_argument("--evidence-chars", type=int, default=4000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="runs/sleep/judge_trainset")
    args = parser.parse_args()

    from rlm.clients.openai import OpenAIClient

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = _CallCache(out / "label_cache.json")

    def client(model: str, kind: str) -> CachedJudge:
        return CachedJudge(
            OpenAIClient(model_name=model, base_url=OPENROUTER_BASE_URL), cache, kind
        )

    proposer = client(args.propose_model, "propose")
    labeler = client(args.labeler_model, "label")

    # Positives: the 71 frontier memories (tier is the label).
    frontier = json.loads(Path(args.prep).read_text())
    labeled = [
        {
            "memory": m["memory"],
            "evidence": "\n\n".join(e["context"] for e in m.get("evidence", [])[:2])[
                : args.evidence_chars
            ],
            "label": {"keep": True, "tier": m["tier"], "reason": "frontier-curated"},
            "source": "frontier",
        }
        for m in frontier
    ]

    # Sample episodes deterministically, generate a candidate, label with Opus.
    episodes = load_claude_export(args.export)
    episodes.sort(key=lambda e: hashlib.sha256(e.episode_id.encode()).hexdigest())
    episodes = episodes[: args.n_episodes]

    def make(ep) -> dict | None:
        evidence = ep.transcript(max_chars=args.evidence_chars)
        try:
            candidate = proposer.completion(GENERATE_PROMPT.format(evidence=evidence)).strip()
            verdict = parse_worthiness(
                labeler.completion(WORTHINESS_PROMPT.format(memory=candidate, evidence=evidence))
            )
        except Exception:  # noqa: BLE001
            return None
        if verdict is None:
            return None
        return {
            "memory": candidate,
            "evidence": evidence[:3000],
            "label": verdict,
            "source": "generated",
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        labeled += [r for r in ex.map(make, episodes) if r is not None]

    print(f"labeled: {len(labeled)} ({Counter(r['label']['tier'] for r in labeled)})", flush=True)

    by_tier: dict[str, list[dict]] = {}
    for r in labeled:
        by_tier.setdefault(r["label"]["tier"], []).append(r)
    train, test = [], []
    for _tier, group in sorted(by_tier.items()):
        group.sort(key=lambda r: hashlib.sha256(r["memory"].encode()).hexdigest())
        n_test = round(len(group) * args.test_fraction)
        test += group[:n_test]
        train += group[n_test:]

    def dump(name: str, rows: list[dict], as_sft: bool) -> None:
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                obj = (
                    {
                        "messages": worthiness_messages(r["memory"], r["evidence"], r["label"]),
                        "tier": r["label"]["tier"],
                        "source": r["source"],
                    }
                    if as_sft
                    else r
                )
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    dump("trainset.jsonl", train, as_sft=True)
    dump("testset.jsonl", test, as_sft=True)
    dump("testset_raw.jsonl", test, as_sft=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "n_labeled": len(labeled),
                "n_train": len(train),
                "n_test": len(test),
                "labeler": args.labeler_model,
                "proposer": args.propose_model,
                "train_tiers": dict(Counter(r["label"]["tier"] for r in train)),
                "test_tiers": dict(Counter(r["label"]["tier"] for r in test)),
            },
            indent=2,
        )
    )
    print(f"train {len(train)} / test {len(test)} -> {out}")


if __name__ == "__main__":
    main()
