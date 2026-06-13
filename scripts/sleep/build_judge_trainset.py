"""Build a frontier-labeled trainset for fine-tuning a memory-worthiness judge.

Positives: the 71 frontier-curated memories (their tier is the label).
Pool: a sample of the audit's 985 high-recall heuristic candidates, labeled
keep/reject+tier by a frontier model (Opus) via OpenRouter — these supply the
rejects the curated set lacks. Combined, deduped, and split into train/test as
chat SFT rows whose target is the worthiness JSON.

    OPENROUTER_API_KEY=... python scripts/sleep/build_judge_trainset.py \
        --n-heuristic 300 --labeler-model anthropic/claude-opus-4.8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rlm.sleep.gold import _CallCache
from rlm.sleep.judge_worthiness import WORTHINESS_PROMPT, parse_worthiness, worthiness_messages
from rlm.sleep.model_ladder import OPENROUTER_BASE_URL, _NonEmpty, purge_invalid

AUDIT = Path("personal_chat_archive/frontier_quality_audit")


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", default="runs/sleep/frontier_memories/prep.json")
    parser.add_argument(
        "--heuristic",
        default=str(AUDIT / "exhaustive_round1/memory_first_heuristic_candidates.jsonl"),
    )
    parser.add_argument("--n-heuristic", type=int, default=300)
    parser.add_argument("--labeler-model", default="anthropic/claude-opus-4.8")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="runs/sleep/judge_trainset")
    args = parser.parse_args()

    from rlm.clients.openai import OpenAIClient

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Positives: the 71 frontier memories (tier is the label).
    frontier = json.loads(Path(args.prep).read_text())
    seen = {norm(m["memory"]) for m in frontier}
    labeled = [
        {
            "memory": m["memory"],
            "evidence": "\n\n".join(e["context"] for e in m.get("evidence", [])[:2])[:3000],
            "label": {"keep": True, "tier": m["tier"], "reason": "frontier-curated"},
            "source": "frontier",
        }
        for m in frontier
    ]

    # Pool: sample heuristic candidates (deterministic), dedup vs the 71.
    heuristic = [json.loads(line) for line in Path(args.heuristic).open()]
    heuristic.sort(key=lambda c: hashlib.sha256(c["id"].encode()).hexdigest())
    pool, pool_seen = [], set(seen)
    for c in heuristic:
        key = norm(c["candidate_memory"])
        if key in pool_seen:
            continue
        pool_seen.add(key)
        pool.append(c)
        if len(pool) >= args.n_heuristic:
            break

    cache = _CallCache(out / f"label_cache_{args.labeler_model.replace('/', '_')}.json")
    purge_invalid(cache)
    lm = _NonEmpty(OpenAIClient(model_name=args.labeler_model, base_url=OPENROUTER_BASE_URL))

    def label(c: dict) -> dict | None:
        evidence = (
            f"[user] {c.get('human_excerpt', '')}\n[assistant] {c.get('assistant_excerpt', '')}"[
                :3000
            ]
        )
        prompt = WORTHINESS_PROMPT.format(memory=c["candidate_memory"], evidence=evidence)
        verdict = parse_worthiness(cache.completions(lm, "worthiness", prompt, 1)[0])
        if verdict is None:
            return None
        return {
            "memory": c["candidate_memory"],
            "evidence": evidence,
            "label": verdict,
            "source": "heuristic",
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        labeled += [r for r in ex.map(label, pool) if r is not None]

    print(f"labeled: {len(labeled)} ({Counter(r['label']['tier'] for r in labeled)})")

    # Stratified train/test split by tier (stable hash on memory text).
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
                if as_sft:
                    f.write(
                        json.dumps(
                            {
                                "messages": worthiness_messages(
                                    r["memory"], r["evidence"], r["label"]
                                ),
                                "tier": r["label"]["tier"],
                                "source": r["source"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                else:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

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
                "train_tiers": dict(Counter(r["label"]["tier"] for r in train)),
                "test_tiers": dict(Counter(r["label"]["tier"] for r in test)),
            },
            indent=2,
        )
    )
    print(f"train {len(train)} / test {len(test)} -> {out}")


if __name__ == "__main__":
    main()
