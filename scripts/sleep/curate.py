"""Two-stage memory curator: cheap local-tier judge + frontier polish.

Turns raw episodes into a curated memory set using the ladder's finding
that judging is cheap (qwen3.6-35b-a3b ≈ frontier) but generation is hard:

  1. propose  — a candidate memory from each episode's evidence (cheap model)
  2. judge     — keep/reject+tier with the cheap stage-1 judge (35b-a3b)
  3. polish    — for kept candidates only, a frontier model rewrites the
                 memory to gold quality (where the spend actually buys quality)

All calls cached; reruns/extensions pay only for new episodes.

    OPENROUTER_API_KEY=... python scripts/sleep/curate.py \
        --source cc --archive ~/Documents/cc-session-archive \
        --judge-model qwen/qwen3.6-35b-a3b --polish-model anthropic/claude-opus-4.8
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rlm.sleep.gold import _CallCache
from rlm.sleep.judge_worthiness import WORTHINESS_PROMPT, parse_worthiness
from rlm.sleep.model_ladder import GENERATE_PROMPT, OPENROUTER_BASE_URL, CachedJudge

POLISH_PROMPT = """\
The following candidate memory was distilled from a user's conversation and \
judged worth keeping. Rewrite it as a single, durable, concrete memory note \
(1-3 sentences) a future agent should retain — preserve the substance, sharpen \
the wording, drop anything session-specific that won't generalize. Respond with \
ONLY the rewritten memory.

CANDIDATE:
{memory}

EVIDENCE:
{evidence}
"""


def load_episodes(args):
    if args.source == "cc":
        from rlm.sleep.cc_traces import load_cc_episodes

        eps = load_cc_episodes(args.archive)
    else:
        from rlm.sleep.claude_export import load_claude_export

        eps = load_claude_export(args.archive)
    if args.limit:
        eps = eps[: args.limit]
    return eps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["cc", "claude-export"], default="cc")
    parser.add_argument("--archive", default="~/Documents/cc-session-archive")
    parser.add_argument("--judge-model", default="qwen/qwen3.6-35b-a3b")
    parser.add_argument("--propose-model", default="qwen/qwen3.6-35b-a3b")
    parser.add_argument("--polish-model", default="anthropic/claude-opus-4.8")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--evidence-chars", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="runs/sleep/curated")
    args = parser.parse_args()

    from rlm.clients.openai import OpenAIClient

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = _CallCache(out / "curate_cache.json")

    def client(model: str, kind: str) -> CachedJudge:
        return CachedJudge(
            OpenAIClient(model_name=model, base_url=OPENROUTER_BASE_URL), cache, kind
        )

    proposer = client(args.propose_model, "propose")
    judge = client(args.judge_model, "judge")
    polisher = client(args.polish_model, "polish")

    episodes = load_episodes(args)
    print(f"curating {len(episodes)} episodes", flush=True)

    def curate(ep) -> dict | None:
        evidence = ep.transcript(max_chars=args.evidence_chars)
        candidate = _safe(proposer, GENERATE_PROMPT.format(evidence=evidence))
        if candidate is None:
            return None
        candidate = candidate.strip()
        raw = _safe(judge, WORTHINESS_PROMPT.format(memory=candidate, evidence=evidence))
        verdict = parse_worthiness(raw) if raw is not None else None
        if verdict is None or not verdict["keep"]:
            return {
                "episode_id": ep.episode_id,
                "candidate": candidate,
                "kept": False,
                "tier": verdict["tier"] if verdict else "unparseable",
            }
        polished = _safe(polisher, POLISH_PROMPT.format(memory=candidate, evidence=evidence))
        return {
            "episode_id": ep.episode_id,
            "source": ep.source,
            "candidate": candidate,
            "kept": True,
            "tier": verdict["tier"],
            "memory": (polished or candidate).strip(),
            "reason": verdict.get("reason", ""),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = [r for r in ex.map(curate, episodes) if r is not None]

    kept = [r for r in results if r.get("kept")]
    (out / "curated.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results)
    )
    with (out / "gold.jsonl").open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Recall the durable memory for this situation.",
                            },
                            {"role": "assistant", "content": r["memory"]},
                        ],
                        "tier": r["tier"],
                        "episode_id": r["episode_id"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    from collections import Counter

    summary = {
        "n_episodes": len(episodes),
        "n_kept": len(kept),
        "keep_rate": round(len(kept) / len(results), 3) if results else 0,
        "tiers": dict(Counter(r.get("tier") for r in results)),
        "judge_model": args.judge_model,
        "polish_model": args.polish_model,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _safe(judge: CachedJudge, prompt: str):
    try:
        return judge.completion(prompt)
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    main()
