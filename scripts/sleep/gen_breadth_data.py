"""Generate breadth training data: one apply-exemplar per archive episode.

The Phase-B/D negative came partly from too few *distinct* memories (49) with
~3 examples each. Disposition-learning wants the opposite: many distinct
memories, ~1 example each. This mines the full claude.ai export (or cc archive)
and, with one frontier call per episode, emits a (durable memory, novel
scenario, ideal response) triple — the apply-exemplar is the SFT row. Eval
stays the held-out 71 frontier-memory probes (independent transfer test).

    OPENROUTER_API_KEY=... python scripts/sleep/gen_breadth_data.py \
        --source claude-export --n-episodes 800 --gen-model anthropic/claude-opus-4.8
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rlm.sleep.gold import _CallCache
from rlm.sleep.judge import extract_json
from rlm.sleep.model_ladder import OPENROUTER_BASE_URL, CachedJudge

GEN_PROMPT = """\
Below is one conversation between a user and an AI assistant.

{evidence}

From this, produce ONE durable, reusable *behavioral memory* a future agent
working with this user should hold — a disposition, correction, preference, or
constraint that generalizes beyond this conversation (not a summary of the
topic). Then write ONE realistic, DIFFERENT future scenario where that memory
should change the agent's behavior, and the ideal assistant response that
visibly applies it.

Respond with ONLY a JSON object:
{{"memory": "<one durable behavioral memory, 1-2 sentences>",
  "scenario": "<a realistic NEW user message where the memory should fire>",
  "response": "<the ideal assistant response applying the memory, <=1200 chars>"}}
"""


def load_episodes(args):
    if args.source == "cc":
        from rlm.sleep.cc_traces import load_cc_episodes

        eps = load_cc_episodes(args.archive)
    else:
        from rlm.sleep.claude_export import load_claude_export

        eps = load_claude_export(args.archive)
    eps.sort(key=lambda e: hashlib.sha256(e.episode_id.encode()).hexdigest())
    return eps[: args.n_episodes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["claude-export", "cc"], default="claude-export")
    parser.add_argument("--archive", default="personal_chat_archive")
    parser.add_argument("--n-episodes", type=int, default=800)
    parser.add_argument("--gen-model", default="anthropic/claude-opus-4.8")
    parser.add_argument("--evidence-chars", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", default="runs/sleep/breadth")
    args = parser.parse_args()

    from rlm.clients.openai import OpenAIClient

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = _CallCache(out / "gen_cache.json")
    gen = CachedJudge(
        OpenAIClient(model_name=args.gen_model, base_url=OPENROUTER_BASE_URL), cache, "gen"
    )

    episodes = load_episodes(args)
    print(f"generating breadth exemplars from {len(episodes)} episodes", flush=True)

    def make(ep) -> dict | None:
        evidence = ep.transcript(max_chars=args.evidence_chars)
        try:
            parsed = extract_json(gen.completion(GEN_PROMPT.format(evidence=evidence)))
            mem, scen, resp = parsed["memory"], parsed["scenario"], parsed["response"]
        except Exception:  # noqa: BLE001
            return None
        if not (mem and scen and resp):
            return None
        return {
            "episode_id": ep.episode_id,
            "memory": str(mem),
            "messages": [
                {"role": "user", "content": str(scen)},
                {"role": "assistant", "content": str(resp)},
            ],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(make, episodes) if r is not None]

    (out / "sft_breadth.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    )
    (out / "summary.json").write_text(
        json.dumps(
            {"n_episodes": len(episodes), "n_exemplars": len(rows), "gen_model": args.gen_model},
            indent=2,
        )
    )
    print(f"wrote {len(rows)} breadth exemplars -> {out / 'sft_breadth.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
