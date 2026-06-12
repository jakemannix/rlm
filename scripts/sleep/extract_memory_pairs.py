"""Mine frontier-authored memory pairs from Claude Code transcripts.

Every Write to a session-memory file is an explicit, in-context curation
decision by the session's model: "this is worth remembering," plus the
distilled artifact itself. This script extracts those moments as
(context -> memory) training pairs — gold targets by construction, with
no judge in the loop.

Example:
    python scripts/sleep/extract_memory_pairs.py \
        --source ~/Documents/cc-session-archive --out runs/sleep/memory_pairs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rlm.sleep.cc_traces import load_memory_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="~/Documents/cc-session-archive")
    parser.add_argument("--include-subagents", action="store_true")
    parser.add_argument("--context-steps", type=int, default=30)
    parser.add_argument("--out", default="runs/sleep/memory_pairs")
    args = parser.parse_args()

    pairs = load_memory_pairs(
        args.source, include_subagents=args.include_subagents, context_steps=args.context_steps
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "memory_pairs.jsonl").open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")
    with (out_dir / "sft_memory_pairs.jsonl").open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps({"messages": pair.to_messages()}, ensure_ascii=False) + "\n")

    sessions = {p.session_id for p in pairs}
    ctx = [len(p.context) for p in pairs] or [0]
    mem = [len(p.memory) for p in pairs] or [0]
    print(
        f"{len(pairs)} pairs from {len(sessions)} sessions -> {out_dir}\n"
        f"context steps/pair: min {min(ctx)} / median {sorted(ctx)[len(ctx) // 2]} / max {max(ctx)}\n"
        f"memory chars: min {min(mem)} / median {sorted(mem)[len(mem) // 2]} / max {max(mem)}"
    )


if __name__ == "__main__":
    main()
