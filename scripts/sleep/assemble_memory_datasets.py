"""Assemble the frontier-memory training/eval datasets.

Inputs:
- ``prep.json``      — 71 resolved candidates (from rlm.sleep.frontier_memories)
- ``authored.json``  — frontier-authored behavioral fields + 3 probes/memory

Outputs under --out (gitignored):
- ``sft_distill.jsonl``   — evidence context -> curated memory (compact windows)
- ``sft_apply.jsonl``     — probe scenario -> exemplar response (probes 1-2)
- ``probes_heldout.jsonl``— probe 3 per memory, for post-training behavioral eval
- ``days/<date>.jsonl``   — the union, sharded by evidence day (nightly training)
- ``manifest.json``       — counts and provenance
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rlm.sleep.frontier_memories import DISTILL_INSTRUCTION, load_frontier_memories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", default="personal_chat_archive/frontier_quality_audit")
    parser.add_argument("--authored", default="runs/sleep/frontier_memories/authored.json")
    parser.add_argument("--out", default="runs/sleep/frontier_memories")
    parser.add_argument("--sft-window", type=int, default=2, help="evidence msgs each side for SFT")
    parser.add_argument("--sft-max-message-chars", type=int, default=1200)
    args = parser.parse_args()

    memories = load_frontier_memories(
        args.audit_dir, window=args.sft_window, max_message_chars=args.sft_max_message_chars
    )
    by_id = {m.memory_id: m for m in memories}
    authored = {a["memory_id"]: a for a in json.loads(Path(args.authored).read_text())}
    missing = sorted(set(by_id) - set(authored))
    if missing:
        print(f"WARNING: {len(missing)} candidates lack authored fields: {missing}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    distill, apply_rows, heldout = [], [], []
    day_rows: dict[str, list[dict]] = defaultdict(list)

    for memory in memories:
        meta = {
            "memory_id": memory.memory_id,
            "tier": memory.tier,
            "memory_type": memory.memory_type,
            "scope": memory.scope,
            "day": memory.day,
        }
        evidence = "\n\n".join(
            f"=== {e.conversation_title} ({e.created_at}) ===\n{e.context}" for e in memory.evidence
        )
        row = {
            "messages": [
                {"role": "user", "content": f"{evidence}\n\n{DISTILL_INSTRUCTION}"},
                {"role": "assistant", "content": memory.memory},
            ],
            **meta,
            "kind": "distill",
        }
        distill.append(row)
        day_rows[memory.day].append(row)

        auth = authored.get(memory.memory_id)
        if not auth:
            continue
        for i, probe in enumerate(auth["probes"]):
            probe_row = {
                **meta,
                "memory": memory.memory,
                "desired_behavior": auth["desired_behavior"],
                "negative_behavior_to_avoid": auth["negative_behavior_to_avoid"],
                **probe,
            }
            if i < 2:
                sft = {
                    "messages": [
                        {"role": "user", "content": probe["scenario"]},
                        {"role": "assistant", "content": probe["exemplar_response"]},
                    ],
                    **meta,
                    "kind": "apply",
                }
                apply_rows.append(sft)
                day_rows[memory.day].append(sft)
            else:
                heldout.append(probe_row)

    def dump(name: str, rows: list[dict]) -> None:
        with (out / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump("sft_distill.jsonl", distill)
    dump("sft_apply.jsonl", apply_rows)
    dump("probes_heldout.jsonl", heldout)
    days_dir = out / "days"
    days_dir.mkdir(exist_ok=True)
    for day, rows in sorted(day_rows.items()):
        with (days_dir / f"{day}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "n_memories": len(memories),
        "n_authored": len(authored),
        "n_distill": len(distill),
        "n_apply": len(apply_rows),
        "n_heldout_probes": len(heldout),
        "n_days": len(day_rows),
        "days": {d: len(r) for d, r in sorted(day_rows.items())},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "days"}, indent=2))
    print(f"day shards: {len(day_rows)} -> {days_dir}")


if __name__ == "__main__":
    main()
