"""Step 1 — oracle write/read (design doc §4).

No meta-learning: an *untrained* skill (orthogonal-init encoders, gates at their
one-shot defaults) ingests each fact with fact-span-only writes, the state is
saved to disk and reloaded into a **fresh session** (so A4 is exercised from day
one), and the verbatim query is scored over a sweep of read scales.  Controls:
empty store and wrong-fact store.

Gate: some read scale where mean first-token lift ≥ 3 nats with top-5, controls
flat.  Passing certifies (injection works) × (store binds & persists); Step 2's
meta-training then only needs to *find* what this script hand-builds.

Runtime: ~minutes for 24 facts on an L4.

    python scripts/memory/oracle_write_read.py --model google/gemma-3-1b-it \
        --facts 24 --d-k 512 --out runs/oracle_write_read.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from rlm.memory.episodes import EpisodeGenerator, EpisodeText
from rlm.memory.linear_store import LinearStoreConfig
from rlm.memory.session import MemorySession
from rlm.memory.skill import MemorySkill, SkillConfig

READ_SCALES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def recall_episode(gen: EpisodeGenerator, fact) -> EpisodeText:
    return EpisodeText("recall", [gen.session_with_facts([fact])], fact.verbatim_prompt, fact.answer, None, [fact], fact)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--facts", type=int, default=24)
    ap.add_argument("--d-k", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--write-everything", action="store_true",
                    help="write all ingest tokens instead of fact-span only (measures untrained-gate erosion)")
    ap.add_argument("--out", default="runs/oracle_write_read.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()

    torch.manual_seed(args.seed)
    d_model = model.get_input_embeddings().weight.shape[1]
    skill = MemorySkill(SkillConfig(d_model=d_model, d_k=args.d_k, store=LinearStoreConfig(chunk_size=4)))
    gen = EpisodeGenerator(seed=args.seed)
    facts = [gen.make_fact() for _ in range(args.facts)]

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(facts):
            ingest = MemorySession(model, tok, skill, device=args.device)
            ingest.ingest_episode(recall_episode(gen, f), fact_only=not args.write_everything)
            state_path = str(Path(td) / f"state_{i}.pt")
            ingest.save(state_path)

            recall = MemorySession(model, tok, skill, device=args.device)  # fresh session: context gone
            recall.load(state_path)
            wrong = MemorySession(model, tok, skill, device=args.device)
            wrong.ingest_episode(recall_episode(gen, facts[(i + 1) % len(facts)]), fact_only=not args.write_everything)

            row = {"prompt": f.verbatim_prompt, "answer": f.answer, "scales": {}}
            for s in READ_SCALES:
                row["scales"][s] = {
                    "mem": recall.score_answer(f.verbatim_prompt, f.answer, read_scale=s),
                    "empty": recall.score_answer(f.verbatim_prompt, f.answer, read_scale=s, use_store=False),
                    "wrong": wrong.score_answer(f.verbatim_prompt, f.answer, read_scale=s),
                }
            rows.append(row)
            if (i + 1) % 8 == 0:
                print(f"scored {i + 1}/{len(facts)} facts", flush=True)

    summary = {}
    print(f"\n{'scale':>6} {'mem lift':>9} {'mem top5':>9} {'empty lift':>11} {'wrong lift':>11}")
    for s in READ_SCALES:
        def mean(key: str, sub: str, s=s) -> float:
            return sum(r["scales"][s][key][sub] for r in rows) / len(rows)

        summary[s] = {
            "mem_first_lift": mean("mem", "first_token_lift_nats"),
            "mem_top5": mean("mem", "first_token_top5"),
            "empty_first_lift": mean("empty", "first_token_lift_nats"),
            "wrong_first_lift": mean("wrong", "first_token_lift_nats"),
        }
        v = summary[s]
        print(f"{s:>6} {v['mem_first_lift']:>9.2f} {v['mem_top5']:>9.2f} {v['empty_first_lift']:>11.2f} {v['wrong_first_lift']:>11.2f}")

    passing = [s for s, v in summary.items()
               if v["mem_first_lift"] >= 3.0 and v["mem_top5"] >= 0.9
               and abs(v["empty_first_lift"]) < 1e-6 and abs(v["wrong_first_lift"]) <= 0.5]
    verdict = {"gate": "step1_oracle_write_read", "pass": bool(passing), "passing_scales": passing}
    print(f"\nVERDICT: {'PASS' if passing else 'FAIL'}  (read_scale ∈ {passing})")
    if not passing:
        print("→ if Step 0 passed, debug key consistency (try keys from the literal answer position) "
              "and per-fact interference before touching the trainer.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"verdict": verdict, "summary": summary, "rows": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
