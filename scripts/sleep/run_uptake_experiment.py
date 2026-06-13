"""Causal memory-uptake experiment with held-out-memory + negative controls.

Two phases (decoupled so generation runs on GPU and grading runs locally
against OpenRouter):

  --phase generate  (GPU/Colab): split the 71 memories into train/heldout,
      train adapters on TRAIN-memory data only (one per seed), and generate
      base + adapted responses on three probe sets — TRAIN-memory held-out
      probes, HELDOUT-memory probes (never trained), and negative-control
      probes (no memory should fire). Writes responses.jsonl.

  --phase grade  (local): grade those cached responses with an OpenRouter
      judge — behavior + CoT on the positive sets, over-application on the
      negative set — plus a blind base-vs-adapted win-rate. Writes report.json.

Headline contrast: (adapted gain on TRAIN probes) − (adapted gain on
HELDOUT probes) = specific learning, separated from generic caution; the
negative-control delta is the over-application cost.

Temporal mode (Phase D): --split temporal --cutoff YYYY-MM-DD trains on
memories dated <= cutoff and probes memories dated after it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rlm.sleep.frontier_memories import split_memories


def load_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).open(encoding="utf-8", errors="replace")
        if line.strip()
    ]


def compute_split(prep: list[dict], args: argparse.Namespace) -> dict[str, list[str]]:
    if args.split == "temporal":
        by_day = {
            m["memory_id"]: min(
                (e["created_at"] for e in m.get("evidence", []) if e.get("created_at")),
                default="9999",
            )
            for m in prep
        }
        train = sorted(mid for mid, d in by_day.items() if d <= args.cutoff)
        heldout = sorted(mid for mid, d in by_day.items() if d > args.cutoff)
        return {"train": train, "heldout": heldout}
    return split_memories(prep, heldout_fraction=args.heldout_fraction, seed=args.split_seed)


def build_train_examples(data: Path, train_ids: set[str]):
    from rlm.sleep.types import TrainingExample

    rows = []
    for shard in sorted((data / "days").glob("*.jsonl")):
        if shard.name.startswith("._"):  # skip macOS AppleDouble sidecars
            continue
        rows += [r for r in load_jsonl(shard) if r.get("memory_id") in train_ids]
    return [
        TrainingExample(
            prompt=r["messages"][0]["content"],
            response=r["messages"][1]["content"],
            lesson=r.get("memory", ""),
            source_episode_id=r.get("memory_id", "?"),
            verified=True,
        )
        for r in rows
    ]


def generate_phase(args: argparse.Namespace) -> None:
    from rlm.sleep.config import AdapterConfig
    from rlm.sleep.local_judge import resolve_torch_dtype
    from rlm.sleep.lora import load_adapter, load_policy, train_lora
    from rlm.sleep.winrate import generate_response

    dtype = resolve_torch_dtype(args.torch_dtype, args.device)
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prep = json.loads((data / "prep.json").read_text())
    split = compute_split(prep, args)
    (out / "split.json").write_text(json.dumps(split, indent=2))
    train_ids, heldout_ids = set(split["train"]), set(split["heldout"])
    print(f"split: {len(train_ids)} train / {len(heldout_ids)} heldout", flush=True)

    heldout_probes = load_jsonl(data / "probes_heldout.jsonl")
    probe_sets = {
        "train": [p for p in heldout_probes if p["memory_id"] in train_ids],
        "heldout": [p for p in heldout_probes if p["memory_id"] in heldout_ids],
        "negative": load_jsonl(args.negative_probes) if Path(args.negative_probes).exists() else [],
    }
    for name, ps in probe_sets.items():
        print(f"  probe set {name}: {len(ps)}", flush=True)

    examples = build_train_examples(data, train_ids)
    print(f"training examples (TRAIN memories only): {len(examples)}", flush=True)

    seeds = [int(s) for s in args.train_seeds.split(",")]
    adapters: dict[int, Path] = {}
    for seed in seeds:
        cfg = AdapterConfig(
            lr=args.lr,
            epochs=args.epochs,
            max_seq_len=args.max_seq_len,
            batch_size=1,
            grad_accum=4,
            seed=seed,
        )
        adir = out / f"adapter_seed{seed}"
        train_lora(examples, args.policy_model, adir, cfg, device=args.device, torch_dtype=dtype)
        adapters[seed] = adir
        meta = json.loads((adir / "training_meta.json").read_text())
        print(
            f"  adapter seed {seed}: loss {meta['mean_loss']:.3f}->{meta['final_loss']:.3f}",
            flush=True,
        )

    base_model, tok = load_policy(args.policy_model, device=args.device, torch_dtype=dtype)
    base_model.eval()

    def gen(model, scenario: str) -> str:
        return generate_response(
            model, tok, [{"role": "user", "content": scenario}], args.device, max_new_tokens=512
        )

    rows = []
    for set_name, ps in probe_sets.items():
        for p in ps:
            scenario = p["scenario"]
            rows.append(
                {
                    "probe_id": p.get("probe_id", p.get("memory_id")),
                    "set": set_name,
                    "memory_id": p.get("memory_id"),
                    "tier": p.get("tier"),
                    "memory_type": p.get("memory_type"),
                    "scenario": scenario,
                    "behavior_marker": p.get("behavior_marker"),
                    "should_not_exhibit": p.get("should_not_exhibit"),
                    "principle": p.get("memory"),
                    "base": gen(base_model, scenario),
                    "adapted": {},
                }
            )
    del base_model
    for seed, adir in adapters.items():
        amodel, _ = load_adapter(args.policy_model, adir, device=args.device, torch_dtype=dtype)
        for row in rows:
            row["adapted"][str(seed)] = gen(amodel, row["scenario"])
        del amodel
        print(f"  generated responses for seed {seed}", flush=True)

    (out / "responses.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(f"wrote {len(rows)} probe responses -> {out / 'responses.jsonl'}", flush=True)


def grade_phase(args: argparse.Namespace) -> None:
    from rlm.clients.openai import OpenAIClient
    from rlm.sleep.gold import _CallCache
    from rlm.sleep.grading import grade_behavior, grade_cot, grade_over_application
    from rlm.sleep.model_ladder import OPENROUTER_BASE_URL, CachedJudge, purge_invalid
    from rlm.sleep.winrate import judged_win_rate

    out = Path(args.out)
    rows = load_jsonl(out / "responses.jsonl")
    cache = _CallCache(out / f"grade_cache_{args.judge_model.replace('/', '_')}.json")
    purge_invalid(cache)
    judge = CachedJudge(
        OpenAIClient(model_name=args.judge_model, base_url=OPENROUTER_BASE_URL), cache, "grade"
    )
    seeds = sorted({s for r in rows for s in r["adapted"]})

    def behavior_hit(row: dict, response: str) -> bool:
        if row["set"] == "negative":
            # over-application is the FAILURE: invert so "hit" = good (no over-apply)
            return not grade_over_application(
                judge,
                row["scenario"],
                response,
                row.get("should_not_exhibit") or "refuse or hedge unnecessarily",
            )
        return grade_behavior(judge, row["scenario"], response, row.get("behavior_marker") or "")

    report_rows = []
    for row in rows:
        rec = {
            "probe_id": row["probe_id"],
            "set": row["set"],
            "memory_type": row.get("memory_type"),
        }
        rec["base_behavior"] = behavior_hit(row, row["base"])
        rec["adapted_behavior"] = [behavior_hit(row, row["adapted"][s]) for s in seeds]
        if row["set"] != "negative":
            rec["base_cot"] = grade_cot(judge, row["base"], row.get("principle") or "")
            rec["adapted_cot"] = [
                grade_cot(judge, row["adapted"][s], row.get("principle") or "") for s in seeds
            ]
        report_rows.append(rec)

    def rate(set_name: str, arm: str, key: str) -> tuple[float, float] | None:
        sub = [r for r in report_rows if r["set"] == set_name and f"{arm}_{key}" in r]
        if not sub:
            return None
        if arm == "base":
            vals = [float(r[f"base_{key}"]) for r in sub]
            return (statistics.mean(vals), 0.0)
        per_seed = [
            statistics.mean(float(r[f"adapted_{key}"][i]) for r in sub) for i in range(len(seeds))
        ]
        return (
            statistics.mean(per_seed),
            statistics.pstdev(per_seed) if len(per_seed) > 1 else 0.0,
        )

    summary = {"judge_model": args.judge_model, "n_seeds": len(seeds), "sets": {}}
    for set_name in ("train", "heldout", "negative"):
        b = rate(set_name, "base", "behavior")
        a = rate(set_name, "adapted", "behavior")
        if b is None:
            continue
        entry = {
            "n": sum(1 for r in report_rows if r["set"] == set_name),
            "base_behavior": round(b[0], 3),
            "adapted_behavior": round(a[0], 3),
            "adapted_behavior_std": round(a[1], 3),
            "delta": round(a[0] - b[0], 3),
        }
        if set_name != "negative":
            bc, ac = rate(set_name, "base", "cot"), rate(set_name, "adapted", "cot")
            entry |= {"base_cot": round(bc[0], 3), "adapted_cot": round(ac[0], 3)}
        summary["sets"][set_name] = entry

    if "train" in summary["sets"] and "heldout" in summary["sets"]:
        summary["specific_learning"] = round(
            summary["sets"]["train"]["delta"] - summary["sets"]["heldout"]["delta"], 3
        )

    # blind win-rate on positive probes (base vs adapted seed0)
    s0 = seeds[0]
    items = [
        (r["scenario"], r["base"], r["adapted"][s0])
        for r in rows
        if r["set"] in ("train", "heldout")
    ]
    wr = judged_win_rate(judge, items, seed=0)
    summary["win_rate_adapted_vs_base"] = wr.to_dict()

    (out / "report.json").write_text(json.dumps(summary, indent=2))
    (out / "graded_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in report_rows)
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["generate", "grade"], required=True)
    parser.add_argument("--data", default="runs/sleep/frontier_memories")
    parser.add_argument(
        "--negative-probes", default="runs/sleep/frontier_memories/negative_probes.jsonl"
    )
    parser.add_argument("--split", choices=["random", "temporal"], default="random")
    parser.add_argument("--heldout-fraction", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--cutoff", default="2026-01-01", help="temporal split date")
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--policy-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--judge-model", default="qwen/qwen3.6-35b-a3b")
    parser.add_argument("--out", default="runs/sleep/uptake")
    args = parser.parse_args()

    if args.phase == "generate":
        generate_phase(args)
    else:
        grade_phase(args)


if __name__ == "__main__":
    main()
