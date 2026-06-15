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

    dtype = resolve_torch_dtype(args.torch_dtype, args.device)
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    heldout_probes = load_jsonl(data / "probes_heldout.jsonl")
    negatives = load_jsonl(args.negative_probes) if Path(args.negative_probes).exists() else []

    if args.train_file:
        # Breadth mode: train on an external SFT file (many distinct memories,
        # disjoint from the probe memories), eval transfer on ALL probes_heldout.
        from rlm.sleep.types import TrainingExample

        train_rows = load_jsonl(args.train_file)
        examples = [
            TrainingExample(
                prompt=r["messages"][0]["content"], response=r["messages"][1]["content"],
                lesson=r.get("memory", ""), source_episode_id=r.get("episode_id", "?"), verified=True,
            )
            for r in train_rows
        ]
        probe_sets = {"train": [], "heldout": heldout_probes, "negative": negatives}
        print(f"breadth mode: {len(examples)} exemplars -> eval {len(heldout_probes)} transfer probes",
              flush=True)
    else:
        prep = json.loads((data / "prep.json").read_text())
        split = compute_split(prep, args)
        (out / "split.json").write_text(json.dumps(split, indent=2))
        train_ids, heldout_ids = set(split["train"]), set(split["heldout"])
        print(f"split: {len(train_ids)} train / {len(heldout_ids)} heldout", flush=True)
        probe_sets = {
            "train": [p for p in heldout_probes if p["memory_id"] in train_ids],
            "heldout": [p for p in heldout_probes if p["memory_id"] in heldout_ids],
            "negative": negatives,
        }
        examples = build_train_examples(data, train_ids)
        print(f"training examples (TRAIN memories only): {len(examples)}", flush=True)
    for name, ps in probe_sets.items():
        print(f"  probe set {name}: {len(ps)}", flush=True)

    seeds = [int(s) for s in args.train_seeds.split(",")]
    adapters: dict[int, Path] = {}
    for seed in seeds:
        adir = out / f"adapter_seed{seed}"
        if args.reuse_adapters and (adir / "training_meta.json").exists():
            print(f"  adapter seed {seed}: reusing existing", flush=True)
        else:
            cfg = AdapterConfig(
                lr=args.lr, epochs=args.epochs, max_seq_len=args.max_seq_len,
                batch_size=1, grad_accum=4, seed=seed,
            )
            if args.lora_target == "all-linear":
                cfg.target_modules = ("all-linear",)
            train_lora(examples, args.policy_model, adir, cfg, device=args.device, torch_dtype=dtype)
            meta = json.loads((adir / "training_meta.json").read_text())
            print(f"  adapter seed {seed}: loss {meta['mean_loss']:.3f}->{meta['final_loss']:.3f}",
                  flush=True)
        adapters[seed] = adir

    import torch

    def batch_generate(model, scenarios: list[str]) -> list[str]:
        """Left-padded batched greedy decode — ~Bx faster than one at a time."""
        tok.padding_side = "left"
        outputs: list[str] = []
        for i in range(0, len(scenarios), args.gen_batch_size):
            chunk = scenarios[i : i + args.gen_batch_size]
            texts = [
                tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False,
                                        add_generation_prompt=True)
                for s in chunk
            ]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                outputs.append(tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outputs

    rows = []
    for set_name, ps in probe_sets.items():
        for p in ps:
            rows.append({
                "probe_id": p.get("probe_id", p.get("memory_id")), "set": set_name,
                "memory_id": p.get("memory_id"), "tier": p.get("tier"),
                "memory_type": p.get("memory_type"), "scenario": p["scenario"],
                "behavior_marker": p.get("behavior_marker"),
                "should_not_exhibit": p.get("should_not_exhibit"),
                "principle": p.get("memory"), "adapted": {},
            })
    scenarios = [r["scenario"] for r in rows]

    base_model, tok = load_policy(args.policy_model, device=args.device, torch_dtype=dtype)
    base_model.eval()
    for row, resp in zip(rows, batch_generate(base_model, scenarios), strict=True):
        row["base"] = resp
    del base_model
    print("  generated base responses", flush=True)
    for seed, adir in adapters.items():
        amodel, _ = load_adapter(args.policy_model, adir, device=args.device, torch_dtype=dtype)
        amodel.eval()
        for row, resp in zip(rows, batch_generate(amodel, scenarios), strict=True):
            row["adapted"][str(seed)] = resp
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
        OpenAIClient(
            model_name=args.judge_model, base_url=OPENROUTER_BASE_URL,
            max_tokens=args.judge_max_tokens,
        ),
        cache, "grade",
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

    def grade_row(row: dict) -> dict:
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
        return rec

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.grade_workers) as pool:
        report_rows = list(pool.map(grade_row, rows))

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
    parser.add_argument("--train-file", default=None,
                        help="breadth mode: SFT jsonl (messages) to train on; eval = all heldout probes")
    parser.add_argument("--lora-target", default=None, choices=[None, "all-linear"],
                        help="'all-linear' targets every linear layer (cross-arch sweep)")
    parser.add_argument("--policy-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--gen-batch-size", type=int, default=16)
    parser.add_argument("--reuse-adapters", action="store_true",
                        help="reuse already-trained adapter dirs (skip retraining)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--judge-model", default="qwen/qwen3.6-35b-a3b")
    parser.add_argument("--grade-workers", type=int, default=8)
    parser.add_argument("--judge-max-tokens", type=int, default=4096,
                        help="cap judge completion length; unset 402s on OpenRouter low balance")
    parser.add_argument("--out", default="runs/sleep/uptake")
    args = parser.parse_args()

    if args.phase == "generate":
        generate_phase(args)
    else:
        grade_phase(args)


if __name__ == "__main__":
    main()
