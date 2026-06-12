"""Nightly LoRA training over per-day memory shards + behavioral probe eval.

Trains one adapter per evidence-day shard (chronological;
``--cumulative`` re-distills from the union so far), then measures
whether trained memories actually *fire* on held-out probe scenarios:

- behavior uptake: a judge checks the response against the probe's
  ``behavior_marker`` ("does this response ask before sampling?")
- CoT surfacing: do the probe's ``cot_marker`` keywords appear in the
  response's visible reasoning?

Both rates are reported for base vs adapted — the delta is the
memory-uptake result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rlm.sleep.config import AdapterConfig
from rlm.sleep.judge import VERDICT_RE
from rlm.sleep.types import TrainingExample

BEHAVIOR_GRADE_PROMPT = """\
You are grading one assistant response against one behavioral criterion.

SCENARIO (user message):
{scenario}

RESPONSE:
{response}

CRITERION: does the response {behavior_marker}?

Answer in one short sentence, then give your verdict on the last line in \
exactly this form:
VERDICT: YES
or
VERDICT: NO
"""


def rows_to_examples(rows: list[dict]) -> list[TrainingExample]:
    return [
        TrainingExample(
            prompt=row["messages"][0]["content"],
            response=row["messages"][1]["content"],
            lesson=row.get("memory", ""),
            source_episode_id=row.get("memory_id", "?"),
            verified=True,
        )
        for row in rows
    ]


def grade_behavior(judge_lm, scenario: str, response: str, behavior_marker: str) -> bool:
    text = judge_lm.completion(
        BEHAVIOR_GRADE_PROMPT.format(
            scenario=scenario[:2000], response=response[:3000], behavior_marker=behavior_marker
        )
    )
    lines = [line for line in text.splitlines() if line.strip()]
    match = VERDICT_RE.search(" ".join(lines[-3:])) if lines else None
    return bool(match) and match.group(1).upper() == "YES"


def cot_mentions(response: str, cot_marker: str) -> bool:
    words = [w for w in re.split(r"[,;/\s]+", cot_marker) if len(w) > 3]
    return any(w.lower() in response.lower() for w in words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="runs/sleep/frontier_memories")
    parser.add_argument("--policy-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cumulative", action="store_true")
    parser.add_argument("--max-nights", type=int, default=None)
    parser.add_argument("--probe-limit", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--skip-training", action="store_true", help="evaluate an existing final adapter only"
    )
    parser.add_argument("--out", default="runs/sleep/memory_nights")
    args = parser.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    day_files = sorted((data / "days").glob("*.jsonl"))
    if args.max_nights:
        day_files = day_files[: args.max_nights]

    adapter_dir = out / "adapter_final"
    if not args.skip_training:
        from rlm.sleep.lora import train_lora

        config = AdapterConfig(lr=args.lr, epochs=args.epochs, max_seq_len=args.max_seq_len)
        accumulated: list[TrainingExample] = []
        for i, day_file in enumerate(day_files):
            rows = [json.loads(line) for line in day_file.open()]
            examples = rows_to_examples(rows)
            if args.cumulative:
                accumulated.extend(examples)
                examples = list(accumulated)
            night_dir = out / f"night_{i:03d}_{day_file.stem}"
            train_lora(
                examples,
                args.policy_model,
                night_dir / "adapter",
                config,
                device=args.device,
                torch_dtype=args.torch_dtype,
            )
            meta = json.loads((night_dir / "adapter" / "training_meta.json").read_text())
            print(
                f"[night {i + 1}/{len(day_files)}] {day_file.stem}: "
                f"{len(examples)} examples, loss {meta['mean_loss']:.3f} -> "
                f"{meta['final_loss']:.3f}",
                flush=True,
            )
            adapter_dir = night_dir / "adapter"

    # Behavioral probes: base vs final adapter on held-out scenario #3s.
    from rlm.sleep.local_judge import LocalHFJudge
    from rlm.sleep.lora import load_adapter, load_policy
    from rlm.sleep.winrate import generate_response

    probes = [json.loads(line) for line in (data / "probes_heldout.jsonl").open()]
    if args.probe_limit:
        probes = probes[: args.probe_limit]
    trained_days = {f.stem for f in day_files}
    in_scope = [p for p in probes if p["day"] in trained_days]
    print(
        f"probing {len(in_scope)}/{len(probes)} held-out scenarios "
        f"(memories within the trained nights)",
        flush=True,
    )

    base_model, tokenizer = load_policy(
        args.policy_model, device=args.device, torch_dtype=args.torch_dtype
    )
    base_model.eval()
    adapted_model, _ = load_adapter(
        args.policy_model, adapter_dir, device=args.device, torch_dtype=args.torch_dtype
    )
    judge_lm = LocalHFJudge(
        base_model, tokenizer, model_name=args.policy_model, device=args.device, temperature=0.0
    )

    results = []
    for i, probe in enumerate(in_scope):
        messages = [{"role": "user", "content": probe["scenario"]}]
        row = {
            "memory_id": probe["memory_id"],
            "tier": probe["tier"],
            "memory_type": probe["memory_type"],
        }
        for name, model in (("base", base_model), ("adapted", adapted_model)):
            response = generate_response(
                model, tokenizer, messages, args.device, max_new_tokens=512
            )
            row[name] = {
                "behavior": grade_behavior(
                    judge_lm, probe["scenario"], response, probe["behavior_marker"]
                ),
                "cot": cot_mentions(response, probe["cot_marker"]),
                "response": response,
            }
        results.append(row)
        print(
            f"[probe {i + 1}/{len(in_scope)}] {probe['memory_id']} "
            f"base: beh={row['base']['behavior']} cot={row['base']['cot']} | "
            f"adapted: beh={row['adapted']['behavior']} cot={row['adapted']['cot']}",
            flush=True,
        )

    def pct(key: str, sub: str) -> str:
        hits = sum(1 for r in results if r[key][sub])
        return f"{hits}/{len(results)} ({100 * hits / len(results):.0f}%)" if results else "n/a"

    summary = {
        "n_probes": len(results),
        "behavior_uptake": {"base": pct("base", "behavior"), "adapted": pct("adapted", "behavior")},
        "cot_surfacing": {"base": pct("base", "cot"), "adapted": pct("adapted", "cot")},
        "cumulative": args.cumulative,
        "nights_trained": len(day_files),
    }
    (out / "probe_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results)
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
