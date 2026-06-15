"""Fine-tune an ~8B memory-worthiness judge and compare it to 35b-a3b.

--phase train    (GPU/Colab): LoRA-train the base on the frontier-labeled
    trainset, then run the fine-tuned judge on the held-out testset and
    write its predictions.
--phase compare  (local): run the zero-shot 35b-a3b judge on the same
    testset via OpenRouter and report balanced accuracy / keep-F1 for both
    against the frontier labels — does a small fine-tuned judge match the
    35B zero-shot one?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlm.sleep.judge_worthiness import WORTHINESS_PROMPT, parse_worthiness


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def metrics(rows: list[dict], pred_key: str) -> dict:
    """Balanced accuracy + keep-F1 of pred vs the frontier label."""
    tp = fp = tn = fn = 0
    parsed = 0
    for r in rows:
        gold_keep = r["label"]["keep"]
        pred = r.get(pred_key)
        if pred is None:
            continue
        parsed += 1
        pred_keep = pred["keep"]
        tp += gold_keep and pred_keep
        fp += (not gold_keep) and pred_keep
        tn += (not gold_keep) and (not pred_keep)
        fn += gold_keep and (not pred_keep)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
    return {
        "n_parsed": parsed,
        "balanced_accuracy": round((tpr + tnr) / 2, 3),
        "keep_f1": round(f1, 3),
        "keep_recall": round(tpr, 3),
        "reject_recall": round(tnr, 3),
    }


def train_phase(args: argparse.Namespace) -> None:
    from rlm.sleep.config import AdapterConfig
    from rlm.sleep.local_judge import resolve_torch_dtype
    from rlm.sleep.lora import load_adapter, train_lora
    from rlm.sleep.types import TrainingExample
    from rlm.sleep.winrate import generate_response

    dtype = resolve_torch_dtype(args.torch_dtype, args.device)
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(data / "trainset.jsonl")
    examples = [
        TrainingExample(
            prompt=r["messages"][0]["content"],
            response=r["messages"][1]["content"],
            lesson="",
            source_episode_id=r.get("source", "?"),
            verified=True,
        )
        for r in train_rows
    ]
    cfg = AdapterConfig(
        lr=args.lr,
        epochs=args.epochs,
        max_seq_len=args.max_seq_len,
        batch_size=1,
        grad_accum=8,
        seed=0,
    )
    adapter = out / "judge_adapter"
    print(f"training {args.base_model} judge on {len(examples)} examples", flush=True)
    train_lora(examples, args.base_model, adapter, cfg, device=args.device, torch_dtype=dtype)

    model, tok = load_adapter(args.base_model, adapter, device=args.device, torch_dtype=dtype)
    test = load_jsonl(data / "testset_raw.jsonl")
    preds = []
    for r in test:
        prompt = WORTHINESS_PROMPT.format(memory=r["memory"], evidence=r["evidence"])
        text = generate_response(
            model, tok, [{"role": "user", "content": prompt}], args.device, max_new_tokens=128
        )
        preds.append({**r, "ft_pred": parse_worthiness(text)})
    (out / "ft_predictions.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in preds)
    )
    print("ft judge:", json.dumps(metrics(preds, "ft_pred")), flush=True)


def compare_phase(args: argparse.Namespace) -> None:
    from rlm.clients.openai import OpenAIClient
    from rlm.sleep.gold import _CallCache
    from rlm.sleep.model_ladder import OPENROUTER_BASE_URL, CachedJudge, purge_invalid

    out = Path(args.out)
    preds = load_jsonl(out / "ft_predictions.jsonl")
    cache = _CallCache(out / "compare_cache.json")
    purge_invalid(cache)
    judge = CachedJudge(
        OpenAIClient(model_name=args.baseline_model, base_url=OPENROUTER_BASE_URL),
        cache,
        "worthiness",
    )
    for r in preds:
        prompt = WORTHINESS_PROMPT.format(memory=r["memory"], evidence=r["evidence"])
        try:
            r["baseline_pred"] = parse_worthiness(judge.completion(prompt))
        except Exception:  # noqa: BLE001
            r["baseline_pred"] = None

    report = {
        "base_model": args.base_model,
        "n_test": len(preds),
        f"finetuned_{args.base_model.split('/')[-1]}": metrics(preds, "ft_pred"),
        f"zeroshot_{args.baseline_model.split('/')[-1]}": metrics(preds, "baseline_pred"),
    }
    (out / "judge_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["train", "compare"], required=True)
    parser.add_argument("--data", default="runs/sleep/judge_trainset")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--baseline-model", default="qwen/qwen3.6-35b-a3b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", default="runs/sleep/judge_ft")
    args = parser.parse_args()

    if args.phase == "train":
        train_phase(args)
    else:
        compare_phase(args)


if __name__ == "__main__":
    main()
