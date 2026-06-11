"""Shared CLI plumbing for the sleep scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlm.clients.base_lm import BaseLM
from rlm.sleep.traces import load_hf_traces, synthetic_traces
from rlm.sleep.types import Episode, EvalReport, TrainingExample

MOCK_LEARN_RESPONSE = json.dumps(
    {
        "verdict": "learn",
        "confidence": 0.9,
        "lesson": "Validate tool arguments before calling.",
        "examples": [
            {
                "prompt": "Call a tool that takes a numeric id.",
                "response": "Validate the id is numeric first, then call the tool once.",
            }
        ],
    }
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--synthetic", action="store_true", help="use synthetic traces (offline)")
    parser.add_argument("--n-synthetic", type=int, default=400)
    parser.add_argument("--dataset", default="THUDM/AgentInstruct")
    parser.add_argument("--config", default="os", help="HF dataset config (AgentInstruct domain)")
    parser.add_argument("--limit", type=int, default=None, help="cap loaded episodes")
    parser.add_argument("--policy-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--mock-judge", action="store_true", help="canned judge (offline)")
    parser.add_argument("--episodes-per-day", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run", action="store_true", help="skip LoRA training + NLL evals (no torch)"
    )


def load_episodes(args: argparse.Namespace) -> list[Episode]:
    if args.synthetic:
        return synthetic_traces(args.n_synthetic, seed=0)
    return load_hf_traces(args.dataset, args.config, limit=args.limit)


def build_judge(args: argparse.Namespace) -> BaseLM:
    if args.mock_judge:
        from tests.mock_lm import MockLM

        return MockLM(
            response_fn=lambda p: "YES"
            if "verifying a candidate" in str(p)
            else MOCK_LEARN_RESPONSE
        )
    from rlm.clients.openai import OpenAIClient

    return OpenAIClient(model_name=args.judge_model)


def fake_train_fn(examples: list[TrainingExample], config, out_dir: Path) -> Path:
    """Dry-run trainer: records the example count, trains nothing."""
    adapter = out_dir / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "training_meta.json").write_text(json.dumps({"n_examples": len(examples)}))
    return adapter


def fake_eval_fn(config, adapter_dir, next_day, test) -> EvalReport:
    """Dry-run evaluator: zero metrics, correct shapes."""
    return EvalReport(
        name="adapted" if adapter_dir else "base",
        next_day_nll=0.0,
        test_nll=0.0,
        retention_nll=0.0,
        n_next_day=len(next_day),
        n_test=len(test),
    )
