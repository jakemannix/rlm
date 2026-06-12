"""Shared CLI plumbing for the sleep scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlm.clients.base_lm import BaseLM
from rlm.sleep.config import JudgeConfig, SleepConfig
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
    parser.add_argument(
        "--judge-model",
        default="self",
        help='"self" = the policy model judges itself (higher TTC, no API); '
        "any other value = an OpenAI-compatible model name",
    )
    parser.add_argument("--mock-judge", action="store_true", help="canned judge (offline)")
    parser.add_argument("--episodes-per-day", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help='"auto" = float32 on cpu, bfloat16 on cuda (T4 needs float16 explicitly)',
    )
    parser.add_argument(
        "--use-surprise",
        action="store_true",
        help="add the policy-NLL surprise gate (the selector that matters on "
        "clean expert traces, where failure heuristics rarely fire)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="judge self-consistency samples per episode (default: JudgeConfig.n_samples)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="skip LoRA training + NLL evals (no torch)"
    )


def apply_judge_overrides(config: SleepConfig, args: argparse.Namespace) -> None:
    """Fold optional CLI judge knobs into the config (None = keep the default)."""
    if args.n_samples is not None:
        config.judge.n_samples = args.n_samples


def dump_judge_usage(judge_lm: BaseLM, out_path: Path) -> dict:
    """Persist the judge's token usage next to the run artifacts."""
    summary = judge_lm.get_usage_summary()
    data = {name: usage.to_dict() for name, usage in summary.model_usage_summaries.items()}
    out_path.write_text(json.dumps(data, indent=2))
    return data


def load_episodes(args: argparse.Namespace) -> list[Episode]:
    if args.synthetic:
        return synthetic_traces(args.n_synthetic, seed=0)
    return load_hf_traces(args.dataset, args.config, limit=args.limit)


def build_judge(args: argparse.Namespace, judge_config: JudgeConfig | None = None) -> BaseLM:
    if args.mock_judge:
        from tests.mock_lm import MockLM

        return MockLM(
            response_fn=lambda p: "YES"
            if "verifying a candidate" in str(p)
            else MOCK_LEARN_RESPONSE
        )
    if args.judge_model == "self":
        from rlm.sleep.local_judge import LocalHFJudge

        judge_config = judge_config or JudgeConfig()
        return LocalHFJudge.from_policy(
            args.policy_model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            temperature=judge_config.temperature,
            max_new_tokens=judge_config.max_new_tokens,
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
