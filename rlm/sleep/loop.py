"""The nightly consolidation cycle: gate -> judge -> dataset -> LoRA -> eval.

``run_night`` is dependency-injected: the trainer and evaluator default to
the real LoRA/eval implementations (which require the ``sleep`` extra),
and tests inject lightweight fakes — one code path either way.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from rlm.clients.base_lm import BaseLM
from rlm.sleep.config import SleepConfig
from rlm.sleep.dataset import assert_no_leakage, collect_examples, with_replay, write_jsonl
from rlm.sleep.gate import select_for_reflection
from rlm.sleep.judge import ReflectionJudge
from rlm.sleep.types import DayResult, Episode, EvalReport, TrainingExample

TrainFn = Callable[[list[TrainingExample], SleepConfig, Path], Path]
EvalFn = Callable[[SleepConfig, Path | None, list[Episode], list[Episode]], EvalReport]


def default_train_fn(examples: list[TrainingExample], config: SleepConfig, out_dir: Path) -> Path:
    from rlm.sleep.lora import train_lora

    return train_lora(
        examples,
        model_name=config.policy_model,
        out_dir=out_dir / "adapter",
        config=config.adapter,
        device=config.device,
        torch_dtype=config.torch_dtype,
        system_prompt=config.system_prompt,
    )


def default_eval_fn(
    config: SleepConfig,
    adapter_dir: Path | None,
    next_day: list[Episode],
    test: list[Episode],
) -> EvalReport:
    from rlm.sleep.evals import evaluate_model
    from rlm.sleep.lora import load_adapter, load_policy

    if adapter_dir is None:
        model, tokenizer = load_policy(
            config.policy_model, device=config.device, torch_dtype=config.torch_dtype
        )
        name = "base"
    else:
        model, tokenizer = load_adapter(
            config.policy_model, adapter_dir, device=config.device, torch_dtype=config.torch_dtype
        )
        name = "adapted"
    report = evaluate_model(
        model,
        tokenizer,
        name=name,
        next_day=next_day,
        test=test,
        device=config.device,
        max_seq_len=config.adapter.max_seq_len,
    )
    del model
    return report


def run_night(
    day_index: int,
    day_episodes: list[Episode],
    next_day_episodes: list[Episode],
    test_episodes: list[Episode],
    config: SleepConfig,
    judge_lm: BaseLM,
    out_dir: str | Path,
    train_fn: TrainFn = default_train_fn,
    eval_fn: EvalFn = default_eval_fn,
) -> DayResult:
    """Run one full nightly cycle and persist every artifact under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 — cheap gate over the day's episodes.
    decisions = select_for_reflection(day_episodes, config.gate)
    selected_ids = {d.episode_id for d in decisions if d.selected}
    selected = [ep for ep in day_episodes if ep.episode_id in selected_ids]
    (out_dir / "gate_decisions.json").write_text(
        json.dumps([asdict(d) for d in decisions], indent=2)
    )

    # Stage 2 — expensive reflection on the gated subset.
    judge = ReflectionJudge(judge_lm, config.judge)
    outputs = judge.reflect_all(selected)
    (out_dir / "judge_outputs.json").write_text(json.dumps([asdict(o) for o in outputs], indent=2))

    # Stage 3 — verified examples + replay -> SFT set (with leakage guard).
    examples = collect_examples(outputs)
    assert_no_leakage(examples, next_day_episodes + test_episodes)
    examples = with_replay(examples, config.adapter.replay_ratio, seed=config.adapter.seed)
    write_jsonl(examples, out_dir / "sft_data.jsonl", system_prompt=config.system_prompt)

    n_learn = sum(1 for o in outputs if o.verdict == "learn")
    if not examples:
        result = DayResult(
            day_index=day_index,
            n_episodes=len(day_episodes),
            n_selected=len(selected),
            n_learn_verdicts=n_learn,
            n_examples=0,
            adapter_dir=None,
            base_report=None,
            adapted_report=None,
        )
        (out_dir / "day_result.json").write_text(json.dumps(result.to_dict(), indent=2))
        return result

    # Stage 4 — train tonight's adapter; Stage 5 — evaluate base vs adapted.
    adapter_dir = train_fn(examples, config, out_dir)
    base_report = eval_fn(config, None, next_day_episodes, test_episodes)
    adapted_report = eval_fn(config, adapter_dir, next_day_episodes, test_episodes)

    result = DayResult(
        day_index=day_index,
        n_episodes=len(day_episodes),
        n_selected=len(selected),
        n_learn_verdicts=n_learn,
        n_examples=len(examples),
        adapter_dir=str(adapter_dir),
        base_report=base_report,
        adapted_report=adapted_report,
    )
    (out_dir / "day_result.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result
