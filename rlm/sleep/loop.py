"""The nightly consolidation cycle: gate -> judge -> dataset -> LoRA -> eval.

``run_night`` is dependency-injected: the trainer and evaluator default to
the real LoRA/eval implementations (which require the ``sleep`` extra),
and tests inject lightweight fakes — one code path either way.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from rlm.clients.base_lm import BaseLM
from rlm.sleep.config import SleepConfig
from rlm.sleep.dataset import assert_no_leakage, collect_examples, with_replay, write_jsonl
from rlm.sleep.gate import SurpriseGate, select_for_reflection
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


def build_surprise_gate(config: SleepConfig, judge_lm: BaseLM | None = None) -> SurpriseGate:
    """Construct the NLL surprise gate, reusing a local judge's model if present.

    A :class:`~rlm.sleep.local_judge.LocalHFJudge` already holds the frozen
    policy model in memory; gating with the same instance costs nothing extra.
    Otherwise the policy is loaded fresh.
    """
    model = getattr(judge_lm, "model", None)
    tokenizer = getattr(judge_lm, "tokenizer", None)
    if model is None or tokenizer is None:
        from rlm.sleep.lora import load_policy

        model, tokenizer = load_policy(
            config.policy_model, device=config.device, torch_dtype=config.torch_dtype
        )
        model.eval()
    return SurpriseGate(
        model, tokenizer, device=config.device, max_tokens=config.gate.surprise_max_tokens
    )


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
    surprise_gate: SurpriseGate | None = None,
    judge_cache_path: str | Path | None = None,
    extra_examples: list[TrainingExample] | None = None,
) -> DayResult:
    """Run one full nightly cycle and persist every artifact under ``out_dir``.

    ``extra_examples`` (e.g. prior nights' verified examples, for the
    cumulative re-distill baseline) join tonight's batch before replay
    mixing. A night that distills nothing new still trains nothing — the
    extras alone don't justify an update.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_start = time.perf_counter()
    stage_seconds: dict[str, float] = {}

    def mark(stage: str) -> None:
        nonlocal stage_start
        now = time.perf_counter()
        stage_seconds[stage] = round(now - stage_start, 2)
        stage_start = now

    # Stage 1 — cheap gate over the day's episodes.
    if surprise_gate is None and config.gate.use_surprise:
        surprise_gate = build_surprise_gate(config, judge_lm)
    decisions = select_for_reflection(day_episodes, config.gate, surprise_gate=surprise_gate)
    selected_ids = {d.episode_id for d in decisions if d.selected}
    selected = [ep for ep in day_episodes if ep.episode_id in selected_ids]
    (out_dir / "gate_decisions.json").write_text(
        json.dumps([asdict(d) for d in decisions], indent=2)
    )
    mark("gate")

    # Stage 2 — expensive reflection on the gated subset (cached when asked).
    judge = ReflectionJudge(judge_lm, config.judge, cache_path=judge_cache_path)
    outputs = judge.reflect_all(selected)
    (out_dir / "judge_outputs.json").write_text(json.dumps([asdict(o) for o in outputs], indent=2))
    mark("judge")

    # Stage 3 — verified examples + replay -> SFT set (with leakage guard).
    night_examples = collect_examples(outputs)
    examples = (list(extra_examples) if extra_examples else []) + night_examples
    assert_no_leakage(examples, next_day_episodes + test_episodes)
    if night_examples:
        examples = with_replay(examples, config.adapter.replay_ratio, seed=config.adapter.seed)
    write_jsonl(examples, out_dir / "sft_data.jsonl", system_prompt=config.system_prompt)

    n_learn = sum(1 for o in outputs if o.verdict == "learn")
    if not night_examples:
        result = DayResult(
            day_index=day_index,
            n_episodes=len(day_episodes),
            n_selected=len(selected),
            n_learn_verdicts=n_learn,
            n_examples=0,
            adapter_dir=None,
            base_report=None,
            adapted_report=None,
            stage_seconds=stage_seconds,
        )
        (out_dir / "day_result.json").write_text(json.dumps(result.to_dict(), indent=2))
        return result

    # Stage 4 — train tonight's adapter; Stage 5 — evaluate base vs adapted.
    adapter_dir = train_fn(examples, config, out_dir)
    mark("train")
    base_report = eval_fn(config, None, next_day_episodes, test_episodes)
    adapted_report = eval_fn(config, adapter_dir, next_day_episodes, test_episodes)
    mark("eval")

    result = DayResult(
        day_index=day_index,
        n_episodes=len(day_episodes),
        n_selected=len(selected),
        n_learn_verdicts=n_learn,
        n_examples=len(examples),
        adapter_dir=str(adapter_dir),
        base_report=base_report,
        adapted_report=adapted_report,
        stage_seconds=stage_seconds,
    )
    (out_dir / "day_result.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result
