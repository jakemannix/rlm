"""Sleep-time consolidation: gate -> judge -> LoRA -> eval.

A proof-of-concept of the practical continual-learning loop motivated in
``docs/learning_signal.md``: each "day" of agentic episodes is cheaply
filtered for learnworthy events, reflected on with a high-test-time-compute
LLM-as-a-Judge that distills verified SFT examples, trained into a nightly
LoRA adapter, and evaluated on the next day + a held-out test split with a
retention (forgetting) probe.

Core pipeline (no heavy deps): traces, gate, judge, dataset.
Training/eval (``sleep`` extra: torch/transformers/peft/datasets): lora,
evals, loop, sweep.
"""

from rlm.sleep.config import AdapterConfig, GateConfig, JudgeConfig, SleepConfig
from rlm.sleep.dataset import collect_examples, with_replay, write_jsonl
from rlm.sleep.gate import HeuristicGate, select_for_reflection
from rlm.sleep.judge import ReflectionJudge
from rlm.sleep.traces import load_hf_traces, partition_days, split_episodes, synthetic_traces
from rlm.sleep.types import (
    DayResult,
    Episode,
    EvalReport,
    GateDecision,
    JudgeOutput,
    Step,
    TrainingExample,
)

__all__ = [
    "AdapterConfig",
    "DayResult",
    "Episode",
    "EvalReport",
    "GateConfig",
    "GateDecision",
    "HeuristicGate",
    "JudgeConfig",
    "JudgeOutput",
    "ReflectionJudge",
    "SleepConfig",
    "Step",
    "TrainingExample",
    "collect_examples",
    "load_hf_traces",
    "partition_days",
    "select_for_reflection",
    "split_episodes",
    "synthetic_traces",
    "with_replay",
    "write_jsonl",
]
