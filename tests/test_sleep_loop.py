"""End-to-end nightly-cycle test with injected fake train/eval (no torch)."""

import json
from pathlib import Path

from rlm.sleep.config import SleepConfig
from rlm.sleep.loop import run_night
from rlm.sleep.sweep import expand_grid, run_sweep
from rlm.sleep.traces import partition_days, split_episodes, synthetic_traces
from rlm.sleep.types import EvalReport
from tests.mock_lm import MockLM

LEARN_RESPONSE = json.dumps(
    {
        "verdict": "learn",
        "confidence": 0.9,
        "lesson": "Validate arguments first.",
        "examples": [{"prompt": "Use the tool safely.", "response": "Validate, then call once."}],
    }
)


def judge_lm() -> MockLM:
    """Mock judge: always 'learn', always verifies YES."""

    def respond(prompt) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        return "YES" if "verifying a candidate" in text else LEARN_RESPONSE

    return MockLM(response_fn=respond)


def fake_train_fn(examples, config, out_dir: Path) -> Path:
    adapter = out_dir / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "training_meta.json").write_text(json.dumps({"n_examples": len(examples)}))
    return adapter


def fake_eval_fn(config, adapter_dir, next_day, test) -> EvalReport:
    # The "adapter" pretends to improve next-day NLL and slightly hurt retention.
    adapted = adapter_dir is not None
    return EvalReport(
        name="adapted" if adapted else "base",
        next_day_nll=1.5 if adapted else 2.0,
        test_nll=1.8 if adapted else 2.0,
        retention_nll=1.05 if adapted else 1.0,
        n_next_day=len(next_day),
        n_test=len(test),
    )


def make_splits():
    episodes = synthetic_traces(120, seed=8)
    train, _, test = split_episodes(episodes)
    days = partition_days(train, episodes_per_day=40)
    return days[0], days[1], test[:10]


def test_run_night_end_to_end(tmp_path):
    day0, day1, test = make_splits()
    config = SleepConfig(episodes_per_day=40)
    result = run_night(
        day_index=0,
        day_episodes=day0,
        next_day_episodes=day1,
        test_episodes=test,
        config=config,
        judge_lm=judge_lm(),
        out_dir=tmp_path / "day_0",
        train_fn=fake_train_fn,
        eval_fn=fake_eval_fn,
    )
    assert result.n_episodes == 40
    assert 0 < result.n_selected <= 10  # budget_fraction=0.25
    assert result.n_learn_verdicts == result.n_selected
    assert result.n_examples > result.n_learn_verdicts  # replay mixed in
    assert result.adapter_dir is not None
    assert result.base_report.next_day_nll == 2.0
    assert result.adapted_report.next_day_nll == 1.5

    # All artifacts persisted.
    for name in ["gate_decisions.json", "judge_outputs.json", "sft_data.jsonl", "day_result.json"]:
        assert (tmp_path / "day_0" / name).exists(), name


def test_run_night_quiet_day_trains_nothing(tmp_path):
    """If the judge skips everything, no adapter is trained."""
    day0, day1, test = make_splits()
    skip = json.dumps({"verdict": "skip", "confidence": 0.9, "lesson": "", "examples": []})
    result = run_night(
        day_index=0,
        day_episodes=day0,
        next_day_episodes=day1,
        test_episodes=test,
        config=SleepConfig(),
        judge_lm=MockLM(response_fn=lambda _: skip),
        out_dir=tmp_path / "day_0",
        train_fn=fake_train_fn,
        eval_fn=fake_eval_fn,
    )
    assert result.n_examples == 0
    assert result.adapter_dir is None
    assert result.base_report is None


def test_run_night_includes_extra_examples(tmp_path):
    from rlm.sleep.types import TrainingExample

    day0, day1, test = make_splits()
    extras = [
        TrainingExample(
            prompt=f"Prior night lesson {i}",
            response="Apply it.",
            lesson="old",
            source_episode_id=f"prior-{i}",
            verified=True,
        )
        for i in range(5)
    ]
    captured = {}

    def capturing_train_fn(examples, config, out_dir):
        captured["examples"] = examples
        return fake_train_fn(examples, config, out_dir)

    result = run_night(
        day_index=1,
        day_episodes=day0,
        next_day_episodes=day1,
        test_episodes=test,
        config=SleepConfig(),
        judge_lm=judge_lm(),
        out_dir=tmp_path / "day_1",
        train_fn=capturing_train_fn,
        eval_fn=fake_eval_fn,
        extra_examples=extras,
    )
    trained_prompts = {ex.prompt for ex in captured["examples"]}
    assert all(e.prompt in trained_prompts for e in extras)
    assert result.n_examples == len(captured["examples"])
    assert result.n_examples > len(extras)  # tonight's examples + replay too


def test_run_night_quiet_day_ignores_extras(tmp_path):
    """Extras alone don't justify an update: no new lessons -> no adapter."""
    from rlm.sleep.types import TrainingExample

    day0, day1, test = make_splits()
    skip = json.dumps({"verdict": "skip", "confidence": 0.9, "lesson": "", "examples": []})
    extras = [
        TrainingExample(
            prompt="Old lesson",
            response="x",
            lesson="old",
            source_episode_id="prior-0",
            verified=True,
        )
    ]
    result = run_night(
        day_index=1,
        day_episodes=day0,
        next_day_episodes=day1,
        test_episodes=test,
        config=SleepConfig(),
        judge_lm=MockLM(response_fn=lambda _: skip),
        out_dir=tmp_path / "day_1",
        train_fn=fake_train_fn,
        eval_fn=fake_eval_fn,
        extra_examples=extras,
    )
    assert result.adapter_dir is None
    assert result.n_examples == 0


class FakeSurpriseGate:
    """Stands in for the NLL gate: 'surprising' iff the episode id ends in 0."""

    def score(self, episode) -> float:
        return 1.0 if episode.episode_id.endswith("0") else 0.0


def test_run_night_uses_injected_surprise_gate(tmp_path):
    day0, day1, test = make_splits()
    run_night(
        day_index=0,
        day_episodes=day0,
        next_day_episodes=day1,
        test_episodes=test,
        config=SleepConfig(),
        judge_lm=judge_lm(),
        out_dir=tmp_path / "day_0",
        train_fn=fake_train_fn,
        eval_fn=fake_eval_fn,
        surprise_gate=FakeSurpriseGate(),
    )
    decisions = json.loads((tmp_path / "day_0" / "gate_decisions.json").read_text())
    # The fake gate's signal is recorded for every episode and averaged into
    # the final score (so a maximally surprising episode scores >= 0.5).
    assert all("surprise" in d["signals"] for d in decisions)
    for d in decisions:
        expected = 1.0 if d["episode_id"].endswith("0") else 0.0
        assert d["signals"]["surprise"] == expected
        if expected == 1.0:
            assert d["score"] >= 0.5


def test_build_surprise_gate_reuses_local_judge_model():
    from rlm.sleep.loop import build_surprise_gate

    class JudgeWithModel:
        model = object()
        tokenizer = object()

    gate = build_surprise_gate(SleepConfig(), JudgeWithModel())
    assert gate.model is JudgeWithModel.model
    assert gate.tokenizer is JudgeWithModel.tokenizer


def test_expand_grid():
    grid = {"adapter.lr": [1e-4, 2e-4], "adapter.rank": [8, 16]}
    points = expand_grid(grid)
    assert len(points) == 4
    assert {"adapter.lr": 1e-4, "adapter.rank": 8} in points


def test_run_sweep_writes_csv(tmp_path):
    day0, day1, test = make_splits()
    csv_path = run_sweep(
        grid={"adapter.lr": [1e-4, 2e-4]},
        base_config=SleepConfig(),
        day_episodes=day0[:10],
        next_day_episodes=day1[:5],
        test_episodes=test[:5],
        judge_lm=judge_lm(),
        out_dir=tmp_path / "sweep",
        train_fn=fake_train_fn,
        eval_fn=fake_eval_fn,
        progress=lambda _: None,
    )
    content = csv_path.read_text().splitlines()
    assert len(content) == 3  # header + 2 points
    assert "next_day_delta" in content[0]
    assert "adapter.lr" in content[0]
